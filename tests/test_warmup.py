"""升温起调安排：分段线性、保守策略按热源类型、执行时序、版本冻结。"""

from __future__ import annotations

import dataclasses
import math

import pytest

from heatnet.config import default_config
from heatnet.errors import InterlockError
from heatnet.source.warmup import (
    CURVE_SHAPE,
    PENDING,
    STARTED,
    WARMING,
    REACHED,
    CANCELLED,
    WarmupRunner,
    build_warmup_plan,
    effective_rate_c_s,
    policy_for,
)
from heatnet.temp.versioning import CurveVersion, CurveVersionRegistry


def _plan(**overrides) -> object:
    params = dict(
        plan_id="w1",
        source_type="coal_fired",
        current_c=70.0,
        target_c=91.0,
        rated_rate_c_s=0.35,
        now=1000.0,
        curve_epoch=3,
        curve_outdoor_c=-10.0,
        deadline_at=2000.0,
    )
    params.update(overrides)
    return build_warmup_plan(**params)


def test_curve_shape_is_pinned_to_piecewise_linear():
    assert CURVE_SHAPE == "piecewise_linear"
    plan = _plan()
    assert plan.curve.shape == "piecewise_linear"
    assert plan.as_dict()["curve_shape"] == "piecewise_linear"


def test_policy_is_decided_by_source_type():
    assert policy_for("coal_fired") == "guaranteed"
    for source_type in ("gas_fired", "electric", "chp"):
        assert policy_for(source_type) == "historical_mean"
    with pytest.raises(ValueError):
        policy_for("oil_fired")


def test_guaranteed_rate_uses_slowest_credible_with_margin():
    # 最慢可信速率取额定 60%，给保守口径留足余量。
    assert effective_rate_c_s("guaranteed", 0.35) == pytest.approx(0.21)


def test_historical_mean_uses_observed_average_but_stays_in_band():
    assert effective_rate_c_s("historical_mean", 0.35) == pytest.approx(0.35)
    assert effective_rate_c_s("historical_mean", 0.35, [0.28, 0.32]) == pytest.approx(0.30)
    # 均值不得超过额定联锁上限，也不得低于保守底。
    assert effective_rate_c_s("historical_mean", 0.35, [0.9, 0.9]) == pytest.approx(0.35)
    assert effective_rate_c_s("historical_mean", 0.35, [0.01, 0.01]) == pytest.approx(0.21)


def test_plan_backschedules_start_time_from_deadline():
    plan = _plan()
    # delta=21℃，0.21℃/s → 100s；保守提前量为升温耗时 25% → 25s。
    assert plan.ramp_s == pytest.approx(100.0)
    assert plan.lead_s == pytest.approx(25.0)
    assert plan.start_at == pytest.approx(1875.0)
    assert plan.reach_at == pytest.approx(1975.0)
    assert plan.deadline_at == pytest.approx(2000.0)


def test_historical_mean_plan_uses_tighter_lead_margin():
    plan = _plan(source_type="gas_fired", observed_rates_c_s=[0.30])
    assert plan.policy == "historical_mean"
    # 21℃ / 0.30℃/s = 70s，提前量 10% = 7s。
    assert plan.ramp_s == pytest.approx(70.0)
    assert plan.lead_s == pytest.approx(7.0)
    assert plan.start_at == pytest.approx(2000.0 - 70.0 - 7.0)


def test_plan_is_split_into_linear_segments():
    plan = _plan()
    assert len(plan.segments) == 3  # ceil(21/8)
    first, last = plan.segments[0], plan.segments[-1]
    assert first.start_c == pytest.approx(70.0)
    assert last.end_c == pytest.approx(91.0)
    # 段内匀速，端点衔接，不允许平滑曲线式的非线性。
    mid = (first.start_s + first.end_s) / 2
    assert first.expected_c(mid) == pytest.approx((first.start_c + first.end_c) / 2)
    assert plan.expected_c(plan.start_at - 50) == pytest.approx(70.0)
    assert plan.expected_c(plan.reach_at + 50) == pytest.approx(91.0)
    for left, right in zip(plan.segments, plan.segments[1:]):
        assert left.end_c == pytest.approx(right.start_c)


def test_no_warmup_needed_when_already_at_target():
    plan = _plan(current_c=91.0, target_c=90.8)
    assert plan.required is False
    assert plan.segments == ()
    assert WarmupRunner(plan).state() == REACHED


def test_runner_enforces_start_warm_reach_order():
    plan = _plan()
    runner = WarmupRunner(plan)
    # 没起调不允许确认到目标。
    with pytest.raises(InterlockError):
        runner.complete(91.0, 1975.0)
    runner.start(1875.0)
    assert runner.state() == STARTED
    with pytest.raises(InterlockError):
        runner.start(1876.0)
    # 起调 → 进入第 0 段（升温）。
    assert "enter_segment:0" in runner.track(70.0, 1875.0)["transitions"]
    assert runner.state() == WARMING
    # 不允许跳段。
    with pytest.raises(InterlockError):
        runner._enter_or_advance(2, 1900.0)
    assert runner._enter_or_advance(1, 1910.0) == "advance_segment:1"
    assert runner._enter_or_advance(2, 1945.0) == "advance_segment:2"
    # 段走完后不许再跨段，只能到目标确认。
    with pytest.raises(InterlockError):
        runner._enter_or_advance(3, 1980.0)
    # 实测没到目标不许收尾。
    with pytest.raises(InterlockError):
        runner.complete(88.0, 1975.0)
    result = runner.complete(91.0, 1975.0)
    assert result["state"] == REACHED
    with pytest.raises(InterlockError):
        runner.complete(91.0, 1976.0)


def test_runner_flags_lag_against_piecewise_expectation():
    plan = _plan(now=0.0, deadline_at=None, target_c=90.0, current_c=70.0)
    runner = WarmupRunner(plan)
    runner.start(0.0)
    progress = runner.track(plan.expected_c(40.0) - 5.0, 40.0)
    assert progress["lagging"] is True
    progress = runner.track(plan.expected_c(40.0), 40.0)
    assert progress["lagging"] is False


def test_cancelled_plan_cannot_restart():
    runner = WarmupRunner(_plan())
    runner.cancel("curve_revision")
    assert runner.state() == CANCELLED
    with pytest.raises(InterlockError):
        runner.start(1.0)


def test_plan_roundtrips_and_rejects_unknown_shape():
    plan = _plan()
    restored = type(plan).from_dict(plan.as_dict())
    assert restored.plan_id == plan.plan_id
    assert restored.curve.epoch == 3
    raw = plan.as_dict()
    raw["curve_shape"] = "smooth_spline"
    with pytest.raises(ValueError):
        type(plan).from_dict(raw)


def test_version_registry_is_append_only_and_traceable():
    registry = CurveVersionRegistry()
    registry.register(CurveVersion.create(1, -8.0, 83.0, 100.0))
    registry.register(CurveVersion.create(2, -12.0, 87.0, 200.0))
    assert registry.current().epoch == 2
    # 同一代际内容不一致时禁止覆盖，老版保持原样。
    with pytest.raises(ValueError):
        registry.register(CurveVersion.create(1, -8.0, 84.0, 100.0))
    assert registry.get(1).supply_c == pytest.approx(83.0)
    with pytest.raises(ValueError):
        registry.register(CurveVersion.create(0, -1.0, 60.0, 50.0))


def test_config_rejects_unknown_source_type():
    from heatnet.config import SourceLimits

    with pytest.raises(Exception):
        SourceLimits.from_mapping(
            {
                "supply_target_c": 92.0,
                "supply_max_c": 115.0,
                "return_min_c": 35.0,
                "flow_min_t_h": 900.0,
                "flow_max_t_h": 4200.0,
                "quality_confirm_s": 12.0,
                "adjust_rate_c_s": 0.35,
                "source_type": "nuclear",
            }
        )


# ---- 平台级 -----------------------------------------------------------

def _gas_config():
    base = default_config()
    return dataclasses.replace(
        base, source=dataclasses.replace(base.source, source_type="gas_fired")
    )


def test_platform_issues_pins_curve_epoch_and_survives_revision(platform):
    platform.push_weather(-10.0)
    platform.adjust_source(65.0, 2000.0, "test")
    issued = platform.issue_warmup_plan(deadline_at=platform.clock.now() + 2000.0)
    assert issued["plan"]["curve_shape"] == "piecewise_linear"
    assert issued["plan"]["curve"]["epoch"] == 1
    assert issued["plan"]["policy"] == "guaranteed"

    # 曲线换版：已下发安排仍按下发时那一版解释。
    platform.push_weather(-18.0)
    runner = platform.warmup_runners[issued["plan_id"]]
    assert runner.plan.curve.epoch == 1
    revisions = [r for r in platform.audit.records(500) if r.action == "curve_revision"]
    assert {r.payload["epoch"] for r in revisions} == {1, 2}


def test_platform_issuing_new_plan_cancels_old_with_audit(platform):
    platform.push_weather(-10.0)
    platform.adjust_source(65.0, 2000.0, "test")
    first = platform.issue_warmup_plan(deadline_at=platform.clock.now() + 2000.0)
    second = platform.issue_warmup_plan(deadline_at=platform.clock.now() + 2000.0)
    assert platform.warmup_runners[first["plan_id"]].state() == CANCELLED
    assert platform.warmup_runners[second["plan_id"]].state() == PENDING
    cancelled = [r for r in platform.audit.records(500) if r.action == "warmup_cancelled"]
    assert f"superseded_by:{second['plan_id']}" in cancelled[-1].payload["reason"]


def test_platform_warmup_sequence_and_report_consistency(platform):
    platform.push_weather(-10.0)
    target = platform.curve.supply_c()
    platform.adjust_source(65.0, 2000.0, "test")
    issued = platform.issue_warmup_plan()  # 立即起调建议
    assert issued["plan"]["start_at"] == pytest.approx(issued["plan"]["issued_at"])
    platform.start_warmup(issued["plan_id"])
    platform.adjust_source(target, 2000.0, "warmup")
    platform.cycle(dt=1.0)
    completed = platform.complete_warmup(issued["plan_id"])
    assert completed["state"] == REACHED

    report = platform.warmup_report(issued["plan_id"])
    assert report["curve_shape"] == "piecewise_linear"
    assert report["measured_c"] == round(target, 2)
    assert report["plans"][0]["state"] == REACHED
    assert report["plans"][0]["plan"]["curve"]["epoch"] == platform.curve.epoch()
    assert platform.audit.count("warmup_reached") == 1


def test_gas_source_uses_historical_mean(tmp_path):
    from heatnet.app import build_platform
    from heatnet.clock import ManualClock

    gas = build_platform(_gas_config(), clock=ManualClock(), storage_root=str(tmp_path / "gas"))
    gas.push_weather(-10.0)
    gas.adjust_source(65.0, 2000.0, "test")
    issued = gas.issue_warmup_plan(deadline_at=gas.clock.now() + 2000.0)
    assert issued["plan"]["policy"] == "historical_mean"
    plan = gas.warmup_runners[issued["plan_id"]].plan
    assert plan.lead_s == pytest.approx(plan.ramp_s * 0.10)


def test_warmup_persists_across_restart(tmp_path):
    from heatnet.app import build_platform
    from heatnet.clock import ManualClock

    clock = ManualClock()
    root = str(tmp_path / "state")
    platform = build_platform(_gas_config(), clock=clock, storage_root=root)
    platform.push_weather(-10.0)
    platform.adjust_source(65.0, 2000.0, "test")
    issued = platform.issue_warmup_plan(deadline_at=clock.now() + 2000.0)
    assert issued["plan"]["policy"] == "historical_mean"
    platform.start_warmup(issued["plan_id"])

    rebuilt = build_platform(_gas_config(), clock=clock, storage_root=root)
    runner = rebuilt.warmup_runners[issued["plan_id"]]
    assert runner.state() == STARTED
    # 曲线版本台账同样持久化，安排可追溯到当时那一版。
    assert rebuilt.curve.epoch() == 1
    assert rebuilt.curve.versions.get(1) is not None
