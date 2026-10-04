"""热源起调安排：分段线性排程、速率策略、执行时序与版本追溯。"""

from __future__ import annotations

import dataclasses

import pytest

from heatnet.app import build_platform
from heatnet.config import ConfigError, PlatformConfig, config_to_mapping, default_config
from heatnet.console.api import Api
from heatnet.errors import InterlockError
from heatnet.source.ramp import RampTracker, build_ramp_plan


def _gas_limits():
    return dataclasses.replace(default_config().ramp, source_kind="gas")


def test_segments_tile_a_piecewise_linear_path():
    plan = build_ramp_plan("p1", 70.0, 90.0, 1000.0, default_config().ramp, 1, 0.35)
    assert len(plan.segments) == 5  # ceil(20 / 4℃)
    assert plan.segments[0].from_c == pytest.approx(70.0)
    assert plan.segments[-1].to_c == pytest.approx(90.0)
    for left, right in zip(plan.segments, plan.segments[1:]):
        assert left.to_c == pytest.approx(right.from_c)
        assert left.end_at == pytest.approx(right.start_at)
    assert plan.start_at == pytest.approx(1000.0)
    assert plan.arrive_at == pytest.approx(plan.segments[-1].end_at)
    # 折线取值：段内线性插值，起调前与到点后夹取端点
    segment = plan.segments[1]
    mid = (segment.start_at + segment.end_at) / 2.0
    assert plan.expected_c(mid) == pytest.approx((segment.from_c + segment.to_c) / 2.0)
    assert plan.expected_c(plan.start_at - 1.0) == pytest.approx(70.0)
    assert plan.expected_c(plan.arrive_at + 1.0) == pytest.approx(90.0)


def test_segment_count_is_capped_and_last_segment_takes_remainder():
    plan = build_ramp_plan("p1", 60.0, 95.0, 0.0, default_config().ramp, 1, 0.35)
    assert len(plan.segments) == default_config().ramp.segments_max
    step = (95.0 - 60.0) / default_config().ramp.segments_max
    assert plan.segments[0].to_c == pytest.approx(60.0 + step)
    assert plan.segments[-1].to_c == pytest.approx(95.0)


def test_strategy_is_fixed_by_source_kind():
    now, deadline = 5000.0, 8600.0
    coal = build_ramp_plan("c", 70.0, 90.0, now, default_config().ramp, 1, 0.35, arrive_by=deadline)
    gas = build_ramp_plan("g", 70.0, 90.0, now, _gas_limits(), 1, 0.35, arrive_by=deadline)
    assert coal.strategy == "conservative"
    assert gas.strategy == "average"
    # 保守按最慢速率并把余量前置到起调时刻；燃气按历史均值、不留余量
    assert coal.rate_c_s == pytest.approx(0.2)
    assert gas.rate_c_s == pytest.approx(0.3)
    assert coal.lead_margin_s == pytest.approx(600.0)
    assert gas.lead_margin_s == pytest.approx(0.0)
    assert coal.start_at < gas.start_at
    assert gas.start_at == pytest.approx(deadline - 20.0 / 0.3)


def test_planned_rate_never_exceeds_source_capability():
    plan = build_ramp_plan("p1", 70.0, 90.0, 0.0, _gas_limits(), 1, adjust_rate_c_s=0.1)
    assert plan.rate_c_s == pytest.approx(0.1)


def test_non_ramp_target_is_rejected():
    with pytest.raises(InterlockError):
        build_ramp_plan("p1", 80.0, 80.0, 0.0, default_config().ramp, 1, 0.35)
    with pytest.raises(InterlockError):
        build_ramp_plan("p1", 80.0, 70.0, 0.0, default_config().ramp, 1, 0.35)


def test_deadline_already_missed_marks_late_and_starts_now():
    plan = build_ramp_plan("p1", 70.0, 90.0, 1000.0, default_config().ramp, 1, 0.35, arrive_by=1100.0)
    # 升温要 100s、余量 600s，只剩 100s：已晚 600s，立即起调
    assert plan.late_s == pytest.approx(600.0)
    assert plan.start_at == pytest.approx(1000.0)


def test_execution_sequence_is_one_way():
    limits = default_config().ramp
    tracker = RampTracker(limits)
    plan = build_ramp_plan("p1", 70.0, 82.0, 1000.0, limits, 3, 0.35, arrive_by=2320.0)
    assert plan.start_at == pytest.approx(1660.0)  # 2320 - 60s升温 - 600s余量
    assert tracker.issue(plan) is None
    # 未到起调时刻，即便实际供温已到目标也不许跳到"到目标"
    assert tracker.advance(1200.0, 82.0) == []
    assert tracker.progress(1200.0, 82.0)["state"] == "pending"
    # 起调 → 分段推进 → 到目标，顺序不能乱
    first = tracker.advance(1660.0, 70.5)
    assert [item["action"] for item in first] == ["ramp_started", "ramp_segment"]
    assert tracker.progress(1660.0, 70.5)["state"] == "ramping"
    second = tracker.advance(1700.0, 82.0)
    assert [item["action"] for item in second] == ["ramp_segment", "ramp_segment", "ramp_reached"]
    assert tracker.progress(1700.0, 82.0)["state"] == "reached"
    # 到目标后状态不回退
    assert tracker.advance(1800.0, 60.0) == []
    assert tracker.progress(1800.0, 60.0)["state"] == "reached"


def test_issue_supersedes_unfinished_plan():
    tracker = RampTracker(default_config().ramp)
    tracker.issue(build_ramp_plan("p1", 70.0, 80.0, 0.0, default_config().ramp, 1, 0.35))
    superseded = tracker.issue(build_ramp_plan("p2", 70.0, 85.0, 10.0, default_config().ramp, 2, 0.35))
    assert superseded is not None
    assert superseded["plan_id"] == "p1"
    assert superseded["state"] == "superseded"
    assert [item["plan_id"] for item in tracker.history()] == ["p1"]
    assert tracker.active().plan_id == "p2"


def test_broken_ramp_config_is_rejected():
    mapping = config_to_mapping(default_config())
    mapping["limits"]["ramp"]["rate_avg_c_s"] = 0.05  # 历史均值低于最慢速率
    with pytest.raises(ConfigError):
        PlatformConfig.from_mapping(mapping)
    mapping = config_to_mapping(default_config())
    mapping["limits"]["ramp"]["source_kind"] = "nuclear"
    with pytest.raises(ConfigError):
        PlatformConfig.from_mapping(mapping)


def test_plan_defaults_to_curve_target_and_pins_epoch(online):
    with pytest.raises(InterlockError):
        online.plan_ramp()  # 已按曲线供温运行，无需起调
    report = online.plan_ramp(85.0)
    assert report["ok"] is True
    plan = report["plan"]
    assert plan["curve_epoch"] == online.curve.epoch()
    assert plan["strategy"] == "conservative"
    assert plan["source_kind"] == "coal"
    assert plan["segments"]


def test_gas_source_plans_on_average_rate(tmp_path, clock):
    config = dataclasses.replace(default_config(), ramp=_gas_limits())
    platform = build_platform(config, clock=clock, storage_root=str(tmp_path / "state"))
    platform.start()
    platform.push_weather(-8.0)
    platform.startup(-8.0, 60.0)
    plan = platform.plan_ramp(85.0)["plan"]
    assert plan["strategy"] == "average"
    assert plan["rate_c_s"] == pytest.approx(0.3)
    assert plan["lead_margin_s"] == pytest.approx(0.0)


def test_plan_with_deadline_starts_ahead_with_margin(online, clock):
    plan = online.plan_ramp(85.0, arrive_in_s=3600.0)["plan"]
    assert plan["start_at"] > clock.now()  # 不到时刻不起调
    assert plan["arrive_at"] <= plan["deadline_at"]
    assert plan["lead_margin_s"] == pytest.approx(600.0)
    assert plan["deadline_at"] == pytest.approx(clock.now() + 3600.0)


def test_curve_revision_keeps_issued_plan_on_its_epoch(online):
    plan = online.plan_ramp(85.0)["plan"]
    online.push_weather(-12.0)  # 曲线换版
    report = online.ramp_report()
    assert report["curve_revised"] is True
    assert report["plan"]["curve_epoch"] == plan["curve_epoch"]
    assert report["plan"]["target_c"] == pytest.approx(85.0)
    revised = [entry for entry in online.audit.tail(500) if entry["action"] == "ramp_curve_revised"]
    assert revised
    assert revised[-1]["payload"]["plan_epoch"] == plan["curve_epoch"]
    assert revised[-1]["payload"]["curve_epoch"] == online.curve.epoch()


def test_execution_matches_actuals_and_audit(online):
    online.plan_ramp(85.0)
    online.cycle(dt=1.0)
    online.adjust_source(85.0, 1500.0)
    online.cycle(dt=1.0)
    actions = [entry["action"] for entry in online.audit.tail(500)]
    assert "ramp_planned" in actions
    assert actions.index("ramp_started") < actions.index("ramp_reached")
    reached = [entry for entry in online.audit.tail(500) if entry["action"] == "ramp_reached"]
    assert reached[-1]["payload"]["target_c"] == pytest.approx(85.0)
    progress = online.ramp_report()["progress"]
    assert progress["state"] == "reached"
    assert progress["actual_c"] == pytest.approx(online.source.temperature_c())


def test_progress_flags_lag_when_actual_falls_behind(online):
    online.plan_ramp(85.0)
    for _ in range(60):
        online.cycle(dt=1.0)
    progress = online.ramp_report()["progress"]
    assert progress["state"] == "ramping"
    assert progress["late_s"] > 0.0
    assert progress["deviation_c"] < 0.0  # 实际供温落后于折线


def test_new_plan_supersedes_active_one_in_audit(online):
    first = online.plan_ramp(85.0)["plan"]
    second = online.plan_ramp(88.0)["plan"]
    report = online.ramp_report()
    assert report["plan"]["plan_id"] == second["plan_id"]
    assert [item["plan_id"] for item in report["history"]] == [first["plan_id"]]
    assert report["history"][0]["state"] == "superseded"
    actions = [entry["action"] for entry in online.audit.tail(500)]
    assert "ramp_superseded" in actions


def test_status_exposes_ramp_section(online):
    online.plan_ramp(85.0)
    status = online.status()
    assert status["ramp"]["plan"]["target_c"] == pytest.approx(85.0)
    assert status["ramp"]["progress"]["state"] == "pending"
    assert status["ramp"]["curve_revised"] is False


def test_console_ramp_endpoints(online):
    api = Api(online)
    status, payload = api.handle("POST", "/api/ramp", {}, {"target_c": 85.0, "origin": "console"})
    assert status == 200
    assert payload["ok"] is True
    status, report = api.handle("GET", "/api/ramp", {}, {})
    assert status == 200
    assert report["plan"]["target_c"] == pytest.approx(85.0)
    assert report["strategy"] == "conservative"
    status, rejected = api.handle("POST", "/api/ramp", {}, {"target_c": 50.0})
    assert status == 409
    assert rejected["component"] == "ramp"
