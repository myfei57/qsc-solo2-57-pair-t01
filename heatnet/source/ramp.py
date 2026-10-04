"""热源起调安排。

值班员面对的是"曲线给的目标温度一点点往上爬，不知道什么时候该提前起调"。这里
把起调安排定死成**分段线性**：按当前供温、目标供温与升温速率算出几段等温升的
爬升段，每段给到多少、什么时候到，段与段首尾相接，不按平滑曲线走——执行、展示
与审计都按同一条折线对表。

偏保守（按最慢速率并把余量前置到起调时刻）还是按历史均值博一把，由热源类型
决定（:class:`~heatnet.config.RampLimits`），不在排程时临时选。计划速率再快也
不允许超过热源自身的可调速率，否则安排根本执行不了。

安排一旦下发就钉死当时的曲线代际：中途曲线换版属于版本变化，已下发的安排仍按
当时那一版的目标与分段解释；新版曲线要生效就重新下发一份安排，旧安排转入历史。
执行时序单向推进：待起调 → 升温中 → 到目标，不允许回退；每一次推进都把计划
温度与实际供温一起交还给平台落审计，三处口径一致。
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

from ..config import RampLimits
from ..errors import InterlockError

PENDING = "pending"
RAMPING = "ramping"
REACHED = "reached"
SUPERSEDED = "superseded"


@dataclass(frozen=True)
class RampSegment:
    """一段等速爬升：从 ``from_c`` 到 ``to_c``，首尾时刻与邻段相接。"""

    index: int
    from_c: float
    to_c: float
    start_at: float
    end_at: float

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "from_c": round(self.from_c, 2),
            "to_c": round(self.to_c, 2),
            "start_at": round(self.start_at, 3),
            "end_at": round(self.end_at, 3),
        }


@dataclass(frozen=True)
class RampPlan:
    """一份已下发的起调安排（下发后不再改动，曲线换版也不影响它）。"""

    plan_id: str
    issued_at: float
    curve_epoch: int
    source_kind: str
    strategy: str
    start_c: float
    target_c: float
    rate_c_s: float
    start_at: float
    arrive_at: float
    deadline_at: float | None
    lead_margin_s: float
    late_s: float
    segments: tuple[RampSegment, ...]

    def expected_c(self, at: float) -> float:
        """分段线性折线上 ``at`` 时刻应有的供温。"""

        moment = float(at)
        if moment <= self.start_at:
            return self.start_c
        if moment >= self.arrive_at:
            return self.target_c
        for segment in self.segments:
            if segment.start_at <= moment < segment.end_at:
                ratio = (moment - segment.start_at) / (segment.end_at - segment.start_at)
                return segment.from_c + (segment.to_c - segment.from_c) * ratio
        return self.target_c

    def segment_index(self, at: float) -> int:
        """``at`` 时刻落在第几段（未起调为 -1，过到点后停在最后一段）。"""

        moment = float(at)
        if moment < self.start_at:
            return -1
        for segment in self.segments:
            if segment.start_at <= moment < segment.end_at:
                return segment.index
        return len(self.segments) - 1

    def as_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "issued_at": round(self.issued_at, 3),
            "curve_epoch": self.curve_epoch,
            "source_kind": self.source_kind,
            "strategy": self.strategy,
            "start_c": round(self.start_c, 2),
            "target_c": round(self.target_c, 2),
            "rate_c_s": round(self.rate_c_s, 4),
            "start_at": round(self.start_at, 3),
            "arrive_at": round(self.arrive_at, 3),
            "deadline_at": None if self.deadline_at is None else round(self.deadline_at, 3),
            "lead_margin_s": round(self.lead_margin_s, 1),
            "late_s": round(self.late_s, 1),
            "segments": [segment.as_dict() for segment in self.segments],
        }


def build_ramp_plan(
    plan_id: str,
    start_c: float,
    target_c: float,
    now: float,
    limits: RampLimits,
    curve_epoch: int,
    adjust_rate_c_s: float,
    arrive_by: float | None = None,
) -> RampPlan:
    """按当前供温、目标供温与升温速率算出一份分段线性的起调安排。"""

    start = float(start_c)
    target = float(target_c)
    moment = float(now)
    delta = target - start
    if delta <= 0:
        raise InterlockError("目标供温不高于当前供温，无需起调", component="ramp")
    strategy = limits.strategy()
    rate = limits.rate_slow_c_s if strategy == "conservative" else limits.rate_avg_c_s
    rate = min(rate, float(adjust_rate_c_s))
    margin = limits.lead_margin_s if strategy == "conservative" else 0.0

    count = min(limits.segments_max, max(1, math.ceil(delta / limits.segment_step_c)))
    step_c = delta / count
    duration = delta / rate
    deadline = None if arrive_by is None else float(arrive_by)
    late_s = 0.0
    if deadline is None:
        start_at = moment
    else:
        start_at = deadline - duration - margin
        if start_at < moment:
            late_s = moment - start_at
            start_at = moment
    step_s = duration / count
    segments = tuple(
        RampSegment(
            index,
            start + step_c * index,
            start + step_c * (index + 1),
            start_at + step_s * index,
            start_at + step_s * (index + 1),
        )
        for index in range(count)
    )
    return RampPlan(
        plan_id=str(plan_id),
        issued_at=moment,
        curve_epoch=int(curve_epoch),
        source_kind=limits.source_kind,
        strategy=strategy,
        start_c=start,
        target_c=target,
        rate_c_s=rate,
        start_at=start_at,
        arrive_at=start_at + duration,
        deadline_at=deadline,
        lead_margin_s=margin,
        late_s=late_s,
        segments=segments,
    )


class RampExecution:
    """一份安排的执行侧状态（单向推进，不回退）。"""

    def __init__(self) -> None:
        self.state = PENDING
        self.started_at: float | None = None
        self.reached_at: float | None = None
        self.segment = -1


class RampTracker:
    """起调安排的登记、推进与历史。"""

    def __init__(self, limits: RampLimits, history_limit: int = 32) -> None:
        self._limits = limits
        self._history_limit = max(1, history_limit)
        self._current: tuple[RampPlan, RampExecution] | None = None
        self._history: list[dict] = []
        self._lock = threading.RLock()

    def issue(self, plan: RampPlan) -> dict | None:
        """下发一份新安排；旧安排未走完则标记被取代并转入历史。"""

        with self._lock:
            superseded = None
            if self._current is not None:
                old_plan, old_execution = self._current
                if old_execution.state in (PENDING, RAMPING):
                    old_execution.state = SUPERSEDED
                    superseded = self._archive(old_plan, old_execution)
                else:
                    self._archive(old_plan, old_execution)
            self._current = (plan, RampExecution())
            return superseded

    def active(self) -> RampPlan | None:
        """返回仍在执行窗口内的安排（待起调或升温中）。"""

        with self._lock:
            if self._current is None:
                return None
            plan, execution = self._current
            if execution.state in (PENDING, RAMPING):
                return plan
            return None

    def advance(self, now: float, actual_c: float) -> list[dict]:
        """按当前时刻与实际供温推进执行状态，返回本次发生的跃迁（供审计落盘）。"""

        moment = float(now)
        actual = float(actual_c)
        with self._lock:
            if self._current is None:
                return []
            plan, execution = self._current
            transitions: list[dict] = []
            if execution.state == PENDING and moment >= plan.start_at:
                execution.state = RAMPING
                execution.started_at = moment
                transitions.append(
                    {
                        "action": "ramp_started",
                        "message": f"起调开始（{plan.plan_id}）",
                        "plan_id": plan.plan_id,
                        "curve_epoch": plan.curve_epoch,
                        "planned_c": round(plan.start_c, 2),
                        "actual_c": round(actual, 2),
                    }
                )
            if execution.state == RAMPING:
                index = plan.segment_index(moment)
                while execution.segment < index:
                    execution.segment += 1
                    segment = plan.segments[execution.segment]
                    transitions.append(
                        {
                            "action": "ramp_segment",
                            "message": f"第 {segment.index + 1} 段到 {segment.to_c:.1f}℃",
                            "plan_id": plan.plan_id,
                            "segment": segment.index,
                            "expected_c": round(segment.to_c, 2),
                            "actual_c": round(actual, 2),
                            "deviation_c": round(actual - segment.to_c, 2),
                        }
                    )
                if actual >= plan.target_c - self._limits.reach_band_c:
                    execution.state = REACHED
                    execution.reached_at = moment
                    transitions.append(
                        {
                            "action": "ramp_reached",
                            "message": f"到达目标 {plan.target_c:.1f}℃（{plan.plan_id}）",
                            "plan_id": plan.plan_id,
                            "curve_epoch": plan.curve_epoch,
                            "target_c": round(plan.target_c, 2),
                            "actual_c": round(actual, 2),
                            "late_s": round(max(0.0, moment - plan.arrive_at), 1),
                        }
                    )
            return transitions

    def progress(self, now: float, actual_c: float) -> dict:
        """返回当前安排的对表进度（计划温度、实际供温与偏差）。"""

        moment = float(now)
        actual = float(actual_c)
        with self._lock:
            if self._current is None:
                return {"state": "none"}
            plan, execution = self._current
            expected = plan.expected_c(moment)
            late = 0.0
            if execution.state != REACHED and moment > plan.arrive_at:
                late = moment - plan.arrive_at
            return {
                "state": execution.state,
                "plan_id": plan.plan_id,
                "segment": execution.segment,
                "segments": len(plan.segments),
                "expected_c": round(expected, 2),
                "actual_c": round(actual, 2),
                "deviation_c": round(actual - expected, 2),
                "late_s": round(late, 1),
                "started_at": None if execution.started_at is None else round(execution.started_at, 3),
                "reached_at": None if execution.reached_at is None else round(execution.reached_at, 3),
            }

    def history(self) -> list[dict]:
        """返回已完结安排的摘要（按完结先后排序）。"""

        with self._lock:
            return [dict(item) for item in self._history]

    def report(self, now: float, actual_c: float) -> dict:
        """汇总当前安排、执行进度与历史，供控制台展示。"""

        with self._lock:
            plan = None if self._current is None else self._current[0]
        return {
            "plan": None if plan is None else plan.as_dict(),
            "progress": self.progress(now, actual_c),
            "history": self.history(),
        }

    def _archive(self, plan: RampPlan, execution: RampExecution) -> dict:
        """把一份离开生效位的安排折成摘要转入历史。"""

        summary = {
            "plan_id": plan.plan_id,
            "curve_epoch": plan.curve_epoch,
            "issued_at": round(plan.issued_at, 3),
            "target_c": round(plan.target_c, 2),
            "strategy": plan.strategy,
            "state": execution.state,
            "started_at": None if execution.started_at is None else round(execution.started_at, 3),
            "reached_at": None if execution.reached_at is None else round(execution.reached_at, 3),
        }
        self._history.append(summary)
        if len(self._history) > self._history_limit:
            del self._history[: len(self._history) - self._history_limit]
        return summary
