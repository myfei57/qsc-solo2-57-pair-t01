"""热源升温起调安排。

值班员靠人工判断什么时候开始起调，天一冷就慢半拍。这里把"什么时候起调、分几段
爬、每段爬到多少"算成一份可执行的安排，并由 :class:`WarmupRunner` 状态机看守
执行时序：起调 → 逐段升温 → 到目标，顺序不能颠倒。

定死的口径（不接受运行时切换）：

* 升温曲线形状是**分段线性**（:data:`CURVE_SHAPE`）。每一段内部温度随时间匀速
  爬升，断点处就是控制台展示和审计核对的确认点。平滑曲线的口径被明确排除。
* 保守策略按热源类型在 :data:`SOURCE_POLICY` 里定死：燃煤炉惯性大、压火代价高，
  按最慢可信速率留足余量（``guaranteed``，起调提前量也给得更宽）；燃气炉、电热源
  与热电联产调节灵活，按历史观测均值博一把（``historical_mean``，提前量收紧）。

版本化：安排在下发那一刻冻结当版曲线（:class:`~heatnet.temp.versioning.CurveVersion`
的代际与目标供温），中途曲线换版不影响已下发安排的解释，审计里能追溯到那一版。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import SOURCE_TYPES
from ..errors import InterlockError

# 升温曲线形状：定死为分段线性。
CURVE_SHAPE = "piecewise_linear"

# 按热源类型定死的保守策略。
POLICY_GUARANTEED = "guaranteed"        # 按最慢可信速率留足余量
POLICY_HISTORICAL_MEAN = "historical_mean"  # 按历史均值博一把
SOURCE_POLICY: dict[str, str] = {
    "coal_fired": POLICY_GUARANTEED,
    "gas_fired": POLICY_HISTORICAL_MEAN,
    "electric": POLICY_HISTORICAL_MEAN,
    "chp": POLICY_HISTORICAL_MEAN,
}
assert set(SOURCE_POLICY) == set(SOURCE_TYPES)

# 最慢可信速率取额定最大升温速率的 60%（再慢说明设备已经异常，不该当余量用）。
GUARANTEED_RATE_FRACTION = 0.6
# 起调提前量：保守口径留 25% 余量，博一把口径只留 10%。
LEAD_MARGIN = {POLICY_GUARANTEED: 0.25, POLICY_HISTORICAL_MEAN: 0.10}
# 分段：每段温升降幅 8℃，至少 2 段、最多 8 段（避免安排碎到不可执行）。
SEGMENT_STEP_C = 8.0
MIN_SEGMENTS = 2
MAX_SEGMENTS = 8
# 判定"已经在目标上"的容差。
TARGET_BAND_C = 0.5
# 到目标时刻之后宽限多久算滞后（审计与控制台用）。
OVERDUE_GRACE_S = 30.0


@dataclass(frozen=True)
class RampSegment:
    """一段线性爬升：在 ``start_s`` 从 ``start_c`` 匀速爬到 ``end_c``。"""

    index: int
    start_s: float
    end_s: float
    start_c: float
    end_c: float
    rate_c_s: float

    def expected_c(self, now: float) -> float:
        """分段线性插值出 ``now`` 时刻的期望供温，段外夹取到端点。"""

        if now <= self.start_s:
            return self.start_c
        if now >= self.end_s:
            return self.end_c
        ratio = (now - self.start_s) / (self.end_s - self.start_s)
        return self.start_c + (self.end_c - self.start_c) * ratio

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "start_s": round(self.start_s, 1),
            "end_s": round(self.end_s, 1),
            "start_c": round(self.start_c, 2),
            "end_c": round(self.end_c, 2),
            "rate_c_s": round(self.rate_c_s, 4),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "RampSegment":
        return cls(
            index=int(raw["index"]),
            start_s=float(raw["start_s"]),
            end_s=float(raw["end_s"]),
            start_c=float(raw["start_c"]),
            end_c=float(raw["end_c"]),
            rate_c_s=float(raw["rate_c_s"]),
        )


@dataclass(frozen=True)
class CurveRef:
    """安排所依据的曲线版本坐标（下发即冻结）。"""

    epoch: int
    supply_c: float
    outdoor_c: float
    shape: str = CURVE_SHAPE

    def as_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "supply_c": round(self.supply_c, 2),
            "outdoor_c": round(self.outdoor_c, 2),
            "shape": self.shape,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "CurveRef":
        return cls(
            epoch=int(raw["epoch"]),
            supply_c=float(raw["supply_c"]),
            outdoor_c=float(raw["outdoor_c"]),
            shape=str(raw.get("shape", CURVE_SHAPE)),
        )


@dataclass(frozen=True)
class WarmupPlan:
    """一份可执行的升温起调安排。"""

    plan_id: str
    source_type: str
    policy: str
    current_c: float
    target_c: float
    delta_c: float
    rate_c_s: float
    lead_s: float
    ramp_s: float
    issued_at: float
    start_at: float
    reach_at: float
    deadline_at: float | None
    segments: tuple[RampSegment, ...]
    curve: CurveRef
    required: bool = True
    note: str = ""

    def segment_at(self, now: float) -> RampSegment | None:
        """返回 ``now`` 所在（或最近一个已经起算）的升温段。"""

        active = None
        for segment in self.segments:
            if now >= segment.start_s:
                active = segment
            else:
                break
        return active

    def expected_c(self, now: float) -> float:
        """分段线性给出 ``now`` 时刻的期望供温（整段安排外夹取端点）。"""

        if not self.segments:
            return self.target_c
        if now <= self.start_at:
            return self.current_c
        if now >= self.reach_at:
            return self.target_c
        return self.segment_at(now).expected_c(now)

    def as_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "source_type": self.source_type,
            "policy": self.policy,
            "curve_shape": CURVE_SHAPE,
            "required": self.required,
            "current_c": round(self.current_c, 2),
            "target_c": round(self.target_c, 2),
            "delta_c": round(self.delta_c, 2),
            "rate_c_s": round(self.rate_c_s, 4),
            "lead_s": round(self.lead_s, 1),
            "ramp_s": round(self.ramp_s, 1),
            "issued_at": round(self.issued_at, 1),
            "start_at": round(self.start_at, 1),
            "reach_at": round(self.reach_at, 1),
            "deadline_at": None if self.deadline_at is None else round(self.deadline_at, 1),
            "segments": [segment.as_dict() for segment in self.segments],
            "curve": self.curve.as_dict(),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "WarmupPlan":
        """从落盘字段还原一份安排；形状不是定死的分段线性时拒绝还原。"""

        if str(raw.get("curve_shape", CURVE_SHAPE)) != CURVE_SHAPE:
            raise ValueError(f"未知起调曲线形状 {raw.get('curve_shape')!r}，只支持 {CURVE_SHAPE}")
        curve = CurveRef.from_dict(raw["curve"])
        return cls(
            plan_id=str(raw["plan_id"]),
            source_type=str(raw["source_type"]),
            policy=str(raw["policy"]),
            current_c=float(raw["current_c"]),
            target_c=float(raw["target_c"]),
            delta_c=float(raw["delta_c"]),
            rate_c_s=float(raw["rate_c_s"]),
            lead_s=float(raw["lead_s"]),
            ramp_s=float(raw["ramp_s"]),
            issued_at=float(raw["issued_at"]),
            start_at=float(raw["start_at"]),
            reach_at=float(raw["reach_at"]),
            deadline_at=(None if raw.get("deadline_at") is None else float(raw["deadline_at"])),
            segments=tuple(RampSegment.from_dict(item) for item in raw.get("segments", [])),
            curve=curve,
            required=bool(raw.get("required", True)),
            note=str(raw.get("note", "")),
        )


def policy_for(source_type: str) -> str:
    """按热源类型给出定死的保守策略。"""

    try:
        return SOURCE_POLICY[source_type]
    except KeyError:
        raise ValueError(f"未知热源类型 {source_type!r}")


def effective_rate_c_s(
    policy: str, rated_rate_c_s: float, observed_rates_c_s: list[float] | None = None
) -> float:
    """按策略计算安排采用的升温速率（℃/s），不允许超过额定联锁上限。"""

    rated = float(rated_rate_c_s)
    floor = rated * GUARANTEED_RATE_FRACTION
    if policy == POLICY_GUARANTEED:
        return floor
    if policy == POLICY_HISTORICAL_MEAN:
        samples = [abs(float(item)) for item in (observed_rates_c_s or []) if item and item > 0]
        if not samples:
            # 没有历史观测时退回额定值：设备联锁上限是唯一可信口径。
            return rated
        return min(rated, max(floor, sum(samples) / len(samples)))
    raise ValueError(f"未知起调策略 {policy!r}")


def build_warmup_plan(
    *,
    plan_id: str,
    source_type: str,
    current_c: float,
    target_c: float,
    rated_rate_c_s: float,
    now: float,
    curve_epoch: int,
    curve_outdoor_c: float,
    deadline_at: float | None = None,
    observed_rates_c_s: list[float] | None = None,
) -> WarmupPlan:
    """按当前供温、目标供温与升温速率算出一份起调安排。

    有 ``deadline_at``（必须到目标的最晚时刻）时，安排倒排：起调时刻 = 最晚时刻
    - 升温耗时 - 起调提前量；不给期限时给一份"立即起调"的正排建议，同时算出按
    保守口径必须的提前量供控制台提示。
    """

    current = float(current_c)
    target = float(target_c)
    moment = float(now)
    delta = target - current
    if delta <= TARGET_BAND_C:
        return WarmupPlan(
            plan_id=str(plan_id),
            source_type=source_type,
            policy=policy_for(source_type),
            current_c=current,
            target_c=target,
            delta_c=0.0,
            rate_c_s=float(rated_rate_c_s),
            lead_s=0.0,
            ramp_s=0.0,
            issued_at=moment,
            start_at=moment,
            reach_at=moment,
            deadline_at=None if deadline_at is None else float(deadline_at),
            segments=(),
            curve=CurveRef(curve_epoch, target, float(curve_outdoor_c)),
            required=False,
            note="当前供温不低于目标，无需起调",
        )

    policy = policy_for(source_type)
    rate = effective_rate_c_s(policy, rated_rate_c_s, observed_rates_c_s)
    ramp_s = delta / rate
    lead_s = ramp_s * LEAD_MARGIN[policy]

    if deadline_at is None:
        start_at = moment
        reach_at = moment + ramp_s
    else:
        latest_start = float(deadline_at) - ramp_s
        start_at = latest_start - lead_s
        reach_at = start_at + ramp_s

    count = min(MAX_SEGMENTS, max(MIN_SEGMENTS, int(-(-delta // SEGMENT_STEP_C))))
    segments: list[RampSegment] = []
    span = ramp_s / count
    for index in range(count):
        seg_start_c = current + delta * index / count
        seg_end_c = current + delta * (index + 1) / count
        segments.append(
            RampSegment(
                index=index,
                start_s=start_at + span * index,
                end_s=start_at + span * (index + 1),
                start_c=seg_start_c,
                end_c=seg_end_c,
                rate_c_s=(seg_end_c - seg_start_c) / span,
            )
        )

    note = "立即起调建议" if deadline_at is None else "按到目标期限倒排"
    if start_at < moment:
        note += "；已错过保守起调点，安排滞后"

    return WarmupPlan(
        plan_id=str(plan_id),
        source_type=source_type,
        policy=policy,
        current_c=current,
        target_c=target,
        delta_c=delta,
        rate_c_s=rate,
        lead_s=lead_s,
        ramp_s=ramp_s,
        issued_at=moment,
        start_at=start_at,
        reach_at=reach_at,
        deadline_at=None if deadline_at is None else float(deadline_at),
        segments=tuple(segments),
        curve=CurveRef(curve_epoch, target, float(curve_outdoor_c)),
        required=True,
        note=note,
    )


# ---- 执行状态机 -------------------------------------------------------

PENDING = "pending"
STARTED = "started"
WARMING = "warming"
REACHED = "reached"
CANCELLED = "cancelled"
WARMUP_STATES = (PENDING, STARTED, WARMING, REACHED, CANCELLED)


class WarmupRunner:
    """看守一份安排的执行时序：起调 → 升温 → 到目标。"""

    def __init__(self, plan: WarmupPlan) -> None:
        self._plan = plan
        self._state = PENDING if plan.required else REACHED
        self._segment = 0
        self._started_at: float | None = None
        self._reached_at: float | None = None

    @property
    def plan(self) -> WarmupPlan:
        return self._plan

    @property
    def plan_id(self) -> str:
        return self._plan.plan_id

    def state(self) -> str:
        return self._state

    def started_at(self) -> float | None:
        return self._started_at

    def reached_at(self) -> float | None:
        return self._reached_at

    def is_overdue(self, now: float) -> bool:
        """是否已超过到目标时刻（含宽限）却还没到目标。"""

        if self._state in (REACHED, CANCELLED):
            return False
        return float(now) > self._plan.reach_at + OVERDUE_GRACE_S

    def lagging(self, measured_c: float, now: float, band_c: float = TARGET_BAND_C) -> bool:
        """实测供温是否明显落后于安排在该时刻的分段线性期望值。"""

        if self._state not in (STARTED, WARMING):
            return False
        return float(measured_c) < self._plan.expected_c(now) - band_c

    def start(self, now: float) -> dict:
        """起调。只能从待执行开始；不允许跳过起调直接升温。"""

        if self._state == CANCELLED:
            raise InterlockError("安排已作废，禁止起调", component="source")
        if self._state != PENDING:
            raise InterlockError(f"安排当前为 {self._state}，不能重复起调", component="source")
        self._state = STARTED
        self._started_at = float(now)
        return {"plan_id": self.plan_id, "state": self._state, "started_at": round(float(now), 1)}

    def _enter_or_advance(self, index: int, now: float) -> str | None:
        """按段号进入下一段；时序不对直接拒绝。

        第一次调用从起调态进入第 0 段（起调 → 升温），之后每次只能推进到相邻的
        下一段，跳段、倒段都抛联锁。
        """

        if self._state == STARTED:
            if index != self._segment:
                raise InterlockError(
                    f"升温段顺序错乱：当前未起段，收到第 {index} 段（应为第 {self._segment} 段）",
                    component="source",
                )
            self._state = WARMING
            return f"enter_segment:{index}"
        if self._state != WARMING:
            raise InterlockError(f"安排当前为 {self._state}，不能进入升温段", component="source")
        expected = self._segment + 1
        if index != expected:
            raise InterlockError(
                f"升温段顺序错乱：应进入第 {expected} 段，收到第 {index} 段",
                component="source",
            )
        if index >= len(self._plan.segments):
            raise InterlockError("升温段已走完，应执行到目标确认", component="source")
        self._segment = index
        return f"advance_segment:{index}"

    def complete(self, measured_c: float, now: float, band_c: float = TARGET_BAND_C) -> dict:
        """到目标确认：实测必须落在目标带内，且只能在起调之后收尾。"""

        if self._state == CANCELLED:
            raise InterlockError("安排已作废，禁止确认到目标", component="source")
        if self._state == PENDING:
            raise InterlockError("尚未起调，禁止确认到目标", component="source")
        if self._state == REACHED:
            raise InterlockError("安排已到目标，不能重复确认", component="source")
        if abs(float(measured_c) - self._plan.target_c) > band_c:
            raise InterlockError(
                f"实测供温 {float(measured_c):.2f}℃ 未到目标 {self._plan.target_c:.2f}℃，禁止收尾",
                component="source",
            )
        self._state = REACHED
        self._reached_at = float(now)
        return {
            "plan_id": self.plan_id,
            "state": self._state,
            "reached_at": round(float(now), 1),
            "measured_c": round(float(measured_c), 2),
            "overdue": now > self._plan.reach_at + OVERDUE_GRACE_S,
        }

    def cancel(self, reason: str) -> dict:
        """作废安排（曲线换版或人工取消）。作废后冻结，不能复活。"""

        if self._state == REACHED:
            raise InterlockError("安排已到目标，无需作废", component="source")
        self._state = CANCELLED
        return {"plan_id": self.plan_id, "state": self._state, "reason": str(reason)}

    def track(self, measured_c: float, now: float) -> dict:
        """控制周期推进：按分段线性期望自动走段，并给出滞后/到期判定。

        只推进"起调之后、到目标之前"的段间推进；起调本身必须由值班员显式触发、
        到目标必须由值班员凭实测显式确认，保证时序是显式的
        起调 → 升温 → 到目标，这里不自动越权收尾。
        """

        moment = float(now)
        transitions: list[str] = []
        if self._state in (STARTED, WARMING):
            # 起调态先看当前段是否已到点；升温态从下一段开始看是否该推进。
            first = self._segment if self._state == STARTED else self._segment + 1
            for segment in self._plan.segments[first:]:
                if moment + 1e-9 < segment.start_s:
                    break
                entered = self._enter_or_advance(segment.index, moment)
                if entered:
                    transitions.append(entered)

        return {
            "plan_id": self.plan_id,
            "state": self._state,
            "segment": self._segment,
            "expected_c": round(self._plan.expected_c(moment), 2),
            "measured_c": round(float(measured_c), 2),
            "lagging": self.lagging(measured_c, moment),
            "overdue": self.is_overdue(moment),
            "transitions": transitions,
        }

    def as_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "state": self._state,
            "segment": self._segment,
            "started_at": None if self._started_at is None else round(self._started_at, 1),
            "reached_at": None if self._reached_at is None else round(self._reached_at, 1),
            "plan": self._plan.as_dict(),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "WarmupRunner":
        """从落盘字段还原一个执行器。"""

        runner = cls(WarmupPlan.from_dict(raw["plan"]))
        state = str(raw.get("state", PENDING))
        if state not in WARMUP_STATES:
            raise ValueError(f"未知起调执行状态 {state!r}")
        runner._state = state
        runner._segment = int(raw.get("segment", 0))
        runner._started_at = None if raw.get("started_at") is None else float(raw["started_at"])
        runner._reached_at = None if raw.get("reached_at") is None else float(raw["reached_at"])
        return runner
