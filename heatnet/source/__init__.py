"""热源调质调量。

热源必须先调质（把一次网供水温度提到曲线给定的目标）再调量（把循环流量提到
目标），两步都由 :class:`~heatnet.source.source.HeatSource` 的状态机看守；调整
要么整体生效、要么整体不生效，中间态不允许被下游读到。

升温起调安排见 :mod:`heatnet.source.warmup`：按当前供温、目标供温与升温速率
算出"什么时候起调、分几段爬、每段到多少"，并由状态机看守起调 → 升温 → 到目标
的时序。
"""

from __future__ import annotations

from .adjust import Adjustment, apply_adjustment
from .source import HeatSource
from .state import SourceSnapshot
from .warmup import (
    CURVE_SHAPE,
    POLICY_GUARANTEED,
    POLICY_HISTORICAL_MEAN,
    RampSegment,
    WarmupPlan,
    WarmupRunner,
    build_warmup_plan,
    effective_rate_c_s,
    policy_for,
)
from .weather import WeatherPush, push_weather

__all__ = [
    "Adjustment",
    "CURVE_SHAPE",
    "HeatSource",
    "POLICY_GUARANTEED",
    "POLICY_HISTORICAL_MEAN",
    "RampSegment",
    "SourceSnapshot",
    "WarmupPlan",
    "WarmupRunner",
    "WeatherPush",
    "apply_adjustment",
    "build_warmup_plan",
    "effective_rate_c_s",
    "policy_for",
    "push_weather",
]
