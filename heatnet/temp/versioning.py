"""供热曲线版本台账。

曲线每次换版都是一次"版本变化"，必须留得住：已下发的起调安排按它下发那一刻的
版本解释，事后曲线再改版也不能回头改老安排。:class:`CurveVersion` 因此是不可变
快照，冻结当版的室外温度、目标供温、断点表与生成时刻；:class:`CurveVersionRegistry`
按代际（epoch）顺序追加，永不就地改写。回放在审计流水之外另留一条独立查询路径，
控制台用它回答"这份安排当时依据的是哪一版"。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CurveVersion:
    """某一版曲线的不可变快照。"""

    epoch: int
    outdoor_c: float
    supply_c: float
    issued_at: float
    points: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    note: str = ""

    @classmethod
    def create(
        cls,
        epoch: int,
        outdoor_c: float,
        supply_c: float,
        issued_at: float,
        points: tuple[tuple[float, float], ...] = (),
        note: str = "",
    ) -> "CurveVersion":
        return cls(
            epoch=int(epoch),
            outdoor_c=float(outdoor_c),
            supply_c=float(supply_c),
            issued_at=float(issued_at),
            points=tuple((float(o), float(s)) for o, s in points),
            note=str(note),
        )

    def as_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "outdoor_c": round(self.outdoor_c, 2),
            "supply_c": round(self.supply_c, 2),
            "issued_at": round(self.issued_at, 3),
            "points": [{"outdoor_c": round(o, 2), "supply_c": round(s, 2)} for o, s in self.points],
            "note": self.note,
        }


class CurveVersionRegistry:
    """按代际登记曲线版本，只追加、不改写。"""

    def __init__(self) -> None:
        self._versions: list[CurveVersion] = []
        self._lock = threading.RLock()

    def register(self, version: CurveVersion) -> CurveVersion:
        """登记一版；epoch 必须严格递增，重复登记同一版原样返回。"""

        with self._lock:
            if self._versions and version.epoch == self._versions[-1].epoch:
                existing = self._versions[-1]
                if existing != version:
                    raise ValueError(f"曲线代际 {version.epoch} 已登记且内容不一致，禁止覆盖")
                return existing
            if self._versions and version.epoch < self._versions[-1].epoch:
                raise ValueError("曲线代际不能回退")
            self._versions.append(version)
            return version

    def current(self) -> CurveVersion | None:
        """返回当前最新一版。"""

        with self._lock:
            return self._versions[-1] if self._versions else None

    def get(self, epoch: int) -> CurveVersion | None:
        """按代际取一版，查不到返回 ``None``。"""

        with self._lock:
            for version in self._versions:
                if version.epoch == int(epoch):
                    return version
        return None

    def history(self, limit: int | None = None) -> list[dict]:
        """按时间升序返回版本清单。"""

        with self._lock:
            versions = list(self._versions)
        if limit is not None:
            versions = versions[-max(0, int(limit)) :]
        return [version.as_dict() for version in versions]

    def count(self) -> int:
        """返回已登记版本数。"""

        with self._lock:
            return len(self._versions)

    def as_list(self) -> list[dict]:
        """导出全部版本（按时间升序），供落盘。"""

        with self._lock:
            return [version.as_dict() for version in self._versions]

    def absorb(self, raw: list[dict]) -> int:
        """把落盘版本合并进本台账（装配时的基线已在其中），返回版本数。

        落盘里可能也带着第 0 版基线：与当前基线代际相同时直接沿用，不重复登记；
        其余版本按 epoch 顺序追加。
        """

        for item in raw:
            version = CurveVersion.create(
                epoch=int(item["epoch"]),
                outdoor_c=float(item["outdoor_c"]),
                supply_c=float(item["supply_c"]),
                issued_at=float(item["issued_at"]),
                points=tuple(
                    (float(point["outdoor_c"]), float(point["supply_c"]))
                    for point in item.get("points", [])
                ),
                note=str(item.get("note", "")),
            )
            with self._lock:
                if any(existing.epoch == version.epoch for existing in self._versions):
                    continue
            self.register(version)
        return self.count()
