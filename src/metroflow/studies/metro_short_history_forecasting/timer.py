from __future__ import annotations

from .common import *

class _StageTimer:
    """Collect wall-clock timings for the main experiment stages."""

    def __init__(self, echo: bool = True):
        self.echo = bool(echo)
        self.rows: list[dict[str, object]] = []
        self._order = 0

    def _append(self, stage: str, seconds: float, status: str = "ok") -> None:
        self._order += 1
        seconds = float(seconds)
        self.rows.append({
            "order": int(self._order),
            "stage": stage,
            "seconds": seconds,
            "minutes": seconds / 60.0,
            "status": status,
        })
        if self.echo:
            suffix = "" if status == "ok" else f" [{status}]"
            print(f"[timing] {stage}: {seconds:.2f} s ({seconds / 60.0:.2f} min){suffix}", flush=True)

    @contextmanager
    def track(self, stage: str):
        started = perf_counter()
        status = "ok"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            self._append(stage, perf_counter() - started, status=status)

    def record(self, stage: str, seconds: float, status: str = "ok") -> None:
        self._append(stage, seconds, status=status)

    def to_frame(self) -> pd.DataFrame:
        table = pd.DataFrame(self.rows)
        if table.empty:
            return table
        total_mask = table["stage"].eq("TOTAL run_multistation_forecasting_experiment")
        if total_mask.any():
            total_seconds = float(table.loc[total_mask, "seconds"].iloc[-1])
        else:
            total_seconds = float(table["seconds"].sum())
        table["share_of_total_pct"] = np.where(
            total_seconds > 0,
            100.0 * table["seconds"].astype(float) / total_seconds,
            np.nan,
        )
        return table.sort_values("order").reset_index(drop=True)

__all__ = [
    "_StageTimer",
]
