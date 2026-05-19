"""Lightweight HTML dashboard renderer (no external JS dependencies)."""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional

import pandas as pd


class DashboardRenderer:
    def __init__(self, target_disease: str) -> None:
        self.target_disease = target_disease

    def render(
        self,
        lgdi_timeline: pd.DataFrame,
        rdi_timeline: Optional[pd.DataFrame] = None,
        alerts: Optional[pd.DataFrame] = None,
        per_group_heatmap: Optional[pd.DataFrame] = None,
        out_path: str | Path = "dashboard.html",
    ) -> Path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_path = Path(out_path)

        def _fig_to_b64(fig) -> str:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
            plt.close(fig)
            return base64.b64encode(buf.getvalue()).decode("ascii")

        imgs: list[tuple[str, str]] = []

        if not lgdi_timeline.empty:
            fig, ax = plt.subplots(figsize=(9, 3.2))
            t = lgdi_timeline.sort_values("week")
            ax.plot(pd.to_datetime(t["week"]), t["lgdi"], "-o", lw=1.0, ms=3, color="#c0392b")
            ax.axhline(0, color="grey", lw=0.6, ls="--")
            ax.set_title(f"{self.target_disease} — LGDI timeline")
            ax.set_ylabel("LGDI")
            imgs.append(("LGDI timeline", _fig_to_b64(fig)))

        if rdi_timeline is not None and not rdi_timeline.empty:
            fig, ax = plt.subplots(figsize=(9, 3.2))
            t = rdi_timeline.sort_values("week")
            ax.plot(pd.to_datetime(t["week"]), t["rdi"], "-o", lw=1.0, ms=3, color="#2c3e50")
            ax.axhline(0, color="grey", lw=0.6, ls="--")
            ax.set_title(f"{self.target_disease} — Pearson RDI timeline")
            ax.set_ylabel("RDI")
            imgs.append(("Pearson RDI timeline", _fig_to_b64(fig)))

        if per_group_heatmap is not None and not per_group_heatmap.empty:
            pivot = per_group_heatmap.pivot_table(index="group", columns="week", values="S")
            fig, ax = plt.subplots(figsize=(10, max(2, 0.4 * len(pivot))))
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlBu_r")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xticks([])
            ax.set_title(f"{self.target_disease} — per-group MASE heatmap")
            fig.colorbar(im, ax=ax, label="MASE")
            imgs.append(("Per-group heatmap", _fig_to_b64(fig)))

        alert_rows = ""
        if alerts is not None and not alerts.empty:
            alert_rows = alerts.to_html(index=False, classes="alerts", border=0)

        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{self.target_disease} surveillance dashboard</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}}
h1{{font-size:1.4em}} h2{{font-size:1.1em;margin-top:28px;color:#444}}
.disclaimer{{background:#fff8e1;border:1px solid #ffc107;padding:10px 14px;border-radius:6px;font-size:0.92em}}
img{{max-width:100%;border:1px solid #eee;border-radius:4px}}
table.alerts{{border-collapse:collapse;font-size:0.9em}}
table.alerts td,table.alerts th{{border:1px solid #ddd;padding:4px 8px}}
</style></head><body>
<h1>{self.target_disease} — Surveillance Dashboard</h1>
<p class="disclaimer">Retrospective proof-of-concept. Not a medical device. Not validated for
prospective surveillance. Data and alerts are derived from user-supplied EHR records.</p>
""" + "\n".join(
            f'<h2>{title}</h2><img src="data:image/png;base64,{b64}"/>'
            for title, b64 in imgs
        ) + (f'<h2>Alerts</h2>{alert_rows}' if alert_rows else "") + "</body></html>"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        return out_path
