from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from prometheus_client import Gauge, start_http_server


RMSE_GAUGE = Gauge("rossmann_model_rmse", "Latest RMSE from monitoring report")
MAE_GAUGE = Gauge("rossmann_model_mae", "Latest MAE from monitoring report")
RMSPE_GAUGE = Gauge("rossmann_model_val_rmspe", "Latest validation RMSPE from monitoring report")
MAX_PSI_GAUGE = Gauge("rossmann_model_drift_max_psi", "Maximum PSI value in latest monitoring report")
SEVERE_DRIFT_COUNT_GAUGE = Gauge(
    "rossmann_model_drift_severe_count", "Number of severe drift features in latest monitoring report"
)
ALERT_FLAG_GAUGE = Gauge("rossmann_model_alert_active", "1 if latest monitoring report contains alert, otherwise 0")


def _resolve_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _read_last_json_line(file_path: Path) -> dict[str, Any] | None:
    if not file_path.exists():
        return None

    lines = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return None

    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None
    return payload


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _update_metrics_from_report(report: dict[str, Any]) -> None:
    performance = report.get("performance", {})
    if isinstance(performance, dict):
        rmse = _safe_float(performance.get("rmse"))
        mae = _safe_float(performance.get("mae"))
        val_rmspe = _safe_float(performance.get("val_rmspe"))
        if rmse is not None:
            RMSE_GAUGE.set(rmse)
        if mae is not None:
            MAE_GAUGE.set(mae)
        if val_rmspe is not None:
            RMSPE_GAUGE.set(val_rmspe)

    drift_items = report.get("drift", [])
    if isinstance(drift_items, list):
        psi_values: list[float] = []
        severe_count = 0
        for item in drift_items:
            if not isinstance(item, dict):
                continue
            psi = _safe_float(item.get("psi"))
            if psi is not None:
                psi_values.append(psi)
            if item.get("status") == "severe_drift":
                severe_count += 1

        if psi_values:
            MAX_PSI_GAUGE.set(max(psi_values))
        SEVERE_DRIFT_COUNT_GAUGE.set(float(severe_count))

    alert_text = report.get("alert")
    ALERT_FLAG_GAUGE.set(1.0 if alert_text else 0.0)


def main() -> None:
    report_file = _resolve_path(os.getenv("MONITORING_REPORT_FILE", "logs/monitoring_report.jsonl"))
    listen_port = int(os.getenv("MODEL_EXPORTER_PORT", "9108"))
    poll_seconds = int(os.getenv("MODEL_EXPORTER_POLL_SECONDS", "15"))

    start_http_server(listen_port)
    while True:
        report = _read_last_json_line(report_file)
        if report is not None:
            _update_metrics_from_report(report)
        time.sleep(max(5, poll_seconds))


if __name__ == "__main__":
    main()
