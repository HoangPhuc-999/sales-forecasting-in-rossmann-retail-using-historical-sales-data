from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import requests
from fastapi import FastAPI
from pydantic import BaseModel


class AlertItem(BaseModel):
    status: str
    labels: dict[str, str] = {}
    annotations: dict[str, str] = {}
    startsAt: str | None = None
    endsAt: str | None = None


class AlertManagerPayload(BaseModel):
    receiver: str | None = None
    status: str | None = None
    alerts: list[AlertItem] = []
    groupLabels: dict[str, str] = {}
    commonLabels: dict[str, str] = {}
    commonAnnotations: dict[str, str] = {}


app = FastAPI(title="Rossmann Alert Notifier", version="0.1.0")


def _build_message(payload: AlertManagerPayload) -> str:
    timestamp = datetime.utcnow().isoformat()
    lines = [f"[Rossmann Alert] {payload.status or 'unknown'} at {timestamp} UTC"]

    for idx, alert in enumerate(payload.alerts, start=1):
        alert_name = alert.labels.get("alertname", "unnamed_alert")
        summary = alert.annotations.get("summary", "")
        description = alert.annotations.get("description", "")
        severity = alert.labels.get("severity", "unknown")
        lines.append(f"{idx}. {alert_name} | severity={severity} | status={alert.status}")
        if summary:
            lines.append(f"   summary: {summary}")
        if description:
            lines.append(f"   desc: {description}")

    return "\n".join(lines)


def _send_slack(message: str) -> None:
    slack_webhook = os.getenv("SLACK_WEBHOOK_URL", "").strip()
    if not slack_webhook:
        return

    requests.post(slack_webhook, json={"text": message}, timeout=10).raise_for_status()


def _send_telegram(message: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        return

    telegram_url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    requests.post(telegram_url, json=payload, timeout=10).raise_for_status()


@app.post("/alert")
def receive_alert(payload: AlertManagerPayload) -> dict[str, Any]:
    message = _build_message(payload)

    errors: list[str] = []
    for sender in (_send_slack, _send_telegram):
        try:
            sender(message)
        except Exception as exc:  # pragma: no cover
            errors.append(str(exc))

    return {
        "received": len(payload.alerts),
        "forwarded": len(errors) == 0,
        "errors": errors,
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
