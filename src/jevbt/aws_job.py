"""Scheduled AWS job for forward paper trading (EventBridge Scheduler → ECS Fargate task → Alpaca paper + SNS email).

Container entry point: `python -m jevbt.aws_job trade|report` (exit code 1 on failure). `handler(event, context)`
also works as a Lambda handler. Two scheduled modes, Monday–Friday (America/New_York):
  {"mode": "trade"}  09:00 — on the first session of the week only: compute targets and submit market-on-open
                     orders (paper.run_paper), store the run log in S3, email the submitted orders (or "no orders").
  {"mode": "report"} 10:00 — after the open: email today's jevbt fills (positions opened / closed) and any order
                     that was not filled. Silent when there were no jevbt orders today.
Any exception is emailed and re-raised (so CloudWatch counts the error).

Configuration (Lambda environment): JEVBT_TICKERS (comma-separated), JEVBT_STRATEGY (trend|baseline),
JEVBT_DIRECTION, JEVBT_MAX_ALLOC, JEVBT_MAX_GROSS, JEVBT_VOL_SIZING, JEVBT_TIF, JEVBT_LOG_BUCKET,
JEVBT_SNS_TOPIC_ARN, JEVBT_SSM_PREFIX (SecureString parameters <prefix>alpaca_api_key_id / alpaca_api_secret_key),
JEVBT_DATA_DIR (/tmp/data), ALPACA_DATA_FEED. AWS credentials come from the task role.
"""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
SECRETS = {"alpaca_api_key_id": "ALPACA_API_KEY_ID", "alpaca_api_secret_key": "ALPACA_API_SECRET_KEY"}


@dataclass
class Deps:
    """Injectable side effects (tests replace them; Lambda builds the real ones lazily)."""

    publish: Callable[[str, str], None] | None = None   # (subject, message) → SNS email
    upload: Callable[[Path], str] | None = None         # local file → S3 URI
    broker_factory: Callable | None = None              # settings → AlpacaPaperBroker
    run_paper: Callable | None = None
    today: date | None = None
    extra: dict = field(default_factory=dict)


# ---------- configuration and AWS clients ----------

def _env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f"missing environment variable {name}")
    return value


def load_secrets_from_ssm(ssm=None) -> None:
    """Copy the Alpaca keys from SSM SecureString parameters into the environment (once per container)."""
    if all(os.environ.get(v) for v in SECRETS.values()):
        return
    if ssm is None:
        import boto3  # provided by the Lambda runtime

        ssm = boto3.client("ssm")
    prefix = _env("JEVBT_SSM_PREFIX", "/jevbt/")
    resp = ssm.get_parameters(Names=[prefix + n for n in SECRETS], WithDecryption=True)
    found = {p["Name"][len(prefix):]: p["Value"] for p in resp["Parameters"]}
    missing = [n for n in SECRETS if n not in found or found[n] in ("", "CHANGE_ME")]
    if missing:
        raise RuntimeError(f"SSM parameters not set: {', '.join(prefix + n for n in missing)}")
    for name, var in SECRETS.items():
        os.environ[var] = found[name]


def _sns_publisher() -> Callable[[str, str], None]:
    import boto3

    topic, sns = _env("JEVBT_SNS_TOPIC_ARN"), boto3.client("sns")
    return lambda subject, message: sns.publish(TopicArn=topic, Subject=subject[:100], Message=message)


def _s3_uploader() -> Callable[[Path], str]:
    import boto3

    bucket, s3 = _env("JEVBT_LOG_BUCKET"), boto3.client("s3")

    def upload(path: Path) -> str:
        key = f"paper/{path.name}"
        s3.upload_file(str(path), bucket, key)
        return f"s3://{bucket}/{key}"

    return upload


def build_strategy():
    from jevbt.strategy import BaselineStrategy, TrendConfidenceStrategy

    max_alloc = float(_env("JEVBT_MAX_ALLOC", str(1 / 12)))
    direction = _env("JEVBT_DIRECTION", "long")
    if _env("JEVBT_STRATEGY", "trend") == "baseline":
        return BaselineStrategy(max_alloc=max_alloc, direction=direction)
    return TrendConfidenceStrategy(max_alloc=max_alloc, direction=direction,
                                   vol_sizing=_env("JEVBT_VOL_SIZING", "false").lower() == "true")


# ---------- modes ----------

def trade(deps: Deps) -> dict:
    from jevbt.config import load_settings
    from jevbt.paper import first_session_of_week

    settings = load_settings()
    broker = deps.broker_factory(settings)
    today = deps.today
    sessions = broker.calendar((today - timedelta(days=7)).isoformat(), (today + timedelta(days=7)).isoformat())
    if not first_session_of_week(sessions, today):
        return {"mode": "trade", "skipped": f"{today} is not the first session of the week"}
    tickers = [t.strip().upper() for t in _env("JEVBT_TICKERS").split(",") if t.strip()]
    max_gross = float(_env("JEVBT_MAX_GROSS", "1.0"))
    run = deps.run_paper(settings, tickers, build_strategy(), max_gross=max_gross, submit=True,
                         time_in_force=_env("JEVBT_TIF", "opg"), broker=broker, today=today)
    log_uri = deps.upload(Path(run["log_path"]))
    orders = run["orders"]
    lines = [f"jevbt paper trading — {today} (first session of the week)",
             f"Account equity: {run['equity']:,.2f} USD | universe: {len(tickers)} stocks | max gross {max_gross:.0%}",
             f"Open positions before: {len(run['positions_before'])}", ""]
    if orders:
        lines.append("Orders submitted for today's open (market-on-open):")
        for o in orders:
            lines.append(f"  {o['side'].upper():4} {o['qty']:>6} {o['ticker']:<6} ({o['reason']}, ref close "
                         f"{o['ref_price']:.2f})  status: {o['status']}")
        lines.append("\nFills will be reported in a second email after the open (10:00 New York).")
    else:
        lines.append("No orders this week: no stock passed the entry rules and no open position hit an exit rule.")
    lines += [f"\nNotes: {'; '.join(run['notes'])}" if run["notes"] else "", f"Run log: {log_uri}"]
    subject = f"jevbt paper {today}: {len(orders)} order(s) submitted" if orders else f"jevbt paper {today}: no orders"
    deps.publish(subject, "\n".join(lines))
    return {"mode": "trade", "orders": len(orders), "log": log_uri}


def _describe(order: dict) -> str:
    """'OPENED LONG' / 'CLOSED SHORT' ... from the jevbt client_order_id (jevbt-YYYYMMDD-TICKER-reason-side)."""
    parts = order.get("client_order_id", "").split("-")
    reason, side = (parts[3], parts[4]) if len(parts) >= 5 else ("?", order.get("side", "?"))
    if reason == "open":
        return "OPENED LONG" if side == "buy" else "OPENED SHORT"
    if reason == "close":
        return "CLOSED LONG" if side == "sell" else "CLOSED SHORT"
    return "ADJUSTED"


def report(deps: Deps) -> dict:
    from jevbt.config import load_settings

    broker = deps.broker_factory(load_settings())
    today = deps.today
    start_utc = datetime.combine(today, datetime.min.time(), NY).astimezone(timezone.utc).isoformat()
    prefix = f"jevbt-{today:%Y%m%d}-"
    mine = [o for o in broker.orders(after=start_utc) if o.get("client_order_id", "").startswith(prefix)]
    if not mine:
        return {"mode": "report", "orders": 0}
    filled = [o for o in mine if o.get("status") == "filled"]
    other = [o for o in mine if o.get("status") != "filled"]
    lines = [f"jevbt paper trading — fills for {today}", ""]
    for o in sorted(filled, key=lambda o: o["symbol"]):
        qty, px = float(o["filled_qty"]), float(o["filled_avg_price"])
        lines.append(f"  {_describe(o):<13} {o['symbol']:<6} {qty:>6.0f} @ {px:,.2f}  (= {qty * px:,.0f} USD)")
    if other:
        lines.append("\nNot (fully) filled:")
        for o in other:
            lines.append(f"  {_describe(o):<13} {o['symbol']:<6} qty {o.get('qty')}  status: {o.get('status')}"
                         f"  filled {o.get('filled_qty', 0)}")
    account, positions = broker.account(), broker.positions()
    lines += ["", f"Account equity: {account['equity']:,.2f} USD | open positions: {len(positions)}"]
    opened = sum(_describe(o).startswith("OPENED") for o in filled)
    closed = sum(_describe(o).startswith("CLOSED") for o in filled)
    subject = f"jevbt paper {today}: {opened} opened, {closed} closed" + (f", {len(other)} NOT filled" if other else "")
    deps.publish(subject, "\n".join(lines))
    return {"mode": "report", "filled": len(filled), "not_filled": len(other)}


# ---------- entry point ----------

def handler(event, context, deps: Deps | None = None):
    deps = deps or Deps()
    mode = (event or {}).get("mode", "trade")
    deps.publish = deps.publish or _sns_publisher()
    try:
        load_secrets_from_ssm(deps.extra.get("ssm"))
        if deps.broker_factory is None:
            from jevbt.broker.alpaca_paper import AlpacaPaperBroker
            deps.broker_factory = AlpacaPaperBroker
        if deps.run_paper is None:
            from jevbt.paper import run_paper
            deps.run_paper = run_paper
        deps.upload = deps.upload or _s3_uploader()
        deps.today = deps.today or datetime.now(NY).date()
        if mode == "trade":
            return trade(deps)
        if mode == "report":
            return report(deps)
        raise ValueError(f"unknown mode {mode!r}")
    except SystemExit as e:  # run_paper refuses to submit (account blocked, not a session...)
        deps.publish(f"jevbt paper: {mode} refused", str(e))
        return {"mode": mode, "refused": str(e)}
    except Exception:
        deps.publish(f"jevbt paper: ERROR in {mode}", traceback.format_exc())
        raise


def main(argv: list[str] | None = None) -> int:
    """Container entry point: python -m jevbt.aws_job trade|report."""
    import json
    import sys

    args = sys.argv[1:] if argv is None else argv
    mode = args[0] if args else "trade"
    try:
        print(json.dumps(handler({"mode": mode}, None), default=str))
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
