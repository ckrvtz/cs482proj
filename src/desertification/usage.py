"""CDSE quota ledger: ``data/api_usage_log.csv`` (ROADMAP §3.4).

One row per batch of API requests. The ``*_before`` / ``*_after`` balances are copied by a
person from the CDSE Sentinel Hub dashboard; they may be left blank and filled in later.
The first row records the starting balance.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Any

COLUMNS = [
    "date",
    "task_id",
    "api",
    "budget_category",
    "request_count",
    "pu_before",
    "pu_after",
    "requests_before",
    "requests_after",
    "purpose",
]
BALANCE_API = "balance"
CATEGORIES = ("setup_pilot", "extraction", "auxiliary_eda", "reserve")
_NUMERIC = ("request_count", "pu_before", "pu_after", "requests_before", "requests_after")


def _to_number(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    return float(value)


def make_entry(
    task_id: str,
    api: str,
    budget_category: str,
    request_count: int,
    purpose: str,
    pu_before: float | None = None,
    pu_after: float | None = None,
    requests_before: float | None = None,
    requests_after: float | None = None,
    on: date | None = None,
) -> dict[str, Any]:
    """Build and validate one ledger row.

    Parameters
    ----------
    task_id : str
        Roadmap task, e.g. ``"P0-T03"``.
    api : str
        ``"catalog"``, ``"process"``, ``"statistical"`` or ``"balance"``.
    budget_category : str
        One of ``CATEGORIES`` (the §3.4 budget rows).
    request_count : int
        HTTP requests this code made in the batch (>= 0), including OAuth token requests.
        The quota charge can be lower (CDSE did not charge the token request in P0-T03);
        the charged amount is the difference of the dashboard balances.
    purpose : str
        Short description.
    pu_before, pu_after, requests_before, requests_after : float, optional
        Dashboard balances (remaining units) before/after the batch.
    on : date, optional
        Defaults to today.

    Returns
    -------
    dict
        Row keyed by ``COLUMNS``; unknown balances are empty strings.
    """
    if budget_category not in CATEGORIES:
        raise ValueError(f"budget_category must be one of {CATEGORIES}, got {budget_category!r}")
    if request_count < 0:
        raise ValueError("request_count must be >= 0")
    for label, before, after in (
        ("pu", pu_before, pu_after),
        ("requests", requests_before, requests_after),
    ):
        if before is not None and after is not None and after > before:
            raise ValueError(f"{label}_after ({after}) is greater than {label}_before ({before})")
    entry = {
        "date": (on or date.today()).isoformat(),
        "task_id": task_id,
        "api": api,
        "budget_category": budget_category,
        "request_count": int(request_count),
        "pu_before": pu_before,
        "pu_after": pu_after,
        "requests_before": requests_before,
        "requests_after": requests_after,
        "purpose": purpose,
    }
    return {k: ("" if v is None else v) for k, v in entry.items()}


def init_usage_log(
    path: Path, pu_total: float, requests_total: float, on: date | None = None
) -> bool:
    """Create the ledger with a starting-balance row if it does not exist yet.

    Returns
    -------
    bool
        True if the file was created, False if it already existed (left untouched).
    """
    if path.exists():
        return False
    start = make_entry(
        task_id="P0-T03",
        api=BALANCE_API,
        budget_category="setup_pilot",
        request_count=0,
        purpose="Starting balance on the CDSE account",
        pu_before=pu_total,
        pu_after=pu_total,
        requests_before=requests_total,
        requests_after=requests_total,
        on=on,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerow(start)
    return True


def read_usage_log(path: Path) -> list[dict[str, Any]]:
    """Read the ledger; numeric columns become floats (or None when blank)."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        for col in _NUMERIC:
            row[col] = _to_number(row.get(col))
    return rows


def append_usage(path: Path, entry: dict[str, Any]) -> None:
    """Append one row built by :func:`make_entry` to an existing ledger."""
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; call init_usage_log first")
    if set(entry) != set(COLUMNS):
        raise ValueError("entry must be built with make_entry")
    with open(path, "a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=COLUMNS).writerow(entry)


def latest_balance(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Most recent known remaining PU and request balances from the dashboard readings.

    Parameters
    ----------
    rows : list of dict
        Ledger rows from :func:`read_usage_log`, in file order.

    Returns
    -------
    dict
        ``{"pu": ..., "requests": ...}``; None if never recorded.
    """
    result: dict[str, float | None] = {"pu": None, "requests": None}
    for row in rows:
        for key in ("pu", "requests"):
            value = row.get(f"{key}_after")
            if value is None:
                value = row.get(f"{key}_before")
            if value is not None:
                result[key] = value
    return result


def pu_used_by_category(rows: list[dict[str, Any]]) -> dict[str, float]:
    """PU spent per budget category, from rows that have both before and after readings."""
    used = dict.fromkeys(CATEGORIES, 0.0)
    for row in rows:
        before, after = row.get("pu_before"), row.get("pu_after")
        if row.get("api") != BALANCE_API and before is not None and after is not None:
            used[row["budget_category"]] += before - after
    return used


def check_cap(
    rows: list[dict[str, Any]], category: str, estimate_pu: float, caps: dict[str, float]
) -> tuple[bool, float]:
    """Would a batch estimated at ``estimate_pu`` stay within its category's cap?

    Parameters
    ----------
    rows : list of dict
        Ledger rows.
    category : str
        Budget category of the planned batch.
    estimate_pu : float
        Estimated PU cost of the batch.
    caps : dict
        ``quota.pu_caps`` from the project config.

    Returns
    -------
    (bool, float)
        Whether it fits, and the PU left in that category after the batch.
    """
    left = caps[category] - pu_used_by_category(rows)[category] - estimate_pu
    return left >= 0, left
