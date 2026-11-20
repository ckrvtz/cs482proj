"""Tests for the quota ledger and the secret-leak check (P0-T03)."""

from __future__ import annotations

from datetime import date

import pytest

from desertification.config import get_path, load_config
from desertification.security import find_leaks, parse_env_text, secrets_to_check
from desertification.usage import (
    COLUMNS,
    append_usage,
    check_cap,
    init_usage_log,
    latest_balance,
    make_entry,
    pu_used_by_category,
    read_usage_log,
)

DAY = date(2026, 11, 20)


# --- usage ledger -----------------------------------------------------------------------


def test_init_creates_starting_balance_once(tmp_path):
    path = tmp_path / "log.csv"
    assert init_usage_log(path, 30000, 30000, on=DAY) is True
    assert init_usage_log(path, 1, 1, on=DAY) is False  # existing file untouched
    rows = read_usage_log(path)
    assert len(rows) == 1
    assert rows[0]["api"] == "balance"
    assert rows[0]["pu_after"] == 30000 and rows[0]["requests_after"] == 30000
    assert path.read_text().splitlines()[0] == ",".join(COLUMNS)


def test_make_entry_validation():
    with pytest.raises(ValueError):
        make_entry("P0-T03", "catalog", "not-a-category", 1, "x")
    with pytest.raises(ValueError):
        make_entry("P0-T03", "catalog", "setup_pilot", -1, "x")
    with pytest.raises(ValueError):
        make_entry("P0-T03", "catalog", "setup_pilot", 1, "x", pu_before=10, pu_after=20)
    entry = make_entry("P0-T03", "catalog", "setup_pilot", 2, "x", on=DAY)
    assert entry["pu_before"] == "" and entry["date"] == "2026-11-20"


def test_append_balances_and_caps(tmp_path):
    path = tmp_path / "log.csv"
    init_usage_log(path, 30000, 30000, on=DAY)
    append_usage(
        path,
        make_entry("M1-T05", "process", "setup_pilot", 4, "pilot", 30000, 29800, 30000, 29996, DAY),
    )
    append_usage(path, make_entry("M1-T05", "catalog", "setup_pilot", 1, "no reading", on=DAY))
    rows = read_usage_log(path)
    assert latest_balance(rows) == {"pu": 29800, "requests": 29996}
    assert pu_used_by_category(rows)["setup_pilot"] == 200
    caps = {"setup_pilot": 1000, "extraction": 20000, "auxiliary_eda": 2000, "reserve": 7000}
    assert check_cap(rows, "setup_pilot", 700, caps) == (True, 100)
    assert check_cap(rows, "setup_pilot", 900, caps) == (False, -100)


def test_append_requires_existing_log(tmp_path):
    with pytest.raises(FileNotFoundError):
        append_usage(tmp_path / "missing.csv", make_entry("x", "catalog", "setup_pilot", 1, "x"))


def test_committed_ledger_has_starting_balance():
    rows = read_usage_log(get_path(load_config(), "api_usage_log"))
    assert rows[0]["api"] == "balance"
    assert latest_balance(rows[:1]) == {"pu": 30000, "requests": 30000}


# --- secret check -----------------------------------------------------------------------


def test_parse_env_text():
    text = '# comment\nSH_CLIENT_ID=abc123456\nexport SH_CLIENT_SECRET="s3cr3t-value"\n\nEMPTY=\n'
    assert parse_env_text(text) == {
        "SH_CLIENT_ID": "abc123456",
        "SH_CLIENT_SECRET": "s3cr3t-value",
        "EMPTY": "",
    }


def test_secrets_to_check_skips_short_and_empty():
    assert secrets_to_check({"A": "short", "B": "", "C": "long-enough-value"}) == [
        "long-enough-value"
    ]


def test_find_leaks_reports_files_not_values():
    files = {"a.py": "x = 1", "nb.ipynb": '{"outputs": ["token s3cr3t-value"]}', "b.md": ""}
    assert find_leaks(files, ["s3cr3t-value"]) == ["nb.ipynb"]
    assert find_leaks(files, []) == []
