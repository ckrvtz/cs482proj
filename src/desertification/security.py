"""Check that secret values from ``.env`` do not appear in files git would commit."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

# Values shorter than this are too generic to search for (e.g. "1", "true").
MIN_SECRET_LENGTH = 8


def parse_env_text(text: str) -> dict[str, str]:
    """Parse ``KEY=value`` lines from a ``.env`` file (comments and blanks ignored).

    Surrounding single or double quotes around a value are removed. ``export KEY=value``
    is accepted.
    """
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def secrets_to_check(env_values: Mapping[str, str], keys: Iterable[str] | None = None) -> list[str]:
    """Pick the secret values worth searching for.

    Parameters
    ----------
    env_values : mapping
        Parsed ``.env`` contents.
    keys : iterable of str, optional
        Restrict to these variable names; defaults to all of them.

    Returns
    -------
    list of str
        Non-empty values at least ``MIN_SECRET_LENGTH`` characters long.
    """
    names = list(env_values) if keys is None else list(keys)
    return [env_values[k] for k in names if len(env_values.get(k, "") or "") >= MIN_SECRET_LENGTH]


def find_leaks(files: Mapping[str, str], secrets: Iterable[str]) -> list[str]:
    """Return the names of files whose text contains any secret value.

    Parameters
    ----------
    files : mapping of str to str
        File path -> file text.
    secrets : iterable of str
        Secret values to look for.

    Returns
    -------
    list of str
        Sorted file paths that contain a secret. The secrets themselves are never returned.
    """
    secrets = [s for s in secrets if s]
    return sorted(path for path, text in files.items() if any(s in text for s in secrets))
