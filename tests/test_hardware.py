"""Tests for CUDA arch compatibility logic (P0-T02)."""

from __future__ import annotations

import pytest

from desertification.hardware import arch_list_supports, parse_arch

# Arch list compiled into torch 2.14.1+cu126 (read from libtorch_cuda.so, 2026-11-20).
CU126_ARCHS = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
GTX_1080_TI = (6, 1)


def test_parse_arch():
    assert parse_arch("sm_61") == ("sm", 6, 1)
    assert parse_arch("compute_90") == ("compute", 9, 0)
    assert parse_arch("sm_90a") == ("sm", 9, 0)
    assert parse_arch("sm_120") == ("sm", 12, 0)


def test_parse_arch_rejects_garbage():
    with pytest.raises(ValueError):
        parse_arch("pascal")


def test_cu126_build_supports_1080_ti():
    # sm_60 binaries run on 6.1 devices (same major, lower minor).
    assert arch_list_supports(GTX_1080_TI, CU126_ARCHS)


def test_build_without_pascal_does_not_support_1080_ti():
    assert not arch_list_supports(GTX_1080_TI, ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100"])


def test_higher_minor_binary_does_not_run_on_lower_minor_device():
    assert not arch_list_supports((6, 0), ["sm_61"])


def test_ptx_jit_covers_newer_devices_only():
    assert arch_list_supports((8, 9), ["compute_80"])
    assert not arch_list_supports((6, 1), ["compute_70"])
