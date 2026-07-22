"""Smoke tests for the first modular Canyon package."""

from canyon_app.core.formatting import compact_text, money, pct, score, to_float
from canyon_app.core.paths import PROJECT_ROOT, project_file
from canyon_app.data.loaders import safe_csv, safe_json
from canyon_app.domain.performance.labels import performance_accent, performance_plain, signal_label
from canyon_app.domain.risk.labels import risk_accent, risk_plain, risk_status_label


def test_formatting_helpers():
    assert to_float("1,234.5%") == 1234.5
    assert compact_text("REDUCE_ONLY") == "REDUCE ONLY"
    assert pct(0.5, already_pct=False) == "50.0%"
    assert score(54.93) == "54.9 / 100"
    assert money(3039.11) == "$3,039"


def test_project_paths_exist():
    assert PROJECT_ROOT.exists()
    assert project_file("canyon_final_v9_step86_dashboard_v3.py").exists()


def test_safe_loaders_do_not_crash_on_missing_files():
    missing = PROJECT_ROOT / "__missing_file_for_test__"
    assert safe_json(missing) == {}
    assert safe_csv(missing).empty


def test_risk_labels_hide_raw_codes():
    text = risk_plain("REDUCE_ONLY; DATA_GAP; NO_NEW_OPTION")
    assert "REDUCE_ONLY" not in text
    assert "DATA_GAP" not in text
    assert "NO_NEW_OPTION" not in text
    assert "no new buying" in text
    assert risk_status_label("SIZE_DOWN") == "Use smaller size"
    assert risk_accent("REDUCE_ONLY") == "#991b1b"


def test_performance_labels_hide_raw_codes():
    text = performance_plain("PROTOTYPE_ONLY and PENDING_FORWARD_RETURNS")
    assert "PROTOTYPE_ONLY" not in text
    assert "PENDING_FORWARD_RETURNS" not in text
    assert "prototype evidence only" in text
    assert signal_label("mom_12m_skip1m") == "12-month momentum"
    assert performance_accent(21, "NOT RELIABLE") == "#991b1b"
