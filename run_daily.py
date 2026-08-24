#!/usr/bin/env python3
"""
Canyon v9 — Daily Pipeline Runner
===================================
Run every morning with:  .venv/bin/python run_daily.py

Order of operations:
  0. Price & signals (step 0)  — downloads today's prices for all ~494 S&P 500
                                  tickers, recomputes momentum/RSI/trend/vol,
                                  refreshes regime_ml_scores.csv + alpha_scores.csv
  1. News fetch (step 99)      — pulls fresh news for all watchlist tickers
  2. Macro signals (step 95)   — updates HYG/TLT/GLD/UUP macro overlay
  3. Core pipeline (step 12)   — steps 3→10: prices, signals, risk, report
  4. Step 500 pipeline          — live signal snapshot + paper trading log
  5. P&L attribution (step 96) — updates attribution tables
  6. Daily alerts (step 98)    — generates desk monitor alerts
  7. Research HTML              — rebuilds canyon_v24_research.html
  8. Email summary              — sends daily briefing email (if configured)

Principles:
  - No broker connection.  No live orders.
  - Research and paper-tracking only.
  - Each step is run independently; one failure does not block the rest.
  - All output is logged to daily_runs/YYYYMMDD_HH MM.log
"""

from __future__ import annotations

import json
import os as _os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Auto-load .env from project root (never commit .env to git)
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            if _k.strip() and _v.strip() and _k.strip() not in _os.environ:
                _os.environ[_k.strip()] = _v.strip()

try:
    import yaml
    _cfg = yaml.safe_load(open(Path(__file__).parent / "config.yaml")) or {}
except Exception:
    _cfg = {}

ROOT   = Path(__file__).parent
# The project's .venv lives on the iCloud-synced Desktop; macOS evicts its files
# to iCloud ("dataless") under storage pressure, so every venv import can hang or
# time out (Errno 60) waiting on an iCloud download — which stalls the whole
# pipeline. Prefer the system framework Python (/Library, never iCloud-managed)
# so automation is iCloud-proof; fall back to the venv only if it's absent.
_FRAMEWORK_PY = Path("/Library/Frameworks/Python.framework/Versions/3.14/bin/python3")
_VENV_PY      = ROOT / ".venv" / "bin" / "python"
PYTHON = _FRAMEWORK_PY if _FRAMEWORK_PY.exists() else _VENV_PY
LOGDIR = ROOT / (_cfg.get("logging", {}).get("dir", "daily_runs"))
LOGDIR.mkdir(exist_ok=True)

RUN_AT = datetime.now()
LOG_FILE     = LOGDIR / RUN_AT.strftime("%Y%m%d_%H%M%S_daily_run.log")
LOG_FILE_JSON = LOGDIR / RUN_AT.strftime("%Y%m%d_%H%M%S_daily_run.jsonl")

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

STEPS = [
    # (label, script, critical, timeout_sec)
    # critical=True  → failure shown as ERROR but pipeline continues
    # critical=False → skipped silently if script missing
    # timeout_sec    → None means use TIMEOUT_DEFAULT
    ("Step 0  — Price & signals",  "step_daily_price_signals.py",                 True,  900),
    ("Step 99 — News (S&P500全覆盖)", "canyon_final_v9_step99_news_aggregator.py", False, 300,
     ["--source", "alpha", "--top", "500", "--workers", "12"]),
    ("Step 129— News Impact Targeting", "canyon_final_v9_step129_news_impact_targeting.py", False, 600),
    ("Step 79 — FinBERT sentiment","canyon_final_v9_step79_finbert_sentiment.py", False, 1500),  # ML 模型推理慢, 600s 不够 → 给 25min 余量 (2026-08-04)
    ("Step 80 — SEC MD&A NLP",    "canyon_final_v9_step80_sec_mda_nlp.py",       False, 3600),
    ("Step 81 — Earnings 8-K NLP","canyon_final_v9_step81_earnings_nlp.py",      False, 2400),
    ("Step 81b— PEAD/SUE signal", "canyon_final_v9_step81_earnings_surprise.py",  False, 600),
    ("Step 82 — Options IV/flow", "canyon_final_v9_step82_options_signals.py",    False, 2400),
    ("Step 85 — IC Optimizer",    "canyon_final_v9_step85_ic_optimizer.py",       False, 600),
    ("Step 86 — Cross-Asset mom",  "canyon_final_v9_step86_cross_asset.py",       False, 300),
    ("Step 87 — 13F Crowding",    "canyon_final_v9_step87_13f_crowding.py",       False, 1800),
    ("Step 89 — XBRL Fundamntls","canyon_final_v9_step89_xbrl_fundamentals.py",  False, 3600),
    ("Step 88 — Factor Risk Mdl", "canyon_final_v9_step88_factor_risk_model.py",  False, 600),
    ("Step 95 — Macro signals",    "canyon_final_v9_step95_macro_signals.py",     False, 300),
    ("Step 48 — Macro breadth",   "canyon_final_v9_step48_l2_macro_regime.py",   False, 300),
    ("Step 49 — Sector rotation", "canyon_final_v9_step49_l3_sector_rotation.py",False, 300),
    ("Step 370 — HMM Regime",     "canyon_final_v9_step370_hmm_regime.py",       False, 900),
    ("Step 371 — Macro Outlook",  "step_macro_regime_outlook.py",                False, 120),
    ("Step 372 — DCF Valuation", "step_dcf_valuation.py",                        False, 1500),
    ("Step 373 — Short Scanner", "step_short_scanner.py",                        False, 900),
    ("Step 374 — Push Alerts",   "step_push_alerts.py",                          False,  60),
    ("Step 375 — Econ Calendar", "step_economic_calendar.py",                    False,  30),
    ("Step 376 — Earnings AI",   "step_earnings_ai.py",                          False, 600),
    ("Step 377 — 8-K NLP",      "step_earnings_8k_nlp.py",                      False, 1200),
    ("Step 378 — FRED Data",    "step_fred_data.py",                             False, 120),
    ("Step 378a— Regime Detector", "canyon_final_v9_step76_regime_detector.py",  False, 180),
    ("Step 378b— Macro Event Engine", "canyon_final_v9_step101_macro_event_engine.py", False, 240),
    ("Step 378c— Macro Regime Outlook", "step_macro_regime_outlook.py",          False, 120),
    ("Step 379 — Social Senti", "step_social_sentiment.py",                      False, 180),
    ("Step 379b— AlphaVantage News", "step_alphavantage.py",                     False, 60),
    ("Step 102— Earnings cal (S&P500)", "canyon_final_v9_step102_earnings_calendar.py", False, 300,
     ["--top", "500"]),
    ("Step 250 — Alt data IC",    "canyon_final_v9_step250_alt_data.py",          False, 600),
    ("Inst   — Layer upgrades",    "step_institutional_upgrades.py",              False, 1800),
    ("Step 12 — Core pipeline",   "canyon_final_v9_step12_daily_runner.py",       True,  1800),
    ("Step 500 — Live signals",   "canyon_final_v9_step500_daily_pipeline.py",    True,  1200),
    ("Step 240 — TC model",       "canyon_final_v9_step240_tc_model.py",          False, 300),
    ("Step 90  — MVO Optimizer",  "canyon_final_v9_step90_portfolio_optimizer.py", False, 120),
    ("Step 379b— Rolling IC",     "step_rolling_ic.py",                           False, 120),
    ("Step 380 — ML Alpha",       "step_ml_alpha.py",                             False, 600),
    ("Step 381 — Backtest 3Bks",  "step_backtest_rigorous.py",                    False, 300),
    ("Step 382 — Alpaca Execute", "step_alpaca_execution.py",                     False, 120),
    ("Step 383 — Alpaca P&L",    "step_alpaca_pnl.py",                            False,  60),
    ("Step 387 — Insider Scan",  "canyon_insider_scanner.py",                     False, 1800),
    ("Step 387b— Insider Sizing","canyon_insider_sizing.py",                      False,  90),
    ("Step 387c— Insider Short", "canyon_insider_short.py",                       False, 1500),
    ("Step 388 — Insider Paper", "canyon_insider_paper.py",                       False,  90),
    ("Step 388b— Insider L/S BT", "canyon_insider_ls_backtest.py",                False, 180),
    ("Step 388c— PM Desk decision","canyon_pm_desk.py",                            False,  60),
    ("Step 388d— Live calibration","canyon_insider_calibration.py",               False,  60),
    ("Step 389 — IC Desk (synth)", "canyon_ic_desk.py",                            False,  60),
    ("Step 384 — Factor Exp",    "step_factor_exposure.py",                       False,  60),
    ("Step 385 — PEAD Tracker", "step_pead_tracker.py",                           False, 120),
    ("Step 386 — Famous 13F",   "step_famous_holdings.py",                        False, 300),
    ("Step 387 — Congress",     "step_congressional_trading.py",                  False, 120),
    ("Step 69  — Paper sim NAV",  "canyon_final_v9_step69_paper_sim.py",          False, 300),
    ("Step 96 — P&L attribution", "canyon_final_v9_step96_pnl_attribution.py",   False, 1200),
    ("Step 260 — Risk infra",     "canyon_final_v9_step260_risk_infra.py",        False, 600),
    ("Step 270 — IC Decay report","canyon_final_v9_step270_reporting.py",         False, 600),
    ("Step 290 — Stress test",    "canyon_final_v9_step290_stress_test.py",       False, 300),
    ("Step 97 — Correlation mon","canyon_final_v9_step97_correlation_monitor.py", False, 300),
    ("Step 104— Signal divrsty", "canyon_final_v9_step104_signal_correlation.py", False, 120),
    # ── Re-added 2026-07-22: orphaned analytics steps (were dropped ~41d ago,
    #    dashboard still reads their outputs). Verified to run + refresh output. ──
    ("Step 142— Sector cycle",   "canyon_final_v9_step142_sector_cycle_linkage.py",       False, 300),
    ("Step 144— Daily workflow", "canyon_final_v9_step144_dynamic_daily_workflow.py",     False, 300),
    ("Step 115— NAV attribution","canyon_final_v9_step115_live_nav_attribution_slippage.py", False, 300),
    ("Step 410— Factor timing",  "canyon_final_v9_step410_factor_timing.py",              False, 600),
    ("Step 100— Walk-fwd OOS",   "canyon_final_v9_step100_walk_forward_oos.py",           False, 600),
    ("Step 221— Factor attrib",  "canyon_final_v9_step221_factor_attribution.py",         False, 900),
    ("Step 380b— PnL attribution","canyon_final_v9_step380_pnl_attribution.py",           False, 600),
    ("Step 114— Kelly sizing",   "canyon_final_v9_step114_drawdown_vol_kelly.py",         False, 300),
    ("Step 121— Data truth ledgr","canyon_final_v9_step121_data_truth_ledger.py",         False, 300),
    ("Step 120— Inst upgrade mstr","canyon_final_v9_step120_institutional_upgrade_master.py", False, 600),
    ("Step 172— Inst depth engine","canyon_final_v9_step172_institutional_depth_upgrade_engine.py", False, 600),
    ("Step TCA— Execution depth", "canyon_execution_tca_depth.py",                        False, 300),
    # ── Re-added 2026-08-04: 4 producers whose dashboard panels had gone ~14d
    #    stale (they were absent from the pipeline). All non-critical + bootstrap
    #    socket-timeout protected, so a slow/failed run can't harm the pipeline;
    #    generous timeouts because they crunch history CSVs. tc_verify verified
    #    ~1s; the other 3 could not be timed cleanly in a loaded sandbox but run
    #    fine on the unstressed 6am cron. ──
    ("Step ICA— IC audit report", "step_ic_audit.py",                                     False, 600),
    ("Step TCV— TC verification", "step_tc_verify.py",                                     False, 180),
    ("Step FCM— Factor corr/VIF", "step_factor_correlation.py",                            False, 600),
    ("Step STR— Stress test",     "step_stress_test.py",                                   False, 600),
    # Re-added producers that were runnable but missing from the daily pipeline.
    ("Step 5yr— 5-year backtest", "canyon_v9_clean_5yr_backtest.py",                      False, 600),
    ("Step HM — Heatmap data",   "step_heatmap_data.py",                                  False, 120),
    ("Step 490— Short squeeze",  "canyon_final_v9_step490_short_squeeze.py",              False, 300),
    ("Step OF — Options flow",   "step_options_flow.py",                                  False, 120),
    ("Step EF — ETF flow",       "step_etf_flow_rt.py",                                   False, 120),
    # Revive 6 orphaned dashboard panels (producers were lost from the repo).
    # Rebuilds paper NAV, live IC, risk gate, desk monitor, event dossier,
    # action readiness from live data. Must run after weights/picks/alpha history.
    ("Revive — Orphaned panels", "step_revive_panels.py",                                 False, 300),
    # 事件驱动主动投资系统 (手册) — 底库永远是标普500
    ("EDGAR — SEC 8-K/Form4事件流", "canyon_edgar_events.py",                              False, 300),
    ("COT   — CFTC商品持仓信号",   "canyon_cftc_cot.py",                                   False, 180),
    ("Macro — 第1层情报评分卡",  "canyon_macro_intel.py",                                 False, 120),
    ("Rotate— 行业板块轮动信号",  "canyon_sector_rotation.py",                             False, 120),
    ("ETF   — 板块龙头ETF指标",   "canyon_sector_etf.py",                                  False, 120),
    ("Detect— 第3层事件自动侦测", "canyon_event_detect.py",                                False, 120),
    ("Life  — 第2层生命周期风格", "canyon_lifecycle.py",                                   False, 120),
    ("Pool  — 建池(标普500底库)", "canyon_build_pool.py",                                  False, 120),
    ("Event — 利润发动机引擎",   "canyon_event_system.py",                                False, 120),
    ("Layer4— 功能池自动分层",   "canyon_pools.py",                                       False, 120),
    ("Size  — 第6层仓位构建风控", "canyon_position_sizing.py",                             False, 120),
    ("TCA   — 执行成本建模",     "canyon_execution_costs.py",                             False, 120),
    ("Intra — 日内感知层",       "canyon_intraday.py",                                    False, 180),
    ("PosMgr— 持仓管理/退出",     "canyon_position_manager.py",                            False, 120),
    ("Review— 周月季复盘节奏",   "canyon_review.py",                                      False, 120),
    ("Valid — 事件打分IC验证",   "canyon_event_validate.py",                              False, 180),
    ("EventBT—8-K事件研究验证",   "canyon_edgar_backtest.py",                              False, 300),
    # Deep price history (incremental) + PIT membership + credible bias-controlled backtest
    ("Deep  — Price history",    "step_extend_price_history.py",                          False, 900),
    ("PIT   — S&P membership",   "step_sp500_pit_membership.py",                          False, 120),
    ("Rigor — Honest backtest",  "step_rigorous_backtest.py",                             False, 600),
    # First real-edge research: EDGAR point-in-time EPS + PEAD validation
    ("EDGAR — PIT quarterly EPS","step_edgar_eps_pit.py",                                 False, 600),
    ("PEAD  — Earnings drift",   "step_pead_strategy.py",                                 False, 300),
    ("Step 98 — Daily alerts",    "canyon_final_v9_step98_daily_alerts.py",       False, 600),
    ("Data   — Parquet sync",     "canyon_data_layer.py",                         False, 120),
    ("HTML — Research page",      "update_research_html.py",                      True,  600),
    ("Email — Daily summary",     "email_summary.py",                             False, 120),
]

TIMEOUT_DEFAULT = 600   # fallback if step doesn't specify


def _log(msg: str, level: str = "INFO", step: str = "", elapsed: float = 0.0):
    ts  = datetime.now().strftime("%H:%M:%S")
    iso = datetime.now().isoformat(timespec="seconds")
    line = f"[{ts}] {msg}"
    print(line)
    # Plain text log
    with open(LOG_FILE, "a") as f:
        clean = msg.replace(GREEN,"").replace(RED,"").replace(YELLOW,"").replace(CYAN,"").replace(BOLD,"").replace(RESET,"")
        f.write(f"[{ts}] {clean}\n")
    # JSON structured log (for monitoring / alerting)
    with open(LOG_FILE_JSON, "a") as fj:
        record = {"ts": iso, "level": level, "step": step, "msg": msg.replace(GREEN,"").replace(RED,"").replace(YELLOW,"").replace(CYAN,"").replace(BOLD,"").replace(RESET,"").strip()}
        if elapsed:
            record["elapsed_s"] = round(elapsed, 1)
        fj.write(json.dumps(record) + "\n")


def run_step(label: str, script: str, critical: bool, timeout_sec: int = None, args: list = None,
             retries: int = 1) -> bool:
    path = ROOT / script
    if not path.exists():
        if critical:
            _log(f"{RED}MISS  {label} — script not found: {script}{RESET}")
            return False  # critical script missing = failure
        _log(f"{YELLOW}SKIP  {label} — optional script not found{RESET}")
        return True

    _log(f"{CYAN}START {BOLD}{label}{RESET}")
    timeout = timeout_sec if timeout_sec is not None else TIMEOUT_DEFAULT
    import os as _os
    env = _os.environ.copy()
    env['RUN_DAILY_ACTIVE'] = '1'          # signals sub-scripts not to re-invoke the pipeline
    env.setdefault('HF_HUB_OFFLINE', '1')  # transformers/HF 不联网, 防 import 超时(FinBERT)
    env.setdefault('TRANSFORMERS_OFFLINE', '1')
    last_err, err_lines = "", []
    # 失败自动重试 (retries 次): 大多数失败是网络限流/超时, 重试即恢复
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            result = subprocess.run(
                # via _canyon_run.py bootstrap: installs a default socket timeout
                # so a hung network read fails fast instead of eating the whole
                # per-step timeout (root cause of the "6 偶发失败" steps).
                [str(PYTHON), str(ROOT / "_canyon_run.py"), str(path)] + (args or []),
                cwd=str(ROOT), capture_output=True, text=True, timeout=timeout, env=env,
            )
            elapsed = time.time() - t0
            if result.returncode == 0:
                tag = f"{GREEN}  OK  {label}  ({elapsed:.1f}s{'·重试成功' if attempt else ''}){RESET}"
                _log(tag)
                for ln in [l.strip() for l in result.stdout.strip().splitlines() if l.strip()][-3:]:
                    _log(f"       {ln}")
                return True
            last_err = f"rc={result.returncode}"
            err_lines = (result.stderr or result.stdout or "").strip().splitlines()[-3:]
        except subprocess.TimeoutExpired:
            last_err = f"超时 >{timeout}s"; elapsed = time.time() - t0; err_lines = []
        except Exception as e:
            last_err = str(e); elapsed = time.time() - t0; err_lines = []
        if attempt < retries:
            _log(f"{YELLOW}  RETRY {label} (第{attempt+1}次失败: {last_err}, 5秒后重试){RESET}")
            time.sleep(5)
        else:
            _log(f"{RED}  ERR {label}  ({elapsed:.1f}s)  {last_err}{RESET}")
            for ln in err_lines:
                _log(f"       {ln.strip()}")
            return False
    return False


def preflight_check() -> list[str]:
    """Syntax-check every pipeline script before running. Returns list of failures."""
    import ast
    failures = []
    for label, script, *_ in STEPS:
        p = ROOT / script
        if not p.exists():
            continue
        try:
            ast.parse(p.read_text())
        except SyntaxError as e:
            failures.append(f"{script}:{e.lineno} — {e.msg}")
    return failures


def data_freshness_check() -> list[str]:
    """Check that key output files were updated today. Returns list of stale files."""
    from datetime import date as _date
    today = _date.today()
    # Files that MUST be fresh after a successful run
    required_fresh = [
        ("alpha_scores.csv",          "Step 0 — Price signals"),
        ("hmm_regime_daily.csv",      "Step 370 — HMM Regime"),
        ("macro_regime_outlook.json", "Step 371 — Macro Outlook"),
        ("macro_signals.json",        "Step 95 — Macro signals"),
        ("sector_rotation_scores.csv","Step 49 — Sector rotation"),
        ("rolling_ic_monitor.csv",    "Step 260 — Risk infra"),
        ("attribution_monthly.csv",   "Step 96 — Attribution"),
        ("correlation_monitor.csv",   "Step 97 — Correlation"),
        ("signal_correlation.csv",    "Step 104 — Signal diversity"),
        ("daily_alerts.json",         "Step 98 — Daily alerts"),
        ("portfolio_risk_decomp.csv", "Step 88 — Factor Risk"),
    ]
    stale = []
    for fname, step_label in required_fresh:
        p = ROOT / fname
        if not p.exists():
            stale.append(f"MISSING  {fname}  [{step_label}]")
        else:
            mtime_date = datetime.fromtimestamp(p.stat().st_mtime).date()
            if mtime_date < today:
                days = (today - mtime_date).days
                stale.append(f"{days}d old  {fname}  [{step_label}]")
    return stale


def main():
    _log(f"{BOLD}═══════════════════════════════════════════════════════════{RESET}")
    _log(f"{BOLD}  Canyon v9 — Daily Pipeline — {RUN_AT.strftime('%Y-%m-%d %H:%M')}{RESET}")
    _log(f"{BOLD}═══════════════════════════════════════════════════════════{RESET}")
    _log(f"  Python:   {PYTHON}")
    _log(f"  Log file: {LOG_FILE}")
    _log(f"  Steps:    {len(STEPS)}")
    _log("")

    # ── Pre-flight: syntax check all scripts before running anything ──────────
    _log(f"{CYAN}PRE   — Syntax check all {len(STEPS)} pipeline scripts{RESET}")
    syntax_fails = preflight_check()
    if syntax_fails:
        for f in syntax_fails:
            _log(f"{RED}  SYNTAX ERROR  {f}{RESET}", level="ERROR")
        _log(f"{RED}  {len(syntax_fails)} script(s) have syntax errors — fix before running{RESET}")
    else:
        _log(f"{GREEN}  ✓ All scripts syntax-clean{RESET}")
    _log("")

    results: list[tuple[str, bool]] = []
    total_start = time.time()

    for label, script, critical, *rest in STEPS:
        timeout_sec = rest[0] if rest else None
        step_args = rest[1] if len(rest) > 1 else None
        ok = run_step(label, script, critical, timeout_sec, step_args)
        results.append((label, ok))
        _log("")

    total_elapsed = time.time() - total_start

    # Post-run: data freshness check
    _log(f"{CYAN}QC    — Post-run data freshness check{RESET}")
    stale_files = data_freshness_check()
    if stale_files:
        for s in stale_files:
            _log(f"  {YELLOW}⚠  STALE  {s}{RESET}", level="WARN")
        _log(f"  {YELLOW}{len(stale_files)} output(s) not updated — the step that produces them likely failed{RESET}")
    else:
        _log(f"  {GREEN}✓ All key outputs updated today{RESET}")

    # W16: Data quality check
    try:
        import sys as _sys; _sys.path.insert(0, str(ROOT))
        from monitoring.data_quality import run_quality_checks, print_report
        qc_df = run_quality_checks()
        warns = qc_df[qc_df["status"].isin(["WARN", "ERROR", "MISSING"])]
        if not warns.empty:
            for _, row in warns.iterrows():
                _log(f"  {YELLOW}⚠{RESET}  QC {row['check']}: {row.get('detail', '')}", level="WARN")
        else:
            _log(f"  {GREEN}✓  External data quality checks passed{RESET}", level="INFO")
    except Exception as e:
        _log(f"  {YELLOW}QC check skipped: {e}{RESET}", level="WARN")

    # Summary
    _log(f"{BOLD}═══════════════════════════════════════════════════════════{RESET}")
    _log(f"{BOLD}  SUMMARY — {RUN_AT.strftime('%Y-%m-%d')} — {total_elapsed:.0f}s total{RESET}")
    _log(f"{BOLD}═══════════════════════════════════════════════════════════{RESET}")
    passed = sum(1 for _, ok in results if ok)
    failed = len(results) - passed
    for label, ok in results:
        icon = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        _log(f"  {icon}  {label}")
    _log("")
    # ── Failure alert: write a machine-readable alert the dashboard surfaces ──
    _alert_path = ROOT / "pipeline_alert.json"
    failed_labels = [label for label, ok in results if not ok]
    try:
        if failed == 0:
            alert = {"status": "ok", "ts": datetime.now().isoformat(timespec="seconds"),
                     "passed": passed, "failed": 0, "failed_steps": []}
        else:
            alert = {"status": "alert", "ts": datetime.now().isoformat(timespec="seconds"),
                     "passed": passed, "failed": failed, "failed_steps": failed_labels}
        json.dump(alert, open(_alert_path, "w"), indent=2)
    except Exception:
        pass

    if failed == 0:
        _log(f"{GREEN}{BOLD}  All {passed} steps completed.{RESET}")
    else:
        _log(f"{RED}{BOLD}  ⚠ {passed} passed · {failed} FAILED: {', '.join(failed_labels[:6])}{RESET}")
        _log(f"{YELLOW}  Alert written to pipeline_alert.json — check log: {LOG_FILE}{RESET}")
        # optional desktop push if step_push_alerts supports it
        try:
            subprocess.run([str(PYTHON), str(ROOT / "step_push_alerts.py"), "--pipeline-failure"],
                           cwd=str(ROOT), timeout=30, capture_output=True)
        except Exception:
            pass
    _log("")
    _log(f"  Research page:  file://{ROOT / 'canyon_v24_research.html'}")
    _log(f"  Dynamic server: http://localhost:8513  (if serve_research.py is running)")
    _log("")


if __name__ == "__main__":
    if not PYTHON.exists():
        print(f"{RED}ERROR: .venv not found at {PYTHON}{RESET}")
        print("  Run:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt")
        sys.exit(1)
    main()
