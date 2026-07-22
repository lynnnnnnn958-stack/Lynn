#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CANYON v9 Step 55 — 10-Layer Dashboard v2

Reads Step 54 v2 matrix and scorecard.
Standalone dashboard. Does not patch old files.
No broker. No live order.
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
import os
import re
import subprocess
import sys
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    import plotly.express as px
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

ROOT = Path.cwd()

FILES = {
    "layer_audit": ROOT / "canyon_layer_status_audit.csv",
    "architecture": ROOT / "canyon_10_layer_architecture.md",
    "build_plan": ROOT / "canyon_layer_build_plan.md",

    "master_v2": ROOT / "master_10_layer_decision_matrix_v2.csv",
    "master_report_v2": ROOT / "master_10_layer_decision_report_v2.md",
    "scorecard": ROOT / "master_10_layer_scorecard.csv",

    "macro_report": ROOT / "macro_regime_report.md",
    "sector_report": ROOT / "sector_rotation_report.md",
    "fund_report": ROOT / "fundamental_report.md",
    "event_report": ROOT / "event_news_sec_insider_report.md",
    "tech_report": ROOT / "technical_microstructure_report.md",
    "action_cards": ROOT / "action_cards.csv",
    "watch_triggers": ROOT / "watch_triggers.csv",
    "pre_trade": ROOT / "pre_trade_checklist.csv",
    "pre_trade_order": ROOT / "pre_trade_order_ticket.csv",
    "data_quality": ROOT / "data_quality_flags.csv",
    "market_snapshot": ROOT / "market_data_snapshot.csv",
    "universe": ROOT / "universe_master.csv",
    "paper_ledger": ROOT / "paper_portfolio_ledger.csv",
    "learning_summary": ROOT / "learning_attribution_summary.csv",
    "learning_suggestions": ROOT / "learning_weight_suggestions.csv",
    "learning_report": ROOT / "learning_attribution_report.md",
    "exposure_warnings": ROOT / "exposure_warnings.csv",
    "scenario_stress": ROOT / "scenario_stress_results.csv",
    "position_sizing": ROOT / "position_sizing_recommendations.csv",
    "stress_report": ROOT / "stress_position_sizing_report.md",
    "exposure": ROOT / "exposure_dashboard.csv",
    "exposure_report": ROOT / "exposure_dashboard.md",
    "macro_signals": ROOT / "macro_regime_signals.csv",
    "index_breadth": ROOT / "index_breadth_dashboard.csv",
    "volatility_regime": ROOT / "volatility_regime.csv",
    "sector_scores": ROOT / "sector_rotation_scores.csv",
    "theme_heatmap": ROOT / "theme_heatmap.csv",
    "fundamentals": ROOT / "fundamental_quality_valuation.csv",
    "valuation_flags": ROOT / "valuation_risk_flags.csv",
    "events": ROOT / "evidence_cards.csv",
    "news_risk": ROOT / "news_event_risk.csv",
    "earnings_check": ROOT / "earnings_calendar_check.csv",
    "insider_signals": ROOT / "insider_form4_signals.csv",
    "technicals": ROOT / "technical_signal_matrix.csv",
    "liquidity_proxy": ROOT / "intraday_liquidity_proxy.csv",
    "tactical_candidates": ROOT / "tactical_candidates.csv",
    "breakout_watchlist": ROOT / "breakout_reversal_watchlist.csv",
    "gamma_candidates": ROOT / "gamma_squeeze_candidates.csv",
    "kill_zone": ROOT / "option_kill_zone_risk.csv",
    "options_decision": ROOT / "options_decision_matrix.csv",
    "options_report": ROOT / "options_gamma_report.md",
    "kill_zone_report": ROOT / "option_kill_zone_report.md",
    "v8_inventory": ROOT / "v8_research_module_inventory.csv",
    "v8_bsm": ROOT / "v8_bsm_greeks_overlay.csv",
    "v8_synthetic_options": ROOT / "v8_synthetic_options_overlay.csv",
    "v8_report": ROOT / "v8_research_bridge_report.md",
    "v8_l9_gate": ROOT / "v8_l9_execution_gate.csv",
    "v8_l9_report": ROOT / "v8_l9_execution_gate_report.md",
    "v8_adv_risk": ROOT / "v8_advanced_risk_summary.csv",
    "v8_pca": ROOT / "v8_pca_factor_exposure.csv",
    "v8_tail": ROOT / "v8_tail_dependence_matrix.csv",
    "v8_adv_risk_report": ROOT / "v8_advanced_risk_report.md",
    "vault_index": ROOT / "canyon_output_vault_index.csv",
    "vault_alerts": ROOT / "canyon_output_shrinkage_alerts.csv",
    "vault_report": ROOT / "canyon_output_vault_report.md",
    "data_source_health": ROOT / "data_source_health.csv",
    "data_source_health_report": ROOT / "data_source_health_report.md",
    # Step 62 — Backtest Engine
    "backtest_signal_ic":   ROOT / "backtest_signal_ic.csv",
    "backtest_monthly_perf": ROOT / "backtest_monthly_perf.csv",
    "backtest_summary":     ROOT / "backtest_summary.csv",
    "backtest_holdings":    ROOT / "backtest_holdings.csv",
    "backtest_report":      ROOT / "backtest_engine_report.md",
    # Step 63 — Portfolio Optimizer
    "portfolio_weights":    ROOT / "portfolio_optimized_weights.csv",
    "portfolio_covariance": ROOT / "portfolio_covariance.csv",
    "portfolio_ef_points":  ROOT / "portfolio_ef_points.csv",
    "portfolio_opt_report": ROOT / "portfolio_optimizer_report.md",
    # Step 64 — Data Layer
    "data_layer_status":    ROOT / "data_layer_status.csv",
    "data_layer_report":    ROOT / "data_layer_report.md",
    "data_layer_prices":    ROOT / "data_layer_prices.csv",
    # Step 65 — Earnings NLP
    "earnings_nlp_scores":  ROOT / "earnings_nlp_scores.csv",
    "earnings_calendar":    ROOT / "earnings_calendar.csv",
    "earnings_nlp_report":  ROOT / "earnings_nlp_report.md",
    # Step 66 — ML Signal Generator
    "ml_signal_scores":     ROOT / "ml_signal_scores.csv",
    "ml_ic_comparison":     ROOT / "ml_ic_comparison.csv",
    "ml_backtest_perf":     ROOT / "ml_backtest_perf.csv",
    "ml_feature_importance": ROOT / "ml_feature_importance.csv",
    "ml_summary":           ROOT / "ml_summary.csv",
    "ml_report":            ROOT / "ml_signal_report.md",
    # Step 67 — SHAP Explainer
    "shap_values_rf":       ROOT / "shap_values_rf.csv",
    "shap_values_ridge":    ROOT / "shap_values_ridge.csv",
    "shap_summary":         ROOT / "shap_summary.csv",
    "shap_per_ticker":      ROOT / "shap_per_ticker.csv",
    "shap_report":          ROOT / "shap_report.md",
    # Step 68 — Fundamental Features
    "fundamental_features": ROOT / "fundamental_features.csv",
    "enhanced_ml_scores":   ROOT / "enhanced_ml_scores.csv",
    "fundamental_ic":       ROOT / "fundamental_ic_comparison.csv",
    "fundamental_report":   ROOT / "fundamental_report.md",
    # Step 69 — Paper Trading Simulator
    "paper_sim_positions":  ROOT / "paper_sim_positions.csv",
    "paper_sim_trades":     ROOT / "paper_sim_trades.csv",
    "paper_sim_summary":    ROOT / "paper_sim_summary.csv",
    "paper_sim_report":     ROOT / "paper_sim_report.md",
    "paper_sim_nav":        ROOT / "paper_sim_nav.csv",
    # Step 70 — Daily Batch Runner
    "run_daily_log":        ROOT / "run_daily_all_log.csv",
    "run_daily_report":     ROOT / "run_daily_all_report.md",
    # Step 71 — Alert System
    "alerts":               ROOT / "alerts.csv",
    "alerts_report":        ROOT / "alerts_report.md",
    # Step 72 — Weekly Report
    "weekly_report":        ROOT / "weekly_report_latest.html",
    # Step 73 — Factor Attribution
    "factor_attribution":   ROOT / "factor_attribution.csv",
    "factor_marginal_risk": ROOT / "factor_marginal_risk.csv",
    "factor_report":        ROOT / "factor_report.md",
    # Step 74 — Options Chain (L7)
    "options_chain_summary": ROOT / "options_chain_summary.csv",
    "options_chain_detail":  ROOT / "options_chain_detail.csv",
    "options_chain_report":  ROOT / "options_chain_report.md",
    # Step 78 — Deep Fundamentals
    "fundamental_deep_scores":  ROOT / "fundamental_deep_scores.csv",
    "fundamental_quality_rank": ROOT / "fundamental_quality_rank.csv",
    "fundamental_deep_report":  ROOT / "fundamental_deep_report.md",
    # Step 79 — FinBERT Sentiment
    "finbert_sentiment":        ROOT / "finbert_sentiment.csv",
    "finbert_report":           ROOT / "finbert_sentiment_report.md",
    # Step 75 — Universe Expansion
    "ic_by_regime_full":     ROOT / "ic_by_regime_full.csv",
    "ic_by_regime_raw":      ROOT / "ic_by_regime_raw.csv",
    "extended_backtest_perf": ROOT / "extended_backtest_perf.csv",
    "extended_backtest_summary": ROOT / "extended_backtest_summary.csv",
    "universe_expansion_report": ROOT / "universe_expansion_report.md",
    # Step 76 — Regime Detector
    "regime_history":        ROOT / "regime_history.csv",
    "regime_transitions":    ROOT / "regime_transitions.csv",
    "regime_current":        ROOT / "regime_current.json",
    "regime_report":         ROOT / "regime_report.md",
    # Step 77 — Regime-Conditional ML
    "regime_ml_scores":      ROOT / "regime_ml_scores.csv",
    "regime_backtest_ic":    ROOT / "regime_backtest_ic.csv",
    "regime_ic_summary":     ROOT / "regime_ic_summary.csv",
    "regime_ml_report":      ROOT / "regime_ml_report.md",
}

CHINESE_TEXT_REPLACEMENTS = [    ('\u53ea\u7814\u7a76\uff0c\u4e0d\u505a paper\u3002', 'Research only; do not start a paper test.'),    ('\u53ea\u7814\u7a76\uff0c\u4e0d\u505a paper', 'Research only; do not start a paper test'),    ('\u53ea\u7814\u7a76\uff0c\u4e0d\u5efa paper', 'Research only; do not start a paper test'),    ('\u7814\u7a76', 'Research'),    ('\u5efa\u4ed3 / paper', 'Open a position / paper test'),    ('\u5148\u8865\u57fa\u672c\u9762/\u4e8b\u4ef6\u8bc1\u636e\u3002', 'Add company/basic and event evidence first.'),    ('Kill Zone \u672a\u663e\u793a\u9ad8\u98ce\u9669\uff0c\u4f46\u4ecd\u9700\u4eba\u5de5\u68c0\u67e5 spread \u548c\u4e8b\u4ef6\u3002', 'The danger zone is not high, but spread and event checks still need human review.'),    ('\u4eca\u665a\u8df3\u8fc7\uff0c\u4e0d\u6d6a\u8d39\u6ce8\u610f\u529b\u3002', 'Skip tonight; do not spend attention here.'),    ('\u8df3\u8fc7\uff1b\u4eca\u665a\u4e0d\u505a', 'Skip; do nothing tonight'),    ('\u65e0', 'None'),    ('\u5f3a\u884c\u627e\u4ea4\u6613\u7406\u7531', 'Force a trade reason'),    ('\u6ca1\u6709\u89e6\u53d1\u6761\u4ef6\uff1b\u8df3\u8fc7\u3002', 'No trigger condition; skip.'),    ('\u98ce\u9669\u706f RED \u4e14 gamma \u4f18\u52bf\u4e0d\u8db3\uff0c\u5148\u7814\u7a76\u4e0d\u4ea4\u6613\u3002', 'Risk light is red and gamma edge is not strong enough; research only, no trade.'),    ('Pre-trade \u5df2\u963b\u65ad\u3002', 'Before-action check already blocks this.'),    ('\u5df2\u7ecf paper closed\uff0c\u4e0d\u8981\u91cd\u590d\u9020\u6837\u672c\u3002', 'This paper sample is already closed; do not create a duplicate sample.'),    ('\u751f\u6210\u65f6\u95f4', 'Generated time'),    ('\u6570\u636e\u771f\u5b9e\u6027', 'Data Trust'),    ('\u6570\u636e\u72b6\u6001', 'Data Status'),    ('\u5408\u6210\u6570\u636e\u89c4\u5219', 'Synthetic Data Rule'),    ('\u4e0b\u8f7d\u5931\u8d25\u65f6\u7cfb\u7edf\u505c\u6b62\uff0c\u4e0d\u4f7f\u7528\u5408\u6210\u884c\u60c5\u3002', 'If download fails, stop the system; do not use synthetic market data.'),    ('\u7ed3\u8bba', 'Conclusion'),    ('\u5f53\u524d\u6d41\u7a0b\u53ea\u5e94\u4f7f\u7528\u771f\u5b9e\u4e0b\u8f7d\u6570\u636e\uff1b\u4e0b\u8f7d\u5931\u8d25\u65f6\u5e94\u505c\u6b62\uff0c\u4e0d\u5e94\u56de\u9000\u5230 synthetic data \u505a\u51b3\u7b56\u3002', 'Use downloaded real data only; if download fails, stop instead of falling back to synthetic data for decisions.'),    ('\u5f53\u524d\u5e02\u573a\u72b6\u6001\u4e0e\u4e09\u8d26\u6237\u8d44\u91d1\u8ba1\u5212', 'Current Market State And Three Account Plan'),    ('\u5f53\u524d\u5e02\u573a\u72b6\u6001', 'Current Market State'),    ('\u666e\u901a\u725b\u5e02\uff08\u8fdb\u653b\u4e3a\u4e3b\uff09', 'Normal bull market, offense first'),    ('\u53ea\u505a\u5019\u9009\uff0c\u4e0d\u81ea\u52a8\u4e0b\u5355\uff1b\u5fc5\u987b\u5148\u4eba\u5de5\u68c0\u67e5\u65b0\u95fb/\u8d22\u62a5/\u6d41\u52a8\u6027\u3002', 'Candidate only, no automatic order; manually check news, earnings, and liquidity first.'),    ('\u662f\u538b\u8231\u77f3\uff0c\u4e0d\u56e0\u4e3a\u77ed\u7ebf\u5174\u594b\u968f\u610f\u632a\u8d70\u3002', 'is the stabilizer sleeve; do not move it because of short-term excitement.'),    ('\u6bcf\u5468\u590d\u6838\uff0c\u4e0d\u505a\u65e5\u5185\u9891\u7e41\u5207\u6362\u3002', 'Review weekly; do not switch frequently intraday.'),    ('\u89e3\u91ca', 'Explanation'),    ('\u77ed\u7ebf\u5019\u9009\u8d26\u6237\uff0c\u4e0d\u662f\u81ea\u52a8\u4ea4\u6613\u8d26\u6237', 'short-term candidate account, not an automatic trading account'),    ('\u538b\u8231\u77f3', 'stabilizer sleeve'),    ('Evidence / SEC / Event \u8bc1\u636e\u6458\u8981', 'Evidence / SEC / Event Summary'),    ('\u4ef7\u683c\u4e0e\u98ce\u9669\u8bc1\u636e', 'Price And Risk Evidence'),    ('\u8d8b\u52bf\u8bc1\u636e', 'Trend Evidence'),    ('\u4ef7\u683c\u572820\u65e5/50\u65e5\u5747\u7ebf\u4e0a\u65b9\uff0c\u8d8b\u52bf\u7ed3\u6784\u504f\u591a\u3002', 'Price is above the 20-day and 50-day averages; trend leans bullish.'),    ('\u77ed\u7ebf\u5019\u9009\u5fc5\u987b\u4eba\u5de5\u68c0\u67e5\uff1a\u65b0\u95fb\u3001\u8d22\u62a5\u65e5\u671f\u3001\u76d8\u524d\u76d8\u540e\u5f02\u52a8\u3001bid-ask/liquidity\u3002', 'Short-term candidates need manual checks: news, earnings date, pre/post-market moves, bid-ask spread, and liquidity.'),    ('\u907f\u9669\u8bc1\u636e', 'Hedge Evidence'),    ('\u9ec4\u91d1\u53ef\u4f5c\u4e3a\u80a1\u7968\u98ce\u9669\u548c\u5b8f\u89c2\u4e0d\u786e\u5b9a\u6027\u7684\u7f13\u51b2\u3002', 'Gold can buffer equity risk and macro uncertainty.'),    ('\u8d8b\u52bf\u98ce\u9669', 'Trend Risk'),    ('\u4ef7\u683c\u572820\u65e5/50\u65e5\u5747\u7ebf\u4e0b\u65b9\uff0c\u53cd\u5f39\u53ef\u80fd\u53ea\u662f\u5f31\u53cd\u5f39\u3002', 'Price is below the 20-day and 50-day averages; any rebound may be weak.'),    ('\u76f8\u5bf9\u5f3a\u5ea6\u98ce\u9669', 'Relative Strength Risk'),    ('\u8fd120\u65e5\u8dd1\u8f93SPY\u3002', 'Underperformed SPY over the last 20 days.'),    ('\u91cf\u80fd\u8bc1\u636e', 'Volume Evidence'),    ('\u6210\u4ea4\u91cf\u660e\u663e\u9ad8\u4e8e\u8fd120\u65e5\u5747\u503c\uff0c\u8d44\u91d1\u5173\u6ce8\u5ea6\u4e0a\u5347\u3002', 'Volume is clearly above the 20-day average; attention is rising.'),    ('SEC / Event \u8bc1\u636e', 'SEC / Event Evidence'),    ('\u8d22\u62a5\u65e5\u671f', 'earnings date'),    ('\u65b0\u95fb', 'news'),    ('\u7ebf\u7d22', 'clues'),    ('\u98ce\u9669', 'Risk'),    ('90\u5929\u51858-K\u8f83\u591a\uff1a\u4e8b\u4ef6\u5bc6\u96c6\uff0c\u77ed\u7ebf\u6ce2\u52a8\u548c\u4fe1\u606f\u98ce\u9669\u8f83\u9ad8\u3002', 'Many 8-K filings in 90 days: event density is high, so short-term volatility and information risk are higher.'),    ('\u6ce8\u610f', 'Note'),    ('\u6570\u91cf\u4e0d\u662f\u4e70\u5165\u4fe1\u53f7\u3002\u5fc5\u987b\u6253\u5f00\u539f\u59cb Form 4 \u5224\u65ad\u662f\u4e70\u5165\u3001\u5356\u51fa\u3001\u671f\u6743\u5f52\u5c5e\u3001\u7a0e\u52a1\u5356\u51fa\uff0c\u8fd8\u662f 10b5-1 \u8ba1\u5212\u4ea4\u6613\u3002', 'count is not a buy signal. Open the original Form 4 to classify buys, sells, option vesting, tax sales, or 10b5-1 trades.'),    ('\u4e70\u5165', 'buy'),    ('\u5356\u51fa', 'sell'),    ('\u671f\u6743\u5f52\u5c5e', 'option vesting'),    ('\u7a0e\u52a1\u5356\u51fa', 'tax sale'),    ('\u8ba1\u5212\u4ea4\u6613', 'planned trade'),    ('Trade Journal \u72b6\u6001', 'Trade Journal Status'),    ('\u65e5\u5fd7\u5019\u9009\u603b\u6570', 'Total journal candidates'),    ('\u771f\u5b9e\u4ea4\u6613\u6570', 'real trades'),    ('Execution Gate \u72b6\u6001', 'Execution Gate Status'),    ('\u5ba1\u6838\u8868\u5019\u9009\u6570', 'Checklist candidate count'),    ('\u5206\u5e03', 'distribution'),    ('\u4eba\u5de5\u68c0\u67e5\u662f\u5426\u5168\u90e8\u5b8c\u6210', 'Are all manual checks complete'),    ('Order Ticket \u72b6\u6001', 'Order Ticket Status'),    ('\u8ba2\u5355\u8349\u7a3f\u6570', 'Order draft count'),    ('\u65e0\u8ba2\u5355\u8349\u7a3f\u3002\u8fd9\u901a\u5e38\u662f\u6b63\u5e38\u7684\uff1a\u4eba\u5de5\u68c0\u67e5\u672a\u5b8c\u6210\u65f6\u4e0d\u751f\u6210\u8ba2\u5355\u8349\u7a3f\u3002', 'No order drafts. This is usually correct: no order draft is generated before manual checks are complete.'),    ('\u4eca\u65e5\u64cd\u4f5c\u7eaa\u5f8b', 'Today Action Discipline'),    ('\u5b58\u5728 PENDING_MANUAL_CHECKS\uff1a\u65b0\u95fb\u3001\u8d22\u62a5\u65e5\u671f\u3001\u6d41\u52a8\u6027\u3001\u4ef7\u5dee\u3001thesis \u5c1a\u672a\u5168\u90e8\u786e\u8ba4\u3002', 'There are pending human checks: news, earnings date, liquidity, spread, and thesis are not all confirmed.'),    ('\u4e3a\u7a7a\u6216\u4e0d\u5b58\u5728\uff0c\u8bf4\u660e\u6ca1\u6709\u5f85\u53d1\u9001\u8ba2\u5355\u3002', 'is empty or missing, which means there are no orders waiting to send.'),    ('\u5c11\u4e8e 5 \u7b14\uff0cLearning Engine \u4e0d\u5e94\u81ea\u52a8\u8c03\u6743\u3002', 'fewer than 5 samples; the learning engine should not auto-adjust weights.'),    ('\u53ea\u4ee3\u8868\u53ef\u8fdb\u5165\u5ba1\u6838\uff0c\u4e0d\u4ee3\u8868\u5141\u8bb8\u771f\u5b9e\u4e0b\u5355\u3002', 'only means it can enter review; it does not allow live orders.'),    ('\u9ed8\u8ba4\u5148\u51cf\u534a\u6216\u8df3\u8fc7\u3002', 'default is half-size or skip.'),    ('\u5355\u7968\u9ed8\u8ba4\u4e0a\u9650', 'default single-name cap'),    ('\u9ad8\u76f8\u5173\u4e3b\u9898\u4e0d\u5141\u8bb8\u91cd\u590d\u6253\u6ee1\u3002', 'do not max out multiple highly correlated themes.'),    ('\u9ed8\u8ba4\u8ba2\u5355\u7c7b\u578b', 'default order type'),    ('\u4e0d\u8981\u7528 market order \u8ffd\u6da8\u3002', 'do not chase with market orders.'),    ('\u6240\u6709\u4eba\u5de5\u68c0\u67e5\u9879\u5fc5\u987b\u4e3a YES\uff0c\u4e14 order_intent \u5fc5\u987b\u4e3a PAPER \u6216 LIVE\uff0c\u624d\u5141\u8bb8\u751f\u6210\u8ba2\u5355\u8349\u7a3f\u3002', 'All human checks must be YES and order_intent must be PAPER or LIVE before an order draft is allowed.'),    ('\u4e0b\u4e00\u6b65', 'Next Step'),    ('\u5e94\u8be5\u505a Portfolio Exposure Dashboard\uff1a\u628a\u534a\u5bfc\u4f53\u3001\u79d1\u6280\u3001\u5927\u76d8\u3001\u4e2a\u80a1\u3001ETF \u91cd\u590d\u66b4\u9732\u4ee5\u8868\u683c\u5f62\u5f0f\u6c47\u603b\uff0c\u907f\u514d\u770b\u4f3c\u5206\u6563\u3001\u5b9e\u5219\u540c\u4e00\u4e2a\u4e3b\u9898\u8fc7\u5ea6\u96c6\u4e2d\u3002', 'should build the Portfolio Exposure Dashboard: summarize repeated exposure across semiconductors, tech, broad market, single names, and ETFs to avoid hidden concentration.'),    ('\u5f53\u524d\u6ca1\u6709', 'Currently missing'),    ('\u6240\u4ee5\u4e0d\u80fd\u8ba1\u7b97', 'so it cannot calculate'),    ('\u8fd9\u4e0d\u662f\u4ee3\u7801\u5931\u8d25\u3002', 'This is not a code failure.'),    ('\u5b83\u8bf4\u660e\u7cfb\u7edf\u6ca1\u6709\u671f\u6743\u94fe\u6570\u636e\u3002', 'It means the system has no options-chain data.'),    ('\u5148\u8fd0\u884c Step 23\uff0c\u5e76\u63d0\u4f9b Polygon API key \u6216\u624b\u52a8 `options_chain_input.csv`\u3002', 'Run Step 23 first and provide a Polygon API key or manual `options_chain_input.csv`.'),    ('\u8fd9\u5c42\u8981\u89e3\u51b3\u4ec0\u4e48\uff1f', 'What does this layer solve?'),    ('\u4f60\u8bf4\u7684\u201c\u505a\u5e02\u5546\u6740\u671f\u6743\u201d\u66f4\u4e25\u8c28\u5730\u62c6\u6210\u51e0\u7c7b\uff1a', 'The idea of market makers hurting options buyers is split into several risk types:'),    ('\u5230\u671f\u65e5\u524d\uff0c\u4ef7\u683c\u53ef\u80fd\u56f4\u7ed5\u9ad8 OI strike \u6ce2\u52a8\uff0c\u77ed\u671f\u671f\u6743\u4e70\u65b9\u88ab theta \u6d88\u8017\u3002', 'Before expiration, price may hover around high-OI strikes and short-term option buyers can lose to theta.'),    ('\u9ad8 OI strike \u9644\u8fd1\u53ef\u80fd\u5f62\u6210\u963b\u529b/\u652f\u6491\u6216\u7a81\u7834\u540e\u7684\u52a0\u901f\u70b9\u3002', 'High-OI strikes can become resistance/support or acceleration points after a break.'),    ('\u8d22\u62a5\u3001\u4e8b\u4ef6\u3001FOMC\u3001CPI \u540e\uff0c\u65b9\u5411\u5bf9\u4e86\u4f46 IV \u6389\u5f97\u592a\u5feb\uff0c\u671f\u6743\u4e5f\u53ef\u80fd\u4e8f\u3002', 'After earnings, events, FOMC, or CPI, options can lose even when direction is right if IV drops quickly.'),    ('\u592a\u5bbd\uff0c\u4e70\u8fdb\u53bb\u5c31\u4e8f\u4e00\u622a\u3002', 'is too wide, so the position loses immediately after entry.'),    ('\u5468\u5ea6 OTM options \u5982\u679c\u6ca1\u6709\u5feb\u901f\u7a81\u7834\uff0c\u4f1a\u88ab\u65f6\u95f4\u4ef7\u503c\u6d88\u8017\u3002', 'Weekly OTM options lose time value if there is no quick breakout.'),    ('\u5982\u679c dealer \u88ab\u8feb\u52a8\u6001\u5bf9\u51b2\uff0c\u53ef\u80fd\u9020\u6210 pinning\uff0c\u4e5f\u53ef\u80fd\u9020\u6210\u8d8b\u52bf\u52a0\u901f\u3002\u4f46 OI \u672c\u8eab\u4e0d\u80fd\u8bc1\u660e dealer \u771f\u4ed3\u4f4d\u3002', 'If dealers must dynamically hedge, that can cause pinning or trend acceleration, but OI alone does not prove dealer positioning.'),    ('\u5f53\u524d\u884c\u52a8', 'Current Action'),    ('\u5148\u4e0d\u8981\u628a gamma/market-maker \u903b\u8f91\u5199\u6210\u786e\u5b9a\u7ed3\u8bba\u3002\u5fc5\u987b\u7b49 Step 23 \u6709\u771f\u5b9e\u671f\u6743\u94fe\u6570\u636e\u540e\u518d\u6253\u5206\u3002', 'Do not write gamma or market-maker logic as a certain conclusion. Score it only after Step 23 has real options-chain data.'),    ('\u5927\u79d1\u6280\u66b4\u9732\uff1a\u4e0eQQQ/XLK\u76f8\u5173\u6027\u9ad8\uff0c\u9700\u68c0\u67e5\u79d1\u6280\u603b\u6743\u91cd\u3002', 'Mega-cap tech exposure: highly correlated with QQQ/XLK; check total tech weight.'),    ('\u957f\u503a\u66b4\u9732\uff1a\u5bf9\u5229\u7387\u53d8\u5316\u654f\u611f\uff0c\u6536\u76ca\u7387\u4e0a\u884c\u65f6\u53ef\u80fd\u62d6\u7d2f\u7ec4\u5408\u3002', 'Long-bond exposure: sensitive to rates; rising yields may drag the portfolio.'),    ('\u91cd\u590d\u66b4\u9732\uff1aSMH \u540c\u65f6\u51fa\u73b0\u5728\u591a\u4e2asleeve\uff0c\u603b\u66b4\u9732=12.9%\u3002', 'Duplicate exposure: SMH appears in multiple sleeves; total exposure = 12.9%.'),    ('\u91cd\u590d\u66b4\u9732\uff1aXLK \u540c\u65f6\u51fa\u73b0\u5728\u591a\u4e2asleeve\uff0c\u603b\u66b4\u9732=12.9%\u3002', 'Duplicate exposure: XLK appears in multiple sleeves; total exposure = 12.9%.'),    ('\u534a\u5bfc\u4f53\u66b4\u9732\uff1a\u9ad8\u5ea6\u53d7AI capex\u3001\u5e93\u5b58\u5468\u671f\u3001\u5229\u7387\u548c\u98ce\u9669\u504f\u597d\u5f71\u54cd\u3002', 'Semiconductor exposure: highly affected by AI capex, inventory cycle, rates, and risk appetite.'),    ('\u6536\u5165\u540c\u6bd4\u4e3a\u8d1f\uff1a\u82e5\u505a\u591a\uff0c\u9700\u8981\u66f4\u5f3a\u7684\u4e8b\u4ef6/\u4f30\u503c/\u53cd\u8f6c\u7406\u7531\u3002', 'Revenue growth is negative year-over-year; a long idea needs stronger event, valuation, or reversal support.'),    ('\u4e2d\u6027', 'Neutral'),    ('\u4ef7\u683c\u4e0a\u6da81%\u65f6\uff0c\u505a\u5e02\u5546\u5bf9\u51b2\u6d41\uff1a\u5356\u51fa', 'If price rises 1%, estimated dealer hedge flow: sell '),    ('\u80a1\uff08\u6291\u5236\u4e0a\u6da8\u52a8\u80fd\uff09', ' shares (dampens upside momentum)'),    ('\u505a\u5e02\u5546', 'dealer'),    ('\u5bf9\u51b2\u6d41', 'hedging flow'),    ('\u4e0a\u6da8\u52a8\u80fd', 'upside momentum'),]
RAW_STATUS_TEXT_REPLACEMENTS = {
    "PENDING_MANUAL_CHECKS": "Needs Human Check",
    "NO_DATA": "No Data",
    "RISK_REDUCTION_FIRST": "Reduce Risk First",
    "TINY_PAPER_ONLY": "Tiny Paper Test Only",
    "RESEARCH_ONLY": "Research Only",
    "FIX_DATA_FIRST": "Fix Data First",
    "DO_NOT_TOUCH": "Do Not Touch",
    "PRICE_DATA_UNAVAILABLE": "Price Data Unavailable",
    "NO_PRICE_PROXY": "No Usable Price",
    "NO_LIVE_NO_OPTIONS": "No Live Orders, No Options",
    "ONLY_SMALL_PAPER_AFTER_CHECKS": "Only Small Paper After Checks",
}
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
def translate_visible_text(value):
    if not isinstance(value, str) or not value:
        return value
    s = value
    for old, new in CHINESE_TEXT_REPLACEMENTS:
        s = s.replace(old, new)
    for old, new in RAW_STATUS_TEXT_REPLACEMENTS.items():
        s = s.replace(old, new)
    s = s.replace("\uFF0C", ", ").replace("\u3002", ".").replace("\u3001", ", ")
    if CJK_RE.search(s):
        lines = []
        for line in s.splitlines():
            if CJK_RE.search(line):
                lines.append("Legacy source note hidden in the English UI; the original file keeps the exact wording.")
            else:
                lines.append(line)
        s = "\n".join(lines)
    return s

def translate_dataframe_for_ui(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(translate_visible_text)
    return out

REPORT_ARCHIVE = [
    ("Daily", "Daily PM Report", "daily_pm_report.md"),
    ("Daily", "Tonight Plan", "tonight_action_plan.md"),
    ("Daily", "Action Board", "action_cards.md"),
    ("Portfolio And Risk", "Portfolio Overview", "exposure_dashboard.md"),
    ("Portfolio And Risk", "Stress Test And Size Plan", "stress_position_sizing_report.md"),
    ("Portfolio And Risk", "System Health Check", "system_health_check.md"),
    ("Action Check", "Before-Action Checklist", "pre_trade_checklist.md"),
    ("Action Check", "Paper Log Summary", "paper_ledger_summary.md"),
    ("Review Learning", "Review Learning Report", "learning_attribution_report.md"),
    ("Data And Layers", "Data Quality Report", "data_quality_report.md"),
    ("Data And Layers", "Layer 2 Market Mood", "macro_regime_report.md"),
    ("Data And Layers", "Layer 3 Sector Strength", "sector_rotation_report.md"),
    ("Data And Layers", "Layer 4 Company Basics", "fundamental_report.md"),
    ("Data And Layers", "Layer 5 News And Events", "event_news_sec_insider_report.md"),
    ("Data And Layers", "Layer 6 Price Trend", "technical_microstructure_report.md"),
    ("Options", "Yahoo Options Fetch", "yfinance_options_fetch_report.md"),
    ("Options", "Options Heat Report", "options_gamma_report.md"),
    ("Options", "Options Check", "options_gamma_diagnostics.md"),
    ("Options", "Short-Term Options Danger Zone", "option_kill_zone_report.md"),
    ("Options", "Options Layer Decision Table", "options_decision_matrix.md"),
    ("Options", "Options Daily Run Log", "options_daily_runner_log.md"),
    ("Options", "Options Decision Run Log", "options_decision_daily_runner_log.md"),
    ("Options", "Safe Options Action Log", "safe_options_action_runner_log.md"),
    ("10-Layer System", "10-Layer Structure", "canyon_10_layer_architecture.md"),
    ("10-Layer System", "Layer Build Plan", "canyon_layer_build_plan.md"),
    ("10-Layer System", "Main Decision v1", "master_10_layer_decision_report.md"),
    ("10-Layer System", "Main Decision v2", "master_10_layer_decision_report_v2.md"),
    ("Run Logs", "Missing Layers Run Log", "build_missing_layers_runner_log.md"),
    ("Run Logs", "Full Daily Run Log", "full_daily_pipeline_log.md"),
    ("Run Logs", "Full 10-Layer Run Log", "full_10_layer_daily_runner_v2_log.md"),
    ("Old Code Link", "Old Code Link Report", "v8_research_bridge_report.md"),
    ("Old Code Link", "Old Action Check Report", "v8_l9_execution_gate_report.md"),
    ("Old Code Link", "Old Risk Check Report", "v8_advanced_risk_report.md"),
    ("System", "Output Backup Report", "canyon_output_vault_report.md"),
    ("System", "Data Source Health Report", "data_source_health_report.md"),
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return translate_dataframe_for_ui(pd.read_csv(path, dtype=str).fillna(""))
    except Exception:
        return pd.DataFrame()


def read_md(path: Path) -> str:
    if not path.exists():
        return f"_Missing file: `{path.name}`_"
    try:
        return friendly_markdown_text(path.read_text(encoding="utf-8"))
    except Exception as e:
        return f"_Could not read `{path.name}`: {e}_"


def file_age_hours(path: Path) -> float | None:
    if not path.exists():
        return None
    modified = datetime.fromtimestamp(path.stat().st_mtime)
    return max((datetime.now() - modified).total_seconds() / 3600, 0)


def build_run_status() -> pd.DataFrame:
    required = [
        ("L1 data integrity", "market_data_snapshot.csv", 24),
        ("L2 macro", "macro_regime_signals.csv", 24),
        ("L3 sector rotation", "sector_rotation_scores.csv", 24),
        ("L4 fundamentals", "fundamental_quality_valuation.csv", 72),
        ("L5 event/news/SEC", "evidence_cards.csv", 24),
        ("L6 technical", "technical_signal_matrix.csv", 24),
        ("L7 options chain", "options_chain_snapshot.csv", 12),
        ("L7 options decision", "options_decision_matrix.csv", 12),
        ("L8 risk", "stress_position_sizing_report.md", 24),
        ("L9 pre-trade", "pre_trade_checklist.csv", 24),
        ("L10 learning", "learning_attribution_report.md", 72),
        ("Old research link", "v8_research_bridge_report.md", 24),
        ("Old action check", "v8_l9_execution_gate.csv", 24),
        ("Old risk check", "v8_advanced_risk_summary.csv", 24),
        ("Main decision", "master_10_layer_decision_matrix_v2.csv", 12),
        ("Data source health", "data_source_health.csv", 12),
        ("Output vault", "canyon_output_vault_index.csv", 24),
        ("Shrinkage alerts", "canyon_output_shrinkage_alerts.csv", 24),
        ("Dashboard action cards", "action_cards.csv", 12),
        ("Full runner log", "full_10_layer_daily_runner_v2_log.md", 12),
    ]

    rows = []
    for area, filename, stale_after in required:
        path = ROOT / filename
        age = file_age_hours(path)
        if age is None:
            status = "MISSING"
            age_display = ""
            modified = ""
        else:
            status = "FRESH" if age <= stale_after else "STALE"
            age_display = f"{age:.1f}"
            modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        rows.append({
            "status": status,
            "area": area,
            "file": filename,
            "age_hours": age_display,
            "stale_after_hours": stale_after,
            "last_modified": modified,
        })
    return pd.DataFrame(rows)


def build_report_archive_index() -> pd.DataFrame:
    rows = []
    for category, title, filename in REPORT_ARCHIVE:
        path = ROOT / filename
        exists = path.exists()
        age = file_age_hours(path)
        rows.append({
            "status": "FOUND" if exists else "MISSING",
            "category": category,
            "report": title,
            "file": filename,
            "age_hours": "" if age is None else f"{age:.1f}",
            "last_modified": "" if not exists else datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        })
    return pd.DataFrame(rows)


def classify_output_file(filename: str) -> str:
    name = filename.lower()
    if "vault" in name or "shrinkage" in name:
        return "System"
    if "data_source_health" in name:
        return "System"
    if name.startswith("v8_") or "v8_legacy" in name:
        return "V8 Research Bridge"
    if "option" in name or "gamma" in name or "kill_zone" in name:
        return "Options"
    if "risk" in name or "stress" in name or "exposure" in name or "position_sizing" in name:
        return "Portfolio / Risk"
    if "ledger" in name or "trade" in name or "pre_trade" in name or "execution" in name:
        return "Execution"
    if "learning" in name:
        return "Learning"
    if "macro" in name or "sector" in name or "fundamental" in name or "event" in name or "technical" in name:
        return "Data / Layers"
    if "data_quality" in name or "universe" in name or "market_data" in name or "layer" in name:
        return "Data / Layers"
    if "master" in name or "10_layer" in name or "architecture" in name:
        return "10-Layer"
    if "log" in name or "runner" in name or "pipeline" in name:
        return "Pipeline Logs"
    if "daily" in name or "tonight" in name or "action_cards" in name:
        return "Daily / PM"
    return "General"


def build_output_file_index() -> pd.DataFrame:
    rows = []
    seen = set()
    for pattern in ("*.md", "*.csv"):
        for path in sorted(ROOT.glob(pattern)):
            if path.name in seen:
                continue
            seen.add(path.name)
            age = file_age_hours(path)
            rows.append({
                "status": "FOUND",
                "category": classify_output_file(path.name),
                "kind": path.suffix.replace(".", "").upper(),
                "file": path.name,
                "size_kb": f"{path.stat().st_size / 1024:.1f}",
                "age_hours": "" if age is None else f"{age:.1f}",
                "last_modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            })
    return pd.DataFrame(rows)


def color_action(val):
    s = str(val).upper()
    if "TACTICAL_REVIEW" in s:
        return "background-color:#e9f8ef;color:#1e6b3a;"
    if "LONG_TERM_REVIEW" in s:
        return "background-color:#e8f2ff;color:#164a8b;"
    if "WAIT" in s:
        return "background-color:#e8f2ff;color:#164a8b;"
    if "PAPER" in s:
        return "background-color:#f2e8ff;color:#5b2c83;"
    if "RISK" in s:
        return "background-color:#ffeaea;color:#922222;"
    if "SKIP" in s or "DO_NOT_REPEAT" in s:
        return "background-color:#eeeeee;color:#555;"
    if "RESEARCH" in s:
        return "background-color:#e5fbff;color:#0d6370;"
    return ""


def color_state(val):
    s = str(val).upper()
    if s in {"OK", "RISK_ON", "LEADER", "QUALITY_HOLD_CANDIDATE", "EVENT_SUPPORT", "TACTICAL_CANDIDATE", "GREEN", "HAS_SAMPLE"}:
        return "background-color:#e9f8ef;color:#1e6b3a;"
    if s in {"MIXED_RISK", "PARTIAL", "NEUTRAL_OR_NO_EVENT", "AMBER", "PENDING_MANUAL_CHECKS"}:
        return "background-color:#e5fbff;color:#0d6370;"
    if s in {"RED", "RISK_OFF_OR_CHOPPY", "EVENT_RISK", "NO_TECH_EDGE"}:
        return "background-color:#ffeaea;color:#922222;"
    if "NO_DATA" in s or "NO_SAMPLE" in s:
        return "background-color:#eeeeee;color:#555;"
    if "SYNTHETIC" in s or "ETF_EVENT_CONTEXT" in s or "BENCHMARK_CONTEXT" in s or "HEDGE_CONTEXT" in s or "ETF_SECTOR_CONTEXT" in s or "OPTIONS_DATA_UNAVAILABLE" in s or "NO_LISTED_OPTIONS_CONTEXT" in s or "DATA_UNAVAILABLE" in s or "LEARNING_SAMPLE_PENDING" in s:
        return "background-color:#e5fbff;color:#0d6370;"
    if "ETF_NOT_FUNDAMENTAL" in s:
        return "background-color:#e8f2ff;color:#164a8b;"
    if "WAIT" in s or "WATCH" in s or "FUNDAMENTAL_WATCH" in s:
        return "background-color:#e8f2ff;color:#164a8b;"
    if "PAPER" in s:
        return "background-color:#f2e8ff;color:#5b2c83;"
    if "SKIP" in s or "BLOCKED" in s:
        return "background-color:#eeeeee;color:#555;"
    return ""


def status_kind(val):
    """Simplified 3-color system: risk (red) | warn (amber) | ok (green) | plain (no color)."""
    s = str(val).upper()
    # ── RED: needs immediate action ─────────────────────────────────────
    if any(x in s for x in [
        "RISK", "FAIL", "MISSING", "RED", "HIGH", "CRITICAL",
        "AT_TRIGGER", "FIX_DATA_FIRST", "EVENT_RISK", "DO_NOT_TOUCH",
        "HIGH_GAMMA", "HIGH_OPTION_KILL_ZONE", "STOP",
    ]) and "RISK_ON" not in s:
        return "risk"
    # ── GREEN: confirmed good ───────────────────────────────────────────
    if any(x in s for x in [
        "OK", "PASS", "FRESH", "FOUND", "YES", "TRUE", "ALLOW",
        "OPERATIONAL", "UPTREND", "LEADER", "HAS_SAMPLE",
        "CLOSED_PAPER", "CLOSED_REAL", "RISK_ON",
    ]):
        return "ok"
    # ── AMBER: review / caution / pending ───────────────────────────────
    if any(x in s for x in [
        "WARN", "STALE", "MEDIUM", "REVIEW", "WAIT", "WATCH",
        "PENDING", "NEAR_TRIGGER", "REDUCE", "USABLE", "RESEARCH",
        "PAPER", "OPEN_PAPER", "NEUTRAL", "MIXED", "LOW", "BACKLOG",
        "SYNTHETIC", "LEARNING", "NO_DATA", "NO_SAMPLE", "MANUAL",
        "OVERSIZED", "SKIP",
    ]):
        return "warn"
    # ── PLAIN: informational only ────────────────────────────────────────
    return "plain"


FRIENDLY_COLUMNS = {
    "status": "Status",
    "priority": "Priority",
    "ticker": "Ticker",
    "layer": "Layer",
    "section": "Section",
    "signal": "Signal Meaning",
    "score": "Score",
    "evidence": "Evidence",
    "next_action": "Next Step",
    "source_type": "Source Type",
    "source_file": "Source File",
    "source_detail": "Source Detail",
    "operator_read": "How To Use",
    "monitor": "Check Item",
    "reading": "Current Read",
    "origin_layer": "Source Layer",
    "source": "Source",
    "operator_action": "What To Do Now",
    "desk_signal": "Desk Alert",
    "source_layer": "Source Layer",
    "step": "#",
    "station": "Workflow Step",
    "what_to_do": "What To Do",
    "go_to": "Where To Look",
    "why_this_step_exists": "Why This Step Exists",
    "memo_line": "Memo Item",
    "answer": "Answer",
    "operator_takeaway": "Key Reminder",
    "question": "Question",
    "state": "Status",
    "count": "Count",
    "meaning": "Meaning",
    "stage": "Gate",
    "file": "File",
    "category": "Category",
    "report": "Report",
    "age_hours": "Age Hours",
    "last_modified": "Last Updated",
    "area": "Area",
    "kind": "Type",
    "size_kb": "Size KB",
    "message": "Note",
    "item": "Item",
    "level": "Level",
    "lane": "Queue",
    "gap_type": "Gap Type",
    "impact": "Impact",
    "next_fix": "How To Fix",
    "note": "Notes",
    "decision": "Decision",
    "urgency": "Urgency",
    "spot": "Current Price",
    "one_liner": "One-Line Rule",
    "allowed_action": "Allowed Action",
    "forbidden_action": "Forbidden Action",
    "trigger_rule": "Trigger Rule",
    "live_allowed": "Live Orders Allowed",
    "paper_allowed": "Paper Test Allowed",
    "final_status": "Final Status",
    "reasons": "Reason",
    "risk_light": "Risk Light",
    "risk_detail": "Risk Detail",
    "master_action": "Final Decision",
    "master_reason": "Final Reason",
    "stack_score_avg": "Average Score",
    "stack_score_min": "Lowest Score",
    "focus_bucket": "Focus Group",
    "focus_score": "Focus Score",
    "desk_status": "Today Status",
    "blocked_by": "Blocked By",
    "next_station": "Next Stop",
    "trigger_status": "Trigger Status",
    "nearest_trigger_distance_pct": "Distance To Trigger",
    "pretrade_status": "Check Status",
    "portfolio_risk_light": "Portfolio Risk Light",
    "final_options_decision": "Options Layer Decision",
    "rule": "Rule",
    "explanation": "Explanation",
    "gamma_squeeze_label": "Squeeze Heat",
    "option_kill_zone_label": "Short-Term Options Danger Zone",
    "gamma_label": "Squeeze Heat",
    "kill_zone_label": "Short-Term Options Danger Zone",
    "breakout_trigger": "Upside Trigger",
    "breakdown_trigger": "Downside Trigger",
    "call_wall_breakout_trigger": "Upper Key Price",
    "put_wall_breakdown_trigger": "Lower Key Price",
    "manual_news_check": "News Check",
    "earnings_date_check": "Earnings Date Check",
    "liquidity_check": "Liquidity Check",
    "spread_check": "Bid-Ask Spread Check",
    "duplicate_exposure_check": "Duplicate Exposure Check",
    "stress_check": "Stress Test Check",
    "suggested_action": "Suggested Action",
    "suggested_weight": "Suggested Size",
    "effective_weight": "Current Size",
    "sleeve": "Account Group",
    "risk_bucket": "Risk Group",
    "ledger_status": "Log Status",
    "data_status": "Data Status",
    "data_confidence": "Data Trust",
    "best_price_proxy": "Usable Price",
    "price_source_file": "Price Source",
    "source_age_hours": "Source Age Hours",
}

for _i in range(1, 11):
    FRIENDLY_COLUMNS[f"L{_i}_state"] = f"Layer {_i} Status"
    FRIENDLY_COLUMNS[f"L{_i}_score"] = f"Layer {_i} Score"
    FRIENDLY_COLUMNS[f"L{_i}_note"] = f"Layer {_i} Note"


FRIENDLY_VALUES = {
    "RISK_REDUCTION_FIRST": "Reduce Risk First",
    "TINY_PAPER_ONLY": "Tiny Paper Test Only",
    "PAPER_ONLY": "Paper Test Only",
    "RESEARCH_ONLY": "Research Only, Do Nothing Yet",
    "RESEARCH_ONLY_NO_NEW_RISK": "Research Only, Add No New Risk",
    "SKIP": "Skip",
    "BLOCKED": "Blocked",
    "DO_NOT_TOUCH": "Do Not Touch",
    "FIX_DATA_FIRST": "Fix Data First",
    "PENDING_MANUAL_CHECKS": "Needs Human Check",
    "ALREADY_CLOSED_DO_NOT_REPEAT": "Already Closed, Do Not Repeat",
    "WAIT": "Wait For Confirmation",
    "WATCH": "Watch",
    "ACTIVE_WATCH": "Main Watch",
    "PRIMARY_WATCH": "Priority Watch",
    "BACKLOG": "Backlog",
    "HIGH": "High",
    "MEDIUM": "Medium",
    "LOW": "Low",
    "RED": "Red Light, High Risk",
    "GREEN": "Green Light, Supportive",
    "AMBER": "Needs Review",
    "OK": "OK",
    "NO": "No",
    "YES": "is",
    "TRUE": "is",
    "FALSE": "No",
    "FOUND": "Found",
    "MISSING": "Missing",
    "FRESH": "Fresh",
    "STALE": "Stale",
    "REVIEW": "Needs Check",
    "WARN": "Warning",
    "RISK": "Risk",
    "NO_DATA": "No Data",
    "PRICE_DATA_UNAVAILABLE": "Price Data Unavailable",
    "NO_PRICE_PROXY": "No Usable Price",
    "DATA_UNAVAILABLE": "Data Unavailable",
    "OPTIONS_DATA_UNAVAILABLE": "Options Data Unavailable",
    "NO_LISTED_OPTIONS_CONTEXT": "No Listed Options Context",
    "MIXED_RISK": "Mixed Opportunity And Risk",
    "RISK_OFF_OR_CHOPPY": "Choppy Or Risk-Off Market",
    "RISK_ON": "Supportive Market",
    "LEADER": "Leader",
    "LAGGARD": "Laggard",
    "UPTREND": "Uptrend",
    "DOWNTREND": "Downtrend",
    "MIXED": "Mixed",
    "NEUTRAL": "Neutral",
    "NEUTRAL_OR_NO_EVENT": "No Clear Event",
    "EVENT_SUPPORT": "Event Support",
    "EVENT_RISK": "Event Risk",
    "TACTICAL_CANDIDATE": "Short-Term Watch Candidate",
    "NO_TECH_EDGE": "No Price-Trend Edge",
    "ETF_NOT_FUNDAMENTAL": "ETF, Not A Single Company",
    "ETF_SECTOR_CONTEXT": "ETF Sector Context Only",
    "MACRO_BENCHMARK_CONTEXT": "Market Benchmark Context",
    "BENCHMARK_CONTEXT": "Benchmark Context",
    "HEDGE_CONTEXT": "Hedge Context",
    "LEARNING_SAMPLE_PENDING": "Not Enough Learning Samples",
    "OPEN_OR_WATCH_SAMPLE": "Open Or Watch Sample",
    "HAS_SAMPLE": "Has Sample",
    "NO_SAMPLE": "No Sample",
    "AT_TRIGGER": "At Trigger",
    "NEAR_TRIGGER": "Near Trigger",
    "HIGH_GAMMA": "High Options Heat",
    "MEDIUM_GAMMA": "Medium Options Heat",
    "LOW_GAMMA": "Low Options Heat",
    "HIGH_OPTION_KILL_ZONE": "High Short-Term Options Danger",
    "MEDIUM_OPTION_KILL_ZONE": "Medium Short-Term Options Danger",
    "LOW_OPTION_KILL_ZONE": "Low Short-Term Options Danger",
    "OPERATIONAL": "Working",
    "USABLE": "Usable",
}


def friendly_label(value: str) -> str:
    s = str(value)
    return FRIENDLY_COLUMNS.get(s, s.replace("_", " ").title() if "_" in s else s)


def friendly_value(value) -> str:
    s = str(value)
    key = s.upper()
    if key in FRIENDLY_VALUES:
        return FRIENDLY_VALUES[key]
    return s


def friendly_sentence(value) -> str:
    s = friendly_value(value)
    replacements = {
        "Next:": "Next:",
        "Blocked:": "Blocked:",
        "Risk / Risk Control": "Risk / Risk Control",
        "Daily Desk / Ticker Drilldown": "Today / Single Ticker",
        "Research / Pre-Trade Gate": "Research / Before-Action Check",
        "L8 RED": "Red Risk Light",
        "high data gaps": " high-priority data gaps",
        "source risks": " source risks",
        "Portfolio risk is RED; reduce concentration before adding new ideas.": "Portfolio risk is red; reduce concentration before looking for new ideas.",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    for old, new in RAW_STATUS_TEXT_REPLACEMENTS.items():
        s = s.replace(old, new)
    return s


def friendly_markdown_text(value) -> str:
    """Make archived reports readable without changing the saved source files."""
    s = translate_visible_text(str(value))
    phrase_replacements = {
        "master_action": "Final Decision",
        "stack_score_avg": "Average Score",
        "stack_score_min": "Lowest Score",
        "ticker": "Ticker",
        "count": "Count",
        "states": "States",
        "layer": "Layer",
        "avg_score": "Average Score",
        "min_score": "Lowest Score",
        "generated": "Generated",
        "Generated:": "Generated:",
    }
    for raw, label in sorted(FRIENDLY_COLUMNS.items(), key=lambda kv: len(kv[0]), reverse=True):
        s = re_safe_word_replace(s, raw, label)
    for raw, label in sorted(FRIENDLY_VALUES.items(), key=lambda kv: len(kv[0]), reverse=True):
        s = re_safe_word_replace(s, raw, label)
    for raw, label in phrase_replacements.items():
        s = re_safe_word_replace(s, raw, label)
    return s


def re_safe_word_replace(text: str, raw: str, label: str) -> str:
    import re
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(str(raw)) + r"(?![A-Za-z0-9_])"
    return re.sub(pattern, str(label), text)


FRIENDLY_SECTION_LABELS = {
    "Macro Signals": "Market Mood Signals",
    "Breadth": "Market Breadth",
    "Volatility": "Volatility",
    "Raw Report": "Full Report",
    "Sector Scores": "Sector Scores",
    "Theme Heatmap": "Theme Heatmap",
    "Fundamental Matrix": "Basics Table",
    "Valuation Flags": "Valuation Alerts",
    "Evidence Cards": "Event Evidence Cards",
    "News": "News",
    "Earnings": "Earnings",
    "Insider": "Insider",
    "Technical Matrix": "Price Trend Table",
    "Liquidity": "Trading Activity",
    "Tactical Candidates": "Watch Candidates",
    "Breakout Watchlist": "Breakout Watchlist",
    "Options Decision": "Options Layer Decision",
    "V8 Synthetic Overlay": "Old Options Reference",
    "Gamma Candidates": "Options Heat List",
    "Kill Zone": "Options Danger Zone",
    "Raw Options Report": "Raw Options Report",
    "Raw Kill Zone Report": "Raw Danger-Zone Report",
    "Exposure": "Portfolio Exposure",
    "Warnings": "Warning",
    "Stress": "Stress Test",
    "Sizing": "Size Plan",
    "Advanced Risk": "More Risk Checks",
    "Raw Stress Report": "Raw Stress Report",
    "Raw Exposure Report": "Raw Portfolio Report",
    "Raw Advanced Risk": "Full Risk Check Report",
    "Pre-Trade Gate": "Before-Action Check",
    "V8 L9 Bridge": "Old Action Check",
    "Order Ticket": "Order Safety Ticket",
    "Paper Ledger": "Paper Log",
    "Raw Pre-Trade Report": "Raw Before-Action Report",
    "Raw V8 L9 Report": "Raw Old Check Report",
    "Learning Summary": "Review Summary",
    "Weight Suggestions": "Weight Suggestions",
    "Raw Learning Report": "Raw Review Report",
}


def friendly_section_label(name: str) -> str:
    return FRIENDLY_SECTION_LABELS.get(str(name), str(name))


def render_badge_table(df: pd.DataFrame, height=620):
    if df.empty:
        st.info("No data yet.")
        return

    # Auto-size: expand to show all rows without scrolling where possible.
    # max-height means small tables stay compact; large tables cap at 900px.
    _row_px = 48          # approx px per data row (matches canyon-table td padding)
    _head_px = 54         # thead + a bit of breathing room
    _auto_h = _head_px + len(df) * _row_px
    height = min(_auto_h, 900)   # never taller than 900px; never cuts rows unnecessarily

    html = [
        f'<div class="canyon-table-wrap" style="max-height:{height}px;">',
        '<table class="canyon-table">',
        "<thead><tr>",
    ]
    for col in df.columns:
        html.append(f"<th>{escape(friendly_label(str(col)))}</th>")
    html.append("</tr></thead><tbody>")

    badge_cols = {
        "master_action",
        "L8_state",
        "L9_state",
        "urgency",
        "risk_light",
        "final_status",
        "live_allowed",
        "live_allowed_effective",
        "priority",
        "status",
        "pretrade_status",
        "portfolio_risk_light",
        "final_options_decision",
        "trigger_status",
        "severity",
        "desk_status",
    }
    for _, row in df.iterrows():
        html.append("<tr>")
        for col in df.columns:
            value = row.get(col, "")
            cell = escape(friendly_sentence(value))
            classes = ["canyon-reason"] if col == "master_reason" else []
            if col in badge_cols:
                classes.extend(["canyon-status-cell", f"canyon-{status_kind(value)}"])
            html.append(f'<td class="{" ".join(classes)}">{cell}</td>')
        html.append("</tr>")

    html.append("</tbody></table></div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def show_df(df: pd.DataFrame, height=520, style=True):
    if df.empty:
        st.info("No data yet.")
        return
    # Auto-size: show all rows up to 900px
    _auto_h = 56 + len(df) * 38   # Streamlit dataframe row is ~38px
    height = min(_auto_h, 900)
    obj = df
    if style:
        obj = df.style.map(color_state)
        if "master_action" in df.columns:
            obj = obj.map(color_action, subset=["master_action"])
    try:
        st.dataframe(obj, hide_index=True, width="stretch", height=height)
    except TypeError:
        st.dataframe(obj, hide_index=True, use_container_width=True, height=height)


def query_value(name: str, default: str = "") -> str:
    try:
        value = st.query_params.get(name, default)
    except Exception:
        return default
    if isinstance(value, list):
        return str(value[0]) if value else default
    return str(value)


def as_ticker_list(df: pd.DataFrame, action: str) -> str:
    if df.empty or "ticker" not in df.columns or "master_action" not in df.columns:
        return ""
    rows = df[df["master_action"].astype(str).str.upper().eq(action)]
    return ", ".join(rows["ticker"].astype(str).tolist())


def top_values(df: pd.DataFrame, label_col: str, value_col: str, n: int = 5, ascending: bool = False) -> str:
    if df.empty or label_col not in df.columns or value_col not in df.columns:
        return "None"
    tmp = df.copy()
    tmp["_value"] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=["_value"]).sort_values("_value", ascending=ascending).head(n)
    if tmp.empty:
        return "None"
    return ", ".join(f"{r[label_col]} ({r['_value']:.1f})" for _, r in tmp.iterrows())


def count_value(df: pd.DataFrame, col: str, value: str) -> int:
    if df.empty or col not in df.columns:
        return 0
    return int(df[col].astype(str).str.upper().eq(value.upper()).sum())


def count_contains(df: pd.DataFrame, col: str, text: str) -> int:
    if df.empty or col not in df.columns:
        return 0
    return int(df[col].astype(str).str.upper().str.contains(text.upper(), regex=False).sum())


def render_layer_workbench_header(layer_id: str, title: str, thesis: str, metrics: list[tuple[str, str | int, str]]):
    _KIND_CSS = {
        # RED: danger / stop / fail
        "risk":       "background:#fef2f2;color:#b91c1c;",
        # BLUE: ok / active / noteworthy
        "ok":         "background:#eff6ff;color:#1e40af;",
        "warn":       "background:#eff6ff;color:#1e40af;",
        "supportive": "background:#eff6ff;color:#1e40af;",
        "watch":      "background:#eff6ff;color:#1e40af;",
        "cyan":       "background:#eff6ff;color:#1e40af;",
        "wait":       "background:#eff6ff;color:#1e40af;",
        "paper":      "background:#eff6ff;color:#1e40af;",
        # NO COLOR: neutral / informational
        "plain":      "background:transparent;color:#6b7280;",
        "blocked":    "background:transparent;color:#9ca3af;",
        "weak":       "background:transparent;color:#9ca3af;",
    }
    pills = []
    for label, value, kind in metrics:
        css = _KIND_CSS.get(kind, _KIND_CSS["plain"])
        pills.append(
            f'<span style="{css}border-radius:4px;padding:3px 10px;'
            f'font-size:12px;font-weight:600;white-space:nowrap;'
            f'font-family:\'IBM Plex Mono\',monospace;line-height:1.6;">'
            f'{escape(str(label))}&nbsp;&nbsp;{escape(str(value))}'
            f'</span>'
        )
    pills_html = '<div style="display:flex;gap:6px;flex-wrap:wrap;">' + ''.join(pills) + '</div>'
    st.markdown(
        f'<div style="border-bottom:1px solid #e5e7eb;padding:10px 0 14px 0;margin:0 0 20px 0;">'
        f'<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:6px;">'
        f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;font-weight:700;'
        f'color:#9ca3af;letter-spacing:0.10em;text-transform:uppercase;white-space:nowrap;">'
        f'{escape(layer_id)}</span>'
        f'<span style="font-size:16px;font-weight:700;color:#111827;white-space:nowrap;">'
        f'{escape(title)}</span>'
        f'</div>'
        f'<div style="margin-bottom:8px;">{pills_html}</div>'
        f'<div style="font-size:12px;color:#9ca3af;line-height:1.5;">{escape(thesis)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_layer_tables(tables: list[tuple[str, pd.DataFrame, list[str] | None, int]]):
    available = [(name, df, cols, height) for name, df, cols, height in tables if df is not None]
    if not available:
        st.info("No layer tables available yet.")
        return
    tabs = st.tabs([friendly_section_label(name) for name, _, _, _ in available])
    for tab, (_name, df, cols, height) in zip(tabs, available):
        with tab:
            if df.empty:
                st.info("No data yet.")
            elif _name == "Raw Report" and "content" in df.columns:
                st.markdown(str(df.iloc[0].get("content", "")))
            else:
                view = df[[c for c in cols if c in df.columns]] if cols else df
                render_badge_table(view, height=height)


def render_terminal_header():
    st.markdown("""
    <div class="classic-header">
      <div class="classic-title">Canyon v9 — 10-Layer Research Terminal</div>
      <div class="classic-subtitle">Full-stack decision view &nbsp;·&nbsp; Options = L7 only &nbsp;·&nbsp; No broker &nbsp;·&nbsp; No live order</div>
    </div>
    """, unsafe_allow_html=True)


def render_command_shell():
    master = read_csv(FILES["master_v2"])
    run_status = build_run_status()
    source_health = read_csv(FILES["data_source_health"])
    vault_alerts = read_csv(FILES["vault_alerts"])

    risk_light = "UNKNOWN"
    if not master.empty and "L8_state" in master.columns:
        risk_values = master["L8_state"].astype(str).str.upper()
        if risk_values.str.contains("RED", regex=False).any():
            risk_light = "RED"
        elif risk_values.str.contains("YELLOW", regex=False).any():
            risk_light = "CYAN"
        else:
            risk_light = "OK"

    stale_or_missing = 0
    if not run_status.empty and "status" in run_status.columns:
        stale_or_missing = int(run_status["status"].astype(str).str.upper().isin(["STALE", "MISSING"]).sum())

    source_risk = 0
    if not source_health.empty and "status" in source_health.columns:
        source_risk = int(source_health["status"].astype(str).str.upper().eq("RISK").sum())

    alert_count = 0
    if not vault_alerts.empty:
        if "status" in vault_alerts.columns:
            alert_mask = ~vault_alerts["status"].astype(str).str.upper().isin(["OK", ""])
            alert_count = int(alert_mask.sum())
        else:
            alert_count = len(vault_alerts)
    master_rows = len(master)
    now_label = datetime.now().strftime("%Y-%m-%d %H:%M")

    shell_class = "shell-risk" if risk_light == "RED" else "shell-cyan" if risk_light == "CYAN" else "shell-watch"
    stale_class = "shell-risk" if stale_or_missing else "shell-watch"
    source_class = "shell-risk" if source_risk else "shell-watch"
    vault_class = "shell-risk" if alert_count else "shell-watch"

    st.markdown(f"""
    <div class="product-shell">
      <div class="shell-left">
        <div class="shell-kicker">System Check</div>
        <div class="shell-title">Check If The System Is Reliable</div>
        <div class="shell-subtitle">Check risk, freshness, data source health, and missing outputs before reading tickers.</div>
      </div>
      <div class="shell-grid">
        <div class="shell-tile {shell_class}">
          <div class="shell-label">Risk Light</div>
          <div class="shell-value">{escape(friendly_value(risk_light))}</div>
        </div>
        <div class="shell-tile shell-plain">
          <div class="shell-label">Ticker Count</div>
          <div class="shell-value">{master_rows}</div>
        </div>
        <div class="shell-tile {stale_class}">
          <div class="shell-label">Stale / Missing</div>
          <div class="shell-value">{stale_or_missing}</div>
        </div>
        <div class="shell-tile {source_class}">
          <div class="shell-label">Source Risk</div>
          <div class="shell-value">{source_risk}</div>
        </div>
        <div class="shell-tile {vault_class}">
          <div class="shell-label">Backup Alerts</div>
          <div class="shell-value">{alert_count}</div>
        </div>
        <div class="shell-tile shell-plain">
          <div class="shell-label">Local Time</div>
          <div class="shell-value shell-time">{escape(now_label)}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_target_quad(queue: pd.DataFrame, title: str = "Top Four Targets"):
    if queue.empty:
        return
    cards = []
    for _, row in queue.head(4).iterrows():
        priority = str(row.get("priority", ""))
        ticker = str(row.get("ticker", ""))
        status = str(row.get("desk_status", row.get("master_action", "")))
        score = str(row.get("focus_score", ""))
        station = str(row.get("next_station", ""))
        blocked_by = str(row.get("blocked_by", ""))
        reason = str(row.get("master_reason", ""))
        if len(reason) > 120:
            reason = reason[:117].rstrip() + "..."
        cards.append(
            f'<div class="target-card target-{status_kind(priority or status)}">'
            f'<div class="target-top"><span class="target-priority">{escape(friendly_value(priority))}</span>'
            f'<span class="target-score">{escape(score)}</span></div>'
            f'<div class="target-ticker">{escape(ticker)}</div>'
            f'<div class="target-status">{escape(friendly_value(status))}</div>'
            f'<div class="target-line"><b>Next:</b> {escape(friendly_sentence(station))}</div>'
            f'<div class="target-line"><b>Blocked:</b> {escape(friendly_sentence(blocked_by))}</div>'
            f'<div class="target-reason">{escape(friendly_sentence(reason))}</div>'
            f'</div>'
        )
    st.markdown(
        f'<div class="target-section"><div class="target-section-title">{escape(title)}</div>'
        f'<div class="target-grid">{"".join(cards)}</div></div>',
        unsafe_allow_html=True,
    )


def render_workflow_steps(workflow_rows: pd.DataFrame):
    if workflow_rows.empty:
        st.info("No workflow steps yet.")
        return

    rows = workflow_rows.copy().reset_index(drop=True)
    rows["step"] = rows.index + 1

    def step_kind(status: str) -> str:
        s = str(status).upper()
        if s in {"RED", "RISK", "NO", "BLOCKED"}:
            return "risk" if s != "NO" else "blocked"
        if s in {"WAIT", "WATCH"}:
            return "wait"
        if s in {"PAPER"}:
            return "paper"
        if s in {"OK", "GREEN"}:
            return "supportive"
        return "cyan"

    detail_rules = {
        "Check Risk First": {
            "done_when": "Risk light, stress case, and concentration are understood before looking for new ideas.",
            "do_not_skip": "Do not let a strong ticker story override portfolio risk.",
            "output": "Risk posture for the day.",
        },
        "Check If Data Can Be Trusted": {
            "done_when": "Freshness, source health, and backup alerts are checked.",
            "do_not_skip": "Missing or fallback data cannot become a stronger signal.",
            "output": "Data trust status.",
        },
        "Pick Today’s Tickers First": {
            "done_when": "The first names to inspect are chosen from the queue, not from attention noise.",
            "do_not_skip": "Do not jump to options or triggers before choosing the right names.",
            "output": "A short focus list.",
        },
        "Read The Single-Ticker Note": {
            "done_when": "The 10-layer note explains support, blockers, allowed action, and forbidden action.",
            "do_not_skip": "One label is not enough to act on a ticker.",
            "output": "Ticker-level thesis and blocker stack.",
        },
        "Finish Human Checks": {
            "done_when": "News, earnings, liquidity, spread, duplicate exposure, and stress checks are complete.",
            "do_not_skip": "Manual checks are a gate, not a footnote.",
            "output": "Before-action clearance status.",
        },
        "Check Tickers Near Trigger": {
            "done_when": "Trigger, options heat, danger zone, risk, and action gate are checked together.",
            "do_not_skip": "A trigger is a review signal, not an entry order.",
            "output": "Trigger review result.",
        },
        "Write Today’s No-List": {
            "done_when": "Forbidden actions are explicit: no live orders, no short-term options chase, no full-size jump.",
            "do_not_skip": "The no-list protects attention and portfolio risk.",
            "output": "Hard boundaries for the day.",
        },
        "Record Learning Samples": {
            "done_when": "Any paper observation is logged cleanly for later attribution.",
            "do_not_skip": "Unlogged paper tests cannot teach Layer 10.",
            "output": "Clean paper-log trail.",
        },
    }

    counts = rows["status"].astype(str).str.upper().value_counts()
    risk_steps = int(counts.get("RED", 0) + counts.get("RISK", 0) + counts.get("NO", 0))
    review_steps = int(counts.get("REVIEW", 0))
    wait_steps = int(counts.get("WAIT", 0) + counts.get("WATCH", 0))
    first_station = str(rows.iloc[0].get("station", "")) if not rows.empty else ""

    st.markdown(f"""
    <div class="workflow-route-shell">
      <div class="workflow-route-head">
        <div>
          <div class="workflow-route-kicker">Daily Workflow</div>
          <div class="workflow-route-title">Morning Route From Risk To Review</div>
          <div class="workflow-route-text">Follow these steps in order. A later signal cannot skip an earlier risk, data, or human-check gate.</div>
        </div>
        <div class="workflow-route-metrics">
          <div><span>Steps</span><b>{len(rows)}</b></div>
          <div><span>Risk / No</span><b>{risk_steps}</b></div>
          <div><span>Review</span><b>{review_steps}</b></div>
          <div><span>Wait</span><b>{wait_steps}</b></div>
        </div>
      </div>
      <div class="workflow-first-gate"><b>First gate:</b> {escape(first_station)}</div>
    </div>
    """, unsafe_allow_html=True)

    compact = rows[[c for c in [
        "step", "status", "station", "what_to_do", "go_to", "origin_layer", "source_file"
    ] if c in rows.columns]].copy()
    render_badge_table(compact, height=min(520, 96 + len(compact) * 48))

    for _, row in rows.iterrows():
        station = str(row.get("station", ""))
        status = str(row.get("status", ""))
        idx = int(row.get("step", 0))
        rules = detail_rules.get(station, {
            "done_when": "This step has been reviewed and its source evidence is understood.",
            "do_not_skip": "Do not skip source review.",
            "output": "Reviewed decision input.",
        })
        with st.expander(f"Step {idx}: {station} — source and finish line"):
            cols = st.columns([1.1, 1.4, 1.4])
            with cols[0]:
                st.markdown(
                    f'<span class="canyon-swatch canyon-{step_kind(status)}">{escape(friendly_value(status))}</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Where to look:** {row.get('go_to', '')}")
                st.caption(f"Output: {rules['output']}")
            with cols[1]:
                st.markdown("**Why this step exists**")
                st.write(row.get("why_this_step_exists", ""))
                st.markdown("**Done when**")
                st.write(rules["done_when"])
            with cols[2]:
                st.markdown("**Source trail**")
                st.write(row.get("origin_layer", ""))
                st.caption(row.get("source_type", ""))
                st.code(str(row.get("source_file", "")), language="text")
                st.markdown("**Do not skip**")
                st.write(rules["do_not_skip"])


def build_today_workflow(
    risk_state: str,
    queue: pd.DataFrame,
    pretrade: pd.DataFrame,
    run_status: pd.DataFrame,
    health: pd.DataFrame,
    vault_risk: int,
) -> pd.DataFrame:
    stale_missing = 0
    if not run_status.empty and "status" in run_status.columns:
        stale_missing = int(run_status["status"].astype(str).str.upper().isin(["STALE", "MISSING"]).sum())
    source_risk = count_value(health, "status", "RISK")
    pending_manual = count_value(pretrade, "final_status", "PENDING_MANUAL_CHECKS")
    trigger_names = ""
    if not queue.empty and "trigger_status" in queue.columns:
        trigger_rows = queue[queue["trigger_status"].astype(str).str.upper().isin(["AT_TRIGGER", "NEAR_TRIGGER"])]
        trigger_names = ", ".join(trigger_rows["ticker"].astype(str).head(4).tolist()) if "ticker" in trigger_rows.columns else ""

    rows = [
        {
            "status": "RED" if str(risk_state).upper() == "RED" else "REVIEW",
            "station": "Check Risk First",
            "what_to_do": "Check the risk light, stress tests, concentration, and whether any new risk is allowed today.",
            "go_to": "Risk / Risk Control",
            "origin_layer": "Layer 8: Portfolio Risk",
            "source_type": "Layer Result + Stress Test Report",
            "source_file": "master_10_layer_decision_matrix_v2.csv; stress_position_sizing_report.md; exposure_warnings.csv",
            "why_this_step_exists": f"The risk layer is {friendly_value(risk_state)}; it is more important than options or price-trend signals.",
            "source_detail": "Risk state comes from the main decision table. Stress and concentration come from risk reports.",
        },
    ]

    if stale_missing or source_risk or vault_risk:
        rows.append({
            "status": "REVIEW",
            "station": "Check If Data Can Be Trusted",
            "what_to_do": f"Check stale/missing={stale_missing}, source risk={source_risk}, backup alerts={vault_risk} before trusting signals.",
            "go_to": "System / Run Status",
            "origin_layer": "Layer 1: Data Trust + System Check",
            "source_type": "Yahoo/yfinance Connection + Local Output Check",
            "source_file": "data_source_health.csv; canyon_output_shrinkage_alerts.csv; canyon_output_vault_index.csv",
            "why_this_step_exists": "Some freshness or source checks are not clean.",
            "source_detail": "The data source page shows Yahoo connection issues and whether local fallback data was used.",
        })

    rows.extend([
        {
            "status": "REVIEW",
            "station": "Pick Today’s Tickers First",
            "what_to_do": "Start with the four tickers most worth attention; ignore lower-value noise.",
            "go_to": "Today / Top Four",
            "origin_layer": "Main Decision + Focus Ranking",
            "source_type": "10-Layer Score Blend",
            "source_file": "master_10_layer_decision_matrix_v2.csv; watch_triggers.csv; pre_trade_checklist.csv",
            "why_this_step_exists": "The queue ranks by final decision, focus score, trigger distance, data gaps, and action checks.",
            "source_detail": "The top-four cards come from today’s queue, combining decisions, triggers, data gaps, and before-action checks.",
        },
        {
            "status": "REVIEW",
            "station": "Read The Single-Ticker Note",
            "what_to_do": "Read the full ticker explanation before options or trigger levels.",
            "go_to": "Today / Single Ticker",
            "origin_layer": "Full Layer 1-10 Check",
            "source_type": "Single-Ticker Research Note",
            "source_file": "master_10_layer_decision_matrix_v2.csv; action_cards.csv; evidence_cards.csv; technical_signal_matrix.csv; options_decision_matrix.csv",
            "why_this_step_exists": "Do not act from one label; the note explains evidence, blockers, and allowed/forbidden actions.",
            "source_detail": "The long explanation uses all 10 layers plus events, price trend, options, risk, and action checks.",
        },
    ])

    if pending_manual:
        rows.append({
            "status": "REVIEW",
            "station": "Finish Human Checks",
            "what_to_do": f"There are {pending_manual} rows needing news, earnings, liquidity, spread, duplicate exposure, and stress checks.",
            "go_to": "Research / Before-Action Check",
            "origin_layer": "Layer 5 Events + Layer 9 Action Check",
            "source_type": "Human Checklist + Event Evidence",
            "source_file": "pre_trade_checklist.csv; evidence_cards.csv; event_news_sec_insider_report.md",
            "why_this_step_exists": f"There are {pending_manual} rows still need before-action human checks.",
            "source_detail": "This checks news, earnings, SEC/insider context, liquidity, spreads, duplicate exposure, and stress.",
        })

    if trigger_names:
        rows.append({
            "status": "WAIT",
            "station": "Check Tickers Near Trigger",
            "what_to_do": f"Tickers near trigger: {trigger_names}. A trigger means review, not entry.",
            "go_to": "Today / Trigger Levels",
            "origin_layer": "Layer 6 Price Trend + Layer 7 Options",
            "source_type": "Trigger Distance + Options Heat/Danger Zone",
            "source_file": "watch_triggers.csv; options_decision_matrix.csv; option_kill_zone_risk.csv; gamma_squeeze_candidates.csv",
            "why_this_step_exists": f"One or more tickers are at or near trigger: {trigger_names}.",
            "source_detail": "Trigger levels come from trigger tables and options key levels/danger zones. Risk and action checks still apply.",
        })

    rows.extend([
        {
            "status": "NO",
            "station": "Write Today’s No-List",
            "what_to_do": "Confirm what is forbidden today: no live orders, no short-term options chase, no full-size conversion.",
            "go_to": "Today / If-Then Rules",
            "origin_layer": "System Safety Rules + Layer 9 Action Check",
            "source_type": "Hard Rule",
            "source_file": "pre_trade_checklist.csv; v8_l9_execution_gate.csv; action_cards.csv",
            "why_this_step_exists": "Canyon is research-only, with no broker connection and no live order path.",
            "source_detail": "Forbidden actions come from Layer 9 checks and system safety design. If risk is red or checks are incomplete, stay tiny or do nothing.",
        },
        {
            "status": "REVIEW",
            "station": "Record Learning Samples",
            "what_to_do": "If a paper observation changes, log it so Layer 10 can learn later, without over-trusting small samples.",
            "go_to": "Risk / Paper Log",
            "origin_layer": "Layer 10: Review Learning",
            "source_type": "Paper Log + Review Attribution",
            "source_file": "paper_portfolio_ledger.csv; learning_attribution_report.md; learning_weight_suggestions.csv",
            "why_this_step_exists": "Learning becomes useful only after enough paper samples; for now, record only and do not auto-change strategy.",
            "source_detail": "Layer 10 reads the paper log and review summaries. Small samples cannot directly change strategy weights.",
        },
    ])
    return pd.DataFrame(rows)


def build_overview_control_blotter(
    risk_state: str,
    master: pd.DataFrame,
    action_counts: pd.Series,
    health_counts: pd.Series,
    run_counts: pd.Series,
    vault_risk: int,
    pretrade: pd.DataFrame,
) -> pd.DataFrame:
    pending_manual = count_value(pretrade, "final_status", "PENDING_MANUAL_CHECKS")
    stale_missing = int(run_counts.get("STALE", 0)) + int(run_counts.get("MISSING", 0))
    source_risk = int(health_counts.get("RISK", 0))
    return pd.DataFrame([
        {
            "status": risk_state,
            "monitor": "Portfolio Risk Light",
            "reading": friendly_value(risk_state),
            "origin_layer": "Layer 8: Portfolio Risk",
            "source": "master_10_layer_decision_matrix_v2.csv; stress_position_sizing_report.md",
            "operator_action": "If risk is red, protect capital and review before adding exposure.",
        },
        {
            "status": "REVIEW" if source_risk else "OK",
            "monitor": "Data Source Health",
            "reading": f"{source_risk} rows have source risk",
            "origin_layer": "Layer 1: Data Trust",
            "source": "data_source_health.csv",
            "operator_action": "If Yahoo/yfinance fails, treat price and options data as unavailable or fallback.",
        },
        {
            "status": "REVIEW" if stale_missing else "OK",
            "monitor": "Was Today’s Run Fresh",
            "reading": f"{stale_missing} outputs are stale or missing",
            "origin_layer": "System Check",
            "source": "build_run_status(); generated files",
            "operator_action": "Run the full daily workflow before relying on conclusions.",
        },
        {
            "status": "RISK" if vault_risk else "OK",
            "monitor": "Output Backup",
            "reading": f"{vault_risk} shrinkage alerts",
            "origin_layer": "System Output Check",
            "source": "canyon_output_shrinkage_alerts.csv; canyon_output_vault_index.csv",
            "operator_action": "If outputs suddenly shrink, investigate before trusting the new run.",
        },
        {
            "status": "REVIEW" if pending_manual else "OK",
            "monitor": "Human Checks",
            "reading": f"{pending_manual} rows still need checks",
            "origin_layer": "Layer 5 Events + Layer 9 Action Check",
            "source": "pre_trade_checklist.csv; evidence_cards.csv",
            "operator_action": "Finish news, earnings, liquidity, spread, duplicate exposure, and stress checks.",
        },
        {
            "status": "REVIEW" if int(action_counts.get("RISK_REDUCTION_FIRST", 0)) else "OK",
            "monitor": "Reduce-Risk-First Queue",
            "reading": f"{int(action_counts.get('RISK_REDUCTION_FIRST', 0))} rows",
            "origin_layer": "Main Decision",
            "source": "master_10_layer_decision_matrix_v2.csv",
            "operator_action": "These tickers are not new opportunities until risk improves.",
        },
        {
            "status": "PAPER" if int(action_counts.get("TINY_PAPER_ONLY", 0)) else "OK",
            "monitor": "Tiny Paper Test Queue",
            "reading": f"{int(action_counts.get('TINY_PAPER_ONLY', 0))} rows",
            "origin_layer": "Main Decision + Layer 9",
            "source": "master_10_layer_decision_matrix_v2.csv; pre_trade_checklist.csv",
            "operator_action": "Tiny paper test means simulated tracking only, not live orders.",
        },
        {
            "status": "OK" if not master.empty else "MISSING",
            "monitor": "Main Decision Coverage",
            "reading": f"{len(master)} rows",
            "origin_layer": "Layer 1-10 Combined",
            "source": "master_10_layer_decision_matrix_v2.csv",
            "operator_action": "Open the single-ticker note before making any judgment.",
        },
    ])


def build_daily_desk_signal_blotter(queue: pd.DataFrame, pretrade: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    if queue.empty:
        return pd.DataFrame()

    high = count_value(queue, "priority", "HIGH")
    risk_reduction = count_value(queue, "desk_status", "RISK_REDUCTION_FIRST")
    fix_first = count_value(queue, "desk_status", "FIX_DATA_FIRST")
    manual = count_value(pretrade, "final_status", "PENDING_MANUAL_CHECKS")
    top_names = ", ".join(queue["ticker"].astype(str).head(4).tolist()) if "ticker" in queue.columns else ""
    risk_state = "NO_DATA"
    if not master.empty and "L8_state" in master.columns:
        _rs_mode = master["L8_state"].astype(str).mode()
        risk_state = _rs_mode.iloc[0] if not _rs_mode.empty else "UNKNOWN"

    return pd.DataFrame([
        {
            "status": "RED" if str(risk_state).upper() == "RED" else "REVIEW",
            "desk_signal": "Risk Decides What Is Allowed Today",
            "reading": f"Risk layer={friendly_value(risk_state)}; reduce-risk-first rows={risk_reduction}",
            "source_layer": "Layer 8 Portfolio Risk + Main Decision",
            "source_file": "master_10_layer_decision_matrix_v2.csv; stress_position_sizing_report.md",
            "operator_action": "Do not let options heat or price trend override portfolio risk.",
        },
        {
            "status": "HIGH" if high else "OK",
            "desk_signal": "What To Watch First Today",
            "reading": f"{high} high-priority rows; first tickers: {top_names}",
            "source_layer": "Main Decision + Focus Ranking",
            "source_file": "master_10_layer_decision_matrix_v2.csv; watch_triggers.csv; pre_trade_checklist.csv",
            "operator_action": "Start with these names, then open the single-ticker note.",
        },
        {
            "status": "REVIEW" if manual else "OK",
            "desk_signal": "Human Checks Are Not Finished",
            "reading": f"{manual} rows still need human checks",
            "source_layer": "Layer 5 Events + Layer 9 Action Check",
            "source_file": "pre_trade_checklist.csv; evidence_cards.csv",
            "operator_action": "Finish news, earnings, liquidity, spread, duplicate exposure, and stress checks.",
        },
        {
            "status": "REVIEW" if fix_first else "OK",
            "desk_signal": "Fix Data And Risk First",
            "reading": f"{fix_first} rows need fixing first",
            "source_layer": "Layer 1 Data + Layer 8/9 Checks",
            "source_file": "data_quality_flags.csv; market_data_snapshot.csv; master_10_layer_decision_matrix_v2.csv",
            "operator_action": "Fix source, price, and risk gaps before treating the ticker as researchable.",
        },
        {
            "status": "NO",
            "desk_signal": "What Is Clearly Forbidden Today",
            "reading": "No live orders; no short-term options chase; no full-size conversion",
            "source_layer": "System Safety Rules + Layer 9",
            "source_file": "pre_trade_checklist.csv; v8_l9_execution_gate.csv; action_cards.csv",
            "operator_action": "Treat this as today’s hard no-list.",
        },
    ])


def generate_daily_brief() -> str:
    master = read_csv(FILES["master_v2"])
    sectors = read_csv(FILES["sector_scores"])
    events = read_csv(FILES["events"])
    stress = read_csv(FILES["scenario_stress"])
    gaps = build_gap_queue(master, read_csv(FILES["market_snapshot"]))
    options_decision = read_csv(FILES["options_decision"])
    ledger = read_csv(FILES["paper_ledger"])
    pretrade = read_csv(FILES["pre_trade"])

    risk_state = "NO_DATA"
    if not master.empty and "L8_state" in master.columns:
        _rs_mode = master["L8_state"].astype(str).mode()
        risk_state = _rs_mode.iloc[0] if not _rs_mode.empty else "UNKNOWN"

    tiny = as_ticker_list(master, "TINY_PAPER_ONLY") or "None"
    reduce_first = as_ticker_list(master, "RISK_REDUCTION_FIRST") or "None"
    research = as_ticker_list(master, "RESEARCH_ONLY") or "None"
    skip = as_ticker_list(master, "SKIP") or "None"

    worst = "N/A"
    if not stress.empty and "estimated_loss" in stress.columns:
        loss = pd.to_numeric(stress["estimated_loss"], errors="coerce")
        if not loss.dropna().empty:
            idx = loss.idxmin()
            scenario = stress.loc[idx, "scenario"] if "scenario" in stress.columns else "N/A"
            worst = f"{scenario} ({loss.loc[idx] * 100:.2f}%)"

    event_risk = "None"
    if not events.empty and "event_label" in events.columns:
        rows = events[events["event_label"].astype(str).eq("EVENT_RISK")]
        if not rows.empty:
            event_risk = ", ".join(rows["ticker"].astype(str).tolist())

    paper_only = 0
    if not options_decision.empty and "final_options_decision" in options_decision.columns:
        paper_only = int((options_decision["final_options_decision"] == "PAPER_ONLY").sum())

    closed = 0
    needed = 5
    if not ledger.empty and "status" in ledger.columns:
        closed = int(ledger["status"].astype(str).str.upper().isin(["CLOSED_PAPER", "CLOSED_REAL"]).sum())
        needed = max(5 - closed, 0)

    pending = 0
    blocked = 0
    if not pretrade.empty and "final_status" in pretrade.columns:
        pending = int((pretrade["final_status"] == "PENDING_MANUAL_CHECKS").sum())
        blocked = int((pretrade["final_status"] == "BLOCKED").sum())

    high_gaps = 0 if gaps.empty else int((gaps["priority"] == "HIGH").sum())

    return f"""# Canyon v9 Daily Brief

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Top Line
- Risk light: **{risk_state}**
- Worst stress scenario: **{worst}**
- Live trading: **NO**
- Short-dated options: **NO**
- Current max expression: **tiny stock/ETF paper only, after manual checks**

## Today Actions
- Tiny paper only: **{tiny}**
- Risk reduction first: **{reduce_first}**
- Research only: **{research}**
- Skip: **{skip}**

## Research Stack
- Sector leaders: **{top_values(sectors, "ticker", "rotation_score", n=4)}**
- Sector laggards: **{top_values(sectors, "ticker", "rotation_score", n=4, ascending=True)}**
- Event risk tickers: **{event_risk}**

## Options Layer
- Paper-only options decisions: **{paper_only}**
- Rule: options are L7 only; L8/L9 still override.
- Current expression: stock/ETF paper, not weekly OTM options.

## Execution Gate
- Pending manual checks: **{pending}**
- Blocked rows: **{blocked}**
- Required checks: news, earnings date, liquidity, spread, duplicate exposure, stress.

## Data / Learning
- High-priority data gaps: **{high_gaps}**
- Closed paper samples: **{closed}**
- More closed samples needed before L10 adjustment: **{needed}**
"""


def merge_master_action(master: pd.DataFrame, cards: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame()
    base_cols = [c for c in [
        "ticker", "master_action", "master_reason", "stack_score_avg",
        "L3_state", "L6_state", "L7_state", "L8_state", "L9_state"
    ] if c in master.columns]
    out = master[base_cols].copy()
    if not cards.empty and "ticker" in cards.columns:
        card_cols = [c for c in [
            "ticker", "allowed_action", "forbidden_action", "trigger_rule",
            "breakout_trigger", "breakdown_trigger", "live_allowed"
        ] if c in cards.columns]
        out = out.merge(cards[card_cols], on="ticker", how="left")
    return out.fillna("")


def first_row(df: pd.DataFrame, ticker: str) -> dict:
    if df.empty or "ticker" not in df.columns:
        return {}
    rows = df[df["ticker"].astype(str).eq(str(ticker))]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def ticker_rows(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df.empty or "ticker" not in df.columns:
        return pd.DataFrame()
    return df[df["ticker"].astype(str).str.upper().eq(str(ticker).upper())].copy()


def row_value(row: dict, *keys: str, default: str = "") -> str:
    for key in keys:
        value = row.get(key, "")
        if str(value).strip():
            return str(value)
    return default


def build_ticker_dossier(
    ticker: str,
    master_row: dict,
    card_row: dict,
    trigger_row: dict,
    pretrade_row: dict,
    options_row: dict,
    v8_gate_row: dict,
    tech_row: dict,
    event_row: dict,
    gap_rows: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "status": row_value(master_row, "master_action", default="NO_DATA"),
            "section": "Final action",
            "evidence": row_value(master_row, "master_reason", default="No master decision yet."),
            "next_action": "Respect L8/L9 before any expression.",
        },
        {
            "status": row_value(master_row, "L8_state", pretrade_row.get("risk_light", ""), default="NO_DATA"),
            "section": "Portfolio risk",
            "evidence": row_value(master_row, "L8_note", pretrade_row.get("risk_detail", ""), default="No risk note."),
            "next_action": "If RED, reduce risk first and do not use options.",
        },
        {
            "status": row_value(pretrade_row, "final_status", v8_gate_row.get("final_status", ""), default="NO_DATA"),
            "section": "Execution gate",
            "evidence": row_value(pretrade_row, "reasons", v8_gate_row.get("reasons", ""), default="No execution row."),
            "next_action": row_value(pretrade_row, "paper_allowed", v8_gate_row.get("paper_allowed", ""), default="Research only until checked."),
        },
        {
            "status": row_value(options_row, "final_options_decision", master_row.get("L7_state", ""), default="NO_DATA"),
            "section": "Options layer",
            "evidence": row_value(options_row, "explanation", master_row.get("L7_note", ""), default="No options edge."),
            "next_action": "Options are L7 only; never override L8/L9.",
        },
        {
            "status": row_value(tech_row, "technical_label", master_row.get("L6_state", ""), default="NO_DATA"),
            "section": "Technical timing",
            "evidence": row_value(tech_row, "reasons", master_row.get("L6_note", ""), default="No technical confirmation."),
            "next_action": "Use only as timing confirmation, not standalone thesis.",
        },
        {
            "status": row_value(event_row, "event_label", master_row.get("L5_state", ""), default="NO_DATA"),
            "section": "Event / news",
            "evidence": row_value(event_row, "reasons", master_row.get("L5_note", ""), default="No event edge."),
            "next_action": "Manual news/earnings/SEC check before paper action.",
        },
        {
            "status": "HIGH" if not gap_rows.empty and (gap_rows["priority"] == "HIGH").any() else ("OK" if gap_rows.empty else "REVIEW"),
            "section": "Data gaps",
            "evidence": "No ticker-specific gaps." if gap_rows.empty else f"{len(gap_rows)} gap rows; highest priority: {gap_rows['priority'].iloc[0]}",
            "next_action": "Fix high-priority gaps before trusting price-sensitive conclusions.",
        },
        {
            "status": row_value(card_row, "decision", trigger_row.get("decision", ""), default=ticker),
            "section": "Trigger plan",
            "evidence": row_value(card_row, "trigger_rule", trigger_row.get("action", ""), default="No trigger rule."),
            "next_action": row_value(card_row, "allowed_action", default="Research only."),
        },
    ])


def build_ticker_intelligence_panel(
    ticker: str,
    master_row: dict,
    card_row: dict,
    trigger_row: dict,
    pretrade_row: dict,
    options_row: dict,
    v8_gate_row: dict,
    tech_row: dict,
    event_row: dict,
    market_row: dict,
    gap_rows: pd.DataFrame,
) -> pd.DataFrame:
    gap_count = len(gap_rows) if gap_rows is not None else 0
    high_gaps = 0
    if gap_rows is not None and not gap_rows.empty and "priority" in gap_rows.columns:
        high_gaps = int(gap_rows["priority"].astype(str).str.upper().eq("HIGH").sum())

    market_source = row_value(market_row, "price_source_file", default="market_data_snapshot.csv")
    market_detail = (
        f"price={row_value(market_row, 'best_price_proxy', default='N/A')}; "
        f"confidence={row_value(market_row, 'data_confidence', default='N/A')}; "
        f"age_hours={row_value(market_row, 'source_age_hours', default='N/A')}"
    )

    return pd.DataFrame([
        {
            "status": row_value(master_row, "L1_state", default="NO_DATA"),
            "layer": "L1 Data Trust",
            "signal": row_value(market_row, "data_confidence", master_row.get("L1_note", ""), default="NO_DATA"),
            "score": row_value(master_row, "L1_score", default=""),
            "source_type": "Local price/source integrity",
            "source_file": f"market_data_snapshot.csv; data_quality_flags.csv; {market_source}",
            "source_detail": market_detail,
            "operator_read": "If this layer is weak, every downstream price/options/technical read becomes conditional.",
        },
        {
            "status": row_value(master_row, "L2_state", default="NO_DATA"),
            "layer": "L2 Macro Regime",
            "signal": row_value(master_row, "L2_note", default="No macro state."),
            "score": row_value(master_row, "L2_score", default=""),
            "source_type": "Index breadth, volatility, rates, credit proxy",
            "source_file": "macro_regime_signals.csv; index_breadth_dashboard.csv; volatility_regime.csv; macro_regime_report.md",
            "source_detail": "Read from master v2 L2 fields generated by Step 48.",
            "operator_read": "Macro decides whether the ticker is swimming with or against the broader tape.",
        },
        {
            "status": row_value(master_row, "L3_state", default="NO_DATA"),
            "layer": "L3 Sector / Theme",
            "signal": row_value(master_row, "L3_note", default="No sector state."),
            "score": row_value(master_row, "L3_score", default=""),
            "source_type": "Sector and theme rotation",
            "source_file": "sector_rotation_scores.csv; theme_heatmap.csv; sector_rotation_report.md",
            "source_detail": "Read from master v2 L3 fields generated by Step 49.",
            "operator_read": "Sector leadership changes attention priority, but it does not bypass risk or execution gates.",
        },
        {
            "status": row_value(master_row, "L4_state", default="NO_DATA"),
            "layer": "L4 Quality / Valuation",
            "signal": row_value(master_row, "L4_note", default="No fundamental state."),
            "score": row_value(master_row, "L4_score", default=""),
            "source_type": "Fundamental, ETF context, valuation flags",
            "source_file": "fundamental_quality_valuation.csv; valuation_risk_flags.csv; fundamental_report.md",
            "source_detail": "ETF rows may intentionally show ETF_NOT_FUNDAMENTAL instead of company fundamentals.",
            "operator_read": "Separates long-term hold quality from a short-term tactical setup.",
        },
        {
            "status": row_value(event_row, "event_label", master_row.get("L5_state", ""), default="NO_DATA"),
            "layer": "L5 Event / News",
            "signal": row_value(event_row, "reasons", master_row.get("L5_note", ""), default="No event state."),
            "score": row_value(event_row, "event_score", master_row.get("L5_score", ""), default=""),
            "source_type": "Event card plus manual-news gate",
            "source_file": "evidence_cards.csv; news_event_risk.csv; earnings_calendar_check.csv; insider_form4_signals.csv",
            "source_detail": "Local generated evidence card; live external news still requires manual confirmation when marked pending.",
            "operator_read": "Fresh event risk can block an otherwise attractive setup.",
        },
        {
            "status": row_value(tech_row, "technical_label", master_row.get("L6_state", ""), default="NO_DATA"),
            "layer": "L6 Technical Timing",
            "signal": row_value(tech_row, "reasons", master_row.get("L6_note", ""), default="No technical state."),
            "score": row_value(tech_row, "technical_score", master_row.get("L6_score", ""), default=""),
            "source_type": "Trend, RSI/ATR, range, liquidity proxy",
            "source_file": "technical_signal_matrix.csv; tactical_candidates.csv; breakout_reversal_watchlist.csv; intraday_liquidity_proxy.csv",
            "source_detail": (
                f"close={row_value(tech_row, 'close', default='N/A')}; "
                f"ret_20d={row_value(tech_row, 'ret_20d', default='N/A')}; "
                f"rsi14={row_value(tech_row, 'rsi14', default='N/A')}"
            ),
            "operator_read": "Technical confirmation is timing context, not a standalone permission slip.",
        },
        {
            "status": row_value(options_row, "final_options_decision", master_row.get("L7_state", ""), default="NO_DATA"),
            "layer": "L7 Options / Gamma",
            "signal": row_value(options_row, "explanation", master_row.get("L7_note", ""), default="No options state."),
            "score": row_value(master_row, "L7_score", default=row_value(options_row, "gamma_squeeze_score", default="")),
            "source_type": "Options OI/volume proxy, walls, kill zone",
            "source_file": "options_decision_matrix.csv; gamma_squeeze_candidates.csv; option_kill_zone_risk.csv; options_gamma_report.md",
            "source_detail": (
                f"gamma={row_value(options_row, 'gamma_squeeze_label', default='N/A')}; "
                f"kill_zone={row_value(options_row, 'option_kill_zone_label', default='N/A')}; "
                f"max_pain={row_value(options_row, 'max_pain_proxy', default='N/A')}"
            ),
            "operator_read": "Options are pressure context only. L7 cannot override L8 risk or L9 execution.",
        },
        {
            "status": row_value(master_row, "L8_state", pretrade_row.get("risk_light", ""), default="NO_DATA"),
            "layer": "L8 Portfolio Risk",
            "signal": row_value(master_row, "L8_note", pretrade_row.get("risk_detail", ""), default="No portfolio risk state."),
            "score": row_value(master_row, "L8_score", default=""),
            "source_type": "Stress, sizing, concentration, exposure",
            "source_file": "stress_position_sizing_report.md; scenario_stress_results.csv; position_sizing_recommendations.csv; exposure_warnings.csv",
            "source_detail": row_value(pretrade_row, "risk_detail", master_row.get("L8_note", ""), default="No L8 detail."),
            "operator_read": "This layer controls the maximum expression size. RED means risk reduction first.",
        },
        {
            "status": row_value(pretrade_row, "final_status", v8_gate_row.get("final_status", ""), master_row.get("L9_state", ""), default="NO_DATA"),
            "layer": "L9 Execution Gate",
            "signal": row_value(pretrade_row, "reasons", v8_gate_row.get("reasons", ""), master_row.get("L9_note", ""), default="No execution gate."),
            "score": row_value(master_row, "L9_score", default=""),
            "source_type": "Pre-trade checklist, manual gates, v8 bridge",
            "source_file": "pre_trade_checklist.csv; pre_trade_order_ticket.csv; v8_l9_execution_gate.csv; action_cards.csv",
            "source_detail": (
                f"paper_allowed={row_value(pretrade_row, 'paper_allowed', v8_gate_row.get('paper_allowed', ''), default='N/A')}; "
                f"live_allowed={row_value(pretrade_row, 'live_allowed', v8_gate_row.get('live_allowed', ''), card_row.get('live_allowed', ''), default='NO')}"
            ),
            "operator_read": "This is the permission boundary. No live order path is enabled.",
        },
        {
            "status": "HIGH" if high_gaps else row_value(master_row, "L10_state", default="NO_DATA"),
            "layer": "L10 Learning / Gaps",
            "signal": f"{gap_count} active ticker gap rows; high={high_gaps}. {row_value(master_row, 'L10_note', default='')}",
            "score": row_value(master_row, "L10_score", default=""),
            "source_type": "Learning sample plus gap queue",
            "source_file": "learning_attribution_summary.csv; learning_weight_suggestions.csv; canyon_gap_queue derived in dashboard",
            "source_detail": "Learning remains conservative until enough closed paper trades exist.",
            "operator_read": "Record outcomes, but do not overfit a tiny sample or hide unresolved source gaps.",
        },
    ])


def render_ticker_source_trail(panel: pd.DataFrame):
    if panel.empty:
        return
    with st.expander("Open Why Each Layer Made This Decision"):
        for _, row in panel.iterrows():
            st.markdown(
                f"**{escape(str(row.get('layer', 'Layer')))}** "
                f"`{escape(str(row.get('status', '')))} `"
            )
            st.caption(f"Source: {row.get('source_file', '')}")
            st.write(row.get("operator_read", ""))
            detail = pd.DataFrame([{
                "status": row.get("status", ""),
                "signal": row.get("signal", ""),
                "source_type": row.get("source_type", ""),
                "source_detail": row.get("source_detail", ""),
            }])
            render_badge_table(detail, height=150)


def build_ticker_memo(
    ticker: str,
    master_row: dict,
    card_row: dict,
    trigger_row: dict,
    pretrade_row: dict,
    v8_gate_row: dict,
    gap_rows: pd.DataFrame,
) -> pd.DataFrame:
    action = row_value(master_row, "master_action", default="NO_DATA")
    l8_state = row_value(master_row, "L8_state", pretrade_row.get("risk_light", ""), default="NO_DATA")
    l9_state = row_value(pretrade_row, "final_status", v8_gate_row.get("final_status", ""), default=row_value(master_row, "L9_state", default="NO_DATA"))
    allowed = row_value(card_row, "allowed_action", pretrade_row.get("paper_allowed", ""), v8_gate_row.get("paper_allowed", ""), default="Research only")
    forbidden = row_value(card_row, "forbidden_action", default="Live orders / short-dated options / forced entries")
    trigger_status = row_value(trigger_row, "trigger_status", card_row.get("urgency", ""), default="NO_TRIGGER")
    one_liner = row_value(card_row, "one_liner", master_row.get("master_reason", ""), default="No one-line rule yet.")
    master_reason = row_value(master_row, "master_reason", default="")
    if master_reason and master_reason not in one_liner:
        decision_answer = f"{action}: {master_reason} Current operating rule: {one_liner}"
    else:
        decision_answer = f"{action}: {one_liner}"

    blockers = []
    if str(l8_state).upper() == "RED":
        blockers.append("L8 portfolio risk is RED")
    if "PENDING" in str(l9_state).upper() or "BLOCKED" in str(l9_state).upper():
        blockers.append(f"L9 execution gate is {l9_state}")
    if not gap_rows.empty:
        high = int((gap_rows["priority"].astype(str).str.upper() == "HIGH").sum()) if "priority" in gap_rows.columns else len(gap_rows)
        blockers.append(f"{high} high-priority data/risk gaps" if high else f"{len(gap_rows)} data gaps")
    if not blockers:
        blockers.append("No hard blocker found, but manual review still applies.")

    if str(l8_state).upper() == "RED":
        next_station = "Risk / Risk Control"
    elif "PENDING" in str(l9_state).upper() or "BLOCKED" in str(l9_state).upper():
        next_station = "Research / Pre-Trade Gate"
    elif not gap_rows.empty:
        next_station = "System / Data Gaps"
    elif str(trigger_status).upper() in {"AT_TRIGGER", "NEAR_TRIGGER"}:
        next_station = "Daily Desk / Trigger Board"
    else:
        next_station = "Daily Desk / Focus List"

    return pd.DataFrame([
        {
            "status": action,
            "memo_line": "Decision",
            "answer": decision_answer,
            "operator_takeaway": "Treat this as the current research conclusion, not a trade instruction. The final action must still pass L8 portfolio risk, L9 execution gate, and the manual checklist.",
        },
        {
            "status": l8_state,
            "memo_line": "Main blocker",
            "answer": "; ".join(blockers),
            "operator_takeaway": "This is the reason the ticker cannot be upgraded just because one layer looks interesting. If the blocker stack includes RED risk, pending execution checks, or data gaps, the name remains research-first.",
        },
        {
            "status": l9_state,
            "memo_line": "Allowed now",
            "answer": allowed,
            "operator_takeaway": "Allowed means the maximum permitted research expression under the current system state. It does not mean buy, and it never enables live execution.",
        },
        {
            "status": "NO",
            "memo_line": "Forbidden now",
            "answer": forbidden,
            "operator_takeaway": "Forbidden actions stay forbidden even if price is moving or options flow looks exciting. No weekly OTM options, no live orders, and no forced entries.",
        },
        {
            "status": trigger_status,
            "memo_line": "Trigger context",
            "answer": row_value(card_row, "trigger_rule", trigger_row.get("action", ""), default="No trigger rule available."),
            "operator_takeaway": "A trigger is only a reason to re-open the checklist. It is not automatic entry, and it still needs spread, liquidity, event, stress, and duplicate exposure review.",
        },
        {
            "status": "REVIEW",
            "memo_line": "Next station",
            "answer": next_station,
            "operator_takeaway": f"Start there before spending more time on {ticker}; the next station is where the current decision can actually improve.",
        },
    ])


def build_ticker_decision_narrative(
    ticker: str,
    master_row: dict,
    card_row: dict,
    trigger_row: dict,
    pretrade_row: dict,
    options_row: dict,
    v8_gate_row: dict,
    tech_row: dict,
    event_row: dict,
    gap_rows: pd.DataFrame,
) -> str:
    action = row_value(master_row, "master_action", default="NO_DATA")
    reason = row_value(master_row, "master_reason", default="No master reason is available yet.")
    stack_score = row_value(master_row, "stack_score_avg", default="N/A")
    min_score = row_value(master_row, "stack_score_min", default="N/A")
    l1 = row_value(master_row, "L1_state", default="NO_DATA")
    l2 = row_value(master_row, "L2_state", default="NO_DATA")
    l3 = row_value(master_row, "L3_state", default="NO_DATA")
    l4 = row_value(master_row, "L4_state", default="NO_DATA")
    l5 = row_value(event_row, "event_label", master_row.get("L5_state", ""), default="NO_DATA")
    l6 = row_value(tech_row, "technical_label", master_row.get("L6_state", ""), default="NO_DATA")
    l7 = row_value(options_row, "final_options_decision", master_row.get("L7_state", ""), default="NO_DATA")
    l8 = row_value(master_row, "L8_state", pretrade_row.get("risk_light", ""), default="NO_DATA")
    l9 = row_value(pretrade_row, "final_status", v8_gate_row.get("final_status", ""), master_row.get("L9_state", ""), default="NO_DATA")
    l10 = row_value(master_row, "L10_state", default="NO_DATA")
    allowed = row_value(card_row, "allowed_action", pretrade_row.get("paper_allowed", ""), v8_gate_row.get("paper_allowed", ""), default="Research only.")
    forbidden = row_value(card_row, "forbidden_action", default="Live order, full-size trade, short-dated options, and forced entry.")
    trigger_rule = row_value(card_row, "trigger_rule", trigger_row.get("action", ""), default="No trigger rule is available.")
    option_note = row_value(options_row, "explanation", master_row.get("L7_note", ""), default="No options explanation is available.")
    risk_note = row_value(master_row, "L8_note", pretrade_row.get("risk_detail", ""), default="No portfolio risk note is available.")
    execution_note = row_value(pretrade_row, "reasons", v8_gate_row.get("reasons", ""), default="No execution checklist note is available.")

    gap_summary = "No ticker-specific data gap is currently listed."
    if not gap_rows.empty:
        high = int((gap_rows["priority"].astype(str).str.upper() == "HIGH").sum()) if "priority" in gap_rows.columns else 0
        medium = int((gap_rows["priority"].astype(str).str.upper() == "MEDIUM").sum()) if "priority" in gap_rows.columns else 0
        lanes = ", ".join(gap_rows["lane"].astype(str).drop_duplicates().head(3).tolist()) if "lane" in gap_rows.columns else "data gap queue"
        gap_summary = f"{len(gap_rows)} ticker-specific gap rows are active: high={high}, medium={medium}. Main lanes: {lanes}."

    decision_posture = "research-only"
    if "TINY_PAPER" in str(action).upper() or "PAPER" in str(allowed).upper():
        decision_posture = "tiny paper only after manual checks"
    if str(l8).upper() == "RED":
        decision_posture = "risk-reduction first; no new aggressive expression"

    return f"""
### Decision Narrative

**Current answer for {ticker}: {action}.** The system is not saying "buy" or "sell"; it is saying the current maximum posture is **{decision_posture}**. The reason is: {reason}

The 10-layer stack is mixed rather than clean. The average stack score is **{stack_score}** and the weakest layer score is **{min_score}**. L1 is **{l1}**, which matters because missing or weak price/source data makes every downstream timing conclusion less reliable. L2 is **{l2}**, L3 is **{l3}**, and L4 is **{l4}**, so the top-down context has to be read as part of the decision rather than ignored. L5 is **{l5}** and L6 is **{l6}**, meaning the event and technical layers are not allowed to become a standalone trade thesis without the rest of the stack.

The options layer is **{l7}**. Its note is: {option_note} Options remain only L7; they can explain pressure, pinning, squeeze risk, or kill-zone risk, but they cannot override portfolio risk or execution gates. The risk layer is **{l8}**: {risk_note} The execution layer is **{l9}**: {execution_note} L10 is **{l10}**, so learning feedback should be recorded but not overfit.

**Allowed now:** {allowed}  
**Forbidden now:** {forbidden}  
**Trigger rule:** {trigger_rule}  
**Data / blocker summary:** {gap_summary}

The practical next step is to resolve the blocking layer first, then revisit the ticker from the top of the stack. If L8 remains RED or L9 remains pending, the correct behavior is to keep this as research-only or tiny underlying paper after manual checks, not to convert it into options or live execution.
"""


def build_layer_thesis_stack(master_row: dict) -> pd.DataFrame:
    layer_questions = {
        "L1": "Can we trust ticker, price, timestamp, and source?",
        "L2": "Is the broad regime supportive or hostile?",
        "L3": "Is sector/theme leadership helping this name?",
        "L4": "Is this business/ETF context worth holding or only trading?",
        "L5": "Is there fresh event/news/SEC/insider risk or support?",
        "L6": "Is price action confirming the thesis?",
        "L7": "Are options/dealer gamma adding pressure or kill-zone risk?",
        "L8": "Does portfolio risk allow any new expression?",
        "L9": "What exactly is allowed before execution?",
        "L10": "Do we have enough learning samples to adapt?",
    }
    rows = []
    for i in range(1, 11):
        layer = f"L{i}"
        rows.append({
            "layer": layer,
            "question": layer_questions[layer],
            "status": master_row.get(f"{layer}_state", "NO_DATA"),
            "score": master_row.get(f"{layer}_score", ""),
            "evidence": master_row.get(f"{layer}_note", ""),
        })
    return pd.DataFrame(rows)


def build_what_would_change(
    ticker: str,
    master_row: dict,
    pretrade_row: dict,
    v8_gate_row: dict,
    gap_rows: pd.DataFrame,
    tech_row: dict,
) -> pd.DataFrame:
    action = row_value(master_row, "master_action", default="NO_DATA")
    l2 = row_value(master_row, "L2_state", default="NO_DATA")
    l6 = row_value(tech_row, "technical_label", master_row.get("L6_state", ""), default="NO_DATA")
    l8 = row_value(master_row, "L8_state", pretrade_row.get("risk_light", ""), default="NO_DATA")
    l9 = row_value(pretrade_row, "final_status", v8_gate_row.get("final_status", ""), default=row_value(master_row, "L9_state", default="NO_DATA"))
    high_gaps = 0
    if gap_rows is not None and not gap_rows.empty and "priority" in gap_rows.columns:
        high_gaps = int(gap_rows["priority"].astype(str).str.upper().eq("HIGH").sum())

    rows = []

    # Upgrade conditions
    if str(l8).upper() == "RED":
        rows.append({
            "direction": "UPGRADE",
            "if_this_happens": "L8 portfolio risk improves from RED",
            "then_this_changes": "Paper trades may become allowed; smaller sizing can be considered",
            "how_to_watch_for_it": "Reduce concentration / close stressed positions first",
            "where_to_check": "Portfolio Risk › Risk Control",
        })
    if "PENDING" in str(l9).upper() or "BLOCKED" in str(l9).upper():
        rows.append({
            "direction": "UPGRADE",
            "if_this_happens": "Execution gate (L9) clears from PENDING or BLOCKED",
            "then_this_changes": "Before-action checklist passes; paper entry may be allowed",
            "how_to_watch_for_it": "Resolve manual checklist items in Pre-Trade Gate",
            "where_to_check": "Research Room › Before-Action Check",
        })
    if high_gaps > 0:
        rows.append({
            "direction": "UPGRADE",
            "if_this_happens": f"{high_gaps} high-priority data gaps close",
            "then_this_changes": "More layers have real data; decision confidence rises",
            "how_to_watch_for_it": "Run the full daily runner; wait for data refresh",
            "where_to_check": "System Check › Gap List",
        })
    if str(l2).upper() not in {"SUPPORTIVE", "OK", "BULL", "NO_DATA", ""}:
        rows.append({
            "direction": "UPGRADE",
            "if_this_happens": "Market mood (L2) turns supportive",
            "then_this_changes": "Broad regime no longer blocks ticker-level aggression",
            "how_to_watch_for_it": "Macro data shifts; typically confirmed over days to weeks",
            "where_to_check": "Middle Layers › Market Mood",
        })
    if str(l6).upper() in {"NO_TECH_EDGE", "WAIT", "NO_DATA", "WEAK"}:
        rows.append({
            "direction": "UPGRADE",
            "if_this_happens": "Technical timing (L6) confirms — trigger level reached",
            "then_this_changes": "Price action aligns with thesis; name moves to tactical candidate",
            "how_to_watch_for_it": "Check breakout / breakdown trigger levels daily",
            "where_to_check": "Middle Layers › Price Trend; Daily Plan › Trigger Levels",
        })

    # Downgrade conditions
    rows.append({
        "direction": "DOWNGRADE",
        "if_this_happens": "Event risk appears — earnings, SEC filing, or insider sell",
        "then_this_changes": "L5 event flag forces decision back to Research Only until event clears",
        "how_to_watch_for_it": "Check earnings calendar; review insider form 4 weekly",
        "where_to_check": "Middle Layers › Events And News",
    })
    if str(l8).upper() != "RED":
        rows.append({
            "direction": "DOWNGRADE",
            "if_this_happens": "Portfolio stress hits RED (L8 turns RED)",
            "then_this_changes": "Risk protection overrides everything; no new position aggression",
            "how_to_watch_for_it": "Monitor exposure and stress test after any large market move",
            "where_to_check": "Portfolio Risk › Risk Control; Portfolio Risk › Stress Test",
        })
    rows.append({
        "direction": "DOWNGRADE",
        "if_this_happens": "Data source becomes unavailable or fallback-only",
        "then_this_changes": "L1 data trust degrades; all downstream decisions become conditional",
        "how_to_watch_for_it": "Run status goes stale or data source turns RISK",
        "where_to_check": "System Check › Data Sources; System Check › Run Status",
    })

    if not rows:
        rows.append({
            "direction": "STABLE",
            "if_this_happens": "No specific change condition identified",
            "then_this_changes": f"Current decision ({action}) appears stable under current data",
            "how_to_watch_for_it": "Continue monitoring via daily workflow",
            "where_to_check": "Daily Plan › Today Overview",
        })

    return pd.DataFrame(rows)


def build_action_board_table() -> pd.DataFrame:
    cards = read_csv(FILES["action_cards"])
    master = read_csv(FILES["master_v2"])
    options = read_csv(FILES["options_decision"])
    pretrade = read_csv(FILES["pre_trade"])

    if cards.empty:
        return pd.DataFrame()

    out = cards.copy()
    if not master.empty and "ticker" in master.columns:
        master_cols = [c for c in [
            "ticker", "master_action", "master_reason", "stack_score_avg",
            "L1_state", "L2_state", "L3_state", "L6_state", "L7_state", "L8_state", "L9_state"
        ] if c in master.columns]
        out = out.merge(master[master_cols], on="ticker", how="left")

    if not options.empty and "ticker" in options.columns:
        option_cols = [c for c in [
            "ticker", "final_options_decision", "portfolio_risk_light",
            "pretrade_status", "paper_allowed", "rule", "explanation"
        ] if c in options.columns]
        out = out.merge(options[option_cols], on="ticker", how="left", suffixes=("", "_option"))

    if not pretrade.empty and "ticker" in pretrade.columns:
        pre_cols = [c for c in [
            "ticker", "final_status", "manual_news_check", "earnings_date_check",
            "liquidity_check", "spread_check", "duplicate_exposure_check", "stress_check",
            "paper_allowed", "live_allowed", "reasons"
        ] if c in pretrade.columns]
        renamed = pretrade[pre_cols].rename(columns={
            "paper_allowed": "pretrade_paper_allowed",
            "live_allowed": "pretrade_live_allowed",
            "reasons": "pretrade_reasons",
        })
        out = out.merge(renamed, on="ticker", how="left")

    if "live_allowed" in out.columns and "pretrade_live_allowed" in out.columns:
        out["live_allowed_effective"] = out["pretrade_live_allowed"].where(out["pretrade_live_allowed"].astype(str).str.strip().ne(""), out["live_allowed"])
    elif "live_allowed" in out.columns:
        out["live_allowed_effective"] = out["live_allowed"]
    else:
        out["live_allowed_effective"] = ""

    return out.fillna("")


def build_research_lab_routes() -> pd.DataFrame:
    macro = read_csv(FILES["macro_signals"])
    sectors = read_csv(FILES["sector_scores"])
    fundamentals = read_csv(FILES["fundamentals"])
    events = read_csv(FILES["events"])
    technicals = read_csv(FILES["technicals"])
    options = read_csv(FILES["options_decision"])
    v8_inventory = read_csv(FILES["v8_inventory"])
    pretrade = read_csv(FILES["pre_trade"])

    return pd.DataFrame([
        {
            "status": "REVIEW" if count_value(macro, "data_status", "NO_DATA") else "OK",
            "station": "Layer 2: Market Mood",
            "what_it_answers": "Is the broad market helping or hurting?",
            "current_signal": f"{len(macro)} signals; NO_DATA={count_value(macro, 'data_status', 'NO_DATA')}",
            "go_to": "Middle Layers / Market Mood",
        },
        {
            "status": "REVIEW" if count_value(sectors, "rotation_label", "NO_DATA") else "OK",
            "station": "Layer 3: Sectors",
            "what_it_answers": "Which sector or theme deserves attention?",
            "current_signal": f"leaders={count_value(sectors, 'rotation_label', 'LEADER')}; NO_DATA={count_value(sectors, 'rotation_label', 'NO_DATA')}",
            "go_to": "Middle Layers / Sectors",
        },
        {
            "status": "REVIEW" if count_value(fundamentals, "fundamental_label", "NO_DATA") else "OK",
            "station": "Layer 4: Company Basics",
            "what_it_answers": "Is this worth owning, or only watching short term?",
            "current_signal": f"ETF context={count_value(fundamentals, 'fundamental_label', 'ETF_NOT_FUNDAMENTAL')}; NO_DATA={count_value(fundamentals, 'fundamental_label', 'NO_DATA')}",
            "go_to": "Middle Layers / Company Basics",
        },
        {
            "status": "RISK" if count_value(events, "event_label", "EVENT_RISK") else "OK",
            "station": "Layer 5: News And Events",
            "what_it_answers": "Can fresh news or earnings change the setup?",
            "current_signal": f"event risk={count_value(events, 'event_label', 'EVENT_RISK')}; rows={len(events)}",
            "go_to": "Research Room / Evidence Board",
        },
        {
            "status": "REVIEW" if count_value(technicals, "technical_label", "NO_TECH_EDGE") else "OK",
            "station": "Layer 6: Price Trend",
            "what_it_answers": "Is the chart confirming the timing?",
            "current_signal": f"candidates={count_value(technicals, 'technical_label', 'TACTICAL_CANDIDATE')}; no edge={count_value(technicals, 'technical_label', 'NO_TECH_EDGE')}",
            "go_to": "Middle Layers / Price Trend",
        },
        {
            "status": "RESEARCH_ONLY" if count_value(options, "final_options_decision", "RESEARCH_ONLY") else "OK",
            "station": "Layer 7: Options",
            "what_it_answers": "Is options pressure helpful, dangerous, or unavailable?",
            "current_signal": f"research only={count_value(options, 'final_options_decision', 'RESEARCH_ONLY')}; paper only={count_value(options, 'final_options_decision', 'PAPER_ONLY')}",
            "go_to": "Last Four Layers / Options",
        },
        {
            "status": "OK",
            "station": "Old Code Link",
            "what_it_answers": "Which older research modules are available, connected, or blocked?",
            "current_signal": f"available={count_value(v8_inventory, 'integration_status', 'AVAILABLE_RESEARCH')}; blocked={count_value(v8_inventory, 'integration_status', 'BLOCKED_NO_LIVE')}",
            "go_to": "Research Room / Old Code Link",
        },
        {
            "status": "REVIEW" if count_value(pretrade, "final_status", "PENDING_MANUAL_CHECKS") else "OK",
            "station": "Before-Action Check",
            "what_it_answers": "What is allowed, forbidden, and still manually blocked?",
            "current_signal": f"pending={count_value(pretrade, 'final_status', 'PENDING_MANUAL_CHECKS')}; blocked={count_value(pretrade, 'final_status', 'BLOCKED')}",
            "go_to": "Research Room / Before-Action Check",
        },
    ])


def build_risk_control_routes() -> pd.DataFrame:
    exposure = read_csv(FILES["exposure"])
    warnings = read_csv(FILES["exposure_warnings"])
    stress = read_csv(FILES["scenario_stress"])
    sizing = read_csv(FILES["position_sizing"])
    adv = read_csv(FILES["v8_adv_risk"])
    ledger = read_csv(FILES["paper_ledger"])

    largest = "N/A"
    if not exposure.empty and {"ticker", "effective_weight"}.issubset(exposure.columns):
        tmp = exposure.copy()
        tmp["effective_weight_num"] = pd.to_numeric(tmp["effective_weight"], errors="coerce")
        row = tmp.sort_values("effective_weight_num", ascending=False).head(1)
        if not row.empty:
            largest = f"{row.iloc[0]['ticker']} ({row.iloc[0]['effective_weight_num'] * 100:.1f}%)"

    worst = "N/A"
    if not stress.empty and "estimated_loss" in stress.columns:
        losses = pd.to_numeric(stress["estimated_loss"], errors="coerce")
        if not losses.dropna().empty:
            idx = losses.idxmin()
            scenario = stress.loc[idx, "scenario"] if "scenario" in stress.columns else "stress"
            worst = f"{scenario}: {losses.loc[idx] * 100:.2f}%"

    high_warn = count_value(warnings, "level", "HIGH")
    med_warn = count_value(warnings, "level", "MEDIUM")
    adv_risk = count_value(adv, "status", "RISK")
    closed = count_contains(ledger, "status", "CLOSED")

    return pd.DataFrame([
        {
            "status": "RED" if high_warn else ("WARN" if med_warn else "OK"),
            "station": "Portfolio Map",
            "risk_question": "Where is exposure concentrated?",
            "current_signal": f"largest={largest}; high warnings={high_warn}; medium warnings={med_warn}",
            "control_action": "Reduce overlap and avoid stacking ETF/component exposure.",
            "go_to": "Portfolio Risk / Portfolio Map",
        },
        {
            "status": "RISK" if worst != "N/A" else "NO_DATA",
            "station": "Stress Test",
            "risk_question": "What happens if the regime shifts against us?",
            "current_signal": f"worst={worst}",
            "control_action": "Use stress breach flags before allowing any paper expression.",
            "go_to": "Portfolio Risk / Stress Test",
        },
        {
            "status": "RISK" if adv_risk else "OK",
            "station": "More Risk Checks",
            "risk_question": "Is the portfolio secretly relying on one hidden risk?",
            "current_signal": f"advanced risk flags={adv_risk}; data source={adv.iloc[0].get('data_source', 'UNKNOWN') if not adv.empty else 'NO_DATA'}",
            "control_action": "Treat synthetic fallback as plumbing validation until online data refreshes.",
            "go_to": "Portfolio Risk / More Risk Checks",
        },
        {
            "status": "REVIEW" if not sizing.empty else "NO_DATA",
            "station": "Sizing",
            "risk_question": "What size is allowed after concentration and stress?",
            "current_signal": f"sizing rows={len(sizing)}; reduced={count_contains(sizing, 'suggested_action', 'REDUCED')}",
            "control_action": "Use suggested size as a cap, not a target.",
            "go_to": "Last Four Layers / Portfolio Risk",
        },
        {
            "status": "HAS_SAMPLE" if closed >= 30 else "LEARNING_SAMPLE_PENDING",
            "station": "Paper Log",
            "risk_question": "Are paper samples clean enough to learn from?",
            "current_signal": f"closed samples={closed}; ledger rows={len(ledger)}",
            "control_action": "Keep paper tests tiny and attribution-clean.",
            "go_to": "Portfolio Risk / Paper Log",
        },
    ])


def build_system_control_routes() -> pd.DataFrame:
    reports = build_report_archive_index()
    generated = build_output_file_index()
    run_status = build_run_status()
    health = read_csv(FILES["data_source_health"])
    vault_alerts = read_csv(FILES["vault_alerts"])
    master = read_csv(FILES["master_v2"])
    gaps = build_gap_queue(master, read_csv(FILES["market_snapshot"]))
    checks = build_qa_checks()

    report_missing = count_value(reports, "status", "MISSING")
    run_missing = count_value(run_status, "status", "MISSING")
    stale = count_value(run_status, "status", "STALE")
    data_risk = count_value(health, "status", "RISK")
    qa_high = count_value(checks, "severity", "HIGH")
    real_alerts = int((vault_alerts["status"].astype(str).str.upper() != "OK").sum()) if not vault_alerts.empty and "status" in vault_alerts.columns else 0

    return pd.DataFrame([
        {
            "status": "MISSING" if report_missing else "OK",
            "station": "Report Archive",
            "system_question": "Are the original reports still present?",
            "current_signal": f"tracked={len(reports)}; missing={report_missing}; generated files={len(generated)}",
            "control_action": "Use Report Archive before assuming old reports disappeared.",
            "go_to": "System / Report Archive",
        },
        {
            "status": "RISK" if real_alerts else "OK",
            "station": "Output Vault",
            "system_question": "Did any output shrink or disappear after rebuilds?",
            "current_signal": f"alerts={real_alerts}; snapshots={read_csv(FILES['vault_index'])['snapshot_id'].nunique() if FILES['vault_index'].exists() else 0}",
            "control_action": "If alerts appear, compare latest snapshot before trusting the page.",
            "go_to": "System / Output Vault",
        },
        {
            "status": "RISK" if data_risk else "OK",
            "station": "Data Source Health",
            "system_question": "Can online data fetches be trusted right now?",
            "current_signal": f"risk={data_risk}; warn={count_value(health, 'status', 'WARN')}; ok={count_value(health, 'status', 'OK')}",
            "control_action": "When DNS is RISK, treat yfinance-dependent layers as fallback/research-only.",
            "go_to": "System / Data Source Health",
        },
        {
            "status": "RISK" if qa_high else "OK",
            "station": "System QA",
            "system_question": "Are there consistency or safety issues?",
            "current_signal": f"high={qa_high}; medium={count_value(checks, 'severity', 'MEDIUM')}; low={count_value(checks, 'severity', 'LOW')}",
            "control_action": "Resolve HIGH issues before decision-ready use.",
            "go_to": "System / System QA",
        },
        {
            "status": "REVIEW" if not gaps.empty else "OK",
            "station": "Data Gaps",
            "system_question": "Which tickers/layers need source repair?",
            "current_signal": f"gaps={len(gaps)}; high={count_value(gaps, 'priority', 'HIGH')}; affected={gaps['ticker'].nunique() if not gaps.empty and 'ticker' in gaps.columns else 0}",
            "control_action": "Fix high-priority L1/L8/L9 gaps before trusting action output.",
            "go_to": "System / Data Gaps",
        },
        {
            "status": "MISSING" if run_missing else ("STALE" if stale else "FRESH"),
            "station": "Run Status",
            "system_question": "Are core outputs fresh enough?",
            "current_signal": f"fresh={count_value(run_status, 'status', 'FRESH')}; stale={stale}; missing={run_missing}",
            "control_action": "Run the full v2 daily workflow before relying on stale outputs.",
            "go_to": "System / Run Status",
        },
    ])


def build_gap_queue(master: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame()

    rows = []
    market_lookup = {}
    if not market.empty and "ticker" in market.columns:
        market_lookup = {str(r["ticker"]): r.to_dict() for _, r in market.iterrows()}

    layer_names = {
        "L1": "Data integrity",
        "L2": "Macro regime",
        "L3": "Sector/theme",
        "L4": "Fundamental",
        "L5": "Event/SEC/insider",
        "L6": "Technical",
        "L7": "Options/gamma",
        "L8": "Portfolio risk",
        "L9": "Execution gate",
        "L10": "Learning sample",
    }

    for _, row in master.iterrows():
        ticker = str(row.get("ticker", ""))
        action = str(row.get("master_action", ""))
        for i in range(1, 11):
            layer = f"L{i}"
            state = str(row.get(f"{layer}_state", ""))
            note = str(row.get(f"{layer}_note", ""))
            score = str(row.get(f"{layer}_score", ""))

            if layer == "L8" and state == "RED":
                rows.append({
                    "priority": "HIGH",
                    "lane": "Risk / Portfolio Blocker",
                    "ticker": ticker,
                    "gap_type": "Risk override",
                    "layer": layer,
                    "layer_name": layer_names[layer],
                    "state": state,
                    "score": score,
                    "impact": "Caps every action to tiny paper or research only.",
                    "next_fix": "Reduce concentration, review stress exposure, and do not use options until risk improves.",
                    "note": note,
                    "master_action": action,
                })
            elif "NO_PRICE" in state or (layer == "L1" and "NO_DATA" in state):
                m = market_lookup.get(ticker, {})
                rows.append({
                    "priority": "HIGH",
                    "lane": "Data Blocker",
                    "ticker": ticker,
                    "gap_type": "Missing price proxy",
                    "layer": layer,
                    "layer_name": layer_names[layer],
                    "state": state,
                    "score": score,
                    "impact": "Blocks action before the research stack can be trusted.",
                    "next_fix": "Refresh market snapshot or add current spot/close for this ticker.",
                    "note": m.get("notes", note),
                    "master_action": action,
                })
            elif "NO_DATA" in state:
                optional_context = layer == "L3" and ticker in {"SPY", "QQQ", "GLD", "TLT"}
                priority = "LOW" if optional_context or action in {"RESEARCH_ONLY", "RISK_REDUCTION_FIRST"} else "MEDIUM"
                lane = "Context Backlog" if optional_context else "Research Backlog"
                rows.append({
                    "priority": priority,
                    "lane": lane,
                    "ticker": ticker,
                    "gap_type": "Layer input missing",
                    "layer": layer,
                    "layer_name": layer_names[layer],
                    "state": state,
                    "score": score,
                    "impact": "Weakens the 10-layer explanation and may keep ticker in research only.",
                    "next_fix": f"Populate {layer_names[layer]} input for this ticker or mark it explicitly not applicable.",
                    "note": note,
                    "master_action": action,
                })
            elif layer == "L9" and ("PENDING" in state or "BLOCKED" in state):
                rows.append({
                    "priority": "HIGH" if "BLOCKED" in state else "MEDIUM",
                    "lane": "Execution Blocker",
                    "ticker": ticker,
                    "gap_type": "Execution/manual gate",
                    "layer": layer,
                    "layer_name": layer_names[layer],
                    "state": state,
                    "score": score,
                    "impact": "Prevents clean paper action until checklist is reviewed.",
                    "next_fix": "Complete manual news, earnings, liquidity, spread, duplicate exposure, and stress checks.",
                    "note": note,
                    "master_action": action,
                })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    out["_rank"] = out["priority"].map(rank).fillna(9)
    out = out.sort_values(["_rank", "lane", "ticker", "layer"]).drop(columns=["_rank"])
    return out


def layer_card_html(layer: str, state: str, score: str, note: str) -> str:
    kind = status_kind(state)
    return (
        f'<div class="layer-card layer-{kind}">'
        f'<div class="layer-head">'
        f'<span class="layer-name">{escape(layer)}</span>'
        f'<span class="layer-score">{escape(str(score))}</span>'
        f'</div>'
        f'<div class="layer-state">{escape(str(state))}</div>'
        f'<div class="layer-note">{escape(str(note))}</div>'
        f'</div>'
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PROFESSIONAL SCORECARD BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_perf_metrics(ledger: pd.DataFrame, market: pd.DataFrame) -> dict:
    """Calculate Sharpe, Sortino, Calmar, max drawdown, win rate, alpha vs SPY."""
    empty = {
        "n_closed": 0, "total_return": "N/A", "avg_return": "N/A",
        "win_rate": "N/A", "max_drawdown": "N/A",
        "sharpe": "N/A", "sortino": "N/A", "calmar": "N/A",
        "spy_20d": "N/A", "alpha": "N/A", "ir": "N/A",
        "pnl_series": pd.Series(dtype=float),
        "cumulative": pd.Series(dtype=float),
        "drawdown_series": pd.Series(dtype=float),
    }
    if ledger.empty or "pnl_pct" not in ledger.columns:
        return empty
    closed = ledger.copy()
    if "status" in ledger.columns:
        closed = ledger[ledger["status"].astype(str).str.upper().str.startswith("CLOSED")]
    pnl = pd.to_numeric(closed["pnl_pct"], errors="coerce").dropna()
    if pnl.empty:
        return empty
    n = len(pnl)
    mean_r = float(pnl.mean())
    std_r  = float(pnl.std()) if n > 1 else 1e-6
    neg    = pnl[pnl < 0]
    down_std = float(neg.std()) if len(neg) > 1 else 1e-6
    cumulative   = (1 + pnl.reset_index(drop=True)).cumprod()
    rolling_max  = cumulative.expanding().max()
    drawdown_ser = (cumulative - rolling_max) / (rolling_max + 1e-10)
    total_ret    = float(cumulative.iloc[-1] - 1)
    max_dd       = float(drawdown_ser.min())
    win_rate     = float((pnl > 0).mean())
    rf_pt        = 0.05 / 252
    scale        = n ** 0.5
    sharpe   = (mean_r - rf_pt) / (std_r + 1e-10) * scale
    sortino  = (mean_r - rf_pt) / (down_std + 1e-10) * scale
    calmar   = total_ret / (abs(max_dd) + 1e-10)
    spy_20d_val = None
    if not market.empty and "ticker" in market.columns and "ret_20d" in market.columns:
        spy_rows = market[market["ticker"].astype(str).str.upper() == "SPY"]
        if not spy_rows.empty:
            v = pd.to_numeric(spy_rows["ret_20d"].iloc[0], errors="coerce")
            if not pd.isna(v):
                spy_20d_val = float(v)
    # NOTE: alpha_val compares strategy CUMULATIVE return (all trades, any horizon)
    # against SPY's most recent 20-day return. These are different time windows.
    # Interpret as directional only — not a rigorous Jensen's alpha.
    # Per-trade alpha = mean_r - (spy_20d_val / 20) is a closer per-trade comparison.
    alpha_val = None
    alpha_per_trade = None
    ir_val    = None
    if spy_20d_val is not None:
        alpha_val = total_ret - spy_20d_val                      # cumulative vs 20d — rough
        alpha_per_trade = mean_r - (spy_20d_val / 20.0)          # per-trade avg vs SPY daily
        ir_val    = alpha_per_trade / (std_r + 1e-10)
    return {
        "n_closed":      n,
        "total_return":  f"{total_ret * 100:+.2f}%",
        "avg_return":    f"{mean_r * 100:+.2f}%",
        "win_rate":      f"{win_rate * 100:.1f}%",
        "max_drawdown":  f"{max_dd * 100:.2f}%",
        "sharpe":        f"{sharpe:.2f}",
        "sortino":       f"{sortino:.2f}",
        "calmar":        f"{calmar:.2f}",
        "spy_20d":       f"{spy_20d_val * 100:+.2f}%" if spy_20d_val is not None else "N/A",
        "alpha":         f"{alpha_val * 100:+.2f}% (total vs SPY-20d)" if alpha_val is not None else "N/A",
        "alpha_per_trade": f"{alpha_per_trade * 100:+.4f}%/trade" if alpha_per_trade is not None else "N/A",
        "ir":            f"{ir_val:.2f}" if ir_val is not None else "N/A",
        "pnl_series":    pnl.reset_index(drop=True),
        "cumulative":    cumulative,
        "drawdown_series": drawdown_ser,
    }


def build_macro_calendar() -> pd.DataFrame:
    """Return upcoming macro events for the next 90 days (hardcoded 2026 schedule)."""
    today = datetime.now().date()
    events_raw = [
        # FOMC meetings 2026
        ("FOMC Decision",        "2026-01-29", "HIGH", "Fed rate decision. IV spikes before; options decay after."),
        ("FOMC Decision",        "2026-03-19", "HIGH", "Fed rate decision. IV spikes before; options decay after."),
        ("FOMC Decision",        "2026-05-07", "HIGH", "Fed rate decision. IV spikes before; options decay after."),
        ("FOMC Decision",        "2026-06-18", "HIGH", "Fed rate decision. IV spikes before; options decay after."),
        ("FOMC Decision",        "2026-07-30", "HIGH", "Fed rate decision. IV spikes before; options decay after."),
        ("FOMC Decision",        "2026-09-17", "HIGH", "Fed rate decision. IV spikes before; options decay after."),
        ("FOMC Decision",        "2026-11-05", "HIGH", "Fed rate decision. IV spikes before; options decay after."),
        ("FOMC Decision",        "2026-12-17", "HIGH", "Fed rate decision. IV spikes before; options decay after."),
        # CPI releases (approx 2nd week each month)
        ("CPI Inflation",        "2026-01-14", "HIGH", "CPI print. Rates and growth names react hard."),
        ("CPI Inflation",        "2026-02-11", "HIGH", "CPI print. Rates and growth names react hard."),
        ("CPI Inflation",        "2026-03-11", "HIGH", "CPI print."),
        ("CPI Inflation",        "2026-04-10", "HIGH", "CPI print."),
        ("CPI Inflation",        "2026-05-13", "HIGH", "CPI print."),
        ("CPI Inflation",        "2026-06-10", "HIGH", "CPI print."),
        ("CPI Inflation",        "2026-07-14", "HIGH", "CPI print."),
        ("CPI Inflation",        "2026-08-12", "HIGH", "CPI print."),
        # Non-Farm Payrolls (first Friday each month)
        ("Non-Farm Payrolls",    "2026-01-09", "MEDIUM", "Jobs data. Surprised NFP moves rates and dollar."),
        ("Non-Farm Payrolls",    "2026-02-06", "MEDIUM", "Jobs data."),
        ("Non-Farm Payrolls",    "2026-03-06", "MEDIUM", "Jobs data."),
        ("Non-Farm Payrolls",    "2026-04-03", "MEDIUM", "Jobs data."),
        ("Non-Farm Payrolls",    "2026-05-01", "MEDIUM", "Jobs data."),
        ("Non-Farm Payrolls",    "2026-06-05", "MEDIUM", "Jobs data."),
        ("Non-Farm Payrolls",    "2026-07-02", "MEDIUM", "Jobs data."),
        ("Non-Farm Payrolls",    "2026-08-07", "MEDIUM", "Jobs data."),
        # PCE (last business day of month)
        ("PCE Inflation",        "2026-01-30", "MEDIUM", "Fed's preferred inflation gauge."),
        ("PCE Inflation",        "2026-02-27", "MEDIUM", "Fed's preferred inflation gauge."),
        ("PCE Inflation",        "2026-03-27", "MEDIUM", "Fed's preferred inflation gauge."),
        ("PCE Inflation",        "2026-04-29", "MEDIUM", "Fed's preferred inflation gauge."),
        ("PCE Inflation",        "2026-05-29", "MEDIUM", "Fed's preferred inflation gauge."),
        ("PCE Inflation",        "2026-06-26", "MEDIUM", "Fed's preferred inflation gauge."),
        ("PCE Inflation",        "2026-07-31", "MEDIUM", "Fed's preferred inflation gauge."),
        # Earnings seasons
        ("Earnings Season Start","2026-01-13", "MEDIUM", "Q4 2025 earnings begin. Banks report first."),
        ("Earnings Season Start","2026-04-14", "MEDIUM", "Q1 2026 earnings begin."),
        ("Earnings Season Start","2026-07-14", "MEDIUM", "Q2 2026 earnings begin."),
        ("Earnings Season Start","2026-10-13", "MEDIUM", "Q3 2026 earnings begin."),
    ]
    rows = []
    for event, date_str, impact, what_to_do in events_raw:
        try:
            evt_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except Exception:
            continue
        days = (evt_date - today).days
        if days < -7 or days > 90:
            continue
        status = "RISK" if days <= 5 else ("WARN" if days <= 14 else "OK")
        rows.append({
            "status":        status,
            "event":         event,
            "date":          date_str,
            "days_to_event": days,
            "impact":        impact,
            "what_to_do":    what_to_do,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("days_to_event").reset_index(drop=True)


def build_atr_sizing(technicals: pd.DataFrame, exposure: pd.DataFrame,
                     account_size: float = 100_000.0,
                     risk_pct: float = 0.01) -> pd.DataFrame:
    """ATR-based position sizing: max_weight = risk_pct / ATR14_pct, capped at 15%."""
    if technicals.empty or "ticker" not in technicals.columns:
        return pd.DataFrame()
    tech = technicals[["ticker"]].copy()
    if "atr14_pct" in technicals.columns:
        tech["atr14_pct"] = pd.to_numeric(technicals["atr14_pct"], errors="coerce")
    else:
        return pd.DataFrame()
    tech = tech.dropna(subset=["atr14_pct"])
    tech["atr14_pct_num"] = tech["atr14_pct"].clip(lower=0.002)
    tech["atr_suggested_weight"] = (risk_pct / tech["atr14_pct_num"]).clip(upper=0.15)
    tech["atr_dollar_risk_per_1pct"] = account_size * risk_pct
    if not exposure.empty and "ticker" in exposure.columns and "effective_weight" in exposure.columns:
        exp = exposure[["ticker", "effective_weight"]].copy()
        exp["effective_weight"] = pd.to_numeric(exp["effective_weight"], errors="coerce")
        tech = tech.merge(exp, on="ticker", how="left")
    else:
        tech["effective_weight"] = float("nan")
    tech["status"] = "OK"
    if "effective_weight" in tech.columns:
        eff = pd.to_numeric(tech["effective_weight"], errors="coerce")
        sug = tech["atr_suggested_weight"]
        tech["status"] = "OK"
        tech.loc[eff > sug * 1.5, "status"] = "OVERSIZED"
        tech.loc[eff.isna(), "status"] = "NO_DATA"
    cols = ["status", "ticker", "atr14_pct", "atr_suggested_weight", "effective_weight"]
    cols = [c for c in cols if c in tech.columns]
    out = tech[cols].copy()
    for c in ["atr14_pct", "atr_suggested_weight", "effective_weight"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").map(
                lambda x: "" if pd.isna(x) else f"{x * 100:.2f}%"
            )
    return out.reset_index(drop=True)


def build_circuit_breaker(ledger: pd.DataFrame) -> dict:
    """
    Monitors for risk rules that should halt paper trading.
    Rules:
      - Consecutive losing trades: WARN at 3, STOP at 5
      - Monthly drawdown (sum of pnl_pct this calendar month): WARN at -10%, STOP at -15%
    Returns dict:
      status: "STOP" | "WARN" | "OK" | "NO_DATA"
      consecutive_losses: int
      monthly_pnl_pct: float or None
      monthly_pnl_str: str
      consecutive_status: "STOP"|"WARN"|"OK"
      monthly_status: "STOP"|"WARN"|"OK"
      message: str (human-readable action to take)
      rules: list of dicts with keys: status, rule, value, threshold, action
    """
    empty = {
        "status": "NO_DATA",
        "consecutive_losses": 0,
        "monthly_pnl_pct": None,
        "monthly_pnl_str": "N/A",
        "consecutive_status": "OK",
        "monthly_status": "OK",
        "message": "No closed trades yet. No circuit breaker triggers.",
        "rules": [],
    }
    if ledger.empty or "pnl_pct" not in ledger.columns:
        return empty
    closed = ledger[ledger["status"].astype(str).str.upper().str.startswith("CLOSED")].copy()
    if closed.empty:
        return empty

    pnl = pd.to_numeric(closed["pnl_pct"], errors="coerce").fillna(0)

    # Sort by exit_date if available, else entry_date, else original order
    for date_col in ["exit_date", "entry_date"]:
        if date_col in closed.columns:
            closed = closed.copy()
            closed["_sort_date"] = pd.to_datetime(closed[date_col], errors="coerce")
            closed = closed.sort_values("_sort_date", na_position="last")
            pnl = pd.to_numeric(closed["pnl_pct"], errors="coerce").fillna(0)
            break

    # Consecutive losses (count from end)
    consecutive = 0
    for v in reversed(pnl.tolist()):
        if v < 0:
            consecutive += 1
        else:
            break

    # Monthly P&L (current calendar month)
    monthly_pnl = None
    if "exit_date" in closed.columns or "entry_date" in closed.columns:
        date_col = "exit_date" if "exit_date" in closed.columns else "entry_date"
        closed["_date"] = pd.to_datetime(closed[date_col], errors="coerce")
        today = pd.Timestamp.today()
        this_month = closed[
            (closed["_date"].dt.year == today.year) &
            (closed["_date"].dt.month == today.month)
        ]
        if not this_month.empty:
            monthly_pnl = float(pd.to_numeric(this_month["pnl_pct"], errors="coerce").fillna(0).sum())

    # Statuses
    consec_status = "STOP" if consecutive >= 5 else ("WARN" if consecutive >= 3 else "OK")
    monthly_status = "OK"
    monthly_pnl_str = "N/A"
    if monthly_pnl is not None:
        monthly_pnl_str = f"{monthly_pnl * 100:.2f}%"
        monthly_status = "STOP" if monthly_pnl <= -0.15 else ("WARN" if monthly_pnl <= -0.10 else "OK")

    overall = "STOP" if "STOP" in [consec_status, monthly_status] else (
        "WARN" if "WARN" in [consec_status, monthly_status] else "OK"
    )

    # Message
    if overall == "STOP":
        message = "🛑 TRADING HALTED — Review all open paper positions before adding any new risk."
    elif overall == "WARN":
        message = "⚠ WARNING — Reduce position sizes by 50% until streak or drawdown clears."
    else:
        message = "✅ All clear — Circuit breaker is not triggered. Normal sizing rules apply."

    rules = [
        {
            "status": consec_status,
            "rule": "Consecutive Losses",
            "value": str(consecutive),
            "threshold": "WARN ≥3 / STOP ≥5",
            "action": "STOP new paper" if consec_status == "STOP" else ("Half-size only" if consec_status == "WARN" else "Normal"),
        },
        {
            "status": monthly_status,
            "rule": "Monthly Drawdown",
            "value": monthly_pnl_str,
            "threshold": "WARN ≤ -10% / STOP ≤ -15%",
            "action": "STOP new paper" if monthly_status == "STOP" else ("Half-size only" if monthly_status == "WARN" else "Normal"),
        },
    ]
    return {
        "status": overall,
        "consecutive_losses": consecutive,
        "monthly_pnl_pct": monthly_pnl,
        "monthly_pnl_str": monthly_pnl_str,
        "consecutive_status": consec_status,
        "monthly_status": monthly_status,
        "message": message,
        "rules": rules,
    }


def build_hold_time_analysis(ledger: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes average holding period for winners vs losers.
    Needs: entry_date, exit_date (or holding_days), pnl_pct columns.
    Returns DataFrame: outcome, count, avg_days, median_days, min_days, max_days
    Plus a 'status' column: "RISK" if losers held longer than winners (disposition effect).
    """
    if ledger.empty or "pnl_pct" not in ledger.columns:
        return pd.DataFrame()

    closed = ledger[ledger["status"].astype(str).str.upper().str.startswith("CLOSED")].copy()
    if closed.empty:
        return pd.DataFrame()

    pnl = pd.to_numeric(closed["pnl_pct"], errors="coerce")

    # Compute holding days
    if "holding_days" in closed.columns:
        days = pd.to_numeric(closed["holding_days"], errors="coerce")
    elif "entry_date" in closed.columns and "exit_date" in closed.columns:
        entry = pd.to_datetime(closed["entry_date"], errors="coerce")
        exit_ = pd.to_datetime(closed["exit_date"], errors="coerce")
        days = (exit_ - entry).dt.days
    else:
        return pd.DataFrame()

    closed["_pnl"] = pnl
    closed["_days"] = days
    valid = closed.dropna(subset=["_pnl", "_days"])
    if valid.empty:
        return pd.DataFrame()

    winners = valid[valid["_pnl"] > 0]["_days"]
    losers  = valid[valid["_pnl"] <= 0]["_days"]
    all_    = valid["_days"]

    rows = []
    for label, series in [("Winners", winners), ("Losers", losers), ("All Trades", all_)]:
        if series.empty:
            continue
        rows.append({
            "outcome": label,
            "count": len(series),
            "avg_days": round(series.mean(), 1),
            "median_days": round(series.median(), 1),
            "min_days": int(series.min()),
            "max_days": int(series.max()),
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Detect disposition effect: losers held longer than winners
    w_avg = df.loc[df["outcome"] == "Winners", "avg_days"].values
    l_avg = df.loc[df["outcome"] == "Losers",  "avg_days"].values
    disposition_effect = (len(w_avg) > 0 and len(l_avg) > 0 and l_avg[0] > w_avg[0])
    df["status"] = "OK"
    if disposition_effect:
        df.loc[df["outcome"] == "Losers", "status"] = "RISK"
    return df.reset_index(drop=True)


def build_ev_kelly(pm: dict) -> dict:
    """
    Computes Expected Value per trade and Kelly Criterion optimal fraction.
    Uses pre-computed perf metrics from build_perf_metrics().
    EV = win_rate * avg_win - (1 - win_rate) * avg_loss
    Kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss)
    Returns dict: ev_pct, kelly_fraction, half_kelly, ev_status, kelly_status, ev_str, kelly_str, half_kelly_str
    """
    empty = {
        "ev_pct": None, "kelly_fraction": None, "half_kelly": None,
        "ev_status": "NO_DATA", "kelly_status": "NO_DATA",
        "ev_str": "N/A", "kelly_str": "N/A", "half_kelly_str": "N/A",
        "interpretation": "Need closed trades to compute EV and Kelly.",
    }
    if pm.get("n_closed", 0) < 2:
        return empty

    pnl_series = pm.get("pnl_series")
    if pnl_series is None or (hasattr(pnl_series, "empty") and pnl_series.empty):
        return empty

    pnl = pnl_series
    winners = pnl[pnl > 0]
    losers  = pnl[pnl <= 0]
    if winners.empty or losers.empty:
        return empty

    win_rate  = len(winners) / len(pnl)
    avg_win   = float(winners.mean())
    avg_loss  = float(abs(losers.mean()))

    ev = win_rate * avg_win - (1 - win_rate) * avg_loss
    # Guard both avg_win > 0 AND avg_loss > 0 before Kelly division
    kelly = win_rate - (1 - win_rate) / (avg_win / avg_loss) if (avg_win > 0 and avg_loss > 1e-10) else 0
    kelly = max(kelly, 0)  # Kelly can't be negative (means don't bet)
    half_kelly = kelly / 2

    ev_status     = "supportive" if ev > 0 else "risk"
    kelly_status  = "supportive" if kelly > 0.05 else ("cyan" if kelly > 0 else "risk")

    return {
        "ev_pct":        ev,
        "kelly_fraction": kelly,
        "half_kelly":    half_kelly,
        "ev_status":     ev_status,
        "kelly_status":  kelly_status,
        "ev_str":        f"{ev * 100:.3f}% per trade",
        "kelly_str":     f"{kelly * 100:.1f}% of account",
        "half_kelly_str":f"{half_kelly * 100:.1f}% of account (recommended)",
        "interpretation": (
            f"EV {ev*100:+.3f}% per trade. "
            f"Full Kelly = {kelly*100:.1f}%; use Half-Kelly ({half_kelly*100:.1f}%) to reduce variance. "
            + ("Positive EV — edge exists." if ev > 0 else "Negative EV — strategy losing money on average.")
        ),
    }


def build_rolling_performance(ledger: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rolling performance across 1M / 3M / 6M trailing windows.
    Needs: pnl_pct + exit_date (or entry_date) columns.
    Returns DataFrame: period, n_trades, total_return_pct, win_rate_pct, avg_return_pct, status
    """
    if ledger.empty or "pnl_pct" not in ledger.columns:
        return pd.DataFrame()

    closed = ledger[ledger["status"].astype(str).str.upper().str.startswith("CLOSED")].copy()
    if closed.empty:
        return pd.DataFrame()

    date_col = "exit_date" if "exit_date" in closed.columns else "entry_date" if "entry_date" in closed.columns else None
    if date_col is None:
        return pd.DataFrame()

    closed["_date"] = pd.to_datetime(closed[date_col], errors="coerce")
    closed["_pnl"]  = pd.to_numeric(closed["pnl_pct"], errors="coerce")
    closed = closed.dropna(subset=["_date", "_pnl"]).sort_values("_date")
    if closed.empty:
        return pd.DataFrame()

    today = pd.Timestamp.today()
    rows = []
    for label, months in [("Last 1 Month", 1), ("Last 3 Months", 3), ("Last 6 Months", 6)]:
        cutoff = today - pd.DateOffset(months=months)
        window = closed[closed["_date"] >= cutoff]
        if window.empty:
            rows.append({
                "status": "NO_DATA", "period": label, "n_trades": 0,
                "total_return_pct": "—", "win_rate_pct": "—", "avg_return_pct": "—",
            })
            continue
        pnl = window["_pnl"]
        total_ret = float((1 + pnl).prod() - 1)
        win_rate  = float((pnl > 0).mean())
        avg_ret   = float(pnl.mean())
        status    = "supportive" if total_ret > 0 else "risk"
        rows.append({
            "status": status,
            "period": label,
            "n_trades": len(pnl),
            "total_return_pct": f"{total_ret * 100:.2f}%",
            "win_rate_pct":     f"{win_rate * 100:.1f}%",
            "avg_return_pct":   f"{avg_ret * 100:.3f}%",
        })
    return pd.DataFrame(rows)


def build_regime_breakdown(ledger: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """
    Splits performance by SPY market regime.
    Regimes: BULL = SPY trend_state contains 'BULL' or 'UP', BEAR = 'BEAR' or 'DOWN', CHOP = else.
    Since paper trades don't store the regime at entry time, we approximate:
    if market snapshot is available, use current SPY trend_state as label and note it's approximate.
    Falls back to splitting by trade outcome date vs SPY ret_20d sign.
    Returns DataFrame: status, regime, n_trades, win_rate_pct, avg_return_pct, total_return_pct, note
    """
    if ledger.empty or "pnl_pct" not in ledger.columns:
        return pd.DataFrame()

    closed = ledger[ledger["status"].astype(str).str.upper().str.startswith("CLOSED")].copy()
    if closed.empty:
        return pd.DataFrame()

    closed["_pnl"] = pd.to_numeric(closed["pnl_pct"], errors="coerce")
    closed = closed.dropna(subset=["_pnl"])
    if closed.empty:
        return pd.DataFrame()

    # Try to get SPY regime info from market snapshot
    spy_trend = "UNKNOWN"
    if not market.empty and "ticker" in market.columns and "trend_state" in market.columns:
        spy_row = market[market["ticker"].astype(str).str.upper() == "SPY"]
        if not spy_row.empty:
            spy_trend = str(spy_row.iloc[0]["trend_state"]).upper()

    # Classify current regime
    if any(x in spy_trend for x in ["BULL", "UP", "ABOVE"]):
        current_regime = "BULL"
    elif any(x in spy_trend for x in ["BEAR", "DOWN", "BELOW"]):
        current_regime = "BEAR"
    else:
        current_regime = "CHOP"

    # Since we can't retroactively know regime per trade, show:
    # 1. All trades labeled under current regime as "Current Regime"
    # 2. A note explaining limitation
    pnl = closed["_pnl"]
    total_ret = float((1 + pnl).prod() - 1)
    win_rate  = float((pnl > 0).mean())
    avg_ret   = float(pnl.mean())

    rows = [
        {
            "status": "cyan",
            "regime": f"Current SPY Regime: {current_regime}",
            "n_trades": len(pnl),
            "win_rate_pct": f"{win_rate * 100:.1f}%",
            "avg_return_pct": f"{avg_ret * 100:.3f}%",
            "total_return_pct": f"{total_ret * 100:.2f}%",
            "note": f"SPY trend_state = {spy_trend}. All closed trades shown under current regime (retroactive regime data not stored).",
        },
    ]

    # Add win/loss breakdown by quarters if enough data
    if len(pnl) >= 8:
        pnl_sorted = pnl.reset_index(drop=True)
        mid = len(pnl_sorted) // 2
        first_half = pnl_sorted.iloc[:mid]
        second_half = pnl_sorted.iloc[mid:]
        for label, half in [("First Half Of Sample (older)", first_half), ("Second Half Of Sample (recent)", second_half)]:
            t = float((1 + half).prod() - 1)
            w = float((half > 0).mean())
            a = float(half.mean())
            rows.append({
                "status": "supportive" if t > 0 else "risk",
                "regime": label,
                "n_trades": len(half),
                "win_rate_pct": f"{w * 100:.1f}%",
                "avg_return_pct": f"{a * 100:.3f}%",
                "total_return_pct": f"{t * 100:.2f}%",
                "note": "Trend comparison: improving = second half better than first half.",
            })

    return pd.DataFrame(rows)


def build_sizing_compliance(ledger: pd.DataFrame, exposure: pd.DataFrame, technicals: pd.DataFrame) -> dict:
    """
    Checks how often actual paper positions follow ATR-based sizing rules.
    Compares effective_weight in exposure vs atr_suggested_weight from build_atr_sizing.
    Returns dict:
      compliance_rate: float 0-1
      compliance_str: str
      oversized_tickers: list of str
      avg_deviation_pct: float
      status: "OK"|"WARN"|"RISK"|"NO_DATA"
      detail_df: DataFrame with ticker, atr_max, actual, deviation, status
    """
    empty = {
        "compliance_rate": None, "compliance_str": "N/A",
        "oversized_tickers": [], "avg_deviation_pct": None,
        "status": "NO_DATA", "detail_df": pd.DataFrame(),
    }
    if exposure.empty or technicals.empty:
        return empty
    if "ticker" not in exposure.columns or "effective_weight" not in exposure.columns:
        return empty
    if "ticker" not in technicals.columns or "atr14_pct" not in technicals.columns:
        return empty

    exp = exposure[["ticker", "effective_weight"]].copy()
    exp["effective_weight"] = pd.to_numeric(exp["effective_weight"], errors="coerce")

    tech = technicals[["ticker", "atr14_pct"]].copy()
    tech["atr14_pct"] = pd.to_numeric(tech["atr14_pct"], errors="coerce")
    tech["atr_max"] = (0.01 / tech["atr14_pct"].clip(lower=0.002)).clip(upper=0.15)

    # Use left join so tickers in exposure but missing from technicals are counted as unknown
    merged = exp.merge(tech[["ticker", "atr_max"]], on="ticker", how="left")
    merged = merged.dropna(subset=["effective_weight"])  # must have weight; atr_max may be NaN
    if merged.empty:
        return empty

    # Rows with no ATR data: mark as NO_ATR_DATA (unknown compliance, not assumed compliant)
    no_atr_mask = merged["atr_max"].isna()
    merged.loc[no_atr_mask, "atr_max"] = float("nan")
    merged["deviation"] = (merged["effective_weight"] - merged["atr_max"]).fillna(0.0)
    merged["compliant"] = merged.apply(
        lambda r: True if pd.isna(r["atr_max"]) else (r["effective_weight"] <= r["atr_max"] * 1.05),
        axis=1,
    )
    merged["status"] = merged.apply(
        lambda r: ("NO_ATR_DATA" if pd.isna(r["atr_max"])
                   else ("OK" if r["compliant"] else ("RISK" if r["deviation"] > 0.05 else "WARN"))),
        axis=1,
    )
    # Compliance rate only over rows with ATR data (can't penalise unknown)
    has_atr = merged[~no_atr_mask]
    compliance_rate = float(has_atr["compliant"].mean()) if not has_atr.empty else 1.0
    n_missing_atr = int(no_atr_mask.sum())
    oversized = merged[~merged["compliant"]]["ticker"].astype(str).tolist()
    avg_dev = float(merged["deviation"].mean()) * 100

    status = "OK" if compliance_rate >= 0.9 else ("WARN" if compliance_rate >= 0.7 else "RISK")

    detail_df = merged[["status", "ticker", "atr_max", "effective_weight", "deviation"]].copy()
    for c in ["atr_max", "effective_weight", "deviation"]:
        detail_df[c] = detail_df[c].map(lambda x: f"{x * 100:.2f}%")

    return {
        "compliance_rate": compliance_rate,
        "compliance_str": f"{compliance_rate * 100:.1f}% (of {len(has_atr)} w/ ATR data)",
        "oversized_tickers": oversized,
        "avg_deviation_pct": avg_dev,
        "n_missing_atr": n_missing_atr,
        "status": status,
        "detail_df": detail_df.reset_index(drop=True),
    }


def build_entry_timing(ledger: pd.DataFrame) -> pd.DataFrame:
    """
    Analyzes what day-of-week entries were made and win rate by day.
    Needs: entry_date, pnl_pct columns.
    Returns DataFrame: status, day_of_week, n_trades, win_rate_pct, avg_return_pct, note
    """
    if ledger.empty or "pnl_pct" not in ledger.columns or "entry_date" not in ledger.columns:
        return pd.DataFrame()

    closed = ledger[ledger["status"].astype(str).str.upper().str.startswith("CLOSED")].copy()
    if closed.empty:
        return pd.DataFrame()

    closed["_pnl"]  = pd.to_numeric(closed["pnl_pct"], errors="coerce")
    closed["_date"] = pd.to_datetime(closed["entry_date"], errors="coerce")
    closed = closed.dropna(subset=["_pnl", "_date"])
    if closed.empty:
        return pd.DataFrame()

    closed["_dow"] = closed["_date"].dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    closed["_dow_num"] = closed["_date"].dt.dayofweek

    rows = []
    best_win_rate = -1
    best_day = ""
    for dow_num, dow_name in enumerate(day_order):
        subset = closed[closed["_dow_num"] == dow_num]
        if subset.empty:
            continue
        pnl = subset["_pnl"]
        win_rate = float((pnl > 0).mean())
        avg_ret  = float(pnl.mean())
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_day = dow_name
        rows.append({
            "day_of_week": dow_name,
            "n_trades": len(pnl),
            "win_rate_pct": f"{win_rate * 100:.1f}%",
            "avg_return_pct": f"{avg_ret * 100:.3f}%",
            "_win_rate_num": win_rate,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["status"] = df["day_of_week"].apply(
        lambda d: "supportive" if d == best_day else "OK"
    )
    df["note"] = df["day_of_week"].apply(
        lambda d: "Best win-rate day" if d == best_day else ""
    )
    return df[["status", "day_of_week", "n_trades", "win_rate_pct", "avg_return_pct", "note"]].reset_index(drop=True)


def build_watchlist_aging(focus: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    """
    Shows how long each focus-list ticker has been watched without a paper trade action.
    If 'added_date' exists in focus, use it. Otherwise estimate from ledger entry dates.
    Returns DataFrame: status, ticker, days_watched, has_paper_trade, last_action, action_needed
    """
    if focus.empty or "ticker" not in focus.columns:
        return pd.DataFrame()

    today = pd.Timestamp.today()
    tickers = focus["ticker"].astype(str).tolist()

    # Tickers with paper trades
    paper_tickers = set()
    last_action_map = {}
    if not ledger.empty and "ticker" in ledger.columns:
        for t in tickers:
            rows = ledger[ledger["ticker"].astype(str).str.upper() == t.upper()]
            if not rows.empty:
                paper_tickers.add(t.upper())
                date_col = "entry_date" if "entry_date" in rows.columns else None
                if date_col:
                    last_date = pd.to_datetime(rows[date_col], errors="coerce").max()
                    last_action_map[t.upper()] = last_date

    # Estimate days watched
    days_watched_map = {}
    if "added_date" in focus.columns:
        for _, row in focus.iterrows():
            t = str(row["ticker"]).upper()
            dt = pd.to_datetime(row.get("added_date"), errors="coerce")
            if pd.notna(dt):
                days_watched_map[t] = (today - dt).days

    result = []
    for _, row in focus.iterrows():
        t = str(row.get("ticker", "")).upper()
        has_paper = t in paper_tickers
        days = days_watched_map.get(t, None)
        last_action = last_action_map.get(t, None)
        last_action_str = last_action.strftime("%Y-%m-%d") if last_action and pd.notna(last_action) else "—"
        days_str = str(days) if days is not None else "—"

        if days is not None and days > 45 and not has_paper:
            status = "WARN"
            action_needed = "Review or remove — 45+ days with no paper trade"
        elif days is not None and days > 90 and not has_paper:
            status = "RISK"
            action_needed = "Stale — consider removing from watch list"
        elif has_paper:
            status = "OK"
            action_needed = "Paper trade exists"
        else:
            status = "cyan"
            action_needed = "Still building evidence"

        result.append({
            "status": status,
            "ticker": t,
            "days_watched": days_str,
            "has_paper_trade": "Yes" if has_paper else "No",
            "last_action": last_action_str,
            "action_needed": action_needed,
        })

    return pd.DataFrame(result)


# ── 8 new builder functions ───────────────────────────────────────────────────

def build_beta_adjusted_exposure(exposure: pd.DataFrame, technicals: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    if exposure.empty or "ticker" not in exposure.columns or "effective_weight" not in exposure.columns:
        return pd.DataFrame()
    df = exposure[["ticker", "sector", "effective_weight"]].copy() if "sector" in exposure.columns else exposure[["ticker", "effective_weight"]].copy()
    if "sector" not in df.columns:
        df["sector"] = "—"
    df["effective_weight"] = pd.to_numeric(df["effective_weight"], errors="coerce").fillna(0)
    df["ticker_upper"] = df["ticker"].astype(str).str.upper()

    # Build beta map: technicals first, then market
    beta_map = {}
    if not technicals.empty and "ticker" in technicals.columns and "beta" in technicals.columns:
        for _, r in technicals.iterrows():
            t = str(r["ticker"]).upper()
            b = pd.to_numeric(r.get("beta"), errors="coerce")
            if pd.notna(b) and b > 0:
                beta_map[t] = float(b)
    if not market.empty and "ticker" in market.columns and "beta" in market.columns:
        for _, r in market.iterrows():
            t = str(r["ticker"]).upper()
            if t not in beta_map:
                b = pd.to_numeric(r.get("beta"), errors="coerce")
                if pd.notna(b) and b > 0:
                    beta_map[t] = float(b)

    df["_has_real_beta"] = df["ticker_upper"].map(beta_map).notna()
    df["beta"] = df["ticker_upper"].map(beta_map).fillna(1.0)
    df["beta_adj_weight"] = df["effective_weight"] * df["beta"]
    n_default_beta = int((~df["_has_real_beta"]).sum())

    def _row_status(row):
        w = row["beta_adj_weight"]
        if w > 0.25: return "RISK"
        if w > 0.15: return "WARN"
        return "OK"

    df["status"] = df.apply(_row_status, axis=1)
    df["note"] = df.apply(
        lambda r: (
            f"Beta {r['beta']:.1f}x amplifies nominal {r['effective_weight']*100:.1f}% to {r['beta_adj_weight']*100:.1f}% market exposure"
            if r["_has_real_beta"]
            else f"Beta=1.0 (no data — actual beta unknown; exposure estimate may be off)"
        ), axis=1
    )
    if n_default_beta > 0:
        df.loc[~df["_has_real_beta"], "status"] = df.loc[~df["_has_real_beta"], "status"].replace("OK", "WARN")

    # Sort by beta_adj_weight desc
    df = df.sort_values("beta_adj_weight", ascending=False)

    # Format display
    out = df[["status", "ticker", "sector", "effective_weight", "beta", "beta_adj_weight", "note"]].copy()
    out["effective_weight"] = out["effective_weight"].map(lambda x: f"{x*100:.2f}%")
    out["beta"] = out["beta"].map(lambda x: f"{x:.2f}")
    out["beta_adj_weight"] = out["beta_adj_weight"].map(lambda x: f"{x*100:.2f}%")
    return out.reset_index(drop=True)


def build_earnings_window(earnings: pd.DataFrame, exposure: pd.DataFrame, n_days: int = 14) -> pd.DataFrame:
    if earnings.empty:
        return pd.DataFrame()

    today = pd.Timestamp.today().normalize()

    # Normalize earnings df
    e = earnings.copy()
    date_col = next((c for c in ["earnings_date", "event_date", "date"] if c in e.columns), None)
    if date_col is None:
        return pd.DataFrame()
    e["_date"] = pd.to_datetime(e[date_col], errors="coerce")
    e = e.dropna(subset=["_date"])
    e["_days"] = (e["_date"] - today).dt.days
    # Only upcoming (0 to n_days)
    e = e[(e["_days"] >= 0) & (e["_days"] <= n_days)].copy()
    if e.empty:
        return pd.DataFrame()

    ticker_col = next((c for c in ["ticker", "symbol"] if c in e.columns), None)
    if ticker_col is None:
        return pd.DataFrame()
    e["_ticker"] = e[ticker_col].astype(str).str.upper()

    # Only tickers we have exposure in
    held_tickers = set()
    weight_map = {}
    if not exposure.empty and "ticker" in exposure.columns and "effective_weight" in exposure.columns:
        for _, r in exposure.iterrows():
            t = str(r["ticker"]).upper()
            held_tickers.add(t)
            weight_map[t] = pd.to_numeric(r.get("effective_weight", 0), errors="coerce") or 0.0

    rows = []
    for _, r in e.iterrows():
        t = r["_ticker"]
        if t not in held_tickers:
            continue
        days = int(r["_days"])
        status = "RISK" if days <= 7 else "WARN"
        weight = weight_map.get(t, 0.0)
        rows.append({
            "status": status,
            "ticker": t,
            "earnings_date": r["_date"].strftime("%Y-%m-%d"),
            "days_away": days,
            "current_weight": f"{weight*100:.1f}%",
            "action_required": "Reduce or exit before earnings" if days <= 7 else "Review thesis — earnings within 14 days",
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("days_away").reset_index(drop=True)


def build_discipline_rate(ledger: pd.DataFrame, pretrade: pd.DataFrame) -> dict:
    empty = {"compliance_rate": None, "compliance_str": "N/A", "total_trades": 0,
             "compliant_trades": 0, "non_compliant_tickers": [], "status": "NO_DATA", "detail_df": pd.DataFrame()}
    if ledger.empty or "ticker" not in ledger.columns:
        return empty

    closed = ledger[ledger["status"].astype(str).str.upper().str.startswith("CLOSED")].copy()
    if closed.empty:
        return empty

    check_cols = [c for c in ["news_check", "earnings_check", "liquidity_check", "spread_check", "thesis_check"] if not pretrade.empty and c in pretrade.columns]

    rows = []
    compliant = 0
    non_compliant = []

    for _, trade in closed.iterrows():
        t = str(trade.get("ticker", "")).upper()
        # Match pretrade row
        pt_row = pretrade[pretrade["ticker"].astype(str).str.upper() == t] if not pretrade.empty and "ticker" in pretrade.columns else pd.DataFrame()

        if pt_row.empty:
            all_pass = False
            checks_str = "No pretrade record"
        elif check_cols:
            check_vals = [str(pt_row.iloc[0].get(c, "")).upper() for c in check_cols]
            all_pass = all(v in {"YES", "PASS", "TRUE", "1", "DONE", "OK"} for v in check_vals)
            checks_str = " | ".join(f"{c.replace('_check','')}:{v}" for c, v in zip(check_cols, check_vals))
        elif "pretrade_status" in (pt_row.columns if not pt_row.empty else []):
            ps = str(pt_row.iloc[0].get("pretrade_status", "")).upper()
            all_pass = any(x in ps for x in ["PASS", "ALL_CLEAR", "OK"])
            checks_str = f"pretrade_status={ps}"
        else:
            all_pass = False
            checks_str = "No check data"

        entry_date = str(trade.get("entry_date", "—"))
        pnl = str(trade.get("pnl_pct", "—"))

        if all_pass:
            compliant += 1
            s = "OK"
        else:
            non_compliant.append(t)
            s = "WARN"

        rows.append({"status": s, "ticker": t, "entry_date": entry_date, "pnl_pct": pnl, "all_checks_passed": "Yes" if all_pass else "No", "checks_detail": checks_str})

    n = len(rows)
    rate = compliant / n if n > 0 else None
    overall_status = "OK" if rate is not None and rate >= 0.9 else ("WARN" if rate is not None and rate >= 0.7 else "RISK")

    return {
        "compliance_rate": rate,
        "compliance_str": f"{rate*100:.1f}%" if rate is not None else "N/A",
        "total_trades": n,
        "compliant_trades": compliant,
        "non_compliant_tickers": list(set(non_compliant)),
        "status": overall_status,
        "detail_df": pd.DataFrame(rows),
    }


def build_stop_compliance(ledger: pd.DataFrame) -> dict:
    empty = {"honored_rate": None, "honored_str": "N/A", "honored_count": 0,
             "violated_count": 0, "status": "NO_DATA", "large_loss_trades": [], "detail_df": pd.DataFrame()}
    if ledger.empty or "pnl_pct" not in ledger.columns:
        return empty

    closed = ledger[ledger["status"].astype(str).str.upper().str.startswith("CLOSED")].copy()
    if closed.empty:
        return empty

    closed["_pnl"] = pd.to_numeric(closed["pnl_pct"], errors="coerce")
    losers = closed[closed["_pnl"] < 0].copy()
    if losers.empty:
        return {"honored_rate": 1.0, "honored_str": "100.0%", "honored_count": 0,
                "violated_count": 0, "status": "OK", "large_loss_trades": [],
                "detail_df": pd.DataFrame([{"status": "OK", "note": "No losing trades yet"}])}

    rows = []
    honored = 0
    violated = 0
    large_losses = []

    has_stop = "stop_price" in closed.columns and "exit_price" in closed.columns and "entry_price" in closed.columns

    for _, r in losers.iterrows():
        t = str(r.get("ticker", "—"))
        pnl = float(r["_pnl"])
        entry_date = str(r.get("entry_date", "—"))

        if has_stop:
            stop = pd.to_numeric(r.get("stop_price"), errors="coerce")
            exit_p = pd.to_numeric(r.get("exit_price"), errors="coerce")
            entry_p = pd.to_numeric(r.get("entry_price"), errors="coerce")

            if pd.notna(stop) and pd.notna(exit_p) and pd.notna(entry_p) and entry_p > 0:
                expected_loss = (stop - entry_p) / entry_p
                # Honored if actual loss not worse than 150% of intended stop
                honor = pnl >= expected_loss * 1.5
                stop_str = f"stop={stop:.2f}, exit={exit_p:.2f}"
            else:
                # Fall back to large-loss heuristic
                honor = pnl >= -0.12  # 12% loss = stop probably wasn't honored
                stop_str = "No stop data"
        else:
            honor = pnl >= -0.12
            stop_str = "No stop data"

        if honor:
            honored += 1
            s = "OK"
        else:
            violated += 1
            s = "RISK"
            if pnl < -0.10:
                large_losses.append(t)

        rows.append({"status": s, "ticker": t, "entry_date": entry_date, "pnl_pct": f"{pnl*100:.2f}%", "stop_honored": "Yes" if honor else "No", "detail": stop_str})

    n = len(rows)
    rate = honored / n if n > 0 else 1.0
    overall = "OK" if rate >= 0.85 else ("WARN" if rate >= 0.70 else "RISK")

    return {
        "honored_rate": rate,
        "honored_str": f"{rate*100:.1f}%",
        "honored_count": honored,
        "violated_count": violated,
        "status": overall,
        "large_loss_trades": list(set(large_losses)),
        "detail_df": pd.DataFrame(rows),
    }


def build_thesis_quality(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty or "pnl_pct" not in ledger.columns:
        return pd.DataFrame()

    closed = ledger[ledger["status"].astype(str).str.upper().str.startswith("CLOSED")].copy()
    if closed.empty or "thesis" not in closed.columns:
        return pd.DataFrame()

    rows = []
    for _, r in closed.iterrows():
        t = str(r.get("ticker", "—"))
        thesis = str(r.get("thesis", "") or "").strip()
        pnl = pd.to_numeric(r.get("pnl_pct"), errors="coerce")

        words = thesis.split()
        wc = len(words)
        thesis_lower = thesis.lower()

        score = 0
        if wc >= 50: score += 2
        if wc >= 100: score += 2
        has_target = any(x in thesis_lower for x in ["$", "% target", "price target", "pt ", "objective"])
        if has_target: score += 2
        has_stop = any(x in thesis_lower for x in ["stop", "cut", "exit if", "invalidate", "close if"])
        if has_stop: score += 2
        has_catalyst = any(x in thesis_lower for x in ["earnings", "catalyst", "fda", "event", "fomc", "cpi", "breakout", "reversal", "upgrade", "acquisition"])
        if has_catalyst: score += 2

        if score >= 7: s = "OK"
        elif score >= 4: s = "WARN"
        else: s = "RISK"

        preview = (thesis[:80] + "...") if len(thesis) > 80 else (thesis if thesis else "— no thesis —")

        rows.append({
            "status": s,
            "ticker": t,
            "pnl_pct": f"{pnl*100:.2f}%" if pd.notna(pnl) else "—",
            "score": f"{score}/10",
            "word_count": wc,
            "has_target": "Yes" if has_target else "No",
            "has_stop": "Yes" if has_stop else "No",
            "has_catalyst": "Yes" if has_catalyst else "No",
            "thesis_preview": preview,
        })

    return pd.DataFrame(rows).sort_values("score").reset_index(drop=True) if rows else pd.DataFrame()


def build_mae_analysis(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty or "pnl_pct" not in ledger.columns:
        return pd.DataFrame()

    closed = ledger[ledger["status"].astype(str).str.upper().str.startswith("CLOSED")].copy()
    if closed.empty:
        return pd.DataFrame()

    closed["_pnl"] = pd.to_numeric(closed["pnl_pct"], errors="coerce")
    closed = closed.dropna(subset=["_pnl"])
    if closed.empty:
        return pd.DataFrame()

    has_stop = "stop_price" in closed.columns and "entry_price" in closed.columns

    rows = []
    for _, r in closed.iterrows():
        t = str(r.get("ticker", "—"))
        pnl = float(r["_pnl"])

        if has_stop:
            stop = pd.to_numeric(r.get("stop_price"), errors="coerce")
            entry = pd.to_numeric(r.get("entry_price"), errors="coerce")
            if pd.notna(stop) and pd.notna(entry) and entry > 0:
                mae_proxy = (stop - entry) / entry  # negative for typical stop below entry
            else:
                mae_proxy = -0.08
        else:
            mae_proxy = -0.08

        outcome = "Winner" if pnl > 0 else "Loser"
        mae_abs = abs(mae_proxy)

        if outcome == "Winner":
            # Risk/reward: how much did we risk (mae_abs) vs what we gained (pnl)
            rr = pnl / mae_abs if mae_abs > 0 else 0
            if rr >= 2.0:
                s = "OK"
                interp = f"Good R/R: risked {mae_abs*100:.1f}%, gained {pnl*100:.2f}%"
            elif rr >= 1.0:
                s = "OK"
                interp = f"Acceptable R/R: {rr:.1f}:1"
            else:
                s = "WARN"
                interp = f"Low R/R: risked {mae_abs*100:.1f}% to gain {pnl*100:.2f}% — lucky or stop too wide"
        else:  # Loser
            if pnl < mae_proxy * 1.5:
                s = "RISK"
                interp = f"Stop blown: lost {pnl*100:.2f}% vs intended stop {mae_abs*100:.1f}%"
                rr = pnl / mae_abs if mae_abs > 0 else 0
            else:
                s = "OK"
                interp = f"Stop honored: lost {pnl*100:.2f}% within intended {mae_abs*100:.1f}%"
                rr = pnl / mae_abs if mae_abs > 0 else 0

        rows.append({
            "status": s,
            "ticker": t,
            "outcome": outcome,
            "pnl_pct": f"{pnl*100:.2f}%",
            "mae_proxy": f"{mae_abs*100:.1f}%",
            "rr_ratio": f"{abs(pnl/mae_abs):.2f}:1" if mae_abs > 0 else "—",
            "interpretation": interp,
        })

    return pd.DataFrame(rows).sort_values("status").reset_index(drop=True) if rows else pd.DataFrame()


def build_opportunity_cost(focus_df: pd.DataFrame, market: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    if focus_df.empty or market.empty:
        return pd.DataFrame()
    if "ticker" not in focus_df.columns:
        return pd.DataFrame()

    # Build set of traded tickers
    traded = set()
    if not ledger.empty and "ticker" in ledger.columns:
        traded = set(ledger["ticker"].astype(str).str.upper().tolist())

    # Market returns map
    ret_map = {}
    if "ret_20d" in market.columns:
        for _, r in market.iterrows():
            t = str(r["ticker"]).upper()
            v = pd.to_numeric(r.get("ret_20d"), errors="coerce")
            if pd.notna(v):
                ret_map[t] = float(v)

    rows = []
    for _, r in focus_df.iterrows():
        t = str(r.get("ticker", "")).upper()
        if not t:
            continue
        score = pd.to_numeric(r.get("focus_score"), errors="coerce")
        was_traded = t in traded
        ret = ret_map.get(t)
        ret_str = f"{ret*100:.2f}%" if ret is not None else "—"

        if not was_traded and ret is not None:
            if ret > 0.05:
                s = "WARN"
                note = f"Missed +{ret*100:.1f}% move — was this avoidable? Review decision to skip."
            elif ret < -0.05:
                s = "OK"
                note = f"Skipping this saved {ret*100:.1f}% loss — good discipline."
            else:
                s = "plain"
                note = "Flat while on watch — no major opportunity missed."
        elif was_traded:
            s = "OK"
            note = "Traded ✓"
        else:
            s = "plain"
            note = "No 20d return data available."

        rows.append({
            "status": s,
            "ticker": t,
            "focus_score": f"{score:.1f}" if pd.notna(score) else "—",
            "traded": "Yes" if was_traded else "No",
            "ret_20d": ret_str,
            "note": note,
        })

    return pd.DataFrame(rows).sort_values("status").reset_index(drop=True) if rows else pd.DataFrame()


def build_layer_signal_power(ledger: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty or master.empty or "ticker" not in ledger.columns or "ticker" not in master.columns:
        return pd.DataFrame()

    closed = ledger[ledger["status"].astype(str).str.upper().str.startswith("CLOSED")].copy()
    if closed.empty or "pnl_pct" not in closed.columns:
        return pd.DataFrame()

    closed["_pnl"] = pd.to_numeric(closed["pnl_pct"], errors="coerce")
    closed = closed.dropna(subset=["_pnl"])
    if len(closed) < 5:
        return pd.DataFrame()

    # Find score columns in master
    score_cols = [c for c in master.columns if c.startswith("L") and ("score" in c.lower() or c[1:].split("_")[0].isdigit())]
    # Also try L1_state etc — convert state to numeric score
    state_cols = [c for c in master.columns if c.startswith("L") and "state" in c.lower()]

    if not score_cols and not state_cols:
        return pd.DataFrame()

    # Merge
    m = master[["ticker"] + score_cols + state_cols].copy()
    # Left join: trades without master scores are retained with NaN scores (excluded per-column)
    merged = closed[["ticker", "_pnl"]].merge(m, on="ticker", how="left")
    if len(merged) < 3:
        return pd.DataFrame()

    rows = []
    for col in score_cols + state_cols:
        vals = pd.to_numeric(merged[col], errors="coerce")
        valid = merged[["_pnl"]].copy()
        valid["_score"] = vals
        valid = valid.dropna()
        if len(valid) < 3:
            continue
        try:
            corr = float(valid["_pnl"].corr(valid["_score"]))
        except Exception:
            continue
        if abs(corr) > 0.30:
            s = "OK"
            interp = f"Strong signal (r={corr:.2f}) — weight this layer more heavily in decisions"
        elif abs(corr) > 0.15:
            s = "WARN"
            interp = f"Moderate signal (r={corr:.2f}) — useful but not decisive alone"
        else:
            s = "plain"
            interp = f"Weak signal (r={corr:.2f}) — may not add predictive value"

        # Clean layer name
        layer_name = col.replace("_score", "").replace("_state", "").upper()
        rows.append({
            "status": s,
            "layer": layer_name,
            "correlation": f"{corr:.3f}",
            "n_samples": len(valid),
            "interpretation": interp,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("correlation", key=lambda x: x.abs() if hasattr(x, "abs") else x.map(lambda v: abs(float(v))), ascending=False).reset_index(drop=True)


# ── NEW BUILDER FUNCTIONS ──────────────────────────────────────────────────────

def build_pnl_calendar(ledger: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly P&L summary for calendar heatmap.
    Groups closed trades by year-month, computes total return and trade count.
    Returns DataFrame: year, month, month_label, total_return_pct, n_trades, win_rate_pct, status
    status: "risk" if total_return < -5%, "ok" if > +5%, "plain" else
    """
    if ledger.empty or "pnl_pct" not in ledger.columns:
        return pd.DataFrame()
    closed = ledger[ledger["status"].astype(str).str.upper().str.startswith("CLOSED")].copy()
    if closed.empty:
        return pd.DataFrame()
    date_col = next((c for c in ["exit_date","entry_date"] if c in closed.columns), None)
    if not date_col:
        return pd.DataFrame()
    closed["_date"] = pd.to_datetime(closed[date_col], errors="coerce")
    closed["_pnl"] = pd.to_numeric(closed["pnl_pct"], errors="coerce")
    closed = closed.dropna(subset=["_date","_pnl"])
    if closed.empty:
        return pd.DataFrame()
    closed["_ym"] = closed["_date"].dt.to_period("M")
    rows = []
    for period, grp in closed.groupby("_ym"):
        pnl = grp["_pnl"]
        total = float((1 + pnl).prod() - 1)
        wr = float((pnl > 0).mean())
        s = "risk" if total < -0.05 else ("ok" if total > 0.05 else "plain")
        rows.append({
            "status": s,
            "year": period.year,
            "month": period.month,
            "month_label": period.strftime("%b %Y"),
            "total_return_pct": round(total * 100, 2),
            "n_trades": len(pnl),
            "win_rate_pct": round(wr * 100, 1),
        })
    return pd.DataFrame(rows).sort_values(["year","month"]).reset_index(drop=True)


def build_catalyst_winrate(ledger: pd.DataFrame) -> pd.DataFrame:
    """
    Win rate by catalyst type, parsed from the thesis text of each closed trade.
    Catalyst categories (keyword-based, case-insensitive):
      Earnings: earnings, q1, q2, q3, q4, quarterly, report, eps, revenue guidance
      Technical: breakout, reversal, support, resistance, moving average, ma cross, rsi, macd, pattern
      Macro: fomc, fed, cpi, nfp, pcfe, rate, gdp, inflation, macro
      Insider: insider, form 4, 10b5, director, officer, bought, purchase
      Event: merger, acquisition, m&a, fda, approval, spinoff, ipo, catalyst
      Momentum: momentum, trend, relative strength, rs, 52-week, new high
    A trade can match multiple categories — count each category separately.
    Returns DataFrame: status, catalyst_type, n_trades, win_rate_pct, avg_return_pct, total_return_pct
    status: "ok" if win_rate > 55%, "plain" if 40-55%, "risk" if < 40%
    """
    if ledger.empty or "pnl_pct" not in ledger.columns or "thesis" not in ledger.columns:
        return pd.DataFrame()
    closed = ledger[ledger["status"].astype(str).str.upper().str.startswith("CLOSED")].copy()
    if closed.empty:
        return pd.DataFrame()
    closed["_pnl"] = pd.to_numeric(closed["pnl_pct"], errors="coerce")
    closed = closed.dropna(subset=["_pnl"])
    if closed.empty:
        return pd.DataFrame()

    CATS = {
        "Earnings":   ["earnings","q1","q2","q3","q4","quarterly","eps","revenue","guidance","beat","miss"],
        "Technical":  ["breakout","reversal","support","resistance","moving average","ma cross","rsi","macd","pattern","channel","wedge","flag"],
        "Macro":      ["fomc","fed ","cpi","nfp","pce","rate cut","rate hike","gdp","inflation","macro","powell"],
        "Insider":    ["insider","form 4","10b5","director bought","officer","purchased shares"],
        "Event":      ["merger","acquisition","m&a","fda","approval","spinoff","ipo","catalyst","deal","buyout"],
        "Momentum":   ["momentum","trend","relative strength"," rs ","52-week","new high","breakout","leading"],
    }
    buckets = {c: [] for c in CATS}
    for _, r in closed.iterrows():
        thesis = str(r.get("thesis","") or "").lower()
        pnl = float(r["_pnl"])
        for cat, kws in CATS.items():
            if any(kw in thesis for kw in kws):
                buckets[cat].append(pnl)

    rows = []
    for cat, pnls in buckets.items():
        if not pnls:
            continue
        arr = pd.Series(pnls)
        wr = float((arr > 0).mean())
        avg = float(arr.mean())
        total = float((1 + arr).prod() - 1)
        s = "ok" if wr > 0.55 else ("risk" if wr < 0.40 else "plain")
        rows.append({
            "status": s,
            "catalyst_type": cat,
            "n_trades": len(arr),
            "win_rate_pct": f"{wr*100:.1f}%",
            "avg_return_pct": f"{avg*100:.3f}%",
            "total_return_pct": f"{total*100:.2f}%",
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("win_rate_pct", ascending=False).reset_index(drop=True)


def build_var(pm: dict) -> dict:
    """
    Historical Value at Risk from the trade P&L series.
    VaR_95 = 5th percentile of pnl_pct distribution (worst loss exceeded only 5% of time)
    VaR_99 = 1st percentile
    CVaR_95 (Expected Shortfall) = mean of losses beyond VaR_95
    Also: best trade, worst trade, skewness, kurtosis
    Returns dict with all metrics as formatted strings + status
    """
    empty = {
        "var_95": "N/A", "var_99": "N/A", "cvar_95": "N/A",
        "best_trade": "N/A", "worst_trade": "N/A",
        "skewness": "N/A", "kurtosis": "N/A",
        "status": "NO_DATA", "n": 0,
    }
    pnl = pm.get("pnl_series")
    if pnl is None or (hasattr(pnl,"empty") and pnl.empty):
        return empty
    if len(pnl) < 5:
        return empty
    import numpy as np
    arr = pnl.values.astype(float)
    var95 = float(np.percentile(arr, 5))
    var99 = float(np.percentile(arr, 1))
    cvar95 = float(arr[arr <= var95].mean()) if (arr <= var95).any() else var95
    best = float(arr.max())
    worst = float(arr.min())
    # Skewness and kurtosis
    try:
        from scipy import stats as sp_stats
        skew = float(sp_stats.skew(arr))
        kurt = float(sp_stats.kurtosis(arr))
    except Exception:
        mean = arr.mean(); std = arr.std()
        skew = float(((arr - mean)**3).mean() / (std**3 + 1e-10)) if std > 0 else 0.0
        kurt = float(((arr - mean)**4).mean() / (std**4 + 1e-10) - 3) if std > 0 else 0.0

    status = "risk" if var95 < -0.10 else ("plain" if var95 < -0.05 else "ok")
    return {
        "var_95":     f"{var95*100:.2f}%",
        "var_99":     f"{var99*100:.2f}%",
        "cvar_95":    f"{cvar95*100:.2f}%",
        "best_trade": f"{best*100:.2f}%",
        "worst_trade":f"{worst*100:.2f}%",
        "skewness":   f"{skew:.3f}",
        "kurtosis":   f"{kurt:.3f}",
        "status":     status,
        "n":          len(arr),
        "var_95_raw": var95,
        "pnl_series": pnl,
    }


def build_trade_frequency(ledger: pd.DataFrame) -> dict:
    """
    Measures trading frequency: trades per month, days between trades, total turnover.
    Professional benchmark: 2-8 new paper trades per month.
    Returns dict: avg_trades_per_month, avg_days_between, total_trades, turnover_note, status, monthly_df
    monthly_df: DataFrame with month_label, n_trades, status
    """
    empty = {"avg_trades_per_month": None, "avg_trades_str": "N/A",
             "avg_days_between": None, "total_trades": 0,
             "turnover_note": "No data", "status": "NO_DATA", "monthly_df": pd.DataFrame()}
    if ledger.empty or "ticker" not in ledger.columns:
        return empty
    date_col = next((c for c in ["entry_date","exit_date"] if c in ledger.columns), None)
    if not date_col:
        return empty
    df = ledger.copy()
    df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["_date"]).sort_values("_date")
    if df.empty:
        return empty

    total = len(df)
    df["_ym"] = df["_date"].dt.to_period("M")
    monthly = df.groupby("_ym").size().reset_index(name="n_trades")
    avg_per_month = float(monthly["n_trades"].mean())

    # Days between consecutive trades
    dates_sorted = df["_date"].sort_values().reset_index(drop=True)
    if len(dates_sorted) > 1:
        gaps = dates_sorted.diff().dt.days.dropna()
        avg_gap = float(gaps.mean())
    else:
        avg_gap = None

    # Status
    if avg_per_month > 12:
        status = "risk"
        note = f"Over-trading: {avg_per_month:.1f} trades/month. High turnover erodes edge through slippage. Target: 2-8/month."
    elif avg_per_month > 8:
        status = "plain"
        note = f"Slightly elevated: {avg_per_month:.1f} trades/month. Monitor for noise-driven entries."
    elif avg_per_month >= 2:
        status = "ok"
        note = f"Good frequency: {avg_per_month:.1f} trades/month. Within professional target range (2-8)."
    else:
        status = "plain"
        note = f"Low frequency: {avg_per_month:.1f} trades/month. May be missing opportunities or being too selective."

    monthly["month_label"] = monthly["_ym"].astype(str)
    monthly["status"] = monthly["n_trades"].apply(
        lambda x: "risk" if x > 12 else ("ok" if 2 <= x <= 8 else "plain")
    )
    monthly_out = monthly[["status","month_label","n_trades"]].copy()

    return {
        "avg_trades_per_month": avg_per_month,
        "avg_trades_str": f"{avg_per_month:.1f}/month",
        "avg_days_between": avg_gap,
        "total_trades": total,
        "turnover_note": note,
        "status": status,
        "monthly_df": monthly_out,
    }


def build_rolling_sharpe(ledger: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Rolling Sharpe ratio over the last `window` closed trades.
    If fewer trades than window, uses all available.
    Returns DataFrame with columns: trade_num, rolling_sharpe, n_in_window, status
    status: "ok" if sharpe > 1.0, "plain" if 0-1.0, "risk" if < 0
    Also computes a simple linear trend to detect edge decay.
    """
    if ledger.empty or "pnl_pct" not in ledger.columns:
        return pd.DataFrame()
    closed = ledger[ledger["status"].astype(str).str.upper().str.startswith("CLOSED")].copy()
    if len(closed) < 5:
        return pd.DataFrame()
    date_col = next((c for c in ["exit_date","entry_date"] if c in closed.columns), None)
    if date_col:
        closed["_sort"] = pd.to_datetime(closed[date_col], errors="coerce")
        closed = closed.sort_values("_sort", na_position="last")
    closed["_pnl"] = pd.to_numeric(closed["pnl_pct"], errors="coerce")
    closed = closed.dropna(subset=["_pnl"]).reset_index(drop=True)
    if len(closed) < 3:
        return pd.DataFrame()

    RF = 0.0001  # ~4% annual / 252 trading days / assumed ~1 trade/day proxy
    rows = []
    for i in range(len(closed)):
        start = max(0, i - window + 1)
        w = closed["_pnl"].iloc[start:i+1]
        if len(w) < 3:
            rows.append({"trade_num": i+1, "rolling_sharpe": None, "n_in_window": len(w), "status": "plain"})
            continue
        mean_r = float(w.mean())
        std_r = float(w.std())
        if std_r < 1e-8:  # All trades identical — Sharpe undefined
            rows.append({"trade_num": i+1, "rolling_sharpe": None, "n_in_window": len(w), "status": "plain"})
            continue
        sharpe = (mean_r - RF) / std_r * (len(w)**0.5)
        s = "ok" if sharpe > 1.0 else ("risk" if sharpe < 0 else "plain")
        rows.append({"trade_num": i+1, "rolling_sharpe": round(sharpe, 3), "n_in_window": len(w), "status": s})

    return pd.DataFrame(rows)


def build_adv_check(exposure: pd.DataFrame, technicals: pd.DataFrame, account_size: float = 100_000.0) -> pd.DataFrame:
    """
    Checks if position sizes exceed 10% of Average Daily Volume (ADV).
    Institutional rule: never be more than 10% of average daily volume to avoid market impact.
    ADV proxy: use volume or avg_volume column from technicals if available.
    position_dollar = effective_weight * account_size
    adv_dollar = avg_volume * last_close (estimated from technicals/market)
    liquidity_ratio = position_dollar / adv_dollar
    Returns DataFrame: status, ticker, nominal_weight, position_dollar, adv_est, liquidity_ratio, note
    status: "risk" if ratio > 0.10, "plain" if <= 0.10
    """
    if exposure.empty or "ticker" not in exposure.columns or "effective_weight" not in exposure.columns:
        return pd.DataFrame()

    exp = exposure[["ticker","effective_weight"]].copy()
    exp["effective_weight"] = pd.to_numeric(exp["effective_weight"], errors="coerce").fillna(0)
    exp["position_dollar"] = exp["effective_weight"] * account_size

    # Try to get ADV from technicals
    adv_map = {}
    price_map = {}
    if not technicals.empty and "ticker" in technicals.columns:
        for vol_col in ["avg_volume","volume","avg_vol","vol_20d"]:
            if vol_col in technicals.columns:
                for _, r in technicals.iterrows():
                    t = str(r["ticker"]).upper()
                    v = pd.to_numeric(r.get(vol_col), errors="coerce")
                    if pd.notna(v) and v > 0:
                        adv_map[t] = float(v)
                break
        for price_col in ["last_close","close","price","last_price"]:
            if price_col in technicals.columns:
                for _, r in technicals.iterrows():
                    t = str(r["ticker"]).upper()
                    p = pd.to_numeric(r.get(price_col), errors="coerce")
                    if pd.notna(p) and p > 0:
                        price_map[t] = float(p)
                break

    rows = []
    for _, r in exp.iterrows():
        t = str(r["ticker"]).upper()
        w = float(r["effective_weight"])
        pos_dollar = float(r["position_dollar"])
        adv_shares = adv_map.get(t)
        price = price_map.get(t)
        if adv_shares and price:
            adv_dollar = adv_shares * price
            ratio = pos_dollar / adv_dollar if adv_dollar > 0 else None
        else:
            adv_dollar = None
            ratio = None

        if ratio is None:
            s = "plain"
            note = "No ADV data - cannot check liquidity"
            ratio_str = "N/A"
            adv_str = "N/A"
        elif ratio > 0.10:
            s = "risk"
            note = f"Position is {ratio*100:.1f}% of daily volume - may cause slippage at this size"
            ratio_str = f"{ratio*100:.2f}%"
            adv_str = f"${adv_dollar:,.0f}"
        else:
            s = "ok"
            note = f"Liquid - position is only {ratio*100:.2f}% of daily volume"
            ratio_str = f"{ratio*100:.2f}%"
            adv_str = f"${adv_dollar:,.0f}"

        rows.append({
            "status": s,
            "ticker": t,
            "nominal_weight": f"{w*100:.2f}%",
            "position_dollar": f"${pos_dollar:,.0f}",
            "adv_est": adv_str,
            "pct_of_adv": ratio_str,
            "note": note,
        })
    return pd.DataFrame(rows).sort_values("status").reset_index(drop=True) if rows else pd.DataFrame()


def build_multi_benchmark(pm: dict, market: pd.DataFrame) -> pd.DataFrame:
    """
    Compares strategy return vs multiple benchmarks: SPY, QQQ, IWM, and a 60/40 proxy.
    Uses ret_20d from market snapshot for each benchmark ticker.
    Returns DataFrame: status, benchmark, benchmark_ret_20d, strategy_ret, alpha, note
    """
    if pm.get("n_closed", 0) == 0 or market.empty:
        return pd.DataFrame()

    strategy_ret = pm.get("total_return", "N/A")
    strategy_float = None
    if strategy_ret != "N/A":
        try:
            strategy_float = float(str(strategy_ret).replace("%","")) / 100
        except Exception:
            pass

    benchmarks = {
        "SPY (S&P 500)": "SPY",
        "QQQ (Nasdaq 100)": "QQQ",
        "IWM (Russell 2000)": "IWM",
        "GLD (Gold)": "GLD",
    }

    ret_map = {}
    if "ret_20d" in market.columns and "ticker" in market.columns:
        for _, r in market.iterrows():
            t = str(r["ticker"]).upper()
            v = pd.to_numeric(r.get("ret_20d"), errors="coerce")
            if pd.notna(v):
                ret_map[t] = float(v)

    rows = []
    for label, sym in benchmarks.items():
        bret = ret_map.get(sym.upper())
        bret_str = f"{bret*100:.2f}%" if bret is not None else "N/A"
        if strategy_float is not None and bret is not None:
            alpha = strategy_float - bret
            # NOTE: strategy_float = cumulative total return (all-time horizon);
            # bret = benchmark 20-day return. Different windows — directional only.
            alpha_str = f"{alpha*100:.2f}% (total vs 20d)"
            if alpha > 0.02:
                s = "ok"
                note = f"Total return leads {sym} 20d by {alpha*100:.2f}% (not same-window — directional guide only)"
            elif alpha < -0.02:
                s = "risk"
                note = f"Total return trails {sym} 20d by {abs(alpha)*100:.2f}% (not same-window — directional guide only)"
            else:
                s = "plain"
                note = f"Roughly in line with {sym} 20d (not same-window comparison)"
        else:
            s = "plain"
            alpha_str = "N/A"
            note = "Insufficient data for comparison"
        rows.append({
            "status": s,
            "benchmark": label,
            "benchmark_20d": bret_str,
            "strategy_return": strategy_ret,
            "alpha": alpha_str,
            "note": note,
        })

    # Add synthetic 60/40 (SPY 60% + AGG/TLT 40%)
    spy_ret = ret_map.get("SPY")
    bond_ret = ret_map.get("TLT") or ret_map.get("AGG") or ret_map.get("BND")
    if spy_ret is not None and bond_ret is not None:
        port_6040 = spy_ret * 0.60 + bond_ret * 0.40
        if strategy_float is not None:
            alpha_6040 = strategy_float - port_6040
            s_6040 = "ok" if alpha_6040 > 0.02 else ("risk" if alpha_6040 < -0.02 else "plain")
            alpha_6040_str = f"{alpha_6040*100:.2f}%"
        else:
            s_6040 = "plain"; alpha_6040_str = "N/A"
        rows.append({
            "status": s_6040,
            "benchmark": "60/40 (SPY 60% + Bond 40%)",
            "benchmark_20d": f"{port_6040*100:.2f}%",
            "strategy_return": strategy_ret,
            "alpha": alpha_6040_str,
            "note": "Traditional balanced portfolio benchmark",
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def tab_strategy_scorecard():
    st.header("Strategy Scorecard")
    ledger      = read_csv(FILES["paper_ledger"])
    market      = read_csv(FILES["market_snapshot"])
    technicals  = read_csv(FILES["technicals"])
    exposure    = read_csv(FILES["exposure"])
    focus       = build_focus_list()

    pm  = build_perf_metrics(ledger, market)
    cb  = build_circuit_breaker(ledger)
    evk = build_ev_kelly(pm)
    n   = pm["n_closed"]

    # Header status driven by circuit breaker
    _cb_kind = "risk" if cb["status"] == "STOP" else ("wait" if cb["status"] == "WARN" else ("supportive" if cb["status"] == "OK" else "blocked"))
    render_layer_workbench_header(
        "Scorecard",
        "Strategy Performance Scorecard",
        "Paper trade outcomes vs SPY benchmark. Need 30+ closed samples for statistical meaning.",
        [
            ("Closed Samples",   n,                    "supportive" if n >= 30 else ("cyan" if n > 0 else "blocked")),
            ("Circuit Breaker",  cb["status"],          _cb_kind),
            ("EV / Trade",       evk["ev_str"],         evk["ev_status"] if evk["ev_status"] != "NO_DATA" else "blocked"),
            ("Half-Kelly Size",  evk["half_kelly_str"], evk["kelly_status"] if evk["kelly_status"] != "NO_DATA" else "blocked"),
        ],
    )

    if n < 30:
        st.warning(
            f"⚠  Only **{n}** closed paper samples. Need at least **30** for statistically meaningful conclusions. "
            "Metrics are directional only — do not adjust strategy weights yet."
        )

    scorecard_rows = pd.DataFrame([
        {"status": "OK",     "metric": "Closed Samples",    "value": str(n),              "need_30_plus": "Yes — under 30 is directional only", "interpretation": "Minimum 30 for inference"},
        {"status": "OK",     "metric": "Total Return",       "value": pm["total_return"],  "need_30_plus": "—",  "interpretation": "Compounded P&L across all closed trades"},
        {"status": "OK",     "metric": "Win Rate",           "value": pm["win_rate"],      "need_30_plus": "—",  "interpretation": "% of trades closed with positive P&L"},
        {"status": "OK",     "metric": "Avg Return/Trade",   "value": pm["avg_return"],    "need_30_plus": "—",  "interpretation": "Mean P&L per closed trade"},
        {"status": "RISK",   "metric": "Max Drawdown",       "value": pm["max_drawdown"],  "need_30_plus": "—",  "interpretation": "Largest peak-to-trough loss in the sequence"},
        {"status": "REVIEW", "metric": "Sharpe Ratio",       "value": pm["sharpe"],        "need_30_plus": "Yes", "interpretation": ">1.0 is acceptable; >2.0 is strong"},
        {"status": "REVIEW", "metric": "Sortino Ratio",      "value": pm["sortino"],       "need_30_plus": "Yes", "interpretation": "Penalises only downside vol; >1.5 is solid"},
        {"status": "REVIEW", "metric": "Calmar Ratio",       "value": pm["calmar"],        "need_30_plus": "Yes", "interpretation": "Return / Max Drawdown; >1.0 is good"},
        {"status": "REVIEW", "metric": "EV Per Trade",       "value": evk["ev_str"],       "need_30_plus": "Yes", "interpretation": "Expected value per trade; must be positive for viable strategy"},
        {"status": "REVIEW", "metric": "Kelly Fraction",     "value": evk["kelly_str"],    "need_30_plus": "Yes", "interpretation": "Theoretical max bet size; use half-Kelly in practice"},
        {"status": "REVIEW", "metric": "Half-Kelly (Rec.)",  "value": evk["half_kelly_str"], "need_30_plus": "Yes", "interpretation": "Recommended position size limit per trade"},
        {"status": "REVIEW", "metric": "SPY 20d Return",     "value": pm["spy_20d"],       "need_30_plus": "—",  "interpretation": "Benchmark — recent 20-day SPY return"},
        {"status": "REVIEW", "metric": "Alpha vs SPY",       "value": pm["alpha"],         "need_30_plus": "Yes", "interpretation": "Positive alpha = outperforming benchmark"},
        {"status": "REVIEW", "metric": "Information Ratio",  "value": pm["ir"],            "need_30_plus": "Yes", "interpretation": ">0.5 shows consistent benchmark outperformance"},
    ])

    sc_tabs = st.tabs([
        "Overview", "Circuit Breaker", "Rolling Performance",
        "Hold Time", "Regime Breakdown", "Entry Timing",
        "Sizing Compliance", "Drawdown", "Trade Log",
        "vs Benchmark", "ATR Sizing",
        "Discipline", "Stop & MAE", "Opportunity Cost", "Signal Power",
        "P&L Calendar", "Catalyst Analysis", "VaR & Distribution",
        "Trade Frequency", "Multi Benchmark",
    ])

    # ── Overview ───────────────────────────────────────────────────────────
    with sc_tabs[0]:
        st.subheader("Risk-Adjusted Metrics")
        render_badge_table(scorecard_rows, height=480)
        if _PLOTLY and not pm["pnl_series"].empty:
            _cum = pm["cumulative"]
            _fig_cum = go.Figure()
            _fig_cum.add_trace(go.Scatter(
                x=list(range(1, len(_cum) + 1)),
                y=((_cum - 1) * 100).tolist(),
                mode="lines+markers",
                line=dict(color="#22d3ee", width=2),
                name="Portfolio",
                hovertemplate="Trade %{x}<br>Return: %{y:.2f}%<extra></extra>",
            ))
            _fig_cum.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
            _fig_cum.update_layout(
                height=280, margin=dict(l=10, r=10, t=28, b=20),
                title=dict(text="Cumulative Return By Trade Sequence", font=dict(size=13), x=0),
                xaxis_title="Trade #", yaxis_title="Cumulative Return %",
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(gridcolor="#e5e7eb"),
                yaxis=dict(gridcolor="#e5e7eb", ticksuffix="%"),
                font=dict(family="Inter,sans-serif", size=12),
            )
            st.plotly_chart(_fig_cum, use_container_width=True)
        if evk["ev_pct"] is not None:
            st.info(evk["interpretation"])

    # ── Circuit Breaker ────────────────────────────────────────────────────
    with sc_tabs[1]:
        st.subheader("Circuit Breaker Status")
        st.caption("Hard stops that override all other signals. If triggered, halt paper trading until root cause is resolved.")

        _cb_color = "#b91c1c" if cb["status"] == "STOP" else ("#f59e0b" if cb["status"] == "WARN" else "#16a34a")
        st.markdown(
            f'<div style="background:{_cb_color};color:white;padding:16px 20px;border-radius:6px;'
            f'font-size:16px;font-weight:700;margin-bottom:16px;">{cb["message"]}</div>',
            unsafe_allow_html=True,
        )

        _cb_c1, _cb_c2, _cb_c3 = st.columns(3)
        _cb_c1.metric("Overall Status",        cb["status"])
        _cb_c2.metric("Consecutive Losses",    cb["consecutive_losses"])
        _cb_c3.metric("Monthly P&L",           cb["monthly_pnl_str"])

        if cb["rules"]:
            render_badge_table(pd.DataFrame(cb["rules"]), height=160)

        if _PLOTLY and not pm["pnl_series"].empty:
            _pnl_colors = ["#b91c1c" if v < 0 else "#16a34a" for v in pm["pnl_series"].tolist()]
            _fig_pnl = go.Figure(go.Bar(
                x=list(range(1, len(pm["pnl_series"]) + 1)),
                y=(pm["pnl_series"] * 100).tolist(),
                marker_color=_pnl_colors,
                hovertemplate="Trade %{x}<br>P&L: %{y:.3f}%<extra></extra>",
            ))
            _fig_pnl.add_hline(y=0, line_color="#111827", line_width=1)
            _fig_pnl.update_layout(
                height=240, margin=dict(l=10, r=10, t=28, b=20),
                title=dict(text="Trade-by-Trade P&L (red = loss)", font=dict(size=13), x=0),
                xaxis_title="Trade #", yaxis_title="P&L %",
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(gridcolor="#e5e7eb"),
                yaxis=dict(gridcolor="#e5e7eb", ticksuffix="%"),
                font=dict(family="Inter,sans-serif", size=12),
            )
            st.plotly_chart(_fig_pnl, use_container_width=True)

        st.markdown("""
**Circuit Breaker Rules:**
| Rule | WARN | STOP |
|------|------|------|
| Consecutive Losses | 3 in a row | 5 in a row |
| Monthly Drawdown | ≤ -10% this month | ≤ -15% this month |

**When WARN:** Reduce all new paper sizes to 50%. Review thesis for active positions.
**When STOP:** No new paper trades. Close positions above ATR limit. Root-cause review required before resuming.
""")

    # ── Rolling Performance ────────────────────────────────────────────────
    with sc_tabs[2]:
        st.subheader("Rolling Performance Windows")
        st.caption("Is the edge consistent over time? A strategy that only worked in one period may have been luck.")
        _roll_df = build_rolling_performance(ledger)
        if _roll_df.empty:
            st.info("Need closed trades with exit_date or entry_date to compute rolling performance.")
        else:
            render_badge_table(_roll_df, height=200)
            if _PLOTLY:
                _roll_valid = _roll_df[_roll_df["total_return_pct"] != "—"].copy()
                if not _roll_valid.empty:
                    _roll_vals = _roll_valid["total_return_pct"].str.replace("%", "").astype(float)
                    _roll_colors = ["#16a34a" if v >= 0 else "#b91c1c" for v in _roll_vals]
                    _fig_roll = go.Figure(go.Bar(
                        x=_roll_valid["period"].tolist(),
                        y=_roll_vals.tolist(),
                        marker_color=_roll_colors,
                        text=[f"{v:.2f}%" for v in _roll_vals],
                        textposition="outside",
                        hovertemplate="%{x}<br>Return: %{y:.2f}%<extra></extra>",
                    ))
                    _fig_roll.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
                    _fig_roll.update_layout(
                        height=280, margin=dict(l=10, r=10, t=28, b=20),
                        title=dict(text="Rolling Return By Period", font=dict(size=13), x=0),
                        yaxis_title="Total Return %", xaxis_title="",
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(gridcolor="#e5e7eb"),
                        yaxis=dict(gridcolor="#e5e7eb", ticksuffix="%"),
                        font=dict(family="Inter,sans-serif", size=12),
                    )
                    st.plotly_chart(_fig_roll, use_container_width=True)
        st.markdown("""
**What to look for:**
- **All three windows positive** → consistent edge, not a one-hit wonder
- **1M positive, 6M negative** → recent recovery; watch for reversion
- **1M negative, 6M positive** → recent drawdown; check if regime changed
- **All negative** → review strategy thesis before adding more paper
""")

    # ── Hold Time Analysis ─────────────────────────────────────────────────
    with sc_tabs[3]:
        st.subheader("Winner vs Loser Holding Time")
        st.caption("Disposition effect: holding losers too long and cutting winners too early is the most common systematic bias.")
        _ht_df = build_hold_time_analysis(ledger)
        if _ht_df.empty:
            st.info("Need closed trades with entry_date, exit_date (or holding_days) and pnl_pct.")
        else:
            render_badge_table(_ht_df, height=200)
            _w = _ht_df[_ht_df["outcome"] == "Winners"]["avg_days"].values
            _l = _ht_df[_ht_df["outcome"] == "Losers"]["avg_days"].values
            if len(_w) > 0 and len(_l) > 0:
                if _l[0] > _w[0]:
                    st.error(f"⚠ **Disposition Effect Detected** — Losers held {_l[0]:.1f} days avg vs Winners {_w[0]:.1f} days. Cut losers faster.")
                else:
                    st.success(f"✅ **No Disposition Effect** — Winners held {_w[0]:.1f} days avg vs Losers {_l[0]:.1f} days. Good discipline.")
            if _PLOTLY and not _ht_df.empty:
                _ht_plot = _ht_df[_ht_df["outcome"] != "All Trades"].copy()
                if not _ht_plot.empty:
                    _ht_colors = ["#16a34a" if o == "Winners" else "#b91c1c" for o in _ht_plot["outcome"]]
                    _fig_ht = go.Figure(go.Bar(
                        x=_ht_plot["outcome"].tolist(),
                        y=_ht_plot["avg_days"].tolist(),
                        marker_color=_ht_colors,
                        text=[f"{v:.1f}d" for v in _ht_plot["avg_days"]],
                        textposition="outside",
                        hovertemplate="%{x}<br>Avg days: %{y:.1f}<extra></extra>",
                    ))
                    _fig_ht.update_layout(
                        height=260, margin=dict(l=10, r=10, t=28, b=20),
                        title=dict(text="Average Holding Days: Winners vs Losers", font=dict(size=13), x=0),
                        yaxis_title="Avg Holding Days", xaxis_title="",
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(gridcolor="#e5e7eb"),
                        yaxis=dict(gridcolor="#e5e7eb"),
                        font=dict(family="Inter,sans-serif", size=12),
                    )
                    st.plotly_chart(_fig_ht, use_container_width=True)
        st.markdown("""
**Rule of thumb:**
- `avg_loser_days < avg_winner_days` → healthy; you're cutting losers and riding winners
- `avg_loser_days > avg_winner_days` → disposition effect; set hard stop-loss rules
- Median is more robust than average when sample size < 30
""")

    # ── Regime Breakdown ──────────────────────────────────────────────────
    with sc_tabs[4]:
        st.subheader("Market Regime Performance")
        st.caption("Does this strategy work in all market conditions, or only in bull markets?")
        _rg_df = build_regime_breakdown(ledger, market)
        if _rg_df.empty:
            st.info("Need closed trades to compute regime breakdown.")
        else:
            render_badge_table(_rg_df, height=220)
            if _PLOTLY and "total_return_pct" in _rg_df.columns:
                _rg_plot = _rg_df[_rg_df["total_return_pct"] != "—"].copy()
                if not _rg_plot.empty:
                    _rg_vals = _rg_plot["total_return_pct"].str.replace("%", "").astype(float)
                    _rg_colors = ["#16a34a" if v >= 0 else "#b91c1c" for v in _rg_vals]
                    _fig_rg = go.Figure(go.Bar(
                        x=_rg_plot["regime"].tolist(),
                        y=_rg_vals.tolist(),
                        marker_color=_rg_colors,
                        text=[f"{v:.2f}%" for v in _rg_vals],
                        textposition="outside",
                        hovertemplate="%{x}<br>Return: %{y:.2f}%<extra></extra>",
                    ))
                    _fig_rg.add_hline(y=0, line_dash="dash", line_color="#9ca3af")
                    _fig_rg.update_layout(
                        height=300, margin=dict(l=10, r=10, t=28, b=20),
                        title=dict(text="Return By Market Regime / Sample Period", font=dict(size=13), x=0),
                        yaxis_title="Total Return %", xaxis_title="",
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(gridcolor="#e5e7eb", tickangle=-15),
                        yaxis=dict(gridcolor="#e5e7eb", ticksuffix="%"),
                        font=dict(family="Inter,sans-serif", size=12),
                    )
                    st.plotly_chart(_fig_rg, use_container_width=True)
        st.info("⚠ Regime data is approximated from current SPY trend_state. Future versions will store regime at trade-entry time for exact breakdown.")

    # ── Entry Timing ──────────────────────────────────────────────────────
    with sc_tabs[5]:
        st.subheader("Entry Timing By Day Of Week")
        st.caption("Which days produce the best results? Avoid chasing Monday opens or Friday shorts.")
        _et_df = build_entry_timing(ledger)
        if _et_df.empty:
            st.info("Need closed trades with entry_date and pnl_pct to compute timing analysis.")
        else:
            render_badge_table(_et_df, height=260)
            if _PLOTLY:
                _et_plot = _et_df.copy()
                _et_wr = _et_plot["win_rate_pct"].str.replace("%", "").astype(float)
                _et_colors = ["#16a34a" if v >= 55 else "#22d3ee" if v >= 45 else "#b91c1c" for v in _et_wr]
                _fig_et = go.Figure(go.Bar(
                    x=_et_plot["day_of_week"].tolist(),
                    y=_et_wr.tolist(),
                    marker_color=_et_colors,
                    text=[f"{v:.1f}%" for v in _et_wr],
                    textposition="outside",
                    hovertemplate="%{x}<br>Win rate: %{y:.1f}%<extra></extra>",
                ))
                _fig_et.add_hline(y=50, line_dash="dash", line_color="#9ca3af", annotation_text="50% baseline")
                _fig_et.update_layout(
                    height=260, margin=dict(l=10, r=10, t=28, b=20),
                    title=dict(text="Win Rate By Entry Day Of Week", font=dict(size=13), x=0),
                    yaxis_title="Win Rate %", xaxis_title="",
                    yaxis=dict(gridcolor="#e5e7eb", ticksuffix="%", range=[0, 100]),
                    xaxis=dict(gridcolor="#e5e7eb"),
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Inter,sans-serif", size=12),
                )
                st.plotly_chart(_fig_et, use_container_width=True)
        st.markdown("""
**What to look for:**
- **Monday entries** are often risky — gap risk from weekend news not priced in
- **Friday entries** carry weekend risk — avoid unless thesis is very strong
- **Best win-rate day** highlighted green — but needs >10 samples per day to be meaningful
""")

    # ── Sizing Compliance ─────────────────────────────────────────────────
    with sc_tabs[6]:
        st.subheader("ATR Sizing Compliance")
        st.caption("Are you actually following your own position sizing rules? Deviation from ATR limits is a behavioral red flag.")
        _sc = build_sizing_compliance(ledger, exposure, technicals)
        _sc_c1, _sc_c2, _sc_c3 = st.columns(3)
        _sc_c1.metric("Compliance Rate",    _sc["compliance_str"])
        _sc_c2.metric("Oversized Tickers",  len(_sc["oversized_tickers"]))
        _sc_c3.metric("Avg Deviation",      f'{_sc["avg_deviation_pct"]:.2f}%' if _sc["avg_deviation_pct"] is not None else "N/A")
        _sc_missing_atr = _sc.get("n_missing_atr", 0)
        if _sc_missing_atr > 0:
            st.info(f"ℹ {_sc_missing_atr} position(s) have no ATR data in technicals — excluded from rate but shown as NO_ATR_DATA. Run daily runner to populate ATR.")
        if _sc["status"] == "RISK":
            st.error("🚨 Low compliance — frequently exceeding ATR sizing limits. This increases risk of large unexpected losses.")
        elif _sc["status"] == "WARN":
            st.warning("⚠ Moderate compliance — some positions exceed ATR limits. Review oversized tickers.")
        elif _sc["status"] == "OK":
            st.success("✅ Good compliance — positions are within ATR sizing bounds.")
        if _sc["oversized_tickers"]:
            st.markdown(f"**Oversized tickers:** {', '.join(_sc['oversized_tickers'])}")
        if not _sc["detail_df"].empty:
            render_badge_table(_sc["detail_df"], height=320)
            if _PLOTLY:
                _scd = _sc["detail_df"].copy()
                if "ticker" in _scd.columns and "atr_max" in _scd.columns and "effective_weight" in _scd.columns:
                    _scd_t = _scd["ticker"].astype(str).tolist()
                    _scd_atr = _scd["atr_max"].str.replace("%", "").astype(float)
                    _scd_eff = _scd["effective_weight"].str.replace("%", "").astype(float)
                    _fig_sc = go.Figure()
                    _fig_sc.add_trace(go.Bar(
                        x=_scd_t, y=_scd_atr.tolist(), name="ATR Max", marker_color="#22d3ee",
                        hovertemplate="%{x}<br>ATR Max: %{y:.2f}%<extra></extra>",
                    ))
                    _fig_sc.add_trace(go.Bar(
                        x=_scd_t, y=_scd_eff.tolist(), name="Actual Weight", marker_color="#f59e0b",
                        hovertemplate="%{x}<br>Actual: %{y:.2f}%<extra></extra>",
                    ))
                    _fig_sc.update_layout(
                        barmode="group",
                        height=280, margin=dict(l=10, r=10, t=28, b=20),
                        title=dict(text="ATR Max vs Actual Weight Per Ticker", font=dict(size=13), x=0),
                        yaxis_title="Weight %", xaxis_title="",
                        yaxis=dict(gridcolor="#e5e7eb", ticksuffix="%"),
                        xaxis=dict(gridcolor="#e5e7eb"),
                        plot_bgcolor="white", paper_bgcolor="white",
                        font=dict(family="Inter,sans-serif", size=12),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(_fig_sc, use_container_width=True)
        else:
            st.info("No overlap between exposure and technicals. Run the daily runner to generate both files.")
        st.markdown("""
**Compliance rule:** `effective_weight ≤ atr_suggested_weight × 1.05` (5% tolerance).
**Oversized** = actual weight exceeds ATR max by more than 5%.
Fix: Reduce the position or wait for ATR to expand (lower volatility = larger allowed size).
""")

    # ── Drawdown ─────────────────────────────────────────────────────────
    with sc_tabs[7]:
        st.subheader("Drawdown — Underwater Chart")
        st.caption("Shows how far the strategy was below its previous peak at each trade. Red bars = losses vs peak.")
        if _PLOTLY and not pm["drawdown_series"].empty:
            _dd = pm["drawdown_series"]
            _dd_colors = ["#b91c1c" if v < -0.05 else "#f87171" if v < 0 else "#22d3ee" for v in _dd]
            _fig_dd = go.Figure(go.Bar(
                x=list(range(1, len(_dd) + 1)),
                y=(_dd * 100).tolist(),
                marker_color=_dd_colors,
                hovertemplate="Trade %{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
            ))
            _fig_dd.add_hline(y=0, line_dash="solid", line_color="#111827", line_width=1)
            _fig_dd.update_layout(
                height=300, margin=dict(l=10, r=10, t=28, b=20),
                title=dict(text=f"Max Drawdown: {pm['max_drawdown']}", font=dict(size=13), x=0),
                xaxis_title="Trade #", yaxis_title="Drawdown %",
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(gridcolor="#e5e7eb"),
                yaxis=dict(gridcolor="#e5e7eb", ticksuffix="%"),
                font=dict(family="Inter,sans-serif", size=12),
            )
            st.plotly_chart(_fig_dd, use_container_width=True)
        else:
            st.info("Need at least 2 closed trades to plot drawdown.")
        st.markdown("""
**Interpretation guide:**
- **Deep red bars (< -5%)** — the strategy was losing significantly vs its own peak at that point.
- **Short red bars** — minor dips, acceptable in any strategy.
- **Recovery speed** — how quickly bars return to zero after a dip shows strategy resilience.
- **Rule of thumb:** Max drawdown > 20% in a paper portfolio → review position sizing.
""")

    # ── Trade Log ─────────────────────────────────────────────────────────
    with sc_tabs[8]:
        st.subheader("Closed Trade Log")
        if ledger.empty:
            st.info("No ledger data.")
        else:
            cols = [c for c in [
                "trade_id", "ticker", "sleeve", "status", "entry_date", "entry_price",
                "exit_date", "exit_price", "pnl_pct", "holding_days", "thesis", "notes"
            ] if c in ledger.columns]
            render_badge_table(ledger[cols], height=560)

    # ── vs Benchmark ──────────────────────────────────────────────────────
    with sc_tabs[9]:
        st.subheader("Portfolio vs Benchmark")
        st.caption("Rough comparison using available return data. SPY returns from the market snapshot.")
        if not market.empty:
            spy_cols = [c for c in ["ticker", "last_close", "ret_5d", "ret_20d", "ret_63d", "trend_state"] if c in market.columns]
            spy_row = market[market["ticker"].astype(str).str.upper() == "SPY"][spy_cols] if spy_cols else pd.DataFrame()
            qqq_row = market[market["ticker"].astype(str).str.upper() == "QQQ"][spy_cols] if spy_cols else pd.DataFrame()
            benchmark_df = pd.concat([spy_row, qqq_row], ignore_index=True) if not spy_row.empty else pd.DataFrame()
            if not benchmark_df.empty:
                render_badge_table(benchmark_df, height=120)
        comparison = pd.DataFrame([
            {"status": "REVIEW", "dimension": "Total Return (paper trades)", "portfolio": pm["total_return"],  "spy_20d": pm["spy_20d"],           "note": "Portfolio is compounded paper-trade P&L; SPY is simple 20-day return"},
            {"status": "REVIEW", "dimension": "Alpha",                        "portfolio": pm["alpha"],         "spy_20d": "0.00%",                  "note": "Positive = outperforming benchmark over this sample"},
            {"status": "REVIEW", "dimension": "EV Per Trade",                 "portfolio": evk["ev_str"],       "spy_20d": "Buy-and-hold = ~∞",      "note": "Positive EV = strategy has mathematical edge"},
            {"status": "REVIEW", "dimension": "Max Drawdown",                 "portfolio": pm["max_drawdown"],  "spy_20d": "Varies",                 "note": "SPY 2024 max drawdown was ~8%; 2022 was ~25%"},
            {"status": "REVIEW", "dimension": "Sharpe Ratio",                 "portfolio": pm["sharpe"],        "spy_20d": "~0.4–0.6 historical avg","note": "SPY long-run Sharpe ≈ 0.4–0.6; this is per-trade not annualised"},
            {"status": "WARN",   "dimension": "Sample size",                  "portfolio": str(n),              "spy_20d": "30+ years",              "note": "Comparison is not valid with under 30 samples"},
        ])
        render_badge_table(comparison, height=300)
        st.info("⚠  This comparison is approximate. SPY return is the last 20 days from market snapshot, not matched to the same date range as the paper trades. Use it as rough directional context only.")

    # ── ATR Sizing ────────────────────────────────────────────────────────
    with sc_tabs[10]:
        st.subheader("ATR-Based Position Sizing Guide")
        st.caption("ATR14 % determines how large each position can be for a fixed 1% account risk per trade. Formula: max_weight = 1% / ATR14%")
        atr_df = build_atr_sizing(technicals, exposure)
        if atr_df.empty:
            st.info("No ATR data. Run the daily runner to generate technical_signal_matrix.csv.")
        else:
            render_badge_table(atr_df, height=380)
            if _PLOTLY and "ticker" in technicals.columns and "atr14_pct" in technicals.columns:
                _atr = technicals.copy()
                _atr["atr14_pct"] = pd.to_numeric(_atr["atr14_pct"], errors="coerce")
                _atr = _atr.dropna(subset=["atr14_pct"]).sort_values("atr14_pct", ascending=True)
                _atr_sug = (0.01 / (_atr["atr14_pct"].clip(lower=0.002))).clip(upper=0.15)
                _fig_atr = go.Figure()
                _fig_atr.add_trace(go.Bar(
                    x=(_atr_sug * 100).tolist(),
                    y=_atr["ticker"].astype(str).tolist(),
                    name="ATR Max Weight",
                    orientation="h",
                    marker_color="#22d3ee",
                    hovertemplate="<b>%{y}</b><br>ATR max: %{x:.1f}%<extra></extra>",
                ))
                _fig_atr.update_layout(
                    height=max(200, len(_atr) * 34 + 60),
                    margin=dict(l=10, r=50, t=28, b=20),
                    title=dict(text="ATR-Based Max Position Weight (1% account risk)", font=dict(size=12), x=0),
                    xaxis_title="Max Weight %", yaxis_title="",
                    xaxis=dict(gridcolor="#e5e7eb", ticksuffix="%"),
                    yaxis=dict(gridcolor="#e5e7eb"),
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Inter,sans-serif", size=12),
                )
                st.plotly_chart(_fig_atr, use_container_width=True)
        st.markdown("""
**How to use ATR sizing:**
- Higher ATR = more volatile = smaller allowed position
- Lower ATR = more stable = larger allowed position
- `max_weight = 1% / ATR14%` limits your daily loss on this name to ≈1% of account
- Adjust `account_size` (default $100k) and `risk_pct` (default 1%) to your actual situation
- This is a *guide*, not an order — L8 and L9 still control final sizing
""")

    # ── Discipline ────────────────────────────────────────────────────────
    with sc_tabs[11]:
        st.subheader("Pre-Trade Discipline")
        st.caption("Were all checklist items completed before each paper trade entry? Discipline rate measures process adherence.")
        pretrade = read_csv(FILES["pre_trade"])
        disc = build_discipline_rate(ledger, pretrade)

        _d_c1, _d_c2, _d_c3 = st.columns(3)
        _d_c1.metric("Compliance Rate", disc["compliance_str"])
        _d_c2.metric("Total Trades", disc["total_trades"])
        _d_c3.metric("Compliant", disc["compliant_trades"])

        if disc["status"] == "RISK":
            st.error("Low discipline rate — many trades entered without completing all checks. This increases behavioral risk.")
        elif disc["status"] == "WARN":
            st.warning("Moderate compliance — some trades skipped checks. Review non-compliant entries.")
        elif disc["status"] == "OK":
            st.success("Strong discipline — process is being followed consistently.")

        if disc["non_compliant_tickers"]:
            st.markdown(f"**Non-compliant entries:** {', '.join(disc['non_compliant_tickers'])}")

        if not disc["detail_df"].empty:
            render_badge_table(disc["detail_df"], height=380)
        else:
            st.info("Need closed paper trades with pretrade records to compute discipline rate.")

        st.divider()
        st.subheader("Thesis Quality Scores")
        st.caption("Each closed trade's thesis is scored 0–10 for specificity: price target, stop reference, and catalyst citation.")
        tq_df = build_thesis_quality(ledger)
        if tq_df.empty:
            st.info("Need closed trades with thesis text to score quality.")
        else:
            # Summary
            scores_raw = tq_df["score"].str.replace("/10", "").astype(float)
            avg_score = scores_raw.mean()
            low_quality = int((scores_raw < 4).sum())
            st.markdown(f"**Average thesis score:** {avg_score:.1f}/10 &nbsp;·&nbsp; **Low quality (<4):** {low_quality}")
            render_badge_table(tq_df, height=380)
            if _PLOTLY:
                _tq_bins = pd.cut(scores_raw, bins=[0, 3, 6, 10], labels=["Low (0-3)", "Medium (4-6)", "High (7-10)"])
                _tq_counts = _tq_bins.value_counts().reindex(["Low (0-3)", "Medium (4-6)", "High (7-10)"]).fillna(0)
                _fig_tq = go.Figure(go.Bar(
                    x=_tq_counts.index.tolist(),
                    y=_tq_counts.values.tolist(),
                    marker_color=["#fee2e2", "#dbeafe", "#dcfce7"],
                    text=[str(int(v)) for v in _tq_counts.values],
                    textposition="outside",
                ))
                _fig_tq.update_layout(
                    height=240, margin=dict(l=10, r=10, t=28, b=20),
                    title=dict(text="Thesis Quality Distribution", font=dict(size=13), x=0),
                    yaxis_title="Count", xaxis_title="",
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(gridcolor="#e5e7eb"), yaxis=dict(gridcolor="#e5e7eb"),
                    font=dict(family="Inter,sans-serif", size=12),
                )
                st.plotly_chart(_fig_tq, use_container_width=True)
        st.markdown("""
**Thesis quality rubric (0–10):**
- +2 pts: thesis ≥ 50 words
- +2 pts: thesis ≥ 100 words
- +2 pts: contains price target or % objective
- +2 pts: contains stop / exit condition
- +2 pts: contains catalyst (earnings, FOMC, breakout, etc.)

**Rule:** Never enter a paper trade with thesis score < 4. Low-quality thesis = high-emotion trade.
""")

    # ── Stop & MAE ────────────────────────────────────────────────────────
    with sc_tabs[12]:
        st.subheader("Stop-Loss Compliance")
        st.caption("Were stops honored? Losses much larger than the planned stop indicate stop-moving behavior — the #1 discipline failure.")
        sc_stop = build_stop_compliance(ledger)

        _ss_c1, _ss_c2, _ss_c3 = st.columns(3)
        _ss_c1.metric("Stop Honored Rate", sc_stop["honored_str"])
        _ss_c2.metric("Honored", sc_stop["honored_count"])
        _ss_c3.metric("Violated", sc_stop["violated_count"])

        if sc_stop["status"] == "RISK":
            st.error("Low stop compliance — stops are being moved or ignored. This leads to large unexpected losses.")
        elif sc_stop["status"] == "WARN":
            st.warning("Some stops were not honored. Review the violated trades.")
        elif sc_stop["status"] == "OK":
            st.success("Good stop discipline — losses are being controlled within planned limits.")

        if sc_stop["large_loss_trades"]:
            st.markdown(f"**Large loss trades (>10%):** {', '.join(sc_stop['large_loss_trades'])}")

        if not sc_stop["detail_df"].empty:
            render_badge_table(sc_stop["detail_df"], height=320)

        st.divider()
        st.subheader("Maximum Adverse Excursion (MAE)")
        st.caption("How much heat did each trade take before reaching its outcome? High MAE on winners = lucky, not skilled.")
        mae_df = build_mae_analysis(ledger)
        if mae_df.empty:
            st.info("Need closed trades with entry_price and stop_price for MAE analysis.")
        else:
            render_badge_table(mae_df, height=360)
            if _PLOTLY and "rr_ratio" in mae_df.columns:
                _mae_rr = mae_df["rr_ratio"].str.replace(":1", "").replace("—", "0").apply(lambda x: pd.to_numeric(x, errors="coerce")).fillna(0)
                _mae_colors = ["#fee2e2" if s == "RISK" else "#dbeafe" if s == "WARN" else "#dcfce7" for s in mae_df["status"]]
                _fig_mae = go.Figure(go.Bar(
                    x=mae_df["ticker"].tolist(),
                    y=_mae_rr.tolist(),
                    marker_color=_mae_colors,
                    text=[f"{v:.2f}" for v in _mae_rr],
                    textposition="outside",
                    hovertemplate="%{x}<br>R/R: %{y:.2f}<extra></extra>",
                ))
                _fig_mae.add_hline(y=1.0, line_dash="dash", line_color="#9ca3af", annotation_text="1:1 R/R")
                _fig_mae.add_hline(y=2.0, line_dash="dash", line_color="#16a34a", annotation_text="2:1 target")
                _fig_mae.update_layout(
                    height=260, margin=dict(l=10, r=10, t=28, b=20),
                    title=dict(text="Risk/Reward Ratio Per Trade (target ≥ 2:1)", font=dict(size=13), x=0),
                    yaxis_title="R/R Ratio", xaxis_title="",
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(gridcolor="#e5e7eb"), yaxis=dict(gridcolor="#e5e7eb"),
                    font=dict(family="Inter,sans-serif", size=12),
                )
                st.plotly_chart(_fig_mae, use_container_width=True)
        st.markdown("""
**MAE interpretation:**
- **Winners with R/R ≥ 2:1** → strong trades — you risked 1 to make 2+
- **Winners with R/R < 1:1** → lucky trades — you risked more than you gained; dangerous pattern
- **Losers with stop blown** → stop was moved after entry; fix this immediately
- **Rule of thumb:** If >30% of winners have R/R < 1:1, the strategy is not reproducible
""")

    # ── Opportunity Cost ──────────────────────────────────────────────────
    with sc_tabs[13]:
        st.subheader("Opportunity Cost — Missed Trades")
        st.caption("Tickers watched but never traded: what did the market do while you sat on the sidelines?")
        focus_df = build_focus_list()
        oc_df = build_opportunity_cost(focus_df, market, ledger)
        if oc_df.empty:
            st.info("Need focus list and market snapshot data to compute opportunity cost.")
        else:
            # Summary stats
            missed = oc_df[oc_df["traded"] == "No"].copy()
            if not missed.empty and "ret_20d" in oc_df.columns:
                _oc_rets = missed["ret_20d"].str.replace("%", "").apply(lambda x: pd.to_numeric(x, errors="coerce")).dropna()
                if not _oc_rets.empty:
                    _oc_avg = _oc_rets.mean()
                    _oc_pos = int((_oc_rets > 0).sum())
                    st.markdown(f"**{len(missed)} untouched tickers** — avg 20d return: {_oc_avg:.2f}% &nbsp;·&nbsp; {_oc_pos} went up")
            render_badge_table(oc_df, height=400)
            if _PLOTLY and not missed.empty:
                _oc_vals = missed["ret_20d"].str.replace("%", "").apply(lambda x: pd.to_numeric(x, errors="coerce")).fillna(0)
                _oc_colors = ["#fee2e2" if v < -0.02 else "#dbeafe" if v > 0.02 else "#f3f4f6" for v in _oc_vals]
                _fig_oc = go.Figure(go.Bar(
                    x=missed["ticker"].tolist(),
                    y=_oc_vals.tolist(),
                    marker_color=_oc_colors,
                    text=[f"{v:.1f}%" for v in _oc_vals],
                    textposition="outside",
                    hovertemplate="%{x}<br>20d return: %{y:.2f}%<extra></extra>",
                ))
                _fig_oc.add_hline(y=0, line_dash="solid", line_color="#9ca3af", line_width=1)
                _fig_oc.update_layout(
                    height=280, margin=dict(l=10, r=10, t=28, b=20),
                    title=dict(text="20-Day Return of Untouched Watch-List Tickers", font=dict(size=13), x=0),
                    yaxis_title="20d Return %", xaxis_title="",
                    yaxis=dict(gridcolor="#e5e7eb", ticksuffix="%"),
                    xaxis=dict(gridcolor="#e5e7eb"),
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Inter,sans-serif", size=12),
                )
                st.plotly_chart(_fig_oc, use_container_width=True)
        st.markdown("""
**What this tells you:**
- **Missed big winners** → your entry threshold may be too conservative — review the filter criteria
- **Missed big losers** → your risk filter worked well — discipline was correct
- **Most tickers flat** → the watch list is doing its job of eliminating noise
- **Rule:** If consistently missing >10% moves, recalibrate trigger sensitivity
""")

    # ── Signal Power ──────────────────────────────────────────────────────
    with sc_tabs[14]:
        st.subheader("Layer Signal Predictive Power")
        st.caption("Which of the 10 research layers actually correlates with profitable outcomes? Focus attention on high-signal layers.")
        master_df = read_csv(FILES["master_v2"])
        sp_df = build_layer_signal_power(ledger, master_df)
        if sp_df.empty:
            st.info("Need 5+ closed trades matched to master data to compute layer signal correlations.")
        else:
            render_badge_table(sp_df, height=400)
            if _PLOTLY and "correlation" in sp_df.columns:
                _sp_corr = sp_df["correlation"].astype(float)
                _sp_colors = ["#fee2e2" if abs(v) < 0.15 else "#dbeafe" if abs(v) < 0.30 else "#dcfce7" for v in _sp_corr]
                _fig_sp = go.Figure(go.Bar(
                    x=sp_df["layer"].tolist(),
                    y=_sp_corr.tolist(),
                    marker_color=_sp_colors,
                    text=[f"{v:.3f}" for v in _sp_corr],
                    textposition="outside",
                    hovertemplate="%{x}<br>r = %{y:.3f}<extra></extra>",
                ))
                _fig_sp.add_hline(y=0.30, line_dash="dash", line_color="#16a34a", annotation_text="Strong (r=0.30)")
                _fig_sp.add_hline(y=-0.30, line_dash="dash", line_color="#16a34a")
                _fig_sp.add_hline(y=0.15, line_dash="dot", line_color="#9ca3af", annotation_text="Moderate (r=0.15)")
                _fig_sp.add_hline(y=-0.15, line_dash="dot", line_color="#9ca3af")
                _fig_sp.update_layout(
                    height=300, margin=dict(l=10, r=10, t=28, b=20),
                    title=dict(text="Pearson Correlation: Layer Score vs Trade P&L", font=dict(size=13), x=0),
                    yaxis_title="Correlation (r)", xaxis_title="",
                    yaxis=dict(gridcolor="#e5e7eb", range=[-1, 1]),
                    xaxis=dict(gridcolor="#e5e7eb"),
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Inter,sans-serif", size=12),
                )
                st.plotly_chart(_fig_sp, use_container_width=True)
        st.markdown("""
**Interpreting correlation (r):**
- **|r| > 0.30** → strong predictive layer — trust this signal and weight it heavily
- **|r| 0.15–0.30** → moderate signal — useful in combination, not alone
- **|r| < 0.15** → weak signal — this layer may not be adding information; consider simplifying
- **Need 30+ samples** for correlations to be statistically meaningful
- **Negative r** → high score on this layer associated with worse outcomes — investigate why
""")

    # ── P&L Calendar ─────────────────────────────────────────────────────
    with sc_tabs[15]:
        st.subheader("Monthly P&L Calendar")
        st.caption("Month-by-month performance summary. Reveals seasonality, streaks, and whether losses cluster in specific periods.")
        cal_df = build_pnl_calendar(ledger)
        if cal_df.empty:
            st.info("Need closed trades with exit_date or entry_date to build P&L calendar.")
        else:
            render_badge_table(cal_df[["status","month_label","total_return_pct","n_trades","win_rate_pct"]], height=max(200, len(cal_df)*46+54))
            if _PLOTLY:
                _cal_colors = ["#fee2e2" if s == "risk" else "#dbeafe" if s == "plain" else "#dcfce7" for s in cal_df["status"]]
                _fig_cal = go.Figure(go.Bar(
                    x=cal_df["month_label"].tolist(),
                    y=cal_df["total_return_pct"].tolist(),
                    marker_color=_cal_colors,
                    text=[f"{v:+.1f}%" for v in cal_df["total_return_pct"]],
                    textposition="outside",
                    hovertemplate="%{x}<br>Return: %{y:+.2f}%<extra></extra>",
                ))
                _fig_cal.add_hline(y=0, line_dash="solid", line_color="#9ca3af", line_width=1)
                _fig_cal.update_layout(
                    height=300, margin=dict(l=10,r=10,t=28,b=40),
                    title=dict(text="Monthly Return (green = >5%, red = <-5%)", font=dict(size=13), x=0),
                    yaxis_title="Return %", xaxis_title="",
                    yaxis=dict(gridcolor="#e5e7eb", ticksuffix="%"),
                    xaxis=dict(gridcolor="#e5e7eb", tickangle=-30),
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Inter,sans-serif", size=12),
                )
                st.plotly_chart(_fig_cal, use_container_width=True)
        st.markdown("""
**What to look for:**
- **Red months clustering** → strategy struggles in certain market regimes; investigate what changed
- **All green but one blowup** → risk management failure in that month; review position sizes
- **Consistent small gains** → compounding works; this is the goal
- **High variance month-to-month** → strategy may need tighter circuit breakers
""")

    # ── Catalyst Analysis ─────────────────────────────────────────────────
    with sc_tabs[16]:
        st.subheader("Win Rate By Catalyst Type")
        st.caption("Parsed from thesis text. Reveals which trade setups actually generate positive alpha — focus research where the edge is.")
        cat_df = build_catalyst_winrate(ledger)
        if cat_df.empty:
            st.info("Need closed trades with thesis text mentioning specific catalyst types (earnings, breakout, FOMC, insider, etc.).")
        else:
            render_badge_table(cat_df, height=max(200, len(cat_df)*46+54))
            if _PLOTLY and "win_rate_pct" in cat_df.columns:
                _wr_vals = cat_df["win_rate_pct"].str.replace("%","").astype(float)
                _cat_colors = ["#fee2e2" if v < 40 else "#dbeafe" if v < 55 else "#dcfce7" for v in _wr_vals]
                _fig_cat = go.Figure(go.Bar(
                    x=cat_df["catalyst_type"].tolist(),
                    y=_wr_vals.tolist(),
                    marker_color=_cat_colors,
                    text=[f"{v:.1f}%" for v in _wr_vals],
                    textposition="outside",
                    hovertemplate="%{x}<br>Win rate: %{y:.1f}%<extra></extra>",
                ))
                _fig_cat.add_hline(y=50, line_dash="dash", line_color="#9ca3af", annotation_text="50% baseline")
                _fig_cat.update_layout(
                    height=280, margin=dict(l=10,r=10,t=28,b=20),
                    title=dict(text="Win Rate By Catalyst Category", font=dict(size=13), x=0),
                    yaxis_title="Win Rate %", xaxis_title="",
                    yaxis=dict(gridcolor="#e5e7eb", ticksuffix="%", range=[0,100]),
                    xaxis=dict(gridcolor="#e5e7eb"),
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Inter,sans-serif", size=12),
                )
                st.plotly_chart(_fig_cat, use_container_width=True)
        st.markdown("""
**How to use this:**
- **Highest win-rate catalyst** → double down on this type of research; it's where your edge is
- **Lowest win-rate catalyst** → consider eliminating this setup entirely
- **All below 50%** → strategy edge is not catalyst-specific; review the overall approach
- Note: a trade can appear in multiple categories if thesis mentions multiple catalysts
""")

    # ── VaR & Distribution ────────────────────────────────────────────────
    with sc_tabs[17]:
        st.subheader("Value at Risk & Return Distribution")
        st.caption("Historical VaR at 95% and 99% confidence. The P&L distribution shape reveals whether losses are fat-tailed.")
        var = build_var(pm)
        _v_c1, _v_c2, _v_c3, _v_c4 = st.columns(4)
        _v_c1.metric("VaR 95%",  var["var_95"],  help="Worst loss exceeded only 5% of the time")
        _v_c2.metric("VaR 99%",  var["var_99"],  help="Worst loss exceeded only 1% of the time")
        _v_c3.metric("CVaR 95%", var["cvar_95"], help="Average loss when VaR 95% is breached")
        _v_c4.metric("Skewness", var["skewness"],help="Negative = fat left tail (more bad surprises)")
        if var["status"] == "risk":
            st.error(f"High VaR: 95% VaR = {var['var_95']} — losses can be significant. Review position sizing.")
        elif var["status"] == "ok":
            st.success(f"VaR within acceptable range ({var['var_95']} at 95% confidence).")
        if _PLOTLY and var.get("pnl_series") is not None and not var["pnl_series"].empty:
            _pnl_arr = (var["pnl_series"] * 100).tolist()
            _fig_dist = go.Figure()
            _fig_dist.add_trace(go.Histogram(
                x=_pnl_arr, nbinsx=max(10, len(_pnl_arr)//3),
                marker_color="#dbeafe", marker_line_color="#1e40af", marker_line_width=1,
                name="Trade Returns",
                hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
            ))
            try:
                var95_line = float(var["var_95"].replace("%",""))
                _fig_dist.add_vline(x=var95_line, line_dash="dash", line_color="#991b1b",
                                    annotation_text=f"VaR 95%: {var['var_95']}", annotation_position="top left")
            except Exception:
                pass
            _fig_dist.add_vline(x=0, line_dash="solid", line_color="#9ca3af", line_width=1)
            _fig_dist.update_layout(
                height=280, margin=dict(l=10,r=10,t=28,b=20),
                title=dict(text=f"Trade P&L Distribution (n={var['n']})", font=dict(size=13), x=0),
                xaxis_title="P&L %", yaxis_title="Frequency",
                xaxis=dict(gridcolor="#e5e7eb", ticksuffix="%"),
                yaxis=dict(gridcolor="#e5e7eb"),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Inter,sans-serif", size=12),
                bargap=0.05,
            )
            st.plotly_chart(_fig_dist, use_container_width=True)
        var_table = pd.DataFrame([
            {"status":"plain","metric":"VaR 95% (Historical)","value":var["var_95"],"interpretation":"Worst loss exceeded only 5% of trades"},
            {"status":"plain","metric":"VaR 99% (Historical)","value":var["var_99"],"interpretation":"Worst loss exceeded only 1% of trades"},
            {"status":"plain","metric":"CVaR 95% (Exp. Shortfall)","value":var["cvar_95"],"interpretation":"Avg loss when you breach VaR 95%"},
            {"status":"plain","metric":"Best Trade","value":var["best_trade"],"interpretation":"Single best outcome"},
            {"status":"risk" if var["var_95"]!="N/A" else "plain","metric":"Worst Trade","value":var["worst_trade"],"interpretation":"Single worst outcome"},
            {"status":"plain","metric":"Skewness","value":var["skewness"],"interpretation":"Negative = fat left tail; target > 0 (positive skew)"},
            {"status":"plain","metric":"Excess Kurtosis","value":var["kurtosis"],"interpretation":"> 0 = fat tails; tail risk larger than normal distribution"},
        ])
        render_badge_table(var_table, height=380)
        st.markdown("""
**VaR interpretation:**
- **VaR 95% = -5%** → 95% of trades lose less than 5%; 5% of the time you lose more
- **CVaR (Expected Shortfall)** → on the bad days, how bad is it on average? More important than VaR alone
- **Negative skewness** → the strategy has fat left tail — occasional large losses; consider tighter stops
- **Target:** VaR 95% < -8%, CVaR 95% < -12%, Skewness > -0.5
""")

    # ── Trade Frequency ───────────────────────────────────────────────────
    with sc_tabs[18]:
        st.subheader("Trade Frequency & Turnover Monitor")
        st.caption("Professional target: 2-8 new paper trades per month. Over-trading signals noise-driven decisions.")
        tf = build_trade_frequency(ledger)
        _tf_c1, _tf_c2, _tf_c3 = st.columns(3)
        _tf_c1.metric("Avg Trades/Month", tf["avg_trades_str"])
        _tf_c2.metric("Total Trades", tf["total_trades"])
        _tf_c3.metric("Avg Days Between", f"{tf['avg_days_between']:.1f}d" if tf["avg_days_between"] else "N/A")
        if tf["status"] == "risk":
            st.error(f"{tf['turnover_note']}")
        elif tf["status"] == "ok":
            st.success(f"{tf['turnover_note']}")
        else:
            st.info(tf["turnover_note"])
        if not tf["monthly_df"].empty:
            render_badge_table(tf["monthly_df"], height=max(200, len(tf["monthly_df"])*46+54))
            if _PLOTLY:
                _tf_df = tf["monthly_df"]
                _tf_colors = ["#fee2e2" if s == "risk" else "#dcfce7" if s == "ok" else "#dbeafe" for s in _tf_df["status"]]
                _fig_tf = go.Figure(go.Bar(
                    x=_tf_df["month_label"].tolist(),
                    y=_tf_df["n_trades"].tolist(),
                    marker_color=_tf_colors,
                    text=_tf_df["n_trades"].tolist(),
                    textposition="outside",
                    hovertemplate="%{x}<br>Trades: %{y}<extra></extra>",
                ))
                _fig_tf.add_hline(y=8, line_dash="dash", line_color="#991b1b", annotation_text="Max (8)")
                _fig_tf.add_hline(y=2, line_dash="dash", line_color="#15803d", annotation_text="Min (2)")
                _fig_tf.update_layout(
                    height=280, margin=dict(l=10,r=10,t=28,b=40),
                    title=dict(text="Trades Per Month (target: 2-8)", font=dict(size=13), x=0),
                    yaxis_title="Number of Trades", xaxis_title="",
                    yaxis=dict(gridcolor="#e5e7eb"),
                    xaxis=dict(gridcolor="#e5e7eb", tickangle=-30),
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Inter,sans-serif", size=12),
                )
                st.plotly_chart(_fig_tf, use_container_width=True)

        st.divider()
        st.subheader("Rolling Sharpe — Edge Decay Monitor")
        st.caption("Rolling 20-trade Sharpe ratio. Declining trend = strategy edge is weakening.")
        _n_closed_rs = int((ledger["status"].astype(str).str.upper() == "CLOSED").sum()) if "status" in ledger.columns else 0
        if _n_closed_rs < 20:
            st.info(f"Rolling Sharpe needs 20 closed trades — you have {_n_closed_rs}. First full window will appear after {20 - _n_closed_rs} more closed positions.")
        rs_df = build_rolling_sharpe(ledger)
        if rs_df.empty or rs_df["rolling_sharpe"].dropna().empty:
            st.info("Need 5+ closed trades to compute rolling Sharpe.")
        else:
            valid_rs = rs_df.dropna(subset=["rolling_sharpe"])
            last_sharpe = float(valid_rs["rolling_sharpe"].iloc[-1]) if not valid_rs.empty else None
            first_sharpe = float(valid_rs["rolling_sharpe"].iloc[0]) if len(valid_rs) > 1 else None
            if last_sharpe is not None and first_sharpe is not None:
                trend = "improving" if last_sharpe > first_sharpe else "declining"
                if trend == "declining" and last_sharpe < 0:
                    st.error(f"Rolling Sharpe is negative ({last_sharpe:.2f}) and declining — strategy edge may have deteriorated.")
                elif trend == "declining":
                    st.warning(f"Rolling Sharpe trending down: {first_sharpe:.2f} → {last_sharpe:.2f}. Monitor closely.")
                else:
                    st.success(f"Rolling Sharpe improving: {first_sharpe:.2f} → {last_sharpe:.2f}.")
            if _PLOTLY:
                _rs_valid = rs_df.dropna(subset=["rolling_sharpe"])
                _rs_colors = ["#fee2e2" if v < 0 else "#dcfce7" if v > 1.0 else "#dbeafe" for v in _rs_valid["rolling_sharpe"]]
                _fig_rs = go.Figure()
                _fig_rs.add_trace(go.Scatter(
                    x=_rs_valid["trade_num"].tolist(),
                    y=_rs_valid["rolling_sharpe"].tolist(),
                    mode="lines+markers",
                    line=dict(color="#1e40af", width=2),
                    marker=dict(color=_rs_colors, size=6),
                    name="Rolling Sharpe",
                    hovertemplate="Trade %{x}<br>Sharpe: %{y:.3f}<extra></extra>",
                ))
                _fig_rs.add_hline(y=1.0, line_dash="dash", line_color="#15803d", annotation_text="Target (1.0)")
                _fig_rs.add_hline(y=0, line_dash="solid", line_color="#991b1b", line_width=1)
                _fig_rs.update_layout(
                    height=260, margin=dict(l=10,r=10,t=28,b=20),
                    title=dict(text="Rolling 20-Trade Sharpe Ratio", font=dict(size=13), x=0),
                    xaxis_title="Trade #", yaxis_title="Sharpe Ratio",
                    yaxis=dict(gridcolor="#e5e7eb"),
                    xaxis=dict(gridcolor="#e5e7eb"),
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Inter,sans-serif", size=12),
                )
                st.plotly_chart(_fig_rs, use_container_width=True)

        st.divider()
        st.subheader("ADV Liquidity Check")
        st.caption("Position dollar size vs average daily volume. Positions > 10% of ADV face real market impact in live trading.")
        _adv_tech = read_csv(FILES["technicals"])
        _acct_size_input = st.number_input(
            "Account Size ($)", min_value=10_000, max_value=100_000_000,
            value=int(st.session_state.get("account_size", 100_000)),
            step=10_000, key="adv_account_size",
            help="Used to estimate your position dollar size. Adjust to match your actual paper portfolio size."
        )
        st.session_state["account_size"] = float(_acct_size_input)
        adv_df = build_adv_check(exposure, _adv_tech, account_size=float(_acct_size_input))
        if adv_df.empty:
            st.info("Need exposure data and technicals with volume/price columns.")
        else:
            _adv_risk = len(adv_df[adv_df["status"] == "risk"])
            if _adv_risk > 0:
                st.warning(f"{_adv_risk} positions exceed 10% of estimated ADV — paper prices may not reflect true execution cost.")
            render_badge_table(adv_df, height=max(200, len(adv_df)*46+54))
        st.markdown("""
**ADV liquidity rule:** `position_dollar <= 10% x average_daily_volume_dollar`
**Why it matters:** At paper stage this doesn't apply, but establishing the habit of checking liquidity prevents scaling into illiquid names. A $10k position in a $50k ADV stock = 20% of daily volume — real execution would move the price against you.
""")

    # ── Multi Benchmark ───────────────────────────────────────────────────
    with sc_tabs[19]:
        st.subheader("Multi-Benchmark Comparison")
        st.caption("Alpha vs SPY, QQQ, IWM, Gold, and a 60/40 portfolio. True alpha = outperforming all benchmarks, not just the one you pick.")
        mb_df = build_multi_benchmark(pm, market)
        if mb_df.empty:
            st.info("Need closed trades and market snapshot with SPY/QQQ/IWM data.")
        else:
            render_badge_table(mb_df, height=max(200, len(mb_df)*46+54))
            if _PLOTLY and "benchmark_20d" in mb_df.columns and "alpha" in mb_df.columns:
                _mb_benchmarks = mb_df["benchmark"].tolist()
                _mb_bench_rets = mb_df["benchmark_20d"].str.replace("%","").apply(lambda x: pd.to_numeric(x, errors="coerce")).fillna(0)
                _mb_alphas = mb_df["alpha"].str.replace("%","").apply(lambda x: pd.to_numeric(x, errors="coerce")).fillna(0)
                _mb_colors = ["#fee2e2" if v < -0.02 else "#dcfce7" if v > 0.02 else "#dbeafe" for v in _mb_alphas]
                _fig_mb = go.Figure()
                _fig_mb.add_trace(go.Bar(
                    x=_mb_benchmarks, y=_mb_bench_rets.tolist(),
                    name="Benchmark 20d Return", marker_color="#e5e7eb",
                    hovertemplate="%{x}<br>Benchmark: %{y:.2f}%<extra></extra>",
                ))
                _fig_mb.add_trace(go.Bar(
                    x=_mb_benchmarks, y=_mb_alphas.tolist(),
                    name="Alpha vs Benchmark", marker_color=_mb_colors,
                    hovertemplate="%{x}<br>Alpha: %{y:.2f}%<extra></extra>",
                ))
                _fig_mb.update_layout(
                    barmode="group",
                    height=280, margin=dict(l=10,r=10,t=28,b=40),
                    title=dict(text="Strategy Alpha vs Multiple Benchmarks (20-day window)", font=dict(size=13), x=0),
                    yaxis_title="Return %", xaxis_title="",
                    yaxis=dict(gridcolor="#e5e7eb", ticksuffix="%"),
                    xaxis=dict(gridcolor="#e5e7eb", tickangle=-20),
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Inter,sans-serif", size=12),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(_fig_mb, use_container_width=True)
        st.markdown("""
**Why multi-benchmark matters:**
- **Beat SPY but lag QQQ** → you're running a tech-momentum strategy in disguise; not true alpha
- **Beat all benchmarks** → genuine alpha, not just factor exposure
- **Lag all benchmarks** → reconsider the strategy's core thesis
- **Alpha vs 60/40** → the true test for any active strategy vs passive alternatives
- Note: 20-day window is short — meaningful comparison needs 6-12 months of consistent data
""")


def tab_ticker_drilldown():
    st.header("Single-Ticker Notebook")

    master = read_csv(FILES["master_v2"])
    cards = read_csv(FILES["action_cards"])
    triggers = read_csv(FILES["watch_triggers"])
    pretrade = read_csv(FILES["pre_trade"])
    options = read_csv(FILES["options_decision"])
    v8_gate = read_csv(FILES["v8_l9_gate"])
    technicals = read_csv(FILES["technicals"])
    events = read_csv(FILES["events"])
    market = read_csv(FILES["market_snapshot"])
    fundamentals = read_csv(FILES["fundamentals"])
    sectors = read_csv(FILES["sector_scores"])
    ledger = read_csv(FILES["paper_ledger"])
    gaps = build_gap_queue(master, market)

    if master.empty or "ticker" not in master.columns:
        st.warning("Run Step 54 first.")
        return

    tickers = master["ticker"].astype(str).tolist()
    # Check if Home page pre-selected a ticker
    _preselect = st.session_state.get("home_active_ticker", None)
    default_idx = 0
    _priority_list = ([_preselect] if _preselect else []) + ["SOXX", "SMH", "QQQ", "SPY"]
    for preferred in _priority_list:
        if preferred in tickers:
            default_idx = tickers.index(preferred)
            break

    ticker = st.selectbox("Choose Ticker", tickers, index=default_idx, key="notebook_ticker_select")
    row = first_row(master, ticker)
    card = first_row(cards, ticker)
    trigger = first_row(triggers, ticker)
    check = first_row(pretrade, ticker)
    option = first_row(options, ticker)
    v8_check = first_row(v8_gate, ticker)
    fund_row = first_row(fundamentals, ticker)
    sector_row = first_row(sectors, ticker)
    ledger_rows = ticker_rows(ledger, ticker) if not ledger.empty else pd.DataFrame()
    tech = first_row(technicals, ticker)
    event = first_row(events, ticker)
    market_row = first_row(market, ticker)
    ticker_gaps = ticker_rows(gaps, ticker)

    action = row.get("master_action", "NO_DATA")
    reason = row.get("master_reason", "")
    score = row.get("stack_score_avg", "")
    min_score = row.get("stack_score_min", "")

    st.markdown(f"""
    <div class="ticker-hero ticker-{status_kind(action)}">
      <div>
        <div class="ticker-label">Final Decision</div>
        <div class="ticker-title">{escape(ticker)} · {escape(str(action))}</div>
        <div class="ticker-reason">{escape(str(reason))}</div>
      </div>
      <div class="ticker-scorebox">
        <div class="ticker-score-label">Average Score / Lowest Score</div>
        <div class="ticker-score">{escape(str(score))} / {escape(str(min_score))}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    dossier_tabs = st.tabs(["Notebook", "10-Layer Evidence", "Is Action Allowed", "Options And Risk", "Missing Data", "What Would Change", "Quick Compare"])

    with dossier_tabs[0]:
        # ── Momentum & Technical Snapshot ────────────────────────────────────
        if _PLOTLY and tech:
            _snap_cols = st.columns([3, 2, 2])
            with _snap_cols[0]:
                # Return bars: 5d / 20d / 63d vs SPY
                _spy_tech = first_row(technicals, "SPY")
                _periods_map = [("ret_5d", "5d"), ("ret_20d", "20d"), ("ret_63d", "63d")]
                _ret_labels, _ret_vals, _spy_vals = [], [], []
                for _col, _lbl in _periods_map:
                    _tv = pd.to_numeric(tech.get(_col, None), errors="coerce")
                    _sv = pd.to_numeric(_spy_tech.get(_col, None), errors="coerce")
                    _ret_labels.append(_lbl)
                    _ret_vals.append(float(_tv) * 100 if pd.notna(_tv) else 0)
                    _spy_vals.append(float(_sv) * 100 if pd.notna(_sv) else 0)
                _fig_snap = go.Figure()
                _fig_snap.add_trace(go.Bar(
                    name=ticker, x=_ret_labels, y=_ret_vals,
                    marker_color=["#16a34a" if v >= 0 else "#dc2626" for v in _ret_vals],
                    text=[f"{v:+.1f}%" for v in _ret_vals], textposition="outside",
                ))
                _fig_snap.add_trace(go.Scatter(
                    name="SPY", x=_ret_labels, y=_spy_vals,
                    mode="markers+lines",
                    marker=dict(color="#9ca3af", size=8),
                    line=dict(color="#9ca3af", width=1.5, dash="dot"),
                ))
                _fig_snap.add_hline(y=0, line_width=1, line_color="#e5e7eb")
                _fig_snap.update_layout(
                    height=220, margin=dict(l=4, r=12, t=28, b=8),
                    title=dict(text="Returns vs SPY", font=dict(size=12), x=0),
                    yaxis=dict(ticksuffix="%", gridcolor="#e5e7eb", zerolinecolor="#d1d5db"),
                    xaxis=dict(gridcolor="#e5e7eb"),
                    plot_bgcolor="white", paper_bgcolor="white",
                    legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="left", x=0, font=dict(size=11)),
                    font=dict(family="Inter,sans-serif", size=12),
                )
                st.plotly_chart(_fig_snap, use_container_width=True)

            with _snap_cols[1]:
                # RSI gauge (horizontal bar 0-100)
                _rsi = float(pd.to_numeric(tech.get("rsi14", 50), errors="coerce") or 50)
                _rsi_color = "#dc2626" if _rsi > 70 else ("#16a34a" if _rsi < 30 else "#2563eb")
                _fig_rsi = go.Figure(go.Bar(
                    x=[_rsi], y=["RSI"], orientation="h",
                    marker_color=_rsi_color,
                    text=[f"RSI {_rsi:.1f}"], textposition="inside",
                    textfont=dict(color="white", size=12),
                ))
                _fig_rsi.add_vline(x=70, line_width=1, line_color="#dc2626", line_dash="dot")
                _fig_rsi.add_vline(x=30, line_width=1, line_color="#16a34a", line_dash="dot")
                _fig_rsi.add_vline(x=50, line_width=1, line_color="#9ca3af")
                _fig_rsi.update_layout(
                    height=100, margin=dict(l=4, r=12, t=28, b=8),
                    title=dict(text="RSI 14", font=dict(size=12), x=0),
                    xaxis=dict(range=[0, 100], gridcolor="#e5e7eb"),
                    yaxis=dict(showticklabels=False),
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Inter,sans-serif", size=12),
                )
                st.plotly_chart(_fig_rsi, use_container_width=True)

                # Volume Z-score
                _volz = float(pd.to_numeric(tech.get("volume_z60", 0), errors="coerce") or 0)
                _volz_color = "#16a34a" if _volz > 0.5 else ("#dc2626" if _volz < -0.5 else "#9ca3af")
                _fig_volz = go.Figure(go.Bar(
                    x=[_volz], y=["Vol Z"], orientation="h",
                    marker_color=_volz_color,
                    text=[f"Vol Z {_volz:+.2f}σ"], textposition="outside" if abs(_volz) < 0.5 else "inside",
                    textfont=dict(color="white" if abs(_volz) >= 0.5 else "#374151", size=12),
                ))
                _fig_volz.add_vline(x=0, line_width=1, line_color="#9ca3af")
                _fig_volz.update_layout(
                    height=100, margin=dict(l=4, r=12, t=28, b=8),
                    title=dict(text="Volume vs 60d Avg", font=dict(size=12), x=0),
                    xaxis=dict(range=[min(-2, _volz-0.2), max(2, _volz+0.2)], gridcolor="#e5e7eb"),
                    yaxis=dict(showticklabels=False),
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Inter,sans-serif", size=12),
                )
                st.plotly_chart(_fig_volz, use_container_width=True)

            with _snap_cols[2]:
                # MA position pills + tech score
                _above_20 = str(tech.get("above_20dma", "")).lower() == "true"
                _above_50 = str(tech.get("above_50dma", "")).lower() == "true"
                _above_100 = str(tech.get("above_100dma", "")).lower() == "true"
                _tech_score = tech.get("technical_score", "N/A")
                _atr_pct = pd.to_numeric(tech.get("atr14_pct", None), errors="coerce")
                _close = pd.to_numeric(tech.get("close", None), errors="coerce")
                def _pill(label, ok):
                    bg = "#dcfce7" if ok else "#fee2e2"
                    fg = "#14532d" if ok else "#7f1d1d"
                    sym = "▲" if ok else "▼"
                    return (f'<span style="display:inline-block;padding:3px 10px;background:{bg};'
                            f'color:{fg};border-radius:3px;font-size:11px;font-weight:600;'
                            f'margin:2px 4px 2px 0;">{sym} {label}</span>')
                st.markdown(
                    f'<div style="font-size:10px;font-weight:700;letter-spacing:.10em;'
                    f'text-transform:uppercase;color:#6b7280;margin-bottom:8px;">MA Position</div>'
                    + _pill("Above 20 DMA", _above_20)
                    + _pill("Above 50 DMA", _above_50)
                    + _pill("Above 100 DMA", _above_100)
                    + f'<div style="margin-top:14px;font-size:10px;font-weight:700;'
                    f'letter-spacing:.10em;text-transform:uppercase;color:#6b7280;">Tech Score</div>'
                    + f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:28px;'
                    f'font-weight:700;color:#111827;margin:4px 0 8px 0;">{_tech_score}</div>'
                    + (f'<div style="font-size:12px;color:#6b7280;">ATR 14: '
                       f'{float(_atr_pct)*100:.2f}% of price</div>' if pd.notna(_atr_pct) else "")
                    + (f'<div style="font-size:12px;color:#374151;font-weight:600;margin-top:4px;">'
                       f'Close: ${float(_close):.2f}</div>' if pd.notna(_close) else ""),
                    unsafe_allow_html=True,
                )

        st.subheader("What This Ticker Means")
        intelligence = build_ticker_intelligence_panel(
            ticker, row, card, trigger, check, option, v8_check, tech, event, market_row, ticker_gaps
        )
        display_cols = [c for c in [
            "status", "layer", "signal", "score", "source_type", "source_file", "operator_read"
        ] if c in intelligence.columns]
        render_badge_table(intelligence[display_cols], height=520)
        render_ticker_source_trail(intelligence)

        st.markdown(build_ticker_decision_narrative(
            ticker, row, card, trigger, check, option, v8_check, tech, event, ticker_gaps
        ))

        st.subheader("Short Note For Yourself")
        memo = build_ticker_memo(
            ticker, row, card, trigger, check, v8_check, ticker_gaps
        )
        render_badge_table(memo, height=300)

        st.subheader("Decision Evidence List")
        dossier = build_ticker_dossier(
            ticker, row, card, trigger, check, option, v8_check, tech, event, ticker_gaps
        )
        render_badge_table(dossier, height=360)

        st.subheader("One-Line Action Rule")
        rule_text = row_value(card, "one_liner", default=row_value(row, "master_reason", default="Research only."))
        st.markdown(f"""
        <div class="ticker-rule ticker-{status_kind(action)}">
          <div class="ticker-label">{escape(ticker)} Current Rule</div>
          <div class="ticker-rule-text">{escape(rule_text)}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── L4 Fundamentals & Valuation ──────────────────────────────────────
        st.subheader("Fundamentals & Valuation (L4)")
        if fund_row:
            def _vs(pe):
                try:
                    pe = float(pe)
                    if pe < 15: return "Cheap"
                    if pe <= 28: return "Fair"
                    return "Expensive"
                except Exception:
                    return "No PE Data"

            fpe = fund_row.get("forward_pe", "")
            tpe = fund_row.get("trailing_pe", "")
            val_sig = _vs(fpe) if fpe not in ("", None) else _vs(tpe)
            fund_label = str(fund_row.get("fundamental_label", ""))
            if "ETF_NOT_FUNDAMENTAL" in fund_label:
                val_sig = "ETF Context"

            _fa, _fb, _fc, _fd = st.columns(4)
            _fa.metric("Val Signal", val_sig)
            _fb.metric("Forward PE", f"{float(fpe):.1f}" if fpe not in ("", None) else "N/A")
            _fc.metric("Quality Score", fund_row.get("quality_score", "N/A"))
            _fd.metric("Current Price", f"${float(fund_row.get('current_price', 0)):.2f}" if fund_row.get("current_price") else "N/A")

            _fe, _ff, _fg, _fh = st.columns(4)
            _rev = fund_row.get("revenue_growth", "")
            _mgn = fund_row.get("gross_margin", "")
            _opm = fund_row.get("operating_margin", "")
            _dte = fund_row.get("debt_to_equity", "")
            _fe.metric("Revenue Growth", f"{float(_rev)*100:.1f}%" if _rev not in ("", None) else "N/A")
            _ff.metric("Gross Margin", f"{float(_mgn)*100:.1f}%" if _mgn not in ("", None) else "N/A")
            _fg.metric("Operating Margin", f"{float(_opm)*100:.1f}%" if _opm not in ("", None) else "N/A")
            _fh.metric("Debt / Equity", f"{float(_dte):.1f}x" if _dte not in ("", None) else "N/A")

            _fund_note = fund_label.split(":")[1].strip() if ":" in fund_label else fund_label
            if _fund_note:
                st.caption(f"L4 label: **{_fund_note[:220]}**")
        else:
            st.info("No fundamentals data for this ticker. Run Step 54 to populate L4.")

        # ── Sector Rotation Context ───────────────────────────────────────────
        st.subheader("Sector Rotation Context (L3)")
        if sector_row:
            _ROT_BADGE = {"LEADER": "🟢", "WATCH": "🟡", "NEUTRAL": "⚪", "LAGGARD": "🔴"}
            rot_label = str(sector_row.get("rotation_label", "N/A")).upper()
            rot_score = sector_row.get("rotation_score", "N/A")
            theme = sector_row.get("theme", "N/A")
            ret_5d = sector_row.get("ret_5d", "")
            ret_20d = sector_row.get("ret_20d", "")
            rel_spy = sector_row.get("relative_20d_vs_spy", "")
            _sa, _sb, _sc, _sd = st.columns(4)
            _sa.metric("Theme / Sector", theme)
            _sb.metric("Rotation Label", f"{_ROT_BADGE.get(rot_label, '')} {rot_label}")
            _sc.metric("Rotation Score", f"{float(rot_score):.1f}" if rot_score not in ("", None, "N/A") else "N/A")
            _sd.metric("Rel vs SPY 20d", f"{float(rel_spy)*100:+.1f}%" if rel_spy not in ("", None) else "N/A")
        else:
            st.info("No sector rotation row for this ticker.")

        # ── Paper Trade History ───────────────────────────────────────────────
        st.subheader("Paper Trade History")
        if not ledger_rows.empty:
            closed_lr = ledger_rows[ledger_rows.get("status", pd.Series()).astype(str).str.upper().isin(["CLOSED_PAPER", "CLOSED_REAL"])] if "status" in ledger_rows.columns else pd.DataFrame()
            open_lr = ledger_rows[~ledger_rows.index.isin(closed_lr.index)] if not closed_lr.empty else ledger_rows

            if not closed_lr.empty:
                st.markdown("**Closed Paper Trades**")
                _cl_cols = [c for c in ["trade_id", "sleeve", "status", "entry_date", "entry_price",
                                        "exit_date", "exit_price", "pnl_pct", "holding_days", "notes"]
                            if c in closed_lr.columns]
                render_badge_table(closed_lr[_cl_cols])

                if _PLOTLY and "pnl_pct" in closed_lr.columns:
                    _pnl = pd.to_numeric(closed_lr["pnl_pct"], errors="coerce").dropna() * 100
                    if not _pnl.empty:
                        _fig_pnl = go.Figure(go.Bar(
                            x=list(range(len(_pnl))),
                            y=_pnl.tolist(),
                            marker_color=["#16a34a" if v >= 0 else "#dc2626" for v in _pnl],
                            text=[f"{v:+.2f}%" for v in _pnl],
                            textposition="outside",
                        ))
                        _fig_pnl.update_layout(
                            height=200, margin=dict(l=10, r=10, t=10, b=30),
                            xaxis_title="Trade #", yaxis_title="Return %",
                            plot_bgcolor="white", paper_bgcolor="white",
                            xaxis=dict(gridcolor="#e5e7eb"), yaxis=dict(gridcolor="#e5e7eb"),
                            font=dict(family="Inter,sans-serif", size=12),
                        )
                        st.plotly_chart(_fig_pnl, use_container_width=True)

            if not open_lr.empty:
                st.markdown("**Open / Watch**")
                _op_cols = [c for c in ["trade_id", "sleeve", "status", "suggested_action",
                                        "effective_weight", "risk_note", "thesis"]
                            if c in open_lr.columns]
                render_badge_table(open_lr[_op_cols])
        else:
            st.info("No paper ledger rows for this ticker yet.")

    with dossier_tabs[1]:
        st.subheader("10-Layer Evidence Map")
        layer_html = ['<div class="layer-grid">']
        for i in range(1, 11):
            prefix = f"L{i}"
            layer_html.append(layer_card_html(
                prefix,
                row.get(f"{prefix}_state", "NO_DATA"),
                row.get(f"{prefix}_score", ""),
                row.get(f"{prefix}_note", ""),
            ))
        layer_html.append("</div>")
        st.markdown("".join(layer_html), unsafe_allow_html=True)

        st.subheader("What Each Layer Says")
        render_badge_table(build_layer_thesis_stack(row), height=460)

    with dossier_tabs[2]:
        st.subheader("Action Limits")
        detail_cols = st.columns(3)
        with detail_cols[0]:
            st.markdown("**Allowed Action**")
            st.write(card.get("allowed_action", v8_check.get("paper_allowed", "No action data.")))
        with detail_cols[1]:
            st.markdown("**Forbidden Action**")
            st.write(card.get("forbidden_action", "No action data."))
        with detail_cols[2]:
            st.markdown("**When To Recheck**")
            st.write(card.get("trigger_rule", "No trigger data."))

        compact = pd.DataFrame([
            {
                "ticker": ticker,
                "spot": card.get("spot", trigger.get("spot", "")),
                "breakout_trigger": card.get("breakout_trigger", trigger.get("call_wall_breakout_trigger", "")),
                "breakdown_trigger": card.get("breakdown_trigger", trigger.get("put_wall_breakdown_trigger", "")),
                "gamma_label": card.get("gamma_label", trigger.get("gamma_label", "")),
                "kill_zone_label": card.get("kill_zone_label", trigger.get("kill_zone_label", "")),
                "risk_light": check.get("risk_light", row.get("L8_state", "")),
                "final_status": check.get("final_status", v8_check.get("final_status", row.get("master_action", ""))),
                "live_allowed": card.get("live_allowed", check.get("live_allowed", v8_check.get("live_allowed", ""))),
            }
        ])
        render_badge_table(compact, height=140)

        v8_rows = ticker_rows(v8_gate, ticker)
        if not v8_rows.empty:
            st.subheader("Old Check Result")
            v8_cols = [c for c in [
                "ticker", "decision", "risk_light", "suggested_action", "paper_allowed",
                "live_allowed", "final_status", "reasons", "sizing_reason"
            ] if c in v8_rows.columns]
            render_badge_table(v8_rows[v8_cols], height=220)

    with dossier_tabs[3]:
        st.subheader("Options Layer Decision")
        option_rows = ticker_rows(options, ticker)
        if option_rows.empty:
            st.info("No options decision row for this ticker.")
        else:
            option_cols = [c for c in [
                "ticker", "spot", "gamma_squeeze_label", "option_kill_zone_label",
                "pretrade_status", "portfolio_risk_light", "paper_allowed",
                "live_allowed", "final_options_decision", "rule", "explanation"
            ] if c in option_rows.columns]
            render_badge_table(option_rows[option_cols], height=260)

        st.subheader("Price Trend And Event Evidence")
        evidence_rows = []
        if tech:
            evidence_rows.append({
                "status": tech.get("technical_label", ""),
                "section": "L6 Technical",
                "score": tech.get("technical_score", ""),
                "evidence": tech.get("reasons", ""),
            })
        if event:
            evidence_rows.append({
                "status": event.get("event_label", ""),
                "section": "L5 Event",
                "score": event.get("event_score", ""),
                "evidence": event.get("reasons", ""),
            })
        render_badge_table(pd.DataFrame(evidence_rows), height=180)

    with dossier_tabs[4]:
        st.subheader("Ticker Missing Data")
        if ticker_gaps.empty:
            st.success("No ticker-specific gap rows in the current queue.")
        else:
            cols = [c for c in [
                "priority", "lane", "ticker", "gap_type", "layer", "state",
                "impact", "next_fix", "note"
            ] if c in ticker_gaps.columns]
            render_badge_table(ticker_gaps[cols], height=420)

        market_rows = ticker_rows(market, ticker)
        if not market_rows.empty:
            st.subheader("Price Data Snapshot")
            render_badge_table(market_rows, height=180)

    with dossier_tabs[5]:
        st.subheader("What Would Change This Decision")
        st.caption(
            "This shows what conditions would upgrade or downgrade the current research conclusion. "
            "It does not mean these conditions are expected — it means these are the things to watch."
        )
        change_table = build_what_would_change(ticker, row, check, v8_check, ticker_gaps, tech)
        render_badge_table(change_table, height=460)

        st.markdown("""
**How to read this:**
- **UPGRADE** rows show what would allow more expression (e.g., paper trade or higher confidence).
- **DOWNGRADE** rows show what would force the decision back to Research Only.
- Nothing here is a trade instruction — these are conditions to monitor, not triggers to act on automatically.
- Live orders are never allowed regardless of which conditions are met.
""")

    # ── Quick Compare ─────────────────────────────────────────────────────────
    with dossier_tabs[6]:
        st.subheader("Quick Compare — Up To 5 Tickers")
        _all_tickers = master["ticker"].astype(str).tolist() if not master.empty else []
        _default_compare = [ticker]
        for _t in ["SPY", "QQQ", "SOXX"]:
            if _t in _all_tickers and _t not in _default_compare and len(_default_compare) < 3:
                _default_compare.append(_t)
        _compare_sel = st.multiselect(
            "Select tickers to compare (max 5)",
            options=_all_tickers,
            default=[t for t in _default_compare if t in _all_tickers],
            max_selections=5,
            key="qc_multiselect",
        )
        if not _compare_sel:
            st.info("Select at least one ticker to compare.")
        else:
            _tech_all = read_csv(FILES["technicals"])
            _fund_all = read_csv(FILES["fundamentals"])
            _sec_all  = read_csv(FILES["sector_scores"])

            # ── Momentum chart: grouped bars ret_5d / ret_20d / ret_63d ──────
            if _PLOTLY and not _tech_all.empty:
                st.markdown("**Return Profile (5d / 20d / 63d)**")
                _periods = ["ret_5d", "ret_20d", "ret_63d"]
                _period_labels = ["5-Day", "20-Day", "63-Day"]
                _MOM_COLORS = ["#22d3ee", "#2563eb", "#7c3aed"]
                _fig_mom = go.Figure()
                for _pi, (_col, _plbl, _col_c) in enumerate(zip(_periods, _period_labels, _MOM_COLORS)):
                    _xs, _ys, _texts, _bar_colors = [], [], [], []
                    for _t in _compare_sel:
                        _tr = first_row(_tech_all, _t)
                        _val = pd.to_numeric(_tr.get(_col, None), errors="coerce")
                        _xs.append(_t)
                        _ys.append(float(_val) * 100 if pd.notna(_val) else 0)
                        _texts.append(f"{float(_val)*100:+.1f}%" if pd.notna(_val) else "N/A")
                        _bar_colors.append("#16a34a" if (pd.notna(_val) and _val > 0) else "#dc2626")
                    _fig_mom.add_trace(go.Bar(
                        name=_plbl,
                        x=_xs,
                        y=_ys,
                        marker_color=_col_c,
                        text=_texts,
                        textposition="outside",
                        offsetgroup=_pi,
                    ))
                _fig_mom.add_hline(y=0, line_width=1, line_color="#9ca3af")
                _fig_mom.update_layout(
                    barmode="group",
                    height=340,
                    margin=dict(l=10, r=10, t=24, b=20),
                    yaxis_title="Return %", yaxis_ticksuffix="%",
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(gridcolor="#e5e7eb"),
                    yaxis=dict(gridcolor="#e5e7eb", zerolinecolor="#9ca3af"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    font=dict(family="Inter,sans-serif", size=13),
                )
                st.plotly_chart(_fig_mom, use_container_width=True)

            # ── Technical score comparison bar ────────────────────────────────
            if _PLOTLY and not _tech_all.empty:
                _tscore_x, _tscore_y, _tscore_rsi, _tscore_vol = [], [], [], []
                for _t in _compare_sel:
                    _tr = first_row(_tech_all, _t)
                    _tscore_x.append(_t)
                    _tscore_y.append(float(pd.to_numeric(_tr.get("technical_score", 0), errors="coerce") or 0))
                    _tscore_rsi.append(float(pd.to_numeric(_tr.get("rsi14", 50), errors="coerce") or 50))
                    _tscore_vol.append(float(pd.to_numeric(_tr.get("volume_z60", 0), errors="coerce") or 0))

                _tc1, _tc2 = st.columns(2)
                with _tc1:
                    st.markdown("**Technical Score vs RSI**")
                    _fig_ts = go.Figure()
                    _fig_ts.add_trace(go.Bar(
                        name="Tech Score",
                        x=_tscore_x, y=_tscore_y,
                        marker_color="#22d3ee",
                        text=[f"{v:.0f}" for v in _tscore_y],
                        textposition="outside",
                    ))
                    _fig_ts.add_trace(go.Scatter(
                        name="RSI",
                        x=_tscore_x, y=_tscore_rsi,
                        mode="markers+lines",
                        marker=dict(color="#f59e0b", size=10),
                        line=dict(color="#f59e0b", width=2),
                        yaxis="y2",
                    ))
                    _fig_ts.update_layout(
                        height=280, margin=dict(l=10, r=40, t=24, b=20),
                        yaxis=dict(title="Tech Score", range=[0, 110], gridcolor="#e5e7eb"),
                        yaxis2=dict(title="RSI", range=[0, 100], overlaying="y", side="right",
                                    showgrid=False, tickfont=dict(color="#f59e0b")),
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(gridcolor="#e5e7eb"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        font=dict(family="Inter,sans-serif", size=12),
                        barmode="group",
                    )
                    st.plotly_chart(_fig_ts, use_container_width=True)

                with _tc2:
                    st.markdown("**Volume Z-Score (vs 60-day avg)**")
                    _vcols = ["#16a34a" if v > 0 else "#dc2626" for v in _tscore_vol]
                    _fig_vol = go.Figure(go.Bar(
                        x=_tscore_x, y=_tscore_vol,
                        marker_color=_vcols,
                        text=[f"{v:+.2f}σ" for v in _tscore_vol],
                        textposition="outside",
                    ))
                    _fig_vol.add_hline(y=0, line_width=1, line_color="#9ca3af")
                    _fig_vol.update_layout(
                        height=280, margin=dict(l=10, r=10, t=24, b=20),
                        yaxis_title="Volume Z-Score",
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(gridcolor="#e5e7eb"),
                        yaxis=dict(gridcolor="#e5e7eb", zerolinecolor="#9ca3af"),
                        font=dict(family="Inter,sans-serif", size=12),
                    )
                    st.plotly_chart(_fig_vol, use_container_width=True)

            # ── Side-by-side data table ───────────────────────────────────────
            st.markdown("**10-Layer Score Comparison**")
            _l_cols = [f"L{i}_score" for i in range(1, 11)] + [f"L{i}_state" for i in range(1, 11)]
            _cmp_rows = []
            for _t in _compare_sel:
                _mr = first_row(master, _t)
                _tr = first_row(_tech_all, _t)
                _fr = first_row(_fund_all, _t)
                _sr = first_row(_sec_all, _t)
                _row_d = {"ticker": _t}
                for _i in range(1, 11):
                    _row_d[f"L{_i}"] = f"{_mr.get(f'L{_i}_score','')}/{_mr.get(f'L{_i}_state','')}"
                _row_d["decision"] = _mr.get("master_action", "")
                _row_d["stack_avg"] = _mr.get("stack_score_avg", "")
                _row_d["tech_score"] = _tr.get("technical_score", "")
                _row_d["rotation"] = _sr.get("rotation_score", "")
                _row_d["fwd_pe"] = _fr.get("forward_pe", "")
                _row_d["quality"] = _fr.get("quality_score", "")
                _cmp_rows.append(_row_d)
            render_badge_table(pd.DataFrame(_cmp_rows), height=min(56 + len(_cmp_rows) * 48, 420))

            # ── Rotation score radar ──────────────────────────────────────────
            if _PLOTLY and not _sec_all.empty and len(_compare_sel) > 1:
                _rot_tickers = []
                _rot_scores  = []
                for _t in _compare_sel:
                    _sr = first_row(_sec_all, _t)
                    _rv = pd.to_numeric(_sr.get("rotation_score", None), errors="coerce")
                    if pd.notna(_rv):
                        _rot_tickers.append(_t)
                        _rot_scores.append(float(_rv))
                if _rot_tickers:
                    st.markdown("**Rotation Score Comparison**")
                    _rot_colors = ["#16a34a" if v > 0 else "#dc2626" for v in _rot_scores]
                    _fig_rot = go.Figure(go.Bar(
                        x=_rot_tickers, y=_rot_scores,
                        marker_color=_rot_colors,
                        text=[f"{v:+.1f}" for v in _rot_scores],
                        textposition="outside",
                    ))
                    _fig_rot.add_hline(y=0, line_width=1, line_color="#9ca3af")
                    _fig_rot.update_layout(
                        height=260, margin=dict(l=10, r=10, t=24, b=20),
                        yaxis_title="Rotation Score",
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(gridcolor="#e5e7eb"),
                        yaxis=dict(gridcolor="#e5e7eb", zerolinecolor="#9ca3af"),
                        font=dict(family="Inter,sans-serif", size=13),
                    )
                    st.plotly_chart(_fig_rot, use_container_width=True)


def tab_system_control():
    st.header("System Control")
    st.caption("Check whether data can be trusted, files are fresh, reports are archived, and signals are reliable.")

    routes = build_system_control_routes()
    reports = build_report_archive_index()
    generated = build_output_file_index()
    run_status = build_run_status()
    health = read_csv(FILES["data_source_health"])
    vault_alerts = read_csv(FILES["vault_alerts"])
    checks = build_qa_checks()

    report_missing = count_value(reports, "status", "MISSING")
    real_alerts = int((vault_alerts["status"].astype(str).str.upper() != "OK").sum()) if not vault_alerts.empty and "status" in vault_alerts.columns else 0
    data_risk = count_value(health, "status", "RISK")
    qa_high = count_value(checks, "severity", "HIGH")
    stale_or_missing = count_value(run_status, "status", "STALE") + count_value(run_status, "status", "MISSING")

    system_state = "RISK" if data_risk or qa_high or real_alerts else ("REVIEW" if report_missing or stale_or_missing else "OK")
    render_layer_workbench_header(
        "System",
        "Pipeline Trust & Output Control",
        "Keeps old reports visible, detects output shrinkage, monitors data-source failure, and checks whether the local run is fresh.",
        [
            ("System State", system_state, status_kind(system_state)),
            ("Reports Missing", report_missing, "blocked" if report_missing else "supportive"),
            ("Vault Alerts", real_alerts, "risk" if real_alerts else "supportive"),
            ("Data Source Risk", data_risk, "risk" if data_risk else "supportive"),
        ],
    )

    # ── Data Refresh control panel ─────────────────────────────────────────────
    _runner = ROOT / "canyon_final_v9_step56_full_10_layer_daily_runner_v2.py"
    _runner_exists = _runner.exists()
    _master_mtime = ""
    _master_age_h = None
    _master_file = FILES.get("master_v2", ROOT / "master_10_layer_decision_matrix_v2.csv")
    if Path(_master_file).exists():
        import time as _time
        _age_s = _time.time() - Path(_master_file).stat().st_mtime
        _master_age_h = _age_s / 3600
        if _master_age_h < 1:
            _master_mtime = f"{int(_age_s / 60)} min ago"
        elif _master_age_h < 24:
            _master_mtime = f"{_master_age_h:.1f} h ago"
        else:
            _master_mtime = f"{_master_age_h / 24:.1f} days ago"
    else:
        _master_mtime = "not found"

    _freshness_color = "#34d399" if (_master_age_h is not None and _master_age_h < 6) else ("#f87171" if (_master_age_h is None or _master_age_h > 24) else "#facc15")

    _no_runner_html = "&nbsp;&middot;&nbsp;<span style='color:#f87171;font-weight:600;'>Runner script not found</span>" if not _runner_exists else ""
    _freshness_html = (
        f'<div style="border-top:2px solid #111827;padding:14px 0 16px 0;margin:0 0 20px 0;'
        f'display:flex;justify-content:space-between;align-items:center;gap:24px;">'
        f'<div>'
        f'<div style="font-size:10px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#4b5563;margin-bottom:4px;">Data Freshness</div>'
        f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:18px;font-weight:600;color:#111827;">'
        f'Last run:&nbsp;<span style="color:{_freshness_color};">{escape(_master_mtime)}</span>'
        f'</div>'
        f'<div style="font-size:12px;color:#6b7280;margin-top:3px;">master_10_layer_decision_matrix_v2.csv{_no_runner_html}</div>'
        f'</div>'
        f'<div style="font-size:12px;color:#6b7280;max-width:380px;">'
        f'Click <b style="color:#111827;">Run Full Update</b> to re-run all 10 layers and refresh outputs. '
        f'Runs in the background &#8212; reload this page after ~3 minutes to see updated data.'
        f'</div>'
        f'</div>'
    )
    st.markdown(_freshness_html, unsafe_allow_html=True)

    _btn_col, _msg_col = st.columns([2, 5])
    with _btn_col:
        if _runner_exists:
            if st.button("▶  Run Full Update", key="sys_ctrl_run_refresh", type="primary"):
                try:
                    subprocess.Popen(
                        [sys.executable, str(_runner)],
                        cwd=str(ROOT),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    st.toast("Daily runner started in background. Reload in ~3 minutes.", icon="✅")
                    st.success("Runner launched. No broker. No live order.")
                except Exception as _e:
                    st.error(f"Could not start runner: {_e}")
        else:
            st.warning("Runner script not found — cannot launch from dashboard.")

    st.subheader("How To Use The System Page")
    render_badge_table(routes, height=330)

    if _PLOTLY and not run_status.empty and "status" in run_status.columns:
        _sc_counts = run_status["status"].value_counts().reset_index()
        _sc_counts.columns = ["status", "count"]
        _sc_colors = [
            "#16a34a" if str(s).upper() == "FRESH"
            else "#b91c1c" if str(s).upper() == "MISSING"
            else "#facc15"
            for s in _sc_counts["status"]
        ]
        _fig_sc = go.Figure(go.Bar(
            x=_sc_counts["count"].tolist(),
            y=_sc_counts["status"].astype(str).tolist(),
            orientation="h",
            marker_color=_sc_colors,
            text=_sc_counts["count"].astype(str).tolist(),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x} output(s)<extra></extra>",
        ))
        _fig_sc.update_layout(
            height=max(80, len(_sc_counts) * 46 + 30),
            margin=dict(l=10, r=40, t=10, b=10),
            xaxis_title="Count", yaxis_title="",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#e5e7eb"),
            yaxis=dict(gridcolor="#e5e7eb", autorange="reversed"),
            font=dict(family="Inter,sans-serif", size=13),
        )
        st.plotly_chart(_fig_sc, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Are Files Fresh?")
        if run_status.empty:
            st.info("No run status available.")
        else:
            render_badge_table(run_status.head(10), height=340)

    with c2:
        st.subheader("Data Source Health")
        if health.empty:
            st.info("No data source health file — run Step 61 to generate it.")
            st.code("python3 -u canyon_final_v9_step61_data_source_health.py", language="bash")
        else:
            render_badge_table(health.head(10), height=340)

    st.subheader("Report Archive Coverage")
    coverage = pd.DataFrame([
        {
            "status": "OK" if report_missing == 0 else "MISSING",
            "area": "Tracked markdown reports",
            "count": len(reports),
            "issue_count": report_missing,
            "note": "All original reports are indexed in Report Archive.",
        },
        {
            "status": "OK" if not generated.empty else "MISSING",
            "area": "Generated CSV/MD files",
            "count": len(generated),
            "issue_count": 0 if not generated.empty else 1,
            "note": "All generated outputs are discoverable from System.",
        },
        {
            "status": "OK" if real_alerts == 0 else "RISK",
            "area": "Output shrinkage",
            "count": len(vault_alerts),
            "issue_count": real_alerts,
            "note": "OK placeholder rows are not counted as alerts.",
        },
        {
            "status": "OK" if qa_high == 0 else "RISK",
            "area": "QA high severity",
            "count": len(checks),
            "issue_count": qa_high,
            "note": "High severity issues should be reviewed before decision-ready use.",
        },
    ])
    render_badge_table(coverage, height=220)


def tab_data_gaps():
    st.header("Gap List")

    master = read_csv(FILES["master_v2"])
    market = read_csv(FILES["market_snapshot"])
    quality = read_csv(FILES["data_quality"])
    universe = read_csv(FILES["universe"])
    gaps = build_gap_queue(master, market)

    _gap_high = int((gaps["priority"].astype(str).str.upper() == "HIGH").sum()) if not gaps.empty and "priority" in gaps.columns else 0
    _gap_med  = int((gaps["priority"].astype(str).str.upper() == "MEDIUM").sum()) if not gaps.empty and "priority" in gaps.columns else 0
    _gap_state = "RISK" if _gap_high else ("REVIEW" if _gap_med else "OK")
    render_layer_workbench_header(
        "Gaps",
        "Data Gap List",
        "Missing or unreliable data reduces decision confidence. Fix HIGH-priority gaps before acting on any research conclusion.",
        [
            ("Gap State",     _gap_state,  status_kind(_gap_state)),
            ("High Priority", _gap_high,   "risk"       if _gap_high else "supportive"),
            ("Medium",        _gap_med,    "wait"       if _gap_med  else "supportive"),
            ("Total Gaps",    len(gaps),   "cyan"),
        ],
    )

    if gaps.empty:
        st.success("No major data gaps detected in the current matrix.")
        gap_summary_rows = [
            {
                "status": "OK",
                "gap_question": "Are there any data gaps?",
                "current_read": "No gaps detected",
                "why_it_matters": "Clean data means downstream decisions are based on real signals, not fallback assumptions.",
                "what_to_do": "Continue running the full daily runner to keep data fresh.",
                "source_file": "build_gap_queue(); master_10_layer_decision_matrix_v2.csv",
            },
        ]
        st.subheader("What To Fix First")
        render_badge_table(pd.DataFrame(gap_summary_rows), height=130)
    else:
        high = int((gaps["priority"] == "HIGH").sum())
        medium = int((gaps["priority"] == "MEDIUM").sum())
        low = int((gaps["priority"] == "LOW").sum())
        affected = int(gaps["ticker"].nunique())
        data_blockers = int((gaps["lane"] == "Data Blocker").sum()) if "lane" in gaps.columns else 0
        risk_blockers = int(gaps["lane"].isin(["Risk / Portfolio Blocker", "Execution Blocker"]).sum()) if "lane" in gaps.columns else 0

        # Find the highest-priority lane and its first fix action
        top_lane = "N/A"
        top_next_fix = "Run the full daily runner first."
        if not gaps.empty and "lane" in gaps.columns:
            high_gaps = gaps[gaps["priority"].astype(str).str.upper() == "HIGH"]
            if not high_gaps.empty:
                top_lane = str(high_gaps["lane"].iloc[0])
                top_next_fix = str(high_gaps["next_fix"].iloc[0]) if "next_fix" in high_gaps.columns else "Resolve HIGH priority gap."
            else:
                top_lane = str(gaps["lane"].iloc[0])
                top_next_fix = str(gaps["next_fix"].iloc[0]) if "next_fix" in gaps.columns else "Resolve gap."

        overall_status = "RISK" if high > 0 else ("REVIEW" if medium > 0 else "OK")
        gap_summary_rows = [
            {
                "status": overall_status,
                "gap_question": "What is the overall gap status?",
                "current_read": f"high={high}; medium={medium}; low={low}; affected tickers={affected}",
                "why_it_matters": "Gaps mean some layers are using fallback or missing data, which lowers decision confidence.",
                "what_to_do": "Fix HIGH priority gaps before relying on any research conclusions.",
                "source_file": "build_gap_queue(); master_10_layer_decision_matrix_v2.csv",
            },
            {
                "status": "RISK" if data_blockers > 0 else "REVIEW",
                "gap_question": "Are there data blockers or risk/action blockers?",
                "current_read": f"data blockers={data_blockers}; risk/action blockers={risk_blockers}",
                "why_it_matters": "Data blockers mean specific tickers have no reliable price or source. Risk blockers constrain sizing and allowed actions.",
                "what_to_do": "Check Data Sources tab for blocked/unavailable rows. Run full daily runner and reload.",
                "source_file": "data_source_health.csv; market_data_snapshot.csv",
            },
            {
                "status": "REVIEW" if top_lane != "N/A" else "OK",
                "gap_question": "What should I fix first?",
                "current_read": f"Top lane: {top_lane}",
                "why_it_matters": "Fixing the top lane resolves the most decision-critical data gaps.",
                "what_to_do": top_next_fix,
                "source_file": "canyon_final_v9_step56_full_10_layer_daily_runner_v2.py",
            },
        ]
        st.subheader("What To Fix First")
        render_badge_table(pd.DataFrame(gap_summary_rows), height=280)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("High-Priority Gaps", high)
        c2.metric("Medium-Priority Gaps", medium)
        c3.metric("Low-Priority To-Dos", low)
        c4.metric("Data Blockers", data_blockers)

        c5, c6 = st.columns(2)
        c5.metric("Affected Tickers", affected)
        c6.metric("Risk / Action Blockers", risk_blockers)

        lane_counts = gaps.groupby(["lane", "priority"], as_index=False).size().rename(columns={"size": "count"})
        st.subheader("Gap Summary")
        if _PLOTLY and not lane_counts.empty:
            _gap_lc = lane_counts.copy()
            _gap_lc_by_lane = _gap_lc.groupby("lane")["count"].sum().sort_values(ascending=True)
            _gap_lane_colors = ["#b91c1c" if n >= 3 else "#f87171" if n >= 1 else "#22d3ee" for n in _gap_lc_by_lane]
            _fig_gap = go.Figure(go.Bar(
                x=_gap_lc_by_lane.values.tolist(),
                y=_gap_lc_by_lane.index.tolist(),
                orientation="h",
                marker_color=_gap_lane_colors,
                text=[str(v) for v in _gap_lc_by_lane.values],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>%{x} gap(s)<extra></extra>",
            ))
            _fig_gap.update_layout(
                height=max(120, len(_gap_lc_by_lane) * 36 + 50),
                margin=dict(l=10, r=40, t=10, b=10),
                xaxis_title="Gap count", yaxis_title="",
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(gridcolor="#e5e7eb"),
                yaxis=dict(gridcolor="#e5e7eb", autorange="reversed"),
                font=dict(family="Inter,sans-serif", size=12),
            )
            st.plotly_chart(_fig_gap, use_container_width=True)
        render_badge_table(lane_counts, height=220)

        st.subheader("Fix Queue")
        lanes = ["All"] + sorted(gaps["lane"].dropna().astype(str).unique().tolist()) if "lane" in gaps.columns else ["All"]
        priorities = ["All", "HIGH", "MEDIUM", "LOW"]
        left, right = st.columns(2)
        with left:
            lane_filter = st.selectbox("Queue", lanes)
        with right:
            priority_filter = st.selectbox("Priority", priorities)

        view = gaps.copy()
        if lane_filter != "All" and "lane" in view.columns:
            view = view[view["lane"].astype(str) == str(lane_filter)]
        if str(priority_filter) != "All":
            view = view[view["priority"].astype(str) == str(priority_filter)]

        cols = [c for c in [
            "priority", "lane", "ticker", "gap_type", "layer", "state", "impact",
            "next_fix", "master_action", "note"
        ] if c in view.columns]
        render_badge_table(view[cols], height=520)

        with st.expander("Layer-by-layer backlog count"):
            by_layer = gaps.groupby(["layer", "layer_name", "priority"], as_index=False).size().rename(columns={"size": "count"})
            render_badge_table(by_layer, height=360)

    st.subheader("Price Data Coverage")
    if not market.empty:
        render_badge_table(market, height=280)
    else:
        st.info("No market snapshot file found.")

    st.subheader("Raw Data Quality Warnings")
    if not quality.empty:
        render_badge_table(quality, height=260)
    else:
        st.info("No data quality flags.")

    with st.expander("Universe source coverage"):
        show_df(universe, height=320, style=True)


def tab_run_status():
    st.header("Run Status")

    status = build_run_status()
    if status.empty:
        st.info("No run status available. Run the full daily runner first.")
        st.code("python3 -u canyon_final_v9_step56_full_10_layer_daily_runner_v2.py", language="bash")
        return

    counts = status["status"].value_counts()
    fresh   = int(counts.get("FRESH",   0))
    stale   = int(counts.get("STALE",   0))
    missing = int(counts.get("MISSING", 0))

    render_layer_workbench_header(
        "Run",
        "Pipeline Freshness",
        "Check whether all 10-layer outputs are fresh before trusting decisions. Stale outputs = old signals.",
        [
            ("Fresh",   fresh,   "supportive" if fresh and not stale and not missing else "cyan"),
            ("Stale",   stale,   "risk" if stale else "watch"),
            ("Missing", missing, "blocked" if missing else "watch"),
            ("Total",   fresh + stale + missing, "cyan"),
        ],
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Fresh", fresh)
    c2.metric("Stale", stale)
    c3.metric("Missing", missing)

    stale_or_missing = status[status["status"].isin(["STALE", "MISSING"])]
    fix_status = "OK" if stale_or_missing.empty else ("RISK" if int(counts.get("MISSING", 0)) else "REVIEW")
    oldest = "N/A"
    if not status.empty and "age_hours" in status.columns:
        ages = pd.to_numeric(status["age_hours"], errors="coerce")
        if not ages.dropna().empty:
            idx = ages.idxmax()
            oldest = f"{status.loc[idx, 'area']} · {ages.loc[idx]:.1f}h" if "area" in status.columns else f"{ages.loc[idx]:.1f}h"

    run_fix_summary = pd.DataFrame([
        {
            "status": fix_status,
            "system_question": "Can today's dashboard be trusted?",
            "current_read": f"fresh={int(counts.get('FRESH', 0))}; stale={int(counts.get('STALE', 0))}; missing={int(counts.get('MISSING', 0))}",
            "why_it_matters": "Old outputs can make the research stack look cleaner than it is.",
            "what_to_do": "If stale or missing exists, run the full daily workflow before relying on decisions.",
            "source_file": "build_run_status(); generated output files",
        },
        {
            "status": "REVIEW" if oldest != "N/A" else "NO_DATA",
            "system_question": "What is the oldest output?",
            "current_read": oldest,
            "why_it_matters": "The oldest file usually shows which part of the pipeline needs attention.",
            "what_to_do": "Refresh the pipeline, then reload the dashboard.",
            "source_file": "file timestamps",
        },
        {
            "status": "OK",
            "system_question": "Which command should run next?",
            "current_read": "Step 56 full daily workflow",
            "why_it_matters": "One full run is safer than refreshing one layer and missing dependencies.",
            "what_to_do": "Run: python3 -u canyon_final_v9_step56_full_10_layer_daily_runner_v2.py",
            "source_file": "canyon_final_v9_step56_full_10_layer_daily_runner_v2.py",
        },
    ])
    st.subheader("Run Fix Summary")
    render_badge_table(run_fix_summary, height=230)

    if _PLOTLY and not status.empty and "area" in status.columns and "age_hours" in status.columns:
        _rs_df = status.copy()
        _rs_df["age_hours"] = pd.to_numeric(_rs_df["age_hours"], errors="coerce").fillna(0)
        _rs_df = _rs_df.sort_values("age_hours", ascending=True)
        _rs_bar_colors = [
            "#22d3ee" if str(r.get("status","")).upper() == "FRESH"
            else "#f87171" if str(r.get("status","")).upper() == "MISSING"
            else "#facc15"
            for _, r in _rs_df.iterrows()
        ]
        _fig_rs = go.Figure(go.Bar(
            x=_rs_df["age_hours"].tolist(),
            y=_rs_df["area"].astype(str).tolist(),
            orientation="h",
            marker_color=_rs_bar_colors,
            text=_rs_df["age_hours"].map(lambda x: f"{x:.1f}h").tolist(),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Age: %{x:.1f} hours<extra></extra>",
        ))
        _fig_rs.update_layout(
            height=max(180, len(_rs_df) * 32 + 60),
            margin=dict(l=10, r=60, t=28, b=20),
            title=dict(text="Output Age By Area (cyan=fresh, yellow=stale, red=missing)", font=dict(size=12), x=0),
            xaxis_title="Age (hours)", yaxis_title="",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#e5e7eb"),
            yaxis=dict(gridcolor="#e5e7eb"),
            font=dict(family="Inter,sans-serif", size=12),
        )
        st.plotly_chart(_fig_rs, use_container_width=True)

    st.subheader("Is The Daily Run Fresh?")
    render_badge_table(status, height=420)

    if stale_or_missing.empty:
        st.success("Core pipeline outputs look fresh. You can review Command Center first.")
    else:
        st.warning("Some pipeline outputs are stale or missing. Re-run the full v2 workflow before relying on decisions.")

    st.subheader("Daily Refresh Commands")
    st.code(
        """cd ~/Desktop/canyon_quant
source .venv/bin/activate

python3 -u canyon_final_v9_step56_full_10_layer_daily_runner_v2.py
streamlit run canyon_final_v9_step55_10_layer_dashboard_v2.py
""",
        language="bash",
    )

    with st.expander("Latest full runner log"):
        st.markdown(read_md(ROOT / "full_10_layer_daily_runner_v2_log.md"))


def tab_system_qa():
    st.header("System Check")

    checks = build_qa_checks()
    reports = build_report_archive_index()
    generated = build_output_file_index()
    run_status = build_run_status()
    master = read_csv(FILES["master_v2"])
    completeness = build_layer_completeness(master)

    severity_counts = checks["severity"].value_counts() if not checks.empty and "severity" in checks.columns else pd.Series(dtype=int)
    report_counts = reports["status"].value_counts() if not reports.empty else pd.Series(dtype=int)
    run_counts = run_status["status"].value_counts() if not run_status.empty else pd.Series(dtype=int)
    layer_gaps = int(pd.to_numeric(completeness["gaps"], errors="coerce").fillna(0).sum()) if not completeness.empty else 0

    _qa_high  = int(severity_counts.get("HIGH",   0))
    _qa_med   = int(severity_counts.get("MEDIUM", 0))
    _qa_state = "RISK" if _qa_high else ("REVIEW" if _qa_med else "OK")
    render_layer_workbench_header(
        "QA",
        "System Health And Completeness",
        "Checks data source integrity, pipeline freshness, report coverage, and 10-layer gaps. HIGH severity items block trust.",
        [
            ("Health State",   _qa_state,           status_kind(_qa_state)),
            ("High Issues",    _qa_high,             "risk"       if _qa_high else "supportive"),
            ("Medium Issues",  _qa_med,              "wait"       if _qa_med  else "supportive"),
            ("Layer Gaps",     layer_gaps,           "cyan"       if layer_gaps else "supportive"),
        ],
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("High-Risk Health Issues", int(severity_counts.get("HIGH", 0)))
    c2.metric("Medium Health Issues", int(severity_counts.get("MEDIUM", 0)))
    c3.metric("Layer Status Gaps", layer_gaps)
    c4.metric("Reports Found", int(report_counts.get("FOUND", 0)))
    c5.metric("Generated Files", len(generated))

    if int(severity_counts.get("HIGH", 0)) > 0:
        st.warning("High-priority QA items exist. Review them before treating the dashboard as decision-ready.")
    elif int(run_counts.get("STALE", 0)) > 0 or int(run_counts.get("MISSING", 0)) > 0:
        st.info("No high QA issues, but some pipeline outputs are stale or missing.")
    else:
        st.success("Core QA checks look clear for the current local files.")

    st.subheader("Health Findings")
    if _PLOTLY and not checks.empty and "severity" in checks.columns:
        _qa_sev = checks["severity"].value_counts().reset_index()
        _qa_sev.columns = ["severity", "count"]
        _qa_sev_colors = [
            "#b91c1c" if str(s).upper() == "HIGH"
            else "#facc15" if str(s).upper() == "MEDIUM"
            else "#22d3ee"
            for s in _qa_sev["severity"]
        ]
        _fig_qa = go.Figure(go.Bar(
            x=_qa_sev["count"].tolist(),
            y=_qa_sev["severity"].astype(str).tolist(),
            orientation="h",
            marker_color=_qa_sev_colors,
            text=_qa_sev["count"].astype(str).tolist(),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x} issue(s)<extra></extra>",
        ))
        _fig_qa.update_layout(
            height=max(100, len(_qa_sev) * 44 + 40),
            margin=dict(l=10, r=40, t=24, b=10),
            title=dict(text="QA Issues By Severity", font=dict(size=12), x=0),
            xaxis_title="Count", yaxis_title="",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#e5e7eb"),
            yaxis=dict(gridcolor="#e5e7eb", autorange="reversed"),
            font=dict(family="Inter,sans-serif", size=13),
        )
        st.plotly_chart(_fig_qa, use_container_width=True)
    render_badge_table(checks, height=420)

    st.subheader("10-Layer Completeness")
    render_badge_table(completeness, height=360)

    st.subheader("Coverage Snapshot")
    coverage = pd.DataFrame([
        {
            "status": "OK" if int(report_counts.get("MISSING", 0)) == 0 else "MISSING",
            "area": "Markdown reports",
            "found": int(report_counts.get("FOUND", 0)),
            "missing": int(report_counts.get("MISSING", 0)),
            "note": "Original report archive coverage",
        },
        {
            "status": "OK" if not generated.empty else "MISSING",
            "area": "Generated files",
            "found": len(generated),
            "missing": 0 if not generated.empty else 1,
            "note": "All .md and .csv outputs visible from Report Archive",
        },
        {
            "status": "OK" if int(run_counts.get("MISSING", 0)) == 0 else "MISSING",
            "area": "Core pipeline outputs",
            "found": int(run_counts.get("FRESH", 0)) + int(run_counts.get("STALE", 0)),
            "missing": int(run_counts.get("MISSING", 0)),
            "note": "Freshness details live in Run Status",
        },
        {
            "status": "OK" if not master.empty else "MISSING",
            "area": "Main decision table",
            "found": len(master),
            "missing": 0 if not master.empty else 1,
            "note": "Ticker-level final decision table",
        },
    ])
    render_badge_table(coverage, height=220)

    with st.expander("Core pipeline freshness"):
        render_badge_table(run_status, height=360)


def tab_paper_ledger():
    st.header("Paper Log And Review Learning")

    ledger = read_csv(FILES["paper_ledger"])
    summary = read_csv(FILES["learning_summary"])
    suggestions = read_csv(FILES["learning_suggestions"])

    if ledger.empty:
        st.warning("No paper ledger found yet.")
        st.caption("Add a paper-test entry to paper_portfolio_ledger.csv to start the log.")
        return

    closed_mask = ledger["status"].astype(str).str.upper().isin(["CLOSED_PAPER", "CLOSED_REAL"]) if "status" in ledger.columns else pd.Series(False, index=ledger.index)
    closed = ledger[closed_mask]
    watch = ledger[~closed_mask]
    closed_count = len(closed)
    needed = max(30 - closed_count, 0)

    avg_pnl = ""
    win_rate = ""
    if not closed.empty and "pnl_pct" in closed.columns:
        pnl = pd.to_numeric(closed["pnl_pct"], errors="coerce").dropna()
        if not pnl.empty:
            avg_pnl = f"{pnl.mean() * 100:.2f}%"
            win_rate = f"{(pnl > 0).mean() * 100:.1f}%"

    render_layer_workbench_header(
        "L10",
        "Paper Log And Learning Review",
        "Record paper-test outcomes. Do not adjust strategy weights until at least 30 closed samples exist.",
        [
            ("Closed Samples",  closed_count,                     "supportive" if closed_count >= 30 else "cyan"),
            ("Still Needed",    needed,                           "blocked" if needed else "supportive"),
            ("Avg Return",      avg_pnl or "N/A",                 "supportive" if avg_pnl.startswith("+") else ("risk" if avg_pnl.startswith("-") else "wait")),
            ("Win Rate",        win_rate or "N/A",                "supportive"),
        ],
    )
    # ── performance metrics ───────────────────────────────────────────────────
    _pm_ledger = build_perf_metrics(ledger, read_csv(FILES["market_snapshot"]))
    if _pm_ledger["n_closed"] >= 2:
        _perf_c1, _perf_c2, _perf_c3, _perf_c4 = st.columns(4)
        _perf_c1.metric("Total Return", _pm_ledger["total_return"])
        _perf_c2.metric("Sharpe (per-trade)", _pm_ledger["sharpe"])
        _perf_c3.metric("Max Drawdown", _pm_ledger["max_drawdown"])
        _perf_c4.metric("vs SPY Alpha", _pm_ledger["alpha"])
        if _PLOTLY and not _pm_ledger["drawdown_series"].empty:
            _dd_ser = _pm_ledger["drawdown_series"]
            _dd_col = ["#b91c1c" if v < -0.05 else "#f87171" if v < 0 else "#22d3ee" for v in _dd_ser]
            _fig_dd_l = go.Figure(go.Bar(
                x=list(range(1, len(_dd_ser) + 1)),
                y=(_dd_ser * 100).tolist(),
                marker_color=_dd_col,
                hovertemplate="Trade %{x}<br>DD: %{y:.2f}%<extra></extra>",
            ))
            _fig_dd_l.add_hline(y=0, line_color="#111827", line_width=1)
            _fig_dd_l.update_layout(
                height=200, margin=dict(l=10, r=10, t=24, b=10),
                title=dict(text=f"Drawdown — Max: {_pm_ledger['max_drawdown']}", font=dict(size=12), x=0),
                xaxis_title="Trade #", yaxis_title="Drawdown %",
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(gridcolor="#e5e7eb"), yaxis=dict(gridcolor="#e5e7eb", ticksuffix="%"),
                font=dict(family="Inter,sans-serif", size=12),
            )
            st.plotly_chart(_fig_dd_l, use_container_width=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Log Rows", len(ledger))
    c2.metric("Closed Samples", closed_count)
    c3.metric("Samples Still Needed", needed)
    c4.metric("Average Closed Return", avg_pnl or "N/A")

    # ── Paper P&L Charts ─────────────────────────────────────────────────────
    if _PLOTLY:
        _ch1, _ch2 = st.columns(2)

        with _ch1:
            st.markdown("**Closed Trade Returns**")
            if not closed.empty and "pnl_pct" in closed.columns:
                _cr = closed[["ticker", "pnl_pct", "notes"]].copy()
                _cr["pnl_pct"] = pd.to_numeric(_cr["pnl_pct"], errors="coerce")
                _cr = _cr.dropna(subset=["pnl_pct"])
                _cr["return_pct"] = _cr["pnl_pct"] * 100
                _cr["bar_color"] = _cr["return_pct"].apply(lambda x: "#16a34a" if x >= 0 else "#dc2626")
                _cr["label"] = _cr["ticker"]
                _cr = _cr.sort_values("return_pct", ascending=True)
                _fig_ret = go.Figure(go.Bar(
                    x=_cr["return_pct"],
                    y=_cr["label"],
                    orientation="h",
                    marker_color=_cr["bar_color"].tolist(),
                    text=_cr["return_pct"].map(lambda x: f"{x:+.2f}%"),
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Return: %{x:+.2f}%<extra></extra>",
                ))
                _fig_ret.update_layout(
                    height=max(240, 70 + len(_cr) * 56),
                    margin=dict(l=10, r=60, t=24, b=20),
                    xaxis_title="Return %",
                    yaxis_title="",
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    xaxis=dict(gridcolor="#e5e7eb", zerolinecolor="#9ca3af",
                               ticksuffix="%", tickfont=dict(size=12)),
                    yaxis=dict(gridcolor="#e5e7eb", tickfont=dict(size=13)),
                    font=dict(family="Inter,sans-serif", size=13),
                )
                st.plotly_chart(_fig_ret, use_container_width=True)
            else:
                st.info("No closed paper trades to chart yet.")

        with _ch2:
            st.markdown("**Open Position Weights by Sleeve**")
            if not watch.empty and "effective_weight" in watch.columns and "sleeve" in watch.columns:
                _SLEEVE_COLOR = {
                    "CORE_HEDGE": "#2563eb",
                    "SECTOR_ROTATION": "#16a34a",
                    "TACTICAL": "#7c3aed",
                }
                _wt = watch[["ticker", "sleeve", "effective_weight"]].copy()
                _wt["effective_weight"] = pd.to_numeric(_wt["effective_weight"], errors="coerce").fillna(0)
                _wt["bar_color"] = _wt["sleeve"].map(
                    lambda s: _SLEEVE_COLOR.get(str(s).upper(), "#6b7280"))
                _wt["weight_pct"] = _wt["effective_weight"] * 100
                _wt["label"] = _wt["ticker"] + " (" + _wt["sleeve"].astype(str) + ")"
                _wt = _wt.sort_values("weight_pct", ascending=True)
                _fig_wt = go.Figure(go.Bar(
                    x=_wt["weight_pct"],
                    y=_wt["label"],
                    orientation="h",
                    marker_color=_wt["bar_color"].tolist(),
                    text=_wt["weight_pct"].map(lambda x: f"{x:.1f}%"),
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Weight: %{x:.1f}%<extra></extra>",
                ))
                # sleeve legend chips
                _legend_html = " ".join(
                    f'<span style="display:inline-block;width:10px;height:10px;'
                    f'border-radius:2px;background:{v};margin-right:4px;"></span>'
                    f'<span style="font-size:12px;color:#374151;margin-right:12px;">{k}</span>'
                    for k, v in _SLEEVE_COLOR.items()
                )
                st.markdown(
                    f'<div style="margin:-8px 0 4px 0">{_legend_html}</div>',
                    unsafe_allow_html=True,
                )
                _fig_wt.update_layout(
                    height=max(240, 70 + len(_wt) * 44),
                    margin=dict(l=10, r=70, t=8, b=20),
                    xaxis_title="Weight %",
                    yaxis_title="",
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    xaxis=dict(gridcolor="#e5e7eb", zerolinecolor="#9ca3af",
                               ticksuffix="%", tickfont=dict(size=12)),
                    yaxis=dict(gridcolor="#e5e7eb", tickfont=dict(size=12)),
                    font=dict(family="Inter,sans-serif", size=13),
                )
                st.plotly_chart(_fig_wt, use_container_width=True)
            else:
                st.info("No open positions to chart yet.")
    else:
        st.warning("Plotly not installed — charts unavailable. Run: pip install plotly")

    if closed_count < 5:
        st.info("L10 is still in record-only mode. At least 5 closed paper trades are required before weight changes should be considered.")
    else:
        st.success("L10 has enough closed paper samples for cautious review signals.")

    learning_status = "REVIEW" if closed_count < 5 else "OK"
    learning_summary = pd.DataFrame([
        {
            "status": learning_status,
            "learning_question": "Can learning change strategy weights yet?",
            "current_read": f"{closed_count} closed samples; {needed} still needed",
            "why_it_matters": "A tiny sample can teach notes, but it should not rewrite the strategy.",
            "what_to_do": "Keep L10 in record-only mode until enough clean paper samples exist.",
            "source_file": "paper_portfolio_ledger.csv; learning_attribution_report.md",
        },
        {
            "status": "OK" if avg_pnl else "NO_DATA",
            "learning_question": "What do closed samples show so far?",
            "current_read": f"avg={avg_pnl or 'N/A'}; win_rate={win_rate or 'N/A'}",
            "why_it_matters": "Early results are useful for review, not for automatic confidence.",
            "what_to_do": "Record the result and explain why it worked or failed.",
            "source_file": "learning_attribution_summary.csv",
        },
        {
            "status": "REVIEW" if not watch.empty else "OK",
            "learning_question": "What still needs observation?",
            "current_read": f"{len(watch)} open/watch rows",
            "why_it_matters": "Open paper rows should not be counted as proven learning.",
            "what_to_do": "Keep notes clean: thesis, risk note, entry, exit, and reason.",
            "source_file": "paper_portfolio_ledger.csv",
        },
    ])
    st.subheader("Learning Summary")
    render_badge_table(learning_summary, height=230)

    st.subheader("Paper Logs Still Being Watched")
    if not watch.empty:
        watch_cols = [c for c in [
            "trade_id", "ticker", "sleeve", "decision", "risk_bucket", "suggested_action",
            "status", "suggested_weight", "risk_note", "manual_news_check",
            "earnings_date_check", "liquidity_check", "spread_check"
        ] if c in watch.columns]
        render_badge_table(watch[watch_cols], height=380)
    else:
        st.write("No open/watch ledger rows.")

    st.subheader("Closed Paper Samples")
    if not closed.empty:
        closed_cols = [c for c in [
            "trade_id", "ticker", "sleeve", "risk_bucket", "status", "entry_date",
            "entry_price", "exit_date", "exit_price", "pnl_pct", "holding_days", "notes"
        ] if c in closed.columns]
        render_badge_table(closed[closed_cols], height=260)
    else:
        st.write("No closed samples yet.")

    st.subheader("Review Results")
    if not summary.empty:
        render_badge_table(summary, height=260)
    else:
        st.write("No attribution summary yet.")

    st.subheader("Weight Suggestions")
    if not suggestions.empty:
        render_badge_table(suggestions, height=260)
    else:
        st.write("No learning suggestions yet.")

    with st.expander("Raw learning report"):
        st.markdown(read_md(FILES["learning_report"]))


def format_percent_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            nums = pd.to_numeric(out[col], errors="coerce")
            out[col] = nums.map(lambda x: "" if pd.isna(x) else f"{x * 100:.2f}%")
    return out


def aggregate_weight(df: pd.DataFrame, group_col: str, weight_col: str = "effective_weight") -> pd.DataFrame:
    if df.empty or group_col not in df.columns or weight_col not in df.columns:
        return pd.DataFrame()
    tmp = df.copy()
    tmp[weight_col] = pd.to_numeric(tmp[weight_col], errors="coerce")
    out = (
        tmp.groupby(group_col, dropna=False)[weight_col]
        .sum()
        .reset_index()
        .sort_values(weight_col, ascending=False)
    )
    return out


def chart_weight(df: pd.DataFrame, group_col: str, weight_col: str = "effective_weight"):
    agg = aggregate_weight(df, group_col, weight_col)
    if agg.empty:
        st.info("No chart data.")
        return
    if not _PLOTLY:
        chart_df = agg.set_index(group_col)
        st.bar_chart(chart_df, y=weight_col, height=260)
        return
    agg[weight_col] = pd.to_numeric(agg[weight_col], errors="coerce").fillna(0)
    agg["_pct"] = agg[weight_col] * 100
    agg = agg.sort_values("_pct", ascending=True)
    bar_colors = ["#b91c1c" if v >= 20 else "#2563eb" if v >= 10 else "#22d3ee" if v >= 5 else "#6b7280"
                  for v in agg["_pct"]]
    _fig = go.Figure(go.Bar(
        x=agg["_pct"],
        y=agg[group_col].astype(str),
        orientation="h",
        marker_color=bar_colors,
        text=agg["_pct"].map(lambda x: f"{x:.1f}%"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>",
    ))
    _fig.update_layout(
        height=max(200, 60 + len(agg) * 46),
        margin=dict(l=10, r=60, t=10, b=20),
        xaxis_title="Weight %",
        yaxis_title="",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#e5e7eb", zerolinecolor="#9ca3af", ticksuffix="%"),
        yaxis=dict(gridcolor="#e5e7eb"),
        font=dict(family="Inter,sans-serif", size=13),
    )
    st.plotly_chart(_fig, use_container_width=True)


def chart_sleeve_breakdown(sizing_df: pd.DataFrame):
    """Stacked horizontal bar: sector on y-axis, stacked by sleeve."""
    if not _PLOTLY or sizing_df.empty:
        return
    needed = {"sleeve", "effective_weight"}
    if not needed.issubset(sizing_df.columns):
        return
    _df = sizing_df.copy()
    _df["effective_weight"] = pd.to_numeric(_df["effective_weight"], errors="coerce").fillna(0)
    _df["_pct"] = _df["effective_weight"] * 100
    _grp_col = "sector" if "sector" in _df.columns else ("risk_bucket" if "risk_bucket" in _df.columns else "sleeve")
    pivot = _df.groupby([_grp_col, "sleeve"])["_pct"].sum().reset_index()
    totals = pivot.groupby(_grp_col)["_pct"].sum().sort_values(ascending=True)
    grp_order = totals.index.tolist()
    _SLEEVE_COLOR = {
        "CORE_HEDGE":      "#2563eb",
        "SECTOR_ROTATION": "#16a34a",
        "TACTICAL":        "#7c3aed",
    }
    sleeves = sorted(pivot["sleeve"].unique().tolist())
    _fig2 = go.Figure()
    for sleeve in sleeves:
        sub = pivot[pivot["sleeve"] == sleeve].set_index(_grp_col).reindex(grp_order).fillna(0)
        _fig2.add_trace(go.Bar(
            name=sleeve,
            x=sub["_pct"].tolist(),
            y=grp_order,
            orientation="h",
            marker_color=_SLEEVE_COLOR.get(str(sleeve).upper(), "#6b7280"),
            text=[f"{v:.1f}%" if v > 0.5 else "" for v in sub["_pct"].tolist()],
            textposition="inside",
            textfont=dict(color="white", size=11),
            hovertemplate=f"<b>%{{y}}</b> · {sleeve}<br>%{{x:.1f}}%<extra></extra>",
        ))
    _fig2.update_layout(
        barmode="stack",
        height=max(220, 70 + len(grp_order) * 46),
        margin=dict(l=10, r=80, t=10, b=20),
        xaxis_title="Weight %",
        yaxis_title="",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#e5e7eb", zerolinecolor="#9ca3af", ticksuffix="%"),
        yaxis=dict(gridcolor="#e5e7eb"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=12)),
        font=dict(family="Inter,sans-serif", size=13),
    )
    st.plotly_chart(_fig2, use_container_width=True)


def build_trigger_board(triggers: pd.DataFrame, cards: pd.DataFrame) -> pd.DataFrame:
    if triggers.empty:
        return pd.DataFrame()

    out = triggers.copy()
    if "call_wall_distance" in out.columns:
        out["breakout_distance_pct"] = pd.to_numeric(out["call_wall_distance"], errors="coerce") * 100
    elif "breakout_distance" in out.columns:
        out["breakout_distance_pct"] = pd.to_numeric(
            out["breakout_distance"].astype(str).str.replace("%", "", regex=False),
            errors="coerce",
        )

    if "put_wall_distance" in out.columns:
        out["breakdown_distance_pct"] = pd.to_numeric(out["put_wall_distance"], errors="coerce") * 100
    elif "breakdown_distance" in out.columns:
        out["breakdown_distance_pct"] = pd.to_numeric(
            out["breakdown_distance"].astype(str).str.replace("%", "", regex=False),
            errors="coerce",
        )

    if "call_wall_breakout_trigger" in out.columns:
        out["breakout_trigger"] = out["call_wall_breakout_trigger"]
    if "put_wall_breakdown_trigger" in out.columns:
        out["breakdown_trigger"] = out["put_wall_breakdown_trigger"]

    out["nearest_trigger_distance_pct"] = out[["breakout_distance_pct", "breakdown_distance_pct"]].abs().min(axis=1)
    out["trigger_status"] = "WAIT"
    out.loc[out["nearest_trigger_distance_pct"] <= 0.25, "trigger_status"] = "NEAR_TRIGGER"
    out.loc[out["nearest_trigger_distance_pct"] <= 0.10, "trigger_status"] = "AT_TRIGGER"

    if not cards.empty and "ticker" in cards.columns:
        card_cols = [c for c in ["ticker", "allowed_action", "forbidden_action", "trigger_rule"] if c in cards.columns]
        out = out.merge(cards[card_cols], on="ticker", how="left")

    return out.sort_values("nearest_trigger_distance_pct", na_position="last").fillna("")


def build_decision_playbook() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "condition": "L8_state = RED",
            "meaning": "Portfolio risk overrides attractive research/options signals.",
            "allowed": "Research; tiny stock/ETF paper only after manual checks.",
            "forbidden": "Live trades; full-size conversion; short-dated options.",
            "next_check": "Stress Test, Portfolio Map, Before-Action Check",
        },
        {
            "condition": "trigger_status = AT_TRIGGER or NEAR_TRIGGER",
            "meaning": "Price is close to a monitored breakout/breakdown level.",
            "allowed": "Re-check gamma, kill zone, spread, liquidity, news, and stress.",
            "forbidden": "Chasing weekly OTM options or treating trigger as automatic entry.",
            "next_check": "Trigger Levels, Options Watch, Before-Action Check",
        },
        {
            "condition": "master_action = TINY_PAPER_ONLY",
            "meaning": "Research stack has enough interest, but risk/execution gates cap expression.",
            "allowed": "Tiny underlying paper test if manual checks are cleared.",
            "forbidden": "Live conversion; options expression; increasing reviewed exposure.",
            "next_check": "Single-Ticker Notebook, Paper Log",
        },
        {
            "condition": "master_action = RESEARCH_ONLY",
            "meaning": "Ticker still needs more evidence or cleaner data before expression.",
            "allowed": "Study the setup and fill missing data.",
            "forbidden": "Paper trade before L1/L5/L9 gaps are resolved.",
            "next_check": "Gap List, Evidence Board, Single-Ticker Notebook",
        },
        {
            "condition": "L1_state = NO_PRICE_OR_NO_DATA",
            "meaning": "The model cannot trust price-sensitive conclusions.",
            "allowed": "Refresh market data or add verified spot/close.",
            "forbidden": "Acting on stale/missing price assumptions.",
            "next_check": "Gap List, Run Status",
        },
        {
            "condition": "L5_state = EVENT_RISK",
            "meaning": "There may be earnings, SEC, insider, or news uncertainty.",
            "allowed": "Manual event review before any paper expression.",
            "forbidden": "Ignoring event timing just because technicals are strong.",
            "next_check": "Evidence Board, Before-Action Check",
        },
        {
            "condition": "L7 = gamma watch + kill zone",
            "meaning": "Options attention exists, but pinning/IV/theta risk can punish direction.",
            "allowed": "Watch levels; consider underlying paper only.",
            "forbidden": "Weekly OTM options, large premium spend, or chasing into wide spreads.",
            "next_check": "Options Watch, Trigger Levels",
        },
        {
            "condition": "L10 closed samples < 5",
            "meaning": "Learning layer is not statistically mature.",
            "allowed": "Record paper trades and attribution.",
            "forbidden": "Changing strategy weights from tiny test samples.",
            "next_check": "Paper Log",
        },
    ])


# ── Economic cycle stage playbook ──────────────────────────────────────────
_CYCLE_PLAYBOOK: dict = {
    "Mid-Cycle Growth": {
        "kicker": "EXPANSION INTACT",
        "description": "Tech and growth are leading. Small caps healthy. Rates backing up. Breadth broad.",
        "color": "#16a34a",
        "overweight": ["XLK", "SOXX", "SMH", "XLI", "XLY"],
        "neutral":    ["XLF", "SPY", "QQQ", "XLV"],
        "underweight":["XLU", "XLP", "TLT", "GLD"],
        "action": "Ride sector momentum leaders. Reduce defensives and long bonds. Favor growth over value.",
    },
    "Late Cycle": {
        "kicker": "GROWTH NARROWING",
        "description": "Expansion aging. Tech momentum fading. Energy and commodities starting to lead. Small caps lagging.",
        "color": "#d97706",
        "overweight": ["XLE", "XLF", "GLD", "XLV"],
        "neutral":    ["XLI", "SPY", "XLP"],
        "underweight":["XLK", "SOXX", "SMH", "TLT", "XLY"],
        "action": "Rotate from growth to value and energy. Trim semis/tech. Watch for cycle-peak signals.",
    },
    "Early Recovery": {
        "kicker": "RECOVERY STARTING",
        "description": "Emerging from a downturn. VIX falling. Breadth improving. Cyclicals and financials waking up first.",
        "color": "#2563eb",
        "overweight": ["XLF", "XLY", "SOXX", "SMH", "XLK"],
        "neutral":    ["QQQ", "XLI"],
        "underweight":["XLU", "XLP", "GLD", "TLT"],
        "action": "Buy early leaders: cyclicals, financials, semis. Underweight defensives. Size small until momentum confirms.",
    },
    "Defensive / Risk-Off": {
        "kicker": "RISK APPETITE SHRINKING",
        "description": "Macro headwinds. Breadth deteriorating. Avoid new cyclical longs.",
        "color": "#dc2626",
        "overweight": ["TLT", "GLD", "XLU", "XLP", "XLV"],
        "neutral":    ["SPY"],
        "underweight":["SOXX", "SMH", "XLK", "XLY", "XLE"],
        "action": "Rotate to defensives and bonds. No new cyclical or growth longs. Tiny paper only.",
    },
    "Recession / Crisis": {
        "kicker": "CAPITAL PRESERVATION MODE",
        "description": "Broad selloff. VIX extreme. Preserve capital first. No new longs.",
        "color": "#7f1d1d",
        "overweight": ["TLT", "GLD"],
        "neutral":    ["XLU"],
        "underweight":["All cyclical sectors"],
        "action": "No new positions. Reduce existing paper. Cash and bonds only.",
    },
    "Mixed / Uncertain": {
        "kicker": "NO CLEAR PHASE",
        "description": "Conflicting signals. No broad rotation call. Stay selective and ticker-specific.",
        "color": "#0891b2",
        "overweight": [],
        "neutral":    ["SPY", "QQQ"],
        "underweight":[],
        "action": "No broad rotation play. Focus on individual high-conviction setups with multi-layer support.",
    },
}


def build_cycle_stage(macro: pd.DataFrame, vol: pd.DataFrame, breadth: pd.DataFrame) -> dict:
    """Infer economic cycle stage from macro signals, VIX, and breadth data."""
    def _macro_row(ticker: str) -> dict:
        if macro.empty or "ticker" not in macro.columns:
            return {}
        rows = macro[macro["ticker"].astype(str).str.upper() == ticker.upper()]
        return rows.iloc[0].to_dict() if not rows.empty else {}

    spy = _macro_row("SPY"); qqq = _macro_row("QQQ")
    iwm = _macro_row("IWM"); tlt = _macro_row("TLT")

    spy_trend = str(spy.get("trend_state", "")).upper()
    iwm_trend = str(iwm.get("trend_state", "")).upper()
    tlt_trend = str(tlt.get("trend_state", "")).upper()
    qqq_ret20 = pd.to_numeric(qqq.get("ret_20d", 0), errors="coerce") or 0.0
    spy_ret20 = pd.to_numeric(spy.get("ret_20d", 0), errors="coerce") or 0.0
    qqq_vs_spy = qqq_ret20 - spy_ret20

    vix_level, vix_direction, vol_regime_str = 20.0, 0.0, "UNKNOWN"
    if not vol.empty and "metric" in vol.columns and "value" in vol.columns:
        def _v(m):
            r = vol[vol["metric"].astype(str).eq(m)]
            return r.iloc[0]["value"] if not r.empty else None
        vl = _v("vix_level");     vix_level     = float(pd.to_numeric(vl, errors="coerce") or 20.0) if vl is not None else 20.0
        vd = _v("vix_20d_change"); vix_direction = float(pd.to_numeric(vd, errors="coerce") or 0.0)  if vd is not None else 0.0
        vr = _v("vol_regime");    vol_regime_str = str(vr).upper() if vr is not None else "UNKNOWN"

    breadth_pct = 0.5
    if not breadth.empty and "above_50dma" in breadth.columns:
        breadth_pct = breadth["above_50dma"].astype(str).str.upper().isin(["TRUE", "1", "YES"]).mean()

    # ── stage logic ────────────────────────────────────────────────────────
    is_up   = spy_trend in ("UPTREND", "BULL")
    is_down = spy_trend in ("DOWNTREND", "BEAR")
    hi_vol  = vol_regime_str in ("HIGH", "EXTREME")
    lo_vol  = vol_regime_str == "LOW"

    if is_down and hi_vol and vix_direction > 0.05:
        stage = "Recession / Crisis"
    elif is_down or hi_vol:
        stage = "Defensive / Risk-Off"
    elif is_up and lo_vol:
        if iwm_trend in ("UPTREND", "BULL") and tlt_trend in ("DOWNTREND", "BEAR"):
            stage = "Mid-Cycle Growth" if qqq_vs_spy > 0.02 else "Late Cycle"
        elif iwm_trend in ("DOWNTREND", "BEAR"):
            stage = "Late Cycle"
        else:
            stage = "Mid-Cycle Growth"
    elif is_up and vol_regime_str == "ELEVATED":
        stage = "Early Recovery" if vix_direction < -0.05 and breadth_pct > 0.55 else "Mixed / Uncertain"
    else:
        stage = "Mixed / Uncertain"

    pb = _CYCLE_PLAYBOOK.get(stage, _CYCLE_PLAYBOOK["Mixed / Uncertain"])
    return {**pb, "stage": stage,
            "spy_trend": spy_trend, "iwm_trend": iwm_trend,
            "tlt_trend": tlt_trend, "vix_level": round(vix_level, 1),
            "vol_regime": vol_regime_str,
            "breadth_pct": f"{breadth_pct*100:.0f}%",
            "qqq_vs_spy": f"{qqq_vs_spy*100:+.1f}%"}


def build_focus_list() -> pd.DataFrame:
    master = read_csv(FILES["master_v2"])
    triggers = build_trigger_board(read_csv(FILES["watch_triggers"]), read_csv(FILES["action_cards"]))
    gaps = build_gap_queue(master, read_csv(FILES["market_snapshot"]))
    pretrade = read_csv(FILES["pre_trade"])

    if master.empty:
        return pd.DataFrame()

    out = master.copy()
    out["stack_score_num"] = pd.to_numeric(out.get("stack_score_avg", ""), errors="coerce").fillna(0)

    action_points = {
        "TINY_PAPER_ONLY": 30,
        "RESEARCH_ONLY": 10,
        "SKIP": -20,
    }
    out["action_points"] = out.get("master_action", "").map(action_points).fillna(0)

    if not triggers.empty and "ticker" in triggers.columns:
        trig_cols = [c for c in [
            "ticker", "trigger_status", "nearest_trigger_distance_pct",
            "breakout_trigger", "breakdown_trigger", "live_allowed"
        ] if c in triggers.columns]
        out = out.merge(triggers[trig_cols], on="ticker", how="left")
    else:
        out["trigger_status"] = ""
        out["nearest_trigger_distance_pct"] = ""

    out["trigger_points"] = 0
    out.loc[out["trigger_status"].astype(str).eq("NEAR_TRIGGER"), "trigger_points"] = 10
    out.loc[out["trigger_status"].astype(str).eq("AT_TRIGGER"), "trigger_points"] = 15

    if not gaps.empty and "ticker" in gaps.columns:
        high_gaps = gaps[gaps["priority"].eq("HIGH")].groupby("ticker").size().rename("high_gap_count")
        med_gaps = gaps[gaps["priority"].eq("MEDIUM")].groupby("ticker").size().rename("medium_gap_count")
        out = out.merge(high_gaps, on="ticker", how="left")
        out = out.merge(med_gaps, on="ticker", how="left")
    else:
        out["high_gap_count"] = 0
        out["medium_gap_count"] = 0

    out["high_gap_count"] = pd.to_numeric(out["high_gap_count"], errors="coerce").fillna(0)
    out["medium_gap_count"] = pd.to_numeric(out["medium_gap_count"], errors="coerce").fillna(0)

    if not pretrade.empty and "ticker" in pretrade.columns:
        pre_cols = [c for c in ["ticker", "final_status", "paper_allowed", "live_allowed"] if c in pretrade.columns]
        pre_one = pretrade[pre_cols].drop_duplicates("ticker", keep="first")
        out = out.merge(pre_one, on="ticker", how="left", suffixes=("", "_pretrade"))

    out["focus_score"] = (
        out["stack_score_num"]
        + out["action_points"]
        + out["trigger_points"]
        - out["high_gap_count"] * 12
        - out["medium_gap_count"] * 3
    )

    out["focus_bucket"] = "BACKLOG"
    out.loc[out["master_action"].astype(str).eq("TINY_PAPER_ONLY"), "focus_bucket"] = "PRIMARY_WATCH"
    out.loc[out["trigger_status"].astype(str).isin(["AT_TRIGGER", "NEAR_TRIGGER"]) & out["master_action"].astype(str).eq("TINY_PAPER_ONLY"), "focus_bucket"] = "ACTIVE_WATCH"
    out.loc[out["high_gap_count"] >= 2, "focus_bucket"] = "FIX_DATA_FIRST"
    out.loc[out["master_action"].astype(str).eq("SKIP"), "focus_bucket"] = "DO_NOT_TOUCH"

    cols = [c for c in [
        "ticker", "focus_bucket", "focus_score", "master_action", "master_reason",
        "stack_score_avg", "trigger_status", "nearest_trigger_distance_pct",
        "breakout_trigger", "breakdown_trigger", "L3_state", "L6_state", "L7_state",
        "L8_state", "L9_state", "high_gap_count", "medium_gap_count",
        "final_status", "paper_allowed", "live_allowed"
    ] if c in out.columns]

    out = out[cols].sort_values(["focus_score", "stack_score_avg"], ascending=False)
    return out.fillna("")


def build_today_action_queue() -> pd.DataFrame:
    focus = build_focus_list()
    pretrade = read_csv(FILES["pre_trade"])
    events = read_csv(FILES["events"])
    health = read_csv(FILES["data_source_health"])

    if focus.empty:
        return pd.DataFrame()

    out = focus.copy()

    if not pretrade.empty and "ticker" in pretrade.columns:
        pre_cols = [c for c in [
            "ticker", "risk_light", "final_status", "manual_news_check",
            "earnings_date_check", "liquidity_check", "spread_check",
            "duplicate_exposure_check", "stress_check", "paper_allowed", "live_allowed",
            "reasons"
        ] if c in pretrade.columns]
        pre_one = pretrade[pre_cols].drop_duplicates("ticker", keep="first")
        out = out.merge(pre_one, on="ticker", how="left", suffixes=("", "_l9"))

    if not events.empty and "ticker" in events.columns:
        event_cols = [c for c in ["ticker", "event_label", "event_reason"] if c in events.columns]
        event_one = events[event_cols].drop_duplicates("ticker", keep="first")
        out = out.merge(event_one, on="ticker", how="left")

    source_risk_count = 0
    if not health.empty and "status" in health.columns:
        source_risk_count = int(health["status"].astype(str).str.upper().eq("RISK").sum())

    def desk_status(row) -> str:
        bucket = str(row.get("focus_bucket", "")).upper()
        action = str(row.get("master_action", "")).upper()
        l8 = str(row.get("L8_state", "")).upper()
        final_status = str(row.get("final_status", row.get("final_status_l9", ""))).upper()
        trigger = str(row.get("trigger_status", "")).upper()
        if bucket == "DO_NOT_TOUCH" or action == "SKIP":
            return "DO_NOT_TOUCH"
        if bucket == "FIX_DATA_FIRST":
            return "FIX_DATA_FIRST"
        if "RED" in l8:
            return "RISK_REDUCTION_FIRST"
        if "PENDING" in final_status:
            return "MANUAL_GATE_FIRST"
        if trigger in {"AT_TRIGGER", "NEAR_TRIGGER"}:
            return "TRIGGER_REVIEW"
        if bucket in {"ACTIVE_WATCH", "PRIMARY_WATCH"}:
            return "WATCHLIST_REVIEW"
        return "BACKLOG"

    def priority(row) -> str:
        status = str(row.get("desk_status", ""))
        score = pd.to_numeric(row.get("focus_score", 0), errors="coerce")
        if status in {"FIX_DATA_FIRST", "RISK_REDUCTION_FIRST", "TRIGGER_REVIEW"}:
            return "HIGH"
        if status in {"MANUAL_GATE_FIRST", "WATCHLIST_REVIEW"}:
            return "MEDIUM"
        if pd.notna(score) and score >= 50:
            return "MEDIUM"
        return "LOW"

    def next_station(row) -> str:
        status = str(row.get("desk_status", ""))
        if status == "FIX_DATA_FIRST":
            return "System / Data Gaps"
        if status == "RISK_REDUCTION_FIRST":
            return "Risk / Risk Control"
        if status == "MANUAL_GATE_FIRST":
            return "Research / Pre-Trade Gate"
        if status == "TRIGGER_REVIEW":
            return "Daily Desk / Trigger Board"
        if status == "WATCHLIST_REVIEW":
            return "Daily Desk / Ticker Drilldown"
        if status == "DO_NOT_TOUCH":
            return "Decision Playbook"
        return "Research Lab"

    def allowed_now(row) -> str:
        l8 = str(row.get("L8_state", "")).upper()
        final_status = str(row.get("final_status", row.get("final_status_l9", ""))).upper()
        action = str(row.get("master_action", "")).upper()
        if action == "SKIP":
            return "Skip / no attention"
        if "RED" in l8:
            return "Research; risk reduction; tiny paper only after checks"
        if "PENDING" in final_status:
            return "Research only until manual checks clear"
        return "Research review"

    def blocked_by(row) -> str:
        blockers = []
        if str(row.get("L8_state", "")).upper() == "RED":
            blockers.append("L8 RED")
        if str(row.get("final_status", row.get("final_status_l9", ""))).upper() in {"PENDING_MANUAL_CHECKS", "BLOCKED"}:
            blockers.append(str(row.get("final_status", row.get("final_status_l9", ""))))
        if pd.to_numeric(row.get("high_gap_count", 0), errors="coerce") > 0:
            blockers.append(f"{int(float(row.get('high_gap_count', 0)))} high data gaps")
        if source_risk_count:
            blockers.append(f"{source_risk_count} source risks")
        event = str(row.get("event_label", "")).upper()
        if event == "EVENT_RISK":
            blockers.append("event risk")
        return "; ".join(blockers) if blockers else "none"

    out["desk_status"] = out.apply(desk_status, axis=1)
    out["priority"] = out.apply(priority, axis=1)
    out["next_station"] = out.apply(next_station, axis=1)
    out["allowed_now"] = out.apply(allowed_now, axis=1)
    out["blocked_by"] = out.apply(blocked_by, axis=1)

    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    out["_priority_rank"] = out["priority"].map(priority_order).fillna(3)
    out["_score_sort"] = pd.to_numeric(out.get("focus_score", 0), errors="coerce").fillna(0)
    out = out.sort_values(["_priority_rank", "_score_sort"], ascending=[True, False])

    cols = [c for c in [
        "priority", "ticker", "desk_status", "focus_bucket", "focus_score",
        "master_action", "trigger_status", "nearest_trigger_distance_pct",
        "allowed_now", "blocked_by", "next_station", "master_reason"
    ] if c in out.columns]
    out = out[cols].copy()
    if "focus_score" in out.columns:
        out["focus_score"] = pd.to_numeric(out["focus_score"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
    if "nearest_trigger_distance_pct" in out.columns:
        out["nearest_trigger_distance_pct"] = pd.to_numeric(out["nearest_trigger_distance_pct"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2f}%")
    return out.fillna("")


def build_layer_completeness(master: pd.DataFrame) -> pd.DataFrame:
    if master.empty:
        return pd.DataFrame([{
            "status": "MISSING",
            "layer": "ALL",
            "covered": 0,
            "gaps": 1,
            "context_rows": 0,
            "total_rows": 0,
            "top_states": "Main decision table missing",
        }])

    rows = []
    context_words = ["CONTEXT", "SYNTHETIC", "ETF_NOT_FUNDAMENTAL", "SECTOR_CONTEXT_ONLY"]
    gap_words = ["NO_DATA", "NO_PRICE", "MISSING"]

    for i in range(1, 11):
        layer = f"L{i}"
        col = f"{layer}_state"
        if col not in master.columns:
            rows.append({
                "status": "MISSING",
                "layer": layer,
                "covered": 0,
                "gaps": len(master),
                "context_rows": 0,
                "total_rows": len(master),
                "top_states": f"{col} missing",
            })
            continue

        states = master[col].astype(str).str.upper().fillna("")
        gap_mask = states.apply(lambda s: any(word in s for word in gap_words))
        context_mask = states.apply(lambda s: any(word in s for word in context_words))
        counts = states.value_counts().head(5)
        rows.append({
            "status": "OK" if int(gap_mask.sum()) == 0 else "MISSING",
            "layer": layer,
            "covered": int((~gap_mask).sum()),
            "gaps": int(gap_mask.sum()),
            "context_rows": int(context_mask.sum()),
            "total_rows": len(master),
            "top_states": "; ".join(f"{idx}={val}" for idx, val in counts.items()),
        })

    return pd.DataFrame(rows)


def build_qa_checks() -> pd.DataFrame:
    master = read_csv(FILES["master_v2"])
    technicals = read_csv(FILES["technicals"])
    fundamentals = read_csv(FILES["fundamentals"])
    pretrade = read_csv(FILES["pre_trade"])
    order_ticket = read_csv(FILES["pre_trade_order"])
    run_status = build_run_status()

    rows = []

    if not master.empty:
        completeness = build_layer_completeness(master)
        gap_layers = completeness[completeness["gaps"].astype(int) > 0]
        for _, gap in gap_layers.iterrows():
            rows.append({
                "severity": "HIGH",
                "area": "Layer completeness",
                "ticker": "SYSTEM",
                "issue": f"{gap['layer']} has {gap['gaps']} state gaps.",
                "evidence": str(gap["top_states"]),
                "fix": "Convert false NO_DATA into explicit context or refresh the source layer.",
            })

        tech_lookup = {}
        if not technicals.empty and "ticker" in technicals.columns:
            tech_lookup = {str(r["ticker"]): r.to_dict() for _, r in technicals.iterrows()}
        fund_lookup = {}
        if not fundamentals.empty and "ticker" in fundamentals.columns:
            fund_lookup = {str(r["ticker"]): r.to_dict() for _, r in fundamentals.iterrows()}

        for _, row in master.iterrows():
            ticker = str(row.get("ticker", ""))
            l1_state = str(row.get("L1_state", ""))
            l4_state = str(row.get("L4_state", ""))
            l6_state = str(row.get("L6_state", ""))
            action = str(row.get("master_action", ""))

            tech = tech_lookup.get(ticker, {})
            fund = fund_lookup.get(ticker, {})

            if "NO_PRICE" in l1_state and tech.get("data_status") == "OK" and str(tech.get("close", "")):
                rows.append({
                    "severity": "HIGH",
                    "area": "L1 vs L6",
                    "ticker": ticker,
                    "issue": "L1 says no price, but L6 has technical close.",
                    "evidence": f"L1={l1_state}; L6 close={tech.get('close')}; L6={l6_state}",
                    "fix": "Update L1 price-source logic to use technical close or refresh market snapshot.",
                })

            if "NO_PRICE" in l1_state and fund.get("data_status") == "OK" and str(fund.get("current_price", "")):
                rows.append({
                    "severity": "MEDIUM",
                    "area": "L1 vs L4",
                    "ticker": ticker,
                    "issue": "L1 says no price, but L4 has current_price.",
                    "evidence": f"L1={l1_state}; L4 current_price={fund.get('current_price')}; L4={l4_state}",
                    "fix": "Decide whether fundamentals current_price is trusted enough for L1.",
                })

            if action == "TINY_PAPER_ONLY" and str(row.get("L8_state", "")) != "RED":
                rows.append({
                    "severity": "MEDIUM",
                    "area": "Decision logic",
                    "ticker": ticker,
                    "issue": "Tiny paper action without L8 RED needs review.",
                    "evidence": f"master_action={action}; L8={row.get('L8_state', '')}",
                    "fix": "Confirm the action cap reason is still valid.",
                })

            if action == "SKIP" and str(row.get("L10_state", "")) == "NO_SAMPLE":
                rows.append({
                    "severity": "LOW",
                    "area": "Learning",
                    "ticker": ticker,
                    "issue": "Skipped ticker has no learning sample.",
                    "evidence": f"master_action={action}; L10={row.get('L10_state', '')}",
                    "fix": "No action required unless ticker returns to research queue.",
                })

    if not pretrade.empty and "live_allowed" in pretrade.columns:
        live_yes = pretrade[pretrade["live_allowed"].astype(str).str.upper().eq("YES")]
        if not live_yes.empty:
            rows.append({
                "severity": "HIGH",
                "area": "Safety",
                "ticker": "MULTI",
                "issue": "Some pre-trade rows allow live trading.",
                "evidence": ", ".join(live_yes["ticker"].astype(str).tolist()),
                "fix": "Review immediately; current project rule is no live orders.",
            })

    if not order_ticket.empty:
        rows.append({
            "severity": "HIGH",
            "area": "Safety",
            "ticker": "ORDER_TICKET",
            "issue": "Pre-trade order ticket is not empty.",
            "evidence": f"rows={len(order_ticket)}",
            "fix": "Confirm these are paper-only drafts and not broker orders.",
        })

    if not run_status.empty:
        stale = run_status[run_status["status"].isin(["STALE", "MISSING"])]
        for _, row in stale.iterrows():
            rows.append({
                "severity": "MEDIUM" if row["status"] == "STALE" else "HIGH",
                "area": "Run freshness",
                "ticker": "SYSTEM",
                "issue": f"{row['area']} output is {row['status']}.",
                "evidence": f"file={row['file']}; age_hours={row['age_hours']}",
                "fix": "Run the full v2 daily refresh before relying on decisions.",
            })

    if not rows:
        return pd.DataFrame([{
            "severity": "OK",
            "area": "System",
            "ticker": "ALL",
            "issue": "No consistency issues detected.",
            "evidence": "",
            "fix": "Continue normal workflow.",
        }])

    out = pd.DataFrame(rows)
    rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "OK": 3}
    out["_rank"] = out["severity"].map(rank).fillna(9)
    return out.sort_values(["_rank", "ticker", "area"]).drop(columns=["_rank"]).fillna("")


def tab_risk_stress():
    st.header("Risk And Stress Test")

    warnings = read_csv(FILES["exposure_warnings"])
    stress = read_csv(FILES["scenario_stress"])
    sizing = read_csv(FILES["position_sizing"])

    high = 0
    medium = 0
    worst = "N/A"
    worst_loss = "N/A"

    if not warnings.empty and "level" in warnings.columns:
        high = int((warnings["level"].astype(str).str.upper() == "HIGH").sum())
        medium = int((warnings["level"].astype(str).str.upper() == "MEDIUM").sum())

    if not stress.empty and "estimated_loss" in stress.columns:
        loss = pd.to_numeric(stress["estimated_loss"], errors="coerce")
        if not loss.dropna().empty:
            idx = loss.idxmin()
            worst = stress.loc[idx, "scenario"] if "scenario" in stress.columns else "N/A"
            worst_loss = f"{loss.loc[idx] * 100:.2f}%"

    render_layer_workbench_header(
        "L8",
        "Risk And Stress Test",
        "Stress tests estimate how much the portfolio would lose in a bad scenario. High warnings block aggressive action.",
        [
            ("Risk Light",  "RED" if high else ("WARN" if medium else "OK"), status_kind("RED" if high else ("WARN" if medium else "OK"))),
            ("High Warns",  high,        "risk"        if high   else "supportive"),
            ("Medium Warns",medium,      "wait"        if medium else "supportive"),
            ("Worst Case",  worst_loss,  "risk"),
        ],
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Light", "Red Light" if high else "Watch")
    c2.metric("High Risk Warnings", high)
    c3.metric("Medium Risk Warnings", medium)
    c4.metric("Worst Case", worst_loss)

    breach_1 = 0
    breach_2 = 0
    breach_5 = 0
    if not stress.empty:
        for col, name in [("breaches_1pct", "breach_1"), ("breaches_2pct", "breach_2"), ("breaches_5pct", "breach_5")]:
            if col in stress.columns:
                val = pd.to_numeric(stress[col], errors="coerce").fillna(0).sum()
                if name == "breach_1":
                    breach_1 = int(val)
                elif name == "breach_2":
                    breach_2 = int(val)
                else:
                    breach_5 = int(val)

    stress_summary = pd.DataFrame([
        {
            "status": "RISK" if worst_loss != "N/A" else "NO_DATA",
            "stress_question": "What is the worst stress case?",
            "current_read": f"{worst}; {worst_loss}",
            "why_it_matters": "This is the scenario to respect before any new paper exposure.",
            "what_to_do": "Use it as a risk cap, not as a prediction.",
            "source_file": "scenario_stress_results.csv",
        },
        {
            "status": "RISK" if high else ("REVIEW" if medium else "OK"),
            "stress_question": "Are concentration warnings already active?",
            "current_read": f"high={high}; medium={medium}",
            "why_it_matters": "Stress is more dangerous when exposure is already concentrated.",
            "what_to_do": "Resolve high warnings before adding new ideas.",
            "source_file": "exposure_warnings.csv",
        },
        {
            "status": "REVIEW" if breach_1 or breach_2 or breach_5 else "OK",
            "stress_question": "Which loss lines were breached?",
            "current_read": f"1%={breach_1}; 2%={breach_2}; 5%={breach_5}",
            "why_it_matters": "Breach counts show whether stress is small noise or account-level pain.",
            "what_to_do": "If breaches exist, reduce size before considering new risk.",
            "source_file": "scenario_stress_results.csv",
        },
    ])
    st.subheader("Stress Test Summary")
    render_badge_table(stress_summary, height=230)

    st.markdown(f"""
    <div class="command-grid">
      <div class="command-panel command-risk">
        <div class="command-label">Main Limit</div>
        <div class="command-title">Risk Layer First</div>
        <div class="command-text">Worst case: {escape(str(worst))}; possible impact: {escape(str(worst_loss))}. Do not let options heat or price strength override account risk.</div>
      </div>
      <div class="command-panel command-cyan">
        <div class="command-label">Size Rule</div>
        <div class="command-title">Reduce Duplicate Exposure First</div>
        <div class="command-text">Tech and semiconductor ETFs overlap with single names, so simplify that risk first.</div>
      </div>
      <div class="command-panel command-blocked">
        <div class="command-label">Not Allowed</div>
        <div class="command-title">Do Not Turn This Into A Large Position</div>
        <div class="command-text">Suggested sizes are only review ideas, not orders.</div>
      </div>
      <div class="command-panel command-paper">
        <div class="command-label">Allowed</div>
        <div class="command-title">Tiny Paper Test Only</div>
        <div class="command-text">Use tiny paper tests for learning, and keep review samples clean.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Position Risk Warnings")
    if not warnings.empty:
        render_badge_table(warnings, height=300)
    else:
        st.info("No exposure warnings.")

    st.subheader("What Happens If The Market Gets Worse")
    if not stress.empty:
        if _PLOTLY and "scenario" in stress.columns and "estimated_loss" in stress.columns:
            _st_df = stress.copy()
            _st_df["estimated_loss"] = pd.to_numeric(_st_df["estimated_loss"], errors="coerce").fillna(0)
            _st_df = _st_df.sort_values("estimated_loss", ascending=True)
            _st_colors = ["#b91c1c" if v <= -0.05 else "#f87171" if v < 0 else "#22d3ee" for v in _st_df["estimated_loss"]]
            _fig_stress = go.Figure(go.Bar(
                x=(_st_df["estimated_loss"] * 100).tolist(),
                y=_st_df["scenario"].astype(str).tolist(),
                orientation="h",
                marker_color=_st_colors,
                text=(_st_df["estimated_loss"] * 100).map(lambda x: f"{x:.2f}%").tolist(),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Loss: %{x:.2f}%<extra></extra>",
            ))
            _fig_stress.update_layout(
                height=max(180, len(_st_df) * 36 + 60),
                margin=dict(l=10, r=70, t=28, b=20),
                title=dict(text="Scenario Estimated Loss %", font=dict(size=12), x=0),
                xaxis_title="Estimated Loss %", yaxis_title="",
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(gridcolor="#e5e7eb", zeroline=True, zerolinecolor="#111827", zerolinewidth=1, ticksuffix="%"),
                yaxis=dict(gridcolor="#e5e7eb"),
                font=dict(family="Inter,sans-serif", size=12),
            )
            st.plotly_chart(_fig_stress, use_container_width=True)
        stress_view = format_percent_columns(stress, ["estimated_pnl", "estimated_loss"])
        render_badge_table(stress_view, height=260)
    else:
        st.info("No scenario stress results.")

    st.subheader("Size Plan")
    if not sizing.empty:
        sizing_view = format_percent_columns(sizing, [
            "planned_weight", "approved_weight", "effective_weight",
            "suggested_weight", "reduction_from_effective"
        ])
        cols = [c for c in [
            "ticker", "sleeve", "decision", "risk_bucket", "effective_weight",
            "suggested_weight", "reduction_from_effective", "suggested_action",
            "sizing_reason"
        ] if c in sizing_view.columns]
        render_badge_table(sizing_view[cols], height=420)
    else:
        st.info("No sizing recommendations.")

    with st.expander("Raw stress and sizing report"):
        st.markdown(read_md(FILES["stress_report"]))


def tab_risk_control():
    st.header("Risk Desk")
    st.caption("Start with portfolio risk: positions, stress tests, advanced risk, sizing, and paper-log discipline.")

    exposure = read_csv(FILES["exposure"])
    warnings = read_csv(FILES["exposure_warnings"])
    stress = read_csv(FILES["scenario_stress"])
    sizing = read_csv(FILES["position_sizing"])
    adv = read_csv(FILES["v8_adv_risk"])
    ledger = read_csv(FILES["paper_ledger"])
    master = read_csv(FILES["master_v2"])
    routes = build_risk_control_routes()

    high_warn = count_value(warnings, "level", "HIGH")
    med_warn = count_value(warnings, "level", "MEDIUM")
    adv_risk = count_value(adv, "status", "RISK")
    closed = count_contains(ledger, "status", "CLOSED")

    total_exposure = "N/A"
    biggest_exposure = "N/A"
    biggest_exposure_note = "No exposure file found."
    if not exposure.empty and "effective_weight" in exposure.columns:
        weights = pd.to_numeric(exposure["effective_weight"], errors="coerce")
        total = weights.sum()
        total_exposure = f"{total * 100:.1f}%"
        if not weights.dropna().empty:
            idx = weights.idxmax()
            ticker = exposure.loc[idx, "ticker"] if "ticker" in exposure.columns else "Largest row"
            sector = exposure.loc[idx, "sector"] if "sector" in exposure.columns else ""
            biggest_exposure = f"{ticker} · {weights.loc[idx] * 100:.1f}%"
            biggest_exposure_note = f"{ticker} is the largest listed exposure. {sector}".strip()

    worst_loss = "N/A"
    worst_scenario = "N/A"
    if not stress.empty and "estimated_loss" in stress.columns:
        losses = pd.to_numeric(stress["estimated_loss"], errors="coerce")
        if not losses.dropna().empty:
            idx = losses.idxmin()
            worst_loss = f"{losses.loc[idx] * 100:.2f}%"
            worst_scenario = stress.loc[idx, "scenario"] if "scenario" in stress.columns else "Worst stress row"

    reduce_first = "N/A"
    reduce_reason = "No sizing plan found."
    if not sizing.empty:
        sizing_view = sizing.copy()
        if "reduction_from_effective" in sizing_view.columns:
            sizing_view["_reduction_num"] = pd.to_numeric(sizing_view["reduction_from_effective"], errors="coerce").fillna(0)
            sizing_view = sizing_view.sort_values("_reduction_num", ascending=False)
        elif "effective_weight" in sizing_view.columns:
            sizing_view["_reduction_num"] = pd.to_numeric(sizing_view["effective_weight"], errors="coerce").fillna(0)
            sizing_view = sizing_view.sort_values("_reduction_num", ascending=False)
        if not sizing_view.empty:
            first = sizing_view.iloc[0]
            ticker = first.get("ticker", "N/A")
            action = friendly_value(first.get("suggested_action", first.get("decision", "Review")))
            reduce_first = f"{ticker} · {action}"
            reduce_reason = str(first.get("sizing_reason", first.get("risk_bucket", "Review size before adding risk.")))

    risk_light = "RED" if high_warn else ("WARN" if med_warn or adv_risk else "OK")
    render_layer_workbench_header(
        "Risk",
        "Portfolio Risk Control Desk",
        "Risk comes before signal expression. This page decides whether the research stack is allowed to become even tiny paper.",
        [
            ("Risk Light", risk_light, status_kind(risk_light)),
            ("Exposure", total_exposure, "cyan"),
            ("Worst Stress", worst_loss, "risk" if worst_loss != "N/A" else "blocked"),
            ("Closed Samples", closed, "supportive" if closed >= 30 else "cyan"),
        ],
    )

    # ── Circuit breaker status banner ──────────────────────────────────────
    _rc_cb = build_circuit_breaker(ledger)
    if _rc_cb["status"] in ("STOP", "WARN"):
        _rc_cb_color = "#b91c1c" if _rc_cb["status"] == "STOP" else "#f59e0b"
        st.markdown(
            f'<div style="background:{_rc_cb_color};color:white;padding:12px 18px;border-radius:6px;'
            f'font-size:14px;font-weight:700;margin-bottom:12px;">🚦 CIRCUIT BREAKER: {_rc_cb["message"]}</div>',
            unsafe_allow_html=True,
        )

    if _PLOTLY and not exposure.empty and "effective_weight" in exposure.columns:
        _exp = exposure.copy()
        _exp["effective_weight"] = pd.to_numeric(_exp["effective_weight"], errors="coerce").fillna(0)
        _by_sector = _exp.groupby("sector", as_index=False)["effective_weight"].sum().sort_values("effective_weight", ascending=True)
        _fig_exp = go.Figure(go.Bar(
            x=(_by_sector["effective_weight"] * 100).round(1).tolist(),
            y=_by_sector["sector"].tolist(),
            orientation="h",
            marker_color="#22d3ee",
            text=[f"{v:.1f}%" for v in (_by_sector["effective_weight"] * 100).round(1)],
            textposition="outside",
        ))
        _fig_exp.update_layout(
            height=max(160, len(_by_sector) * 34),
            margin=dict(l=0, r=40, t=18, b=4),
            title=dict(text="Exposure By Sector (%)", font=dict(size=11, color="#6b7280"), x=0),
            xaxis=dict(ticksuffix="%", gridcolor="#e5e7eb"),
            yaxis=dict(tickfont=dict(size=11)),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Inter,sans-serif", size=11),
        )
        _exp_c1, _exp_c2 = st.columns([2, 1])
        with _exp_c1:
            st.plotly_chart(_fig_exp, use_container_width=True)
        with _exp_c2:
            _total_w = float(_exp["effective_weight"].sum())
            st.metric("Total Exposure", f"{_total_w*100:.1f}%")
            st.metric("Positions", len(_exp))

    risk_desk = pd.DataFrame([
        {
            "status": risk_light,
            "risk_question": "Can we add risk today?",
            "current_read": friendly_value(risk_light),
            "why_it_matters": "Risk controls whether any research idea can become even a tiny paper test.",
            "what_to_do": "If this is red, reduce or review risk before looking for new ideas.",
            "source_file": "master_10_layer_decision_matrix_v2.csv; stress_position_sizing_report.md",
        },
        {
            "status": "RISK" if high_warn else ("REVIEW" if med_warn else "OK"),
            "risk_question": "What is the biggest warning?",
            "current_read": f"high={high_warn}; medium={med_warn}",
            "why_it_matters": "Warnings identify concentration, overlap, and fragile exposures.",
            "what_to_do": "Read high warnings first; do not let options or price strength jump the line.",
            "source_file": "exposure_warnings.csv; exposure_dashboard.csv",
        },
        {
            "status": "REVIEW" if biggest_exposure != "N/A" else "NO_DATA",
            "risk_question": "Where is exposure largest?",
            "current_read": biggest_exposure,
            "why_it_matters": biggest_exposure_note,
            "what_to_do": "Check whether this exposure overlaps with ETFs, themes, or single names.",
            "source_file": "exposure_dashboard.csv",
        },
        {
            "status": "RISK" if worst_loss != "N/A" else "NO_DATA",
            "risk_question": "What is the worst stress case?",
            "current_read": f"{worst_scenario}; {worst_loss}",
            "why_it_matters": "The worst scenario sets the pain point before any new paper idea.",
            "what_to_do": "Use the worst case as a cap, not as a prediction.",
            "source_file": "scenario_stress_results.csv; stress_position_sizing_report.md",
        },
        {
            "status": "REVIEW" if reduce_first != "N/A" else "NO_DATA",
            "risk_question": "What should shrink or stay capped first?",
            "current_read": reduce_first,
            "why_it_matters": reduce_reason,
            "what_to_do": "Treat suggested size as a maximum review cap, not a buy instruction.",
            "source_file": "position_sizing_recommendations.csv",
        },
        {
            "status": "REVIEW" if closed < 5 else "OK",
            "risk_question": "Can learning change risk settings yet?",
            "current_read": f"closed paper samples={closed}",
            "why_it_matters": "A small sample should be recorded, not overfit into new risk weights.",
            "what_to_do": "Keep paper tests tiny and attribution-clean until the sample is larger.",
            "source_file": "paper_portfolio_ledger.csv; learning_attribution_report.md",
        },
    ])
    st.subheader("Risk Desk Summary")
    render_badge_table(risk_desk, height=380)

    with st.expander("Open Risk Source Map"):
        st.markdown("**Exposure:** `exposure_dashboard.csv`, `exposure_warnings.csv`")
        st.markdown("**Stress:** `scenario_stress_results.csv`, `stress_position_sizing_report.md`")
        st.markdown("**Sizing:** `position_sizing_recommendations.csv`")
        st.markdown("**More Risk Checks:** `v8_advanced_risk_summary.csv`, `v8_pca_exposure.csv`, `v8_tail_dependence.csv`")
        st.markdown("**Final Decision:** `master_10_layer_decision_matrix_v2.csv`")
        st.markdown("**Learning:** `paper_portfolio_ledger.csv`, `learning_attribution_report.md`")

    fix_rows = []
    if not warnings.empty:
        warn_view = warnings.copy()
        if "level" in warn_view.columns:
            warn_view["_rank"] = warn_view["level"].astype(str).str.upper().map({"HIGH": 0, "MEDIUM": 1, "LOW": 2}).fillna(3)
            warn_view = warn_view.sort_values("_rank")
        for _, row in warn_view.head(4).iterrows():
            fix_rows.append({
                "status": row.get("level", "REVIEW"),
                "priority_item": row.get("issue", "Exposure warning"),
                "why_first": row.get("detail", "This warning can cap new risk."),
                "next_step": row.get("action", "Review exposure before adding ideas."),
                "source_file": "exposure_warnings.csv",
            })
    if not sizing.empty:
        size_view = sizing.copy()
        if "reduction_from_effective" in size_view.columns:
            size_view["_rank"] = pd.to_numeric(size_view["reduction_from_effective"], errors="coerce").fillna(0)
            size_view = size_view.sort_values("_rank", ascending=False)
        for _, row in size_view.head(4).iterrows():
            ticker = row.get("ticker", "Ticker")
            action = friendly_value(row.get("suggested_action", row.get("decision", "Review")))
            fix_rows.append({
                "status": "REVIEW",
                "priority_item": f"{ticker} size cap",
                "why_first": row.get("sizing_reason", row.get("risk_bucket", "Sizing row needs review.")),
                "next_step": f"Treat {action} as a cap, not an instruction to buy.",
                "source_file": "position_sizing_recommendations.csv",
            })
    if fix_rows:
        st.subheader("What To Fix First")
        render_badge_table(pd.DataFrame(fix_rows).head(8), height=360)

    risk_tickers = []
    for source_df in [sizing, exposure, master]:
        if not source_df.empty and "ticker" in source_df.columns:
            risk_tickers.extend(source_df["ticker"].dropna().astype(str).tolist())
    risk_tickers = sorted(dict.fromkeys(risk_tickers))
    if risk_tickers:
        st.subheader("Open One Ticker Risk Explanation")
        risk_ticker = st.selectbox("Choose Ticker For Risk Review", risk_tickers, key="risk_desk_ticker")
        exposure_row = first_row(exposure, risk_ticker)
        sizing_row = first_row(sizing, risk_ticker)
        master_row = first_row(master, risk_ticker)

        ticker_warning_text = ""
        if not warnings.empty:
            warn_text_cols = [c for c in ["issue", "detail", "action"] if c in warnings.columns]
            if warn_text_cols:
                warn_mask = warnings[warn_text_cols].astype(str).agg(" ".join, axis=1).str.contains(str(risk_ticker), case=False, regex=False, na=False)
                ticker_warnings = warnings[warn_mask]
                if not ticker_warnings.empty:
                    ticker_warning_text = "; ".join(ticker_warnings.head(3)[warn_text_cols].astype(str).agg(" | ".join, axis=1).tolist())

        ticker_risk_rows = pd.DataFrame([
            {
                "status": row_value(master_row, "L8_state", default="NO_DATA"),
                "check": "Risk Light",
                "current_read": row_value(master_row, "L8_note", master_row.get("master_reason", ""), default="No ticker-level risk note."),
                "what_to_do": "If this is red, do not let price trend or options heat upgrade the action.",
                "source_file": "master_10_layer_decision_matrix_v2.csv",
            },
            {
                "status": "REVIEW" if exposure_row else "NO_DATA",
                "check": "Exposure",
                "current_read": (
                    f"weight={row_value(exposure_row, 'effective_weight', default='N/A')}; "
                    f"sector={row_value(exposure_row, 'sector', default='N/A')}; "
                    f"theme={row_value(exposure_row, 'theme', default='N/A')}"
                ),
                "what_to_do": "Check whether this is a core exposure or duplicated through ETFs/themes.",
                "source_file": "exposure_dashboard.csv",
            },
            {
                "status": "REVIEW" if sizing_row else "NO_DATA",
                "check": "Size Cap",
                "current_read": (
                    f"effective={row_value(sizing_row, 'effective_weight', default='N/A')}; "
                    f"suggested={row_value(sizing_row, 'suggested_weight', default='N/A')}; "
                    f"action={friendly_value(row_value(sizing_row, 'suggested_action', sizing_row.get('decision', ''), default='N/A'))}"
                ),
                "what_to_do": row_value(sizing_row, "sizing_reason", default="Treat size as a cap, not a buy instruction."),
                "source_file": "position_sizing_recommendations.csv",
            },
            {
                "status": "REVIEW" if ticker_warning_text else "OK",
                "check": "Warning Match",
                "current_read": ticker_warning_text or "No ticker-specific warning text found.",
                "what_to_do": "If warning text exists, resolve it before adding exposure.",
                "source_file": "exposure_warnings.csv",
            },
        ])
        render_badge_table(ticker_risk_rows, height=310)

    st.subheader("How To Use The Risk Page")
    render_badge_table(routes, height=320)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Most Important Risk Warnings")
        if warnings.empty:
            st.info("No exposure warnings.")
        else:
            render_badge_table(warnings.head(8), height=300)

    with c2:
        st.subheader("Worst Stress Test")
        if stress.empty:
            st.info("No scenario stress output.")
        else:
            view = stress.copy()
            if "estimated_loss" in view.columns:
                view["_loss_num"] = pd.to_numeric(view["estimated_loss"], errors="coerce")
                view = view.sort_values("_loss_num").drop(columns=["_loss_num"])
            view = format_percent_columns(view, ["estimated_pnl", "estimated_loss"])
            render_badge_table(view.head(8), height=300)

    st.subheader("More Risk Checks Snapshot")
    if adv.empty:
        st.info("No advanced risk output yet.")
    else:
        cols = [c for c in ["status", "data_source", "metric", "value", "interpretation"] if c in adv.columns]
        render_badge_table(adv[cols], height=320)


def risk_summary_value(summary: pd.DataFrame, metric: str, percent: bool = True) -> str:
    if summary.empty or "metric" not in summary.columns or "value" not in summary.columns:
        return "N/A"
    rows = summary[summary["metric"].astype(str).eq(metric)]
    if rows.empty:
        return "N/A"
    val = pd.to_numeric(rows["value"].iloc[0], errors="coerce")
    if pd.isna(val):
        return str(rows["value"].iloc[0])
    if percent:
        return f"{val * 100:.2f}%"
    return f"{val:.4f}"


def build_tail_pairs(tail: pd.DataFrame) -> pd.DataFrame:
    if tail.empty or "ticker" not in tail.columns:
        return pd.DataFrame()
    rows = []
    tickers = [str(t) for t in tail["ticker"].tolist()]
    for _, row in tail.iterrows():
        left = str(row.get("ticker", ""))
        for right in tickers:
            if not left or not right or left >= right or right not in tail.columns:
                continue
            val = pd.to_numeric(row.get(right, ""), errors="coerce")
            if pd.isna(val):
                continue
            rows.append({
                "status": "RISK" if val >= 0.45 else ("WARN" if val >= 0.30 else "OK"),
                "pair": f"{left} / {right}",
                "tail_dependence": f"{val:.3f}",
                "interpretation": "Higher means these names tend to fall together in downside tails.",
            })
    return pd.DataFrame(rows).sort_values("tail_dependence", ascending=False).head(15) if rows else pd.DataFrame()


def tab_advanced_risk():
    st.header("More Risk Checks")

    summary = read_csv(FILES["v8_adv_risk"])
    pca = read_csv(FILES["v8_pca"])
    tail = read_csv(FILES["v8_tail"])

    if summary.empty:
        st.warning("No advanced risk output yet. Run Step 59 first.")
        st.code("python3 -u canyon_final_v9_step59_v8_advanced_risk_bridge.py", language="bash")
        return

    data_source = summary["data_source"].iloc[0] if "data_source" in summary.columns else "UNKNOWN"
    risk_flags = int((summary.get("status", pd.Series(dtype=str)).astype(str).str.upper() == "RISK").sum())

    render_layer_workbench_header(
        "L8+",
        "Advanced Risk — Simulation And Factor Checks",
        "Adds GBM simulation, PCA factor exposure, and tail-dependence analysis on top of the core L8 risk check. Research use only.",
        [
            ("Data Source",   data_source,   "wait"),
            ("Risk Flags",    risk_flags,    "risk" if risk_flags else "supportive"),
            ("21d Sim Loss",  risk_summary_value(summary, "gbm_var_21d_95"),    "risk"),
            ("Tail Extreme",  risk_summary_value(summary, "copula_tail_var_5"), "risk"),
        ],
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data Sources", data_source)
    c2.metric("Risk Flags", risk_flags)
    c3.metric("Possible 21-Day Loss", risk_summary_value(summary, "gbm_var_21d_95"))
    c4.metric("Possible Extreme Loss", risk_summary_value(summary, "copula_tail_var_5"))

    more_risk_summary = pd.DataFrame([
        {
            "status": "RISK" if risk_flags else "OK",
            "risk_question": "Did the extra risk checks find flags?",
            "current_read": f"{risk_flags} risk flags",
            "why_it_matters": "These checks look for hidden factor, simulation, and downside-tail risk.",
            "what_to_do": "Use this as review material before any paper test.",
            "source_file": "v8_advanced_risk_summary.csv",
        },
        {
            "status": "REVIEW" if str(data_source).upper() == "SYNTHETIC_FALLBACK" else "OK",
            "risk_question": "Can we trust the data source?",
            "current_read": str(data_source),
            "why_it_matters": "Fallback data is useful for plumbing checks, not strong market conclusions.",
            "what_to_do": "If fallback is active, keep conclusions research-only.",
            "source_file": "v8_advanced_risk_summary.csv; data_source_health.csv",
        },
        {
            "status": "REVIEW",
            "risk_question": "How should this page affect action?",
            "current_read": "Diagnostic only",
            "why_it_matters": "Extra risk models should not create orders or override L8/L9.",
            "what_to_do": "Let this page reduce confidence or size; never use it to upgrade action.",
            "source_file": "v8_pca_exposure.csv; v8_tail_dependence.csv",
        },
    ])
    st.subheader("More Risk Checks Summary")
    render_badge_table(more_risk_summary, height=230)

    panel_class = "command-risk" if risk_flags else "command-cyan"
    source_note = (
        "This run is using simulated fallback data, so treat these numbers as system checks, not market decisions."
        if data_source == "SYNTHETIC_FALLBACK"
        else "This run found online price history."
    )
    st.markdown(f"""
    <div class="command-grid">
      <div class="command-panel {panel_class}">
        <div class="command-label">More Risk Checks</div>
        <div class="command-title">{risk_flags} Risk Flags</div>
        <div class="command-text">These models are diagnostic tools only; they will not change position size by themselves.</div>
      </div>
      <div class="command-panel command-cyan">
        <div class="command-label">Data Sources</div>
        <div class="command-title">{escape(str(data_source))}</div>
        <div class="command-text">{escape(source_note)}</div>
      </div>
      <div class="command-panel command-blocked">
        <div class="command-label">Safety Rules</div>
        <div class="command-title">No Live Orders</div>
        <div class="command-text">This page will not connect to a broker or submit orders.</div>
      </div>
      <div class="command-panel command-paper">
        <div class="command-label">How To Use It</div>
        <div class="command-title">Use It As Risk Review Material</div>
        <div class="command-text">Before any paper test, use it to understand crowding and extreme downside risk.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Main Risk Drivers", "Extreme Co-Move", "Report"])
    with tab1:
        view = summary.copy()
        if "value" in view.columns:
            view["value"] = pd.to_numeric(view["value"], errors="coerce").map(
                lambda x: "" if pd.isna(x) else f"{x:.4f}"
            )
        render_badge_table(view, height=360)

    with tab2:
        st.caption(
            "PCA factor exposure shows which market forces (equity momentum, rate sensitivity, sector tilt, volatility) "
            "most explain the portfolio's variance. A high loading on a single factor means that factor drives P&L. "
            "Use this to check for hidden concentration, not as a trade signal."
        )
        if pca.empty:
            st.info("No PCA factor output yet. Run Step 59 to generate it.")
        else:
            if _PLOTLY and "metric" in pca.columns and "value" in pca.columns:
                _pca_chart = pca.copy()
                _pca_chart["value_num"] = pd.to_numeric(_pca_chart["value"], errors="coerce")
                _pca_chart = _pca_chart.dropna(subset=["value_num"]).sort_values("value_num", ascending=True).head(20)
                if not _pca_chart.empty:
                    _pca_colors = ["#b91c1c" if v >= 0.3 else "#22d3ee" if v >= 0 else "#f87171" for v in _pca_chart["value_num"]]
                    _fig_pca = go.Figure(go.Bar(
                        x=_pca_chart["value_num"].tolist(),
                        y=_pca_chart["metric"].astype(str).tolist(),
                        orientation="h",
                        marker_color=_pca_colors,
                        text=_pca_chart["value_num"].map(lambda x: f"{x:.4f}").tolist(),
                        textposition="outside",
                        hovertemplate="<b>%{y}</b><br>%{x:.4f}<extra></extra>",
                    ))
                    _fig_pca.update_layout(
                        height=max(180, len(_pca_chart) * 32 + 60),
                        margin=dict(l=10, r=70, t=24, b=10),
                        title=dict(text="PCA Factor Loadings", font=dict(size=12), x=0),
                        xaxis_title="Loading", yaxis_title="",
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(gridcolor="#e5e7eb", zeroline=True, zerolinecolor="#111827", zerolinewidth=1),
                        yaxis=dict(gridcolor="#e5e7eb"),
                        font=dict(family="Inter,sans-serif", size=12),
                    )
                    st.plotly_chart(_fig_pca, use_container_width=True)
            pca_view = pca.copy()
            if "value" in pca_view.columns:
                nums = pd.to_numeric(pca_view["value"], errors="coerce")
                pca_view["value"] = [
                    f"{x:.4f}" if not pd.isna(x) else str(v)
                    for x, v in zip(nums, pca_view["value"])
                ]
            render_badge_table(pca_view, height=420)

    with tab3:
        pairs = build_tail_pairs(tail)
        st.subheader("Highest Shared Downside Risk")
        if not pairs.empty:
            render_badge_table(pairs, height=360)
        else:
            st.info("No tail-pair output yet.")
        st.subheader("What Moves Together In Extreme Cases")
        if not tail.empty:
            tail_view = tail.copy()
            for col in tail_view.columns:
                if col != "ticker":
                    nums = pd.to_numeric(tail_view[col], errors="coerce")
                    tail_view[col] = nums.map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
            render_badge_table(tail_view, height=460)
        else:
            st.info("No tail matrix yet.")

    with tab4:
        st.markdown(read_md(FILES["v8_adv_risk_report"]))


def tab_portfolio_map():
    exposure = read_csv(FILES["exposure"])
    sizing = read_csv(FILES["position_sizing"])
    warnings = read_csv(FILES["exposure_warnings"])

    if exposure.empty:
        st.warning("No exposure dashboard found.")
        return

    exposure_num = exposure.copy()
    exposure_num["effective_weight"] = pd.to_numeric(exposure_num["effective_weight"], errors="coerce")
    total = exposure_num["effective_weight"].sum()
    largest_row = exposure_num.sort_values("effective_weight", ascending=False).head(1)
    largest = "N/A"
    if not largest_row.empty:
        largest = f"{largest_row.iloc[0]['ticker']} ({largest_row.iloc[0]['effective_weight'] * 100:.1f}%)"

    risk_bucket = aggregate_weight(exposure, "risk_bucket")
    top_bucket = "N/A"
    if not risk_bucket.empty:
        top_bucket = f"{risk_bucket.iloc[0]['risk_bucket']} ({risk_bucket.iloc[0]['effective_weight'] * 100:.1f}%)"

    sector_weight = aggregate_weight(exposure, "sector")
    top_sector = "N/A"
    if not sector_weight.empty:
        top_sector = f"{sector_weight.iloc[0]['sector']} ({sector_weight.iloc[0]['effective_weight'] * 100:.1f}%)"

    high_warnings = 0 if warnings.empty or "level" not in warnings.columns else int((warnings["level"] == "HIGH").sum())
    medium_warnings = 0 if warnings.empty or "level" not in warnings.columns else int((warnings["level"] == "MEDIUM").sum())

    sizing_review = 0
    if not sizing.empty and "suggested_action" in sizing.columns:
        sizing_review = int(sizing["suggested_action"].astype(str).str.upper().str.contains("REVIEW|REDUCE|SKIP", regex=True, na=False).sum())

    _pm_state = "RISK" if high_warnings else ("WAIT" if (medium_warnings or sizing_review) else "OK")
    render_layer_workbench_header(
        "L8",
        "Portfolio Map — Exposure And Overlap",
        "Shows how exposure is spread by sector, risk group, and ticker. Overlap warnings block new paper action.",
        [
            ("Total Exposure", f"{total * 100:.1f}%",  "cyan"),
            ("Largest Ticker", largest,                 "cyan"),
            ("High Overlap",   high_warnings,           "risk"      if high_warnings   else "supportive"),
            ("Size Review",    sizing_review,           "wait"      if sizing_review   else "supportive"),
        ],
    )

    exposure_map_summary = pd.DataFrame([
        {
            "status": "REVIEW" if largest != "N/A" else "NO_DATA",
            "map_question": "Which ticker is largest?",
            "current_read": largest,
            "why_it_matters": "A single ticker can quietly drive account-level risk.",
            "what_to_do": "Check whether this is intended core exposure or duplicated exposure.",
            "source_file": "exposure_dashboard.csv",
        },
        {
            "status": "REVIEW" if top_bucket != "N/A" else "NO_DATA",
            "map_question": "Which risk group is largest?",
            "current_read": top_bucket,
            "why_it_matters": "A risk group can hide several tickers moving together.",
            "what_to_do": "Avoid adding new ideas from the same risk group until exposure is reviewed.",
            "source_file": "exposure_dashboard.csv",
        },
        {
            "status": "REVIEW" if top_sector != "N/A" else "NO_DATA",
            "map_question": "Which sector is largest?",
            "current_read": top_sector,
            "why_it_matters": "Sector concentration can make different tickers behave like one bet.",
            "what_to_do": "Compare ETF exposure with single-name exposure before adding risk.",
            "source_file": "exposure_dashboard.csv",
        },
        {
            "status": "RISK" if high_warnings else ("REVIEW" if medium_warnings else "OK"),
            "map_question": "How many overlap warnings exist?",
            "current_read": f"high={high_warnings}; medium={medium_warnings}",
            "why_it_matters": "Overlap warnings identify repeated exposure across ETFs and components.",
            "what_to_do": "Resolve high warnings before any new paper expression.",
            "source_file": "exposure_warnings.csv",
        },
        {
            "status": "REVIEW" if sizing_review else "OK",
            "map_question": "How many names need size review?",
            "current_read": str(sizing_review),
            "why_it_matters": "Size review rows should be treated as caps, not buy instructions.",
            "what_to_do": "Open Size Change before treating any ticker as allowed.",
            "source_file": "position_sizing_recommendations.csv",
        },
    ])
    st.subheader("Exposure Map Summary")
    render_badge_table(exposure_map_summary, height=300)

    st.markdown("""
    <div class="command-grid">
      <div class="command-panel command-risk">
        <div class="command-label">Main Problem</div>
        <div class="command-title">Theme Exposure Is Too Concentrated</div>
        <div class="command-text">Tech-growth and semiconductor exposure is large enough to affect total account risk.</div>
      </div>
      <div class="command-panel command-cyan">
        <div class="command-label">Position Discipline</div>
        <div class="command-title">Use One Clean Expression Per Idea</div>
        <div class="command-text">If an ETF and its components overlap, choose the cleaner expression instead of stacking both.</div>
      </div>
      <div class="command-panel command-paper">
        <div class="command-label">Paper Test</div>
        <div class="command-title">Learn From Small Samples</div>
        <div class="command-text">Paper tests should help Layer 10 learning, not quietly become full portfolio risk.</div>
      </div>
      <div class="command-panel command-blocked">
        <div class="command-label">Do Not Do This</div>
        <div class="command-title">Do Not Treat A Watchlist As A Position</div>
        <div class="command-text">This is a risk map, not permission to deploy the whole account.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    sub = st.tabs(["Risk Group", "Sector", "Ticker", "Size Change", "Warning", "Correlation", "Beta Exposure"])

    with sub[0]:
        st.subheader("By Risk Group")
        chart_weight(exposure, "risk_bucket")
        rb = aggregate_weight(exposure, "risk_bucket")
        render_badge_table(format_percent_columns(rb, ["effective_weight"]), height=260)

    with sub[1]:
        st.subheader("By Sector — Total Weight")
        chart_weight(exposure, "sector")
        st.subheader("By Sector — Sleeve Breakdown")
        chart_sleeve_breakdown(sizing)
        sec = aggregate_weight(exposure, "sector")
        render_badge_table(format_percent_columns(sec, ["effective_weight"]), height=300)

    with sub[2]:
        st.subheader("By Ticker")
        tick = exposure.copy()
        tick["effective_weight"] = pd.to_numeric(tick["effective_weight"], errors="coerce")
        tick = tick.sort_values("effective_weight", ascending=False)
        if _PLOTLY and not tick.empty:
            _tk_pct = tick["effective_weight"].fillna(0) * 100
            _tk_colors = ["#b91c1c" if v >= 12 else "#2563eb" if v >= 6 else "#22d3ee"
                          for v in _tk_pct]
            _fig_tk = go.Figure(go.Bar(
                x=_tk_pct.tolist(),
                y=tick["ticker"].astype(str).tolist(),
                orientation="h",
                marker_color=_tk_colors,
                text=_tk_pct.map(lambda x: f"{x:.1f}%").tolist(),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>",
            ))
            _fig_tk.update_layout(
                height=max(240, 60 + len(tick) * 44),
                margin=dict(l=10, r=60, t=10, b=20),
                xaxis_title="Weight %", yaxis_title="",
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(gridcolor="#e5e7eb", ticksuffix="%"),
                yaxis=dict(gridcolor="#e5e7eb", autorange="reversed"),
                font=dict(family="Inter,sans-serif", size=13),
            )
            st.plotly_chart(_fig_tk, use_container_width=True)
        render_badge_table(format_percent_columns(tick, ["effective_weight"]), height=420)

    with sub[3]:
        st.subheader("Current Size vs Suggested Size")
        if not sizing.empty:
            sizing_view = format_percent_columns(sizing, [
                "planned_weight", "approved_weight", "effective_weight",
                "suggested_weight", "reduction_from_effective"
            ])
            cols = [c for c in [
                "ticker", "sleeve", "risk_bucket", "effective_weight", "suggested_weight",
                "reduction_from_effective", "suggested_action", "sizing_reason"
            ] if c in sizing_view.columns]
            render_badge_table(sizing_view[cols], height=460)
        else:
            st.info("No sizing data.")

    with sub[4]:
        st.subheader("Concentration And Overlap Warnings")
        if not warnings.empty:
            render_badge_table(warnings, height=360)
        else:
            st.info("No exposure warnings.")

    with sub[5]:
        st.subheader("Return Correlation Between Holdings")
        st.caption("Based on multi-period returns from technical data. Correlation > 0.7 means these names move together — true diversification is lower than it looks.")
        tech_for_corr = read_csv(FILES["technicals"])
        if tech_for_corr.empty or "ticker" not in tech_for_corr.columns:
            st.info("No technical data for correlation matrix.")
        else:
            _ret_cols = [c for c in ["ret_5d", "ret_20d", "ret_63d"] if c in tech_for_corr.columns]
            if not _ret_cols:
                st.info("No return columns available for correlation.")
            else:
                _corr_pivot = tech_for_corr[["ticker"] + _ret_cols].copy()
                for c in _ret_cols:
                    _corr_pivot[c] = pd.to_numeric(_corr_pivot[c], errors="coerce")
                _corr_pivot = _corr_pivot.dropna(subset=_ret_cols)
                _corr_pivot = _corr_pivot.set_index("ticker")[_ret_cols].T
                if _corr_pivot.shape[1] >= 2:
                    _corr_matrix = _corr_pivot.corr()
                    _tickers_corr = _corr_matrix.columns.tolist()
                    _z = _corr_matrix.values.tolist()
                    _z_text = [[f"{v:.2f}" if isinstance(v, float) and not pd.isna(v) else "" for v in row] for row in _z]
                    if _PLOTLY:
                        _fig_corr = go.Figure(go.Heatmap(
                            z=_z,
                            x=_tickers_corr,
                            y=_tickers_corr,
                            text=_z_text,
                            texttemplate="%{text}",
                            colorscale="RdYlGn",
                            zmin=-1, zmax=1,
                            colorbar=dict(title="Correlation"),
                            hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.2f}<extra></extra>",
                        ))
                        _fig_corr.update_layout(
                            height=max(300, len(_tickers_corr) * 40 + 100),
                            margin=dict(l=10, r=10, t=36, b=10),
                            title=dict(text="Ticker Return Correlation Matrix (green=low, red=high)", font=dict(size=12), x=0),
                            plot_bgcolor="white", paper_bgcolor="white",
                            font=dict(family="Inter,sans-serif", size=11),
                        )
                        st.plotly_chart(_fig_corr, use_container_width=True)
                    st.markdown("""
**Reading the heatmap:**
- **Dark red (near 1.0)** — these tickers move almost identically. Holding both = concentrated risk.
- **Green (near 0 or negative)** — these tickers are genuinely diversifying each other.
- **Rule:** If two names have correlation > 0.8, treat them as ONE position for sizing purposes.
""")
                    render_badge_table(_corr_matrix.reset_index().rename(columns={"index": "ticker"}), height=320)
                else:
                    st.info("Need at least 2 tickers with return data to compute correlations.")

    with sub[6]:  # "Beta Exposure" — new 7th sub-tab
        st.subheader("Beta-Adjusted Market Exposure")
        st.caption("Nominal weight × beta = true market risk. A 20% position in a β=1.6 stock = 32% effective market exposure.")
        _ba_tech = read_csv(FILES["technicals"])
        _ba_mkt  = read_csv(FILES["market_snapshot"])
        beta_df = build_beta_adjusted_exposure(exposure, _ba_tech, _ba_mkt)
        if beta_df.empty:
            st.info("Need exposure and technical data (with beta column) to compute beta-adjusted exposure.")
        else:
            # Summary
            beta_adj_vals = beta_df["beta_adj_weight"].str.replace("%", "").astype(float)
            total_beta_adj = beta_adj_vals.sum()
            oversized = len(beta_df[beta_df["status"] == "RISK"])
            _ba_c1, _ba_c2, _ba_c3 = st.columns(3)
            _n_no_beta = int((beta_df["note"].str.contains("no data", case=False, na=False)).sum())
            _ba_c1.metric("Total Beta-Adj Exposure", f"{total_beta_adj:.1f}%")
            _ba_c2.metric("Positions > 25% Beta-Adj", oversized)
            _ba_c3.metric("Positions", len(beta_df))
            if _n_no_beta > 0:
                st.warning(f"⚠ {_n_no_beta} of {len(beta_df)} positions using default beta=1.0 (no real data) — true exposure estimate may be inaccurate. Run the daily runner to refresh technicals.")
            if total_beta_adj > 100:
                st.error(f"Total beta-adjusted exposure = {total_beta_adj:.1f}% — portfolio is leveraged relative to market")
            elif total_beta_adj > 70:
                st.warning(f"High beta-adjusted exposure ({total_beta_adj:.1f}%) — concentrated market risk")
            render_badge_table(beta_df, height=380)
            if _PLOTLY:
                _ba_sorted = beta_df.copy()
                _ba_sorted["_adj"] = beta_adj_vals
                _ba_sorted = _ba_sorted.sort_values("_adj", ascending=True)
                _ba_colors = ["#fee2e2" if s == "RISK" else "#dbeafe" for s in _ba_sorted["status"]]
                _fig_ba = go.Figure()
                _fig_ba.add_trace(go.Bar(
                    x=_ba_sorted["effective_weight"].str.replace("%", "").astype(float).tolist(),
                    y=_ba_sorted["ticker"].tolist(),
                    name="Nominal Weight",
                    orientation="h",
                    marker_color="#e5e7eb",
                    hovertemplate="<b>%{y}</b><br>Nominal: %{x:.2f}%<extra></extra>",
                ))
                _fig_ba.add_trace(go.Bar(
                    x=_ba_sorted["_adj"].tolist(),
                    y=_ba_sorted["ticker"].tolist(),
                    name="Beta-Adjusted",
                    orientation="h",
                    marker_color=_ba_colors,
                    hovertemplate="<b>%{y}</b><br>Beta-adj: %{x:.2f}%<extra></extra>",
                ))
                _fig_ba.update_layout(
                    barmode="overlay",
                    height=max(200, len(_ba_sorted)*34+60),
                    margin=dict(l=10, r=50, t=28, b=20),
                    title=dict(text="Nominal vs Beta-Adjusted Exposure", font=dict(size=12), x=0),
                    xaxis_title="Exposure %", yaxis_title="",
                    xaxis=dict(gridcolor="#e5e7eb", ticksuffix="%"),
                    yaxis=dict(gridcolor="#e5e7eb"),
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Inter,sans-serif", size=12),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(_fig_ba, use_container_width=True)
        st.markdown("""
**Beta-adjusted exposure formula:** `beta_adj = nominal_weight × beta`
**Why it matters:** A portfolio of 5 × 20% positions looks equally weighted — but if betas are 0.5, 1.0, 1.5, 2.0, 2.5, the true market risks are 10%, 20%, 30%, 40%, 50%.
**Rule:** No single name > 20% beta-adjusted. Total beta-adjusted exposure > 100% = implicit leverage.
""")

    with st.expander("Raw exposure report"):
        st.markdown(read_md(FILES["exposure_report"]))


def tab_trigger_board():
    st.header("Trigger Levels")

    triggers = read_csv(FILES["watch_triggers"])
    cards = read_csv(FILES["action_cards"])
    board = build_trigger_board(triggers, cards)

    if board.empty:
        st.warning("No trigger board data found.")
        return

    near = int(board["trigger_status"].astype(str).isin(["AT_TRIGGER", "NEAR_TRIGGER"]).sum()) if "trigger_status" in board.columns else 0
    paper = int((board["decision"].astype(str) == "PAPER_ONLY").sum()) if "decision" in board.columns else 0
    blocked_live = int((board["live_allowed"].astype(str) == "NO").sum()) if "live_allowed" in board.columns else 0

    closest = "N/A"
    if "nearest_trigger_distance_pct" in board.columns:
        nums = pd.to_numeric(board["nearest_trigger_distance_pct"], errors="coerce")
        if not nums.dropna().empty:
            idx = nums.idxmin()
            closest = f"{board.loc[idx, 'ticker']} ({nums.loc[idx]:.2f}%)"

    render_layer_workbench_header(
        "Triggers",
        "Trigger Levels — Price Gates",
        "Shows how far each ticker is from its breakout or breakdown trigger. Hitting a trigger means re-check, not auto-buy.",
        [
            ("Near Trigger", near,         "cyan"    if near   else "wait"),
            ("Paper Only",   paper,        "paper"),
            ("Closest",      closest,      "cyan"),
            ("Live Blocked", blocked_live, "blocked"),
        ],
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nearest Trigger Distance", closest)
    c2.metric("Near Or At Trigger", near)
    c3.metric("Paper Test Only", paper)
    c4.metric("Live Orders Blocked", blocked_live)

    st.markdown("""
    <div class="command-grid">
      <div class="command-panel command-cyan">
        <div class="command-label">Use</div>
        <div class="command-title">Watch Key Prices; Do Not Chase</div>
        <div class="command-text">This table sorts tickers by distance to upside or downside trigger levels.</div>
      </div>
      <div class="command-panel command-paper">
        <div class="command-label">How To Express It</div>
        <div class="command-title">Tiny Paper Test Only After Checks</div>
        <div class="command-text">Trigger levels only tell you to recheck; they are not trade orders.</div>
      </div>
      <div class="command-panel command-risk">
        <div class="command-label">Do Not Do This</div>
        <div class="command-title">Do Not Use Short-Term Out-Of-The-Money Options</div>
        <div class="command-text">The options danger zone and risk layer still block short-term options.</div>
      </div>
      <div class="command-panel command-blocked">
        <div class="command-label">Gate</div>
        <div class="command-title">Live Orders Are Still Not Allowed</div>
        <div class="command-text">Human checks and portfolio risk must pass first.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Tickers Closest To Triggers")
    if _PLOTLY and not board.empty and "ticker" in board.columns and "nearest_trigger_distance_pct" in board.columns:
        _tb_df = board.copy()
        _tb_df["nearest_trigger_distance_pct"] = pd.to_numeric(_tb_df["nearest_trigger_distance_pct"], errors="coerce")
        _tb_df = _tb_df.dropna(subset=["nearest_trigger_distance_pct"]).sort_values("nearest_trigger_distance_pct", ascending=True).head(20)
        if not _tb_df.empty:
            _tb_colors = [
                "#b91c1c" if str(r.get("trigger_status","")).upper() == "AT_TRIGGER"
                else "#22d3ee" if str(r.get("trigger_status","")).upper() == "NEAR_TRIGGER"
                else "#9ca3af"
                for _, r in _tb_df.iterrows()
            ]
            _fig_tb = go.Figure(go.Bar(
                x=_tb_df["nearest_trigger_distance_pct"].tolist(),
                y=_tb_df["ticker"].astype(str).tolist(),
                orientation="h",
                marker_color=_tb_colors,
                text=_tb_df["nearest_trigger_distance_pct"].map(lambda x: f"{x:.2f}%").tolist(),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Distance: %{x:.2f}%<extra></extra>",
            ))
            _fig_tb.update_layout(
                height=max(180, len(_tb_df) * 34 + 60),
                margin=dict(l=10, r=60, t=28, b=20),
                title=dict(text="Distance To Nearest Trigger (red=at, cyan=near)", font=dict(size=12), x=0),
                xaxis_title="Distance %", yaxis_title="",
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(gridcolor="#e5e7eb", ticksuffix="%"),
                yaxis=dict(gridcolor="#e5e7eb", autorange="reversed"),
                font=dict(family="Inter,sans-serif", size=12),
            )
            st.plotly_chart(_fig_tb, use_container_width=True)

    view = board.copy()
    for col in ["breakout_distance_pct", "breakdown_distance_pct", "nearest_trigger_distance_pct"]:
        if col in view.columns:
            view[col] = pd.to_numeric(view[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2f}%")
    cols = [c for c in [
        "ticker", "trigger_status", "decision", "urgency", "spot",
        "breakout_trigger", "breakout_distance_pct",
        "breakdown_trigger", "breakdown_distance_pct",
        "nearest_trigger_distance_pct", "gamma_label", "kill_zone_label",
        "allowed_action", "forbidden_action", "live_allowed", "trigger_rule"
    ] if c in view.columns]
    render_badge_table(view[cols], height=480)

    st.subheader("Raw Trigger Table")
    render_badge_table(triggers, height=300)


def tab_decision_playbook():
    st.header("If This, Then That")

    master = read_csv(FILES["master_v2"])
    triggers = build_trigger_board(read_csv(FILES["watch_triggers"]), read_csv(FILES["action_cards"]))
    ledger = read_csv(FILES["paper_ledger"])

    risk_state = "NO_DATA"
    if not master.empty and "L8_state" in master.columns:
        _rs_mode = master["L8_state"].astype(str).mode()
        risk_state = _rs_mode.iloc[0] if not _rs_mode.empty else "UNKNOWN"

    at_trigger = 0
    if not triggers.empty and "trigger_status" in triggers.columns:
        at_trigger = int(triggers["trigger_status"].astype(str).isin(["AT_TRIGGER", "NEAR_TRIGGER"]).sum())

    closed = 0
    live_blocked = 0
    if not ledger.empty and "status" in ledger.columns:
        closed = int(ledger["status"].astype(str).str.upper().isin(["CLOSED_PAPER", "CLOSED_REAL"]).sum())
    if not triggers.empty and "live_allowed" in triggers.columns:
        live_blocked = int((triggers["live_allowed"].astype(str).str.upper() == "NO").sum())

    render_layer_workbench_header(
        "Playbook",
        "If This, Then That — Decision Rules",
        "Conditional rules that link market conditions to allowed actions. No condition unlocks live orders.",
        [
            ("Risk Rule",      friendly_value(risk_state), status_kind(risk_state)),
            ("Near Trigger",   at_trigger,    "cyan"       if at_trigger  else "wait"),
            ("Closed Samples", closed,        "supportive" if closed >= 30 else "cyan"),
            ("Live Blocked",   live_blocked,  "blocked"    if live_blocked else "supportive"),
        ],
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Risk Rule", friendly_value(risk_state))
    c2.metric("Near Trigger", at_trigger)
    c3.metric("Closed Learning Samples", closed)
    c4.metric("Live Orders Blocked", live_blocked)

    st.markdown("""
    <div class="command-grid">
      <div class="command-panel command-risk">
        <div class="command-label">First Rule</div>
        <div class="command-title">Risk First, Signals Second</div>
        <div class="command-text">No single layer can override a red risk light or human checks.</div>
      </div>
      <div class="command-panel command-cyan">
        <div class="command-label">How To Use It</div>
        <div class="command-title">Read The Condition Before The Action</div>
        <div class="command-text">The table below translates model status into allowed and forbidden actions.</div>
      </div>
      <div class="command-panel command-paper">
        <div class="command-label">Paper-Test Discipline</div>
        <div class="command-title">Paper Tests Are For Learning</div>
        <div class="command-text">Small tests should create clean review samples, not hidden portfolio risk.</div>
      </div>
      <div class="command-panel command-blocked">
        <div class="command-label">No Shortcut</div>
        <div class="command-title">Trigger Levels Are Not Orders</div>
        <div class="command-text">A trigger level starts a review; it does not skip checks.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if _PLOTLY and not master.empty and "master_action" in master.columns:
        _dp_counts = master["master_action"].value_counts()
        _dp_labels = _dp_counts.index.tolist()
        _dp_vals   = _dp_counts.values.tolist()
        _dp_colors = [
            "#b91c1c" if "RISK" in str(v).upper()
            else "#a855f7" if "PAPER" in str(v).upper()
            else "#22d3ee" if "RESEARCH" in str(v).upper()
            else "#9ca3af"
            for v in _dp_labels
        ]
        _fig_dp = go.Figure(go.Bar(
            x=_dp_vals, y=_dp_labels, orientation="h",
            marker_color=_dp_colors,
            text=[str(v) for v in _dp_vals], textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x} tickers<extra></extra>",
        ))
        _fig_dp.update_layout(
            height=max(120, len(_dp_labels) * 38 + 50),
            margin=dict(l=10, r=50, t=10, b=10),
            xaxis_title="Tickers", yaxis_title="",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#e5e7eb"),
            yaxis=dict(gridcolor="#e5e7eb", autorange="reversed"),
            font=dict(family="Inter,sans-serif", size=13),
        )
        st.caption("Master action distribution — drives all downstream playbook rules.")
        st.plotly_chart(_fig_dp, use_container_width=True)

    st.subheader("Rule List")
    render_badge_table(build_decision_playbook(), height=520)

    st.subheader("Rules Used Today")
    applied = pd.DataFrame([
        {
            "current_state": f"L8_state = {risk_state}",
            "system_response": "Keep live_allowed = NO; cap expression to tiny paper only.",
            "where_to_check": "Stress Test / Portfolio Map",
        },
        {
            "current_state": f"Near or at trigger setups = {at_trigger}",
            "system_response": "Review levels and spreads; do not chase options.",
            "where_to_check": "Trigger Levels / Options Watch",
        },
        {
            "current_state": f"Closed learning samples = {closed}",
            "system_response": "L10 remains record-only until at least 30 closed paper samples.",
            "where_to_check": "Paper Log",
        },
    ])
    render_badge_table(applied, height=180)

    st.subheader("Upcoming Macro Events")
    _dp_cal = build_macro_calendar()
    if _dp_cal.empty:
        st.info("No upcoming events in the next 90 days.")
    else:
        render_badge_table(_dp_cal[["status","event","date","days_to_event","impact"]], height=260)


def tab_research_stack():
    st.header("Research Evidence")

    macro = read_csv(FILES["macro_signals"])
    sectors = read_csv(FILES["sector_scores"])
    fundamentals = read_csv(FILES["fundamentals"])
    events = read_csv(FILES["events"])
    technicals = read_csv(FILES["technicals"])

    _leaders  = 0 if sectors.empty  or "rotation_label" not in sectors.columns   else int((sectors["rotation_label"]   == "LEADER").sum())
    _tactical = 0 if technicals.empty or "technical_label" not in technicals.columns else int((technicals["technical_label"] == "TACTICAL_CANDIDATE").sum())
    _ev_risk  = 0 if events.empty    or "event_label"     not in events.columns    else int((events["event_label"]     == "EVENT_RISK").sum())
    _no_data  = count_value(macro, "data_status", "NO_DATA") + count_value(technicals, "technical_label", "NO_DATA")
    render_layer_workbench_header(
        "L2–L6",
        "Research Evidence Stack",
        "Layers 2-6 build the research case: macro mood → sector rotation → fundamentals → events → price trend.",
        [
            ("Leading Sectors", _leaders,  "supportive" if _leaders else "wait"),
            ("Trend Watch",     _tactical, "cyan"),
            ("Event Risk",      _ev_risk,  "risk" if _ev_risk else "supportive"),
            ("No-Data Rows",    _no_data,  "blocked" if _no_data else "supportive"),
        ],
    )

    leaders = 0 if sectors.empty or "rotation_label" not in sectors.columns else int((sectors["rotation_label"] == "LEADER").sum())
    laggards = 0 if sectors.empty or "rotation_label" not in sectors.columns else int((sectors["rotation_label"] == "LAGGARD").sum())
    event_risk = 0 if events.empty or "event_label" not in events.columns else int((events["event_label"] == "EVENT_RISK").sum())
    tactical = 0 if technicals.empty or "technical_label" not in technicals.columns else int((technicals["technical_label"] == "TACTICAL_CANDIDATE").sum())
    macro_no_data = count_value(macro, "data_status", "NO_DATA")
    sector_no_data = count_value(sectors, "rotation_label", "NO_DATA")
    fundamental_no_data = count_value(fundamentals, "fundamental_label", "NO_DATA")
    technical_no_data = count_value(technicals, "technical_label", "NO_DATA")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Leading Sectors", leaders)
    c2.metric("Lagging Sectors", laggards)
    c3.metric("Event Risk", event_risk)
    c4.metric("Price-Trend Watch", tactical)

    st.markdown("""
    <div class="command-grid">
      <div class="command-panel command-cyan">
        <div class="command-label">Use</div>
        <div class="command-title">Research First, Act Later</div>
        <div class="command-text">Check market mood, sector, basics, events, and price trend before options or sizing.</div>
      </div>
      <div class="command-panel command-paper">
        <div class="command-label">Current Theme</div>
        <div class="command-title">Semiconductors And Tech Are Strong, But Risk Limits Action</div>
        <div class="command-text">Layers 3 and 6 show strength, but Layer 8 still controls final size.</div>
      </div>
      <div class="command-panel command-risk">
        <div class="command-label">Main Caution</div>
        <div class="command-title">Single Stocks Have Event Risk</div>
        <div class="command-text">For tickers like AMD, MU, and GOOGL, check events and earnings before any action.</div>
      </div>
      <div class="command-panel command-blocked">
        <div class="command-label">Do Not Skip This</div>
        <div class="command-title">An ETF Is Not A Single Company</div>
        <div class="command-text">ETF decisions should focus on market mood, sector, price trend, and risk, not company-quality fields.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    evidence_summary = pd.DataFrame([
        {
            "status": "REVIEW" if macro_no_data else ("OK" if not macro.empty else "NO_DATA"),
            "research_area": "Market Mood",
            "current_read": f"{len(macro)} rows; no-data={macro_no_data}",
            "what_to_check": "Broad market, rates, volatility, and breadth before single tickers.",
            "source_file": "macro_regime_signals.csv; index_breadth_dashboard.csv; volatility_regime.csv",
        },
        {
            "status": "REVIEW" if sector_no_data else ("OK" if not sectors.empty else "NO_DATA"),
            "research_area": "Sectors And Themes",
            "current_read": f"leaders={leaders}; laggards={laggards}; no-data={sector_no_data}",
            "what_to_check": "Which sectors are leading, lagging, or only market context.",
            "source_file": "sector_rotation_scores.csv; theme_heatmap.csv",
        },
        {
            "status": "REVIEW" if fundamental_no_data else ("OK" if not fundamentals.empty else "NO_DATA"),
            "research_area": "Company Basics",
            "current_read": f"{len(fundamentals)} rows; no-data={fundamental_no_data}",
            "what_to_check": "Company quality, ETF context, valuation flags, and missing basics.",
            "source_file": "fundamental_quality_valuation.csv; valuation_risk_flags.csv",
        },
        {
            "status": "RISK" if event_risk else ("OK" if not events.empty else "NO_DATA"),
            "research_area": "News And Events",
            "current_read": f"event risk={event_risk}; rows={len(events)}",
            "what_to_check": "News, earnings, SEC filings, and insider activity before paper action.",
            "source_file": "evidence_cards.csv; news_event_risk.csv; earnings_calendar_check.csv; insider_form4_signals.csv",
        },
        {
            "status": "REVIEW" if technical_no_data else ("WATCH" if tactical else ("OK" if not technicals.empty else "NO_DATA")),
            "research_area": "Price Trend",
            "current_read": f"watch candidates={tactical}; rows={len(technicals)}; no-data={technical_no_data}",
            "what_to_check": "Trend, momentum, volatility, range, and trading activity.",
            "source_file": "technical_signal_matrix.csv; tactical_candidates.csv; breakout_reversal_watchlist.csv",
        },
    ])
    st.subheader("Research Evidence Summary")
    render_badge_table(evidence_summary, height=280)

    if _PLOTLY and not sectors.empty and "rotation_score" in sectors.columns and "ticker" in sectors.columns:
        _rs_sec = sectors.copy()
        _rs_sec["rotation_score"] = pd.to_numeric(_rs_sec["rotation_score"], errors="coerce")
        _rs_sec = _rs_sec.dropna(subset=["rotation_score"]).sort_values("rotation_score", ascending=True)
        if not _rs_sec.empty:
            _rs_colors = [
                "#16a34a" if v >= 10
                else "#22d3ee" if v >= 0
                else "#f87171" if v >= -10
                else "#b91c1c"
                for v in _rs_sec["rotation_score"]
            ]
            _fig_rs = go.Figure(go.Bar(
                x=_rs_sec["rotation_score"].tolist(),
                y=_rs_sec["ticker"].astype(str).tolist(),
                orientation="h",
                marker_color=_rs_colors,
                text=_rs_sec["rotation_score"].map(lambda x: f"{x:+.0f}").tolist(),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Rotation score: %{x:+.0f}<extra></extra>",
            ))
            _fig_rs.update_layout(
                height=max(200, len(_rs_sec) * 36 + 60),
                margin=dict(l=10, r=50, t=28, b=20),
                title=dict(text="Layer 3 — Sector Rotation Scores", font=dict(size=13, family="Inter,sans-serif"), x=0),
                xaxis_title="Rotation Score", yaxis_title="",
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(gridcolor="#e5e7eb", zeroline=True, zerolinecolor="#111827", zerolinewidth=1),
                yaxis=dict(gridcolor="#e5e7eb"),
                font=dict(family="Inter,sans-serif", size=12),
            )
            st.caption("Green = strong leaders, cyan = mild positive, red = laggards.")
            st.plotly_chart(_fig_rs, use_container_width=True)

    st.subheader("Layer 2: Market Mood")
    if not macro.empty:
        macro_view = format_percent_columns(macro, ["ret_5d", "ret_20d", "ret_63d", "realized_vol_20d"])
        cols = [c for c in [
            "ticker", "description", "last_close", "ret_5d", "ret_20d", "ret_63d",
            "trend_state", "realized_vol_20d", "data_status"
        ] if c in macro_view.columns]
        render_badge_table(macro_view[cols], height=270)
    else:
        st.info("No macro signals.")

    st.subheader("Layer 3: Sectors And Themes")
    if not sectors.empty:
        sector_view = format_percent_columns(sectors, ["ret_5d", "ret_20d", "ret_63d", "relative_20d_vs_spy", "relative_63d_vs_spy"])
        cols = [c for c in [
            "ticker", "theme", "ret_20d", "relative_20d_vs_spy", "relative_63d_vs_spy",
            "rotation_score", "rotation_label"
        ] if c in sector_view.columns]
        render_badge_table(sector_view[cols], height=320)
    else:
        st.info("No sector scores.")

    st.subheader("Layer 4: Basics And Quality")
    if not fundamentals.empty:
        fund_view = format_percent_columns(fundamentals, ["revenue_growth", "gross_margin", "operating_margin", "profit_margin"])
        cols = [c for c in [
            "ticker", "asset_type", "data_status", "company_name", "sector",
            "revenue_growth", "gross_margin", "forward_pe", "peg_ratio",
            "quality_score", "fundamental_label"
        ] if c in fund_view.columns]
        render_badge_table(fund_view[cols], height=320)
    else:
        st.info("No fundamental file.")

    st.subheader("Layer 5: Events, Filings, And Insider Activity")
    if not events.empty:
        render_badge_table(events, height=260)
    else:
        st.info("No event evidence.")

    st.subheader("Layer 6: Price Trend And Trading Activity")
    if not technicals.empty:
        tech_view = format_percent_columns(technicals, [
            "ret_5d", "ret_20d", "ret_63d", "atr14_pct",
            "distance_to_20d_high", "distance_to_20d_low"
        ])
        cols = [c for c in [
            "ticker", "close", "ret_20d", "ret_63d", "rsi14", "atr14_pct",
            "volume_z60", "technical_score", "technical_label", "reasons"
        ] if c in tech_view.columns]
        render_badge_table(tech_view[cols], height=360)
    else:
        st.info("No technical matrix.")


def tab_research_lab():
    st.header("Research Path")
    st.caption("Confirm research evidence quality before options, sizing, or action checks.")

    routes = build_research_lab_routes()
    master = read_csv(FILES["master_v2"])
    focus = build_focus_list()
    inventory = read_csv(FILES["v8_inventory"])
    options = read_csv(FILES["options_decision"])
    pretrade = read_csv(FILES["pre_trade"])

    review_count = count_value(routes, "status", "REVIEW")
    risk_count = count_value(routes, "status", "RISK")
    research_only = count_value(options, "final_options_decision", "RESEARCH_ONLY")
    live_blocked = count_value(pretrade, "live_allowed", "NO")

    render_layer_workbench_header(
        "Research",
        "Evidence Routing & Research Controls",
        "Use this page to decide which research layer needs attention before any paper expression is considered.",
        [
            ("Review Stations", review_count, "cyan"),
            ("Risk Stations", risk_count, "risk" if risk_count else "supportive"),
            ("Options Research", research_only, "cyan"),
            ("Live Blocked", live_blocked, "blocked"),
        ],
    )

    research_command = pd.DataFrame([
        {
            "status": "REVIEW" if review_count else "OK",
            "desk_question": "Which research layer needs attention first?",
            "read_this_first": "Research route table",
            "why_it_matters": "A ticker idea should not jump straight to options, triggers, or action checks.",
            "source_file": "macro_regime_signals.csv; sector_rotation_scores.csv; fundamental_quality_valuation.csv; evidence_cards.csv; technical_signal_matrix.csv",
        },
        {
            "status": "RISK" if risk_count else "OK",
            "desk_question": "Is any research station flashing risk?",
            "read_this_first": "Risk-marked research rows",
            "why_it_matters": "Event risk, weak data, or red portfolio risk can block an otherwise interesting setup.",
            "source_file": "evidence_cards.csv; pre_trade_checklist.csv; master_10_layer_decision_matrix_v2.csv",
        },
        {
            "status": "REVIEW" if not focus.empty else "NO_DATA",
            "desk_question": "Which tickers deserve attention now?",
            "read_this_first": "Tickers Worth Checking First",
            "why_it_matters": "Attention should follow ranked evidence and blockers, not noise.",
            "source_file": "master_10_layer_decision_matrix_v2.csv; watch_triggers.csv; pre_trade_checklist.csv",
        },
        {
            "status": "RESEARCH_ONLY" if research_only else "OK",
            "desk_question": "Are options only context today?",
            "read_this_first": "Options Watch",
            "why_it_matters": "Options pressure is Layer 7 only; Layer 8 risk and Layer 9 checks still decide what is allowed.",
            "source_file": "options_decision_matrix.csv; gamma_squeeze_candidates.csv; option_kill_zone_risk.csv",
        },
        {
            "status": "NO" if live_blocked else "OK",
            "desk_question": "Can anything become a live order?",
            "read_this_first": "Before-Action Check",
            "why_it_matters": "No broker connection and no live order path are enabled.",
            "source_file": "pre_trade_checklist.csv; v8_l9_execution_gate.csv; action_cards.csv",
        },
    ])
    st.subheader("Research Command Board")
    render_badge_table(research_command, height=300)

    with st.expander("Open Research Source Map"):
        st.markdown("**Market Mood:** `macro_regime_signals.csv`, `index_breadth_dashboard.csv`, `volatility_regime.csv`")
        st.markdown("**Sectors:** `sector_rotation_scores.csv`, `theme_heatmap.csv`")
        st.markdown("**Company Basics:** `fundamental_quality_valuation.csv`, `valuation_risk_flags.csv`")
        st.markdown("**Events:** `evidence_cards.csv`, `news_event_risk.csv`, `earnings_calendar_check.csv`, `insider_form4_signals.csv`")
        st.markdown("**Price Trend:** `technical_signal_matrix.csv`, `tactical_candidates.csv`, `breakout_reversal_watchlist.csv`")
        st.markdown("**Options Context:** `options_decision_matrix.csv`, `gamma_squeeze_candidates.csv`, `option_kill_zone_risk.csv`")
        st.markdown("**Action Check:** `pre_trade_checklist.csv`, `v8_l9_execution_gate.csv`, `action_cards.csv`")

    st.subheader("How To Use The Research Page")
    render_badge_table(routes, height=360)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Tickers Worth Checking First")
        if focus.empty:
            st.info("No focus list yet.")
        else:
            cols = [c for c in [
                "ticker", "focus_bucket", "focus_score", "master_action",
                "trigger_status", "reason"
            ] if c in focus.columns]
            view = focus[cols].head(8).copy()
            if "focus_score" in view.columns:
                view["focus_score"] = pd.to_numeric(view["focus_score"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
            render_badge_table(view, height=330)
    with c2:
        st.subheader("Old Module Status")
        if inventory.empty:
            st.info("No v8 inventory yet.")
        else:
            status_counts = inventory.groupby("integration_status", as_index=False).size().rename(columns={"size": "count"})
            render_badge_table(status_counts, height=180)
            module_cols = [c for c in ["module", "target_layer", "integration_status", "present_in_source"] if c in inventory.columns]
            render_badge_table(inventory[module_cols].head(8), height=250)

    if _PLOTLY and not master.empty and "master_action" in master.columns:
        _rl_counts = master["master_action"].value_counts()
        _rl_labels = _rl_counts.index.tolist()
        _rl_vals   = _rl_counts.values.tolist()
        _rl_colors = [
            "#b91c1c" if "RISK" in str(v).upper()
            else "#a855f7" if "PAPER" in str(v).upper()
            else "#22d3ee" if "RESEARCH" in str(v).upper()
            else "#9ca3af"
            for v in _rl_labels
        ]
        _fig_rl = go.Figure(go.Bar(
            x=_rl_vals, y=_rl_labels, orientation="h",
            marker_color=_rl_colors,
            text=[str(v) for v in _rl_vals], textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x} tickers<extra></extra>",
        ))
        _fig_rl.update_layout(
            height=max(100, len(_rl_labels) * 38 + 40),
            margin=dict(l=10, r=50, t=10, b=10),
            xaxis_title="Tickers", yaxis_title="",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#e5e7eb"),
            yaxis=dict(gridcolor="#e5e7eb", autorange="reversed"),
            font=dict(family="Inter,sans-serif", size=13),
        )
        st.caption("Master action distribution — drives research routing decisions.")
        st.plotly_chart(_fig_rl, use_container_width=True)

    st.subheader("Research-Only Tickers")
    if master.empty:
        st.info("No master matrix yet.")
    else:
        research_view = master[master["master_action"].astype(str).str.contains("RESEARCH|RISK_REDUCTION", regex=True, na=False)].copy() if "master_action" in master.columns else master.copy()
        cols = [c for c in [
            "ticker", "master_action", "stack_score_avg", "L2_state", "L3_state",
            "L4_state", "L5_state", "L6_state", "L7_state", "L8_state", "L9_state",
            "master_reason"
        ] if c in research_view.columns]
        render_badge_table(research_view[cols], height=420)


def tab_v8_research_bridge():
    st.header("Old Code Link")

    inventory = read_csv(FILES["v8_inventory"])
    bsm = read_csv(FILES["v8_bsm"])
    synthetic = read_csv(FILES["v8_synthetic_options"])

    integrated = 0
    blocked = 0
    available = 0
    medium_or_high = 0
    if not inventory.empty and "integration_status" in inventory.columns:
        integrated = int((inventory["integration_status"] == "INTEGRATED_STEP57").sum())
        blocked = int((inventory["integration_status"] == "BLOCKED_NO_LIVE").sum())
        available = int((inventory["integration_status"] == "AVAILABLE_RESEARCH").sum())
    if not synthetic.empty and "squeeze_risk" in synthetic.columns:
        medium_or_high = int(synthetic["squeeze_risk"].astype(str).str.upper().isin(["MEDIUM", "HIGH"]).sum())

    render_layer_workbench_header(
        "V8",
        "Old Code Research Link",
        "Canyon v8 math is connected for diagnostics only. BSM pricing, simulated squeeze, and module inventory are research context — not signals.",
        [
            ("Connected",    integrated,     "supportive" if integrated else "wait"),
            ("Research",     available,      "cyan"),
            ("Live Blocked", blocked,        "blocked"),
            ("Squeeze Watch",medium_or_high, "risk" if medium_or_high else "supportive"),
        ],
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Connected Modules", integrated)
    c2.metric("Research Modules", available)
    c3.metric("Live-Order Modules Blocked", blocked)
    c4.metric("Simulated Squeeze Watch", medium_or_high)

    st.markdown("""
    <div class="command-grid">
      <div class="command-panel command-cyan">
        <div class="command-label">Use</div>
        <div class="command-title">Use Old Math, Keep Safety Gates</div>
        <div class="command-text">Old options pricing, simulated options pressure, and module lists are diagnostic only.</div>
      </div>
      <div class="command-panel command-risk">
        <div class="command-label">Safety Rules</div>
        <div class="command-title">Broker Code Is Disabled</div>
        <div class="command-text">Real trading, automated orders, and execution algorithms are archived only and will not run.</div>
      </div>
      <div class="command-panel command-paper">
        <div class="command-label">Options Reference</div>
        <div class="command-title">Simulated Research Only</div>
        <div class="command-text">It cannot override real Layer 7 options, Layer 8 risk, or Layer 9 action checks.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Module List", "Options Price Reference", "Simulated Options Pressure", "Report"])
    with tab1:
        st.caption("Lists all v8 research modules and their integration status. BLOCKED_NO_LIVE = disabled for live orders; INTEGRATED_STEP57 = connected for research only.")
        if _PLOTLY and not inventory.empty and "integration_status" in inventory.columns:
            _inv_counts = inventory["integration_status"].value_counts()
            _inv_labels = _inv_counts.index.tolist()
            _inv_vals   = _inv_counts.values.tolist()
            _inv_colors = [
                "#b91c1c" if "BLOCKED" in str(v).upper()
                else "#22d3ee" if "INTEGRATED" in str(v).upper()
                else "#9ca3af"
                for v in _inv_labels
            ]
            _fig_inv = go.Figure(go.Bar(
                x=_inv_vals, y=_inv_labels, orientation="h",
                marker_color=_inv_colors,
                text=[str(v) for v in _inv_vals], textposition="outside",
                hovertemplate="<b>%{y}</b><br>%{x} modules<extra></extra>",
            ))
            _fig_inv.update_layout(
                height=max(100, len(_inv_labels) * 42 + 40),
                margin=dict(l=10, r=50, t=10, b=10),
                xaxis_title="Modules", yaxis_title="",
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(gridcolor="#e5e7eb"),
                yaxis=dict(gridcolor="#e5e7eb", autorange="reversed"),
                font=dict(family="Inter,sans-serif", size=13),
            )
            st.plotly_chart(_fig_inv, use_container_width=True)
        render_badge_table(inventory, height=420)
    with tab2:
        st.caption("BSM (Black-Scholes-Merton) theoretical option prices and Greeks per ticker. These are model estimates — not live market quotes. Use for research context only.")
        render_badge_table(bsm, height=460)
    with tab3:
        st.caption("Simulated squeeze and gamma pressure scores. combined_signal and squeeze_score are modeled estimates from historical correlations, not real-time dealer positioning data.")
        render_badge_table(synthetic, height=460)
    with tab4:
        st.markdown(read_md(FILES["v8_report"]))


def tab_options_lab():
    st.header("Options Watch")
    st.warning(
        "⚠  SIMULATED DATA — Options heat (gamma squeeze) and danger zone scores are model estimates "
        "using historical correlations and yfinance public data. They are NOT real-time dealer positioning "
        "or live options chain data. Use as research context only. Never use as the sole basis for an action decision."
    )

    gamma = read_csv(FILES["gamma_candidates"])
    kill = read_csv(FILES["kill_zone"])
    decision = read_csv(FILES["options_decision"])

    paper_only = 0
    skip = 0
    medium_gamma = 0
    medium_kill = 0
    if not decision.empty and "final_options_decision" in decision.columns:
        paper_only = int((decision["final_options_decision"] == "PAPER_ONLY").sum())
        skip = int((decision["final_options_decision"] == "SKIP").sum())
    if not gamma.empty and "gamma_squeeze_label" in gamma.columns:
        medium_gamma = int(gamma["gamma_squeeze_label"].astype(str).str.contains("MEDIUM", na=False).sum())
    if not kill.empty and "option_kill_zone_label" in kill.columns:
        medium_kill = int(kill["option_kill_zone_label"].astype(str).str.contains("MEDIUM", na=False).sum())

    render_layer_workbench_header(
        "L7",
        "Options Watch — Heat And Danger Zone",
        "Options context for timing and risk. L8 and L9 always take priority. Short-term options remain blocked.",
        [
            ("Options Heat",  medium_gamma, "risk" if medium_gamma else "supportive"),
            ("Danger Zone",   medium_kill,  "risk" if medium_kill  else "supportive"),
            ("Paper Only",    paper_only,   "paper"),
            ("Skip",          skip,         "blocked" if skip else "supportive"),
        ],
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Options Heat Watch", medium_gamma)
    c2.metric("Danger Zone", medium_kill)
    c3.metric("Paper Test Only", paper_only)
    c4.metric("Skip", skip)

    st.markdown("""
    <div class="command-grid">
      <div class="command-panel command-cyan">
        <div class="command-label">Note</div>
        <div class="command-title">This Is An Estimate, Not Real Dealer Positioning</div>
        <div class="command-text">This estimates where options attention is concentrated; it does not confirm dealer holdings.</div>
      </div>
      <div class="command-panel command-paper">
        <div class="command-label">How To Express It</div>
        <div class="command-title">Use Stock/ETF Paper Tests Only</div>
        <div class="command-text">Current rules block short-term options. If expressing an idea, use only tiny stock/ETF paper tests.</div>
      </div>
      <div class="command-panel command-risk">
        <div class="command-label">Risk</div>
        <div class="command-title">You Can Be Right And Still Lose</div>
        <div class="command-text">Pinning, falling volatility, time decay, and wide spreads can hurt short-term options.</div>
      </div>
      <div class="command-panel command-blocked">
        <div class="command-label">What Comes First</div>
        <div class="command-title">Layers 8 And 9 Still Decide</div>
        <div class="command-text">Options cannot override a red risk light or human action check.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if _PLOTLY and not decision.empty and "final_options_decision" in decision.columns:
        _opt_counts = decision["final_options_decision"].value_counts()
        _opt_labels = _opt_counts.index.tolist()
        _opt_vals   = _opt_counts.values.tolist()
        _opt_colors = [
            "#a855f7" if "PAPER" in str(v).upper()
            else "#b91c1c" if "KILL" in str(v).upper() or "DANGER" in str(v).upper()
            else "#22d3ee" if "RESEARCH" in str(v).upper()
            else "#9ca3af"
            for v in _opt_labels
        ]
        _fig_opt = go.Figure(go.Bar(
            x=_opt_vals, y=_opt_labels, orientation="h",
            marker_color=_opt_colors,
            text=[str(v) for v in _opt_vals], textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x} tickers<extra></extra>",
        ))
        _fig_opt.update_layout(
            height=max(100, len(_opt_labels) * 40 + 40),
            margin=dict(l=10, r=50, t=10, b=10),
            xaxis_title="Tickers", yaxis_title="",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#e5e7eb"),
            yaxis=dict(gridcolor="#e5e7eb", autorange="reversed"),
            font=dict(family="Inter,sans-serif", size=13),
        )
        st.caption("Options decision distribution — paper only or research only; no live orders.")
        st.plotly_chart(_fig_opt, use_container_width=True)

    st.subheader("Options Layer Decision Table")
    if not decision.empty:
        decision_view = format_percent_columns(decision, [
            "call_wall_distance", "put_wall_distance", "max_pain_distance",
            "near_expiry_oi_ratio", "median_spread_pct"
        ])
        cols = [c for c in [
            "ticker", "spot", "gamma_squeeze_label", "gamma_squeeze_score",
            "call_wall_strike", "call_wall_distance", "put_wall_strike",
            "put_wall_distance", "option_kill_zone_label", "option_kill_zone_score",
            "pretrade_status", "portfolio_risk_light", "final_options_decision",
            "rule", "explanation"
        ] if c in decision_view.columns]
        render_badge_table(decision_view[cols], height=420)
    else:
        st.info("No options decision matrix.")

    st.subheader("Options Heat Watch")
    if not gamma.empty:
        gamma_view = format_percent_columns(gamma, ["call_wall_distance", "put_wall_distance"])
        cols = [c for c in [
            "ticker", "spot", "contracts", "data_confidence", "net_gex_1pct_proxy",
            "call_wall_strike", "call_wall_distance", "put_wall_strike",
            "put_wall_distance", "gamma_squeeze_score", "gamma_squeeze_label", "reasons"
        ] if c in gamma_view.columns]
        render_badge_table(gamma_view[cols], height=340)
    else:
        st.info("No gamma candidates.")

    st.subheader("Short-Term Options Danger Zone")
    if not kill.empty:
        kill_view = format_percent_columns(kill, [
            "max_pain_distance", "call_oi_wall_distance", "put_oi_wall_distance",
            "near_expiry_oi_ratio", "atm_oi_ratio", "median_spread_pct"
        ])
        cols = [c for c in [
            "ticker", "spot", "option_kill_zone_score", "option_kill_zone_label",
            "nearest_expiration", "max_pain_proxy", "max_pain_distance",
            "call_oi_wall", "call_oi_wall_distance", "put_oi_wall",
            "put_oi_wall_distance", "median_spread_pct", "interpretation", "action_rule"
        ] if c in kill_view.columns]
        render_badge_table(kill_view[cols], height=340)
    else:
        st.info("No kill zone file.")

    with st.expander("Raw options reports"):
        st.markdown(read_md(FILES["options_report"]))
        st.markdown(read_md(FILES["kill_zone_report"]))


def tab_pre_trade_gate():
    st.header("Before-Action Check")

    pretrade = read_csv(FILES["pre_trade"])
    v8_gate = read_csv(FILES["v8_l9_gate"])
    order_ticket = read_csv(FILES["pre_trade_order"])

    if pretrade.empty and v8_gate.empty:
        st.warning("No pre-trade checklist found.")
        return

    counts = pretrade["final_status"].value_counts() if "final_status" in pretrade.columns else pd.Series(dtype=int)
    v8_counts = v8_gate["final_status"].value_counts() if "final_status" in v8_gate.columns else pd.Series(dtype=int)
    pending = int(counts.get("PENDING_MANUAL_CHECKS", 0))
    blocked = int(counts.get("BLOCKED", 0))
    closed = int(counts.get("ALREADY_CLOSED_DO_NOT_REPEAT", 0))
    no_new_risk = int(v8_counts.get("RESEARCH_ONLY_NO_NEW_RISK", 0))
    order_rows = len(order_ticket)

    render_layer_workbench_header(
        "L9",
        "Before-Action Check — Human Gate",
        "All manual checks must be YES before a paper test. Live orders are blocked regardless.",
        [
            ("Needs Check", pending,      "cyan"    if pending  else "supportive"),
            ("Blocked",     blocked,      "blocked" if blocked  else "supportive"),
            ("Add No Risk", no_new_risk,  "risk"    if no_new_risk else "watch"),
            ("Orders",      order_rows,   "wait"    if order_rows else "supportive"),
        ],
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Waiting For Human Checks", pending)
    c2.metric("Blocked", blocked)
    c3.metric("Already Closed", closed)
    c4.metric("Add No New Risk", no_new_risk)
    c5.metric("Order Safety Ticket", order_rows)

    st.markdown("""
    <div class="command-grid">
      <div class="command-panel command-risk">
        <div class="command-label">Rule</div>
        <div class="command-title">Red Risk Light Means No Live Orders</div>
        <div class="command-text">The checklist may allow a tiny paper watch, but live orders remain NO.</div>
      </div>
      <div class="command-panel command-cyan">
        <div class="command-label">Human Checks</div>
        <div class="command-title">News, Earnings, Liquidity, And Bid-Ask Spread</div>
        <div class="command-text">Any ticker still marked NO needs human confirmation before even a paper test.</div>
      </div>
      <div class="command-panel command-blocked">
        <div class="command-label">Do Not Touch</div>
        <div class="command-title">Blocked / Already Closed</div>
        <div class="command-text">Do not create duplicate samples or force blocked rows into action.</div>
      </div>
      <div class="command-panel command-paper">
        <div class="command-label">Paper Path</div>
        <div class="command-title">Only After Checks</div>
        <div class="command-text">If a tiny paper test has a reason, use the paper helper instead of hand-editing CSV.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if _PLOTLY and not pretrade.empty and "final_status" in pretrade.columns:
        _pt_counts = pretrade["final_status"].value_counts()
        _pt_labels = _pt_counts.index.tolist()
        _pt_vals   = _pt_counts.values.tolist()
        _pt_colors = [
            "#f87171" if "BLOCKED" in str(v).upper()
            else "#facc15" if "PENDING" in str(v).upper()
            else "#22d3ee" if "PAPER" in str(v).upper()
            else "#9ca3af"
            for v in _pt_labels
        ]
        _fig_pt = go.Figure(go.Bar(
            x=_pt_vals, y=_pt_labels, orientation="h",
            marker_color=_pt_colors,
            text=[str(v) for v in _pt_vals], textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x} ticker(s)<extra></extra>",
        ))
        _fig_pt.update_layout(
            height=max(100, len(_pt_labels) * 40 + 40),
            margin=dict(l=10, r=50, t=10, b=10),
            xaxis_title="Tickers", yaxis_title="",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#e5e7eb"),
            yaxis=dict(gridcolor="#e5e7eb", autorange="reversed"),
            font=dict(family="Inter,sans-serif", size=13),
        )
        st.caption("Gate status distribution — yellow=needs check, red=blocked.")
        st.plotly_chart(_fig_pt, use_container_width=True)

    st.subheader("Rows Still Needing Work")
    pending_rows = pretrade[pretrade["final_status"].astype(str).eq("PENDING_MANUAL_CHECKS")] if "final_status" in pretrade.columns else pd.DataFrame()
    if not pending_rows.empty:
        cols = [c for c in [
            "ticker", "sleeve", "risk_bucket", "suggested_weight", "suggested_action",
            "risk_light", "manual_news_check", "earnings_date_check", "liquidity_check",
            "spread_check", "duplicate_exposure_check", "stress_check",
            "paper_allowed", "live_allowed", "reasons"
        ] if c in pending_rows.columns]
        pending_view = format_percent_columns(pending_rows[cols], ["suggested_weight"])
        render_badge_table(pending_view, height=380)
    else:
        st.info("No pending manual rows.")

    st.subheader("Full Before-Action Check Table")
    full_cols = [c for c in [
        "ticker", "sleeve", "ledger_status", "final_status", "risk_light",
        "suggested_weight", "manual_news_check", "earnings_date_check",
        "liquidity_check", "spread_check", "duplicate_exposure_check",
        "stress_check", "paper_allowed", "live_allowed"
    ] if c in pretrade.columns]
    full_view = format_percent_columns(pretrade[full_cols], ["suggested_weight"])
    render_badge_table(full_view, height=460)

    st.subheader("Old Action Check")
    if not v8_gate.empty:
        v8_cols = [c for c in [
            "ticker", "sleeve", "decision", "final_status", "risk_light",
            "suggested_action", "paper_allowed", "live_allowed",
            "stress_check", "duplicate_exposure_check", "reasons", "sizing_reason"
        ] if c in v8_gate.columns]
        render_badge_table(v8_gate[v8_cols], height=360)
    else:
        st.info("No V8 L9 gate rows.")

    st.subheader("Order Safety Check")
    if order_ticket.empty:
        st.success("No order tickets generated. This is correct while live_allowed is NO and manual checks are incomplete.")
    else:
        st.warning("Order tickets exist; review carefully before any action.")
        render_badge_table(order_ticket, height=220)

    with st.expander("Raw pre-trade report"):
        st.markdown(read_md(ROOT / "pre_trade_checklist.md"))
        st.markdown(read_md(FILES["v8_l9_report"]))


def tab_command_center():
    st.header("Today Overview")

    master = read_csv(FILES["master_v2"])
    cards = read_csv(FILES["action_cards"])
    triggers = read_csv(FILES["watch_triggers"])
    pretrade = read_csv(FILES["pre_trade"])

    if master.empty:
        st.warning("Run Step 54 first.")
        return

    risk_state = "NO_DATA"
    if "L8_state" in master.columns and not master["L8_state"].empty:
        _rs_mode = master["L8_state"].astype(str).mode()
        risk_state = _rs_mode.iloc[0] if not _rs_mode.empty else "UNKNOWN"
    counts = master["master_action"].value_counts() if "master_action" in master.columns else pd.Series(dtype=int)

    render_layer_workbench_header(
        "Today",
        "Today Overview — Decision Summary",
        "Quick view of what the 10-layer system decided today and what needs human attention.",
        [
            ("Risk Light",    friendly_value(risk_state),              status_kind(risk_state)),
            ("Tiny Paper",    int(counts.get("TINY_PAPER_ONLY",   0)), "paper"),
            ("Reduce Risk",   int(counts.get("RISK_REDUCTION_FIRST",0)),"risk"),
            ("Research Only", int(counts.get("RESEARCH_ONLY",     0)), "cyan"),
        ],
    )

    _is_red = str(risk_state).upper() == "RED"
    _risk_color = "#f87171" if _is_red else "#34d399"

    def _ov_stat(label: str, val: str, accent: str = "#111827") -> str:
        return (
            f'<div style="border-left:3px solid {accent};padding:6px 0 6px 14px;">'
            f'<div style="font-size:10px;font-weight:700;letter-spacing:.10em;text-transform:uppercase;'
            f'color:#4b5563;margin-bottom:3px;">{escape(label)}</div>'
            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:20px;font-weight:600;'
            f'color:#111827;">{escape(str(val))}</div>'
            f'</div>'
        )

    st.markdown(
        f'<div style="border-top:2px solid #111827;padding:14px 0 18px 0;margin:0 0 16px 0;'
        f'display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;">'
        + _ov_stat("Risk Light", friendly_value(risk_state), _risk_color)
        + _ov_stat("Tiny Paper", str(int(counts.get("TINY_PAPER_ONLY", 0))), "#c084fc")
        + _ov_stat("Reduce Risk", str(int(counts.get("RISK_REDUCTION_FIRST", 0))), "#f87171")
        + _ov_stat("Research Only", str(int(counts.get("RESEARCH_ONLY", 0))), "#60a5fa")
        + _ov_stat("Skip", str(int(counts.get("SKIP", 0))), "#9ca3af")
        + '</div>',
        unsafe_allow_html=True,
    )

    tiny = as_ticker_list(master, "TINY_PAPER_ONLY") or "None"
    reduce_first = as_ticker_list(master, "RISK_REDUCTION_FIRST") or "None"
    research = as_ticker_list(master, "RESEARCH_ONLY") or "None"
    skip = as_ticker_list(master, "SKIP") or "None"

    st.markdown(f"""
    <div class="command-grid">
      <div class="command-panel command-risk">
        <div class="command-label">Hard Rule</div>
        <div class="command-title">No Live Orders, No Short-Term Options</div>
        <div class="command-text">The risk layer is {escape(friendly_value(risk_state))}. Options pressure cannot override portfolio risk.</div>
      </div>
      <div class="command-panel command-paper">
        <div class="command-label">Allowed</div>
        <div class="command-title">Tiny Paper Test Only</div>
        <div class="command-text">{escape(tiny)}</div>
      </div>
      <div class="command-panel command-risk">
        <div class="command-label">Reduce Risk First</div>
        <div class="command-title">Handle Risk Before New Ideas</div>
        <div class="command-text">{escape(reduce_first)}</div>
      </div>
      <div class="command-panel command-cyan">
        <div class="command-label">Research Only</div>
        <div class="command-title">Look First, Do Nothing Yet</div>
        <div class="command-text">{escape(research)}</div>
      </div>
      <div class="command-panel command-blocked">
        <div class="command-label">Skip</div>
        <div class="command-title">Do Not Spend Attention Here</div>
        <div class="command-text">{escape(skip)}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if _PLOTLY and not master.empty and "master_action" in master.columns:
        _cc_counts = master["master_action"].value_counts()
        _cc_labels = _cc_counts.index.tolist()
        _cc_vals   = _cc_counts.values.tolist()
        _cc_colors = [
            "#b91c1c" if "RISK" in str(v).upper()
            else "#a855f7" if "PAPER" in str(v).upper()
            else "#22d3ee" if "RESEARCH" in str(v).upper()
            else "#9ca3af"
            for v in _cc_labels
        ]
        _fig_cc = go.Figure(go.Bar(
            x=_cc_vals, y=_cc_labels, orientation="h",
            marker_color=_cc_colors,
            text=[str(v) for v in _cc_vals], textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x} tickers<extra></extra>",
        ))
        _fig_cc.update_layout(
            height=max(120, len(_cc_labels) * 38 + 50),
            margin=dict(l=10, r=50, t=10, b=10),
            xaxis_title="Number of tickers", yaxis_title="",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#e5e7eb"),
            yaxis=dict(gridcolor="#e5e7eb", autorange="reversed"),
            font=dict(family="Inter,sans-serif", size=13),
        )
        st.plotly_chart(_fig_cc, use_container_width=True)

    merged = merge_master_action(master, cards)
    if not merged.empty:
        st.subheader("Action Map")
        render_badge_table(merged, height=430)

    if not triggers.empty:
        st.subheader("Trigger Watch")
        trigger_cols = [c for c in [
            "ticker", "decision", "urgency", "spot", "action",
            "call_wall_breakout_trigger", "put_wall_breakdown_trigger",
            "gamma_label", "kill_zone_label", "live_allowed"
        ] if c in triggers.columns]
        render_badge_table(triggers[trigger_cols], height=320)

    if not pretrade.empty:
        st.subheader("Unfinished Human Checks")
        pre_cols = [c for c in [
            "ticker", "risk_light", "final_status", "paper_allowed", "live_allowed",
            "manual_news_check", "earnings_date_check", "liquidity_check",
            "spread_check", "duplicate_exposure_check", "stress_check"
        ] if c in pretrade.columns]
        render_badge_table(pretrade[pre_cols], height=360)


def tab_daily_brief():  # noqa: C901
    st.header("Today Brief")

    # ── load data ─────────────────────────────────────────────────────────────
    master   = read_csv(FILES["master_v2"])
    sectors  = read_csv(FILES["sector_scores"])
    stress   = read_csv(FILES["scenario_stress"])
    pretrade = read_csv(FILES["pre_trade"])
    ledger   = read_csv(FILES["paper_ledger"])
    macro    = read_csv(FILES["macro_signals"])
    vol      = read_csv(FILES["volatility_regime"])
    breadth  = read_csv(FILES["index_breadth"])
    health   = read_csv(FILES["data_source_health"])
    run_st   = build_run_status()
    market   = read_csv(FILES["market_snapshot"])
    gaps     = build_gap_queue(master, market)

    risk_state = "NO_DATA"
    if not master.empty and "L8_state" in master.columns:
        _rs_mode = master["L8_state"].astype(str).mode()
        risk_state = _rs_mode.iloc[0] if not _rs_mode.empty else "UNKNOWN"

    _now = datetime.now().strftime("%Y-%m-%d  %H:%M")
    is_red = str(risk_state).upper() == "RED"

    # ── top alert bar ─────────────────────────────────────────────────────────
    stale_n   = count_value(run_st,   "status",       "STALE")   + count_value(run_st,   "status", "MISSING")
    data_risk = count_value(health,   "status",       "RISK")
    pending   = count_value(pretrade, "final_status", "PENDING_MANUAL_CHECKS")
    high_gaps = 0 if gaps.empty or "priority" not in gaps.columns else int((gaps["priority"].astype(str).str.upper() == "HIGH").sum())
    alerts    = []
    if is_red:     alerts.append("⚠  RISK LIGHT RED")
    if stale_n:    alerts.append(f"{stale_n} STALE FILES")
    if data_risk:  alerts.append(f"{data_risk} DATA RISK")
    if pending:    alerts.append(f"{pending} PENDING CHECKS")
    if high_gaps:  alerts.append(f"{high_gaps} HIGH-PRI GAPS")
    if alerts:
        st.markdown(
            '<p style="font-size:12px;font-weight:700;letter-spacing:.08em;'
            'color:#b91c1c;margin:0 0 10px 0;">' + "  ·  ".join(alerts) + "</p>",
            unsafe_allow_html=True,
        )

    # ── key metrics row ───────────────────────────────────────────────────────
    _closed = 0
    _avg_pnl_str = "N/A"
    if not ledger.empty and "status" in ledger.columns:
        _closed = int(ledger["status"].astype(str).str.upper().str.startswith("CLOSED").sum())
        if "pnl_pct" in ledger.columns:
            _pnl = pd.to_numeric(ledger["pnl_pct"], errors="coerce").dropna()
            if not _pnl.empty:
                _avg_pnl_str = f"{_pnl.mean()*100:+.2f}%"

    _worst_loss = "N/A"
    if not stress.empty and "estimated_loss" in stress.columns:
        _loss = pd.to_numeric(stress["estimated_loss"], errors="coerce")
        if not _loss.dropna().empty:
            _idx = _loss.idxmin()
            _worst_loss = f"{_loss.loc[_idx]*100:.2f}%"

    _brief_closed = count_contains(ledger, "status", "CLOSED")
    _brief_leaders = int(sectors["rotation_label"].eq("LEADER").sum()) if not sectors.empty and "rotation_label" in sectors.columns else 0
    render_layer_workbench_header(
        "Brief",
        "Today Brief — Morning Snapshot",
        "One-page morning view: risk state, stale files, pending checks, sector rotation, and cycle stage.",
        [
            ("Risk Light",    friendly_value(risk_state),  status_kind(risk_state)),
            ("Worst Stress",  _worst_loss,                 "risk"),
            ("Paper Closed",  _brief_closed,               "supportive" if _brief_closed else "wait"),
            ("Leaders",       _brief_leaders,              "supportive" if _brief_leaders else "wait"),
        ],
    )

    # ── Earnings window alert ─────────────────────────────────────────────
    _ew_earn = read_csv(FILES["earnings_check"])
    _ew_exp  = read_csv(FILES["exposure"])
    _ew_df   = build_earnings_window(_ew_earn, _ew_exp, n_days=14)
    if not _ew_df.empty:
        _ew_risk = len(_ew_df[_ew_df["status"] == "RISK"])
        _ew_warn = len(_ew_df[_ew_df["status"] == "WARN"])
        _ew_color = "#991b1b" if _ew_risk > 0 else "#92400e"
        _ew_bg    = "#fee2e2" if _ew_risk > 0 else "#fef9c3"
        st.markdown(
            f'<div style="background:{_ew_bg};border-left:4px solid {_ew_color};'
            f'padding:10px 16px;border-radius:4px;margin-bottom:14px;">'
            f'<span style="font-weight:700;color:{_ew_color};">EARNINGS WINDOW:</span> '
            f'<span style="color:{_ew_color};">{_ew_risk} positions within 7 days · {_ew_warn} within 14 days. '
            f'Review or reduce before reporting dates.</span></div>',
            unsafe_allow_html=True,
        )
        render_badge_table(_ew_df, height=max(100, len(_ew_df)*46+54))

    # ── Concentration warning (single position > 20%) ─────────────────────
    _conc_exp = read_csv(FILES["exposure"])
    if not _conc_exp.empty and "effective_weight" in _conc_exp.columns:
        _conc_exp["_w"] = pd.to_numeric(_conc_exp["effective_weight"], errors="coerce").fillna(0)
        _conc_over = _conc_exp[_conc_exp["_w"] > 0.20]
        if not _conc_over.empty:
            _conc_names = ", ".join(_conc_over["ticker"].astype(str).tolist()) if "ticker" in _conc_over.columns else "unknown"
            st.markdown(
                f'<div style="background:#fee2e2;border-left:4px solid #991b1b;'
                f'padding:10px 16px;border-radius:4px;margin-bottom:14px;">'
                f'<span style="font-weight:700;color:#991b1b;">CONCENTRATION ALERT:</span> '
                f'<span style="color:#991b1b;">{len(_conc_over)} position(s) exceed 20% weight — '
                f'{escape(_conc_names)}. Reduce before adding new ideas.</span></div>',
                unsafe_allow_html=True,
            )

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Risk Light",     risk_state,        delta="RED — reduce first" if is_red else None,
              delta_color="inverse" if is_red else "normal")
    m2.metric("Stale Files",    stale_n,           delta="needs refresh" if stale_n > 4 else None, delta_color="inverse")
    m3.metric("Worst Stress",   _worst_loss)
    m4.metric("Paper Closed",   _closed)
    m5.metric("Avg Paper P&L",  _avg_pnl_str)
    m6.metric("Pending Checks", pending,           delta="manual required" if pending else None, delta_color="inverse")

    st.markdown("<hr style='border:none;border-top:1px solid #e5e7eb;margin:16px 0;'>",
                unsafe_allow_html=True)

    # ── two columns: left = actions, right = cycle + sectors ─────────────────
    _left, _right = st.columns([3, 2], gap="large")

    with _left:
        # Action buckets
        st.markdown(
            '<div style="font-size:10px;font-weight:700;letter-spacing:.14em;'
            'text-transform:uppercase;color:#111827;border-top:2px solid #111827;'
            'padding-top:10px;margin-bottom:16px;">Today Actions</div>',
            unsafe_allow_html=True,
        )
        _bucket_cfg = [
            ("TINY_PAPER_ONLY",       "Paper Possible",       "#22d3ee"),
            ("RISK_REDUCTION_FIRST",  "Reduce Risk First",    "#f87171"),
            ("RESEARCH_ONLY",         "Research Only",        "#9ca3af"),
            ("SKIP",                  "Skip",                 "#d1d5db"),
        ]
        for _bk, _label, _color in _bucket_cfg:
            _tickers = as_ticker_list(master, _bk) if not master.empty else ""
            if not _tickers or _tickers == "None":
                continue
            st.markdown(
                f'<div style="margin-bottom:10px;">'
                f'<div style="font-size:10px;font-weight:700;letter-spacing:.10em;'
                f'text-transform:uppercase;color:{_color};margin-bottom:4px;">{_label}</div>'
                f'<div style="font-size:13px;color:#111827;font-family:\'IBM Plex Mono\',monospace;">'
                f'{escape(_tickers)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Sector leaders & laggards
        st.markdown(
            '<div style="font-size:10px;font-weight:700;letter-spacing:.14em;'
            'text-transform:uppercase;color:#111827;border-top:1px solid #e5e7eb;'
            'padding-top:12px;margin:20px 0 14px 0;">Sector Rotation Today</div>',
            unsafe_allow_html=True,
        )
        if not sectors.empty and "rotation_score" in sectors.columns and "ticker" in sectors.columns:
            _sec = sectors.copy()
            _sec["rotation_score"] = pd.to_numeric(_sec["rotation_score"], errors="coerce")
            _leaders  = _sec.nlargest(4,  "rotation_score")[["ticker", "rotation_label", "rotation_score"]]
            _laggards = _sec.nsmallest(4, "rotation_score")[["ticker", "rotation_label", "rotation_score"]]

            def _chips_row(df_sub, color, bg):
                chips = ""
                for _, r in df_sub.iterrows():
                    chips += (
                        f'<span style="display:inline-block;padding:3px 10px;'
                        f'background:{bg};color:{color};border-radius:4px;'
                        f'font-family:\'IBM Plex Mono\',monospace;font-size:12px;'
                        f'font-weight:600;margin:2px 4px 2px 0;">'
                        f'{escape(str(r["ticker"]))}'
                        f'<span style="font-size:10px;font-weight:400;margin-left:6px;">'
                        f'{float(r["rotation_score"]):+.0f}</span></span>'
                    )
                return chips

            st.markdown(
                '<div style="font-size:10px;font-weight:600;color:#16a34a;'
                'text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;">▲ Leaders</div>'
                + _chips_row(_leaders, "#14532d", "#dcfce7") +
                '<div style="font-size:10px;font-weight:600;color:#dc2626;'
                'text-transform:uppercase;letter-spacing:.08em;margin:10px 0 4px 0;">▼ Laggards</div>'
                + _chips_row(_laggards, "#7f1d1d", "#fee2e2"),
                unsafe_allow_html=True,
            )
        else:
            st.caption("Sector data not yet available.")

    with _right:
        # Economic cycle stage card (reuse build_cycle_stage)
        _cs = build_cycle_stage(macro, vol, breadth)
        _stage = _cs.get("stage", "Mixed / Uncertain")
        _stage_color = _cs.get("color", "#0891b2")
        _kicker = _cs.get("kicker", "")
        st.markdown(
            f'<div style="border-top:2px solid {_stage_color};padding-top:10px;margin-bottom:16px;">'
            f'<div style="font-size:10px;font-weight:700;letter-spacing:.14em;'
            f'text-transform:uppercase;color:{_stage_color};margin-bottom:6px;">'
            f'Cycle Stage</div>'
            f'<div style="font-size:18px;font-weight:700;color:#111827;margin-bottom:4px;">'
            f'{escape(_stage)}</div>'
            f'<div style="font-size:11px;font-weight:600;letter-spacing:.08em;'
            f'text-transform:uppercase;color:{_stage_color};">{escape(_kicker)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Overweight chips
        _ow = _cs.get("overweight", [])
        _uw = _cs.get("underweight", [])
        if _ow:
            st.markdown(
                '<div style="font-size:10px;font-weight:600;color:#16a34a;'
                'letter-spacing:.08em;margin-bottom:4px;">Overweight</div>'
                + "".join(
                    f'<span style="display:inline-block;padding:2px 8px;background:#dcfce7;'
                    f'color:#14532d;border-radius:3px;font-size:11px;font-weight:600;'
                    f'margin:2px 3px 2px 0;">{t}</span>' for t in _ow
                ),
                unsafe_allow_html=True,
            )
        if _uw:
            st.markdown(
                '<div style="font-size:10px;font-weight:600;color:#dc2626;'
                'letter-spacing:.08em;margin:8px 0 4px 0;">Underweight</div>'
                + "".join(
                    f'<span style="display:inline-block;padding:2px 8px;background:#fee2e2;'
                    f'color:#7f1d1d;border-radius:3px;font-size:11px;font-weight:600;'
                    f'margin:2px 3px 2px 0;">{t}</span>' for t in _uw
                ),
                unsafe_allow_html=True,
            )

        # Safety rules reminder
        st.markdown(
            '<div style="border-top:1px solid #e5e7eb;padding-top:12px;margin-top:20px;">'
            '<div style="font-size:10px;font-weight:700;letter-spacing:.14em;'
            'text-transform:uppercase;color:#111827;margin-bottom:8px;">Rules Today</div>'
            '<div style="font-size:11px;color:#374151;line-height:1.7;">'
            '✗ &nbsp;No live orders &nbsp;&nbsp; '
            '✗ &nbsp;No weekly OTM options<br>'
            '✗ &nbsp;No broker connection &nbsp;&nbsp; '
            '✓ &nbsp;Paper only after manual checks'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border:none;border-top:1px solid #e5e7eb;margin:24px 0 16px 0;'>",
                unsafe_allow_html=True)

    # ── sector rotation chart ─────────────────────────────────────────────────
    if _PLOTLY and not sectors.empty and "rotation_score" in sectors.columns and "ticker" in sectors.columns:
        _sc = sectors.copy()
        _sc["rotation_score"] = pd.to_numeric(_sc["rotation_score"], errors="coerce")
        _sc = _sc.dropna(subset=["rotation_score"]).sort_values("rotation_score", ascending=True)
        if not _sc.empty:
            _sc_colors = ["#22d3ee" if v >= 0 else "#f87171" for v in _sc["rotation_score"]]
            _fig_brief_sec = go.Figure(go.Bar(
                x=_sc["rotation_score"].tolist(),
                y=_sc["ticker"].astype(str).tolist(),
                orientation="h",
                marker_color=_sc_colors,
                text=_sc["rotation_score"].map(lambda x: f"{x:+.0f}").tolist(),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Rotation score: %{x:+.0f}<extra></extra>",
            ))
            _fig_brief_sec.update_layout(
                height=max(180, len(_sc) * 34 + 60),
                margin=dict(l=10, r=50, t=26, b=20),
                title=dict(text="Sector Rotation Scores", font=dict(size=13, family="Inter,sans-serif"), x=0),
                xaxis_title="Rotation Score", yaxis_title="",
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(gridcolor="#e5e7eb", zeroline=True, zerolinecolor="#111827", zerolinewidth=1),
                yaxis=dict(gridcolor="#e5e7eb"),
                font=dict(family="Inter,sans-serif", size=12),
            )
            st.caption("Sector rotation scores — cyan = positive (leading), red = negative (lagging).")
            st.plotly_chart(_fig_brief_sec, use_container_width=True)

    # ── macro calendar ────────────────────────────────────────────────────────
    _macro_cal = build_macro_calendar()
    if not _macro_cal.empty:
        st.markdown("<hr style='border:none;border-top:1px solid #e5e7eb;margin:16px 0;'>",
                    unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:10px;font-weight:700;letter-spacing:.14em;'
            'text-transform:uppercase;color:#111827;margin-bottom:10px;">Upcoming Macro Events</div>',
            unsafe_allow_html=True,
        )
        render_badge_table(_macro_cal[["status","event","date","days_to_event","impact","what_to_do"]], height=260)
        if _PLOTLY:
            _mc_near = _macro_cal[_macro_cal["days_to_event"] <= 45].copy()
            if not _mc_near.empty:
                _mc_colors = [
                    "#b91c1c" if d <= 5 else "#facc15" if d <= 14 else "#22d3ee"
                    for d in _mc_near["days_to_event"]
                ]
                _fig_cal = go.Figure(go.Bar(
                    x=_mc_near["days_to_event"].tolist(),
                    y=_mc_near["event"].astype(str).tolist(),
                    orientation="h",
                    marker_color=_mc_colors,
                    text=_mc_near["days_to_event"].map(lambda d: f"{d}d").tolist(),
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>In %{x} days<extra></extra>",
                ))
                _fig_cal.update_layout(
                    height=max(140, len(_mc_near) * 32 + 50),
                    margin=dict(l=10, r=50, t=10, b=10),
                    xaxis_title="Days to Event", yaxis_title="",
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(gridcolor="#e5e7eb"),
                    yaxis=dict(gridcolor="#e5e7eb", autorange="reversed"),
                    font=dict(family="Inter,sans-serif", size=12),
                )
                st.plotly_chart(_fig_cal, use_container_width=True)

    # ── raw text for copy ─────────────────────────────────────────────────────
    with st.expander("Copy / Save Raw Text Brief"):
        brief = generate_daily_brief()
        st.code(brief, language="markdown")


def tab_focus_list():
    st.header("Focus List")

    focus = build_focus_list()
    if focus.empty:
        st.warning("No focus list data.")
        return

    active = int((focus["focus_bucket"] == "ACTIVE_WATCH").sum()) if "focus_bucket" in focus.columns else 0
    primary = int((focus["focus_bucket"] == "PRIMARY_WATCH").sum()) if "focus_bucket" in focus.columns else 0
    fix_first = int((focus["focus_bucket"] == "FIX_DATA_FIRST").sum()) if "focus_bucket" in focus.columns else 0
    do_not_touch = int((focus["focus_bucket"] == "DO_NOT_TOUCH").sum()) if "focus_bucket" in focus.columns else 0

    render_layer_workbench_header(
        "Focus",
        "Focus List — Closest To Useful",
        "Ranks tickers by focus score: how close they are to a valid trigger with clean data and passing checks.",
        [
            ("Main Watch",  active,       "paper"),
            ("Priority",    primary,      "cyan"),
            ("Fix First",   fix_first,    "risk"    if fix_first    else "supportive"),
            ("Do Not Touch",do_not_touch, "blocked" if do_not_touch else "supportive"),
        ],
    )

    def _fl_stat(label: str, val: int, accent: str) -> str:
        return (
            f'<div style="border-left:3px solid {accent};padding:6px 0 6px 14px;">'
            f'<div style="font-size:10px;font-weight:700;letter-spacing:.10em;text-transform:uppercase;'
            f'color:#4b5563;margin-bottom:3px;">{escape(label)}</div>'
            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:20px;font-weight:600;'
            f'color:#111827;">{val}</div>'
            f'</div>'
        )
    st.markdown(
        '<div style="border-top:2px solid #111827;padding:14px 0 18px 0;margin:0 0 16px 0;'
        'display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;">'
        + _fl_stat("Main Watch", active, "#c084fc")
        + _fl_stat("Priority Watch", primary, "#22d3ee")
        + _fl_stat("Fix Data First", fix_first, "#f87171")
        + _fl_stat("Do Not Touch", do_not_touch, "#9ca3af")
        + '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div class="command-grid">
      <div class="command-panel command-paper">
        <div class="command-label">Main Watch</div>
        <div class="command-title">Closest To A Useful Level</div>
        <div class="command-text">These tickers are near triggers, but risk and action checks still limit them.</div>
      </div>
      <div class="command-panel command-cyan">
        <div class="command-label">Priority Watch</div>
        <div class="command-title">Worth Watching</div>
        <div class="command-text">The research evidence matters, but triggers or human checks may not be ready.</div>
      </div>
      <div class="command-panel command-risk">
        <div class="command-label">Fix First</div>
        <div class="command-title">Do Not Treat Missing Data As Confidence</div>
        <div class="command-text">If data or human checks have not passed, fix the input before action.</div>
      </div>
      <div class="command-panel command-blocked">
        <div class="command-label">Do Not Touch</div>
        <div class="command-title">Skipping Also Saves Attention</div>
        <div class="command-text">Do not force closed or blocked rows into new samples.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    display = focus.copy()
    if "nearest_trigger_distance_pct" in display.columns:
        display["nearest_trigger_distance_pct"] = pd.to_numeric(display["nearest_trigger_distance_pct"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2f}%")
    if "focus_score" in display.columns:
        display["focus_score"] = pd.to_numeric(display["focus_score"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.1f}")

    if _PLOTLY and not focus.empty and "focus_score" in focus.columns and "ticker" in focus.columns:
        _fl_df = focus.copy()
        _fl_df["focus_score"] = pd.to_numeric(_fl_df["focus_score"], errors="coerce")
        _fl_df = _fl_df.dropna(subset=["focus_score"]).sort_values("focus_score", ascending=True).tail(20)
        if not _fl_df.empty:
            _fl_colors = [
                "#c084fc" if str(r.get("focus_bucket","")).upper() == "ACTIVE_WATCH"
                else "#22d3ee" if str(r.get("focus_bucket","")).upper() == "PRIMARY_WATCH"
                else "#f87171" if str(r.get("focus_bucket","")).upper() == "FIX_DATA_FIRST"
                else "#9ca3af"
                for _, r in _fl_df.iterrows()
            ]
            _fig_fl = go.Figure(go.Bar(
                x=_fl_df["focus_score"].tolist(),
                y=_fl_df["ticker"].astype(str).tolist(),
                orientation="h",
                marker_color=_fl_colors,
                text=_fl_df["focus_score"].map(lambda x: f"{x:.1f}").tolist(),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>",
            ))
            _fig_fl.update_layout(
                height=max(180, len(_fl_df) * 34 + 60),
                margin=dict(l=10, r=50, t=28, b=20),
                title=dict(text="Focus Score — Top Tickers (purple=active, cyan=priority)", font=dict(size=12), x=0),
                xaxis_title="Focus Score", yaxis_title="",
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(gridcolor="#e5e7eb"),
                yaxis=dict(gridcolor="#e5e7eb"),
                font=dict(family="Inter,sans-serif", size=12),
            )
            st.plotly_chart(_fig_fl, use_container_width=True)

    st.subheader("Ticker Ranking")
    render_badge_table(display, height=620)

    st.divider()
    st.subheader("Watchlist Aging")
    st.caption("Tickers that have been on the focus list too long without a paper trade may indicate anchoring bias.")
    _wl_df = build_watchlist_aging(focus, read_csv(FILES["paper_ledger"]))
    if _wl_df.empty:
        st.info("No focus list data for aging analysis.")
    else:
        _stale = (_wl_df["action_needed"].str.contains("45+|90+|Stale", na=False)).sum() if not _wl_df.empty else 0
        if _stale > 0:
            st.warning(f"⚠ {_stale} ticker(s) have been on the watch list for 45+ days with no paper trade. Review for anchoring bias.")
        render_badge_table(_wl_df, height=max(200, len(_wl_df) * 36 + 40))


def tab_daily_desk():
    queue = build_today_action_queue()
    master = read_csv(FILES["master_v2"])
    pretrade = read_csv(FILES["pre_trade"])

    risk_state = "NO_DATA"
    if not master.empty and "L8_state" in master.columns:
        _rs_mode = master["L8_state"].astype(str).mode()
        risk_state = _rs_mode.iloc[0] if not _rs_mode.empty else "UNKNOWN"
    pending_manual = count_value(pretrade, "final_status", "PENDING_MANUAL_CHECKS")
    high_priority = count_value(queue, "priority", "HIGH")
    blocked_rows = count_contains(queue, "desk_status", "DO_NOT_TOUCH") + count_contains(queue, "desk_status", "FIX_DATA_FIRST")

    render_layer_workbench_header(
        "Today",
        "Start Here In The Morning",
        "This tells you what to check first, what is blocked, and where each ticker should go next.",
        [
            ("Risk Light", friendly_value(risk_state), "risk" if str(risk_state).upper() == "RED" else "cyan"),
            ("High Priority", high_priority, "risk" if high_priority else "watch"),
            ("Needs Human Check", pending_manual, "cyan" if pending_manual else "watch"),
            ("Blocked / Fix First", blocked_rows, "blocked" if blocked_rows else "watch"),
        ],
    )

    run_status = build_run_status()
    health = read_csv(FILES["data_source_health"])
    vault_alerts = read_csv(FILES["vault_alerts"])
    vault_risk = 0
    if not vault_alerts.empty and "status" in vault_alerts.columns:
        vault_risk = int((vault_alerts["status"].astype(str).str.upper() != "OK").sum())

    st.subheader("Daily Workflow")
    workflow_rows = build_today_workflow(risk_state, queue, pretrade, run_status, health, vault_risk)
    render_workflow_steps(workflow_rows)

    st.subheader("Today Action Queue")
    if queue.empty:
        st.info("No daily queue data yet.")
    else:
        render_target_quad(queue, "First Four Tickers To Check")
        if _PLOTLY and "focus_score" in queue.columns and "ticker" in queue.columns:
            _dd_df = queue.copy()
            _dd_df["focus_score"] = pd.to_numeric(_dd_df["focus_score"], errors="coerce")
            _dd_df = _dd_df.dropna(subset=["focus_score"]).sort_values("focus_score", ascending=True).tail(16)
            if not _dd_df.empty:
                _dd_colors = [
                    "#f87171" if str(r.get("desk_status","")).upper() in ("DO_NOT_TOUCH", "FIX_DATA_FIRST")
                    else "#a855f7" if str(r.get("priority","")).upper() == "HIGH"
                    else "#22d3ee"
                    for _, r in _dd_df.iterrows()
                ]
                _fig_dd = go.Figure(go.Bar(
                    x=_dd_df["focus_score"].tolist(),
                    y=_dd_df["ticker"].astype(str).tolist(),
                    orientation="h",
                    marker_color=_dd_colors,
                    text=_dd_df["focus_score"].map(lambda x: f"{x:.1f}").tolist(),
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}<extra></extra>",
                ))
                _fig_dd.update_layout(
                    height=max(160, len(_dd_df) * 32 + 50),
                    margin=dict(l=10, r=50, t=10, b=10),
                    xaxis_title="Focus Score", yaxis_title="",
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(gridcolor="#e5e7eb"),
                    yaxis=dict(gridcolor="#e5e7eb"),
                    font=dict(family="Inter,sans-serif", size=12),
                )
                st.plotly_chart(_fig_dd, use_container_width=True)
        compact_cols = [c for c in [
            "priority", "ticker", "desk_status", "focus_score",
            "master_action", "blocked_by", "next_station"
        ] if c in queue.columns]
        render_badge_table(queue[compact_cols].head(12), height=360)
        with st.expander("Open Full Queue"):
            render_badge_table(queue.head(20), height=520)

    st.subheader("Today's Key Warnings")
    desk_blotter = build_daily_desk_signal_blotter(queue, pretrade, master)
    render_badge_table(desk_blotter, height=260)

    desk_tabs = st.tabs([
        "Today Overview",
        "Today Brief",
        "Focus List",
        "Single Ticker",
        "Trigger Levels",
        "If This, Then That",
    ])
    with desk_tabs[0]:
        tab_command_center()
    with desk_tabs[1]:
        tab_daily_brief()
    with desk_tabs[2]:
        tab_focus_list()
    with desk_tabs[3]:
        tab_ticker_drilldown()
    with desk_tabs[4]:
        tab_trigger_board()
    with desk_tabs[5]:
        tab_decision_playbook()


def tab_overview():  # noqa: C901
    # ── data loading ──────────────────────────────────────────────────────────
    master       = read_csv(FILES["master_v2"])
    health       = read_csv(FILES["data_source_health"])
    run_status   = build_run_status()
    pretrade     = read_csv(FILES["pre_trade"])
    vault_alerts = read_csv(FILES["vault_alerts"])
    queue        = build_today_action_queue()

    risk_state = "NO_DATA"
    if not master.empty and "L8_state" in master.columns:
        _rs_mode = master["L8_state"].astype(str).mode()
        risk_state = _rs_mode.iloc[0] if not _rs_mode.empty else "UNKNOWN"

    run_counts    = run_status["status"].value_counts() if not run_status.empty and "status" in run_status.columns else pd.Series(dtype=int)
    health_counts = health["status"].value_counts()     if not health.empty     and "status" in health.columns     else pd.Series(dtype=int)

    vault_risk     = int((vault_alerts["status"].astype(str).str.upper() != "OK").sum()) if not vault_alerts.empty and "status" in vault_alerts.columns else 0
    total_tickers  = len(master)
    fresh_count    = int(run_counts.get("FRESH", 0))
    stale_count    = int(run_counts.get("STALE", 0)) + int(run_counts.get("MISSING", 0))
    data_risk      = int(health_counts.get("RISK", 0))
    pending_manual = count_value(pretrade, "final_status", "PENDING_MANUAL_CHECKS")
    now_label      = datetime.now().strftime("%b %d, %Y  %H:%M")
    n_queue        = len(queue) if not queue.empty else 0
    high_p         = count_value(queue, "priority", "HIGH") if not queue.empty else 0

    is_risk_red  = str(risk_state).upper() == "RED"
    risk_display = "RED" if is_risk_red else (str(risk_state).upper() if risk_state not in ("NO_DATA", "") else "—")

    # ── one-line alert — plain text, no box, no fill ───────────────────────────
    alerts = []
    if is_risk_red:    alerts.append("RISK RED")
    if stale_count:    alerts.append(f"{stale_count} STALE")
    if data_risk:      alerts.append(f"{data_risk} DATA RISK")
    if pending_manual: alerts.append(f"{pending_manual} PENDING")
    if alerts:
        st.markdown(
            f'<p style="font-size:11px;font-weight:700;letter-spacing:.08em;'
            f'color:#b91c1c;margin:12px 0 6px 0;padding:0;">'
            f'{"  ·  ".join(alerts)}</p>',
            unsafe_allow_html=True,
        )

    # ── two-column layout — no decorative fills anywhere ──────────────────────
    col_left, col_right = st.columns([11, 6], gap="large")

    # ── helper: minimal section label ─────────────────────────────────────────
    def _head(title, sub=""):
        sub_part = (
            f'<span style="font-size:11px;font-weight:400;color:#9ca3af;'
            f'margin-left:12px;letter-spacing:0;">{escape(sub)}</span>'
        ) if sub else ""
        return (
            f'<div style="border-top:1px solid #111827;padding-top:10px;margin-bottom:36px;">'
            f'<span style="font-size:10px;font-weight:700;letter-spacing:.14em;'
            f'text-transform:uppercase;color:#111827;">{escape(title)}</span>'
            f'{sub_part}</div>'
        )

    # ══════════════ LEFT — ticker queue ═══════════════════════════════════════
    with col_left:
        st.markdown(_head("Today's Queue", f"{n_queue} tickers  ·  {high_p} high priority"), unsafe_allow_html=True)

        if queue.empty:
            st.caption("No queue — run the daily runner first.")
        else:
            _action_short = {
                "RISK_REDUCTION_FIRST": "Reduce Risk First",
                "RESEARCH_ONLY":        "Research Only",
                "PAPER_POSSIBLE":       "Paper Possible",
                "WAIT":                 "Wait",
                "SKIP":                 "Skip",
                "WATCH":                "Watch",
                "NO_DATA":              "No Data",
            }
            # header row
            tbl = (
                '<div style="display:flex;padding:0 0 6px 0;'
                'border-bottom:1px solid #d1d5db;">'
                '<div style="flex:0 0 52px;font-size:9px;font-weight:700;'
                'letter-spacing:.12em;text-transform:uppercase;color:#9ca3af;">Pri</div>'
                '<div style="flex:0 0 84px;font-size:9px;font-weight:700;'
                'letter-spacing:.12em;text-transform:uppercase;color:#9ca3af;">Ticker</div>'
                '<div style="flex:1;font-size:9px;font-weight:700;'
                'letter-spacing:.12em;text-transform:uppercase;color:#9ca3af;">Decision</div>'
                '<div style="flex:1;font-size:9px;font-weight:700;'
                'letter-spacing:.12em;text-transform:uppercase;color:#9ca3af;">Go To</div>'
                '</div>'
            )
            for _, row in queue.iterrows():
                pri    = str(row.get("priority", "—"))
                tick   = str(row.get("ticker",   "—"))
                raw_a  = str(row.get("master_action", "—"))
                action = _action_short.get(raw_a, raw_a.replace("_", " ").title())
                nxt    = str(row.get("next_station", "—"))
                is_hi  = pri.upper() == "HIGH"
                pri_c  = "#b91c1c" if is_hi else "#6b7280"
                tbl += (
                    f'<div style="display:flex;padding:7px 0;'
                    f'border-bottom:1px solid #f3f4f6;align-items:baseline;">'
                    f'<div style="flex:0 0 52px;font-size:10px;font-weight:700;'
                    f'color:{pri_c};letter-spacing:.04em;">{escape(pri)}</div>'
                    f'<div style="flex:0 0 84px;font-size:13px;font-weight:700;'
                    f'color:#111827;font-family:\'IBM Plex Mono\',monospace;'
                    f'letter-spacing:.02em;">{escape(tick)}</div>'
                    f'<div style="flex:1;font-size:12px;color:#374151;'
                    f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
                    f'{escape(action)}</div>'
                    f'<div style="flex:1;font-size:12px;color:#4b5563;'
                    f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
                    f'{escape(nxt)}</div>'
                    f'</div>'
                )
            st.markdown(f'<div style="margin-bottom:6px;">{tbl}</div>', unsafe_allow_html=True)

            # ── quick ticker opener ─────────────────────────────────────────
            _ticker_opts = queue["ticker"].dropna().astype(str).drop_duplicates().tolist()
            st.markdown(
                '<div style="margin-top:14px;padding-top:12px;border-top:1px solid #f3f4f6;">'
                '<span style="font-size:10px;font-weight:700;letter-spacing:.12em;'
                'text-transform:uppercase;color:#9ca3af;">Open In Notebook</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.selectbox(
                "open_ticker_label",
                _ticker_opts,
                key="home_active_ticker",
                label_visibility="collapsed",
            )

    # ══════════════ RIGHT — status + checklist ════════════════════════════════
    with col_right:
        # status rows — no box, no fill, just lines + numbers
        st.markdown(_head("Status", now_label), unsafe_allow_html=True)

        def _sr(label, val, red=False, amber=False):
            fg = "#b91c1c" if red else ("#b45309" if amber else "#111827")
            return (
                f'<div style="display:flex;justify-content:space-between;'
                f'align-items:baseline;padding:8px 0 8px 0;border-bottom:1px solid #f3f4f6;">'
                f'<span style="font-size:10px;font-weight:600;color:#4b5563;'
                f'text-transform:uppercase;letter-spacing:.1em;'
                f'font-family:Inter,sans-serif;">{escape(label)}</span>'
                f'<span style="font-size:20px;font-weight:700;color:{fg};'
                f'font-family:\'IBM Plex Mono\',\'Roboto Mono\',monospace;'
                f'letter-spacing:-.02em;margin-right:18px;">{escape(str(val))}</span>'
                f'</div>'
            )

        # ── paper snapshot (compact one row) ──────────────────────────────────
        _ledger = read_csv(FILES.get("paper_ledger", ROOT / "paper_portfolio_ledger.csv"))
        _open_pos   = 0
        _closed_pos = 0
        _avg_pnl    = ""
        if not _ledger.empty:
            _open_pos   = int((_ledger["status"].astype(str).str.upper().isin(["PAPER_CANDIDATE","WATCHLIST","OPEN"])).sum()) if "status" in _ledger.columns else 0
            _closed_pos = int(_ledger["status"].astype(str).str.upper().str.startswith("CLOSED").sum()) if "status" in _ledger.columns else 0
            if "pnl_pct" in _ledger.columns:
                _pnl_vals = pd.to_numeric(_ledger["pnl_pct"], errors="coerce").dropna()
                if not _pnl_vals.empty:
                    _avg_pnl = f"{_pnl_vals.mean()*100:+.1f}%"

        st.markdown(
            "<div>"
            + _sr("Risk",      risk_display, red=is_risk_red)
            + _sr("Tickers",   total_tickers)
            + _sr("Fresh",     fresh_count)
            + _sr("Stale",     stale_count,  red=stale_count > 4, amber=0 < stale_count <= 4)
            + _sr("Data Risk", data_risk,    red=bool(data_risk))
            + _sr("On Desk",   n_queue)
            + _sr("Paper Open", _open_pos)
            + _sr("Paper Closed", _closed_pos)
            + ("" if not _avg_pnl else _sr("Avg P&L", _avg_pnl, red=_avg_pnl.startswith("-")))
            + "</div>",
            unsafe_allow_html=True,
        )

        # ── quick refresh button ────────────────────────────────────────────────
        _runner_home = ROOT / "canyon_final_v9_step56_full_10_layer_daily_runner_v2.py"
        if _runner_home.exists():
            if st.button("▶  Run Update", key="home_run_refresh"):
                try:
                    subprocess.Popen(
                        [sys.executable, str(_runner_home)],
                        cwd=str(ROOT),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    st.toast("Runner started — reload in ~3 min", icon="✅")
                except Exception as _e:
                    st.error(f"Could not launch runner: {_e}")

        # checklist — no box, no numbered badges, just index + text
        st.markdown(
            f'<div style="margin-top:24px;">'
            + _head("Morning Checklist")
            + "</div>",
            unsafe_allow_html=True,
        )
        workflow_rows = build_today_workflow(risk_state, queue, pretrade, run_status, health, vault_risk)
        if workflow_rows.empty:
            st.caption("Run daily runner to populate checklist.")
        else:
            rows_html = "<div>"
            for i, (_, row) in enumerate(workflow_rows.iterrows()):
                step   = str(row.get("station", row.get("workflow_step", row.get("step", f"Step {i+1}"))))
                status = str(row.get("status", "")).upper()
                is_red = any(k in status for k in ("RISK", "RED", "BLOCKED"))
                fg     = "#b91c1c" if is_red else "#374151"
                rows_html += (
                    f'<div style="display:flex;gap:12px;padding:7px 0;'
                    f'border-bottom:1px solid #f3f4f6;align-items:baseline;">'
                    f'<span style="font-size:10px;color:#9ca3af;min-width:14px;">{i+1}</span>'
                    f'<span style="font-size:12px;color:{fg};line-height:1.4;">'
                    f'{escape(step)}</span>'
                    f'</div>'
                )
            st.markdown(rows_html + "</div>", unsafe_allow_html=True)

        # ── decision distribution mini chart ──────────────────────────────────
        if _PLOTLY and not master.empty and "master_action" in master.columns:
            _ac = master["master_action"].value_counts()
            _ac_labels = [friendly_value(str(k)) for k in _ac.index]
            _ac_vals   = _ac.values.tolist()
            _ac_colors = []
            for k in _ac.index:
                k = str(k).upper()
                if "RISK" in k or "REDUCE" in k:    _ac_colors.append("#f87171")
                elif "PAPER" in k or "TINY" in k:   _ac_colors.append("#c084fc")
                elif "RESEARCH" in k:               _ac_colors.append("#22d3ee")
                else:                               _ac_colors.append("#d1d5db")
            _fig_ac = go.Figure(go.Bar(
                x=_ac_vals, y=_ac_labels, orientation="h",
                marker_color=_ac_colors,
                text=[str(v) for v in _ac_vals], textposition="outside",
            ))
            _fig_ac.update_layout(
                height=max(120, len(_ac_labels) * 36),
                margin=dict(l=0, r=20, t=18, b=4),
                title=dict(text="Decision Distribution", font=dict(size=11, color="#6b7280"), x=0),
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(tickfont=dict(size=11)),
                plot_bgcolor="white", paper_bgcolor="white",
                font=dict(family="Inter,sans-serif", size=11),
            )
            st.plotly_chart(_fig_ac, use_container_width=True)

    # footer
    st.markdown(
        f'<div style="margin:32px 0 4px 0;padding-top:10px;border-top:1px solid #f3f4f6;">'
        f'<span style="font-size:10px;color:#d1d5db;letter-spacing:.04em;">'
        f'CANYON V9 · NO BROKER · NO LIVE ORDER · OPTIONS = L7 ONLY'
        f'&emsp;·&emsp;{escape(now_label)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def tab_master():
    st.header("Main Decision")
    master = read_csv(FILES["master_v2"])
    scorecard = read_csv(FILES["scorecard"])
    if master.empty:
        st.warning("Run Step 54 first.")
        return

    counts = master["master_action"].value_counts() if "master_action" in master.columns else pd.Series(dtype=int)
    l8_red = count_value(master, "L8_state", "RED")
    l1_missing = count_contains(master, "L1_state", "UNAVAILABLE") + count_contains(master, "L1_state", "NO_DATA")
    l9_blockers = count_contains(master, "L9_state", "BLOCKED") + count_contains(master, "L9_state", "PENDING")

    render_layer_workbench_header(
        "Main Decision",
        "Final Result From All 10 Layers",
        "This combines data, market mood, sector, basics, events, price trend, options, portfolio risk, action checks, and review learning into one result.",
        [
            ("Ticker Count", len(master), "cyan"),
            ("Reduce Risk First", int(counts.get("RISK_REDUCTION_FIRST", 0)), "risk"),
            ("Skip", int(counts.get("SKIP", 0)), "blocked"),
            ("Red Risk Light", l8_red, "risk" if l8_red else "supportive"),
        ],
    )

    funnel = pd.DataFrame([
        {
            "status": "FOUND",
            "stage": "Ticker Pool",
            "count": len(master),
            "meaning": "Number of tickers going through the 10-layer check.",
        },
        {
            "status": "REVIEW" if l1_missing else "OK",
            "stage": "Can We Trust The Data?",
            "count": len(master) - l1_missing,
            "meaning": f"{l1_missing} rows have unavailable or missing data.",
        },
        {
            "status": "RISK" if l8_red else "OK",
            "stage": "Does Risk Allow It?",
            "count": len(master) - l8_red,
            "meaning": f"{l8_red} rows are limited by the red risk light.",
        },
        {
            "status": "REVIEW" if l9_blockers else "OK",
            "stage": "Before-Action Check",
            "count": len(master) - l9_blockers,
            "meaning": f"{l9_blockers} rows are waiting, blocked, or already closed and cannot be repeated.",
        },
        {
            "status": "LEARNING_SAMPLE_PENDING" if count_contains(master, "L10_state", "PENDING") else "OK",
            "stage": "Review Learning",
            "count": count_contains(master, "L10_state", "HAS_SAMPLE") + count_contains(master, "L10_state", "OPEN_OR_WATCH"),
            "meaning": "Only clean paper-log results can teach Layer 10.",
        },
    ])
    st.subheader("Step-By-Step Funnel To The Final Result")
    render_badge_table(funnel, height=240)

    st.subheader("Filters")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        ticker_query = st.text_input("Ticker Contains", "", key="master_ticker_query")
    with f2:
        action_filter = st.selectbox("Final Decision", ["All"] + sorted(master["master_action"].dropna().astype(str).unique().tolist()) if "master_action" in master.columns else ["All"], key="master_action_filter")
    with f3:
        l8_filter = st.selectbox("Risk Layer Status", ["All"] + sorted(master["L8_state"].dropna().astype(str).unique().tolist()) if "L8_state" in master.columns else ["All"], key="master_l8_filter")
    with f4:
        l9_filter = st.selectbox("Action Check Status", ["All"] + sorted(master["L9_state"].dropna().astype(str).unique().tolist()) if "L9_state" in master.columns else ["All"], key="master_l9_filter")

    if _PLOTLY and not master.empty and "master_action" in master.columns:
        _dist = master["master_action"].value_counts().reset_index()
        _dist.columns = ["action", "count"]
        _dist["label"] = _dist["action"].apply(lambda x: friendly_value(str(x)))
        _dist["color"] = _dist["action"].apply(lambda x: (
            "#f87171" if "RISK" in str(x).upper() or "REDUCE" in str(x).upper()
            else "#c084fc" if "PAPER" in str(x).upper() or "TINY" in str(x).upper()
            else "#22d3ee" if "RESEARCH" in str(x).upper()
            else "#d1d5db"
        ))
        _fig_dist = go.Figure(go.Bar(
            x=_dist["label"], y=_dist["count"],
            marker_color=_dist["color"].tolist(),
            text=_dist["count"].tolist(), textposition="outside",
        ))
        _fig_dist.update_layout(
            height=260, margin=dict(l=10, r=10, t=10, b=20),
            xaxis=dict(gridcolor="#f3f4f6"),
            yaxis=dict(gridcolor="#f3f4f6", title="Tickers"),
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Inter,sans-serif", size=12),
        )
        _c_chart, _c_gap = st.columns([3, 1])
        with _c_chart:
            st.plotly_chart(_fig_dist, use_container_width=True)

    view = master.copy()
    _tq = str(ticker_query).strip().upper()
    if _tq and "ticker" in view.columns:
        view = view[view["ticker"].astype(str).str.upper().str.contains(_tq, regex=False)]
    if str(action_filter) != "All" and "master_action" in view.columns:
        view = view[view["master_action"].astype(str) == str(action_filter)]
    if str(l8_filter) != "All" and "L8_state" in view.columns:
        view = view[view["L8_state"].astype(str) == str(l8_filter)]
    if str(l9_filter) != "All" and "L9_state" in view.columns:
        view = view[view["L9_state"].astype(str) == str(l9_filter)]

    tabs = st.tabs(["Filtered Table", "Layer Status Counts", "Score Sheet", "Full Report"])
    with tabs[0]:
        st.caption(f"Showing {len(view)} / {len(master)} rows.")
        cols = [c for c in [
            "ticker", "master_action", "master_reason", "stack_score_avg", "stack_score_min",
            "L1_state", "L2_state", "L3_state", "L4_state", "L5_state",
            "L6_state", "L7_state", "L8_state", "L9_state", "L10_state"
        ] if c in view.columns]
        render_badge_table(view[cols], height=620)

        if not view.empty and "ticker" in view.columns:
            st.subheader("Open One Ticker Explanation")
            ticker_options = view["ticker"].dropna().astype(str).drop_duplicates().tolist()
            explain_ticker = st.selectbox(
                "Choose Ticker To Explain",
                ticker_options,
                key="master_explain_ticker",
            )
            cards = read_csv(FILES["action_cards"])
            triggers = read_csv(FILES["watch_triggers"])
            pretrade = read_csv(FILES["pre_trade"])
            options = read_csv(FILES["options_decision"])
            v8_gate = read_csv(FILES["v8_l9_gate"])
            technicals = read_csv(FILES["technicals"])
            events = read_csv(FILES["events"])
            market = read_csv(FILES["market_snapshot"])
            gaps = build_gap_queue(master, market)

            row = first_row(master, explain_ticker)
            card = first_row(cards, explain_ticker)
            trigger = first_row(triggers, explain_ticker)
            check = first_row(pretrade, explain_ticker)
            option = first_row(options, explain_ticker)
            v8_check = first_row(v8_gate, explain_ticker)
            tech = first_row(technicals, explain_ticker)
            event = first_row(events, explain_ticker)
            market_row = first_row(market, explain_ticker)
            ticker_gaps = ticker_rows(gaps, explain_ticker)

            action = row.get("master_action", "NO_DATA")
            reason = row.get("master_reason", "")
            st.markdown(f"""
            <div class="ticker-hero ticker-{status_kind(action)}">
              <div>
                <div class="ticker-label">Final Decision</div>
                <div class="ticker-title">{escape(explain_ticker)} · {escape(str(action))}</div>
                <div class="ticker-reason">{escape(str(reason))}</div>
              </div>
              <div class="ticker-scorebox">
                <div class="ticker-score-label">Average / Lowest Score</div>
                <div class="ticker-score">{escape(str(row.get('stack_score_avg', '')))} / {escape(str(row.get('stack_score_min', '')))}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            intelligence = build_ticker_intelligence_panel(
                explain_ticker, row, card, trigger, check, option, v8_check, tech, event, market_row, ticker_gaps
            )
            display_cols = [c for c in [
                "status", "layer", "signal", "score", "source_type", "source_file", "operator_read"
            ] if c in intelligence.columns]
            render_badge_table(intelligence[display_cols], height=520)
            render_ticker_source_trail(intelligence)

    with tabs[1]:
        rows = []
        for i in range(1, 11):
            col = f"L{i}_state"
            if col not in master.columns:
                continue
            for state, count in master[col].astype(str).value_counts().items():
                rows.append({
                    "status": state,
                    "layer": f"L{i}",
                    "state": state,
                    "count": int(count),
                    "share": f"{count / max(len(master), 1) * 100:.1f}%",
                })
        render_badge_table(pd.DataFrame(rows), height=520)

    with tabs[2]:
        if scorecard.empty:
            st.info("No scorecard yet.")
        else:
            render_badge_table(scorecard, height=360)

    with tabs[3]:
        st.markdown(read_md(FILES["master_report_v2"]))


def tab_layers():
    st.header("10-Layer Map")
    audit = read_csv(FILES["layer_audit"])
    if audit.empty:
        st.warning("No layer audit file found.")
        return

    scores = pd.to_numeric(audit.get("maturity_score_0_5", pd.Series(dtype=str)), errors="coerce")
    avg = scores.mean() if not scores.dropna().empty else 0
    operational = count_value(audit, "maturity_status", "Operational")
    usable = count_value(audit, "maturity_status", "Usable")
    missing_outputs = int(audit.get("missing_outputs", pd.Series(dtype=str)).astype(str).str.strip().ne("").sum()) if "missing_outputs" in audit.columns else 0

    render_layer_workbench_header(
        "Structure",
        "Is The 10-Layer System Built?",
        "This checks whether each layer has a clear job, output files, a role in the final decision, and a next build step.",
        [
            ("Average Completion", f"{avg:.1f}/5", "supportive" if avg >= 4 else "cyan"),
            ("Working", operational, "supportive"),
            ("Usable", usable, "wait"),
            ("Missing Outputs", missing_outputs, "risk" if missing_outputs else "supportive"),
        ],
    )

    # ── Radar chart: 10-layer maturity scores ────────────────────────────────
    if _PLOTLY and not audit.empty and "maturity_score_0_5" in audit.columns:
        _layer_names = audit.get("layer_id", pd.Series()).astype(str).tolist()
        _layer_scores = pd.to_numeric(audit["maturity_score_0_5"], errors="coerce").fillna(0).tolist()
        if _layer_names and _layer_scores:
            # Close the polygon
            _rn = _layer_names + [_layer_names[0]]
            _rs = _layer_scores + [_layer_scores[0]]
            _fig_radar = go.Figure(go.Scatterpolar(
                r=_rs,
                theta=_rn,
                fill="toself",
                fillcolor="rgba(34,211,238,0.15)",
                line=dict(color="#22d3ee", width=2),
                marker=dict(color="#22d3ee", size=6),
                hovertemplate="<b>%{theta}</b><br>Score: %{r}/5<extra></extra>",
            ))
            _fig_radar.add_trace(go.Scatterpolar(
                r=[5] * (len(_layer_names) + 1),
                theta=_rn,
                mode="lines",
                line=dict(color="#e5e7eb", width=1, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ))
            _fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(range=[0, 5], showticklabels=True, tickfont=dict(size=10),
                                   gridcolor="#e5e7eb", linecolor="#e5e7eb"),
                    angularaxis=dict(tickfont=dict(size=11, family="IBM Plex Mono,monospace"),
                                     gridcolor="#e5e7eb", linecolor="#e5e7eb"),
                    bgcolor="white",
                ),
                paper_bgcolor="white",
                height=380,
                margin=dict(l=60, r=60, t=30, b=30),
                font=dict(family="Inter,sans-serif", size=12),
                showlegend=False,
            )
            _rc1, _rc2 = st.columns([2, 3])
            with _rc1:
                st.markdown(
                    f'<div style="padding:16px 0;">'
                    f'<div style="font-size:10px;font-weight:700;letter-spacing:.14em;'
                    f'text-transform:uppercase;color:#111827;margin-bottom:8px;">Maturity Score</div>'
                    f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:36px;'
                    f'font-weight:700;color:#111827;">{avg:.1f}<span style="font-size:18px;'
                    f'color:#6b7280;"> / 5</span></div>'
                    f'<div style="font-size:12px;color:#6b7280;margin-top:4px;">'
                    f'avg across 10 layers</div>'
                    f'<div style="margin-top:16px;font-size:12px;color:#374151;line-height:1.6;">'
                    f'Operational: <b>{operational}</b><br>'
                    f'Usable: <b>{usable}</b><br>'
                    f'Missing outputs: <b>{missing_outputs}</b>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with _rc2:
                st.plotly_chart(_fig_radar, use_container_width=True)

    st.subheader("Layer Cards")
    card_html = ['<div class="layer-grid">']
    for _, row in audit.iterrows():
        layer = f"{row.get('layer_id', '')} · {row.get('layer_name', '')}"
        card_html.append(layer_card_html(
            layer,
            row.get("maturity_status", "NO_DATA"),
            f"{row.get('maturity_score_0_5', '')}/5",
            row.get("core_question", row.get("next_build", "")),
        ))
    card_html.append("</div>")
    st.markdown("".join(card_html), unsafe_allow_html=True)

    tabs = st.tabs(["Completion Check", "What This Layer Does", "What To Build Next", "Build Plan"])
    with tabs[0]:
        cols = [c for c in [
            "layer_id", "layer_name", "maturity_score_0_5", "maturity_status",
            "found_outputs", "missing_outputs", "next_build"
        ] if c in audit.columns]
        render_badge_table(audit[cols], height=520)
    with tabs[1]:
        cols = [c for c in [
            "layer_id", "layer_name", "purpose", "core_question",
            "decision_use", "user_vision_mapping"
        ] if c in audit.columns]
        render_badge_table(audit[cols], height=560)
    with tabs[2]:
        next_cols = [c for c in [
            "layer_id", "layer_name", "maturity_status", "missing_outputs", "next_build"
        ] if c in audit.columns]
        render_badge_table(audit[next_cols], height=500)
    with tabs[3]:
        st.markdown(read_md(FILES["build_plan"]))


def tab_l2_l6():  # noqa: C901
    st.header("Middle & Upper Layers")
    sub = st.tabs([
        "Layer 2: Market Mood",
        "Layer 3: Sectors",
        "Layer 4: Basics",
        "Layer 5: Events",
        "Layer 6: Price Trend",
        "Layer 7: Options",
        "Layer 8: Risk",
        "Layer 9: Action Check",
        "Layer 10: Review",
    ])
    with sub[0]:
        signals = read_csv(FILES["macro_signals"])
        breadth = read_csv(FILES["index_breadth"])
        vol = read_csv(FILES["volatility_regime"])
        regime = "NO_DATA"
        if not vol.empty and {"metric", "value"}.issubset(vol.columns):
            match = vol[vol["metric"].astype(str).eq("vol_regime")]
            if not match.empty:
                regime = str(match.iloc[0].get("value", "NO_DATA"))
        render_layer_workbench_header(
            "L2",
            "Market Mood",
            "Check the broad market, rates, volatility, credit, and market breadth before a single ticker.",
            [
                ("Market Status", friendly_value(regime), status_kind(regime)),
                ("Signal Count", len(signals), "cyan"),
                ("No Data", count_value(signals, "data_status", "NO_DATA"), "blocked"),
                ("Breadth Rows", len(breadth), "wait"),
            ],
        )
        _l2_supportive = count_value(signals, "trend_state", "BULL") + count_value(signals, "trend_state", "UPTREND") if not signals.empty else 0
        _l2_hostile = count_value(signals, "trend_state", "BEAR") + count_value(signals, "trend_state", "DOWNTREND") if not signals.empty else 0
        _l2_breadth_above50 = int(breadth["above_50dma"].astype(str).str.upper().isin(["TRUE", "1", "YES"]).sum()) if not breadth.empty and "above_50dma" in breadth.columns else 0
        st.subheader("What Market Mood Says Today")
        render_badge_table(pd.DataFrame([
            {
                "status": regime,
                "layer_question": "What is the current market regime?",
                "today_read": f"Vol regime: {regime}. {_l2_supportive} supportive signals, {_l2_hostile} hostile signals.",
                "what_it_means_for_tickers": "High vol or hostile regime means every ticker needs stronger evidence to act. Tickers swim against the tape.",
                "when_this_layer_blocks_action": "When regime is high-vol, bear, or hostile — all position aggression is held back regardless of single-ticker setup.",
            },
            {
                "status": "REVIEW" if not breadth.empty else "NO_DATA",
                "layer_question": "How healthy is market breadth?",
                "today_read": f"{_l2_breadth_above50} of {len(breadth)} index components above 50-day MA.",
                "what_it_means_for_tickers": "Narrow breadth (few names above moving average) means the rally is concentrated — individual names may be riskier than they look.",
                "when_this_layer_blocks_action": "If breadth is narrow or declining, individual breakouts are less reliable. Treat tactical setups with more skepticism.",
            },
        ]), height=200)
        render_layer_tables([
            ("Macro Signals", signals, ["ticker", "description", "last_close", "ret_5d", "ret_20d", "ret_63d", "trend_state", "realized_vol_20d", "vol_z_1y", "data_status"], 360),
            ("Breadth", breadth, ["ticker", "close", "above_20dma", "above_50dma", "ret_20d", "trend_state"], 320),
            ("Volatility", vol, None, 220),
            ("Raw Report", pd.DataFrame([{"status": "REPORT", "content": read_md(FILES["macro_report"])}]), ["status", "content"], 520),
        ])
    with sub[1]:
        sectors = read_csv(FILES["sector_scores"])
        theme = read_csv(FILES["theme_heatmap"])
        _l3_signals = read_csv(FILES["macro_signals"])
        _l3_vol     = read_csv(FILES["volatility_regime"])
        _l3_breadth = read_csv(FILES["index_breadth"])
        _cs = build_cycle_stage(_l3_signals, _l3_vol, _l3_breadth)

        render_layer_workbench_header(
            "L3",
            "Sectors And Themes",
            f"Cycle stage: {_cs['stage']}  ·  {_cs['kicker']}",
            [
                ("Cycle Stage", _cs["stage"].split("/")[0].strip(), "supportive" if "Growth" in _cs["stage"] or "Recovery" in _cs["stage"] else ("risk" if "Recession" in _cs["stage"] or "Defensive" in _cs["stage"] else "cyan")),
                ("Leaders", count_value(sectors, "rotation_label", "LEADER"), "supportive"),
                ("VIX", _cs["vix_level"], "blocked" if _cs["vix_level"] > 25 else "supportive"),
                ("Breadth ≥50d", _cs["breadth_pct"], "supportive"),
            ],
        )

        # ── Economic Cycle Stage card ───────────────────────────────────────
        def _ticker_chips(tickers: list, bg: str, fg: str) -> str:
            if not tickers:
                return '<span style="color:#9ca3af;font-size:12px;">—</span>'
            chips = "".join(
                f'<span style="display:inline-block;margin:2px 4px 2px 0;padding:3px 10px;'
                f'background:{bg};color:{fg};font-family:\'IBM Plex Mono\',monospace;'
                f'font-size:12px;font-weight:600;border-radius:3px;">{t}</span>'
                for t in tickers
            )
            return chips

        _stage_html = (
            f'<div style="border-top:2px solid {_cs["color"]};padding:20px 0 24px 0;margin:0 0 32px 0;">'
            f'<div style="display:grid;grid-template-columns:minmax(0,1.6fr) minmax(0,1fr);gap:32px;align-items:start;">'
            # left: stage name + description + action
            f'<div>'
            f'<div style="font-size:10px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;'
            f'color:{_cs["color"]};margin-bottom:8px;">{escape(_cs["kicker"])}</div>'
            f'<div style="font-size:26px;font-weight:800;letter-spacing:-0.02em;color:#111827;'
            f'line-height:1.15;margin-bottom:10px;">{escape(_cs["stage"])}</div>'
            f'<div style="font-size:13px;color:#4b5563;line-height:1.55;max-width:520px;'
            f'margin-bottom:16px;">{escape(_cs["description"])}</div>'
            f'<div style="font-size:12px;font-weight:600;color:#111827;padding:10px 14px;'
            f'background:#f9fafb;border-left:3px solid {_cs["color"]};">'
            f'▶ {escape(_cs["action"])}</div>'
            f'</div>'
            # right: signals + overweight/underweight grid
            f'<div>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:20px;">'
            f'<div style="font-size:10px;font-weight:700;letter-spacing:.10em;text-transform:uppercase;'
            f'color:#9ca3af;padding:8px 12px;background:#f9fafb;">SPY {escape(_cs["spy_trend"])}</div>'
            f'<div style="font-size:10px;font-weight:700;letter-spacing:.10em;text-transform:uppercase;'
            f'color:#9ca3af;padding:8px 12px;background:#f9fafb;">IWM {escape(_cs["iwm_trend"])}</div>'
            f'<div style="font-size:10px;font-weight:700;letter-spacing:.10em;text-transform:uppercase;'
            f'color:#9ca3af;padding:8px 12px;background:#f9fafb;">VIX {escape(str(_cs["vix_level"]))} ({escape(_cs["vol_regime"])})</div>'
            f'<div style="font-size:10px;font-weight:700;letter-spacing:.10em;text-transform:uppercase;'
            f'color:#9ca3af;padding:8px 12px;background:#f9fafb;">QQQ vs SPY {escape(_cs["qqq_vs_spy"])}</div>'
            f'</div>'
            # overweight / neutral / underweight
            f'<div style="margin-bottom:10px;">'
            f'<div style="font-size:10px;font-weight:700;letter-spacing:.10em;text-transform:uppercase;'
            f'color:#16a34a;margin-bottom:6px;">▲ Overweight</div>'
            f'{_ticker_chips(_cs["overweight"], "#dcfce7", "#14532d")}'
            f'</div>'
            f'<div style="margin-bottom:10px;">'
            f'<div style="font-size:10px;font-weight:700;letter-spacing:.10em;text-transform:uppercase;'
            f'color:#6b7280;margin-bottom:6px;">= Neutral</div>'
            f'{_ticker_chips(_cs["neutral"], "#f3f4f6", "#374151")}'
            f'</div>'
            f'<div>'
            f'<div style="font-size:10px;font-weight:700;letter-spacing:.10em;text-transform:uppercase;'
            f'color:#dc2626;margin-bottom:6px;">▼ Underweight</div>'
            f'{_ticker_chips(_cs["underweight"], "#fee2e2", "#7f1d1d")}'
            f'</div>'
            f'</div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(_stage_html, unsafe_allow_html=True)

        # ── sector scores with overweight tag appended ──────────────────────
        if not sectors.empty:
            _ow_set = set(_cs["overweight"])
            _uw_set = set(_cs["underweight"])
            def _cycle_tag(t):
                if t in _ow_set: return "OVERWEIGHT"
                if t in _uw_set: return "UNDERWEIGHT"
                return "NEUTRAL"
            sectors = sectors.copy()
            sectors["cycle_signal"] = sectors["ticker"].astype(str).apply(_cycle_tag)
            sview = format_percent_columns(sectors, ["ret_5d", "ret_20d", "ret_63d",
                                                     "relative_20d_vs_spy", "relative_63d_vs_spy"])
            scols = [c for c in ["ticker", "theme", "cycle_signal", "ret_20d",
                                 "relative_20d_vs_spy", "relative_63d_vs_spy",
                                 "rotation_score", "rotation_label"] if c in sview.columns]
            render_badge_table(sview[scols])

        # ── Sector Rotation Heatmap ─────────────────────────────────────────
        if _PLOTLY and not sectors.empty and "rotation_score" in sectors.columns:
            st.subheader("Sector Rotation Heatmap")
            _ROT_COLOR = {
                "LEADER":  "#16a34a",
                "WATCH":   "#d97706",
                "NEUTRAL": "#6b7280",
                "LAGGARD": "#dc2626",
            }
            _CYCLE_ICON = {"OVERWEIGHT": "▲", "UNDERWEIGHT": "▼", "NEUTRAL": "="}
            _sh = sectors.copy()
            _sh["rotation_score"] = pd.to_numeric(_sh.get("rotation_score", 0), errors="coerce").fillna(0)
            _sh["rotation_label"] = _sh.get("rotation_label", "NEUTRAL").fillna("NEUTRAL").astype(str).str.upper()
            _sh["cycle_signal"] = _sh.get("cycle_signal", "NEUTRAL").fillna("NEUTRAL").astype(str).str.upper()
            _sh["bar_color"] = _sh["rotation_label"].map(lambda x: _ROT_COLOR.get(x, "#6b7280"))
            _sh["label"] = _sh.apply(
                lambda r: f"{_CYCLE_ICON.get(r['cycle_signal'], '=')} {r['ticker']}  ({r.get('theme', '')})", axis=1)
            _sh = _sh.sort_values("rotation_score", ascending=True)
            _fig_sec = go.Figure(go.Bar(
                x=_sh["rotation_score"],
                y=_sh["label"],
                orientation="h",
                marker_color=_sh["bar_color"].tolist(),
                text=_sh["rotation_score"].map(lambda x: f"{x:+.1f}"),
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Rotation Score: %{x:.1f}<extra></extra>"
                ),
            ))
            # Add a vertical zero-line annotation
            _fig_sec.add_vline(x=0, line_width=1, line_color="#9ca3af")
            _legend_rot = " ".join(
                f'<span style="display:inline-block;width:10px;height:10px;border-radius:2px;'
                f'background:{v};margin-right:4px;"></span>'
                f'<span style="font-size:12px;color:#374151;margin-right:14px;">{k}</span>'
                for k, v in _ROT_COLOR.items()
            )
            _legend_cyc = (
                '<span style="font-size:12px;color:#374151;margin-right:6px;">Cycle: </span>'
                '<span style="font-size:12px;color:#374151;margin-right:14px;">▲ Overweight</span>'
                '<span style="font-size:12px;color:#374151;margin-right:14px;">= Neutral</span>'
                '<span style="font-size:12px;color:#374151;">▼ Underweight</span>'
            )
            st.markdown(
                f'<div style="margin-bottom:6px">{_legend_rot} &nbsp;&nbsp; {_legend_cyc}</div>',
                unsafe_allow_html=True,
            )
            _fig_sec.update_layout(
                height=max(320, 70 + len(_sh) * 44),
                margin=dict(l=10, r=80, t=10, b=30),
                xaxis_title="Rotation Score",
                yaxis_title="",
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(gridcolor="#e5e7eb", zerolinecolor="#9ca3af",
                           tickfont=dict(size=12)),
                yaxis=dict(gridcolor="#e5e7eb", tickfont=dict(size=12)),
                font=dict(family="Inter,sans-serif", size=13),
            )
            st.plotly_chart(_fig_sec, use_container_width=True)

        render_layer_tables([
            ("Sector Scores", sectors, ["ticker", "theme", "cycle_signal", "ret_5d", "ret_20d", "ret_63d", "relative_20d_vs_spy", "rotation_score", "rotation_label"], 460),
            ("Theme Heatmap", theme, ["ticker", "theme", "ret_5d", "ret_20d", "ret_63d", "relative_20d_vs_spy", "relative_63d_vs_spy", "rotation_score", "rotation_label"], 360),
            ("Raw Report", pd.DataFrame([{"status": "REPORT", "content": read_md(FILES["sector_report"])}]), ["status", "content"], 520),
        ])
    with sub[2]:
        fundamentals = read_csv(FILES["fundamentals"])
        flags = read_csv(FILES["valuation_flags"])
        render_layer_workbench_header(
            "L4",
            "Basics And Valuation",
            "Separate ETF context from single-company quality, valuation, and balance sheet data.",
            [
                ("Rows", len(fundamentals), "cyan"),
                ("ETF Context", count_value(fundamentals, "fundamental_label", "ETF_NOT_FUNDAMENTAL"), "wait"),
                ("Better Quality", count_value(fundamentals, "fundamental_label", "QUALITY_HOLD_CANDIDATE"), "supportive"),
                ("No Data", count_value(fundamentals, "fundamental_label", "NO_DATA"), "blocked"),
            ],
        )
        _l4_etf_count = count_value(fundamentals, "fundamental_label", "ETF_NOT_FUNDAMENTAL")
        _l4_quality = count_value(fundamentals, "fundamental_label", "QUALITY_HOLD_CANDIDATE")
        _l4_val_flags = len(flags) if not flags.empty else 0
        st.subheader("What Basics And Valuation Say Today")
        render_badge_table(pd.DataFrame([
            {
                "status": "wait" if _l4_etf_count > 0 else "REVIEW",
                "layer_question": "Are most tickers ETFs or individual companies here?",
                "today_read": f"{_l4_etf_count} ETF context rows (no fundamental data); {_l4_quality} quality hold candidates.",
                "what_it_means_for_tickers": "ETF rows use ETF_NOT_FUNDAMENTAL because ETFs do not have PE ratios or earnings. The decision for ETFs rests on macro, sector, and technical layers instead.",
                "when_this_layer_blocks_action": "L4 does not directly block ETFs — it marks them as context rows so you do not mistake missing fundamentals for a red flag.",
            },
            {
                "status": "REVIEW" if _l4_val_flags > 0 else "OK",
                "layer_question": "Are there any valuation risk flags?",
                "today_read": f"{_l4_val_flags} valuation flag rows. Quality hold candidates: {_l4_quality}.",
                "what_it_means_for_tickers": "Valuation flags (high PE, high debt, negative margins) mean the business is priced for perfection. Any miss can cause a sharp drop.",
                "when_this_layer_blocks_action": "A ticker with multiple valuation flags should stay research-only even if technical timing looks good.",
            },
        ]), height=200)
        # ── L4 Valuation Signal: Cheap / Fair / Expensive ─────────────────
        if not fundamentals.empty:
            def _val_signal(row):
                lbl = str(row.get("fundamental_label", ""))
                if "ETF_NOT_FUNDAMENTAL" in lbl:
                    return "ETF Context"
                fpe = pd.to_numeric(row.get("forward_pe", None), errors="coerce")
                tpe = pd.to_numeric(row.get("trailing_pe", None), errors="coerce")
                pe = fpe if pd.notna(fpe) else tpe
                if pd.isna(pe):
                    return "No PE Data"
                if pe < 15:
                    return "Cheap"
                if pe <= 28:
                    return "Fair"
                return "Expensive"
            fundamentals = fundamentals.copy()
            fundamentals["val_signal"] = fundamentals.apply(_val_signal, axis=1)
        render_layer_tables([
            ("Fundamental Matrix", fundamentals, ["ticker", "asset_type", "val_signal", "forward_pe", "trailing_pe", "peg_ratio", "revenue_growth", "gross_margin", "profit_margin", "debt_to_equity", "quality_score", "fundamental_label"], 520),
            ("Valuation Flags", flags, None, 320),
            ("Raw Report", pd.DataFrame([{"status": "REPORT", "content": read_md(FILES["fund_report"])}]), ["status", "content"], 520),
        ])
    with sub[3]:
        events = read_csv(FILES["events"])
        news = read_csv(FILES["news_risk"])
        earnings = read_csv(FILES["earnings_check"])
        insider = read_csv(FILES["insider_signals"])
        render_layer_workbench_header(
            "L5",
            "Events And News",
            "Check whether news, earnings, filings, or insider activity could change the decision.",
            [
                ("Evidence Rows", len(events), "cyan"),
                ("Event Risk", count_value(events, "event_label", "EVENT_RISK"), "risk"),
                ("News Errors", count_value(news, "news_status", "ERROR"), "risk"),
                ("ETF Context", count_contains(earnings, "earnings_status", "ETF"), "wait"),
            ],
        )
        _l5_event_risk = count_value(events, "event_label", "EVENT_RISK")
        _l5_news_error = count_value(news, "news_status", "ERROR")
        _l5_insider_sell = count_contains(insider, "net_direction", "SELL") if not insider.empty else 0
        _l5_earnings_soon = count_contains(earnings, "earnings_status", "NEAR") + count_contains(earnings, "earnings_status", "UPCOMING") if not earnings.empty else 0
        _l5_status = "RISK" if (_l5_event_risk > 0 or _l5_insider_sell > 0) else ("REVIEW" if _l5_earnings_soon > 0 else "OK")
        st.subheader("What Events And News Say Today")
        render_badge_table(pd.DataFrame([
            {
                "status": _l5_status,
                "layer_question": "Is there any fresh event risk right now?",
                "today_read": f"Event risk flags: {_l5_event_risk}; insider sell signals: {_l5_insider_sell}; earnings soon: {_l5_earnings_soon}; news errors: {_l5_news_error}.",
                "what_it_means_for_tickers": "Event risk means a binary outcome is approaching. Even a strong technical setup can be wiped out by a bad earnings report or regulatory action.",
                "when_this_layer_blocks_action": "Any ticker with EVENT_RISK label should stay research-only until the event passes and the outcome is absorbed.",
            },
            {
                "status": "REVIEW" if _l5_news_error > 0 else "OK",
                "layer_question": "Is the news data reliable?",
                "today_read": f"News fetch errors: {_l5_news_error}. If errors are high, news risk cannot be confirmed — treat as manual check required.",
                "what_it_means_for_tickers": "News errors mean the system could not pull current headlines. This lowers confidence in the L5 result for affected tickers.",
                "when_this_layer_blocks_action": "When news is unavailable (error), mark that ticker as needing manual news check before any action is considered.",
            },
        ]), height=200)
        render_layer_tables([
            ("Evidence Cards", events, ["ticker", "event_score", "event_label", "reasons"], 360),
            ("News", news, ["ticker", "news_status", "title", "publisher", "risk_label", "notes"], 360),
            ("Earnings", earnings, ["ticker", "asset_type", "earnings_date", "earnings_status", "notes"], 340),
            ("Insider", insider, ["ticker", "insider_status", "recent_transactions", "net_direction", "notes"], 340),
            ("Raw Report", pd.DataFrame([{"status": "REPORT", "content": read_md(FILES["event_report"])}]), ["status", "content"], 520),
        ])
    with sub[4]:
        technicals = read_csv(FILES["technicals"])
        liquidity = read_csv(FILES["liquidity_proxy"])
        tactical = read_csv(FILES["tactical_candidates"])
        breakout = read_csv(FILES["breakout_watchlist"])
        render_layer_workbench_header(
            "L6",
            "Price Trend",
            "After data is trusted, check whether trend, trading activity, and trigger levels support the idea.",
            [
                ("Rows", len(technicals), "cyan"),
                ("Watchable", count_value(technicals, "technical_label", "TACTICAL_CANDIDATE"), "supportive"),
                ("No Edge", count_value(technicals, "technical_label", "NO_TECH_EDGE"), "weak"),
                ("No Price", count_value(technicals, "data_status", "NO_PRICE"), "blocked"),
            ],
        )
        _l6_tactical = count_value(technicals, "technical_label", "TACTICAL_CANDIDATE")
        _l6_no_edge = count_value(technicals, "technical_label", "NO_TECH_EDGE")
        _l6_no_price = count_value(technicals, "data_status", "NO_PRICE")
        _l6_breakouts = len(breakout) if not breakout.empty else 0
        _l6_liquidity_ok = count_value(liquidity, "liquidity_label", "LIQUID") if not liquidity.empty else 0
        _l6_status = "supportive" if _l6_tactical > 0 else ("blocked" if _l6_no_price > 0 else "REVIEW")
        st.subheader("What Price Trend Says Today")
        render_badge_table(pd.DataFrame([
            {
                "status": _l6_status,
                "layer_question": "How many tickers have a real technical edge?",
                "today_read": f"Tactical candidates: {_l6_tactical}; no-edge names: {_l6_no_edge}; no-price names: {_l6_no_price}; breakout watchlist rows: {_l6_breakouts}.",
                "what_it_means_for_tickers": "Tactical candidate means trend, momentum, and trigger level are aligned. No-edge means the price is not telling a clear story — wait for confirmation.",
                "when_this_layer_blocks_action": "L6 does not hard-block, but a no-edge or no-price ticker cannot be used for timing. Any entry without L6 confirmation is a blind guess.",
            },
            {
                "status": "REVIEW" if _l6_liquidity_ok == 0 else "OK",
                "layer_question": "Is liquidity adequate for the tickers being watched?",
                "today_read": f"Liquid names: {_l6_liquidity_ok} of {len(liquidity) if not liquidity.empty else 0}. Low liquidity means wide spreads and difficulty exiting.",
                "what_it_means_for_tickers": "Even a strong technical setup in an illiquid name is dangerous — the entry and exit spreads can absorb most of the edge.",
                "when_this_layer_blocks_action": "If a ticker is marked as illiquid or has low dollar volume, it should stay research-only regardless of trend quality.",
            },
        ]), height=200)
        render_layer_tables([
            ("Technical Matrix", technicals, ["ticker", "data_status", "close", "ret_5d", "ret_20d", "ret_63d", "rsi14", "atr14_pct", "volume_z60", "technical_score", "technical_label", "reasons"], 520),
            ("Liquidity", liquidity, ["ticker", "avg_20d_dollar_volume", "median_20d_volume", "liquidity_label", "notes"], 340),
            ("Tactical Candidates", tactical, None, 300),
            ("Breakout Watchlist", breakout, None, 300),
            ("Raw Report", pd.DataFrame([{"status": "REPORT", "content": read_md(FILES["tech_report"])}]), ["status", "content"], 520),
        ])

    # ── L7 : Options ──────────────────────────────────────────────────────────
    with sub[5]:
        _l7_opts = read_csv(FILES["options_decision"])
        _l7_syn  = read_csv(FILES["v8_synthetic_options"])
        _l7_gam  = read_csv(FILES["gamma_candidates"])
        _l7_kill = read_csv(FILES["kill_zone"])
        render_layer_workbench_header(
            "L7", "Options Heat And Danger Zone",
            "Options help judge pressure and timing. Risk and action checks still come first.",
            [
                ("Rows",            len(_l7_opts), "cyan"),
                ("Research Only",   count_value(_l7_opts, "final_options_decision", "RESEARCH_ONLY"), "cyan"),
                ("Paper Test",      count_value(_l7_opts, "final_options_decision", "PAPER_ONLY"),    "paper"),
                ("Live Allowed",    count_value(_l7_opts, "live_allowed", "YES"), "risk" if count_value(_l7_opts, "live_allowed", "YES") else "blocked"),
            ],
        )
        render_layer_tables([
            ("Options Decision",   _l7_opts, ["ticker","spot","gamma_squeeze_label","option_kill_zone_label","pretrade_status","portfolio_risk_light","paper_allowed","live_allowed","final_options_decision","rule","explanation"], 460),
            ("V8 Synthetic Overlay", _l7_syn, ["ticker","overlay_status","decision_use","spot","combined_signal","squeeze_score","squeeze_risk","gamma_flip","gex_regime","max_pain","dealer_flow"], 360),
            ("Gamma Candidates",   _l7_gam, None, 280),
            ("Kill Zone",          _l7_kill, None, 280),
            ("Raw Options Report", pd.DataFrame([{"status":"REPORT","content":read_md(FILES["options_report"])}]), ["status","content"], 520),
            ("Raw Kill Zone",      pd.DataFrame([{"status":"REPORT","content":read_md(FILES["kill_zone_report"])}]), ["status","content"], 520),
        ])

    # ── L8 : Portfolio Risk ────────────────────────────────────────────────────
    with sub[6]:
        _l8_exp  = read_csv(FILES["exposure"])
        _l8_warn = read_csv(FILES["exposure_warnings"])
        _l8_str  = read_csv(FILES["scenario_stress"])
        _l8_siz  = read_csv(FILES["position_sizing"])
        _l8_adv  = read_csv(FILES["v8_adv_risk"])
        _l8_hw   = count_value(_l8_warn, "level", "HIGH")
        _l8_mw   = count_value(_l8_warn, "level", "MEDIUM")
        render_layer_workbench_header(
            "L8", "Portfolio Risk And Size",
            "Check concentration, stress-test losses, and whether sizing should shrink.",
            [
                ("Risk Light",       "RED" if _l8_hw else ("WARN" if _l8_mw else "OK"), status_kind("RED" if _l8_hw else ("WARN" if _l8_mw else "OK"))),
                ("High Warnings",    _l8_hw, "risk"),
                ("Stress Rows",      len(_l8_str), "cyan"),
                ("Advanced Flags",   count_value(_l8_adv, "status", "RISK"), "risk" if count_value(_l8_adv, "status", "RISK") else "supportive"),
            ],
        )
        render_layer_tables([
            ("Exposure",          _l8_exp, ["ticker","asset_type","sector","theme","risk_bucket","effective_weight"], 360),
            ("Warnings",          _l8_warn, ["level","issue","detail","action"], 360),
            ("Stress",            _l8_str, ["scenario","estimated_pnl","estimated_loss","breaches_1pct","breaches_2pct","breaches_5pct"], 340),
            ("Sizing",            _l8_siz, ["ticker","sleeve","decision","risk_bucket","effective_weight","suggested_weight","reduction_from_effective","suggested_action","sizing_reason"], 420),
            ("Advanced Risk",     _l8_adv, ["status","data_source","metric","value","interpretation"], 360),
            ("Raw Stress Report", pd.DataFrame([{"status":"REPORT","content":read_md(FILES["stress_report"])}]),   ["status","content"], 520),
            ("Raw Exposure",      pd.DataFrame([{"status":"REPORT","content":read_md(FILES["exposure_report"])}]), ["status","content"], 520),
        ])

    # ── L9 : Before-Action Check ───────────────────────────────────────────────
    with sub[7]:
        _l9_pre   = read_csv(FILES["pre_trade"])
        _l9_gate  = read_csv(FILES["v8_l9_gate"])
        _l9_order = read_csv(FILES["pre_trade_order"])
        _l9_ledg  = read_csv(FILES["paper_ledger"])
        _l9_pend  = count_value(_l9_pre, "final_status", "PENDING_MANUAL_CHECKS")
        _l9_blk   = count_value(_l9_pre, "final_status", "BLOCKED")
        _l9_live  = count_value(_l9_pre, "live_allowed", "YES") + count_value(_l9_gate, "live_allowed", "YES")
        render_layer_workbench_header(
            "L9", "Before-Action Check And Paper Log",
            "Translate research into allowed and forbidden actions. Paper only — no live orders.",
            [
                ("Needs Check",      _l9_pend, "cyan"),
                ("Blocked",          _l9_blk,  "blocked"),
                ("Old Check Rows",   len(_l9_gate), "wait"),
                ("Live Allowed",     _l9_live, "risk" if _l9_live else "blocked"),
            ],
        )
        render_layer_tables([
            ("Pre-Trade Gate", _l9_pre, ["ticker","decision","risk_light","manual_news_check","earnings_date_check","liquidity_check","spread_check","duplicate_exposure_check","stress_check","paper_allowed","live_allowed","final_status","reasons"], 500),
            ("V8 L9 Bridge",  _l9_gate, ["ticker","decision","risk_light","suggested_action","paper_allowed","live_allowed","final_status","reasons","sizing_reason"], 440),
            ("Order Ticket",  _l9_order, None, 260),
            ("Paper Ledger",  _l9_ledg, ["trade_id","ticker","sleeve","decision","risk_bucket","status","entry_price","exit_price","pnl_pct","holding_days","thesis","notes"], 420),
            ("Raw Pre-Trade", pd.DataFrame([{"status":"REPORT","content":read_md(ROOT / "pre_trade_checklist.md")}]), ["status","content"], 520),
        ])

    # ── L10 : Review Learning ──────────────────────────────────────────────────
    with sub[8]:
        _l10_ledg = read_csv(FILES["paper_ledger"])
        _l10_sum  = read_csv(FILES["learning_summary"])
        _l10_sug  = read_csv(FILES["learning_suggestions"])
        _l10_cls  = count_contains(_l10_ledg, "status", "CLOSED")
        _l10_noadj = count_value(_l10_sug, "suggestion", "NO_ADJUST")
        render_layer_workbench_header(
            "L10", "Review Learning",
            "Record paper-test results. Do not change strategy until there are at least 30 closed samples.",
            [
                ("Closed Samples",  _l10_cls,   "supportive" if _l10_cls >= 30 else "cyan"),
                ("Log Rows",        len(_l10_ledg), "wait"),
                ("Review Rows",     len(_l10_sum), "cyan"),
                ("No Adjustment",   _l10_noadj, "blocked"),
            ],
        )
        render_layer_tables([
            ("Learning Summary",   _l10_sum, ["level","key","trades","avg_pnl","median_pnl","win_rate","total_weighted_contribution"], 420),
            ("Weight Suggestions", _l10_sug, ["level","key","trades","avg_pnl","win_rate","suggestion","reason"], 420),
            ("Paper Ledger",       _l10_ledg, ["trade_id","ticker","sleeve","decision","risk_bucket","status","entry_price","exit_price","pnl_pct","holding_days","thesis","notes"], 460),
            ("Raw Learning",       pd.DataFrame([{"status":"REPORT","content":read_md(FILES["learning_report"])}]), ["status","content"], 520),
        ])


def tab_l7_l10():
    st.header("Last Four Layers: Options, Risk, Action, Review")
    sub = st.tabs(["Layer 7: Options", "Layer 8: Risk", "Layer 9: Action Check", "Layer 10: Review"])

    with sub[0]:
        options = read_csv(FILES["options_decision"])
        synthetic = read_csv(FILES["v8_synthetic_options"])
        gamma = read_csv(FILES["gamma_candidates"])
        kill = read_csv(FILES["kill_zone"])
        render_layer_workbench_header(
            "L7",
            "Options Heat And Danger Zone",
            "Options help judge pressure and timing, but risk and action checks still come first.",
            [
                ("Rows", len(options), "cyan"),
                ("Research Only", count_value(options, "final_options_decision", "RESEARCH_ONLY"), "cyan"),
                ("Paper Test", count_value(options, "final_options_decision", "PAPER_ONLY"), "paper"),
                ("Live Orders Allowed", count_value(options, "live_allowed", "YES"), "risk" if count_value(options, "live_allowed", "YES") else "blocked"),
            ],
        )
        render_layer_tables([
            ("Options Decision", options, ["ticker", "spot", "gamma_squeeze_label", "option_kill_zone_label", "pretrade_status", "portfolio_risk_light", "paper_allowed", "live_allowed", "final_options_decision", "rule", "explanation"], 460),
            ("V8 Synthetic Overlay", synthetic, ["ticker", "overlay_status", "decision_use", "spot", "combined_signal", "squeeze_score", "squeeze_risk", "gamma_flip", "gex_regime", "max_pain", "dealer_flow"], 360),
            ("Gamma Candidates", gamma, None, 280),
            ("Kill Zone", kill, None, 280),
            ("Raw Options Report", pd.DataFrame([{"status": "REPORT", "content": read_md(FILES["options_report"])}]), ["status", "content"], 520),
            ("Raw Kill Zone Report", pd.DataFrame([{"status": "REPORT", "content": read_md(FILES["kill_zone_report"])}]), ["status", "content"], 520),
        ])

    with sub[1]:
        exposure = read_csv(FILES["exposure"])
        warnings = read_csv(FILES["exposure_warnings"])
        stress = read_csv(FILES["scenario_stress"])
        sizing = read_csv(FILES["position_sizing"])
        adv = read_csv(FILES["v8_adv_risk"])
        high_warnings = count_value(warnings, "level", "HIGH")
        medium_warnings = count_value(warnings, "level", "MEDIUM")
        risk_metric = "RED" if high_warnings else ("WARN" if medium_warnings else "OK")
        render_layer_workbench_header(
            "L8",
            "Portfolio Risk And Size",
            "Check whether positions are too concentrated, how much stress tests could lose, and whether size should shrink.",
            [
                ("Risk Light", friendly_value(risk_metric), status_kind(risk_metric)),
                ("High Risk Warnings", high_warnings, "risk"),
                ("Stress Test Rows", len(stress), "cyan"),
                ("More Risk Checks", count_value(adv, "status", "RISK"), "risk" if count_value(adv, "status", "RISK") else "supportive"),
            ],
        )
        if _PLOTLY and not stress.empty and "scenario" in stress.columns and "estimated_loss" in stress.columns:
            _l8_st = stress.copy()
            _l8_st["estimated_loss"] = pd.to_numeric(_l8_st["estimated_loss"], errors="coerce").fillna(0)
            _l8_st = _l8_st.sort_values("estimated_loss", ascending=True)
            _l8_sc = ["#b91c1c" if v <= -0.05 else "#f87171" if v < 0 else "#22d3ee" for v in _l8_st["estimated_loss"]]
            _fig_l8 = go.Figure(go.Bar(
                x=(_l8_st["estimated_loss"] * 100).tolist(),
                y=_l8_st["scenario"].astype(str).tolist(),
                orientation="h", marker_color=_l8_sc,
                text=(_l8_st["estimated_loss"] * 100).map(lambda x: f"{x:.2f}%").tolist(),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>%{x:.2f}%<extra></extra>",
            ))
            _fig_l8.update_layout(
                height=max(140, len(_l8_st) * 34 + 50),
                margin=dict(l=10, r=60, t=10, b=10),
                xaxis_title="Estimated Loss %", yaxis_title="",
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(gridcolor="#e5e7eb", zeroline=True, zerolinecolor="#111827", zerolinewidth=1, ticksuffix="%"),
                yaxis=dict(gridcolor="#e5e7eb"),
                font=dict(family="Inter,sans-serif", size=12),
            )
            st.plotly_chart(_fig_l8, use_container_width=True)
        render_layer_tables([
            ("Exposure", exposure, ["ticker", "asset_type", "sector", "theme", "risk_bucket", "effective_weight"], 360),
            ("Warnings", warnings, ["level", "issue", "detail", "action"], 360),
            ("Stress", stress, ["scenario", "estimated_pnl", "estimated_loss", "breaches_1pct", "breaches_2pct", "breaches_5pct"], 340),
            ("Sizing", sizing, ["ticker", "sleeve", "decision", "risk_bucket", "effective_weight", "suggested_weight", "reduction_from_effective", "suggested_action", "sizing_reason"], 420),
            ("Advanced Risk", adv, ["status", "data_source", "metric", "value", "interpretation"], 360),
            ("Raw Stress Report", pd.DataFrame([{"status": "REPORT", "content": read_md(FILES["stress_report"])}]), ["status", "content"], 520),
            ("Raw Exposure Report", pd.DataFrame([{"status": "REPORT", "content": read_md(FILES["exposure_report"])}]), ["status", "content"], 520),
            ("Raw Advanced Risk", pd.DataFrame([{"status": "REPORT", "content": read_md(FILES["v8_adv_risk_report"])}]), ["status", "content"], 520),
        ])

    with sub[2]:
        pretrade = read_csv(FILES["pre_trade"])
        v8_gate = read_csv(FILES["v8_l9_gate"])
        order_ticket = read_csv(FILES["pre_trade_order"])
        ledger = read_csv(FILES["paper_ledger"])
        pending = count_value(pretrade, "final_status", "PENDING_MANUAL_CHECKS")
        blocked = count_value(pretrade, "final_status", "BLOCKED")
        live_yes = count_value(pretrade, "live_allowed", "YES") + count_value(v8_gate, "live_allowed", "YES")
        render_layer_workbench_header(
            "L9",
            "Before-Action Check And Paper Log",
            "Translate research into allowed and forbidden actions while keeping the paper-test-only safety rule.",
            [
                ("Needs Check", pending, "cyan"),
                ("Blocked", blocked, "blocked"),
                ("Old Check Rows", len(v8_gate), "wait"),
                ("Live Orders Allowed", live_yes, "risk" if live_yes else "blocked"),
            ],
        )
        render_layer_tables([
            ("Pre-Trade Gate", pretrade, ["ticker", "decision", "risk_light", "manual_news_check", "earnings_date_check", "liquidity_check", "spread_check", "duplicate_exposure_check", "stress_check", "paper_allowed", "live_allowed", "final_status", "reasons"], 500),
            ("V8 L9 Bridge", v8_gate, ["ticker", "decision", "risk_light", "suggested_action", "paper_allowed", "live_allowed", "final_status", "reasons", "sizing_reason"], 440),
            ("Order Ticket", order_ticket, None, 260),
            ("Paper Ledger", ledger, ["trade_id", "ticker", "side", "sleeve", "status", "entry_date", "entry_price", "exit_date", "exit_price", "pnl_pct", "thesis", "risk_note"], 420),
            ("Raw Pre-Trade Report", pd.DataFrame([{"status": "REPORT", "content": read_md(ROOT / "pre_trade_checklist.md")}]), ["status", "content"], 520),
            ("Raw V8 L9 Report", pd.DataFrame([{"status": "REPORT", "content": read_md(FILES["v8_l9_report"])}]), ["status", "content"], 520),
        ])

    with sub[3]:
        ledger = read_csv(FILES["paper_ledger"])
        summary = read_csv(FILES["learning_summary"])
        suggestions = read_csv(FILES["learning_suggestions"])
        closed = count_contains(ledger, "status", "CLOSED")
        no_adjust = count_value(suggestions, "suggestion", "NO_ADJUST")
        render_layer_workbench_header(
            "L10",
            "Review Learning",
            "Record paper-test results, but do not change strategy automatically before there are enough samples.",
            [
                ("Closed Samples", closed, "supportive" if closed >= 30 else "cyan"),
                ("Log Rows", len(ledger), "wait"),
                ("Review Rows", len(summary), "cyan"),
                ("No Adjustment", no_adjust, "blocked"),
            ],
        )
        render_layer_tables([
            ("Learning Summary", summary, ["level", "key", "trades", "avg_pnl", "median_pnl", "win_rate", "total_weighted_contribution"], 420),
            ("Weight Suggestions", suggestions, ["level", "key", "trades", "avg_pnl", "win_rate", "suggestion", "reason"], 420),
            ("Paper Ledger", ledger, ["trade_id", "ticker", "sleeve", "decision", "risk_bucket", "status", "entry_price", "exit_price", "pnl_pct", "holding_days", "thesis", "notes"], 460),
            ("Raw Learning Report", pd.DataFrame([{"status": "REPORT", "content": read_md(FILES["learning_report"])}]), ["status", "content"], 520),
        ])


def tab_action():
    st.header("Action Board")
    st.caption("This shows allowed actions, forbidden actions, trigger levels, and checks that still have not passed.")

    board = build_action_board_table()
    triggers = build_trigger_board(read_csv(FILES["watch_triggers"]), read_csv(FILES["action_cards"]))

    if board.empty:
        st.warning("No action card data found.")
        return

    decision_counts = board["decision"].value_counts() if "decision" in board.columns else pd.Series(dtype=int)
    master_counts = board["master_action"].value_counts() if "master_action" in board.columns else pd.Series(dtype=int)
    live_blocked = count_value(board, "live_allowed_effective", "NO")
    pending = count_value(board, "final_status", "PENDING_MANUAL_CHECKS")

    render_layer_workbench_header(
        "Action",
        "Decision, Trigger Levels, And Permission",
        "Translate the 10-layer result into allowed action, forbidden action, and what to check next.",
        [
            ("Research Only", int(decision_counts.get("RESEARCH_ONLY", 0)), "cyan"),
            ("Reduce Risk First", int(master_counts.get("RISK_REDUCTION_FIRST", 0)), "risk"),
            ("Needs Check", pending, "cyan"),
            ("Live Orders Blocked", live_blocked, "blocked"),
        ],
    )

    left, mid, right = st.columns(3)
    with left:
        decision_filter = st.selectbox("Decision", ["All"] + sorted(board["decision"].dropna().astype(str).unique().tolist()) if "decision" in board.columns else ["All"])
    with mid:
        urgency_filter = st.selectbox("Urgency", ["All"] + sorted(board["urgency"].dropna().astype(str).unique().tolist()) if "urgency" in board.columns else ["All"])
    with right:
        live_filter = st.selectbox("Live Orders Allowed", ["All"] + sorted(board["live_allowed_effective"].dropna().astype(str).unique().tolist()) if "live_allowed_effective" in board.columns else ["All"])

    view = board.copy()
    if str(decision_filter) != "All" and "decision" in view.columns:
        view = view[view["decision"].astype(str) == str(decision_filter)]
    if str(urgency_filter) != "All" and "urgency" in view.columns:
        view = view[view["urgency"].astype(str) == str(urgency_filter)]
    if str(live_filter) != "All" and "live_allowed_effective" in view.columns:
        view = view[view["live_allowed_effective"].astype(str) == str(live_filter)]

    tabs = st.tabs(["Main Table", "Trigger Watch", "Allowed And Forbidden", "Human Checks", "Full Action Cards"])

    with tabs[0]:
        cols = [c for c in [
            "ticker", "master_action", "decision", "urgency", "one_liner",
            "allowed_action", "forbidden_action", "trigger_rule", "L8_state",
            "L9_state", "final_status", "final_options_decision", "live_allowed_effective",
            "master_reason"
        ] if c in view.columns]
        render_badge_table(view[cols], height=560)

        if _PLOTLY and not view.empty and "master_action" in view.columns:
            _ma_counts  = view["master_action"].value_counts()
            _ma_labels  = _ma_counts.index.tolist()
            _ma_vals    = _ma_counts.values.tolist()
            _ma_colors  = [
                "#b91c1c" if "RISK" in str(v).upper()
                else "#a855f7" if "PAPER" in str(v).upper()
                else "#22d3ee" if "RESEARCH" in str(v).upper()
                else "#9ca3af"
                for v in _ma_labels
            ]
            _fig_action = go.Figure(go.Bar(
                x=_ma_vals, y=_ma_labels, orientation="h",
                marker_color=_ma_colors,
                text=[str(v) for v in _ma_vals], textposition="outside",
                hovertemplate="<b>%{y}</b><br>%{x} ticker(s)<extra></extra>",
            ))
            _fig_action.update_layout(
                height=max(120, len(_ma_labels) * 38 + 60),
                margin=dict(l=10, r=60, t=10, b=20),
                xaxis_title="Number of tickers", yaxis_title="",
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(gridcolor="#e5e7eb"),
                yaxis=dict(gridcolor="#e5e7eb", autorange="reversed"),
                font=dict(family="Inter,sans-serif", size=13),
            )
            st.caption("Decision distribution for the current filter.")
            st.plotly_chart(_fig_action, use_container_width=True)

        if not view.empty and "ticker" in view.columns:
            st.subheader("Open One Action Explanation")
            action_tickers = view["ticker"].dropna().astype(str).drop_duplicates().tolist()
            action_ticker = st.selectbox(
                "Choose Ticker To Explain",
                action_tickers,
                key="action_board_explain_ticker",
            )
            action_row = first_row(view, action_ticker)
            master_for_action = read_csv(FILES["master_v2"])
            master_action_row = first_row(master_for_action, action_ticker)
            action_evidence = pd.DataFrame([
                {
                    "status": row_value(master_action_row, "master_action", action_row.get("master_action", ""), default="NO_DATA"),
                    "check": "Final Decision",
                    "what_it_means": row_value(master_action_row, "master_reason", action_row.get("master_reason", ""), action_row.get("one_liner", ""), default="No final reason yet."),
                    "source_file": "master_10_layer_decision_matrix_v2.csv; action_cards.csv",
                },
                {
                    "status": row_value(master_action_row, "L8_state", action_row.get("L8_state", ""), default="NO_DATA"),
                    "check": "Portfolio Risk",
                    "what_it_means": row_value(master_action_row, "L8_note", master_action_row.get("master_reason", ""), default="Risk layer decides how much action is allowed."),
                    "source_file": "stress_position_sizing_report.md; exposure_warnings.csv; master_10_layer_decision_matrix_v2.csv",
                },
                {
                    "status": row_value(action_row, "final_status", action_row.get("L9_state", ""), default="NO_DATA"),
                    "check": "Before-Action Check",
                    "what_it_means": row_value(action_row, "pretrade_reasons", action_row.get("explanation", ""), default="Manual checks decide whether this can move beyond research."),
                    "source_file": "pre_trade_checklist.csv; v8_l9_execution_gate.csv",
                },
                {
                    "status": row_value(action_row, "final_options_decision", default="NO_DATA"),
                    "check": "Options Context",
                    "what_it_means": row_value(action_row, "explanation", default="Options are context only and cannot override portfolio risk."),
                    "source_file": "options_decision_matrix.csv; gamma_squeeze_candidates.csv; option_kill_zone_risk.csv",
                },
                {
                    "status": row_value(action_row, "live_allowed_effective", default="NO"),
                    "check": "Hard Safety Rule",
                    "what_it_means": "No broker connection. No live order path is enabled. Paper/research only.",
                    "source_file": "pre_trade_checklist.csv; action_cards.md",
                },
            ])
            render_badge_table(action_evidence, height=340)
            with st.expander("Source Trail For This Action"):
                st.markdown(f"**Allowed:** {row_value(action_row, 'allowed_action', default='Research only.')}")
                st.markdown(f"**Forbidden:** {row_value(action_row, 'forbidden_action', default='Live order or forced trade.')}")
                st.markdown(f"**Trigger rule:** {row_value(action_row, 'trigger_rule', default='No trigger rule yet.')}")
                st.code(
                    "action_cards.csv\n"
                    "action_cards.md\n"
                    "master_10_layer_decision_matrix_v2.csv\n"
                    "pre_trade_checklist.csv\n"
                    "watch_triggers.csv\n"
                    "options_decision_matrix.csv",
                    language="text",
                )

    with tabs[1]:
        if triggers.empty:
            st.info("No trigger data yet.")
        else:
            trigger_cols = [c for c in [
                "ticker", "decision", "urgency", "spot", "trigger_status",
                "nearest_trigger_distance_pct", "breakout_trigger", "breakdown_trigger",
                "allowed_action", "forbidden_action", "trigger_rule", "live_allowed"
            ] if c in triggers.columns]
            trigger_view = triggers[trigger_cols].copy()
            if "nearest_trigger_distance_pct" in trigger_view.columns:
                trigger_view["nearest_trigger_distance_pct"] = pd.to_numeric(trigger_view["nearest_trigger_distance_pct"], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.2f}%")
            render_badge_table(trigger_view, height=520)

    with tabs[2]:
        permission_cols = [c for c in [
            "ticker", "decision", "allowed_action", "forbidden_action",
            "paper_allowed", "pretrade_paper_allowed", "live_allowed_effective",
            "risk_note", "explanation"
        ] if c in view.columns]
        render_badge_table(view[permission_cols], height=520)

    with tabs[3]:
        gate_cols = [c for c in [
            "ticker", "final_status", "manual_news_check", "earnings_date_check",
            "liquidity_check", "spread_check", "duplicate_exposure_check", "stress_check",
            "pretrade_reasons"
        ] if c in view.columns]
        render_badge_table(view[gate_cols], height=520)

    with tabs[4]:
        raw_cards = read_csv(FILES["action_cards"])
        render_badge_table(raw_cards, height=520)
        with st.expander("Raw action cards markdown"):
            st.markdown(read_md(ROOT / "action_cards.md"))


def tab_architecture():
    st.header("Blueprint")

    audit = read_csv(FILES["layer_audit"])
    master = read_csv(FILES["master_v2"])
    reports = build_report_archive_index()

    layer_count = len(audit)
    avg = pd.to_numeric(audit.get("maturity_score_0_5", pd.Series(dtype=str)), errors="coerce").mean() if not audit.empty else 0
    operational = count_value(audit, "maturity_status", "Operational")
    report_found = count_value(reports, "status", "FOUND")

    render_layer_workbench_header(
        "Blueprint",
        "Canyon v9 Investment System Architecture",
        "The system is not an options dashboard. Options are L7 inside a broader 10-layer research, risk, execution, and learning stack.",
        [
            ("Layers", layer_count, "cyan"),
            ("Maturity", f"{avg:.1f}/5", "supportive" if avg >= 4 else "cyan"),
            ("Operational", operational, "supportive"),
            ("Reports Found", report_found, "wait"),
        ],
    )

    tabs = st.tabs(["Core Idea", "10-Layer Map", "Safety Rules", "Build Roadmap", "Full Structure"])

    with tabs[0]:
        thesis = pd.DataFrame([
            {
                "status": "OK",
                "principle": "Not an options screener",
                "meaning": "Options are only L7; they cannot override data integrity, portfolio risk, execution gates, or learning sample limits.",
            },
            {
                "status": "OK",
                "principle": "Top-down to ticker",
                "meaning": "Macro and sector context should frame ticker decisions before technical or option pressure is considered.",
            },
            {
                "status": "RED",
                "principle": "Risk first",
                "meaning": "When L8 is RED, the system blocks aggressive action and caps expression to research or tiny paper.",
            },
            {
                "status": "NO",
                "principle": "No broker / no live order",
                "meaning": "The dashboard is research and paper-only. Live execution modules stay blocked even when imported from v8 source.",
            },
            {
                "status": "LEARNING_SAMPLE_PENDING",
                "principle": "Conservative learning",
                "meaning": "L10 records paper outcomes but does not auto-adjust weights until there are enough clean closed samples.",
            },
        ])
        render_badge_table(thesis, height=300)

    with tabs[1]:
        if audit.empty:
            st.info("No layer audit available. Generate `canyon_layer_status_audit.csv` by running the full daily runner.")
            st.code("python3 -u canyon_final_v9_step56_full_10_layer_daily_runner_v2.py", language="bash")
        else:
            if _PLOTLY and "layer_name" in audit.columns and "maturity_score_0_5" in audit.columns:
                _arc_df = audit.copy()
                _arc_df["maturity_score_0_5"] = pd.to_numeric(_arc_df["maturity_score_0_5"], errors="coerce").fillna(0)
                _arc_colors = [
                    "#16a34a" if v >= 4.5 else "#22d3ee" if v >= 3 else "#f87171" if v >= 1 else "#9ca3af"
                    for v in _arc_df["maturity_score_0_5"]
                ]
                _fig_arc = go.Figure(go.Bar(
                    x=_arc_df["maturity_score_0_5"].tolist(),
                    y=_arc_df["layer_name"].astype(str).tolist(),
                    orientation="h",
                    marker_color=_arc_colors,
                    text=_arc_df["maturity_score_0_5"].map(lambda x: f"{x:.1f}/5").tolist(),
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Maturity: %{x:.1f}/5<extra></extra>",
                ))
                _fig_arc.update_layout(
                    height=max(240, len(_arc_df) * 36 + 60),
                    margin=dict(l=10, r=50, t=28, b=10),
                    title=dict(text="Layer Maturity Scores (0=planned, 5=operational)", font=dict(size=12), x=0),
                    xaxis=dict(range=[0, 5.4], gridcolor="#e5e7eb"),
                    yaxis=dict(gridcolor="#e5e7eb", autorange="reversed"),
                    xaxis_title="Maturity Score (0–5)", yaxis_title="",
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Inter,sans-serif", size=12),
                )
                st.plotly_chart(_fig_arc, use_container_width=True)
            cols = [c for c in [
                "layer_id", "layer_name", "purpose", "core_question",
                "decision_use", "maturity_score_0_5", "maturity_status"
            ] if c in audit.columns]
            render_badge_table(audit[cols], height=560)

    with tabs[2]:
        guardrails = pd.DataFrame([
            {
                "status": "NO",
                "guardrail": "No live orders",
                "scope": "All pages / all layers",
                "enforcement": "live_allowed remains NO; broker classes are inventory-only.",
            },
            {
                "status": "RED",
                "guardrail": "L8 overrides L7",
                "scope": "Options and action pages",
                "enforcement": "Gamma/kill-zone cannot create an aggressive action while portfolio risk is RED.",
            },
            {
                "status": "REVIEW",
                "guardrail": "Manual event checks",
                "scope": "L5 and L9",
                "enforcement": "News, earnings, liquidity, spread, duplicate exposure, and stress checks must be reviewed.",
            },
            {
                "status": "BLOCKED",
                "guardrail": "Missing data blocks confidence",
                "scope": "L1/L2/L3/L6",
                "enforcement": "NO_DATA and NO_PRICE states remain visible instead of being hidden.",
            },
            {
                "status": "OK",
                "guardrail": "Reports stay recoverable",
                "scope": "System / Output Vault",
                "enforcement": "CSV and markdown outputs are snapshotted and shrinkage-audited.",
            },
        ])
        render_badge_table(guardrails, height=340)

    with tabs[3]:
        if audit.empty:
            st.info("No build roadmap available — `canyon_layer_status_audit.csv` is needed. Maturity levels run 0–5: 0=planned, 1=stub, 2=data wired, 3=logic done, 4=QA pass, 5=operational. Run the full daily runner to generate the audit file.")
            st.code("python3 -u canyon_final_v9_step56_full_10_layer_daily_runner_v2.py", language="bash")
        else:
            roadmap = audit[[c for c in [
                "layer_id", "layer_name", "maturity_status", "missing_outputs", "next_build"
            ] if c in audit.columns]].copy()
            render_badge_table(roadmap, height=480)
        st.subheader("Current Main Decision Status")
        if master.empty:
            st.info("No master matrix yet.")
        else:
            rows = []
            for i in range(1, 11):
                col = f"L{i}_state"
                if col in master.columns:
                    rows.append({
                        "status": "OK" if not count_contains(master, col, "NO_DATA") else "REVIEW",
                        "layer": f"L{i}",
                        "top_states": "; ".join(f"{k}={v}" for k, v in master[col].astype(str).value_counts().head(4).items()),
                    })
            render_badge_table(pd.DataFrame(rows), height=360)

    with tabs[4]:
        st.markdown(read_md(FILES["architecture"]))


def tab_report_archive():
    st.header("Report Archive")

    archive = build_report_archive_index()
    generated = build_output_file_index()
    found = int((archive["status"] == "FOUND").sum())
    missing = int((archive["status"] == "MISSING").sum())

    render_layer_workbench_header(
        "Archive",
        "Report And Output Archive",
        "All markdown reports and generated CSV files are indexed here. Missing reports need a runner re-run to restore.",
        [
            ("Reports Found",   found,            "supportive" if missing == 0 else "cyan"),
            ("Missing",         missing,          "blocked" if missing else "supportive"),
            ("Generated Files", len(generated),   "wait"),
            ("Archive Size",    len(archive),      "cyan"),
        ],
    )
    if missing > 0:
        st.info(f"{missing} report(s) are missing — run the full daily runner to regenerate: `python3 -u canyon_final_v9_step56_full_10_layer_daily_runner_v2.py`")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tracked Reports", len(archive))
    c2.metric("Found", found)
    c3.metric("Missing", missing)
    c4.metric("Generated Files", len(generated))

    if _PLOTLY and not archive.empty and "status" in archive.columns:
        _ra_counts = archive["status"].value_counts().reset_index()
        _ra_counts.columns = ["status", "count"]
        _ra_colors = [
            "#16a34a" if str(s).upper() == "FOUND"
            else "#b91c1c"
            for s in _ra_counts["status"]
        ]
        _fig_ra = go.Figure(go.Bar(
            x=_ra_counts["count"].tolist(),
            y=_ra_counts["status"].astype(str).tolist(),
            orientation="h",
            marker_color=_ra_colors,
            text=_ra_counts["count"].astype(str).tolist(),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x} report(s)<extra></extra>",
        ))
        _fig_ra.update_layout(
            height=max(80, len(_ra_counts) * 48 + 30),
            margin=dict(l=10, r=40, t=10, b=10),
            xaxis_title="Count", yaxis_title="",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#e5e7eb"),
            yaxis=dict(gridcolor="#e5e7eb", autorange="reversed"),
            font=dict(family="Inter,sans-serif", size=13),
        )
        st.plotly_chart(_fig_ra, use_container_width=True)

    report_tab, file_tab = st.tabs(["Text Reports", "All Generated Files"])

    with report_tab:
        st.subheader("All Reports")
        render_badge_table(archive, height=360)

        categories = list(dict.fromkeys(archive["category"].tolist()))
        if not categories:
            st.info("No reports indexed. Run the full daily runner to generate reports.")
        else:
            category = st.selectbox("Category", categories)
            filtered = archive[archive["category"].astype(str) == str(category)]
            report_labels = [f"{row['report']} — {row['file']}" for _, row in filtered.iterrows()]
            if not report_labels:
                st.info("No reports in this category.")
            else:
                chosen = st.selectbox("Report", report_labels)
                if chosen and " — " in chosen:
                    filename = chosen.split(" — ", 1)[1]
                    path = ROOT / filename
                    st.subheader(filename)
                    st.markdown(read_md(path))
                else:
                    st.info("No report selected.")

    with file_tab:
        st.subheader("All Generated Files")
        st.caption("This includes old CSV data, run logs, research tables, reports, and web inputs.")
        render_badge_table(generated, height=420)

        file_choices = generated["file"].tolist()
        if not file_choices:
            st.info("No generated files found. Run the full daily runner to generate output files.")
        else:
            selected_file = st.selectbox("Preview Generated File", file_choices)
            selected_file_str = str(selected_file) if selected_file else ""
            selected_path = ROOT / selected_file_str if selected_file_str else ROOT
            st.subheader(selected_file_str)
            if selected_path.suffix.lower() == ".csv":
                show_df(read_csv(selected_path), height=460)
            else:
                st.markdown(read_md(selected_path))


def tab_output_vault():
    st.header("Output Backup")
    st.caption("This keeps local snapshots of CSV and Markdown outputs so old reports are not quietly overwritten.")

    index = read_csv(FILES["vault_index"])
    alerts = read_csv(FILES["vault_alerts"])

    snapshot_count = index["snapshot_id"].nunique() if not index.empty and "snapshot_id" in index.columns else 0
    file_count = len(index)
    high = int((alerts.get("status", pd.Series(dtype=str)).astype(str).str.upper() == "HIGH").sum()) if not alerts.empty else 0
    medium = int((alerts.get("status", pd.Series(dtype=str)).astype(str).str.upper() == "MEDIUM").sum()) if not alerts.empty else 0

    latest_snapshot = "N/A"
    latest_path = "N/A"
    if not index.empty and {"snapshot_id", "snapshot_path", "created_at"}.issubset(index.columns):
        idx = index.copy()
        idx["_created"] = pd.to_datetime(idx["created_at"], errors="coerce")
        idx = idx.sort_values("_created")
        latest_snapshot = str(idx["snapshot_id"].iloc[-1])
        latest_path = str(Path(idx["snapshot_path"].iloc[-1]).parent)

    _vault_state = "RISK" if high > 0 else ("REVIEW" if medium > 0 or snapshot_count == 0 else "OK")
    render_layer_workbench_header(
        "Vault",
        "Output Backup — Shrinkage Guard",
        "Snapshots all CSV and markdown outputs after each run. Warns when a file loses rows or bytes vs the last snapshot.",
        [
            ("Vault State",  _vault_state,  status_kind(_vault_state)),
            ("Snapshots",    snapshot_count,"supportive" if snapshot_count else "blocked"),
            ("High Alerts",  high,          "risk"       if high          else "supportive"),
            ("Medium Alerts",medium,        "wait"       if medium        else "supportive"),
        ],
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Backup Count", snapshot_count)
    c2.metric("Indexed Files", file_count)
    c3.metric("High Risk Warnings", high)
    c4.metric("Medium Risk Warnings", medium)

    backup_status = "OK" if (high == 0 and snapshot_count > 0) else ("RISK" if high > 0 else "REVIEW")
    vault_summary = pd.DataFrame([
        {
            "status": backup_status,
            "backup_question": "Is the backup system working?",
            "current_read": f"snapshots={snapshot_count}; files indexed={file_count}; latest={latest_snapshot}",
            "why_it_matters": "Without backups you cannot tell if a run quietly deleted or shrank a report.",
            "what_to_do": (
                "Run vault: .venv/bin/python -u canyon_final_v9_step60_output_vault.py --label manual-check"
                if snapshot_count == 0 else "Backup is active. Re-run after any major pipeline change."
            ),
            "source_file": "canyon_output_vault_index.csv; canyon_final_v9_step60_output_vault.py",
        },
        {
            "status": "RISK" if high > 0 else ("REVIEW" if medium > 0 else "OK"),
            "backup_question": "Are there shrinkage warnings to act on?",
            "current_read": f"high risk warnings={high}; medium risk warnings={medium}",
            "why_it_matters": "Shrinkage warnings mean a file has fewer rows or bytes than the previous snapshot — this usually means data was accidentally lost.",
            "what_to_do": (
                "Open Backup Warnings tab and review before trusting recent run outputs."
                if (high > 0 or medium > 0) else "No warnings. Outputs look stable compared to last backup."
            ),
            "source_file": "canyon_output_shrinkage_alerts.csv",
        },
        {
            "status": "OK",
            "backup_question": "What is the restore path if something goes wrong?",
            "current_read": f"latest backup folder: {latest_path}",
            "why_it_matters": "If a report shrinks unexpectedly, the backup folder contains the last known good copy.",
            "what_to_do": "Open Backup Index tab and look for the snapshot with the last good date. Files are stored by snapshot path.",
            "source_file": "canyon_output_vault/ folder",
        },
    ])
    st.subheader("Output Backup Summary")
    render_badge_table(vault_summary, height=270)

    st.markdown(f"""
    <div class="command-grid">
      <div class="command-panel command-cyan">
        <div class="command-label">Latest Backup</div>
        <div class="command-title">{escape(latest_snapshot)}</div>
        <div class="command-text">{escape(latest_path)}</div>
      </div>
      <div class="command-panel command-blocked">
        <div class="command-label">Rule</div>
        <div class="command-title">Do Not Trust A Sudden Output Shrink</div>
        <div class="command-text">If file rows or size suddenly shrink, check warnings before trusting the new run.</div>
      </div>
      <div class="command-panel command-paper">
        <div class="command-label">Restore</div>
        <div class="command-title">Find Old Files By Backup Path</div>
        <div class="command-text">Backups save copied file paths so old reports can be found manually if needed.</div>
      </div>
      <div class="command-panel command-watch">
        <div class="command-label">Area</div>
        <div class="command-title">Tables + Text Reports</div>
        <div class="command-text">Run outputs, layer reports, research tables, logs, and web inputs are included.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if _PLOTLY and not index.empty and "filename" in index.columns and "size_bytes" in index.columns:
        _vault_df = index.copy()
        _vault_df["size_bytes"] = pd.to_numeric(_vault_df["size_bytes"], errors="coerce").fillna(0)
        _vault_top = _vault_df.groupby("filename")["size_bytes"].max().sort_values(ascending=True).tail(15).reset_index()
        if not _vault_top.empty:
            _fig_vault = go.Figure(go.Bar(
                x=(_vault_top["size_bytes"] / 1024).tolist(),
                y=_vault_top["filename"].astype(str).tolist(),
                orientation="h",
                marker_color="#22d3ee",
                text=(_vault_top["size_bytes"] / 1024).map(lambda x: f"{x:.1f}KB").tolist(),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>%{x:.1f} KB<extra></extra>",
            ))
            _fig_vault.update_layout(
                height=max(200, len(_vault_top) * 34 + 60),
                margin=dict(l=10, r=70, t=28, b=20),
                title=dict(text="Largest Backed-Up Files (KB)", font=dict(size=12), x=0),
                xaxis_title="Size (KB)", yaxis_title="",
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(gridcolor="#e5e7eb"),
                yaxis=dict(gridcolor="#e5e7eb"),
                font=dict(family="Inter,sans-serif", size=12),
            )
            st.plotly_chart(_fig_vault, use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["Backup Warnings", "Backup Index", "Report"])
    with tab1:
        if alerts.empty:
            st.info("No shrinkage alert file yet.")
        else:
            render_badge_table(alerts, height=360)
    with tab2:
        if index.empty:
            st.info("No vault index yet.")
        else:
            cols = [c for c in [
                "snapshot_id", "label", "created_at", "filename", "suffix",
                "size_bytes", "rows", "snapshot_path", "source_modified"
            ] if c in index.columns]
            render_badge_table(index[cols].sort_values(["created_at", "filename"], ascending=[False, True]), height=520)
    with tab3:
        st.markdown(read_md(FILES["vault_report"]))


def tab_data_source_health():
    st.header("Data Sources")
    st.caption("This shows whether Canyon uses online data, local fallback data, or simulated fallback data, and confirms research-only status.")

    health = read_csv(FILES["data_source_health"])
    if health.empty:
        st.warning("No data source health file yet. Run Step 61 or the full v2 runner.")
        st.code("python3 -u canyon_final_v9_step61_data_source_health.py")
        return

    counts = health["status"].value_counts() if "status" in health.columns else pd.Series(dtype=int)
    risk = int(counts.get("RISK", 0))
    warn = int(counts.get("WARN", 0))
    ok = int(counts.get("OK", 0))

    _ds_state = "RISK" if risk else ("WARN" if warn else "OK")
    render_layer_workbench_header(
        "L1",
        "Data Source Health",
        "Tracks whether Canyon is using live online data, local fallback, or simulated data for each source.",
        [
            ("Trust Level",  _ds_state, status_kind(_ds_state)),
            ("Risk Sources", risk,      "risk"       if risk else "supportive"),
            ("Warnings",     warn,      "wait"       if warn else "supportive"),
            ("OK Sources",   ok,        "supportive" if ok   else "blocked"),
        ],
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk", risk)
    c2.metric("Warning", warn)
    c3.metric("OK", ok)
    c4.metric("Rows", len(health))

    dns_rows = health[health["area"].astype(str).eq("External DNS")] if "area" in health.columns else pd.DataFrame()
    dns_state = "RISK" if not dns_rows.empty and (dns_rows["status"].astype(str).str.upper() == "RISK").any() else "OK"
    fallback_rows = health[
        health.astype(str).apply(lambda col: col.str.contains("FALLBACK|UNAVAILABLE|NO_PRICE", case=False, regex=True, na=False)).any(axis=1)
    ] if not health.empty else pd.DataFrame()

    trust_summary = pd.DataFrame([
        {
            "status": "RISK" if risk else ("REVIEW" if warn else "OK"),
            "data_question": "Can the system trust today's inputs?",
            "current_read": f"risk={risk}; warning={warn}; ok={ok}",
            "why_it_matters": "Weak data should reduce confidence, not upgrade a ticker.",
            "what_to_do": "Fix risk rows before relying on price, options, or timing conclusions.",
            "source_file": "data_source_health.csv",
        },
        {
            "status": dns_state,
            "data_question": "Is the online data path working?",
            "current_read": friendly_value(dns_state),
            "why_it_matters": "If Yahoo/yfinance is blocked, price and options reads can be stale or unavailable.",
            "what_to_do": "Treat affected layers as research-only until the source refreshes.",
            "source_file": "data_source_health.csv",
        },
        {
            "status": "REVIEW" if not fallback_rows.empty else "OK",
            "data_question": "How much fallback or unavailable data exists?",
            "current_read": f"{len(fallback_rows)} rows",
            "why_it_matters": "Fallback rows are useful for plumbing, but not for confident timing.",
            "what_to_do": "Open Backup Data Focus and inspect the affected layer before acting.",
            "source_file": "data_source_health.csv; market_data_snapshot.csv",
        },
    ])
    st.subheader("Data Trust Summary")
    render_badge_table(trust_summary, height=230)

    panel = "command-risk" if risk else ("command-cyan" if warn else "command-watch")
    st.markdown(f"""
    <div class="command-grid">
      <div class="command-panel {panel}">
        <div class="command-label">Data Sources</div>
        <div class="command-title">{risk} Risk / {warn} Warnings</div>
        <div class="command-text">This explains where data inputs got weaker, but it does not directly change investment decisions.</div>
      </div>
      <div class="command-panel command-risk">
        <div class="command-label">Yahoo Connection</div>
        <div class="command-title">{escape(friendly_value(dns_state))}</div>
        <div class="command-text">If Yahoo is blocked, yfinance price and options data should be treated as unavailable.</div>
      </div>
      <div class="command-panel command-cyan">
        <div class="command-label">Fallback Data Rows</div>
        <div class="command-title">{len(fallback_rows)}</div>
        <div class="command-text">Fallback data is clearly marked and treated conservatively: research only, no timing confirmation, no live orders.</div>
      </div>
      <div class="command-panel command-blocked">
        <div class="command-label">Safety Rules</div>
        <div class="command-title">Missing Data Cannot Upgrade An Action</div>
        <div class="command-text">Missing live data cannot become a stronger action signal.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if _PLOTLY and not health.empty and "status" in health.columns and "area" in health.columns:
        _dsh_df = health.copy()
        _dsh_counts = _dsh_df["status"].value_counts().reset_index()
        _dsh_counts.columns = ["status", "count"]
        _dsh_colors = [
            "#b91c1c" if str(s).upper() == "RISK"
            else "#facc15" if str(s).upper() == "WARN"
            else "#22d3ee"
            for s in _dsh_counts["status"]
        ]
        _fig_dsh = go.Figure(go.Bar(
            x=_dsh_counts["count"].tolist(),
            y=_dsh_counts["status"].astype(str).tolist(),
            orientation="h",
            marker_color=_dsh_colors,
            text=_dsh_counts["count"].astype(str).tolist(),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>%{x} source(s)<extra></extra>",
        ))
        _fig_dsh.update_layout(
            height=max(120, len(_dsh_counts) * 44 + 40),
            margin=dict(l=10, r=40, t=24, b=10),
            title=dict(text="Data Source Status Distribution", font=dict(size=12), x=0),
            xaxis_title="Count", yaxis_title="",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#e5e7eb"),
            yaxis=dict(gridcolor="#e5e7eb", autorange="reversed"),
            font=dict(family="Inter,sans-serif", size=13),
        )
        st.plotly_chart(_fig_dsh, use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["Health Table", "Backup Data Focus", "Report"])
    with tab1:
        render_badge_table(health, height=520)
    with tab2:
        if fallback_rows.empty:
            st.success("No fallback/unavailable rows detected.")
        else:
            render_badge_table(fallback_rows, height=420)
    with tab3:
        st.markdown(read_md(FILES["data_source_health_report"]))


def tab_helper():
    render_layer_workbench_header(
        "RUNBOOK",
        "Runbook Helper",
        "The operating manual for running, debugging, refreshing, and safely extending Canyon v9.",
        [
            ("Daily Path", "Step 56", "watch"),
            ("Dashboard", "Step 55", "cyan"),
            ("QA Gate", "Compile + Vault", "supportive"),
            ("Safety", "Research Only", "risk"),
        ],
    )

    daily_tab, debug_tab, website_tab, data_tab, safety_tab, layers_tab = st.tabs([
        "Daily Run",
        "Debug",
        "Website",
        "Data Source",
        "Safety Rules",
        "10 Layers",
    ])

    with daily_tab:
        st.subheader("Daily Run")
        _hlp_run = build_run_status()
        if _PLOTLY and not _hlp_run.empty and "status" in _hlp_run.columns:
            _hlp_counts = _hlp_run["status"].value_counts().reset_index()
            _hlp_counts.columns = ["status", "count"]
            _hlp_colors = [
                "#16a34a" if str(s).upper() == "FRESH"
                else "#b91c1c" if str(s).upper() == "MISSING"
                else "#facc15"
                for s in _hlp_counts["status"]
            ]
            _fig_hlp = go.Figure(go.Bar(
                x=_hlp_counts["count"].tolist(),
                y=_hlp_counts["status"].astype(str).tolist(),
                orientation="h",
                marker_color=_hlp_colors,
                text=_hlp_counts["count"].astype(str).tolist(),
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>%{x} output(s)<extra></extra>",
            ))
            _fig_hlp.update_layout(
                height=max(80, len(_hlp_counts) * 44 + 30),
                margin=dict(l=10, r=40, t=24, b=10),
                title=dict(text="Live Pipeline Freshness", font=dict(size=12), x=0),
                xaxis_title="Count", yaxis_title="",
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(gridcolor="#e5e7eb"),
                yaxis=dict(gridcolor="#e5e7eb", autorange="reversed"),
                font=dict(family="Inter,sans-serif", size=13),
            )
            st.plotly_chart(_fig_hlp, use_container_width=True)
        workflow = pd.DataFrame([
            {
                "step": "1",
                "command": "source .venv/bin/activate",
                "purpose": "activate the Canyon Python environment",
                "when_to_run": "before any local script",
            },
            {
                "step": "2",
                "command": "python3 -u canyon_final_v9_step56_full_10_layer_daily_runner_v2.py",
                "purpose": "refresh L1-L10, V8 bridges, master matrix, output vault, and data health",
                "when_to_run": "daily refresh",
            },
            {
                "step": "3",
                "command": "streamlit run canyon_final_v9_step55_10_layer_dashboard_v2.py --server.port 8512",
                "purpose": "open the current website/dashboard",
                "when_to_run": "after the runner completes",
            },
            {
                "step": "4",
                "command": "check System Control, Data Source Health, and Output Vault",
                "purpose": "confirm nothing silently shrank or fell back",
                "when_to_run": "before trusting the dashboard",
            },
        ])
        render_badge_table(workflow, height=260)
        st.code(
            """cd ~/Desktop/canyon_quant
source .venv/bin/activate
python3 -u canyon_final_v9_step56_full_10_layer_daily_runner_v2.py
streamlit run canyon_final_v9_step55_10_layer_dashboard_v2.py --server.port 8512
""",
            language="bash",
        )

    with debug_tab:
        st.subheader("Debug Checklist")
        checks = pd.DataFrame([
            {
                "check": "Python compile",
                "command": ".venv/bin/python -m py_compile canyon_final_v9_step55_10_layer_dashboard_v2.py",
                "good_result": "no output and exit code 0",
            },
            {
                "check": "Data source health",
                "command": ".venv/bin/python -u canyon_final_v9_step61_data_source_health.py",
                "good_result": "health csv/report created; Yahoo DNS issues stay explicit",
            },
            {
                "check": "Output vault",
                "command": ".venv/bin/python -u canyon_final_v9_step60_output_vault.py --label manual-check",
                "good_result": "alerts = 0 or listed shrinkage is explained",
            },
            {
                "check": "Full runner twice",
                "command": ".venv/bin/python -u canyon_final_v9_step56_full_10_layer_daily_runner_v2.py",
                "good_result": "Full 10-Layer Daily Runner v2: OK",
            },
        ])
        render_badge_table(checks, height=300)
        st.code(
            """.venv/bin/python -m py_compile canyon_final_v9_step55_10_layer_dashboard_v2.py
.venv/bin/python -u canyon_final_v9_step61_data_source_health.py
.venv/bin/python -u canyon_final_v9_step60_output_vault.py --label manual-check
.venv/bin/python -u canyon_final_v9_step56_full_10_layer_daily_runner_v2.py
.venv/bin/python -u canyon_final_v9_step56_full_10_layer_daily_runner_v2.py
""",
            language="bash",
        )

    with website_tab:
        st.subheader("Website Commands")
        website_rows = pd.DataFrame([
            {
                "task": "Start dashboard on current port",
                "command": "streamlit run canyon_final_v9_step55_10_layer_dashboard_v2.py --server.port 8512",
                "note": "matches the current localhost:8512 browser view",
            },
            {
                "task": "Start dashboard on default Streamlit port",
                "command": "streamlit run canyon_final_v9_step55_10_layer_dashboard_v2.py",
                "note": "uses Streamlit default port if available",
            },
            {
                "task": "Refresh after file change",
                "command": "click Rerun in Streamlit or press R",
                "note": "only reloads UI; it does not refresh source data",
            },
        ])
        render_badge_table(website_rows, height=250)
        st.code(
            """cd ~/Desktop/canyon_quant
source .venv/bin/activate
streamlit run canyon_final_v9_step55_10_layer_dashboard_v2.py --server.port 8512
""",
            language="bash",
        )

    with data_tab:
        st.subheader("Data Refresh Routes")
        data_routes = pd.DataFrame([
            {
                "route": "L1-L6 missing layers",
                "script": "canyon_final_v9_step53_build_missing_layers_runner.py",
                "output": "macro, sector, fundamental, event, technical reports",
            },
            {
                "route": "Main Decision",
                "script": "canyon_final_v9_step54_master_10_layer_decision_v2.py",
                "output": "master_10_layer_decision_matrix_v2.csv",
            },
            {
                "route": "V8 research bridge",
                "script": "canyon_final_v9_step57_v8_research_bridge.py",
                "output": "v8 research overlays, no live execution",
            },
            {
                "route": "V8 execution gate",
                "script": "canyon_final_v9_step58_v8_l9_execution_gate.py",
                "output": "research-only L9 gate",
            },
            {
                "route": "V8 advanced risk",
                "script": "canyon_final_v9_step59_v8_advanced_risk_bridge.py",
                "output": "PCA/tail risk research overlays",
            },
            {
                "route": "Data source health",
                "script": "canyon_final_v9_step61_data_source_health.py",
                "output": "explicit source OK/WARN/RISK table",
            },
        ])
        render_badge_table(data_routes, height=340)
        st.code(
            """python3 -u canyon_final_v9_step53_build_missing_layers_runner.py
python3 -u canyon_final_v9_step54_master_10_layer_decision_v2.py
python3 -u canyon_final_v9_step57_v8_research_bridge.py
python3 -u canyon_final_v9_step58_v8_l9_execution_gate.py
python3 -u canyon_final_v9_step59_v8_advanced_risk_bridge.py
python3 -u canyon_final_v9_step61_data_source_health.py
""",
            language="bash",
        )

    with safety_tab:
        st.subheader("Safety Rules")
        rules = pd.DataFrame([
            {
                "rule": "No broker connection",
                "meaning": "Canyon v9 never connects to a broker account.",
                "result": "research dashboard only",
            },
            {
                "rule": "No live order",
                "meaning": "The system cannot place real trades.",
                "result": "paper ledger and research notes only",
            },
            {
                "rule": "L8 overrides L7",
                "meaning": "Portfolio risk can block an options signal.",
                "result": "Risk RED means no aggressive action",
            },
            {
                "rule": "WAIT is not buy",
                "meaning": "WAIT means confirmation is missing.",
                "result": "do not chase weekly OTM options",
            },
            {
                "rule": "Missing data cannot improve a signal",
                "meaning": "Fallback or unavailable rows stay conservative.",
                "result": "fix data first or keep research-only",
            },
            {
                "rule": "Output vault before trust",
                "meaning": "If reports disappear or shrink, investigate before using conclusions.",
                "result": "run vault snapshot after major edits",
            },
        ])
        render_badge_table(rules, height=360)
        st.info("This helper is intentionally operational: it tells you exactly what to run and which guardrails cannot be bypassed.")

    with layers_tab:
        st.subheader("10 Layers Explained")
        st.caption("Each layer answers one specific question. They must be evaluated top-down — macro to ticker to action.")
        layers_ref = pd.DataFrame([
            {
                "layer": "L1 · Data Trust",
                "job": "Is data real and fresh?",
                "outputs": "data_source_health.csv · data_quality_flags.csv",
                "blocks_action_when": "Source is RISK or FALLBACK — reduces confidence across all downstream layers.",
                "where_to_check": "System Check › Data Sources",
            },
            {
                "layer": "L2 · Market Mood",
                "job": "Is the broad market supportive or hostile?",
                "outputs": "macro_regime_signals.csv · volatility_regime.csv · index_breadth_dashboard.csv",
                "blocks_action_when": "SPY downtrend or VIX elevated — defensive posture regardless of individual tickers.",
                "where_to_check": "All Layers › Layer 2",
            },
            {
                "layer": "L3 · Sectors & Themes",
                "job": "Which sectors are rotating into strength?",
                "outputs": "sector_rotation_scores.csv · theme_heatmap.csv",
                "blocks_action_when": "Ticker sector is LAGGARD — reduces conviction for new positions.",
                "where_to_check": "All Layers › Layer 3",
            },
            {
                "layer": "L4 · Company Basics",
                "job": "Do the fundamentals support the thesis?",
                "outputs": "fundamental_quality_valuation.csv · valuation_risk_flags.csv",
                "blocks_action_when": "ETF_NOT_FUNDAMENTAL: use sector context. High PE + no growth: flag as Expensive.",
                "where_to_check": "All Layers › Layer 4 · Single-Ticker Notebook",
            },
            {
                "layer": "L5 · News & Events",
                "job": "Is there a catalyst or a risk event approaching?",
                "outputs": "evidence_cards.csv · news_event_risk.csv · earnings_calendar_check.csv",
                "blocks_action_when": "Earnings date pending or high 8-K density — hold off until event passes.",
                "where_to_check": "All Layers › Layer 5",
            },
            {
                "layer": "L6 · Price Trend",
                "job": "Does the price action confirm the research thesis?",
                "outputs": "technical_signal_matrix.csv · intraday_liquidity_proxy.csv",
                "blocks_action_when": "Price below 20/50 DMA, RSI overextended, or illiquid ticker — research only.",
                "where_to_check": "All Layers › Layer 6 · Single-Ticker Notebook",
            },
            {
                "layer": "L7 · Options Context",
                "job": "Is options pressure helping or hurting timing?",
                "outputs": "options_decision_matrix.csv · gamma_squeeze_candidates.csv · option_kill_zone_risk.csv",
                "blocks_action_when": "Kill zone HIGH or gamma edge insufficient — paper-only or skip.",
                "where_to_check": "All Layers › Layer 7 · Research Room › Options Watch",
            },
            {
                "layer": "L8 · Portfolio Risk",
                "job": "Does the full portfolio have room for this position?",
                "outputs": "exposure_dashboard.csv · scenario_stress_results.csv · position_sizing_recommendations.csv",
                "blocks_action_when": "Risk light RED or exposure warnings HIGH — no new positions until risk is reduced.",
                "where_to_check": "Portfolio Risk tab",
            },
            {
                "layer": "L9 · Before-Action Check",
                "job": "Did all manual safety checks pass?",
                "outputs": "pre_trade_checklist.csv · pre_trade_order_ticket.csv · v8_l9_execution_gate.csv",
                "blocks_action_when": "Any manual check is NO — no paper test until all checks are YES.",
                "where_to_check": "Research Room › Before-Action Check · Action Board",
            },
            {
                "layer": "L10 · Learning Review",
                "job": "What did the last paper tests teach us?",
                "outputs": "learning_attribution_summary.csv · learning_weight_suggestions.csv · paper_portfolio_ledger.csv",
                "blocks_action_when": "Fewer than 30 closed samples — do not auto-adjust strategy weights.",
                "where_to_check": "Portfolio Risk › Paper Log",
            },
        ])
        render_badge_table(layers_ref, height=560)

        st.subheader("Layer Priority Order")
        st.markdown("""
**L8 overrides L7.** Portfolio risk can block any options signal.
**L9 overrides everything.** Manual checks must pass before any paper test.
**L1 poisons downstream.** Bad data reduces confidence in all layers below it.
**L10 does not auto-adjust.** Fewer than 30 clean closed samples = no weight changes.
**Live orders are never allowed.** Regardless of which conditions are met in any layer.
""")


def tab_portfolio_optimizer():  # noqa: C901
    """Step 63 — LedoitWolf covariance + mean-variance portfolio optimizer."""
    weights_df = read_csv(FILES["portfolio_weights"])
    ef_df      = read_csv(FILES["portfolio_ef_points"])
    cov_df     = read_csv(FILES["portfolio_covariance"])

    # ── header ────────────────────────────────────────────────────────────────
    _ms_sharpe  = "—"
    _rp_sharpe  = "—"
    _iv_sharpe  = "—"
    _n_tickers  = 0
    if not weights_df.empty:
        _n_tickers = len(weights_df)
        if "mu_annual" in weights_df.columns and "vol_annual" in weights_df.columns:
            _ms_w = pd.to_numeric(weights_df.get("max_sharpe", pd.Series(dtype=float)), errors="coerce").fillna(0).values
            _mu   = pd.to_numeric(weights_df["mu_annual"], errors="coerce").fillna(0).values
            _vol  = pd.to_numeric(weights_df["vol_annual"], errors="coerce").fillna(0).values
            if len(_ms_w) and _ms_w.sum() > 0.1:
                _ms_ret = float(np.dot(_ms_w, _mu))
                _ms_vol_est = float(np.dot(_ms_w, _vol))
                if _ms_vol_est > 0.01:
                    _ms_sharpe = f"{(_ms_ret - 0.053) / _ms_vol_est:.2f}"

    _ms_kind = "supportive" if _ms_sharpe not in ("—",) and float(_ms_sharpe) > 1.5 else "cyan"
    render_layer_workbench_header(
        "Step 63",
        "LedoitWolf Portfolio Optimizer",
        "Three optimal portfolios from shrinkage covariance: Max-Sharpe, Min-Variance, Risk-Parity. "
        "Replaces naive inverse-vol sizing with correlation-aware weighting.",
        [
            ("Tickers Optimized", _n_tickers, "supportive" if _n_tickers >= 10 else "cyan"),
            ("Max-Sharpe (est.)", _ms_sharpe, _ms_kind),
            ("Method", "LedoitWolf", "supportive"),
            ("Constraint", "Long-only 25% cap", "wait"),
        ],
    )

    if weights_df.empty:
        st.info(
            "No optimizer results found. Run the engine first:\n\n"
            "```bash\n"
            "cd /Users/renjingru/Desktop/canyon_quant\n"
            "python canyon_final_v9_step63_portfolio_optimizer.py\n"
            "```"
        )
        return

    opt_tabs = st.tabs(["Weights", "Efficient Frontier", "Correlation Heatmap", "How to Run"])

    # ── Tab: Weights ──────────────────────────────────────────────────────────
    with opt_tabs[0]:
        st.subheader("Optimized vs Baseline Weights")
        _pct_cols = ["max_sharpe", "min_var", "risk_parity", "inv_vol"]
        _w = weights_df.copy()
        for _c in _pct_cols:
            if _c in _w.columns:
                _w[_c] = pd.to_numeric(_w[_c], errors="coerce").fillna(0)

        if _PLOTLY:
            _colors = {"max_sharpe": "#0ea5e9", "min_var": "#8b5cf6",
                       "risk_parity": "#10b981", "inv_vol": "#94a3b8"}
            _fig = go.Figure()
            for _col, _name in [("max_sharpe","Max-Sharpe"), ("min_var","Min-Variance"),
                                  ("risk_parity","Risk-Parity"), ("inv_vol","Inv-Vol (base)")]:
                if _col in _w.columns:
                    _fig.add_trace(go.Bar(
                        name=_name,
                        x=_w["ticker"].astype(str).tolist(),
                        y=(_w[_col] * 100).tolist(),
                        marker_color=_colors.get(_col, "#64748b"),
                        hovertemplate=f"<b>{_name}</b><br>%{{x}}: %{{y:.1f}}%<extra></extra>",
                    ))
            _fig.update_layout(
                barmode="group", height=360,
                margin=dict(l=10, r=10, t=32, b=10),
                title=dict(text="Portfolio Weights by Strategy (%)", font=dict(size=13), x=0),
                xaxis=dict(title="", tickangle=-30),
                yaxis=dict(title="Weight %", gridcolor="#f1f5f9"),
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", y=1.08),
                font=dict(family="Inter,sans-serif", size=12),
            )
            st.plotly_chart(_fig, use_container_width=True)

        _display = _w.copy()
        for _c in _pct_cols:
            if _c in _display.columns:
                _display[_c] = _display[_c].apply(lambda v: f"{v*100:.1f}%")
        if "mu_annual" in _display.columns:
            _display["mu_annual"] = pd.to_numeric(_display["mu_annual"], errors="coerce").apply(
                lambda v: f"{v*100:.1f}%" if pd.notna(v) else ""
            )
        if "vol_annual" in _display.columns:
            _display["vol_annual"] = pd.to_numeric(_display["vol_annual"], errors="coerce").apply(
                lambda v: f"{v*100:.1f}%" if pd.notna(v) else ""
            )
        _rename_w = {"ticker": "Ticker", "max_sharpe": "Max-Sharpe", "min_var": "Min-Var",
                     "risk_parity": "Risk-Parity", "inv_vol": "Inv-Vol",
                     "mu_annual": "Exp. Return", "vol_annual": "Annl. Vol"}
        st.dataframe(
            _display[[c for c in ["ticker","max_sharpe","min_var","risk_parity","inv_vol","mu_annual","vol_annual"]
                       if c in _display.columns]].rename(columns=_rename_w),
            use_container_width=True, hide_index=True,
        )
        st.info("💡 **Max-Sharpe** maximises risk-adjusted return. "
                "**Min-Var** minimises drawdown risk. "
                "**Risk-Parity** equal risk contribution — good for uncertain regimes.")

    # ── Tab: Efficient Frontier ───────────────────────────────────────────────
    with opt_tabs[1]:
        st.subheader("Efficient Frontier")
        if ef_df.empty:
            st.info("No frontier data.")
        else:
            _ef = ef_df.copy()
            for _c in ["vol", "ret", "sharpe"]:
                if _c in _ef.columns:
                    _ef[_c] = pd.to_numeric(_ef[_c], errors="coerce")

            if _PLOTLY:
                _ef_colors = (_ef["sharpe"].tolist() if "sharpe" in _ef.columns
                              else ["#0ea5e9"] * len(_ef))
                _fig_ef = go.Figure()
                _fig_ef.add_trace(go.Scatter(
                    x=(_ef["vol"] * 100).tolist(),
                    y=(_ef["ret"] * 100).tolist(),
                    mode="lines+markers",
                    marker=dict(
                        color=_ef["sharpe"].tolist(),
                        colorscale="RdYlGn",
                        size=7,
                        colorbar=dict(title="Sharpe"),
                        showscale=True,
                    ),
                    line=dict(color="#0ea5e9", width=1.5),
                    hovertemplate="<b>Frontier</b><br>Vol: %{x:.1f}%<br>Ret: %{y:.1f}%<br>Sharpe: %{marker.color:.3f}<extra></extra>",
                    name="Efficient Frontier",
                ))
                # Mark key portfolios
                for _pname, _col, _color in [("Max-Sharpe","max_sharpe","#0ea5e9"),
                                              ("Min-Var","min_var","#8b5cf6"),
                                              ("Risk-Parity","risk_parity","#10b981")]:
                    if not weights_df.empty and _col in weights_df.columns:
                        _w_arr = pd.to_numeric(weights_df[_col], errors="coerce").fillna(0).values
                        _mu_arr = pd.to_numeric(weights_df.get("mu_annual", pd.Series(dtype=float)), errors="coerce").fillna(0).values
                        _v_arr  = pd.to_numeric(weights_df.get("vol_annual", pd.Series(dtype=float)), errors="coerce").fillna(0).values
                        if len(_w_arr) == len(_mu_arr) and _w_arr.sum() > 0.1:
                            _pr = float(np.dot(_w_arr, _mu_arr))
                            _pv = float(np.sqrt(max(np.dot(_w_arr ** 2, _v_arr ** 2), 1e-6)))
                            _fig_ef.add_trace(go.Scatter(
                                x=[_pv * 100], y=[_pr * 100],
                                mode="markers",
                                marker=dict(size=12, color=_color, symbol="diamond"),
                                name=_pname,
                                hovertemplate=f"<b>{_pname}</b><br>Vol: %{{x:.1f}}%<br>Ret: %{{y:.1f}}%<extra></extra>",
                            ))
                _fig_ef.update_layout(
                    height=400, margin=dict(l=10, r=10, t=32, b=10),
                    title=dict(text="Efficient Frontier (annualised)", font=dict(size=13), x=0),
                    xaxis=dict(title="Annualised Volatility (%)", gridcolor="#f1f5f9"),
                    yaxis=dict(title="Annualised Expected Return (%)", gridcolor="#f1f5f9"),
                    plot_bgcolor="white", paper_bgcolor="white",
                    legend=dict(orientation="h", y=1.08),
                    font=dict(family="Inter,sans-serif", size=12),
                )
                st.plotly_chart(_fig_ef, use_container_width=True)

            st.caption(
                "Each point = a minimum-variance portfolio at a given target return. "
                "Color = Sharpe ratio (green = higher Sharpe). "
                "Diamonds = named portfolios."
            )

    # ── Tab: Correlation Heatmap ──────────────────────────────────────────────
    with opt_tabs[2]:
        st.subheader("Asset Correlation Heatmap (LedoitWolf Covariance)")
        if cov_df.empty:
            st.info("No covariance data.")
        else:
            try:
                _cov = cov_df.copy()
                _cov.index = _cov.columns
                _stds = np.sqrt(np.diag(_cov.values.astype(float)))
                _outer = np.outer(_stds, _stds)
                _corr = _cov.values.astype(float) / (_outer + 1e-10)
                np.fill_diagonal(_corr, 1.0)
                _corr = np.clip(_corr, -1.0, 1.0)
                _corr_df = pd.DataFrame(_corr, index=_cov.index, columns=_cov.columns)

                if _PLOTLY:
                    _fig_hm = go.Figure(go.Heatmap(
                        z=_corr_df.values.tolist(),
                        x=_corr_df.columns.tolist(),
                        y=_corr_df.index.tolist(),
                        colorscale="RdBu_r",
                        zmid=0, zmin=-1, zmax=1,
                        text=[[f"{v:.2f}" for v in row] for row in _corr_df.values],
                        texttemplate="%{text}",
                        textfont=dict(size=9),
                        hovertemplate="<b>%{y} vs %{x}</b><br>Corr: %{z:.3f}<extra></extra>",
                        colorbar=dict(title="Correlation"),
                    ))
                    _n_tck = len(_corr_df)
                    _fig_hm.update_layout(
                        height=max(400, _n_tck * 32 + 80),
                        margin=dict(l=10, r=10, t=32, b=10),
                        title=dict(text="Pairwise Correlation (shrunk)", font=dict(size=13), x=0),
                        xaxis=dict(tickangle=-30),
                        font=dict(family="Inter,sans-serif", size=10),
                    )
                    st.plotly_chart(_fig_hm, use_container_width=True)
                    # Highlight high-correlation pairs
                    _pairs = []
                    tckrs = _corr_df.index.tolist()
                    for _i in range(len(tckrs)):
                        for _j in range(_i + 1, len(tckrs)):
                            _c = _corr[_i, _j]
                            if abs(_c) > 0.70:
                                _pairs.append({"Asset A": tckrs[_i], "Asset B": tckrs[_j],
                                               "Correlation": round(_c, 3)})
                    if _pairs:
                        st.subheader(f"High-Correlation Pairs (|r| > 0.70) — {len(_pairs)} found")
                        st.caption("These pairs move together — holding both doesn't add much diversification.")
                        _pairs_df = pd.DataFrame(_pairs).sort_values("Correlation", ascending=False)
                        st.dataframe(_pairs_df, use_container_width=True, hide_index=True)
            except Exception as _e:
                st.error(f"Could not compute correlation: {_e}")

    # ── Tab: How to Run ───────────────────────────────────────────────────────
    with opt_tabs[3]:
        st.subheader("How to Run the Portfolio Optimizer")
        st.markdown("""
**File**: `canyon_final_v9_step63_portfolio_optimizer.py`  |  **Runtime**: ~1–2 seconds

```bash
cd /Users/renjingru/Desktop/canyon_quant

# Default (20 core tickers, 1-year lookback, 25% cap)
python canyon_final_v9_step63_portfolio_optimizer.py

# 2-year lookback
python canyon_final_v9_step63_portfolio_optimizer.py --lookback 504

# Tighter position cap
python canyon_final_v9_step63_portfolio_optimizer.py --cap 0.20

# Custom ticker list
python canyon_final_v9_step63_portfolio_optimizer.py --tickers AAPL MSFT NVDA GOOGL META
```

| Output file | Contents |
|---|---|
| `portfolio_optimized_weights.csv` | Ticker weights for Max-Sharpe, Min-Var, Risk-Parity, Inv-Vol |
| `portfolio_covariance.csv` | N×N LedoitWolf shrunk covariance matrix |
| `portfolio_ef_points.csv` | Efficient frontier scatter (vol, ret, sharpe) |
| `portfolio_optimizer_report.md` | Full markdown report |

**Math summary**:
- Log-returns from last `--lookback` trading days
- LedoitWolf shrinkage: `Σ̂ = (1−α)·S + α·μI` (reduces estimation error, sklearn)
- Max-Sharpe: SLSQP with 15 random restarts (avoids local minima)
- Risk-Parity: equal risk contribution per asset
- All portfolios: long-only, 25% max per position
""")
        if st.button("🔄 Re-run Optimizer", type="primary"):
            import subprocess
            _path = str(ROOT / "canyon_final_v9_step63_portfolio_optimizer.py")
            with st.spinner("Running optimizer (~2s)..."):
                try:
                    _r = subprocess.run(["python3", _path], capture_output=True,
                                        text=True, timeout=60, cwd=str(ROOT))
                    if _r.returncode == 0:
                        st.success("✅ Optimizer completed. Refresh tab to see updated results.")
                        if _r.stdout.strip():
                            st.code(_r.stdout[-2000:], language="")
                    else:
                        st.error(f"Error code {_r.returncode}")
                        st.code(_r.stderr[-1000:] if _r.stderr else "(no stderr)", language="")
                except Exception as _e:
                    st.error(f"Could not run: {_e}")


def tab_data_layer():  # noqa: C901
    """Step 64 — Unified multi-source data layer status and configuration."""
    status_df = read_csv(FILES["data_layer_status"])
    report_md = read_md(FILES["data_layer_report"])

    render_layer_workbench_header(
        "Step 64",
        "Unified Data Layer",
        "Three-source market data with priority failover: Polygon.io → Alpaca → yfinance. "
        "Configure API keys to unlock premium data sources.",
        [
            ("Primary Source", "yfinance",
             "supportive" if not status_df.empty and "yfinance" in status_df.get("source", pd.Series(dtype=str)).astype(str).values else "cyan"),
            ("Polygon.io",  "Set POLYGON_API_KEY", "blocked"),
            ("Alpaca",      "Set ALPACA_API_KEY",  "blocked"),
            ("Cache TTL",   "12 hours", "wait"),
        ],
    )

    dl_tabs = st.tabs(["Source Status", "Coverage Detail", "How to Configure"])

    with dl_tabs[0]:
        st.subheader("Data Source Status")

        # Source status cards
        c1, c2, c3 = st.columns(3)
        _has_polygon = bool(os.environ.get("POLYGON_API_KEY", ""))
        _has_alpaca  = bool(os.environ.get("ALPACA_API_KEY", ""))
        with c1:
            st.markdown(
                f'<div style="background:{"#dcfce7" if _has_polygon else "#fee2e2"};'
                f'border-radius:8px;padding:16px 20px;border-left:4px solid '
                f'{"#16a34a" if _has_polygon else "#dc2626"}">'
                f'<div style="font-size:11px;color:#64748b;text-transform:uppercase">Priority 1</div>'
                f'<div style="font-size:18px;font-weight:700">Polygon.io</div>'
                f'<div style="color:{"#16a34a" if _has_polygon else "#dc2626"};font-size:13px">'
                f'{"✓ API key configured" if _has_polygon else "✗ POLYGON_API_KEY not set"}</div>'
                f'<div style="font-size:11px;color:#64748b;margin-top:6px">Institutional grade, sub-second</div>'
                f'</div>', unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div style="background:{"#dcfce7" if _has_alpaca else "#fee2e2"};'
                f'border-radius:8px;padding:16px 20px;border-left:4px solid '
                f'{"#16a34a" if _has_alpaca else "#dc2626"}">'
                f'<div style="font-size:11px;color:#64748b;text-transform:uppercase">Priority 2</div>'
                f'<div style="font-size:18px;font-weight:700">Alpaca</div>'
                f'<div style="color:{"#16a34a" if _has_alpaca else "#dc2626"};font-size:13px">'
                f'{"✓ Keys configured" if _has_alpaca else "✗ Keys not set"}</div>'
                f'<div style="font-size:11px;color:#64748b;margin-top:6px">Free tier, real-time IEX</div>'
                f'</div>', unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                '<div style="background:#dcfce7;border-radius:8px;padding:16px 20px;border-left:4px solid #16a34a">'
                '<div style="font-size:11px;color:#64748b;text-transform:uppercase">Priority 3</div>'
                '<div style="font-size:18px;font-weight:700">yfinance</div>'
                '<div style="color:#16a34a;font-size:13px">✓ Always available</div>'
                '<div style="font-size:11px;color:#64748b;margin-top:6px">Free, EOD only, no key needed</div>'
                '</div>', unsafe_allow_html=True,
            )

        if report_md:
            st.markdown("---")
            st.markdown(report_md)

    with dl_tabs[1]:
        st.subheader("Ticker Coverage Detail")
        if status_df.empty:
            st.info(
                "No coverage data. Run the data layer:\n\n"
                "```bash\npython canyon_final_v9_step64_data_upgrade.py\n```"
            )
        else:
            if "source" in status_df.columns:
                _src_counts = status_df["source"].value_counts()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Tickers", len(status_df))
                c2.metric("yfinance", int(_src_counts.get("yfinance", 0)))
                c3.metric("Polygon",  int(_src_counts.get("polygon", 0)))
                c4.metric("Failed",   int(_src_counts.get("FAILED", 0)))

            _cols = [c for c in ["ticker","source","rows","latency_ms","freshness","as_of"]
                     if c in status_df.columns]
            st.dataframe(status_df[_cols], use_container_width=True, hide_index=True)

    with dl_tabs[2]:
        st.subheader("How to Configure API Keys")
        st.markdown("""
### Polygon.io (Recommended Upgrade)

**Free tier**: 5 API calls/minute, 15-minute delayed data
**Starter ($29/mo)**: Unlimited calls, real-time
https://polygon.io

```bash
# Add to your shell profile (~/.zshrc or ~/.bashrc)
export POLYGON_API_KEY="your_key_here"

# Then run:
python canyon_final_v9_step64_data_upgrade.py
```

### Alpaca (Free Alternative)

**Free tier**: Real-time IEX feed, paper trading
https://alpaca.markets → Sign up → API Keys

```bash
export ALPACA_API_KEY="your_key_id"
export ALPACA_SECRET_KEY="your_secret_key"

python canyon_final_v9_step64_data_upgrade.py
```

### Verify Configuration

```bash
# Check which sources are active
python canyon_final_v9_step64_data_upgrade.py --check

# Download and compare all sources
python canyon_final_v9_step64_data_upgrade.py --benchmark

# Force re-download (bypass cache)
python canyon_final_v9_step64_data_upgrade.py --force
```

### Current Limitation (yfinance only)
- End-of-day closing prices only
- Rate-limited (10-second delay between large batch downloads)
- Occasional data gaps during API maintenance
- **Upgrading to Polygon** eliminates all three limitations
""")
        if st.button("🔍 Run Source Check Now", type="primary"):
            import subprocess
            _path = str(ROOT / "canyon_final_v9_step64_data_upgrade.py")
            with st.spinner("Checking data sources..."):
                try:
                    _r = subprocess.run(["python3", _path, "--check"],
                                        capture_output=True, text=True, timeout=30, cwd=str(ROOT))
                    st.code(_r.stdout[-2000:] if _r.stdout else "(no output)", language="")
                except Exception as _e:
                    st.error(str(_e))


def tab_earnings_nlp():  # noqa: C901
    """Step 65 — Earnings NLP sentiment scorer."""
    scores_df   = read_csv(FILES["earnings_nlp_scores"])
    calendar_df = read_csv(FILES["earnings_calendar"])

    # ── header stats ──────────────────────────────────────────────────────────
    _n_bullish  = 0
    _n_bearish  = 0
    _n_total    = 0
    _top_ticker = "—"
    _top_score  = 0.0
    if not scores_df.empty and "sentiment" in scores_df.columns:
        _n_total   = len(scores_df)
        _n_bullish = int(scores_df["sentiment"].astype(str).str.contains("BULLISH").sum())
        _n_bearish = int(scores_df["sentiment"].astype(str).str.contains("BEARISH").sum())
        if "forward_score" in scores_df.columns and "ticker" in scores_df.columns:
            _fs = pd.to_numeric(scores_df["forward_score"], errors="coerce").fillna(0)
            if not _fs.empty:
                _top_idx = _fs.idxmax()
                _top_ticker = str(scores_df.loc[_top_idx, "ticker"])
                _top_score  = float(_fs.max())

    _bull_kind = "supportive" if _n_bullish > _n_bearish else ("risk" if _n_bearish > _n_bullish * 2 else "cyan")
    render_layer_workbench_header(
        "Step 65",
        "Earnings NLP Scorer",
        "FinBERT-inspired keyword scoring of recent earnings news. "
        "Three tiers: keyword (offline) → GPT-4o-mini → full earnings call transcript.",
        [
            ("Tickers Scored",   _n_total,    "supportive" if _n_total > 0 else "blocked"),
            ("Bullish Signals",  _n_bullish,  _bull_kind),
            ("Bearish Signals",  _n_bearish,  "risk" if _n_bearish > 5 else "cyan"),
            ("Top Score",        f"{_top_ticker} {_top_score:+.3f}", "supportive" if _top_score > 0.2 else "wait"),
        ],
    )

    if scores_df.empty:
        st.info(
            "No NLP scores found. Run the scorer first:\n\n"
            "```bash\n"
            "python canyon_final_v9_step65_earnings_nlp.py\n"
            "# With GPT (requires OPENAI_API_KEY):\n"
            "python canyon_final_v9_step65_earnings_nlp.py --gpt\n"
            "```"
        )
        return

    nlp_tabs = st.tabs(["Sentiment Dashboard", "Score Table", "Earnings Calendar", "How to Use"])

    # ── Sentiment Dashboard ───────────────────────────────────────────────────
    with nlp_tabs[0]:
        st.subheader("Earnings Sentiment Overview")

        # Distribution bar
        if "sentiment" in scores_df.columns:
            _sent_order = ["BULLISH", "MILDLY_BULLISH", "NEUTRAL", "MILDLY_BEARISH", "BEARISH"]
            _sent_colors = {
                "BULLISH":        "#16a34a",
                "MILDLY_BULLISH": "#4ade80",
                "NEUTRAL":        "#94a3b8",
                "MILDLY_BEARISH": "#fb923c",
                "BEARISH":        "#dc2626",
            }
            _sent_counts = scores_df["sentiment"].astype(str).value_counts()

            c1, c2, c3, c4, c5 = st.columns(5)
            for _col, _sent in zip([c1, c2, c3, c4, c5], _sent_order):
                _cnt = int(_sent_counts.get(_sent, 0))
                _col.metric(_sent.replace("_", " "), _cnt)

            if _PLOTLY:
                _fig_bar = go.Figure()
                for _sent in _sent_order:
                    _cnt = int(_sent_counts.get(_sent, 0))
                    if _cnt > 0:
                        _fig_bar.add_trace(go.Bar(
                            name=_sent.replace("_", " "),
                            x=[_sent.replace("_", " ")],
                            y=[_cnt],
                            marker_color=_sent_colors.get(_sent, "#94a3b8"),
                            text=[str(_cnt)],
                            textposition="outside",
                        ))
                _fig_bar.update_layout(
                    height=250, margin=dict(l=10, r=10, t=28, b=10),
                    showlegend=False,
                    plot_bgcolor="white", paper_bgcolor="white",
                    xaxis=dict(title=""),
                    yaxis=dict(title="Count", gridcolor="#f1f5f9"),
                    font=dict(family="Inter,sans-serif", size=12),
                )
                st.plotly_chart(_fig_bar, use_container_width=True)

        # Top movers
        st.markdown("---")
        ca, cb = st.columns(2)
        if "forward_score" in scores_df.columns and "ticker" in scores_df.columns:
            _fs = pd.to_numeric(scores_df["forward_score"], errors="coerce").fillna(0)
            scores_df = scores_df.copy()
            scores_df["_fs_num"] = _fs
            with ca:
                st.subheader("🟢 Top Bullish")
                _top_bull = scores_df.nlargest(5, "_fs_num")[
                    [c for c in ["ticker","_fs_num","sentiment","key_themes","beat_miss"] if c in scores_df.columns]
                ]
                for _, _r in _top_bull.iterrows():
                    st.markdown(
                        f'<div style="background:#f0fdf4;border-left:3px solid #16a34a;'
                        f'padding:8px 12px;margin-bottom:6px;border-radius:4px">'
                        f'<b style="font-size:15px">{_r.get("ticker","")}</b>'
                        f'<span style="color:#16a34a;font-size:13px;margin-left:8px">'
                        f'{float(_r.get("_fs_num",0)):+.3f}</span>'
                        f'<br><span style="color:#64748b;font-size:12px">'
                        f'{str(_r.get("key_themes",""))[:50]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            with cb:
                st.subheader("🔴 Top Bearish")
                _top_bear = scores_df.nsmallest(5, "_fs_num")[
                    [c for c in ["ticker","_fs_num","sentiment","key_themes","beat_miss"] if c in scores_df.columns]
                ]
                for _, _r in _top_bear.iterrows():
                    st.markdown(
                        f'<div style="background:#fef2f2;border-left:3px solid #dc2626;'
                        f'padding:8px 12px;margin-bottom:6px;border-radius:4px">'
                        f'<b style="font-size:15px">{_r.get("ticker","")}</b>'
                        f'<span style="color:#dc2626;font-size:13px;margin-left:8px">'
                        f'{float(_r.get("_fs_num",0)):+.3f}</span>'
                        f'<br><span style="color:#64748b;font-size:12px">'
                        f'{str(_r.get("key_themes",""))[:50]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    # ── Score Table ───────────────────────────────────────────────────────────
    with nlp_tabs[1]:
        st.subheader("Full Score Table")
        _SENT_BADGE_COLORS = {
            "BULLISH":        ("#dcfce7","#166534"),
            "MILDLY_BULLISH": ("#f0fdf4","#15803d"),
            "NEUTRAL":        ("#f1f5f9","#475569"),
            "MILDLY_BEARISH": ("#fff7ed","#9a3412"),
            "BEARISH":        ("#fef2f2","#991b1b"),
        }
        if not scores_df.empty:
            _sd = scores_df.copy()
            if "_fs_num" not in _sd.columns and "forward_score" in _sd.columns:
                _sd["_fs_num"] = pd.to_numeric(_sd["forward_score"], errors="coerce").fillna(0)
            _sd = _sd.sort_values("_fs_num", ascending=False)
            _show_cols = [c for c in ["ticker","forward_score","sentiment","beat_miss",
                                       "guidance_tone","key_themes","next_earnings",
                                       "news_count","tier"] if c in _sd.columns]
            _rename_nlp = {
                "ticker": "Ticker", "forward_score": "Score",
                "sentiment": "Sentiment", "beat_miss": "Beat/Miss",
                "guidance_tone": "Guidance", "key_themes": "Key Themes",
                "next_earnings": "Next Earnings", "news_count": "# News",
                "tier": "Scoring Tier",
            }
            st.dataframe(
                _sd[_show_cols].rename(columns=_rename_nlp),
                use_container_width=True, hide_index=True,
            )

    # ── Earnings Calendar ─────────────────────────────────────────────────────
    with nlp_tabs[2]:
        st.subheader("Earnings Calendar")
        if calendar_df.empty:
            st.info("No calendar data.")
        else:
            _cal = calendar_df.copy()
            _cal = _cal[_cal["next_earnings"].notna()].copy()
            if _cal.empty:
                st.info("No upcoming earnings dates found in last run.")
            else:
                st.dataframe(
                    _cal.rename(columns={"ticker": "Ticker", "next_earnings": "Next Earnings Date",
                                         "eps_surprise_pct": "EPS Surprise %"}),
                    use_container_width=True, hide_index=True,
                )
            st.caption("EPS Surprise % = (Actual EPS − Estimate) / |Estimate|. "
                       "Positive = beat, Negative = miss.")

    # ── How to Use ────────────────────────────────────────────────────────────
    with nlp_tabs[3]:
        st.subheader("How to Use the Earnings NLP Scorer")
        st.markdown("""
**File**: `canyon_final_v9_step65_earnings_nlp.py`  |  **Runtime**: ~20s (22 tickers)

### Tier 1 — Keyword (always works, no API key)
```bash
python canyon_final_v9_step65_earnings_nlp.py
python canyon_final_v9_step65_earnings_nlp.py --tickers AAPL MSFT NVDA
python canyon_final_v9_step65_earnings_nlp.py --days 7   # last 7 days only
```

### Tier 2 — GPT-4o-mini (requires OpenAI key)
```bash
export OPENAI_API_KEY="sk-your-key-here"
python canyon_final_v9_step65_earnings_nlp.py --gpt
```
GPT provides structured output: sentiment, guidance_tone, beat_miss, key_themes, risks, analyst_action.

### Output files
| File | Contents |
|---|---|
| `earnings_nlp_scores.csv` | Per-ticker: score, sentiment, themes, tier, reasoning |
| `earnings_calendar.csv` | Next earnings date + EPS surprise |
| `earnings_nlp_report.md` | Full markdown report |

### Score interpretation
| Range | Label | Action |
|---|---|---|
| > +0.20 | BULLISH | Strong positive momentum in news |
| +0.05 to +0.20 | MILDLY BULLISH | Cautiously positive |
| −0.05 to +0.05 | NEUTRAL | Mixed or no signal |
| −0.20 to −0.05 | MILDLY BEARISH | Negative tone, watch closely |
| < −0.20 | BEARISH | Significant negative news flow |

### Upgrade path
1. Add `OPENAI_API_KEY` → structured GPT analysis
2. Integrate Seeking Alpha / Refinitiv transcripts for full earnings call text
3. Fine-tune FinBERT on your specific equity universe for higher precision
""")
        if st.button("🔄 Re-run NLP Scorer", type="primary"):
            import subprocess
            _path = str(ROOT / "canyon_final_v9_step65_earnings_nlp.py")
            with st.spinner("Scoring earnings sentiment (~20s)..."):
                try:
                    _r = subprocess.run(["python3", _path], capture_output=True,
                                        text=True, timeout=120, cwd=str(ROOT))
                    if _r.returncode == 0:
                        st.success("✅ NLP scorer completed. Refresh tab to see updated results.")
                        if _r.stdout.strip():
                            st.code(_r.stdout[-2000:], language="")
                    else:
                        st.error(f"Error code {_r.returncode}")
                        st.code(_r.stderr[-1000:] if _r.stderr else "(no stderr)", language="")
                except Exception as _e:
                    st.error(str(_e))


def tab_ml_signals():  # noqa: C901
    """Step 66 — Walk-forward ML signal generator results viewer."""
    ic_df  = read_csv(FILES["ml_ic_comparison"])
    perf_df = read_csv(FILES["ml_backtest_perf"])
    fi_df  = read_csv(FILES["ml_feature_importance"])
    sum_df = read_csv(FILES["ml_summary"])
    scores_df = read_csv(FILES["ml_signal_scores"])

    # ── header badges ─────────────────────────────────────────────────────────
    _rf_ic = 0.0; _base_ic = 0.0; _alpha_str = "—"; _n_periods = "—"
    if not ic_df.empty and "signal" in ic_df.columns and "mean_ic" in ic_df.columns:
        _ic_num = pd.to_numeric(ic_df["mean_ic"], errors="coerce").fillna(0)
        _ml_mask  = ic_df["signal"].astype(str).isin(["rf_score","ensemble_score","ridge_score"])
        _base_mask = ~_ml_mask
        if _ml_mask.any():
            _rf_ic   = float(_ic_num[_ml_mask].max())
        if _base_mask.any():
            _base_ic = float(_ic_num[_base_mask].max())
    if not perf_df.empty and "ml_ret" in perf_df.columns:
        _n_periods = str(len(perf_df))
        _ml_rets   = pd.to_numeric(perf_df["ml_ret"],  errors="coerce").fillna(0)
        _spy_rets  = pd.to_numeric(perf_df["spy_ret"], errors="coerce").fillna(0)
        _total_ml  = float((1 + _ml_rets).prod() - 1)
        _total_spy = float((1 + _spy_rets).prod() - 1)
        _alpha_str = f"{(_total_ml - _total_spy)*100:+.1f}%"

    _ic_kind = "supportive" if _rf_ic > 0.10 else ("cyan" if _rf_ic > 0.05 else "risk")
    render_layer_workbench_header(
        "Step 66",
        "Walk-Forward ML Signal Generator",
        "Ridge + Random Forest trained on 10 price-derived features. Walk-forward, "
        "no look-ahead. Monthly rebalance, 252-day rolling train window.",
        [
            ("Best ML IC",        f"{_rf_ic:+.4f}", _ic_kind),
            ("Best Base IC",      f"{_base_ic:+.4f}", "cyan"),
            ("IC Improvement",    f"{(_rf_ic - _base_ic):+.4f}",
             "supportive" if _rf_ic > _base_ic else "risk"),
            ("ML Net Alpha",      _alpha_str,
             "supportive" if "+" in _alpha_str and _alpha_str != "—" else "risk"),
        ],
    )

    if ic_df.empty and perf_df.empty:
        st.info(
            "No ML results found. Run the engine first:\n\n"
            "```bash\ncd /Users/renjingru/Desktop/canyon_quant\n"
            "python canyon_final_v9_step66_ml_signals.py\n```\n\n"
            "Runtime: ~15 seconds."
        )
        return

    ml_tabs = st.tabs([
        "IC Comparison",
        "Feature Importance",
        "ML Performance",
        "Signal Scores",
        "How to Run",
    ])

    # ── Tab 1: IC Comparison ──────────────────────────────────────────────────
    with ml_tabs[0]:
        st.subheader("IC Comparison: ML Models vs Rule-Based Signals")
        st.caption(
            "Spearman IC = correlation between signal/model score at rebalance date T "
            "and forward 21-day return. Walk-forward, no look-ahead. "
            "Higher IC = stronger predictive signal."
        )
        if not ic_df.empty:
            IC_COLORS = {
                "STRONG":   ("#dcfce7","#166534"),
                "USABLE":   ("#dbeafe","#1e40af"),
                "WEAK":     ("#fef9c3","#854d0e"),
                "NEGATIVE": ("#fee2e2","#991b1b"),
                "NO_DATA":  ("#f1f5f9","#64748b"),
            }
            _ic_sorted = ic_df.copy()
            _ic_num = pd.to_numeric(_ic_sorted.get("mean_ic", pd.Series(dtype=float)), errors="coerce").fillna(0)
            _ic_sorted = _ic_sorted.assign(_n=_ic_num).sort_values("_n", ascending=False)

            if _PLOTLY:
                _ml_signals  = ["rf_score","ensemble_score","ridge_score"]
                _colors_bar  = [
                    "#0ea5e9" if str(r["signal"]) in _ml_signals else "#94a3b8"
                    for _, r in _ic_sorted.iterrows()
                ]
                _ic_vals = pd.to_numeric(_ic_sorted["mean_ic"], errors="coerce").fillna(0)
                _fig_ic = go.Figure(go.Bar(
                    x=_ic_sorted["signal"].astype(str).tolist(),
                    y=_ic_vals.tolist(),
                    marker_color=_colors_bar,
                    text=[f"{v:+.4f}" for v in _ic_vals],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>IC: %{y:+.4f}<extra></extra>",
                ))
                _fig_ic.add_hline(y=0.05, line_dash="dot", line_color="#16a34a",
                                  annotation_text="0.05 STRONG threshold")
                _fig_ic.add_hline(y=0,    line_dash="solid", line_color="#94a3b8", line_width=1)
                _fig_ic.update_layout(
                    height=340,
                    margin=dict(l=10, r=10, t=32, b=10),
                    title=dict(text="Mean IC by Signal (blue = ML model, gray = rule-based)",
                               font=dict(size=12), x=0),
                    xaxis=dict(tickangle=-30),
                    yaxis=dict(title="Mean IC", gridcolor="#f1f5f9"),
                    plot_bgcolor="white", paper_bgcolor="white",
                    showlegend=False,
                    font=dict(family="Inter,sans-serif", size=12),
                )
                st.plotly_chart(_fig_ic, use_container_width=True)

            # IC table
            _ic_rows = []
            for _, row in _ic_sorted.iterrows():
                bg, fg = IC_COLORS.get(str(row.get("status","")).upper(), ("#f9fafb","#374151"))
                is_ml = str(row.get("signal","")) in ["rf_score","ensemble_score","ridge_score"]
                label = "🤖 ML" if is_ml else "📐 Rule"
                _ic_rows.append({
                    "Type":    label,
                    "Signal":  str(row.get("signal","")),
                    "Mean IC": f"{float(row.get('mean_ic',0) or 0):+.4f}",
                    "t-stat":  f"{float(row.get('t_stat',0) or 0):+.2f}",
                    "p-value": str(row.get("p_value","")),
                    "IC+ Rate": str(row.get("ic_positive_pct","")),
                    "Status":  str(row.get("status","")),
                })
            st.dataframe(pd.DataFrame(_ic_rows), use_container_width=True, hide_index=True)

            if _rf_ic > 0 and _base_ic > 0:
                _improvement = _rf_ic - _base_ic
                _color = "#dcfce7" if _improvement > 0 else "#fee2e2"
                _icon  = "✅" if _improvement > 0.01 else ("⚠️" if _improvement > 0 else "❌")
                st.markdown(
                    f'<div style="background:{_color};border-radius:6px;padding:12px 16px;margin-top:8px">'
                    f'{_icon} ML best IC <b>{_rf_ic:+.4f}</b> vs best rule-based IC <b>{_base_ic:+.4f}</b> '
                    f'— improvement <b>{_improvement:+.4f}</b> '
                    f'({"ML adds statistically meaningful edge" if abs(_improvement) > 0.02 else "marginal improvement"})'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Tab 2: Feature Importance ─────────────────────────────────────────────
    with ml_tabs[1]:
        st.subheader("Feature Importance (Random Forest, full dataset)")
        if fi_df.empty:
            st.info("No feature importance data.")
        else:
            _fi = fi_df.copy()
            _fi_imp = pd.to_numeric(_fi.get("rf_importance", pd.Series(dtype=float)), errors="coerce").fillna(0)

            if _PLOTLY:
                _fig_fi = go.Figure(go.Bar(
                    x=_fi_imp.tolist(),
                    y=_fi["feature"].astype(str).tolist(),
                    orientation="h",
                    marker_color="#0ea5e9",
                    text=[f"{v:.3f}" for v in _fi_imp],
                    textposition="outside",
                    hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
                ))
                _fig_fi.update_layout(
                    height=max(280, len(_fi) * 36 + 60),
                    margin=dict(l=10, r=60, t=28, b=10),
                    title=dict(text="RF Feature Importance (higher = more predictive)",
                               font=dict(size=12), x=0),
                    xaxis=dict(title="Importance", gridcolor="#f1f5f9"),
                    yaxis=dict(autorange="reversed"),
                    plot_bgcolor="white", paper_bgcolor="white",
                    showlegend=False,
                    font=dict(family="Inter,sans-serif", size=12),
                )
                st.plotly_chart(_fig_fi, use_container_width=True)

            _fi_disp = _fi.copy()
            if "ridge_coef" in _fi_disp.columns:
                _fi_disp["Ridge Direction"] = pd.to_numeric(
                    _fi_disp["ridge_coef"], errors="coerce"
                ).apply(lambda v: "↑ Positive" if v > 0 else "↓ Negative" if v < 0 else "—")
            _show = [c for c in ["rf_rank","feature","rf_importance","Ridge Direction"] if c in _fi_disp.columns]
            _ren  = {"rf_rank": "Rank", "feature": "Feature",
                     "rf_importance": "RF Importance"}
            st.dataframe(_fi_disp[_show].rename(columns=_ren), use_container_width=True, hide_index=True)

            st.markdown("""
**Feature glossary**

| Feature | Description |
|---|---|
| `inv_vol` | 1 / 21-day vol — low-vol anomaly (note: NEGATIVE raw IC, but RF captures non-linear use) |
| `mom_12m_skip1m` | 12-month momentum skipping recent month — strongest linear predictor |
| `mom_6m` / `mom_3m` | Medium-term momentum |
| `rank_mom` / `rank_trend` | Cross-sectional rank (reduces outlier effect) |
| `trend_200` | Price vs 200-day MA — trend following |
| `spy_regime` | Market regime (SPY above/below 200-MA) — regime conditioning |
| `rsi_14` | Reversal signal — **NEGATIVE** IC, RF learns to down-weight |
""")

    # ── Tab 3: ML Performance ─────────────────────────────────────────────────
    with ml_tabs[2]:
        st.subheader("ML Portfolio vs SPY")
        if perf_df.empty:
            st.info("No backtest data.")
        else:
            _p = perf_df.copy()
            _ml_c  = pd.to_numeric(_p.get("ml_cum",  pd.Series(dtype=float)), errors="coerce").fillna(0) * 100
            _spy_c = pd.to_numeric(_p.get("bench_cum",pd.Series(dtype=float)), errors="coerce").fillna(0) * 100
            _x     = _p["period_end"].astype(str).tolist() if "period_end" in _p.columns else list(range(len(_p)))

            c1, c2, c3, c4 = st.columns(4)
            _ml_rets  = pd.to_numeric(_p["ml_ret"],  errors="coerce").fillna(0)
            _spy_rets = pd.to_numeric(_p["spy_ret"], errors="coerce").fillna(0)
            _total_ml  = float((1 + _ml_rets).prod() - 1)
            _total_spy = float((1 + _spy_rets).prod() - 1)
            _sharpe = float(_ml_rets.mean() / (_ml_rets.std() + 1e-10) * np.sqrt(12))
            c1.metric("ML Total Return",   f"{_total_ml*100:.2f}%")
            c2.metric("SPY Total Return",  f"{_total_spy*100:.2f}%")
            c3.metric("Net Alpha",         f"{(_total_ml - _total_spy)*100:+.2f}%")
            c4.metric("Annl. Sharpe",      f"{_sharpe:.2f}")

            if _PLOTLY:
                _fig_p = go.Figure()
                _fig_p.add_trace(go.Scatter(
                    x=_x, y=_ml_c.tolist(), mode="lines",
                    name="ML Portfolio", line=dict(color="#0ea5e9", width=2.5),
                    hovertemplate="<b>ML</b><br>%{x}<br>%{y:.2f}%<extra></extra>",
                ))
                _fig_p.add_trace(go.Scatter(
                    x=_x, y=_spy_c.tolist(), mode="lines",
                    name="SPY", line=dict(color="#94a3b8", width=1.8, dash="dot"),
                    hovertemplate="<b>SPY</b><br>%{x}<br>%{y:.2f}%<extra></extra>",
                ))
                _fig_p.add_hline(y=0, line_dash="solid", line_color="#d1d5db", line_width=1)
                _fig_p.update_layout(
                    height=380,
                    margin=dict(l=10, r=10, t=32, b=10),
                    title=dict(text="ML Portfolio Cumulative Return vs SPY (%)", font=dict(size=13), x=0),
                    xaxis=dict(gridcolor="#f1f5f9", tickangle=-30),
                    yaxis=dict(title="Cumulative Return (%)", gridcolor="#f1f5f9"),
                    plot_bgcolor="white", paper_bgcolor="white",
                    legend=dict(orientation="h", y=1.06),
                    font=dict(family="Inter,sans-serif", size=12),
                )
                st.plotly_chart(_fig_p, use_container_width=True)

            # Formatted table
            _show = [c for c in ["rebalance_date","ml_ret","spy_ret","alpha",
                                  "ml_cum","bench_cum","n_held","turnover_pct"] if c in _p.columns]
            _p_disp = _p[_show].copy()
            for _c in ["ml_ret","spy_ret","alpha","ml_cum","bench_cum"]:
                if _c in _p_disp.columns:
                    _p_disp[_c] = pd.to_numeric(_p_disp[_c], errors="coerce").apply(
                        lambda v: f"{v*100:+.2f}%" if pd.notna(v) else ""
                    )
            _ren_p = {"rebalance_date":"Rebalance","ml_ret":"ML Ret","spy_ret":"SPY",
                      "alpha":"Alpha","ml_cum":"ML Cum.","bench_cum":"SPY Cum.",
                      "n_held":"# Held","turnover_pct":"Turnover %"}
            st.dataframe(_p_disp.rename(columns=_ren_p), use_container_width=True, hide_index=True)

            st.warning(
                "⚠ **Survivorship bias & overfitting risk**: universe = current tickers. "
                "ML models trained on historical patterns may not persist. "
                "Out-of-sample live performance will be lower than backtest."
            )

    # ── Tab 4: Signal Scores ──────────────────────────────────────────────────
    with ml_tabs[3]:
        st.subheader("Latest ML Signal Scores")
        if scores_df.empty:
            st.info("No score data.")
        else:
            _last_date = scores_df["rebalance_date"].astype(str).max() if "rebalance_date" in scores_df.columns else "?"
            st.caption(f"Showing rebalance date: **{_last_date}** — "
                       "higher ensemble_score = stronger ML buy signal")
            _last = scores_df[scores_df["rebalance_date"].astype(str) == _last_date].copy()
            if "ensemble_score" in _last.columns:
                _last["_e"] = pd.to_numeric(_last["ensemble_score"], errors="coerce").fillna(0)
                _last = _last.sort_values("_e", ascending=False)
            _sc = [c for c in ["ticker","ensemble_score","ridge_score","rf_score","n_train"]
                   if c in _last.columns]
            _ren_sc = {"ticker":"Ticker","ensemble_score":"Ensemble","ridge_score":"Ridge",
                       "rf_score":"RF","n_train":"Train N"}
            st.dataframe(_last[_sc].rename(columns=_ren_sc), use_container_width=True, hide_index=True)

    # ── Tab 5: How to Run ─────────────────────────────────────────────────────
    with ml_tabs[4]:
        st.subheader("How to Run the ML Signal Generator")
        st.markdown("""
**File**: `canyon_final_v9_step66_ml_signals.py`  |  **Runtime**: ~15 seconds

```bash
# Default (252-day lookback, top-8, 10bps TC)
python canyon_final_v9_step66_ml_signals.py

# 2-year training window
python canyon_final_v9_step66_ml_signals.py --lookback 504

# Different portfolio size
python canyon_final_v9_step66_ml_signals.py --top 10
```

**Models:**
| Model | Type | Regularisation |
|---|---|---|
| Ridge | Linear regression | L2 (α=50) |
| Random Forest | 80 trees, depth≤4 | min_samples_leaf=10 |
| Ensemble | Ridge + RF average | — |

**Walk-forward training:**
- Train window: 252 trading days (rolling)
- Predict: next rebalance date (out-of-sample)
- No look-ahead: all features `.shift(1)`, targets use forward returns

**10 features:**
`mom_1m`, `mom_3m`, `mom_6m`, `mom_12m_skip1m`, `trend_200`, `rsi_14`, `inv_vol`,
`rank_mom` (cross-sectional rank), `rank_trend`, `spy_regime` (market state)

**Upgrade path:**
- Add fundamental features: P/E ratio, earnings revision, analyst upgrades
- Add macro features: VIX, yield curve slope, credit spreads
- Try LightGBM / XGBoost for stronger gradient boosting
- Add SHAP explainability for per-prediction feature attribution
""")
        if st.button("🔄 Re-run ML Engine", type="primary"):
            import subprocess
            _path = str(ROOT / "canyon_final_v9_step66_ml_signals.py")
            with st.spinner("Training ML models (~15s)..."):
                try:
                    _r = subprocess.run(["python3", _path], capture_output=True,
                                        text=True, timeout=180, cwd=str(ROOT))
                    if _r.returncode == 0:
                        st.success("✅ ML engine completed. Refresh tab to see updated results.")
                        if _r.stdout.strip():
                            st.code(_r.stdout[-3000:], language="")
                    else:
                        st.error(f"Error code {_r.returncode}")
                        st.code(_r.stderr[-1000:] if _r.stderr else "(no stderr)", language="")
                except Exception as _e:
                    st.error(str(_e))


def tab_backtest():  # noqa: C901
    """Walk-forward backtest & signal IC validation viewer (Step 62 engine output)."""
    ic_df       = read_csv(FILES["backtest_signal_ic"])
    monthly_df  = read_csv(FILES["backtest_monthly_perf"])
    summary_df  = read_csv(FILES["backtest_summary"])
    holdings_df = read_csv(FILES["backtest_holdings"])

    # ── header badges ────────────────────────────────────────────────────────
    _strong_ic = 0
    _total_alpha_str = "—"
    _sharpe_str = "—"
    _periods = "—"
    if not summary_df.empty and "metric" in summary_df.columns:
        _sm = dict(zip(summary_df["metric"].astype(str), summary_df["value"].astype(str)))
        _strong_ic     = int(_sm.get("Strong IC Signals", "0").split("/")[0].strip() or 0)
        _total_alpha_str = _sm.get("Total Alpha vs SPY", "—")
        _sharpe_str    = _sm.get("Annualised Sharpe", "—")
        _periods       = _sm.get("Periods Tested (months)", "—")

    _alpha_kind = "supportive" if _strong_ic >= 2 else ("cyan" if _strong_ic == 1 else "risk")
    render_layer_workbench_header(
        "Step 62",
        "Walk-Forward Backtest & Signal IC Engine",
        "No-look-ahead walk-forward backtest. Signals computed at T close, positions held T+1. "
        "Monthly rebalance, inverse-vol weights (25% cap), 10 bps/trade transaction cost.",
        [
            ("Strong IC Signals", f"{_strong_ic} / 7",  _alpha_kind),
            ("Total Alpha vs SPY", _total_alpha_str,    "supportive" if "+" in _total_alpha_str or
             (not _total_alpha_str.startswith("-") and _total_alpha_str != "—") else "risk"),
            ("Annualised Sharpe",  _sharpe_str,         "supportive" if any(c.isdigit() and float(_sharpe_str.replace("%","")) > 1.0
             for c in [_sharpe_str[0]] if _sharpe_str not in ("—",)) else "cyan"),
            ("Months Tested",      _periods,            "supportive" if _periods not in ("—",) and int(_periods) >= 36 else "cyan"),
        ],
    )

    # ── no-data state ─────────────────────────────────────────────────────────
    if ic_df.empty and monthly_df.empty and summary_df.empty:
        st.info(
            "Backtest outputs not found. Run the engine first:\n\n"
            "```bash\n"
            "cd /Users/renjingru/Desktop/canyon_quant\n"
            "python canyon_final_v9_step62_backtest_engine.py\n"
            "```\n\n"
            "Takes ~10 seconds. Outputs: backtest_signal_ic.csv, backtest_monthly_perf.csv, "
            "backtest_summary.csv, backtest_holdings.csv, backtest_engine_report.md"
        )
        return

    bt_tabs = st.tabs([
        "Summary",
        "Signal IC",
        "Cumulative Performance",
        "Monthly Returns",
        "Holdings",
        "How to Run",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Summary
    # ══════════════════════════════════════════════════════════════════════════
    with bt_tabs[0]:
        st.subheader("Backtest Summary")

        if summary_df.empty:
            st.info("No summary data. Run backtest engine first.")
        else:
            # KPI row — pull key metrics
            _sm = {}
            if "metric" in summary_df.columns and "value" in summary_df.columns:
                _sm = dict(zip(summary_df["metric"].astype(str), summary_df["value"].astype(str)))

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Return (Strategy)", _sm.get("Total Return (Strategy)", "—"))
            k2.metric("Total Alpha vs SPY",       _sm.get("Total Alpha vs SPY", "—"))
            k3.metric("Annualised Sharpe",         _sm.get("Annualised Sharpe", "—"))
            k4.metric("Max Drawdown",              _sm.get("Max Drawdown", "—"))

            k5, k6, k7, k8 = st.columns(4)
            k5.metric("Annualised Sortino",  _sm.get("Annualised Sortino", "—"))
            k6.metric("Calmar Ratio",        _sm.get("Calmar Ratio", "—"))
            k7.metric("Monthly Win vs SPY",  _sm.get("Monthly Win Rate vs SPY", "—"))
            k8.metric("Transaction Cost",    _sm.get("Transaction Cost (total)", "—"))

            st.markdown("---")

            # Assessment colour map
            def _assessment_color(a: str) -> str:
                a = str(a).upper()
                if a in ("STRONG",):    return "background-color:#dcfce7;color:#166534"
                if a in ("USABLE",):    return "background-color:#dbeafe;color:#1e40af"
                if a in ("WEAK",):      return "background-color:#fef9c3;color:#854d0e"
                if a in ("PLAIN",):     return ""
                return "background-color:#fee2e2;color:#991b1b"

            _rows = []
            for _, row in summary_df.iterrows():
                _rows.append({
                    "Metric":     str(row.get("metric", "")),
                    "Strategy":   str(row.get("value", "")),
                    "Benchmark":  str(row.get("benchmark", "")),
                    "Assessment": str(row.get("assessment", "")),
                })
            _sum_tbl = pd.DataFrame(_rows)

            def _style_assessment(val):
                return _assessment_color(val)

            st.dataframe(
                _sum_tbl.style.map(_style_assessment, subset=["Assessment"]),
                use_container_width=True,
                hide_index=True,
            )

            # Survivorship bias warning
            st.warning(
                "⚠ **Survivorship bias**: backtest uses current universe (stocks still trading). "
                "Actual live performance would be lower — delisted/bankrupt names are excluded. "
                "Returns are directional evidence, not a live forecast."
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Signal IC
    # ══════════════════════════════════════════════════════════════════════════
    with bt_tabs[1]:
        st.subheader("Signal Information Coefficient (IC) — Spearman Rank")
        st.caption(
            "IC = Spearman rank correlation between signal value at rebalance date and "
            "forward 21-day return. IC > 0.05 with |t| > 2.0 = tradeable. "
            "Computed across all rebalance dates (walk-forward, no look-ahead)."
        )

        if ic_df.empty:
            st.info("No IC data. Run backtest engine first.")
        else:
            IC_STATUS_COLORS = {
                "STRONG":   ("#dcfce7", "#166534"),
                "USABLE":   ("#dbeafe", "#1e40af"),
                "WEAK":     ("#fef9c3", "#854d0e"),
                "NEGATIVE": ("#fee2e2", "#991b1b"),
            }

            def _ic_row_html(r):
                status = str(r.get("status", "")).upper()
                bg, fg = IC_STATUS_COLORS.get(status, ("#f9fafb", "#374151"))
                mean_ic  = float(r.get("mean_ic",  0) or 0)
                t_stat   = float(r.get("t_stat",   0) or 0)
                p_value  = float(r.get("p_value",  1) or 1)
                n_obs    = int(r.get("n_obs",      0) or 0)
                ic_pos   = str(r.get("ic_positive_pct", ""))
                signal   = str(r.get("signal", ""))

                # IC bar (visual width 0..100%)
                bar_pct = min(100, max(0, abs(mean_ic) * 500))  # 0.10 IC → 50% bar
                bar_col = "#16a34a" if mean_ic > 0 else "#dc2626"
                bar_html = (
                    f'<div style="width:{bar_pct:.0f}%;height:8px;'
                    f'background:{bar_col};border-radius:3px;margin-top:3px;"></div>'
                )
                return (
                    f'<tr style="background:{bg};color:{fg}">'
                    f'<td style="padding:8px 10px;font-weight:600">{signal}</td>'
                    f'<td style="padding:8px 10px;text-align:center">{mean_ic:+.4f}{bar_html}</td>'
                    f'<td style="padding:8px 10px;text-align:center">{t_stat:+.2f}</td>'
                    f'<td style="padding:8px 10px;text-align:center">{p_value:.4f}</td>'
                    f'<td style="padding:8px 10px;text-align:center">{n_obs}</td>'
                    f'<td style="padding:8px 10px;text-align:center">{ic_pos}</td>'
                    f'<td style="padding:8px 10px;text-align:center;font-weight:700">'
                    f'<span style="background:{bg};border:1px solid {fg};border-radius:4px;'
                    f'padding:2px 8px">{status}</span></td>'
                    f'</tr>'
                )

            ic_rows_html = "".join(_ic_row_html(row) for _, row in ic_df.iterrows())
            st.markdown(
                f"""<table style="width:100%;border-collapse:collapse;
                    font-family:Inter,sans-serif;font-size:13px">
                  <thead>
                    <tr style="background:#f1f5f9;color:#64748b">
                      <th style="padding:8px 10px;text-align:left">Signal</th>
                      <th style="padding:8px 10px">Mean IC</th>
                      <th style="padding:8px 10px">t-stat</th>
                      <th style="padding:8px 10px">p-value</th>
                      <th style="padding:8px 10px">N Obs</th>
                      <th style="padding:8px 10px">IC+ Rate</th>
                      <th style="padding:8px 10px">Status</th>
                    </tr>
                  </thead>
                  <tbody>{ic_rows_html}</tbody>
                </table>""",
                unsafe_allow_html=True,
            )

            st.markdown("---")
            st.markdown("""
**Interpretation guide**

| Status | IC threshold | |t|-stat | Meaning |
|---|---|---|---|
| **STRONG** | IC > 0.05 | > 2.0 | Statistically significant alpha — use in portfolio construction |
| **USABLE** | IC > 0.03 | > 2.0 | Marginal signal — combine with others, watch decay |
| **WEAK** | IC < 0.03 | < 2.0 | No statistical evidence — noise |
| **NEGATIVE** | IC < 0 | any | Contrarian or harmful — do not include |

**Current tradeable signals**: mom_12m_skip1m (STRONG), trend_200 (USABLE), mom_6m (USABLE)
""")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Cumulative Performance Chart
    # ══════════════════════════════════════════════════════════════════════════
    with bt_tabs[2]:
        st.subheader("Cumulative Return: Strategy vs SPY")

        if monthly_df.empty:
            st.info("No monthly performance data. Run backtest engine first.")
        else:
            _perf = monthly_df.copy()
            # Ensure cumulative columns exist
            if "strategy_cum" not in _perf.columns and "strategy_ret" in _perf.columns:
                _perf["strategy_cum"] = (1 + pd.to_numeric(_perf["strategy_ret"], errors="coerce").fillna(0)).cumprod() - 1
            if "bench_cum" not in _perf.columns and "spy_ret" in _perf.columns:
                _perf["bench_cum"] = (1 + pd.to_numeric(_perf["spy_ret"], errors="coerce").fillna(0)).cumprod() - 1

            _perf["strategy_cum_pct"] = pd.to_numeric(_perf.get("strategy_cum", 0), errors="coerce") * 100
            _perf["bench_cum_pct"]    = pd.to_numeric(_perf.get("bench_cum", 0),    errors="coerce") * 100

            # Date axis
            _x = _perf["period_end"].astype(str).tolist() if "period_end" in _perf.columns else list(range(len(_perf)))

            if _PLOTLY:
                _fig = go.Figure()
                _fig.add_trace(go.Scatter(
                    x=_x, y=_perf["strategy_cum_pct"].tolist(),
                    mode="lines", name="Strategy",
                    line=dict(color="#0ea5e9", width=2.5),
                    hovertemplate="<b>Strategy</b><br>%{x}<br>Cumulative: %{y:.2f}%<extra></extra>",
                ))
                _fig.add_trace(go.Scatter(
                    x=_x, y=_perf["bench_cum_pct"].tolist(),
                    mode="lines", name="SPY",
                    line=dict(color="#94a3b8", width=1.8, dash="dot"),
                    hovertemplate="<b>SPY</b><br>%{x}<br>Cumulative: %{y:.2f}%<extra></extra>",
                ))
                # Alpha fill
                _alpha_vals = (_perf["strategy_cum_pct"] - _perf["bench_cum_pct"]).tolist()
                _fig.add_trace(go.Scatter(
                    x=_x + _x[::-1],
                    y=_perf["strategy_cum_pct"].tolist() + _perf["bench_cum_pct"].tolist()[::-1],
                    fill="toself",
                    fillcolor="rgba(14,165,233,0.10)",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    name="Alpha fill",
                ))
                _fig.add_hline(y=0, line_dash="solid", line_color="#d1d5db", line_width=1)
                _fig.update_layout(
                    height=420,
                    margin=dict(l=10, r=10, t=32, b=10),
                    title=dict(text="Walk-Forward Cumulative Return (%)", font=dict(size=13), x=0),
                    xaxis=dict(title="", gridcolor="#f1f5f9", tickangle=-30),
                    yaxis=dict(title="Cumulative Return (%)", gridcolor="#f1f5f9"),
                    plot_bgcolor="white", paper_bgcolor="white",
                    legend=dict(orientation="h", y=1.06, x=0),
                    font=dict(family="Inter,sans-serif", size=12),
                )
                st.plotly_chart(_fig, use_container_width=True)

                # Alpha bar chart
                if "alpha" in _perf.columns:
                    _perf["alpha_pct"] = pd.to_numeric(_perf["alpha"], errors="coerce") * 100
                    _colors = ["#16a34a" if v >= 0 else "#dc2626" for v in _perf["alpha_pct"].fillna(0)]
                    _fig2 = go.Figure(go.Bar(
                        x=_x, y=_perf["alpha_pct"].tolist(),
                        marker_color=_colors,
                        name="Monthly Alpha",
                        hovertemplate="<b>Alpha</b><br>%{x}<br>%{y:.2f}%<extra></extra>",
                    ))
                    _fig2.add_hline(y=0, line_dash="solid", line_color="#94a3b8", line_width=1)
                    _fig2.update_layout(
                        height=200,
                        margin=dict(l=10, r=10, t=28, b=10),
                        title=dict(text="Monthly Alpha (Strategy − SPY)", font=dict(size=12), x=0),
                        xaxis=dict(gridcolor="#f1f5f9", tickangle=-30, showticklabels=False),
                        yaxis=dict(title="Alpha (%)", gridcolor="#f1f5f9"),
                        plot_bgcolor="white", paper_bgcolor="white",
                        font=dict(family="Inter,sans-serif", size=11),
                        showlegend=False,
                    )
                    st.plotly_chart(_fig2, use_container_width=True)
            else:
                st.line_chart(_perf.set_index("period_end")[["strategy_cum_pct", "bench_cum_pct"]] if "period_end" in _perf.columns else _perf[["strategy_cum_pct", "bench_cum_pct"]])

            # Key stats row
            _final_strat = _perf["strategy_cum_pct"].iloc[-1] if not _perf.empty else 0.0
            _final_spy   = _perf["bench_cum_pct"].iloc[-1]    if not _perf.empty else 0.0
            _final_alpha = _final_strat - _final_spy
            _win_months  = int((pd.to_numeric(_perf["alpha"], errors="coerce").fillna(0) > 0).sum()) if "alpha" in _perf.columns else 0
            _total_months = len(_perf)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Final Strategy Return", f"{_final_strat:.2f}%")
            c2.metric("Final SPY Return",       f"{_final_spy:.2f}%")
            c3.metric("Net Alpha",              f"{_final_alpha:+.2f}%")
            c4.metric("Months Beat SPY",        f"{_win_months} / {_total_months}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — Monthly Returns Table
    # ══════════════════════════════════════════════════════════════════════════
    with bt_tabs[3]:
        st.subheader("Monthly Return Detail")

        if monthly_df.empty:
            st.info("No monthly data. Run backtest engine first.")
        else:
            _m = monthly_df.copy()
            # Format for display
            _pct_cols = ["strategy_ret", "spy_ret", "alpha"]
            for _c in _pct_cols:
                if _c in _m.columns:
                    _m[_c] = pd.to_numeric(_m[_c], errors="coerce").apply(
                        lambda v: f"{v*100:+.2f}%" if pd.notna(v) else ""
                    )
            if "strategy_cum" in _m.columns:
                _m["strategy_cum"] = pd.to_numeric(_m["strategy_cum"], errors="coerce").apply(
                    lambda v: f"{v*100:.2f}%" if pd.notna(v) else ""
                )
            if "bench_cum" in _m.columns:
                _m["bench_cum"] = pd.to_numeric(_m["bench_cum"], errors="coerce").apply(
                    lambda v: f"{v*100:.2f}%" if pd.notna(v) else ""
                )
            if "turnover_pct" in _m.columns:
                _m["turnover_pct"] = pd.to_numeric(_m["turnover_pct"], errors="coerce").apply(
                    lambda v: f"{v:.1f}%" if pd.notna(v) else ""
                )
            if "tc_cost_bps" in _m.columns:
                _m["tc_cost_bps"] = pd.to_numeric(_m["tc_cost_bps"], errors="coerce").apply(
                    lambda v: f"{v:.1f} bps" if pd.notna(v) else ""
                )

            _display_cols = [c for c in [
                "rebalance_date", "period_end",
                "strategy_ret", "spy_ret", "alpha",
                "strategy_cum", "bench_cum",
                "n_held", "turnover_pct", "tc_cost_bps",
            ] if c in _m.columns]
            _rename = {
                "rebalance_date": "Rebalance",
                "period_end":     "Period End",
                "strategy_ret":   "Strategy",
                "spy_ret":        "SPY",
                "alpha":          "Alpha",
                "strategy_cum":   "Strategy Cum.",
                "bench_cum":      "SPY Cum.",
                "n_held":         "# Held",
                "turnover_pct":   "Turnover",
                "tc_cost_bps":    "TC Cost",
            }
            st.dataframe(
                _m[_display_cols].rename(columns=_rename),
                use_container_width=True,
                hide_index=True,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — Last Rebalance Holdings
    # ══════════════════════════════════════════════════════════════════════════
    with bt_tabs[4]:
        st.subheader("Last Rebalance Holdings")

        if holdings_df.empty:
            st.info("No holdings data. Run backtest engine first.")
        else:
            _last_date = holdings_df["rebalance_date"].astype(str).max() if "rebalance_date" in holdings_df.columns else "?"
            st.caption(f"Rebalance date: **{_last_date}** — inverse-volatility weighted, 25% cap per position")

            _h = holdings_df[holdings_df["rebalance_date"].astype(str) == _last_date].copy() if "rebalance_date" in holdings_df.columns else holdings_df.copy()

            if "weight" in _h.columns:
                _h["weight_pct"] = pd.to_numeric(_h["weight"], errors="coerce") * 100
            if "composite_score" in _h.columns:
                _h["composite_score"] = pd.to_numeric(_h["composite_score"], errors="coerce").round(4)

            _h_display = _h[[c for c in ["ticker", "weight_pct", "composite_score"] if c in _h.columns]].copy()
            _h_display.columns = ["Ticker", "Weight %", "Composite Score"][:len(_h_display.columns)]

            if _PLOTLY and not _h.empty and "weight_pct" in _h.columns and "ticker" in _h.columns:
                _fig_h = go.Figure(go.Bar(
                    x=_h["ticker"].astype(str).tolist(),
                    y=_h["weight_pct"].tolist(),
                    marker_color="#0ea5e9",
                    text=[f"{v:.1f}%" for v in _h["weight_pct"].fillna(0)],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>Weight: %{y:.2f}%<extra></extra>",
                ))
                _fig_h.update_layout(
                    height=280,
                    margin=dict(l=10, r=10, t=28, b=10),
                    title=dict(text="Position Weights (%)", font=dict(size=12), x=0),
                    xaxis=dict(tickangle=0),
                    yaxis=dict(title="Weight %", gridcolor="#f1f5f9"),
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Inter,sans-serif", size=12),
                )
                st.plotly_chart(_fig_h, use_container_width=True)

            st.dataframe(_h_display.sort_values("Weight %", ascending=False), use_container_width=True, hide_index=True)

            st.markdown("""
**Weight methodology** — Inverse-volatility weighting:
1. Compute 21-day realized volatility for each ticker at rebalance
2. Weight = 1 / vol, normalized to sum to 100%
3. Cap each position at 25%
4. Low-vol defensives (JNJ, WMT, KO) get higher weights than high-vol growth names
""")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 — How to Run
    # ══════════════════════════════════════════════════════════════════════════
    with bt_tabs[5]:
        st.subheader("How to Run the Backtest Engine")

        st.markdown("""
### Step 62: Walk-Forward Backtest Engine

**File**: `canyon_final_v9_step62_backtest_engine.py`

**Runtime**: ~10 seconds (downloads 5 years of price data, caches for 12 hours)

#### Basic run
```bash
cd /Users/renjingru/Desktop/canyon_quant
python canyon_final_v9_step62_backtest_engine.py
```

#### Options
```bash
# Test longer history (if data available)
python canyon_final_v9_step62_backtest_engine.py --years 7

# Change top-N holdings per period (default: 8)
python canyon_final_v9_step62_backtest_engine.py --top 10

# Change transaction cost assumption (default: 10 bps)
python canyon_final_v9_step62_backtest_engine.py --tc 15

# Combined
python canyon_final_v9_step62_backtest_engine.py --years 5 --top 8 --tc 10
```

#### Output files (auto-written to canyon_quant folder)
| File | Contents |
|---|---|
| `backtest_signal_ic.csv` | IC per signal: mean IC, t-stat, p-value, status |
| `backtest_monthly_perf.csv` | Monthly strategy vs SPY returns, alpha, turnover |
| `backtest_summary.csv` | Summary stats: Sharpe, Sortino, Calmar, alpha |
| `backtest_holdings.csv` | Holdings at each rebalance date with weights |
| `backtest_engine_report.md` | Full markdown report |
| `backtest_price_cache.csv` | 12-hour price cache (auto-refreshes) |

#### Refresh data
Delete `backtest_price_cache.csv` to force a fresh yfinance download.

#### What's validated
- **No look-ahead bias** — signals computed with `.shift(1)` at day T close, held from T+1
- **Walk-forward** — no future data used to choose signals
- **Transaction costs** — 10 bps per one-way trade applied on turnover
- **Survivorship bias** — noted: current universe only, overstates true live performance

#### Signal hierarchy (by IC)
1. `mom_12m_skip1m` — 12-month momentum, skip most recent month (IC=0.0602, **STRONG**)
2. `trend_200` — price vs 200-day MA (IC=0.0496, USABLE)
3. `mom_6m` — 6-month momentum (IC=0.0411, USABLE)
4. `rsi_rev` — RSI reversal (IC=0.0046, WEAK — avoid)
5. `inv_vol` — inverse volatility (IC=−0.0923, NEGATIVE — not alpha; use for sizing only)
""")

        # Re-run button
        st.markdown("---")
        if st.button("🔄 Re-run Backtest Engine Now", type="primary"):
            import subprocess
            _engine_path = str(ROOT / "canyon_final_v9_step62_backtest_engine.py")
            with st.spinner("Running backtest engine (~10s)..."):
                try:
                    _result = subprocess.run(
                        ["python", _engine_path],
                        capture_output=True, text=True, timeout=120,
                        cwd=str(ROOT),
                    )
                    if _result.returncode == 0:
                        st.success("✅ Backtest engine completed. Refresh this tab to see updated results.")
                        if _result.stdout.strip():
                            st.code(_result.stdout[-3000:], language="")
                    else:
                        st.error(f"Engine returned error code {_result.returncode}")
                        st.code(_result.stderr[-2000:] if _result.stderr else "(no stderr)", language="")
                except subprocess.TimeoutExpired:
                    st.error("Engine timed out after 120 seconds.")
                except Exception as _e:
                    st.error(f"Could not launch engine: {_e}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 67 — SHAP Explainability
# ─────────────────────────────────────────────────────────────────────────────
def tab_shap_explainer():  # noqa: C901
    """Step 67 — SHAP feature attribution for ML signal models."""
    st.subheader("SHAP Feature Attribution")
    st.caption("Which features drove each ML prediction?  (Step 67 — canyon_final_v9_step67_shap_explainer.py)")

    _shap_sum_path  = FILES.get("shap_summary")
    _shap_tick_path = FILES.get("shap_per_ticker")
    _shap_rf_path   = FILES.get("shap_values_rf")
    _shap_rep_path  = FILES.get("shap_report")

    s67, s68, s69 = st.tabs(["Global Importance", "Per-Ticker Drivers", "How to Run"])

    # ── tab 1: global mean |SHAP| bar chart ──────────────────────────────────
    with s67:
        st.subheader("Global Feature Importance (mean |SHAP|)")
        if _shap_sum_path and Path(_shap_sum_path).exists():
            _df = pd.read_csv(_shap_sum_path)
            if "mean_abs_shap" in _df.columns and "feature" in _df.columns:
                _df = _df.sort_values("mean_abs_shap", ascending=True)
                fig = go.Figure(go.Bar(
                    x=_df["mean_abs_shap"],
                    y=_df["feature"],
                    orientation="h",
                    marker_color="steelblue",
                ))
                fig.update_layout(
                    title="Mean |SHAP| — Random Forest",
                    xaxis_title="Mean |SHAP value|",
                    yaxis_title="Feature",
                    height=420,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(color="#111"),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(_df, use_container_width=True)
        else:
            st.info("No SHAP summary data yet. Run: `python3 canyon_final_v9_step67_shap_explainer.py`")

        # Ridge SHAP if available
        if _shap_rf_path and Path(_shap_rf_path).exists():
            with st.expander("Raw SHAP values (RF) — full table"):
                _raw = pd.read_csv(_shap_rf_path)
                st.dataframe(_raw, use_container_width=True)

    # ── tab 2: per-ticker explanation ────────────────────────────────────────
    with s68:
        st.subheader("Per-Ticker Top Drivers")
        if _shap_tick_path and Path(_shap_tick_path).exists():
            _pt = pd.read_csv(_shap_tick_path)
            # colour rows by net direction
            def _shap_colour(val):
                try:
                    v = float(val)
                    if v > 0.02:   return "color:#1a7a4a"
                    if v < -0.02:  return "color:#b00020"
                    return ""
                except Exception:
                    return ""

            _show_cols = [c for c in ["ticker", "ml_score", "top_driver_1", "shap_1",
                                       "top_driver_2", "shap_2", "top_driver_3", "shap_3",
                                       "positive_drivers", "negative_drivers"] if c in _pt.columns]
            _disp = _pt[_show_cols].copy() if _show_cols else _pt.copy()

            _num_cols = [c for c in ["shap_1", "shap_2", "shap_3"] if c in _disp.columns]
            if _num_cols:
                for _nc in _num_cols:
                    _disp[_nc] = pd.to_numeric(_disp[_nc], errors="coerce").round(4)
                _styled = _disp.style.map(_shap_colour, subset=_num_cols)
            else:
                _styled = _disp.style

            st.dataframe(_styled, use_container_width=True)

            # highlight top 5
            if "ml_score" in _pt.columns and "top_driver_1" in _pt.columns:
                st.markdown("##### Top 5 tickers by ML score")
                _top5 = _pt.nlargest(5, "ml_score")[["ticker", "ml_score", "top_driver_1", "shap_1",
                                                       "top_driver_2", "shap_2"]] \
                           if "ml_score" in _pt.columns else _pt.head(5)
                st.dataframe(_top5, use_container_width=True, hide_index=True)
        else:
            st.info("No per-ticker SHAP data. Run the SHAP engine first.")

        # report
        if _shap_rep_path and Path(_shap_rep_path).exists():
            with st.expander("SHAP Report"):
                st.markdown(Path(_shap_rep_path).read_text())

    # ── tab 3: how to run ────────────────────────────────────────────────────
    with s69:
        st.subheader("How to Run the SHAP Explainer")
        st.markdown("""
**Requirements:** `pip install shap` (0.41+) — already available.

```bash
cd ~/Desktop/canyon_quant
# Explain latest walk-forward predictions (uses step66 ML output)
python3 canyon_final_v9_step67_shap_explainer.py

# Explain a specific date
python3 canyon_final_v9_step67_shap_explainer.py --date 2024-01-31

# Show detail for one ticker
python3 canyon_final_v9_step67_shap_explainer.py --ticker NVDA
```

**Outputs written to** `~/Desktop/canyon_quant/`:
| File | Contents |
|------|----------|
| `shap_values_rf.csv` | Full SHAP matrix, RF model |
| `shap_values_ridge.csv` | Full SHAP matrix, Ridge model |
| `shap_summary.csv` | Mean |SHAP| per feature (global importance) |
| `shap_per_ticker.csv` | Top 3 drivers + SHAP signs per ticker |
| `shap_report.md` | Human-readable summary |

**Typical runtime:** ~2 seconds
""")
        if st.button("▶ Run SHAP Explainer", key="run_shap67"):
            import subprocess
            _script = Path(__file__).parent / "canyon_final_v9_step67_shap_explainer.py"
            with st.spinner("Running SHAP explainer…"):
                try:
                    _r = subprocess.run(
                        ["python3", str(_script)],
                        capture_output=True, text=True, timeout=120,
                        cwd=str(Path(__file__).parent),
                    )
                    if _r.returncode == 0:
                        st.success("SHAP engine completed. Refresh tab to see updated charts.")
                        st.code(_r.stdout[-2000:] if len(_r.stdout) > 2000 else _r.stdout)
                    else:
                        st.error("SHAP engine error:")
                        st.code(_r.stderr[-2000:])
                except subprocess.TimeoutExpired:
                    st.error("SHAP engine timed out after 120 seconds.")
                except Exception as _e:
                    st.error(f"Could not launch SHAP engine: {_e}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 68 — Fundamental Features
# ─────────────────────────────────────────────────────────────────────────────
def tab_fundamental_features():  # noqa: C901
    """Step 68 — Fundamental feature enrichment for ML signals."""
    st.subheader("Fundamental Feature Enrichment")
    st.caption("P/E, P/B, ROE, earnings growth + 6 more fundamentals merged into ML walk-forward.  (Step 68)")

    _fund_path  = FILES.get("fundamental_features")
    _enh_path   = FILES.get("enhanced_ml_scores")
    _ic_path    = FILES.get("fundamental_ic")
    _rep_path   = FILES.get("fundamental_report")

    f68a, f68b, f68c = st.tabs(["IC Comparison", "Enhanced Scores", "How to Run"])

    # ── tab 1: IC comparison bar chart ───────────────────────────────────────
    with f68a:
        st.subheader("Price-only vs Enhanced IC Comparison")
        if _ic_path and Path(_ic_path).exists():
            _ic = pd.read_csv(_ic_path)
            if not _ic.empty:
                _ic_disp = _ic.copy()
                # show bar chart
                if "model" in _ic_disp.columns and "ic" in _ic_disp.columns:
                    _ic_num = pd.to_numeric(_ic_disp["ic"], errors="coerce")
                    colours = ["steelblue" if v >= 0 else "#d62728" for v in _ic_num]
                    fig = go.Figure(go.Bar(
                        x=_ic_disp["model"],
                        y=_ic_num,
                        marker_color=colours,
                        text=[f"{v:.4f}" for v in _ic_num],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        title="Information Coefficient — Price-only vs Price+Fundamental",
                        yaxis_title="IC (Spearman)",
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        font=dict(color="#111"),
                        height=360,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                st.dataframe(_ic_disp, use_container_width=True, hide_index=True)
        else:
            st.info("No IC data yet. Run: `python3 canyon_final_v9_step68_fundamental_features.py`")

        # fundamental raw features
        if _fund_path and Path(_fund_path).exists():
            with st.expander("Raw fundamental features table"):
                _ff = pd.read_csv(_fund_path)
                _num_cols_ff = [c for c in _ff.select_dtypes("number").columns]
                if _num_cols_ff:
                    _ff[_num_cols_ff] = _ff[_num_cols_ff].round(4)
                st.dataframe(_ff, use_container_width=True)

        if _rep_path and Path(_rep_path).exists():
            with st.expander("Fundamental Report"):
                st.markdown(Path(_rep_path).read_text())

    # ── tab 2: enhanced ML scores ─────────────────────────────────────────────
    with f68b:
        st.subheader("Enhanced ML Signal Scores (Price + Fundamental)")
        if _enh_path and Path(_enh_path).exists():
            _enh = pd.read_csv(_enh_path)
            if not _enh.empty:
                # score column detection
                _score_col = next(
                    (c for c in ["enhanced_score", "score", "ml_score"] if c in _enh.columns),
                    _enh.columns[-1] if len(_enh.columns) > 1 else None,
                )
                if _score_col:
                    _enh_sorted = _enh.sort_values(_score_col, ascending=False).head(25)
                    _num_ec = _enh_sorted.select_dtypes("number").columns.tolist()
                    if _num_ec:
                        _enh_sorted[_num_ec] = _enh_sorted[_num_ec].round(4)
                    st.dataframe(_enh_sorted, use_container_width=True, hide_index=True)
                    # sparkline bar
                    if "ticker" in _enh_sorted.columns:
                        fig2 = go.Figure(go.Bar(
                            x=_enh_sorted["ticker"],
                            y=pd.to_numeric(_enh_sorted[_score_col], errors="coerce"),
                            marker_color="steelblue",
                        ))
                        fig2.update_layout(
                            title="Top 25 Enhanced ML Scores",
                            yaxis_title="Score",
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                            font=dict(color="#111"),
                            height=320,
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.dataframe(_enh, use_container_width=True)
        else:
            st.info("No enhanced score data yet. Run the fundamental engine first.")

    # ── tab 3: how to run ────────────────────────────────────────────────────
    with f68c:
        st.subheader("How to Run Fundamental Features")
        st.markdown("""
**Adds 10 fundamentals from yfinance .info:**
P/E, P/B, P/S, ROE, D/E, earnings growth, revenue growth, profit margin, analyst rating, EPS fwd yield.

```bash
cd ~/Desktop/canyon_quant
python3 canyon_final_v9_step68_fundamental_features.py
```

**Notes:**
- 24-hour disk cache → `fundamental_cache.json`
- ETFs (SPY, QQQ, sector ETFs) have NaN fundamentals — median-imputed at training time
- Requires Step 66 and Step 67 in the same directory (import chain)
- Outputs `enhanced_ml_scores.csv` which Step 69 Paper Sim uses preferentially

**Outputs:**
| File | Contents |
|------|----------|
| `fundamental_features.csv` | Raw fundamental metrics per ticker |
| `enhanced_ml_scores.csv` | Walk-forward scores with fundamentals |
| `fundamental_ic_comparison.csv` | IC: price-only vs enhanced |
| `fundamental_report.md` | Summary report |

**Typical runtime:** ~30 seconds (includes yfinance API calls, cached after first run)
""")
        if st.button("▶ Run Fundamental Engine", key="run_fund68"):
            import subprocess
            _script = Path(__file__).parent / "canyon_final_v9_step68_fundamental_features.py"
            with st.spinner("Running fundamental features engine (~30s)…"):
                try:
                    _r = subprocess.run(
                        ["python3", str(_script)],
                        capture_output=True, text=True, timeout=180,
                        cwd=str(Path(__file__).parent),
                    )
                    if _r.returncode == 0:
                        st.success("Fundamental engine completed. Refresh tab to see updated scores.")
                        st.code(_r.stdout[-2000:] if len(_r.stdout) > 2000 else _r.stdout)
                    else:
                        st.error("Fundamental engine error:")
                        st.code(_r.stderr[-2000:])
                except subprocess.TimeoutExpired:
                    st.error("Fundamental engine timed out after 180 seconds.")
                except Exception as _e:
                    st.error(f"Could not launch fundamental engine: {_e}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 69 — Paper Trading Simulator
# ─────────────────────────────────────────────────────────────────────────────
def tab_paper_sim():  # noqa: C901
    """Step 69 — ML-driven paper trading simulator."""
    st.subheader("ML Paper Trading Simulator")
    st.caption("Risk-filtered paper positions driven by ML scores.  (Step 69 — canyon_final_v9_step69_paper_sim.py)")

    _pos_path = FILES.get("paper_sim_positions")
    _trd_path = FILES.get("paper_sim_trades")
    _sum_path = FILES.get("paper_sim_summary")
    _rep_path = FILES.get("paper_sim_report")
    _nav_path = FILES.get("paper_sim_nav")

    p69a, p69b, p69c, p69d, p69e = st.tabs(["Open Positions", "NAV Curve", "Trade History", "Portfolio Metrics", "How to Run"])

    # ── tab 1: open positions ─────────────────────────────────────────────────
    with p69a:
        st.subheader("Current Open Positions")
        if _pos_path and Path(_pos_path).exists():
            _pos = pd.read_csv(_pos_path)
            if _pos.empty:
                st.info("No open positions yet. Run: `python3 canyon_final_v9_step69_paper_sim.py --rebalance`")
            else:
                # colour P&L
                def _pnl_colour(val):
                    try:
                        v = float(val)
                        if v > 0:   return "color:#1a7a4a; font-weight:bold"
                        if v < 0:   return "color:#b00020; font-weight:bold"
                        return ""
                    except Exception:
                        return ""

                _pnl_cols = [c for c in ["unrealised_pct", "unrealised_pnl"] if c in _pos.columns]
                _num_pos  = _pos.select_dtypes("number").columns.tolist()
                if _num_pos:
                    _pos[_num_pos] = _pos[_num_pos].round(4)
                _styled_pos = _pos.style.map(_pnl_colour, subset=_pnl_cols) if _pnl_cols else _pos.style
                st.dataframe(_styled_pos, use_container_width=True)

                # summary metrics row
                n_open = len(_pos)
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Open Positions", n_open)
                if "unrealised_pnl" in _pos.columns:
                    _total_unreal = pd.to_numeric(_pos["unrealised_pnl"], errors="coerce").sum()
                    col2.metric("Total Unrealised P&L", f"${_total_unreal:+,.0f}")
                if "weight" in _pos.columns:
                    _total_wgt = pd.to_numeric(_pos["weight"], errors="coerce").sum()
                    col3.metric("Total Deployed Weight", f"{_total_wgt*100:.1f}%")
                if "ml_score" in _pos.columns:
                    _avg_score = pd.to_numeric(_pos["ml_score"], errors="coerce").mean()
                    col4.metric("Avg ML Score", f"{_avg_score:.3f}")
        else:
            st.info("No positions file found. Run `--rebalance` to generate first positions.")

    # ── tab 2: NAV curve ──────────────────────────────────────────────────────
    with p69b:
        st.subheader("Portfolio NAV Curve vs SPY")
        if _nav_path and Path(_nav_path).exists():
            _nav = pd.read_csv(_nav_path)
            if not _nav.empty and "date" in _nav.columns and "nav" in _nav.columns:
                _nav["date"] = pd.to_datetime(_nav["date"], errors="coerce")
                _nav_sorted = _nav.sort_values("date")
                _nav_vals = pd.to_numeric(_nav_sorted["nav"], errors="coerce")
                fig_nav = go.Figure()
                fig_nav.add_trace(go.Scatter(
                    x=_nav_sorted["date"], y=_nav_vals,
                    mode="lines+markers", name="Paper Portfolio NAV",
                    line=dict(color="#1e3a5f", width=2),
                    marker=dict(size=5),
                ))
                # SPY baseline at 100
                fig_nav.add_hline(y=100, line_dash="dash", line_color="#9ca3af",
                                  annotation_text="Baseline 100")
                fig_nav.update_layout(
                    title="Paper Portfolio NAV (indexed to 100 at start)",
                    xaxis_title="Date", yaxis_title="NAV",
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(color="#111"), height=380,
                )
                st.plotly_chart(fig_nav, use_container_width=True)

                # show raw NAV table
                if "unrealised_pct" in _nav_sorted.columns:
                    _last = _nav_sorted.iloc[-1]
                    _c1, _c2, _c3 = st.columns(3)
                    _c1.metric("Latest NAV", f"{pd.to_numeric(_last.get('nav', 0), errors='coerce'):.2f}")
                    _c2.metric("Unrealised %", f"{pd.to_numeric(_last.get('unrealised_pct', 0), errors='coerce'):+.2f}%")
                    _c3.metric("Positions", str(_last.get("n_positions", "—")))

                st.dataframe(_nav_sorted, use_container_width=True, hide_index=True)
            else:
                st.info("NAV file exists but is empty or missing columns.")
        else:
            st.info("No NAV history yet. Run: `python3 canyon_final_v9_step69_paper_sim.py --mark-to-market`")

        if st.button("▶ Mark to Market Now", key="mtm_p69b"):
            import subprocess
            _scr = Path(__file__).parent / "canyon_final_v9_step69_paper_sim.py"
            with st.spinner("Marking positions to market…"):
                try:
                    _r = subprocess.run(["python3", str(_scr), "--mark-to-market"],
                                        capture_output=True, text=True, timeout=60,
                                        cwd=str(Path(__file__).parent))
                    if _r.returncode == 0:
                        st.success("MTM complete. Refresh page to see updated NAV.")
                        st.code(_r.stdout[-1500:])
                    else:
                        st.error(_r.stderr[-1000:])
                except Exception as _e:
                    st.error(f"Error: {_e}")

    # ── tab 3: trade history ──────────────────────────────────────────────────
    with p69c:
        st.subheader("Trade History (Closed Trades)")
        if _trd_path and Path(_trd_path).exists():
            _trd = pd.read_csv(_trd_path)
            if _trd.empty:
                st.info("No closed trades yet — positions are still open.")
            else:
                def _side_colour(val):
                    if str(val).upper() == "BUY":   return "color:#1a7a4a; font-weight:bold"
                    if str(val).upper() == "SELL":  return "color:#b00020; font-weight:bold"
                    return ""
                _side_col = [c for c in ["side", "action"] if c in _trd.columns]
                _pnl_col2 = [c for c in ["realised_pnl", "pnl_pct"] if c in _trd.columns]
                _styl_trd = _trd.style
                if _side_col:  _styl_trd = _styl_trd.map(_side_colour, subset=_side_col)
                if _pnl_col2:
                    def _pnl_c2(val):
                        try:
                            v = float(val)
                            return "color:#1a7a4a" if v > 0 else ("color:#b00020" if v < 0 else "")
                        except Exception:
                            return ""
                    _styl_trd = _styl_trd.map(_pnl_c2, subset=_pnl_col2)
                st.dataframe(_styl_trd, use_container_width=True)
        else:
            st.info("No trade history yet.")

    # ── tab 4: portfolio metrics ──────────────────────────────────────────────
    with p69d:
        st.subheader("Portfolio Metrics Summary")
        if _sum_path and Path(_sum_path).exists():
            _sm = pd.read_csv(_sum_path)
            if not _sm.empty:
                # attempt to display as metric cards if key/value format
                if set(["metric", "value"]).issubset(_sm.columns):
                    _pairs = list(zip(_sm["metric"], _sm["value"]))
                    cols = st.columns(min(len(_pairs), 4))
                    for i, (k, v) in enumerate(_pairs):
                        cols[i % 4].metric(str(k), str(v))
                else:
                    st.dataframe(_sm, use_container_width=True)
        else:
            st.info("No summary data yet.")

        if _rep_path and Path(_rep_path).exists():
            with st.expander("Full Paper Sim Report"):
                st.markdown(Path(_rep_path).read_text())

    # ── tab 5: how to run ─────────────────────────────────────────────────────
    with p69e:
        st.subheader("How to Run the Paper Simulator")
        st.markdown("""
**Risk filters applied before every BUY ticket:**
- ML score ≥ 0.55
- Max 10 concurrent positions
- Max 20% weight per name
- Sector cap 40%
- Earnings blackout ±3 days
- Stop-loss 8%, profit target 20%

```bash
cd ~/Desktop/canyon_quant

# Run a rebalance (generate new BUY/SELL tickets)
python3 canyon_final_v9_step69_paper_sim.py --rebalance

# Show current status
python3 canyon_final_v9_step69_paper_sim.py --status

# Reset all positions (start fresh)
python3 canyon_final_v9_step69_paper_sim.py --reset
```

**Outputs:**
| File | Contents |
|------|----------|
| `paper_sim_positions.csv` | Current open positions |
| `paper_sim_trades.csv` | All BUY/SELL tickets |
| `paper_sim_summary.csv` | Portfolio metrics |
| `paper_sim_report.md` | Human-readable summary |

**Signal source (in priority order):**
1. `enhanced_ml_scores.csv` (Step 68 — with fundamentals)
2. `ml_signal_scores.csv` (Step 66 — price-only fallback)

> Note: This is paper simulation only. No live orders are placed.
""")
        col_rb, col_st = st.columns(2)
        with col_rb:
            if st.button("▶ Run Rebalance", key="run_psim69_rebalance", type="primary"):
                import subprocess
                _script = Path(__file__).parent / "canyon_final_v9_step69_paper_sim.py"
                with st.spinner("Running paper sim rebalance…"):
                    try:
                        _r = subprocess.run(
                            ["python3", str(_script), "--rebalance"],
                            capture_output=True, text=True, timeout=120,
                            cwd=str(Path(__file__).parent),
                        )
                        if _r.returncode == 0:
                            st.success("Rebalance complete. Refresh tabs to see updated positions.")
                            st.code(_r.stdout[-2000:] if len(_r.stdout) > 2000 else _r.stdout)
                        else:
                            st.error("Rebalance error:")
                            st.code(_r.stderr[-2000:])
                    except subprocess.TimeoutExpired:
                        st.error("Rebalance timed out after 120 seconds.")
                    except Exception as _e:
                        st.error(f"Could not launch paper sim: {_e}")
        with col_st:
            if st.button("📊 Show Status", key="run_psim69_status"):
                import subprocess
                _script = Path(__file__).parent / "canyon_final_v9_step69_paper_sim.py"
                with st.spinner("Fetching paper sim status…"):
                    try:
                        _r = subprocess.run(
                            ["python3", str(_script), "--status"],
                            capture_output=True, text=True, timeout=60,
                            cwd=str(Path(__file__).parent),
                        )
                        st.code(_r.stdout[-3000:] if len(_r.stdout) > 3000 else _r.stdout)
                    except Exception as _e:
                        st.error(f"Error: {_e}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 70 — Daily Batch Runner
# ─────────────────────────────────────────────────────────────────────────────
def tab_daily_runner():  # noqa: C901
    """Step 70 — One-click chain of all engines."""
    st.subheader("Daily Batch Runner")
    st.caption("Runs all engines in dependency order: 10-Layer → Data Health → Backtest → NLP → ML → SHAP → Fundamentals → Paper Sim → Optimizer  (Step 70)")

    _log_path = FILES.get("run_daily_log")
    _rep_path = FILES.get("run_daily_report")

    d70a, d70b, d70c = st.tabs(["Run History", "Last Report", "Launch"])

    with d70a:
        st.subheader("Engine Run History")
        if _log_path and Path(_log_path).exists():
            _log = pd.read_csv(_log_path)
            if not _log.empty:
                # show last 30 rows newest-first
                _log_disp = _log.tail(60).iloc[::-1].copy()
                def _status_colour(val):
                    s = str(val).upper()
                    if s in ("OK", "PASS"):    return "color:#1a7a4a; font-weight:bold"
                    if s in ("FAILED","ERROR","TIMEOUT"): return "color:#b00020; font-weight:bold"
                    return ""
                _sc = [c for c in ["status"] if c in _log_disp.columns]
                _styl = _log_disp.style.map(_status_colour, subset=_sc) if _sc else _log_disp.style
                st.dataframe(_styl, use_container_width=True, hide_index=True)
                # pass/fail summary
                if "status" in _log.columns:
                    _counts = _log.groupby("status").size().reset_index(name="count")
                    st.dataframe(_counts, hide_index=True)
        else:
            st.info("No run history yet. Use the Launch tab to run all engines.")

    with d70b:
        st.subheader("Last Run Report")
        if _rep_path and Path(_rep_path).exists():
            st.markdown(Path(_rep_path).read_text())
        else:
            st.info("No report yet.")

    with d70c:
        st.subheader("Launch Engines")
        col_l, col_r = st.columns(2)
        with col_l:
            _mode = st.radio("Run mode", ["Full (all engines)", "Fast (skip step56 + step68)", "Dry run (no execution)"], key="d70_mode")
        with col_r:
            _timeout = st.number_input("Per-engine timeout (s)", value=180, min_value=30, max_value=600, key="d70_to")

        _flag_map = {
            "Full (all engines)": [],
            "Fast (skip step56 + step68)": ["--fast"],
            "Dry run (no execution)": ["--dry-run"],
        }
        _flags = _flag_map.get(_mode, [])

        st.markdown("""
**Engine order:**
1. step56 — 10-Layer Runner (~60s)
2. step61 — Data Source Health (~3s)
3. step62 — Backtest Engine (~15s)
4. step65 — Earnings NLP (~10s)
5. step66 — ML Signals (~11s)
6. step67 — SHAP Explainer (~2s)
7. step68 — Fundamental Features (~30s)
8. step69 — Paper Sim Rebalance (~2s)
9. step69 — Paper Sim MTM (~1s)
10. step63 — Portfolio Optimizer (~5s)
""")
        if st.button("▶ Launch Daily Run", key="launch_d70", type="primary"):
            import subprocess
            _script = Path(__file__).parent / "canyon_final_v9_step70_daily_runner_all.py"
            _cmd = ["python3", str(_script)] + _flags + ["--timeout", str(int(_timeout))]
            with st.spinner(f"Running daily batch ({_mode})…"):
                try:
                    _r = subprocess.run(_cmd, capture_output=True, text=True,
                                        timeout=int(_timeout) * 12,
                                        cwd=str(Path(__file__).parent))
                    if _r.returncode == 0:
                        st.success("Daily run completed.")
                    else:
                        st.warning(f"Run finished with some failures (exit {_r.returncode}).")
                    st.code(_r.stdout[-3000:] if len(_r.stdout) > 3000 else _r.stdout)
                except subprocess.TimeoutExpired:
                    st.error("Daily run timed out.")
                except Exception as _e:
                    st.error(f"Launch error: {_e}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 71 — Alert System
# ─────────────────────────────────────────────────────────────────────────────
def tab_alerts():  # noqa: C901
    """Step 71 — Alert monitor: CRITICAL / WARNING / INFO."""
    st.subheader("Alert System")
    st.caption("Real-time alerts: stop-loss, ML signal, regime change, earnings blackout, sector over-weight  (Step 71)")

    _alerts_path = FILES.get("alerts")
    _rep_path    = FILES.get("alerts_report")

    a71a, a71b, a71c = st.tabs(["Active Alerts", "Alert History", "Run Monitor"])

    def _sev_colour(val):
        s = str(val).upper()
        if s == "CRITICAL": return "color:#b00020; font-weight:bold"
        if s == "WARNING":  return "color:#b45309; font-weight:bold"
        if s == "INFO":     return "color:#1a7a4a"
        return ""

    with a71a:
        st.subheader("Active Alerts (last 24h)")
        if _alerts_path and Path(_alerts_path).exists():
            _al = pd.read_csv(_alerts_path)
            if not _al.empty:
                # filter recent
                if "timestamp" in _al.columns:
                    _al["timestamp"] = pd.to_datetime(_al["timestamp"], errors="coerce")
                    _cutoff = pd.Timestamp.now() - pd.Timedelta(hours=24)
                    _recent = _al[_al["timestamp"] >= _cutoff].copy()
                else:
                    _recent = _al.copy()

                # summary counts
                if "severity" in _recent.columns:
                    _crit = int((_recent["severity"] == "CRITICAL").sum())
                    _warn = int((_recent["severity"] == "WARNING").sum())
                    _info = int((_recent["severity"] == "INFO").sum())
                    c1, c2, c3 = st.columns(3)
                    c1.metric("🔴 CRITICAL", _crit)
                    c2.metric("🟡 WARNING",  _warn)
                    c3.metric("🟢 INFO",     _info)

                if not _recent.empty:
                    _sev_cols = [c for c in ["severity"] if c in _recent.columns]
                    _styl = _recent.style.map(_sev_colour, subset=_sev_cols) if _sev_cols else _recent.style
                    st.dataframe(_styl, use_container_width=True, hide_index=True)
                else:
                    st.success("No alerts in last 24 hours.")
            else:
                st.success("No alerts on file.")
        else:
            st.info("No alerts data. Run: `python3 canyon_final_v9_step71_alert_system.py`")

        if _rep_path and Path(_rep_path).exists():
            with st.expander("Alert Report"):
                st.markdown(Path(_rep_path).read_text())

    with a71b:
        st.subheader("Full Alert History")
        if _alerts_path and Path(_alerts_path).exists():
            _all = pd.read_csv(_alerts_path)
            if not _all.empty:
                _sev_c = [c for c in ["severity"] if c in _all.columns]
                _styl2 = _all.style.map(_sev_colour, subset=_sev_c) if _sev_c else _all.style
                st.dataframe(_styl2, use_container_width=True)
        else:
            st.info("No alert history yet.")

    with a71c:
        st.subheader("Run Alert Monitor")
        st.markdown("""
```bash
# Run all checks
python3 canyon_final_v9_step71_alert_system.py

# Show summary only
python3 canyon_final_v9_step71_alert_system.py --summary

# Clear old alerts (keep last 24h)
python3 canyon_final_v9_step71_alert_system.py --clear
```
**Alert types:**
| Type | Severity | Condition |
|------|----------|-----------|
| STOP_LOSS_NEAR | CRITICAL | Position within 2% of stop |
| DRAWDOWN_WARN | CRITICAL | Portfolio unrealised < -5% |
| ML_SCORE_DROPPED | WARNING | Score fell below 0.45 |
| EARNINGS_BLACKOUT | WARNING | Earnings in 1–3 days |
| SECTOR_OVER | WARNING | Sector > 35% weight |
| REGIME_CHANGE | WARNING | SPY MA crossover |
| TARGET_NEAR | INFO | Position within 5% of target |
| ML_HIGH_SIGNAL | INFO | Score > 0.70 |
""")
        col_run, col_clr = st.columns(2)
        with col_run:
            if st.button("▶ Run Alert Check", key="run_a71"):
                import subprocess
                _scr = Path(__file__).parent / "canyon_final_v9_step71_alert_system.py"
                with st.spinner("Running alert monitor…"):
                    try:
                        _r = subprocess.run(["python3", str(_scr)],
                                            capture_output=True, text=True, timeout=60,
                                            cwd=str(Path(__file__).parent))
                        st.code(_r.stdout[-2000:] if len(_r.stdout) > 2000 else _r.stdout)
                        if _r.returncode == 0:
                            st.success("Alert check complete. Refresh Active Alerts tab.")
                    except Exception as _e:
                        st.error(f"Error: {_e}")
        with col_clr:
            if st.button("🗑 Clear Old Alerts", key="clr_a71"):
                import subprocess
                _scr = Path(__file__).parent / "canyon_final_v9_step71_alert_system.py"
                with st.spinner("Clearing…"):
                    try:
                        _r = subprocess.run(["python3", str(_scr), "--clear"],
                                            capture_output=True, text=True, timeout=30,
                                            cwd=str(Path(__file__).parent))
                        st.code(_r.stdout)
                    except Exception as _e:
                        st.error(f"Error: {_e}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 72 — Weekly HTML Report
# ─────────────────────────────────────────────────────────────────────────────
def tab_weekly_report():  # noqa: C901
    """Step 72 — Generate self-contained HTML weekly report."""
    st.subheader("Weekly Research Report")
    st.caption("Auto-generated HTML report: ML signals, paper portfolio, optimizer, SHAP, alerts.  (Step 72)")

    _html_path = FILES.get("weekly_report")

    w72a, w72b = st.tabs(["Preview", "Generate"])

    with w72a:
        st.subheader("Latest Report Preview")
        if _html_path and Path(_html_path).exists():
            _mtime = datetime.fromtimestamp(Path(_html_path).stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            st.caption(f"Last generated: {_mtime}")
            with st.expander("View HTML source (first 200 lines)"):
                _lines = Path(_html_path).read_text().split("\n")[:200]
                st.code("\n".join(_lines), language="html")
            st.markdown(f"**Full report:** open `{_html_path}` in your browser, or click:")
            st.info(f"File path: `{_html_path}`")
        else:
            st.info("No report yet. Use the Generate tab to create the first weekly report.")

    with w72b:
        st.subheader("Generate Report")
        st.markdown("""
```bash
cd ~/Desktop/canyon_quant
python3 canyon_final_v9_step72_weekly_report.py         # generate + save
python3 canyon_final_v9_step72_weekly_report.py --open  # generate + open in browser
```
**Report sections:** Market Regime · ML Signals (top 10) · Paper Portfolio NAV · Optimizer Weights · SHAP Drivers · Earnings Watchlist · Alert Summary · Run Status
""")
        col_gen, col_open = st.columns(2)
        with col_gen:
            if st.button("▶ Generate Report", key="gen_w72", type="primary"):
                import subprocess
                _scr = Path(__file__).parent / "canyon_final_v9_step72_weekly_report.py"
                with st.spinner("Generating weekly report…"):
                    try:
                        _r = subprocess.run(["python3", str(_scr)],
                                            capture_output=True, text=True, timeout=120,
                                            cwd=str(Path(__file__).parent))
                        if _r.returncode == 0:
                            st.success(f"Report saved to weekly_report_latest.html")
                            st.code(_r.stdout[-1500:])
                        else:
                            st.error("Generation failed:")
                            st.code(_r.stderr[-1500:])
                    except Exception as _e:
                        st.error(f"Error: {_e}")
        with col_open:
            if st.button("🌐 Open in Browser", key="open_w72"):
                import subprocess
                _scr = Path(__file__).parent / "canyon_final_v9_step72_weekly_report.py"
                try:
                    subprocess.Popen(["python3", str(_scr), "--open"],
                                     cwd=str(Path(__file__).parent))
                    st.success("Opening in browser…")
                except Exception as _e:
                    st.error(f"Error: {_e}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 73 — Factor Attribution
# ─────────────────────────────────────────────────────────────────────────────
def tab_factor_attribution():  # noqa: C901
    """Step 73 — Barra-style 5-factor return and risk decomposition."""
    st.subheader("Factor Attribution")
    st.caption("Decompose portfolio returns into: Market · Momentum · Low-Vol · Value · Quality  (Step 73 — Barra-style)")

    _fa_path  = FILES.get("factor_attribution")
    _mr_path  = FILES.get("factor_marginal_risk")
    _rep_path = FILES.get("factor_report")

    f73a, f73b, f73c = st.tabs(["Factor Decomposition", "Marginal Risk", "How to Run"])

    with f73a:
        st.subheader("Return & Risk Attribution by Factor")
        if _fa_path and Path(_fa_path).exists():
            _fa = pd.read_csv(_fa_path)
            if not _fa.empty:
                # stacked bar: return contribution by factor
                _ret_col = next((c for c in ["return_contrib_ann", "return_contrib", "contribution"] if c in _fa.columns), None)
                _fac_col = next((c for c in ["factor", "name"] if c in _fa.columns), None)
                if _ret_col and _fac_col:
                    _rets = pd.to_numeric(_fa[_ret_col], errors="coerce")
                    _colours = ["#1a7a4a" if v >= 0 else "#b00020" for v in _rets]
                    fig_fa = go.Figure(go.Bar(
                        x=_fa[_fac_col],
                        y=_rets * 100,
                        marker_color=_colours,
                        text=[f"{v*100:+.1f}%" for v in _rets],
                        textposition="outside",
                    ))
                    fig_fa.update_layout(
                        title="Annualised Return Contribution by Factor (%)",
                        yaxis_title="Return Contribution (%)",
                        plot_bgcolor="white", paper_bgcolor="white",
                        font=dict(color="#111"), height=360,
                    )
                    st.plotly_chart(fig_fa, use_container_width=True)

                # risk share pie / bar
                _var_col = next((c for c in ["var_contrib_pct", "variance_pct", "risk_pct"] if c in _fa.columns), None)
                if _var_col and _fac_col:
                    _vars = pd.to_numeric(_fa[_var_col], errors="coerce").fillna(0)
                    fig_vr = go.Figure(go.Bar(
                        x=_fa[_fac_col],
                        y=_vars,
                        marker_color="steelblue",
                        text=[f"{v:.1f}%" for v in _vars],
                        textposition="outside",
                    ))
                    fig_vr.update_layout(
                        title="Variance Contribution by Factor (%)",
                        yaxis_title="% of Total Variance",
                        plot_bgcolor="white", paper_bgcolor="white",
                        font=dict(color="#111"), height=320,
                    )
                    st.plotly_chart(fig_vr, use_container_width=True)

                # raw table
                _num_fa = _fa.select_dtypes("number").columns.tolist()
                if _num_fa: _fa[_num_fa] = _fa[_num_fa].round(4)
                st.dataframe(_fa, use_container_width=True, hide_index=True)
        else:
            st.info("No factor data yet. Run: `python3 canyon_final_v9_step73_factor_attribution.py`")

        if _rep_path and Path(_rep_path).exists():
            with st.expander("Factor Attribution Report"):
                st.markdown(Path(_rep_path).read_text())

    with f73b:
        st.subheader("Marginal Risk Contribution (per Position)")
        if _mr_path and Path(_mr_path).exists():
            _mr = pd.read_csv(_mr_path)
            if not _mr.empty:
                _mrc_col = next((c for c in ["mrc", "marginal_risk", "risk_contrib"] if c in _mr.columns), None)
                _tk_col  = next((c for c in ["ticker"] if c in _mr.columns), None)
                if _mrc_col and _tk_col:
                    _mr_s = _mr.sort_values(_mrc_col, ascending=False)
                    fig_mr = go.Figure(go.Bar(
                        x=_mr_s[_tk_col],
                        y=pd.to_numeric(_mr_s[_mrc_col], errors="coerce") * 100,
                        marker_color="steelblue",
                    ))
                    fig_mr.update_layout(
                        title="Marginal Risk Contribution per Position (%)",
                        yaxis_title="MRC (%)",
                        plot_bgcolor="white", paper_bgcolor="white",
                        font=dict(color="#111"), height=320,
                    )
                    st.plotly_chart(fig_mr, use_container_width=True)
                _num_mr = _mr.select_dtypes("number").columns.tolist()
                if _num_mr: _mr[_num_mr] = _mr[_num_mr].round(4)
                st.dataframe(_mr, use_container_width=True, hide_index=True)
        else:
            st.info("No marginal risk data yet.")

    with f73c:
        st.subheader("How to Run Factor Attribution")
        st.markdown("""
**5 Factors:**
| Factor | Proxy |
|--------|-------|
| Market | SPY daily returns |
| Momentum | Equal-weight top-quintile 12m-momentum stocks |
| Low-Vol | Equal-weight bottom-quintile 21d-vol stocks |
| Value | Top-quintile 1/PE from fundamental_features.csv |
| Quality | Top-quintile ROE from fundamental_features.csv |

```bash
cd ~/Desktop/canyon_quant
python3 canyon_final_v9_step73_factor_attribution.py
python3 canyon_final_v9_step73_factor_attribution.py --refresh  # force re-download prices
```
**Requires:** step66 ML signal scores (for universe) + step68 fundamentals (for Value/Quality factors)
**Runtime:** ~15s (cached); ~45s (fresh download)
""")
        if st.button("▶ Run Factor Attribution", key="run_fa73", type="primary"):
            import subprocess
            _scr = Path(__file__).parent / "canyon_final_v9_step73_factor_attribution.py"
            with st.spinner("Running factor attribution (~15s)…"):
                try:
                    _r = subprocess.run(["python3", str(_scr)],
                                        capture_output=True, text=True, timeout=180,
                                        cwd=str(Path(__file__).parent))
                    if _r.returncode == 0:
                        st.success("Factor attribution complete. Refresh tabs.")
                        st.code(_r.stdout[-2000:])
                    else:
                        st.error("Error:")
                        st.code(_r.stderr[-2000:])
                except Exception as _e:
                    st.error(f"Error: {_e}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 78 — Deep Fundamental Analysis
# ─────────────────────────────────────────────────────────────────────────────
def tab_deep_fundamentals():
    """Step 78 — FCF yield, accruals, Debt/EBITDA, margin trend, ROE quality scores."""
    st.subheader("Deep Fundamental Quality Scores")
    st.caption(
        "FCF Yield · Accruals Quality · Gross Margin Δ · Debt/EBITDA · Revenue Growth · ROE  (Step 78)"
    )

    _rank_path  = FILES.get("fundamental_quality_rank")
    _score_path = FILES.get("fundamental_deep_scores")
    _rep_path   = FILES.get("fundamental_deep_report")

    if _rank_path and _rank_path.exists():
        try:
            rank_df = read_csv(_rank_path)
            # read_csv returns all-string dtype; coerce numeric cols
            for _nc in ["quality_score", "fcf_yield", "revenue_growth",
                        "debt_ebitda", "roe", "accruals_quality", "gross_margin_delta"]:
                if _nc in rank_df.columns:
                    rank_df[_nc] = pd.to_numeric(rank_df[_nc], errors="coerce")
            n = len(rank_df)

            # Distribution metrics
            if "quality_label" in rank_df.columns:
                vc = rank_df["quality_label"].value_counts()
                cols_m = st.columns(4)
                for i, label in enumerate(["HIGH", "ABOVE_AVG", "BELOW_AVG", "LOW"]):
                    cnt = int(vc.get(label, 0))
                    cols_m[i].metric(label, cnt, f"{cnt/n*100:.0f}% of {n}")

            # Quality score bar chart (top 30)
            st.markdown("#### Top 30 Quality Scores (Bear-Market Shelters)")
            top30 = rank_df.sort_values("quality_score", ascending=False).head(30)
            import plotly.graph_objects as go
            colours = ["#1a7a4a" if v >= 75 else ("#2196F3" if v >= 50 else "#FF9800")
                       for v in pd.to_numeric(top30.get("quality_score", pd.Series()), errors="coerce")]
            fig_q = go.Figure(go.Bar(
                x=top30["ticker"].tolist(),
                y=pd.to_numeric(top30["quality_score"], errors="coerce").tolist(),
                marker_color=colours,
                text=[f"{v:.0f}" for v in pd.to_numeric(top30["quality_score"], errors="coerce")],
                textposition="outside",
            ))
            fig_q.update_layout(
                height=320, title="Quality Score (0=worst, 100=best)",
                yaxis_title="Quality Score", xaxis_title="Ticker",
                plot_bgcolor="white", paper_bgcolor="white",
            )
            fig_q.add_hline(y=75, line_dash="dash", line_color="#1a7a4a",
                            annotation_text="HIGH threshold")
            st.plotly_chart(fig_q, use_container_width=True)

            # FCF Yield chart
            if "fcf_yield" in rank_df.columns:
                st.markdown("#### FCF Yield — Self-Funding Ability")
                fcf_df = rank_df.dropna(subset=["fcf_yield"]).sort_values("fcf_yield", ascending=False).head(20)
                fcf_vals = pd.to_numeric(fcf_df["fcf_yield"], errors="coerce") * 100
                fig_fcf = go.Figure(go.Bar(
                    x=fcf_df["ticker"].tolist(), y=fcf_vals.tolist(),
                    marker_color=["#1a7a4a" if v > 3 else ("#FF9800" if v > 0 else "#b00020")
                                  for v in fcf_vals],
                    text=[f"{v:.1f}%" for v in fcf_vals],
                    textposition="outside",
                ))
                fig_fcf.update_layout(
                    height=280, title="FCF Yield %  (>3% = self-funded)",
                    plot_bgcolor="white", paper_bgcolor="white",
                )
                fig_fcf.add_hline(y=3, line_dash="dash", line_color="#1a7a4a",
                                  annotation_text="3% threshold")
                st.plotly_chart(fig_fcf, use_container_width=True)

            # Full table
            st.markdown("#### Full Fundamental Table")
            num_cols = rank_df.select_dtypes("number").columns.tolist()
            if num_cols:
                rank_df[num_cols] = rank_df[num_cols].round(3)

            def _qual_color(val):
                s = str(val)
                if s == "HIGH":     return "color:#1a7a4a; font-weight:bold"
                if s == "ABOVE_AVG": return "color:#2196F3; font-weight:bold"
                if s == "BELOW_AVG": return "color:#FF9800"
                if s == "LOW":      return "color:#b00020"
                return ""

            styl = rank_df.style.map(_qual_color, subset=["quality_label"]) \
                if "quality_label" in rank_df.columns else rank_df.style
            st.dataframe(styl, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error loading fundamental data: {e}")
    else:
        st.info("No fundamental data yet.")

    # Report
    if _rep_path and _rep_path.exists():
        with st.expander("Full Fundamental Report", expanded=False):
            st.markdown(_rep_path.read_text())

    st.markdown("---")
    st.markdown(
        "**Why fundamentals matter in bear markets:** In bull markets momentum explains "
        "most returns. In bear markets, companies with positive FCF, low debt, and real "
        "earnings (low accruals) outperform. This score identifies those shelters."
    )
    st.code("python3 canyon_final_v9_step78_deep_fundamentals.py", language="bash")
    _scr = Path(__file__).parent / "canyon_final_v9_step78_deep_fundamentals.py"
    if st.button("▶ Run Deep Fundamentals Now", key="btn_step78"):
        if _scr.exists():
            import subprocess
            with st.spinner("Running Step 78 (~3 min for 60 tickers)..."):
                result = subprocess.run(
                    ["python3", str(_scr)],
                    capture_output=True, text=True, timeout=300,
                )
            if result.returncode == 0:
                st.success("Step 78 complete!")
                st.text(result.stdout[-1500:] if result.stdout else "")
            else:
                st.error("Step 78 failed")
                st.text(result.stderr[-1000:] if result.stderr else "")
        else:
            st.error("step78 file not found")


def tab_finbert_sentiment():
    """Step 79 — FinBERT news sentiment analysis."""
    st.subheader("FinBERT News Sentiment")
    st.caption("Pre-trained financial NLP model · Scores recent headlines per ticker · BULLISH / NEUTRAL / BEARISH  (Step 79)")

    _sent_path = FILES.get("finbert_sentiment")
    _rep_path  = FILES.get("finbert_report")

    if _sent_path and _sent_path.exists():
        try:
            df = read_csv(_sent_path)
            for c in ["sentiment_score", "rank_sentiment", "n_headlines", "positive", "negative", "neutral"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            n = len(df)
            bull = int((df["label"] == "BULLISH").sum())   if "label" in df.columns else 0
            bear = int((df["label"] == "BEARISH").sum())   if "label" in df.columns else 0
            neut = int((df["label"] == "NEUTRAL").sum())   if "label" in df.columns else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tickers Scored", n)
            c2.metric("🟢 BULLISH", bull, f"{bull/n*100:.0f}%" if n else "")
            c3.metric("🔴 BEARISH", bear, f"{bear/n*100:.0f}%" if n else "")
            c4.metric("⚪ NEUTRAL",  neut, f"{neut/n*100:.0f}%" if n else "")

            import plotly.graph_objects as go

            # Sentiment score bar chart — top/bottom 20
            st.markdown("#### Sentiment Score — Top BULLISH vs BEARISH")
            df_s = df.dropna(subset=["sentiment_score"]).sort_values("sentiment_score", ascending=False)
            top20  = df_s.head(15)
            bot20  = df_s.tail(10).sort_values("sentiment_score")
            plot_df = pd.concat([top20, bot20])
            colours = ["#1a7a4a" if v >= 0 else "#b00020" for v in plot_df["sentiment_score"]]
            fig = go.Figure(go.Bar(
                x=plot_df["ticker"].tolist(),
                y=plot_df["sentiment_score"].tolist(),
                marker_color=colours,
                text=[f"{v:+.2f}" for v in plot_df["sentiment_score"]],
                textposition="outside",
            ))
            fig.update_layout(
                height=320, title="FinBERT Sentiment Score (−1 = very bearish, +1 = very bullish)",
                yaxis_title="Sentiment Score", xaxis_title="Ticker",
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="#888"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Headline count chart
            if "n_headlines" in df.columns:
                st.markdown("#### News Coverage (# headlines last 30 days)")
                top_cov = df.dropna(subset=["n_headlines"]).sort_values("n_headlines", ascending=False).head(20)
                fig2 = go.Figure(go.Bar(
                    x=top_cov["ticker"].tolist(),
                    y=top_cov["n_headlines"].tolist(),
                    marker_color="#00bcd4",
                    text=top_cov["n_headlines"].astype(int).tolist(),
                    textposition="outside",
                ))
                fig2.update_layout(height=250, title="Headlines per Ticker (more = more market attention)",
                                   plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(fig2, use_container_width=True)

            # Full table
            st.markdown("#### Full Sentiment Table")
            show_cols = [c for c in ["ticker","sentiment_score","rank_sentiment","label",
                                      "n_headlines","positive","negative","neutral","sample_headline"]
                         if c in df.columns]

            def _sent_color(val):
                s = str(val)
                if s == "BULLISH": return "color:#1a7a4a; font-weight:bold"
                if s == "BEARISH": return "color:#b00020; font-weight:bold"
                return ""

            styl = df[show_cols].sort_values("sentiment_score", ascending=False).style
            if "label" in show_cols:
                styl = styl.map(_sent_color, subset=["label"])
            st.dataframe(styl, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error loading sentiment data: {e}")
    else:
        st.info("No sentiment data yet — run Step 79 below.")

    if _rep_path and _rep_path.exists():
        with st.expander("Full Sentiment Report", expanded=False):
            st.markdown(_rep_path.read_text())

    st.markdown("---")
    _scr = Path(__file__).parent / "canyon_final_v9_step79_finbert_sentiment.py"
    st.code("python3 canyon_final_v9_step79_finbert_sentiment.py --top 100", language="bash")
    if st.button("▶ Run FinBERT Sentiment Now (~5 min)", key="btn_step79"):
        if _scr.exists():
            import subprocess
            with st.spinner("Downloading FinBERT model + scoring headlines (~5 min first run)..."):
                result = subprocess.run(
                    ["python3", str(_scr), "--top", "100"],
                    capture_output=True, text=True, timeout=600,
                )
            if result.returncode == 0:
                st.success("Step 79 complete!")
                st.text(result.stdout[-2000:] if result.stdout else "")
            else:
                st.error("Step 79 failed")
                st.text(result.stderr[-1000:] if result.stderr else "")
        else:
            st.error("step79 file not found")


# ─────────────────────────────────────────────────────────────────────────────
# Step 75 — Universe Expansion + Extended Backtest
# ─────────────────────────────────────────────────────────────────────────────
def tab_universe_expansion():
    """Step 75 — S&P 500 universe backtest 2000-2025 with regime IC."""
    st.subheader("Universe Expansion — 97-Ticker, 25-Year Backtest")
    st.caption("Survivorship-bias check · BULL/BEAR/SIDEWAYS IC · 2000-2025 walk-forward  (Step 75)")

    # --- Top-level metrics from ic_by_regime_full.csv ---
    _ic_path = FILES.get("ic_by_regime_full")
    _rp_path = FILES.get("universe_expansion_report")

    if _ic_path and _ic_path.exists():
        try:
            ic_df = read_csv(_ic_path)
            st.markdown("#### Per-Regime IC Summary (25-Year Walk-Forward)")
            cols = st.columns(4)
            regime_colors = {"BULL": "green", "BEAR": "red", "SIDEWAYS": "orange", "ALL": "blue"}
            for i, row in ic_df.iterrows():
                reg = row.get("regime", "")
                mean_ic = row.get("mean_ic", 0)
                t_stat = row.get("t_stat", 0)
                assess = row.get("assessment", "")
                n = row.get("n_months", row.get("n_periods", "?"))
                icon = "✅" if assess == "STRONG" else ("⚠️" if float(t_stat) > 1 else "❌")
                with cols[i % 4]:
                    st.metric(
                        f"{icon} {reg}",
                        f"IC={float(mean_ic):+.4f}",
                        f"t={float(t_stat):+.2f} | n={n}",
                    )
        except Exception as e:
            st.error(f"Error loading IC data: {e}")
    else:
        st.info("No regime IC data yet.")

    # --- Portfolio performance chart ---
    _perf_path = FILES.get("extended_backtest_perf")
    if _perf_path and _perf_path.exists():
        try:
            perf = read_csv(_perf_path)
            if "date" in perf.columns and "portfolio_ret" in perf.columns:
                perf["date"] = pd.to_datetime(perf["date"])
                perf = perf.sort_values("date")
                perf["cum_ret"] = (1 + perf["portfolio_ret"]).cumprod()
                st.markdown("#### Cumulative Return (Monthly Rebalance, Long-Only Top Quintile)")
                import plotly.graph_objects as go
                fig = go.Figure()
                for reg, color in [("BULL", "#2196F3"), ("BEAR", "#F44336"), ("SIDEWAYS", "#FF9800")]:
                    sub = perf[perf["regime"] == reg] if "regime" in perf.columns else perf
                    if not sub.empty:
                        fig.add_trace(go.Scatter(
                            x=sub["date"], y=sub["cum_ret"],
                            mode="markers", name=reg,
                            marker=dict(size=4, color=color),
                        ))
                fig.add_trace(go.Scatter(
                    x=perf["date"], y=perf["cum_ret"],
                    mode="lines", name="ALL", line=dict(color="#888", width=1),
                ))
                fig.update_layout(
                    height=350, title="Portfolio vs Regime",
                    xaxis_title="Date", yaxis_title="Cum Return (1=start)",
                    plot_bgcolor="white", paper_bgcolor="white",
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render performance chart: {e}")

    # --- Report ---
    if _rp_path and _rp_path.exists():
        with st.expander("Full Universe Expansion Report", expanded=False):
            st.markdown(_rp_path.read_text())

    st.markdown("---")
    st.markdown("**Run Step 75:**")
    st.code("python3 canyon_final_v9_step75_universe_expansion.py", language="bash")
    st.caption("Downloads 97+ S&P 500 tickers from 2000, builds 25-year walk-forward IC by regime.")

    _scr = Path(__file__).parent / "canyon_final_v9_step75_universe_expansion.py"
    if st.button("▶ Run Universe Expansion Now", key="btn_step75"):
        if _scr.exists():
            import subprocess
            with st.spinner("Running Step 75 (~4 min)..."):
                result = subprocess.run(
                    ["python3", str(_scr)],
                    capture_output=True, text=True, timeout=360,
                )
            if result.returncode == 0:
                st.success("Step 75 complete!")
                st.text(result.stdout[-2000:] if result.stdout else "")
            else:
                st.error("Step 75 failed")
                st.text(result.stderr[-2000:] if result.stderr else "")
        else:
            st.error("step75 file not found")


# ─────────────────────────────────────────────────────────────────────────────
# Step 76 — Regime Detector
# ─────────────────────────────────────────────────────────────────────────────
def tab_regime_detector():
    """Step 76 — 4-indicator regime classification: BULL/BEAR/SIDEWAYS."""
    st.subheader("Market Regime Detector")
    st.caption("SPY MA-cross · VIX · 20-day momentum · RSI(14) → BULL / BEAR / SIDEWAYS  (Step 76)")

    import json as _json

    # --- Current regime badge ---
    _cur_path = FILES.get("regime_current")
    if _cur_path and _cur_path.exists():
        try:
            cur = _json.loads(_cur_path.read_text())
            reg = cur.get("regime", "N/A")
            score = cur.get("score", 0)
            date  = cur.get("date", "")
            colors = {"BULL": "#2196F3", "BEAR": "#F44336", "SIDEWAYS": "#FF9800"}
            c = colors.get(reg, "#888")
            st.markdown(
                f"<div style='background:{c};color:white;padding:12px 20px;"
                f"border-radius:8px;font-size:1.4em;font-weight:bold;display:inline-block'>"
                f"Current Regime: {reg}  (score={score})</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"As of {date}")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("SPY 50MA", f"{cur.get('ma50', 0):.2f}")
            col2.metric("SPY 200MA", f"{cur.get('ma200', 0):.2f}")
            col3.metric("VIX", f"{cur.get('vix', 0):.1f}")
            col4.metric("RSI(14)", f"{cur.get('rsi', 0):.1f}")
        except Exception as e:
            st.error(f"Error reading regime_current.json: {e}")
    else:
        st.info("No regime data yet. Run Step 76.")

    # --- Regime history chart ---
    _hist_path = FILES.get("regime_history")
    if _hist_path and _hist_path.exists():
        try:
            hist = pd.read_csv(_hist_path, index_col=0, parse_dates=True)
            hist.index.name = "date"
            hist = hist.reset_index()
            st.markdown("#### Regime History (2000–present)")
            import plotly.graph_objects as go

            encode = {"BULL": 2, "SIDEWAYS": 1, "BEAR": 0}
            hist["regime_num"] = hist["regime"].map(encode)

            fig = go.Figure()
            for reg, num, color in [("BULL", 2, "#2196F3"), ("SIDEWAYS", 1, "#FF9800"), ("BEAR", 0, "#F44336")]:
                sub = hist[hist["regime"] == reg]
                if not sub.empty:
                    fig.add_trace(go.Scatter(
                        x=sub["date"], y=[num] * len(sub),
                        mode="markers", name=reg,
                        marker=dict(size=2, color=color, opacity=0.5),
                    ))
            # Overlay SPY price (normalised)
            if "spy_close" in hist.columns:
                spy_norm = hist["spy_close"] / hist["spy_close"].max() * 2
                fig.add_trace(go.Scatter(
                    x=hist["date"], y=spy_norm,
                    mode="lines", name="SPY (norm)",
                    line=dict(color="#333", width=1),
                    yaxis="y2",
                ))
            fig.update_layout(
                height=320, title="Regime Timeline",
                yaxis=dict(tickvals=[0, 1, 2], ticktext=["BEAR", "SIDEWAYS", "BULL"]),
                yaxis2=dict(overlaying="y", side="right", showgrid=False),
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render regime history: {e}")

    # --- Transitions table ---
    _trans_path = FILES.get("regime_transitions")
    if _trans_path and _trans_path.exists():
        try:
            trans = read_csv(_trans_path)
            st.markdown("#### Recent Regime Transitions")
            st.dataframe(trans.tail(15), use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not load transitions: {e}")

    # --- Report ---
    _rp_path = FILES.get("regime_report")
    if _rp_path and _rp_path.exists():
        with st.expander("Full Regime Report", expanded=False):
            st.markdown(_rp_path.read_text())

    st.markdown("---")
    st.code("python3 canyon_final_v9_step76_regime_detector.py", language="bash")
    _scr = Path(__file__).parent / "canyon_final_v9_step76_regime_detector.py"
    if st.button("▶ Refresh Regime Now", key="btn_step76"):
        if _scr.exists():
            import subprocess
            with st.spinner("Running Step 76 (~30s)..."):
                result = subprocess.run(
                    ["python3", str(_scr)],
                    capture_output=True, text=True, timeout=120,
                )
            if result.returncode == 0:
                st.success("Regime updated!")
                st.text(result.stdout[-1000:] if result.stdout else "")
            else:
                st.error("Step 76 failed")
                st.text(result.stderr[-1000:] if result.stderr else "")
        else:
            st.error("step76 file not found")


# ─────────────────────────────────────────────────────────────────────────────
# Step 77 — Regime-Conditional ML
# ─────────────────────────────────────────────────────────────────────────────
def tab_regime_ml():
    """Step 77 — Regime-conditional ML: separate models for BULL/BEAR/SIDEWAYS."""
    st.subheader("Regime-Conditional ML Model")
    st.caption("Trains BULL/BEAR/SIDEWAYS models separately · compares IC vs single all-data model  (Step 77)")

    import json as _json

    # --- IC comparison summary ---
    _sum_path = FILES.get("regime_ic_summary")
    if _sum_path and _sum_path.exists():
        try:
            summary = read_csv(_sum_path)
            st.markdown("#### IC Comparison: All-Data vs Regime-Conditional Models")

            import plotly.graph_objects as go
            fig = go.Figure()
            for model, color in [("ALL_DATA", "#888"), ("REGIME_COND", "#2196F3")]:
                sub = summary[summary["model"] == model] if "model" in summary.columns else summary
                if not sub.empty:
                    fig.add_trace(go.Bar(
                        x=sub["regime"].tolist() if "regime" in sub.columns else [],
                        y=sub["ic_mean"].astype(float).tolist() if "ic_mean" in sub.columns else [],
                        name=model,
                        marker_color=color,
                        text=[f"t={float(t):.2f}" for t in sub.get("t_stat", [0]*len(sub))],
                        textposition="outside",
                    ))
            fig.update_layout(
                height=300, barmode="group", title="IC by Regime & Model",
                xaxis_title="Regime", yaxis_title="Mean IC",
                plot_bgcolor="white", paper_bgcolor="white",
            )
            fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1)
            st.plotly_chart(fig, use_container_width=True)

            # Table
            display_cols = ["regime", "model", "ic_mean", "t_stat", "pct_pos", "meaningful"] \
                if all(c in summary.columns for c in ["regime", "model"]) else summary.columns.tolist()
            st.dataframe(summary[display_cols], use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Error rendering IC summary: {e}")
    else:
        st.info("No regime ML data yet. Run Step 77.")

    # --- Today's LONG signals ---
    _scores_path = FILES.get("regime_ml_scores")
    if _scores_path and _scores_path.exists():
        try:
            scores = read_csv(_scores_path)
            cur_regime = "N/A"
            if FILES.get("regime_current") and FILES["regime_current"].exists():
                try:
                    cur_regime = _json.loads(FILES["regime_current"].read_text()).get("regime", "N/A")
                except Exception:
                    pass
            st.markdown(f"#### Today's Signals — {cur_regime} Regime")
            sig_cols = st.tabs(["LONG", "HOLD", "Avoid"])
            for tab, sig in zip(sig_cols, ["LONG", "HOLD", "SHORT_AVOID"]):
                with tab:
                    sub = scores[scores["signal"] == sig] if "signal" in scores.columns else scores
                    if not sub.empty:
                        show_cols = ["ticker", "predicted_score", "signal"] + \
                            [c for c in ["mom_1m", "mom_3m", "inv_vol"] if c in sub.columns]
                        st.dataframe(
                            sub[show_cols].head(20),
                            use_container_width=True, hide_index=True,
                        )
                    else:
                        st.write("No signals in this category.")
        except Exception as e:
            st.warning(f"Could not load scores: {e}")

    # ── Triple-Confirmation Ranking (ML × Sentiment × Quality) ──────────────
    st.markdown("---")
    st.markdown("### 🔥 Triple-Dimension Composite Ranking — ML Momentum × FinBERT Sentiment × Fundamental Quality")
    st.caption("Weights: ML Momentum 50% · News Sentiment 30% · Fundamental Quality 20%   🔥=Triple Confirm  ✅=Double Confirm  ⚠️=Data Not Covered")

    try:
        _sc  = FILES.get("regime_ml_scores")
        _fd  = FILES.get("fundamental_deep_scores")
        _sn  = FILES.get("finbert_sentiment")

        if _sc and _sc.exists():
            ml_df = read_csv(_sc)
            for c in ["predicted_score", "mom_1m", "mom_3m"]:
                if c in ml_df.columns:
                    ml_df[c] = pd.to_numeric(ml_df[c], errors="coerce")

            # merge fundamentals
            if _fd and _fd.exists():
                fd_df = read_csv(_fd)[["ticker", "quality_score", "quality_label", "revenue_growth"]]
                for c in ["quality_score", "revenue_growth"]:
                    fd_df[c] = pd.to_numeric(fd_df[c], errors="coerce")
                ml_df = ml_df.merge(fd_df, on="ticker", how="left")

            # merge sentiment
            if _sn and _sn.exists():
                sn_df = read_csv(_sn)[["ticker", "sentiment_score", "rank_sentiment", "label"]]
                sn_df.rename(columns={"label": "sent_label"}, inplace=True)
                for c in ["sentiment_score", "rank_sentiment"]:
                    sn_df[c] = pd.to_numeric(sn_df[c], errors="coerce")
                ml_df = ml_df.merge(sn_df, on="ticker", how="left")

            long_df = ml_df[ml_df["signal"] == "LONG"].copy()
            if not long_df.empty:
                long_df["ml_r"]   = long_df["predicted_score"].rank(pct=True)
                long_df["sent_r"] = long_df["rank_sentiment"].rank(pct=True).fillna(0.5)
                long_df["qual_r"] = long_df["quality_score"].rank(pct=True).fillna(0.45)
                long_df["triple_score"] = (
                    long_df["ml_r"] * 0.5 +
                    long_df["sent_r"] * 0.3 +
                    long_df["qual_r"] * 0.2
                )
                long_df = long_df.sort_values("triple_score", ascending=False).head(20)

                # flag column
                def _flag(row):
                    sl = str(row.get("sent_label", ""))
                    ql = str(row.get("quality_label", ""))
                    if sl == "BULLISH" and ql in ["HIGH", "ABOVE_AVG"]:
                        return "🔥"
                    if ql in ["HIGH", "ABOVE_AVG"] or sl == "BULLISH":
                        return "✅"
                    return "⚠️"
                long_df["Confirm"] = long_df.apply(_flag, axis=1)

                # format display
                disp = long_df[["Confirm", "ticker", "triple_score", "predicted_score",
                                 "sent_label", "sentiment_score",
                                 "quality_label", "quality_score",
                                 "mom_1m", "mom_3m"]].copy()
                for c in ["triple_score", "predicted_score", "sentiment_score", "quality_score"]:
                    disp[c] = pd.to_numeric(disp[c], errors="coerce").round(3)
                disp["mom_1m"] = (pd.to_numeric(disp["mom_1m"], errors="coerce") * 100).round(1).astype(str) + "%"
                disp["mom_3m"] = (pd.to_numeric(disp["mom_3m"], errors="coerce") * 100).round(1).astype(str) + "%"
                disp.columns = ["✓", "Ticker", "Composite Score", "ML Score", "Sentiment", "Sentiment Score", "Quality", "Quality Score", "1M Momentum", "3M Momentum"]

                def _triple_style(val):
                    if val == "🔥": return "font-size:1.2em"
                    if val == "✅": return "color:#1a7a4a; font-weight:bold"
                    return "color:#888"
                def _sent_style(val):
                    if str(val) == "BULLISH": return "color:#1a7a4a; font-weight:bold"
                    if str(val) == "BEARISH": return "color:#b00020; font-weight:bold"
                    return ""
                def _qual_style(val):
                    if str(val) == "HIGH": return "color:#1a7a4a; font-weight:bold"
                    if str(val) == "ABOVE_AVG": return "color:#2196F3"
                    if str(val) == "BELOW_AVG": return "color:#FF9800"
                    if str(val) == "LOW": return "color:#b00020"
                    return ""

                styled = (disp.style
                    .map(_triple_style, subset=["✓"])
                    .map(_sent_style,   subset=["Sentiment"])
                    .map(_qual_style,   subset=["Quality"]))
                st.dataframe(styled, use_container_width=True, hide_index=True)

                # Top 5 highlight
                top5 = long_df[long_df["Confirmed"] == "🔥"].head(5)
                if not top5.empty:
                    st.markdown("**🔥 Triple Confirmed (ready to research):**  " +
                                "  ·  ".join(f"**{r['ticker']}**" for _, r in top5.iterrows()))
        else:
            st.info("Run Step 77 first to generate signal data")
    except Exception as e:
        st.warning(f"3D ranking load failed: {e}")

    # --- Report ---
    _rp_path = FILES.get("regime_ml_report")
    if _rp_path and _rp_path.exists():
        with st.expander("Full Regime ML Report", expanded=False):
            st.markdown(_rp_path.read_text())

    st.markdown("---")
    st.code("python3 canyon_final_v9_step77_regime_ml.py", language="bash")
    _scr = Path(__file__).parent / "canyon_final_v9_step77_regime_ml.py"
    if st.button("▶ Run Regime ML Now", key="btn_step77"):
        if _scr.exists():
            import subprocess
            with st.spinner("Running Step 77 (~5 min)..."):
                result = subprocess.run(
                    ["python3", str(_scr)],
                    capture_output=True, text=True, timeout=420,
                )
            if result.returncode == 0:
                st.success("Step 77 complete!")
                st.text(result.stdout[-2000:] if result.stdout else "")
            else:
                st.error("Step 77 failed")
                st.text(result.stderr[-2000:] if result.stderr else "")
        else:
            st.error("step77 file not found")


# ─────────────────────────────────────────────────────────────────────────────
# Step 74 — Options Chain (L7)
# ─────────────────────────────────────────────────────────────────────────────
def tab_options_chain():  # noqa: C901
    """Step 74 — yfinance L7 options chain: PCR, max pain, gamma wall, VIX Rank, IV/HV."""
    st.subheader("Options Chain — Layer 7")
    st.caption("PCR · Max Pain · Gamma Wall · **VIX Rank** · **IV/HV Ratio** · Kill Zone  —  yfinance (Step 74)")

    _sum_path = FILES.get("options_chain_summary")
    _rep_path = FILES.get("options_chain_report")

    o74a, o74b, o74c = st.tabs(["Signal Summary", "Chain Detail", "How to Run"])

    def _signal_colour(val):
        s = str(val)
        if "BULLISH"  in s: return "color:#1a7a4a; font-weight:bold"
        if "BEARISH"  in s: return "color:#b00020; font-weight:bold"
        if "IV_RICH"  in s: return "color:#b45309; font-weight:bold"
        if "IV_CHEAP" in s: return "color:#1e3a5f; font-weight:bold"
        if "PINNED"   in s: return "color:#7c3aed; font-weight:bold"
        return ""

    with o74a:
        st.subheader("Options Signal Summary")
        if _sum_path and Path(_sum_path).exists():
            _sm = pd.read_csv(_sum_path)
            if not _sm.empty:
                # metrics row
                if "signal" in _sm.columns:
                    _sig_counts = _sm["signal"].value_counts()
                    cols_sig = st.columns(min(len(_sig_counts), 5))
                    for i, (sig, cnt) in enumerate(_sig_counts.items()):
                        cols_sig[i % 5].metric(sig, cnt)

                # Two charts: VIX Rank and IV/HV Ratio
                _tk_col  = "ticker" if "ticker" in _sm.columns else None
                _vixrk_col = next((c for c in ["vix_rank", "iv_rank"] if c in _sm.columns), None)
                _ivhv_col  = "iv_hv_ratio" if "iv_hv_ratio" in _sm.columns else None

                if _vixrk_col and _tk_col:
                    _sm_s = _sm.sort_values(_vixrk_col, ascending=False).dropna(subset=[_vixrk_col]).head(20)
                    _vr_vals = pd.to_numeric(_sm_s[_vixrk_col], errors="coerce")
                    _vr_clrs = ["#b00020" if v > 70 else ("#1e3a5f" if v < 30 else "steelblue")
                                for v in _vr_vals]
                    fig_vr = go.Figure(go.Bar(
                        x=_sm_s[_tk_col], y=_vr_vals,
                        marker_color=_vr_clrs,
                        text=[f"{v:.0f}" for v in _vr_vals],
                        textposition="outside",
                    ))
                    fig_vr.update_layout(
                        title="VIX Rank by Ticker  (0=52-week VIX low, 100=52-week VIX high)",
                        yaxis_title="VIX Rank (0–100)",
                        plot_bgcolor="white", paper_bgcolor="white",
                        font=dict(color="#111"), height=320,
                    )
                    fig_vr.add_hline(y=70, line_dash="dash", line_color="#b00020",
                                     annotation_text="Rich (>70)")
                    fig_vr.add_hline(y=25, line_dash="dash", line_color="#1e3a5f",
                                     annotation_text="Cheap (<25)")
                    st.plotly_chart(fig_vr, use_container_width=True)

                if _ivhv_col and _tk_col:
                    _sm_h = _sm.dropna(subset=[_ivhv_col]).sort_values(_ivhv_col, ascending=False).head(20)
                    _hv_vals = pd.to_numeric(_sm_h[_ivhv_col], errors="coerce")
                    _hv_clrs = ["#b00020" if v > 1.5 else ("#1e3a5f" if v < 0.7 else "steelblue")
                                for v in _hv_vals]
                    fig_hv = go.Figure(go.Bar(
                        x=_sm_h[_tk_col], y=_hv_vals,
                        marker_color=_hv_clrs,
                        text=[f"{v:.2f}" for v in _hv_vals],
                        textposition="outside",
                    ))
                    fig_hv.update_layout(
                        title="IV / HV Ratio  (ATM implied vol ÷ 30-day realized vol)",
                        yaxis_title="IV/HV",
                        plot_bgcolor="white", paper_bgcolor="white",
                        font=dict(color="#111"), height=300,
                    )
                    fig_hv.add_hline(y=1.5, line_dash="dash", line_color="#b00020",
                                     annotation_text="Overpriced (>1.5)")
                    fig_hv.add_hline(y=0.7, line_dash="dash", line_color="#1e3a5f",
                                     annotation_text="Cheap (<0.7)")
                    fig_hv.add_hline(y=1.0, line_dash="dot", line_color="#888",
                                     annotation_text="Fair value")
                    st.plotly_chart(fig_hv, use_container_width=True)

                st.caption(
                    "**VIX Rank** = VIX 52-week percentile (market IV environment).  "
                    "**IV/HV** = ATM implied vol ÷ 30-day realized vol (stock-specific richness)."
                )

                # main table
                _sig_c = [c for c in ["signal"] if c in _sm.columns]
                _num_c = _sm.select_dtypes("number").columns.tolist()
                if _num_c: _sm[_num_c] = _sm[_num_c].round(3)
                _styl_sm = _sm.style.map(_signal_colour, subset=_sig_c) if _sig_c else _sm.style
                st.dataframe(_styl_sm, use_container_width=True, hide_index=True)
            else:
                st.info("Options summary is empty.")
        else:
            st.info("No options data yet. Run: `python3 canyon_final_v9_step74_options_chain.py`")

        if _rep_path and Path(_rep_path).exists():
            with st.expander("Options Chain Report"):
                st.markdown(Path(_rep_path).read_text())

    with o74b:
        st.subheader("Raw Chain Detail (top 5 tickers)")
        _det_path = FILES.get("options_chain_detail")
        if _det_path and Path(_det_path).exists():
            _det = pd.read_csv(_det_path)
            if not _det.empty:
                _num_d = _det.select_dtypes("number").columns.tolist()
                if _num_d: _det[_num_d] = _det[_num_d].round(4)
                st.dataframe(_det, use_container_width=True)
        else:
            st.info("No chain detail data yet.")

    with o74c:
        st.subheader("How to Run the Options Chain Engine")
        st.markdown("""
**Free — uses yfinance options chain, no Polygon API key needed.**

```bash
cd ~/Desktop/canyon_quant
# Full run (top 20 ML tickers + SPY + QQQ)
python3 canyon_final_v9_step74_options_chain.py

# Single ticker
python3 canyon_final_v9_step74_options_chain.py --ticker NVDA

# Force specific expiry date
python3 canyon_final_v9_step74_options_chain.py --expiry 2026-06-20
```

**Computed metrics:**
| Metric | Meaning |
|--------|---------|
| PCR (volume) | Put/call volume ratio — >1.2 = bearish hedge |
| PCR (OI) | Put/call open interest ratio |
| Max Pain | Strike where total option buyer loss is maximised |
| Gamma Wall | Strike with peak absolute dealer GEX proxy |
| IV Rank | Current IV vs 52-week range (0–100) |
| Kill Zone | Strikes within ±2% of price with OI > 500 |
| Pin Risk | Price within 1% of highest-OI strike |

**Signal logic:** BULLISH_OPTIONS · BEARISH_OPTIONS · IV_RICH · IV_CHEAP · PINNED · NEUTRAL

> Layer 7 is for context only — options signals cannot override L8 Portfolio Risk or L9 Before-Action Check.
""")
        col_run_opt, col_tick = st.columns(2)
        with col_run_opt:
            if st.button("▶ Run Options Chain", key="run_opt74", type="primary"):
                import subprocess
                _scr = Path(__file__).parent / "canyon_final_v9_step74_options_chain.py"
                with st.spinner("Running options chain engine (~60s)…"):
                    try:
                        _r = subprocess.run(["python3", str(_scr)],
                                            capture_output=True, text=True, timeout=180,
                                            cwd=str(Path(__file__).parent))
                        if _r.returncode == 0:
                            st.success("Options chain complete. Refresh Signal Summary tab.")
                            st.code(_r.stdout[-2000:])
                        else:
                            st.error("Engine error:")
                            st.code(_r.stderr[-2000:])
                    except Exception as _e:
                        st.error(f"Error: {_e}")
        with col_tick:
            _single_tk = st.text_input("Single-ticker run:", placeholder="NVDA", key="opt74_tk")
            if st.button("Run Single Ticker", key="run_opt74_single"):
                if _single_tk:
                    import subprocess
                    _scr = Path(__file__).parent / "canyon_final_v9_step74_options_chain.py"
                    with st.spinner(f"Fetching options for {_single_tk}…"):
                        try:
                            _r = subprocess.run(["python3", str(_scr), "--ticker", _single_tk.upper()],
                                                capture_output=True, text=True, timeout=60,
                                                cwd=str(Path(__file__).parent))
                            st.code(_r.stdout[-2000:])
                        except Exception as _e:
                            st.error(f"Error: {_e}")


def main():
    st.set_page_config(
        page_title="Canyon v9 10-Layer v2",
        page_icon="🏔",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

      /* ── Global font stack ─────────────────────────────────────────────── */
      html, body, [class*="css"], .stApp, button, input, select, textarea,
      div[data-testid="stMarkdownContainer"],
      div[data-testid="stMarkdownContainer"] p,
      div[data-testid="stMarkdownContainer"] span,
      div[data-testid="stMarkdownContainer"] div,
      div[data-testid="stCaptionContainer"],
      .stTabs, label, .stMarkdown, h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
      }

      section[data-testid="stSidebar"] { display:none; }
      div[data-testid="stSidebarCollapsedControl"] { display:none; }
      header[data-testid="stHeader"] { display:none; }
      #MainMenu { visibility:hidden; }
      div[data-testid="stElementToolbar"] { display:none; }
      .block-container {
        max-width: 1480px;
        padding-top: 3.2rem;
        padding-left: 2.4rem;
        padding-right: 2.4rem;
      }
      .terminal-header {
        display:grid; grid-template-columns:minmax(420px, 1.35fr) minmax(520px, 1fr);
        gap:18px; align-items:stretch; margin:0 0 12px 0;
        border:1px solid #253a55; border-radius:6px; background:#081422;
        box-shadow:0 0 0 1px rgba(56,189,248,0.08) inset;
      }
      .terminal-brand {
        padding:16px 18px; border-right:1px solid #253a55;
      }
      .terminal-kicker {
        color:#78d6ff; font-size:11px; font-weight:700; text-transform:uppercase;
        letter-spacing:0; margin-bottom:6px;
      }
      .terminal-title {
        color:#f2f8ff; font-size:30px; font-weight:760; line-height:1.08;
        margin-bottom:8px;
      }
      .terminal-subtitle {
        color:#aebfd4; font-size:13px; line-height:1.35;
      }
      .terminal-meta-grid {
        display:grid; grid-template-columns:repeat(4, minmax(0, 1fr));
      }
      .terminal-meta {
        padding:14px 12px; border-right:1px solid #253a55;
        display:flex; flex-direction:column; justify-content:center; gap:8px;
        background:#0d1a2b;
      }
      .terminal-meta:last-child { border-right:0; }
      .terminal-meta span {
        color:#8ea3bd; font-size:10px; text-transform:uppercase; letter-spacing:0;
      }
      .terminal-meta b {
        color:#edf5ff; font-size:16px; line-height:1.2; overflow-wrap:anywhere;
      }
      .terminal-tape {
        display:grid; grid-template-columns:repeat(6, minmax(0, 1fr));
        gap:0; border:1px solid #253a55; border-radius:6px; overflow:hidden;
        background:#081422; margin:0 0 12px 0;
      }
      .terminal-tape-item {
        min-height:46px; padding:8px 10px; border-right:1px solid #253a55;
        display:flex; flex-direction:column; justify-content:center; gap:4px;
      }
      .terminal-tape-item:last-child { border-right:0; }
      .terminal-tape-item span {
        color:#8ea3bd; font-size:10px; text-transform:uppercase; letter-spacing:0;
      }
      .terminal-tape-item b {
        color:#d7e4f2; font-size:13px; line-height:1.15;
      }
      .canyon-legend {
        display:flex; flex-wrap:wrap; gap:10px; align-items:center;
        padding:12px 14px; border:1px solid #d8dee8; border-radius:8px;
        margin:10px 0 18px 0; background:#fbfcfe;
      }
      .canyon-legend b { margin-right:4px; color:#202638; }
      .canyon-swatch {
        display:inline-flex; align-items:center; justify-content:center;
        min-height:26px; padding:4px 10px; border-radius:4px;
        font-size:12px; font-weight:500; line-height:1.15; white-space:nowrap;
        border:1px solid transparent; letter-spacing:0;
      }
      .canyon-supportive { background:#c9efd7; color:#14592f; border-color:#66c58d; }
      .canyon-watch { background:#c9efd7; color:#14592f; border-color:#66c58d; }
      .canyon-wait { background:#cfe5ff; color:#123f78; border-color:#6ba8ec; }
      .canyon-paper { background:#e3ceff; color:#4d2475; border-color:#a678dc; }
      .canyon-cyan { background:#c3f0f7; color:#0b5864; border-color:#5bc6d5; }
      .canyon-risk { background:#ffcaca; color:#842020; border-color:#e67676; }
      .canyon-weak { background:#e7d7d7; color:#6d3030; border-color:#bd9494; }
      .canyon-blocked { background:#dcdcdc; color:#464646; border-color:#a8a8a8; }
      .canyon-plain { background:#f7f8fb; color:#30384b; border-color:#d8dee8; }
      .product-shell {
        display:grid; grid-template-columns:minmax(320px, 0.9fr) minmax(640px, 1.7fr);
        gap:16px; align-items:stretch; border:1px solid #d8dee8; border-radius:8px;
        background:#fbfcfe; padding:16px; margin:0 0 18px 0;
      }
      .shell-left {
        border-right:1px solid #e5e9f0; padding-right:16px;
        display:flex; flex-direction:column; justify-content:center;
      }
      .shell-kicker {
        font-size:11px; text-transform:uppercase; letter-spacing:0; color:#667085;
        margin-bottom:6px;
      }
      .shell-title {
        font-size:24px; font-weight:680; line-height:1.15; color:#202638;
        margin-bottom:8px;
      }
      .shell-subtitle { font-size:13px; line-height:1.45; color:#4b5563; }
      .shell-grid {
        display:grid; grid-template-columns:repeat(6, minmax(0, 1fr)); gap:8px;
      }
      .shell-tile {
        min-height:82px; border:1px solid #d8dee8; border-radius:6px;
        padding:10px 12px; display:flex; flex-direction:column; justify-content:space-between;
        background:white;
      }
      .shell-label {
        font-size:11px; color:#667085; text-transform:uppercase; letter-spacing:0;
      }
      .shell-value {
        font-size:22px; font-weight:680; line-height:1.1; color:#202638;
        overflow-wrap:anywhere;
      }
      .shell-time { font-size:15px; line-height:1.2; }
      .shell-watch { background:#d8f3e1; border-color:#66c58d; }
      .shell-cyan { background:#d5f5fa; border-color:#5bc6d5; }
      .shell-risk { background:#ffd7d7; border-color:#e67676; }
      .shell-plain { background:#f7f8fb; border-color:#d8dee8; }
      .target-section { margin:6px 0 16px 0; }
      .target-section-title {
        font-size:14px; font-weight:650; color:#374151; margin:0 0 8px 0;
      }
      .target-grid {
        display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:10px;
      }
      .target-card {
        min-height:150px; border:1px solid #d8dee8; border-radius:6px;
        padding:12px 14px; display:flex; flex-direction:column; gap:7px;
        background:#fbfcfe;
      }
      .target-top {
        display:flex; justify-content:space-between; gap:8px; align-items:center;
        font-size:11px; color:#667085; text-transform:uppercase; letter-spacing:0;
      }
      .target-ticker {
        font-size:26px; font-weight:720; line-height:1; color:#202638;
      }
      .target-status {
        font-size:14px; font-weight:650; line-height:1.25; color:#202638;
        overflow-wrap:anywhere;
      }
      .target-line {
        font-size:12px; line-height:1.35; color:#374151; overflow-wrap:anywhere;
      }
      .target-supportive, .workflow-supportive { background:#e0f6e8; border-color:#66c58d; }
      .target-watch, .workflow-watch { background:#e0f6e8; border-color:#66c58d; }
      .target-wait, .workflow-wait { background:#e0efff; border-color:#6ba8ec; }
      .target-paper, .workflow-paper { background:#efddff; border-color:#a678dc; }
      .target-cyan, .workflow-cyan { background:#def8fc; border-color:#5bc6d5; }
      .target-risk, .workflow-risk { background:#ffe0e0; border-color:#e67676; }
      .target-blocked, .workflow-blocked { background:#e7e7e7; border-color:#a8a8a8; }
      .target-plain, .workflow-plain { background:#f7f8fb; border-color:#d8dee8; }
      .workflow-grid {
        display:grid; grid-template-columns:repeat(4, minmax(0, 1fr)); gap:10px;
        margin:8px 0 20px 0;
      }
      .workflow-card {
        min-height:150px; border:1px solid #d8dee8; border-radius:6px;
        padding:12px 14px; display:flex; flex-direction:column; gap:8px;
      }
      .workflow-top {
        display:flex; justify-content:space-between; align-items:center; gap:8px;
        font-size:11px; color:#667085; text-transform:uppercase; letter-spacing:0;
      }
      .workflow-title {
        font-size:17px; font-weight:680; color:#202638; line-height:1.2;
      }
      .workflow-text {
        font-size:13px; line-height:1.4; color:#374151;
      }
      .workflow-route {
        margin-top:auto; padding-top:8px; border-top:1px solid rgba(32,38,56,0.12);
        font-size:12px; color:#667085;
      }
      .canyon-table-wrap {
        overflow:auto; border:1px solid #d8dee8; border-radius:8px; background:white;
      }
      .canyon-table {
        width:100%; min-width:1120px; border-collapse:separate; border-spacing:0;
        font-size:13px; color:#202638;
      }
      .canyon-table th {
        position:sticky; top:0; z-index:1; background:#f5f7fb; color:#374151;
        text-align:left; padding:8px 10px; border-bottom:1px solid #d8dee8;
        font-size:12px; text-transform:uppercase; letter-spacing:0;
      }
      .canyon-table td {
        padding:8px 10px; border-bottom:1px solid #e5e9f0; vertical-align:middle;
        background:white;
      }
      .canyon-table tr:nth-child(even) td { background:#fbfcfe; }
      /* 2-color badge system: red = danger, blue = noteworthy, no color = neutral */
      .canyon-table td.canyon-status-cell.canyon-risk,
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-risk
        { background:#fee2e2; color:#991b1b; font-weight:600; }
      .canyon-table td.canyon-status-cell.canyon-ok,
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-ok,
      .canyon-table td.canyon-status-cell.canyon-warn,
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-warn,
      .canyon-table td.canyon-status-cell.canyon-supportive,
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-supportive,
      .canyon-table td.canyon-status-cell.canyon-watch,
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-watch,
      .canyon-table td.canyon-status-cell.canyon-cyan,
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-cyan,
      .canyon-table td.canyon-status-cell.canyon-wait,
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-wait,
      .canyon-table td.canyon-status-cell.canyon-paper,
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-paper
        { background:#dbeafe; color:#1e40af; font-weight:600; }
      .canyon-table td.canyon-status-cell.canyon-plain,
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-plain,
      .canyon-table td.canyon-status-cell.canyon-blocked,
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-blocked,
      .canyon-table td.canyon-status-cell.canyon-weak,
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-weak
        { background:transparent; color:#9ca3af; }
      .canyon-table .canyon-reason { max-width:360px; min-width:260px; line-height:1.35; }
      .ops-strip {
        display:grid; grid-template-columns:repeat(6, minmax(0, 1fr));
        border:1px solid #d8dee8; border-radius:8px; overflow:hidden;
        background:white; margin:10px 0 20px 0;
      }
      .ops-cell {
        min-height:108px; padding:14px 16px; border-right:1px solid #e5e9f0;
        display:flex; flex-direction:column; justify-content:space-between;
      }
      .ops-cell:last-child { border-right:0; }
      .ops-label {
        font-size:11px; text-transform:uppercase; letter-spacing:0;
        color:#667085; line-height:1.2;
      }
      .ops-value {
        font-size:30px; font-weight:680; color:#202638; line-height:1.05;
        overflow-wrap:anywhere;
      }
      .ops-note {
        font-size:12px; line-height:1.35; color:#4b5563;
      }
      .ops-risk { background:#ffe0e0; }
      .ops-cyan { background:#def8fc; }
      .ops-paper { background:#efddff; }
      .ops-blocked { background:#e7e7e7; }
      .ops-watch { background:#e0f6e8; }
      /* layer-workbench-head: replaced by inline styles in render_layer_workbench_header */
      .command-grid {
        display:block; margin:10px 0 22px 0; border-top:1px solid #e5e9f0;
      }
      .command-panel {
        min-height:0; padding:10px 0 12px 0; border:0; border-bottom:1px solid #e5e9f0;
        border-radius:0; background:transparent;
      }
      .command-label {
        font-size:12px; text-transform:none; letter-spacing:0;
        color:#667085; margin-bottom:4px;
      }
      .command-title {
        font-size:16px; font-weight:650; color:#202638; line-height:1.3;
        margin-bottom:4px;
      }
      .command-text {
        font-size:13px; color:#374151; line-height:1.45;
      }
      .command-risk, .command-paper, .command-cyan, .command-blocked, .command-watch {
        background:transparent; border-color:#e5e9f0;
      }
      .ticker-hero {
        display:flex; justify-content:space-between; gap:18px; align-items:stretch;
        border:1px solid #d8dee8; border-radius:6px; padding:18px 20px;
        margin:12px 0 18px 0;
      }
      .ticker-label, .ticker-score-label {
        font-size:11px; text-transform:uppercase; letter-spacing:0; color:#667085;
        margin-bottom:8px;
      }
      .ticker-title {
        font-size:26px; font-weight:650; line-height:1.2; color:#202638;
        margin-bottom:8px;
      }
      .ticker-reason { font-size:14px; line-height:1.45; color:#374151; }
      .ticker-scorebox {
        min-width:160px; border-left:1px solid rgba(32, 38, 56, 0.16);
        padding-left:18px;
      }
      .ticker-score { font-size:24px; color:#202638; }
      .ticker-paper, .layer-paper { background:#efddff; border-color:#a678dc; }
      .ticker-risk, .layer-risk { background:#ffd7d7; border-color:#e67676; }
      .ticker-cyan, .layer-cyan { background:#def8fc; border-color:#5bc6d5; }
      .ticker-blocked, .layer-blocked { background:#e7e7e7; border-color:#a8a8a8; }
      .ticker-supportive, .layer-supportive,
      .ticker-watch, .layer-watch { background:#d8f3e1; border-color:#66c58d; }
      .ticker-wait, .layer-wait { background:#e0efff; border-color:#6ba8ec; }
      .ticker-weak, .layer-weak { background:#e7d7d7; border-color:#bd9494; }
      .ticker-plain, .layer-plain { background:#f7f8fb; border-color:#d8dee8; }
      .ticker-rule {
        border:1px solid #d8dee8; border-radius:6px; padding:14px 16px;
        margin:10px 0 18px 0;
      }
      .ticker-rule-text {
        font-size:18px; font-weight:650; line-height:1.35; color:#202638;
      }
      .layer-grid {
        display:grid; grid-template-columns:repeat(5, minmax(0, 1fr)); gap:10px;
        margin:8px 0 22px 0;
      }
      .layer-card {
        min-height:150px; border:1px solid #d8dee8; border-radius:6px; padding:12px;
      }
      .layer-head {
        display:flex; justify-content:space-between; gap:8px; align-items:center;
        margin-bottom:10px;
      }
      .layer-name {
        font-size:13px; font-weight:650; color:#202638;
      }
      .layer-score {
        font-size:12px; color:#667085;
      }
      .layer-state {
        font-size:14px; color:#202638; line-height:1.25; margin-bottom:8px;
        overflow-wrap:anywhere;
      }
      .layer-note {
        font-size:12px; color:#4b5563; line-height:1.35; overflow-wrap:anywhere;
      }
      @media (max-width: 900px) {
        .block-container { padding-left:1rem; padding-right:1rem; }
        .terminal-header { grid-template-columns:1fr; }
        .terminal-brand { border-right:0; border-bottom:1px solid #253a55; }
        .terminal-meta-grid { grid-template-columns:1fr 1fr; }
        .product-shell { grid-template-columns:1fr; }
        .shell-left { border-right:0; border-bottom:1px solid #e5e9f0; padding-right:0; padding-bottom:12px; }
        .shell-grid { grid-template-columns:1fr 1fr; }
        .target-grid, .workflow-grid { grid-template-columns:1fr 1fr; }
        .ops-strip { grid-template-columns:1fr 1fr; }
        .ops-cell { border-bottom:1px solid #e5e9f0; }
        .ops-cell:nth-child(even) { border-right:0; }
        .layer-workbench-head { grid-template-columns:1fr; }
        .layer-metric-grid { grid-template-columns:1fr 1fr; }
        .command-grid { grid-template-columns:1fr; }
        .layer-grid { grid-template-columns:1fr; }
        .ticker-hero { flex-direction:column; }
        .ticker-scorebox { border-left:0; padding-left:0; border-top:1px solid rgba(32, 38, 56, 0.16); padding-top:12px; }
      }

      /* Cold professional terminal theme */
      .stApp {
        background:#07111f;
        color:#d7e4f2;
      }
      .block-container {
        max-width:1580px;
        background:linear-gradient(180deg, #07111f 0%, #0a1626 100%);
      }
      h1, h2, h3, h4, h5, h6,
      div[data-testid="stMarkdownContainer"],
      div[data-testid="stMarkdownContainer"] p,
      div[data-testid="stMarkdownContainer"] li,
      div[data-testid="stCaptionContainer"],
      label, .stMarkdown {
        color:#d7e4f2;
      }
      h1 {
        color:#edf5ff;
        font-weight:720;
      }
      h2, h3 {
        color:#e5f0ff;
        letter-spacing:0;
      }
      div[data-testid="stMetric"] {
        background:#0e1b2c;
        border:1px solid #21354d;
        border-radius:6px;
        padding:12px 14px;
      }
      div[data-testid="stMetricLabel"] {
        color:#90a4bc;
      }
      div[data-testid="stMetricValue"] {
        color:#f0f7ff;
      }
      div[data-testid="stVerticalBlockBorderWrapper"] {
        background:#0e1b2c;
        border-color:#2a405d;
        border-radius:6px;
      }
      button[data-baseweb="tab"] {
        color:#aebfd4;
        background:transparent;
        border-radius:0;
      }
      button[data-baseweb="tab"][aria-selected="true"] {
        color:#8fdcff;
        border-bottom-color:#38bdf8;
      }
      div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap:0;
        border-bottom:1px solid #253a55;
      }
      div[data-testid="stTabs"] button[data-baseweb="tab"] {
        padding-top:8px;
        padding-bottom:8px;
        border-right:1px solid rgba(37,58,85,0.55);
      }
      .canyon-legend,
      .product-shell,
      .layer-workbench-head,
      .ticker-hero,
      .ticker-rule {
        background:#0d1a2b;
        border-color:#253a55;
        box-shadow:0 0 0 1px rgba(86, 128, 170, 0.08) inset;
      }
      .shell-left {
        border-right-color:#253a55;
      }
      .shell-kicker,
      .shell-label,
      .target-top,
      .workflow-top,
      .ops-label,
      .layer-metric-label,
      .ticker-label,
      .ticker-score-label,
      .layer-score,
      .workflow-route,
      .command-label {
        color:#8ea3bd;
      }
      .shell-title,
      .layer-workbench-title,
      .ticker-title,
      .target-ticker,
      .target-status,
      .workflow-title,
      .ops-value,
      .layer-metric-value,
      .ticker-score,
      .ticker-rule-text,
      .layer-name,
      .layer-state,
      .command-title {
        color:#edf5ff;
      }
      .shell-subtitle,
      .target-line,
      .target-reason,
      .workflow-text,
      .ops-note,
      .layer-workbench-thesis,
      .ticker-reason,
      .layer-note,
      .command-text {
        color:#b7c8dc;
      }
      .shell-tile,
      .target-card,
      .workflow-card,
      .layer-metric,
      .layer-card {
        background:#0f1e31;
        border-color:#2a405d;
      }
      .shell-plain,
      .target-plain,
      .workflow-plain,
      .layer-metric-plain,
      .ticker-plain,
      .layer-plain,
      .canyon-plain {
        background:#111f32;
        color:#d7e4f2;
        border-color:#2a405d;
      }
      .shell-watch, .target-watch, .workflow-watch, .layer-metric-watch,
      .target-supportive, .workflow-supportive, .layer-metric-supportive,
      .ticker-watch, .layer-watch, .ticker-supportive, .layer-supportive,
      .canyon-watch, .canyon-supportive {
        background:#123421;
        color:#9df0ba;
        border-color:#278455;
      }
      .shell-wait, .target-wait, .workflow-wait, .layer-metric-wait,
      .ticker-wait, .layer-wait, .canyon-wait {
        background:#102b4a;
        color:#9fd0ff;
        border-color:#2f74b9;
      }
      .shell-paper, .target-paper, .workflow-paper, .layer-metric-paper,
      .ticker-paper, .layer-paper, .canyon-paper {
        background:#211a43;
        color:#d1b8ff;
        border-color:#7659c7;
      }
      .shell-cyan, .target-cyan, .workflow-cyan, .layer-metric-cyan,
      .ticker-cyan, .layer-cyan, .canyon-cyan {
        background:#0b3542;
        color:#83ecff;
        border-color:#199ab0;
      }
      .shell-risk, .target-risk, .workflow-risk, .layer-metric-risk,
      .ticker-risk, .layer-risk, .canyon-risk {
        background:#351923;
        color:#ffadb6;
        border-color:#bb5063;
      }
      .target-blocked, .workflow-blocked, .layer-metric-blocked,
      .ticker-blocked, .layer-blocked, .canyon-blocked {
        background:#202b3a;
        color:#c5d0de;
        border-color:#50657d;
      }
      .target-weak, .workflow-weak, .layer-metric-weak,
      .ticker-weak, .layer-weak, .canyon-weak {
        background:#2f2431;
        color:#dcb8cf;
        border-color:#7a5871;
      }
      .canyon-table-wrap {
        background:#0b1727;
        border-color:#2a405d;
      }
      .canyon-table {
        color:#d7e4f2;
      }
      .canyon-table th {
        background:#111f32;
        color:#9fb4cc;
        border-bottom-color:#2a405d;
      }
      .canyon-table td {
        background:#0d1a2b;
        color:#d7e4f2;
        border-bottom-color:#1d3148;
      }
      .canyon-table tr:nth-child(even) td {
        background:#0f1e31;
      }
      /* dark mode: red = danger, blue = noteworthy */
      .canyon-table td.canyon-status-cell.canyon-risk,
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-risk
        { background:#4c1520; color:#fca5a5; }
      .canyon-table td.canyon-status-cell.canyon-ok,
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-ok,
      .canyon-table td.canyon-status-cell.canyon-warn,
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-warn,
      .canyon-table td.canyon-status-cell.canyon-supportive,
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-supportive,
      .canyon-table td.canyon-status-cell.canyon-watch,
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-watch,
      .canyon-table td.canyon-status-cell.canyon-cyan,
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-cyan,
      .canyon-table td.canyon-status-cell.canyon-wait,
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-wait,
      .canyon-table td.canyon-status-cell.canyon-paper,
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-paper
        { background:#1e3a5f; color:#93c5fd; }
      .canyon-table td.canyon-status-cell.canyon-plain,
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-plain,
      .canyon-table td.canyon-status-cell.canyon-blocked,
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-blocked,
      .canyon-table td.canyon-status-cell.canyon-weak,
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-weak
        { background:transparent; color:#6b7280; }
      .ops-strip {
        background:#0d1a2b;
        border-color:#253a55;
      }
      .ops-cell {
        border-right-color:#253a55;
      }
      .ops-risk { background:#351923; }
      .ops-cyan { background:#0b3542; }
      .ops-paper { background:#211a43; }
      .ops-blocked { background:#202b3a; }
      .ops-watch { background:#123421; }
      .command-grid {
        border-top-color:#253a55;
      }
      .command-panel {
        border-bottom-color:#253a55;
      }
      div[data-testid="stExpander"] details {
        background:#0d1a2b;
        border:1px solid #253a55;
        border-radius:6px;
      }
      div[data-testid="stExpander"] summary {
        color:#cfe3fb;
      }
      .stCodeBlock, pre {
        background:#050b13 !important;
        color:#d7e4f2 !important;
        border:1px solid #253a55;
      }


      /* Final clean black-and-white workspace skin */
      .stApp {
        background:#fdfcfb !important;
        color:#2b2f3a !important;
      }
      .block-container {
        max-width:1540px !important;
        background:#fdfcfb !important;
        padding-top:3.8rem !important;
        padding-left:3.2rem !important;
        padding-right:3.2rem !important;
      }
      h1, h2, h3, h4, h5, h6,
      div[data-testid="stMarkdownContainer"],
      div[data-testid="stMarkdownContainer"] p,
      div[data-testid="stMarkdownContainer"] li,
      div[data-testid="stCaptionContainer"],
      label, .stMarkdown {
        color:#2b2f3a !important;
      }
      h1 { color:#2b2f3a !important; font-weight:720 !important; }
      h2, h3 { color:#2b2f3a !important; letter-spacing:0 !important; }
      .terminal-header {
        background:#ffffff !important;
        border:0 !important;
        border-radius:0 !important;
        box-shadow:none !important;
        margin:0 0 18px 0 !important;
        gap:28px !important;
      }
      .terminal-brand {
        padding:8px 0 14px 0 !important;
        border-right:0 !important;
      }
      .terminal-kicker {
        color:#6b7280 !important;
        font-size:11px !important;
        letter-spacing:0 !important;
      }
      .terminal-title {
        color:#2b2f3a !important;
        font-size:32px !important;
        font-weight:720 !important;
        line-height:1.12 !important;
      }
      .terminal-subtitle {
        color:#6b7280 !important;
        font-size:14px !important;
      }
      .terminal-meta-grid {
        gap:12px !important;
      }
      .terminal-meta,
      div[data-testid="stMetric"],
      div[data-testid="stVerticalBlockBorderWrapper"] {
        background:#ffffff !important;
        border:1px solid #d7dbe3 !important;
        border-radius:6px !important;
        box-shadow:none !important;
      }
      .terminal-meta {
        padding:13px 14px !important;
      }
      .terminal-meta span,
      .terminal-tape-item span,
      div[data-testid="stMetricLabel"] {
        color:#6b7280 !important;
      }
      .terminal-meta b,
      .terminal-tape-item b,
      div[data-testid="stMetricValue"] {
        color:#2b2f3a !important;
      }
      .terminal-tape {
        background:#ffffff !important;
        border:1px solid #d7dbe3 !important;
        border-radius:6px !important;
        margin:0 0 20px 0 !important;
      }
      .terminal-tape-item {
        border-right:1px solid #e1e5ec !important;
        padding:10px 14px !important;
        min-height:54px !important;
      }
      div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap:18px !important;
        border-bottom:1px solid #d7dbe3 !important;
        padding-top:10px !important;
        padding-bottom:8px !important;
        overflow-x:auto !important;
      }
      div[data-testid="stTabs"] button[data-baseweb="tab"] {
        color:#2b2f3a !important;
        background:transparent !important;
        border-right:0 !important;
        border-radius:0 !important;
        padding:10px 2px 12px 2px !important;
        margin-right:8px !important;
        min-width:max-content !important;
        font-size:15px !important;
      }
      div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        color:#111827 !important;
        border-bottom:2px solid #111827 !important;
      }
      .canyon-legend,
      .product-shell,
      .layer-workbench-head,
      .ticker-hero,
      .ticker-rule,
      .canyon-table-wrap,
      .ops-strip,
      div[data-testid="stExpander"] details {
        background:#ffffff !important;
        border-color:#d7dbe3 !important;
        box-shadow:none !important;
      }
      .canyon-legend {
        margin:14px 0 24px 0 !important;
        padding:12px 16px !important;
        gap:12px !important;
      }
      .shell-tile,
      .target-card,
      .workflow-card,
      .layer-metric,
      .layer-card {
        background:#ffffff !important;
        border-color:#d7dbe3 !important;
        box-shadow:none !important;
      }
      /* Sharp corners everywhere — no rounded boxes */
      .shell-tile, .target-card, .workflow-card, .layer-metric, .layer-card,
      .canyon-legend, .product-shell, .layer-workbench-head, .ticker-hero,
      .ticker-rule, .canyon-table-wrap, .ops-strip, .canyon-swatch,
      div[data-testid="stExpander"] details, .shell-tile,
      div[data-testid="stMetric"],
      div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 0 !important;
      }
      .shell-kicker,
      .shell-label,
      .target-top,
      .workflow-top,
      .ops-label,
      .layer-metric-label,
      .ticker-label,
      .ticker-score-label,
      .layer-score,
      .workflow-route,
      .command-label {
        color:#6b7280 !important;
      }
      .shell-title,
      .layer-workbench-title,
      .ticker-title,
      .target-ticker,
      .target-status,
      .workflow-title,
      .ops-value,
      .layer-metric-value,
      .ticker-score,
      .ticker-rule-text,
      .layer-name,
      .layer-state,
      .command-title {
        color:#2b2f3a !important;
      }
      .shell-subtitle,
      .target-line,
      .target-reason,
      .workflow-text,
      .ops-note,
      .layer-workbench-thesis,
      .ticker-reason,
      .layer-note,
      .command-text {
        color:#4b5563 !important;
      }
      .target-grid,
      .workflow-grid,
      .layer-metric-grid,
      .layer-grid,
      .shell-grid {
        gap:28px !important;
      }
      .workflow-card,
      .target-card {
        min-height:190px !important;
        padding:20px 22px !important;
      }
      .layer-workbench-head,
      .product-shell {
        padding:28px !important;
        margin-bottom:36px !important;
      }
      .command-grid {
        border-top:1px solid #e1e5ec !important;
        margin:32px 0 56px 0 !important;
      }
      .command-panel {
        background:transparent !important;
        border-bottom:1px solid #e1e5ec !important;
        padding:22px 0 24px 0 !important;
      }
      .canyon-table {
        color:#2b2f3a !important;
        font-size:13px !important;
      }
      .canyon-table th {
        background:#f7f8fa !important;
        color:#374151 !important;
        border-bottom-color:#d7dbe3 !important;
        padding:11px 13px !important;
      }
      .canyon-table td {
        background:#ffffff !important;
        color:#2b2f3a !important;
        border-bottom-color:#e5e7eb !important;
        padding:11px 13px !important;
      }
      .canyon-table tr:nth-child(even) td {
        background:#fafafa !important;
      }
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-supportive,
      .canyon-table td.canyon-status-cell.canyon-supportive,
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-watch,
      .canyon-table td.canyon-status-cell.canyon-watch {
        background:#e7f5ec !important;
        color:#14592f !important;
      }
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-wait,
      .canyon-table td.canyon-status-cell.canyon-wait {
        background:#e8f2ff !important;
        color:#123f78 !important;
      }
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-paper,
      .canyon-table td.canyon-status-cell.canyon-paper {
        background:#f0e8fb !important;
        color:#4d2475 !important;
      }
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-cyan,
      .canyon-table td.canyon-status-cell.canyon-cyan {
        background:#e4f7fa !important;
        color:#0b5864 !important;
      }
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-risk,
      .canyon-table td.canyon-status-cell.canyon-risk {
        background:#fae7e7 !important;
        color:#842020 !important;
      }
      .canyon-table tr:nth-child(even) td.canyon-status-cell.canyon-blocked,
      .canyon-table td.canyon-status-cell.canyon-blocked {
        background:#eeeeee !important;
        color:#464646 !important;
      }
      .ops-cell {
        border-right-color:#e1e5ec !important;
      }
      .stCodeBlock, pre {
        background:#f7f8fa !important;
        color:#111827 !important;
        border:1px solid #d7dbe3 !important;
      }
      .classic-header {
        margin:0 0 20px 0 !important;
        padding:0 !important;
        background:transparent !important;
        border:0 !important;
      }
      .classic-title {
        font-family: 'Inter', sans-serif !important;
        font-size:28px !important;
        font-weight:800 !important;
        line-height:1.15 !important;
        letter-spacing:-0.03em !important;
        color:#111827 !important;
        margin:0 0 6px 0 !important;
      }
      .classic-subtitle {
        font-family: 'Inter', sans-serif !important;
        font-size:13px !important;
        font-weight:400 !important;
        line-height:1.5 !important;
        color:#9ca3af !important;
        letter-spacing:0 !important;
        margin:0 !important;
      }
      .terminal-header, .terminal-tape {
        display:none !important;
      }
      div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap:24px !important;
        border-bottom:1px solid #d7dbe3 !important;
        padding-top:14px !important;
        padding-bottom:0 !important;
      }
      div[data-testid="stTabs"] button[data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-weight:500 !important;
        font-size:13px !important;
        letter-spacing:0 !important;
        margin-right:4px !important;
        padding:10px 0 12px 0 !important;
        font-size:15px !important;
        white-space:nowrap !important;
      }
      div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        color:#ff4b4b !important;
        border-bottom:2px solid #ff4b4b !important;
      }
      @media (max-width: 900px) {
        .block-container { padding-left:1.25rem !important; padding-right:1.25rem !important; }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:12px !important; }
        div[data-testid="stTabs"] button[data-baseweb="tab"] { font-size:14px !important; margin-right:4px !important; }
      }



      /* Restore original light Streamlit-style dashboard polish */
      .block-container {
        max-width:1320px !important;
        padding-top:3.4rem !important;
        padding-left:2.4rem !important;
        padding-right:2.4rem !important;
      }
      .classic-header {
        margin:0 0 22px 0 !important;
      }
      .classic-title {
        font-size:34px !important;
        font-weight:720 !important;
        line-height:1.15 !important;
        letter-spacing:0 !important;
        color:#2b2f3a !important;
        margin-bottom:14px !important;
      }
      .classic-subtitle {
        font-size:14px !important;
        color:#6b7280 !important;
      }
      .canyon-legend {
        background:#ffffff !important;
        border:1px solid #d8dce3 !important;
        border-radius:9px !important;
        padding:8px 12px !important;
        margin:20px 0 14px 0 !important;
        gap:8px !important;
        box-shadow:none !important;
      }
      .canyon-legend b {
        color:#2b2f3a !important;
        font-size:13px !important;
        font-weight:650 !important;
        margin-right:4px !important;
      }
      .canyon-swatch {
        min-height:24px !important;
        padding:3px 9px !important;
        border-radius:4px !important;
        font-size:12px !important;
        font-weight:500 !important;
        box-shadow:none !important;
      }
      .canyon-supportive,
      .canyon-watch {
        background:#dff5e8 !important;
        color:#235b3b !important;
        border-color:#a7dec0 !important;
      }
      .canyon-cyan {
        background:#d8f5f9 !important;
        color:#0c5460 !important;
        border-color:#7dd0db !important;
      }
      .canyon-wait {
        background:#e8f2ff !important;
        color:#264f82 !important;
        border-color:#bcd8f6 !important;
      }
      .canyon-paper {
        background:#f2e8ff !important;
        color:#5b3379 !important;
        border-color:#d5b8ef !important;
      }
      .canyon-risk {
        background:#ffecec !important;
        color:#7d2f2f !important;
        border-color:#efb1b1 !important;
      }
      .canyon-blocked {
        background:#eeeeee !important;
        color:#4f5661 !important;
        border-color:#d4d4d4 !important;
      }
      div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap:0 !important;
        border-bottom:1px solid #e3e6eb !important;
        padding-top:8px !important;
        padding-bottom:0 !important;
      }
      div[data-testid="stTabs"] button[data-baseweb="tab"] {
        background:transparent !important;
        color:#2f3440 !important;
        border-right:0 !important;
        border-bottom:0 !important;
        border-radius:0 !important;
        margin-right:18px !important;
        padding:9px 0 12px 0 !important;
        font-size:13px !important;
        font-weight:500 !important;
        line-height:1.2 !important;
        box-shadow:none !important;
      }
      div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        color:#ff4b4b !important;
        border-bottom:0 !important;
        box-shadow:none !important;
      }
      div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        background:#ff4b4b !important;
        height:2px !important;
      }
      div[data-testid="stTabs"] button[data-baseweb="tab"]::before,
      div[data-testid="stTabs"] button[data-baseweb="tab"]::after {
        display:none !important;
      }
      div[data-testid="stTabs"] [role="tabpanel"] {
        padding-top:18px !important;
      }
      h1 {
        font-size:32px !important;
        line-height:1.15 !important;
      }
      h2 {
        font-size:26px !important;
      }
      h3 {
        font-size:22px !important;
      }
      .canyon-table th {
        background:#ffffff !important;
        color:#2b2f3a !important;
        font-size:12px !important;
        padding:8px 10px !important;
      }
      .canyon-table td {
        padding:8px 10px !important;
      }



      .workflow-route-shell {
        border:1px solid #d8dce3 !important;
        border-radius:8px !important;
        background:#ffffff !important;
        padding:18px 20px !important;
        margin:8px 0 16px 0 !important;
      }
      .workflow-route-head {
        display:grid !important;
        grid-template-columns:minmax(0, 1.4fr) minmax(360px, 0.8fr) !important;
        gap:18px !important;
        align-items:stretch !important;
      }
      .workflow-route-kicker {
        color:#6b7280 !important;
        font-size:11px !important;
        font-weight:650 !important;
        text-transform:uppercase !important;
        margin-bottom:6px !important;
      }
      .workflow-route-title {
        color:#2b2f3a !important;
        font-size:24px !important;
        font-weight:680 !important;
        line-height:1.15 !important;
        margin-bottom:6px !important;
      }
      .workflow-route-text {
        color:#4b5563 !important;
        font-size:13px !important;
        line-height:1.45 !important;
      }
      .workflow-route-metrics {
        display:grid !important;
        grid-template-columns:repeat(4, minmax(0, 1fr)) !important;
        gap:8px !important;
      }
      .workflow-route-metrics div {
        border:1px solid #e1e5ec !important;
        border-radius:6px !important;
        padding:10px 12px !important;
        background:#fafafa !important;
      }
      .workflow-route-metrics span {
        display:block !important;
        color:#6b7280 !important;
        font-size:10px !important;
        text-transform:uppercase !important;
        margin-bottom:6px !important;
      }
      .workflow-route-metrics b {
        color:#2b2f3a !important;
        font-size:22px !important;
        line-height:1 !important;
      }
      .workflow-first-gate {
        border-top:1px solid #e1e5ec !important;
        margin-top:14px !important;
        padding-top:12px !important;
        color:#4b5563 !important;
        font-size:13px !important;
      }
      @media (max-width: 900px) {
        .workflow-route-head { grid-template-columns:1fr !important; }
        .workflow-route-metrics { grid-template-columns:1fr 1fr !important; }
      }

      /* ── IBM Plex Mono for all numeric values site-wide ─────────────────── */
      .layer-metric-value,
      .ticker-score,
      .workflow-route-metrics b,
      .ops-value,
      .target-status {
        font-family: 'IBM Plex Mono', 'Roboto Mono', 'Courier New', monospace !important;
      }

      /* ── Premium workbench header ─────────────────────────────────────────── */
      .layer-workbench-head {
        background: #ffffff !important;
        border: 0 !important;
        border-top: 2px solid #111827 !important;
        border-radius: 0 !important;
        padding: 14px 0 18px 0 !important;
        margin: 0 0 20px 0 !important;
        display: grid !important;
        grid-template-columns: minmax(0, 1.4fr) minmax(440px, 0.9fr) !important;
        gap: 24px !important;
      }
      .layer-workbench-title {
        font-family: 'Inter', sans-serif !important;
        font-size: 24px !important;
        font-weight: 800 !important;
        letter-spacing: -0.025em !important;
        line-height: 1.15 !important;
        color: #111827 !important;
        margin-bottom: 6px !important;
      }
      .ticker-label, .layer-metric-label {
        font-family: 'Inter', sans-serif !important;
        font-size: 10px !important;
        font-weight: 700 !important;
        letter-spacing: 0.10em !important;
        text-transform: uppercase !important;
        color: #4b5563 !important;
      }
      .layer-workbench-thesis {
        font-family: 'Inter', sans-serif !important;
        font-size: 13px !important;
        font-weight: 400 !important;
        color: #4b5563 !important;
        line-height: 1.5 !important;
        max-width: 680px !important;
      }
      .layer-metric {
        border: 0 !important;
        border-radius: 0 !important;
        padding: 8px 0 8px 14px !important;
        border-left: 3px solid #d1d5db !important;
        background: transparent !important;
        min-height: 0 !important;
      }
      .layer-metric-grid {
        display: grid !important;
        grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
        gap: 0 12px !important;
        align-items: start !important;
      }
      .layer-metric-value {
        font-size: 20px !important;
        font-weight: 600 !important;
        color: #111827 !important;
        line-height: 1.1 !important;
        margin-top: 4px !important;
      }
      .layer-metric-supportive, .layer-metric-watch {
        border-left-color: #34d399 !important;
        background: transparent !important;
      }
      .layer-metric-risk { border-left-color: #f87171 !important; background: transparent !important; }
      .layer-metric-cyan { border-left-color: #22d3ee !important; background: transparent !important; }
      .layer-metric-wait { border-left-color: #60a5fa !important; background: transparent !important; }
      .layer-metric-paper { border-left-color: #c084fc !important; background: transparent !important; }
      .layer-metric-blocked { border-left-color: #9ca3af !important; background: transparent !important; }
      .layer-metric-plain { border-left-color: #d1d5db !important; background: transparent !important; }

      /* ── Premium ticker hero ─────────────────────────────────────────────── */
      .ticker-hero {
        border: 0 !important;
        border-top: 2px solid #111827 !important;
        border-radius: 0 !important;
        padding: 14px 0 18px 0 !important;
        background: transparent !important;
      }
      .ticker-title {
        font-family: 'Inter', sans-serif !important;
        font-size: 26px !important;
        font-weight: 800 !important;
        letter-spacing: -0.025em !important;
        color: #111827 !important;
        margin-bottom: 6px !important;
      }
      .ticker-reason {
        font-size: 13px !important;
        color: #4b5563 !important;
      }
      .ticker-scorebox {
        border-left: 1px solid #e5e7eb !important;
        padding-left: 20px !important;
        min-width: 140px !important;
      }
      .ticker-score {
        font-size: 22px !important;
        font-weight: 600 !important;
        color: #111827 !important;
        margin-top: 4px !important;
      }
      .ticker-paper, .ticker-risk, .ticker-cyan, .ticker-blocked,
      .ticker-supportive, .ticker-watch, .ticker-wait, .ticker-weak, .ticker-plain {
        background: transparent !important;
        border-top-width: 2px !important;
      }
      .ticker-risk { border-top-color: #f87171 !important; }
      .ticker-supportive, .ticker-watch { border-top-color: #34d399 !important; }
      .ticker-paper { border-top-color: #c084fc !important; }
      .ticker-wait { border-top-color: #60a5fa !important; }
      .ticker-cyan { border-top-color: #22d3ee !important; }
      .ticker-blocked { border-top-color: #9ca3af !important; }

      /* ── Sharper section titles in tabs ──────────────────────────────────── */
      h1, h2, h3 {
        letter-spacing: -0.02em !important;
        font-weight: 800 !important;
      }
      h2 { font-size: 24px !important; color: #111827 !important; }
      h3 { font-size: 18px !important; color: #1f2937 !important; }
      p, li { color: #374151 !important; line-height: 1.55 !important; }

      /* ══ Breathing room — more space between every section ══════════════════

         Three levers:
           1. Page edges — bigger outer padding so content isn't crushed
           2. Tab panels — larger gap below the tab bar before content starts
           3. Section components — workbench headers, cards, and grids breathe
      ═══════════════════════════════════════════════════════════════════════ */

      /* 1 — Page outer padding */
      .block-container {
        padding-top: 5.5rem !important;
        padding-left: 5rem !important;
        padding-right: 5rem !important;
      }

      /* 2 — Tab panel breathing (top-level and nested) */
      div[data-testid="stTabs"] [role="tabpanel"] {
        padding-top: 48px !important;
      }
      div[data-testid="stTabs"] [role="tabpanel"]
        div[data-testid="stTabs"] [role="tabpanel"] {
        padding-top: 36px !important;
      }

      /* 3 — Streamlit vertical block inter-element gap */
      div[data-testid="stVerticalBlock"] {
        gap: 28px !important;
      }
      /* Columns sit inside a horizontal block — slightly tighter but still roomy */
      div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] {
        gap: 20px !important;
      }

      /* 4 — Section-level component margins */
      .layer-workbench-head {
        margin: 0 0 52px 0 !important;
        padding: 28px 0 36px 0 !important;
      }
      .ticker-hero {
        padding: 28px 0 36px 0 !important;
        margin-bottom: 52px !important;
      }
      .classic-header {
        margin: 0 0 48px 0 !important;
      }
      .command-grid {
        margin: 32px 0 56px 0 !important;
        gap: 28px !important;
      }
      .command-panel {
        padding: 22px 0 24px 0 !important;
      }
      .workflow-route-shell {
        margin: 20px 0 40px 0 !important;
        padding: 28px 32px !important;
      }

      /* 5 — Grid gaps — wider cells and wider gutters */
      .target-grid, .workflow-grid, .shell-grid, .layer-grid {
        gap: 28px !important;
      }
      .layer-metric-grid {
        gap: 0 36px !important;
      }

      /* 6 — Card internal padding */
      .target-card, .workflow-card {
        padding: 20px 22px !important;
        min-height: 190px !important;
      }
      .layer-card {
        padding: 18px 20px !important;
      }
      .layer-metric {
        padding: 16px 18px !important;
      }
      .shell-tile {
        padding: 16px 18px !important;
      }

      /* 7 — Headlines get more vertical space */
      h2 { margin-top: 8px !important; margin-bottom: 28px !important; }
      h3 { margin-top: 36px !important; margin-bottom: 18px !important; }
      h4 { margin-top: 24px !important; margin-bottom: 12px !important; }

      /* 8 — Metric cards get more breathing room */
      div[data-testid="stMetric"] {
        padding: 18px 16px !important;
      }

    </style>
    """, unsafe_allow_html=True)

    render_terminal_header()
    st.markdown("""
    <div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;
                padding:8px 0 14px 0;border-bottom:1px solid #e5e7eb;margin-bottom:12px;">
      <span class="canyon-swatch canyon-supportive">Green = supportive</span>
      <span class="canyon-swatch canyon-cyan">Cyan = mixed / check</span>
      <span class="canyon-swatch canyon-wait">Blue = wait / ETF</span>
      <span class="canyon-swatch canyon-paper">Purple = paper only</span>
      <span class="canyon-swatch canyon-risk">Red = risk</span>
      <span class="canyon-swatch canyon-blocked">Gray = no data</span>
    </div>
    """, unsafe_allow_html=True)

    top_tabs = st.tabs([
        "Home",
        "Main Decision",
        "Scorecard",
        "10-Layer Map",
        "All Layers",
        "Action Board",
        "Blueprint",
        "Help",
        "Daily Plan",
        "Research Room",
        "Portfolio Risk",
        "System Check",
        "Alpha Engine",
    ])

    with top_tabs[0]:
        tab_overview()
    with top_tabs[1]:
        tab_master()
    with top_tabs[2]:
        tab_strategy_scorecard()
    with top_tabs[3]:
        tab_layers()
    with top_tabs[4]:
        tab_l2_l6()
    with top_tabs[5]:
        tab_action()
    with top_tabs[6]:
        tab_architecture()
    with top_tabs[7]:
        tab_helper()
    with top_tabs[8]:
        tab_daily_desk()
    with top_tabs[9]:
        research_tabs = st.tabs(["Research Path", "Evidence Board", "Old Code Link", "Options Watch", "Before-Action Check", "Earnings NLP"])
        with research_tabs[0]:
            tab_research_lab()
        with research_tabs[1]:
            tab_research_stack()
        with research_tabs[2]:
            tab_v8_research_bridge()
        with research_tabs[3]:
            tab_options_lab()
        with research_tabs[4]:
            tab_pre_trade_gate()
        with research_tabs[5]:
            tab_earnings_nlp()
    with top_tabs[10]:
        risk_tabs = st.tabs(["Risk Control", "Portfolio Map", "Stress Test", "More Risk Checks", "Paper Log", "Optimizer", "Paper Sim", "Factor Attribution"])
        with risk_tabs[0]:
            tab_risk_control()
        with risk_tabs[1]:
            tab_portfolio_map()
        with risk_tabs[2]:
            tab_risk_stress()
        with risk_tabs[3]:
            tab_advanced_risk()
        with risk_tabs[4]:
            tab_paper_ledger()
        with risk_tabs[5]:
            tab_portfolio_optimizer()
        with risk_tabs[6]:
            tab_paper_sim()
        with risk_tabs[7]:
            tab_factor_attribution()
    with top_tabs[11]:
        system_tabs = st.tabs(["Control Room", "Report Archive", "Output Backup", "Data Sources", "Data Layer", "System Check", "Gap List", "Run Status", "Daily Runner", "Alerts", "Weekly Report"])
        with system_tabs[0]:
            tab_system_control()
        with system_tabs[1]:
            tab_report_archive()
        with system_tabs[2]:
            tab_output_vault()
        with system_tabs[3]:
            tab_data_source_health()
        with system_tabs[4]:
            tab_data_layer()
        with system_tabs[5]:
            tab_system_qa()
        with system_tabs[6]:
            tab_data_gaps()
        with system_tabs[7]:
            tab_run_status()
        with system_tabs[8]:
            tab_daily_runner()
        with system_tabs[9]:
            tab_alerts()
        with system_tabs[10]:
            tab_weekly_report()
    with top_tabs[12]:
        alpha_tabs = st.tabs(["Signal Backtest", "ML Signals", "SHAP", "Fundamentals",
                              "Options L7", "Universe (S75)", "Regime (S76)", "Regime ML (S77)",
                              "Deep Funds (S78)", "Sentiment (S79)"])
        with alpha_tabs[0]:
            tab_backtest()
        with alpha_tabs[1]:
            tab_ml_signals()
        with alpha_tabs[2]:
            tab_shap_explainer()
        with alpha_tabs[3]:
            tab_fundamental_features()
        with alpha_tabs[4]:
            tab_options_chain()
        with alpha_tabs[5]:
            tab_universe_expansion()
        with alpha_tabs[6]:
            tab_regime_detector()
        with alpha_tabs[7]:
            tab_regime_ml()
        with alpha_tabs[8]:
            tab_deep_fundamentals()
        with alpha_tabs[9]:
            tab_finbert_sentiment()


if __name__ == "__main__":
    main()
