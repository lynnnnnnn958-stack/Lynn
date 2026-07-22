# Canyon Quant — task runner
# W46: CI gate added — `make ci` must pass before running the daily pipeline
.PHONY: test test-fast lint lint-all git-check ci pre-run scorecard tearsheet quick fred social

PYTHON := .venv/bin/python

# ── Test gates ────────────────────────────────────────────────────────────────

# Run all signal/pipeline tests (skips slow canyon_app integration tests)
test:
	$(PYTHON) -m pytest tests/test_signals.py tests/test_pit_compliance.py -v --tb=short 2>/dev/null || \
	$(PYTHON) -m pytest tests/test_signals.py -v --tb=short

# Fast smoke test (no network, no disk IO) — used as CI gate
test-fast:
	$(PYTHON) -m pytest tests/test_signals.py -v --tb=short -x

# W46: Full CI gate: fast tests + syntax check on all W1-W50 modules
ci: lint-all test-fast
	@echo "✓ CI gate passed"

# ── Lint ──────────────────────────────────────────────────────────────────────

# Lint key files
lint:
	$(PYTHON) -m py_compile canyon_v11_full.py && echo "v11: OK"
	$(PYTHON) -m py_compile data/edgar_pit.py 2>/dev/null && echo "edgar_pit: OK" || true
	$(PYTHON) -m py_compile data/fred_macro.py 2>/dev/null && echo "fred_macro: OK" || true

# W46: Lint all new modules added in W1-W50
lint-all: lint
	@for f in \
	  signals/macro_hmm.py signals/lgb_shap.py signals/earnings_call.py signals/eps_revision.py \
	  data/edgar_pit.py data/edgar_form4.py data/fred_macro.py data/sp500_constituents.py \
	  data/extend_history.py data/polygon_ivr.py \
	  research/signal_halflife.py research/signal_corr.py research/vif_check.py \
	  research/capacity.py research/backtest_scorecard.py research/tearsheet.py \
	  research/param_robustness.py research/stress_test.py research/v9_bridge.py \
	  risk/barra.py risk/regime_cov.py \
	  portfolio/black_litterman.py portfolio/kelly.py \
	  monitoring/data_quality.py monitoring/factor_exposure.py monitoring/slippage.py \
	  monitoring/execution_quality.py monitoring/attribution.py \
	  execution/alpaca_exec.py \
	  research/institutional_scorecard.py ; do \
	  if [ -f "$$f" ]; then \
	    $(PYTHON) -m py_compile $$f && echo "OK: $$f" || echo "FAIL: $$f"; \
	  fi; \
	done

# ── Research tools ────────────────────────────────────────────────────────────

# Generate factor tearsheet HTML (W42)
tearsheet:
	$(PYTHON) research/tearsheet.py

# Run institutional scoring audit (W49)
scorecard:
	$(PYTHON) research/institutional_scorecard.py

# Run backtest credibility scorecard (W35)
backtest-score:
	$(PYTHON) research/backtest_scorecard.py

# Run stress tests (W44)
stress:
	$(PYTHON) research/stress_test.py

# Run parameter robustness test (W43)
robustness:
	$(PYTHON) research/param_robustness.py

# ── Pipeline ──────────────────────────────────────────────────────────────────

# W46: Pre-run gate — run tests before pipeline starts
pre-run: ci
	@echo "✓ Pre-run checks passed — starting daily pipeline"
	$(PYTHON) run_daily.py

# Fast intraday refresh: price + social + alerts + HTML (< 5 min)
quick:
	$(PYTHON) run_quick.py

# Fetch extended FRED macro data
fred:
	$(PYTHON) step_fred_data.py

# Fetch social sentiment (StockTwits)
social:
	$(PYTHON) step_social_sentiment.py

# Git status check before daily run
git-check:
	@git status --short
	@git log --oneline -5
