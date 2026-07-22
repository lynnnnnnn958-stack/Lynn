# Canyon v9 Modularization Plan

Generated: 2026-06-06

Current active dashboard:

`/Users/renjingru/Desktop/canyon_quant/canyon_final_v9_step86_dashboard_v3.py`

Current size:

- About 10,146 lines
- About 226 top-level functions
- 8 page entry functions
- About 89 render/helper UI functions

This document describes how the current Streamlit monolith should be divided into a normal software project without breaking the working localhost dashboard.

## North Star

The dashboard should eventually follow this rule:

```text
app.py = starts Streamlit and routes pages
ui/pages = page layout and user-facing text
ui/components = reusable visual blocks
domain = quant/business interpretation logic
data = safe file loading and validation
runners = scripts that generate CSV/JSON outputs
tests = smoke tests and logic tests
```

The dashboard should not keep growing as one giant file. It should read prepared files, translate them into clear human decisions, and hide raw evidence until requested.

## Safe Refactor Rule

Do not rewrite the whole dashboard at once.

The correct migration path is:

1. Extract pure helpers that do not depend on Streamlit.
2. Add tests for those helpers.
3. Extract one page at a time.
4. Browser-check the page after each extraction.
5. Keep the old monolith as the fallback until the new entrypoint passes the same checks.

## Proposed Structure

```text
canyon_app/
  core/
    paths.py
    formatting.py
    status_labels.py
    source_health.py
    table_helpers.py

  data/
    loaders.py
    validators.py
    manifest.py

  ui/
    layout.py
    navigation.py
    styles.py
    components/
      cards.py
      tables.py
      command_center.py
      source_badges.py
    pages/
      home.py
      today.py
      ideas.py
      news.py
      risk.py
      performance.py
      live_paper.py
      system.py

  domain/
    risk/
      labels.py
      summary.py
      repair_path.py
      optimizer.py
      execution_liquidity.py
    performance/
      sharpe.py
      backtest_credibility.py
      signal_ic_decay.py
      repair_queue.py
    news/
      labels.py
      causal_chain.py
      help_hurt.py
      industry_map.py
    ideas/
      horizon.py
      vehicle.py
      options.py
    live/
      paper.py
      feedback.py
    system/
      run_center.py
      qa.py

runners/
  daily_runner.py
  risk_runner.py
  performance_runner.py
  news_runner.py

tests/
  test_modular_core.py
  test_risk_labels.py
  test_performance_summary.py
  test_news_help_hurt.py
  test_dashboard_smoke.py
```

## Current Function Groups

### Core helpers

Current functions near the top of the monolith should move to:

`canyon_app/core/formatting.py`

- `_to_float`
- `_money`
- `_pct_display`
- `_human_text`
- `_friendly_value_text`
- `_humanize_key_value_text`
- `_friendly_col`
- `_plain_status`
- `_clean_display`

Current table and HTML helpers should move to:

`canyon_app/ui/components/`

- `_show_status_table`
- `_render_html`
- `_simple_card`
- `_humanize_df`

Current source file helpers should move to:

`canyon_app/core/source_health.py`

- `_friendly_source_label`
- `_section_source_health`

### Pages

Each tab function should become one page module:

| Current function | New file | New public function |
|---|---|---|
| `tab_today_workflow` | `canyon_app/ui/pages/today.py` | `render_today_page()` |
| `tab_ideas_workflow` | `canyon_app/ui/pages/ideas.py` | `render_ideas_page()` |
| `tab_news_room` | `canyon_app/ui/pages/news.py` | `render_news_page()` |
| `tab_risk_portfolio` | `canyon_app/ui/pages/risk.py` | `render_risk_page()` |
| `tab_performance` | `canyon_app/ui/pages/performance.py` | `render_performance_page()` |
| `tab_live_paper_monitor` | `canyon_app/ui/pages/live_paper.py` | `render_live_paper_page()` |
| `tab_system_status` / `tab_run_system` | `canyon_app/ui/pages/system.py` | `render_system_page()` |
| Home helpers | `canyon_app/ui/pages/home.py` | `render_home_page()` |

### Risk

Move plain risk interpretation to:

`canyon_app/domain/risk/labels.py`

- `_risk_accent`
- `_risk_human_action`
- `_risk_status_plain`
- `_risk_limit_plain_name`
- `_risk_code_plain`
- `_risk_desk_plain`

Move risk page sections to:

`canyon_app/domain/risk/summary.py`

- `_risk_loss_estimate_line`
- `_risk_worst_macro_line`
- `_risk_crisis_line`
- `_risk_sector_line`
- `_risk_factor_line`

Move repair logic to:

`canyon_app/domain/risk/repair_path.py`

- `_risk_first_repair_ticker`
- `_risk_repair_action_plain`
- `_risk_route_plain`
- `_risk_unlock_steps`

Keep render functions in:

`canyon_app/ui/pages/risk.py`

- `_render_risk_command_center`
- `_render_risk_repair_path`
- `_render_risk_unlock_ladder`
- `_render_risk_verdict_board`

### Performance

Move plain labels to:

`canyon_app/domain/performance/sharpe.py`

- `_perf_plain`
- `_perf_signal_label`
- `_perf_accent`
- `_perf_number`
- `_perf_score_text`

Move score extraction to:

`canyon_app/domain/performance/backtest_credibility.py`

- `_perf_module_score`
- credibility score calculations

Keep render functions in:

`canyon_app/ui/pages/performance.py`

- `_render_performance_command_center`
- `_render_perf_reason_cards`
- `_render_perf_next_actions_preview`
- `_render_perf_signal_cards`

### News

Move labels and chain logic to:

`canyon_app/domain/news/`

- `_news_plain`
- `_news_tone_label`
- `_news_decision_label`
- `_news_route_label`
- `_news_help_hurt_lines`
- `_news_card_story`
- `_news_chain_sentence`
- `_news_proof_status_sentence`

Keep page rendering in:

`canyon_app/ui/pages/news.py`

- `_render_news_command_center`
- `_render_news_story_board`
- `_render_news_industry_proof_preview`

### Ideas

Move route logic to:

`canyon_app/domain/ideas/`

- `_ideas_plain`
- `_ideas_route_permission`
- `_ideas_horizon_summary`
- `_ideas_vehicle_now`
- `_ideas_call_line`
- `_ideas_put_line`
- `_ideas_unlock_line`

Keep page rendering in:

`canyon_app/ui/pages/ideas.py`

- `_render_ideas_command_center`
- `_render_ideas_horizon_lanes`
- `_render_ideas_vehicle_board`
- `_render_ideas_cards`

### Today

Move workflow translation to:

`canyon_app/domain/system/` or `canyon_app/domain/live/`

- `_today_plain`
- `_today_gate_label`
- `_today_action_sentence`

Keep page rendering in:

`canyon_app/ui/pages/today.py`

- `_render_today_command_board`
- `_render_today_station_flow`
- `_render_today_ticker_queue_cards`
- `_render_today_proof_plan`

## Migration Phases

### Phase 1: Pure Helpers

Already started:

- `canyon_app/core/paths.py`
- `canyon_app/core/formatting.py`
- `canyon_app/data/loaders.py`
- `canyon_app/ui/components/cards.py`
- `tests/test_modular_core.py`

Next:

- Move source health helpers.
- Move status label dictionaries.
- Add tests for code-to-human translation.

No dashboard behavior should change in this phase.

### Phase 2: Risk Page Extraction

Why first:

- Risk is a veto page.
- Risk already has a clear command center.
- It has many label helpers that can be tested independently.

Steps:

1. Create `canyon_app/domain/risk/labels.py`.
2. Move `_risk_code_plain`, `_risk_status_plain`, `_risk_accent`.
3. Write tests that confirm no raw code leaks:
   - `REDUCE_ONLY`
   - `SIZE_DOWN`
   - `DATA_GAP`
   - `NO_NEW_OPTION`
4. Move Risk render sections to `canyon_app/ui/pages/risk.py`.
5. Keep `tab_risk_portfolio()` in the monolith as a wrapper until verified.

### Phase 3: Performance Page Extraction

Move:

- Sharpe summary
- proof-adjusted Sharpe logic
- signal label logic
- repair queue cards

Test:

- headline Sharpe and proof Sharpe are displayed
- Sharpe 4 is not claimed when `claim_allowed=false`
- no raw labels leak on default page

### Phase 4: News and Ideas Extraction

News and Ideas should be extracted after Risk and Performance because they depend on risk gates and proof language.

Must preserve:

- Help/hurt explanation
- source article links
- industry chain mapping
- stock / call / put / wait route
- default pages in simple English

### Phase 5: New App Entrypoint

Only after at least Risk and Performance are extracted:

Create:

`app.py`

It should:

1. Configure Streamlit.
2. Render shared header/nav.
3. Route to `render_home_page`, `render_risk_page`, etc.
4. Keep old dashboard file available as fallback.

## Do Not Do Yet

Do not immediately delete:

- `canyon_final_v9_step86_dashboard_v3.py`
- existing CSV/JSON outputs
- old Step scripts
- output vault snapshots

Do not move output files into a new folder yet. The current scripts expect files in the project root.

Do not rename active CSV files until runners are updated.

Do not change visual style during refactor. The user wants the existing white/black professional style.

## Standard Smoke Test

Run after every extraction:

```bash
cd /Users/renjingru/Desktop/canyon_quant
python3 -m py_compile canyon_final_v9_step86_dashboard_v3.py
python3 -m py_compile $(find canyon_app -name '*.py' -print)
python3 -m pytest tests/test_modular_core.py
```

Then browser-check:

```text
http://localhost:8512/?page=Risk
http://localhost:8512/?page=Performance
http://localhost:8512/?page=News
http://localhost:8512/?page=Ideas
```

Default pages should not show internal raw status codes unless the user opens a technical detail table.

## Success Criteria

The refactor is successful only if:

- current dashboard still runs
- default pages are still simple English
- Risk still vetoes Ideas and Options
- Performance does not overclaim Sharpe 4
- News still shows help/hurt and source proof
- Ideas still separates short / medium / long horizon and stock / call / put / wait routes
- every extracted helper has basic tests

