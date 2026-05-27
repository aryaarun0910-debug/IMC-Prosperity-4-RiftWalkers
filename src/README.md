# Source

Code for the IMC Prosperity 4 run. See the [main writeup](../README.md) and [pipeline architecture](../docs/08-pipeline-architecture.md) for context.

```
src/
├── engine/      The submission engine and backtester
│   ├── trader.py        Main single-file trading engine (9 strategy archetypes, stdlib only)
│   ├── trader_r5.py     Round 5 mean-reversion engine (50-product alpha)
│   ├── datamodel.py     Exchange type definitions (Order, OrderDepth, TradingState)
│   ├── backtest.py      Order-book-reconstruction backtester
│   └── grid_search.py   Parameter search
├── tools/       Research and validation tooling
│   ├── r5_backtest.py            Integrated 3-day backtester (ground-truth validation)
│   ├── r5_walkforward.py         Walk-forward validation (train on subset, test held-out)
│   ├── r5_systematic_mr.py       Mean-reversion config grid search
│   ├── r5_cluster_analysis.py    Per-category structural metrics
│   ├── r5_pair_deepdive.py       Within-category return-correlation analysis
│   ├── r5_obi_study.py           Order-book-imbalance study
│   ├── r5_cross_category.py      50x50 cross-category correlation
│   ├── r5_autocorr_lag.py        Long-lag autocorrelation
│   ├── gauntlet.py               Regression harness (pre-ship gate)
│   ├── submit.py                 Strips the engine for upload (<100KB check)
│   └── ...                       Additional round-specific tooling
├── analysis/    Offline analysis (numpy/pandas/sklearn permitted — never shipped)
│   ├── bot_fingerprinter.py      Counterparty profiling
│   ├── train_macaron_classifier.py  Offline model whose coefficients ship as constants
│   └── ...
└── manual/      Manual-round solvers
    └── r5_manual_optimizer.py    Quadratic-fee allocation optimizer + Monte-Carlo
```

The submission engine (`engine/trader.py`) is standard-library only by competition rule. The `analysis/` scripts run offline and may use scientific Python; any model trained there is distilled to plain constants before being embedded in the engine.

Note: the research tooling reads from competition data that is not included in this repository (see the top-level `.gitignore`). The scripts are provided to show methodology rather than to run out of the box.
