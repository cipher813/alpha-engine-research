# alpha-engine-research

Autonomous investment research pipeline for US equities. Part of the Alpha Engine system.

## Overview

LangGraph-based pipeline that maintains rolling investment theses for a tracked universe of stocks and scans the S&P 500/400 for buy candidates. Delivers a consolidated morning research brief via email on NYSE trading days.

## Stack

- Python 3.11, AWS Lambda
- Anthropic Claude (Haiku + Sonnet)
- LangGraph state machine orchestration
- S3 for data persistence and inter-module communication

## Related Repos

- `alpha-engine` — Executor (trade execution via IB Gateway)
- `alpha-engine-predictor` — GBM model (5-day alpha predictions)
- `alpha-engine-backtester` — Signal quality analysis & parameter optimization
- `alpha-engine-dashboard` — Streamlit monitoring dashboard

## License

See [LICENSE](LICENSE).
