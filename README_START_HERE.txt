MARKET DATA PIPELINE - QUICK START

Requirements:
- Python 3.11
- Git
- Internet connection

1. Open PowerShell in the project directory.

2. Create a virtual environment:

   python -m venv .venv

3. Activate it:

   .\.venv\Scripts\Activate.ps1

4. Install dependencies:

   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt

5. Download historical BTCUSDT data:

   python -m market_data.cli download --symbol BTCUSDT --interval 1m --days 180

6. Validate data:

   python -m market_data.cli validate --symbol BTCUSDT --interval 1m

7. Run analytics:

   python -m market_data.cli analyze --symbol BTCUSDT --interval 1m

8. Run baseline backtest:

   python -m market_data.cli backtest --symbol BTCUSDT --interval 1m

9. Run walk-forward research:

   python -m market_data.walk_forward

10. Generate research summary:

    python -m market_data.research_summary

11. Run automated tests:

    python -m pytest -v

IMPORTANT:
- Activate the virtual environment before running commands.
- Downloaded datasets are excluded from Git.
- Walk-forward research requires approximately 180 days of data.
- This is an educational research project, not a trading recommendation.
