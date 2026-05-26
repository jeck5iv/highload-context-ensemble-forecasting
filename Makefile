.PHONY: install install-full quick-check test run-forecasting run-grid compare-fixture notebook

install:
	python -m pip install -e .

install-full:
	python -m pip install -e '.[full]'

quick-check:
	PYTHONPATH=src pytest tests/test_quick_checks.py

test:
	PYTHONPATH=src pytest

run-forecasting:
	PYTHONPATH=src python scripts/run_metro_forecasting.py

run-grid:
	PYTHONPATH=src python scripts/run_horizon_bucket_grid.py --profile classical --n-bootstrap 0

compare-fixture:
	PYTHONPATH=src python scripts/compare_regression_outputs.py --old-dir tests/fixtures/h060m_b015m --new-dir tests/fixtures/h060m_b015m

notebook:
	PYTHONPATH=src jupyter notebook notebooks/00_repository_check.ipynb
