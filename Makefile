test:
	PYTHONPATH=. pytest -v

test_coverage:
	PYTHONPATH=. pytest --cov=bayes_gmm --cov-report=term-missing
