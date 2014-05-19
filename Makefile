test:
	nosetests -v

test_coverage:
	nosetests --with-coverage --cover-package=bayes_gmm .
