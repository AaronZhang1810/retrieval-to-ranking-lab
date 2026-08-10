.PHONY: data demo quick publish app test lint release-check clean

data:
	python download_data.py --accept-license

demo:
	python run_demo.py

quick:
	python run_demo.py --quick

publish:
	python run_demo.py --publish-reports

app:
	python app.py

test:
	python -m pytest

lint:
	python -m ruff check .

release-check:
	python public_release.py --check
	python public_release.py --check --staged
	git diff --cached --check

clean:
	@echo "Local data/artifacts are ignored; remove them manually only if you no longer need them."
