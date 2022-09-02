setup: venv
	. .venv/bin/activate && python -m pip install --upgrade pip
	. .venv/bin/activate && python -m pip install -e .

setup-win: venv
	. .venv/Scripts/activate && python -m pip install --upgrade pip
	. .venv/Scripts/activate && python -m pip install -e .

venv:
	test -d .venv || python -m venv .venv

clean:
	rm -rf .venv
	rm -rf simreal.egg-info
	find . -name "__pycache__" -exec rm -fr {} +

ADR:
	. .venv/bin/activate && python ./src/main.py

ADR-win:
	. .venv/Scripts/activate && python ./src/main.py
