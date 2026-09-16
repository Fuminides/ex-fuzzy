"""Execute the demo notebooks in place, refreshing their stored outputs.

Run it before a release so the published notebooks reflect the current code::

    python Demos/run_notebooks.py              # every numbered notebook
    python Demos/run_notebooks.py 05_regression.ipynb

It needs ``nbclient`` and ``ipykernel`` (both come with ``pip install jupyter``).
Notebooks run with the Demos folder as working directory. The Titanic and
California housing notebooks download their data through scikit-learn on the
first run and cache it in your home directory.
"""
import pathlib
import sys
import time

import nbformat
from nbclient import NotebookClient

DEMOS = pathlib.Path(__file__).resolve().parent


def main(names: list[str]) -> None:
    targets = [DEMOS / name for name in names] if names else sorted(DEMOS.glob('0*.ipynb'))
    for path in targets:
        notebook = nbformat.read(str(path), as_version=4)
        start = time.perf_counter()
        NotebookClient(notebook, timeout=3600, kernel_name='python3', record_timing=False,
                       resources={'metadata': {'path': str(DEMOS)}}).execute()
        nbformat.write(notebook, str(path))
        print(f'{path.name}: {time.perf_counter() - start:.0f}s')


if __name__ == '__main__':
    main(sys.argv[1:])
