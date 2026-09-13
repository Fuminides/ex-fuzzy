"""Reader for the KEEL classification collection used by the benchmark suite.

The files are KEEL's ARFF dialect: a header of ``@attribute`` declarations
followed by ``@inputs``/``@outputs`` and a comma separated ``@data`` block.
Nominal attributes become ordinal codes and are reported through a categorical
mask, so callers can hand them to estimators that fuzzify categories directly.
The collection is not redistributed here; point ``EX_FUZZY_KEEL_ROOT`` at a
local copy or keep it in one of the default locations below.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re

import numpy as np

#: Searched in order when ``EX_FUZZY_KEEL_ROOT`` is unset.
DEFAULT_ROOTS = (
    Path.home() / 'Datasets' / 'keel_datasets',
    Path.home() / 'Documentos' / 'keel_datasets',
    Path.home() / 'keel_datasets',
)

_ATTRIBUTE = re.compile(r"""^@attribute\s+('[^']*'|"[^"]*"|[^\s{]+)\s*(.*)$""", re.IGNORECASE)
_MISSING = '?'


@dataclass(frozen=True)
class KeelDataset:
    """One parsed KEEL classification problem."""

    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    categorical_mask: np.ndarray
    dropped_rows: int

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def n_categorical(self) -> int:
        return int(self.categorical_mask.sum())

    def summary(self) -> dict:
        """Shape metadata worth recording next to a measurement."""
        return dict(dataset=self.name, n_samples=self.n_samples,
                    n_features=self.n_features, n_classes=self.n_classes,
                    n_categorical=self.n_categorical,
                    dropped_rows=self.dropped_rows)


def keel_root(root: str | os.PathLike | None = None) -> Path:
    """Locate the collection, preferring an explicit path then the env var."""
    candidates = []
    if root is not None:
        candidates.append(Path(root).expanduser())
    env = os.environ.get('EX_FUZZY_KEEL_ROOT')
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(DEFAULT_ROOTS)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        'KEEL collection not found. Set EX_FUZZY_KEEL_ROOT to a directory '
        'holding one subdirectory per dataset, each with a <name>.dat file. '
        'Looked in: ' + ', '.join(str(c) for c in candidates))


def dataset_path(name: str, root: str | os.PathLike | None = None) -> Path:
    """Return the ``.dat`` file for ``name``."""
    base = keel_root(root) / name
    direct = base / f'{name}.dat'
    if direct.is_file():
        return direct
    found = sorted(base.glob('*.dat')) if base.is_dir() else []
    if not found:
        raise FileNotFoundError(f'No .dat file for KEEL dataset {name!r} under {base}')
    return found[0]


def available_datasets(root: str | os.PathLike | None = None) -> list[str]:
    """Names of every dataset the collection exposes, sorted."""
    base = keel_root(root)
    names = []
    for child in sorted(base.iterdir()):
        if child.is_dir() and any(child.glob('*.dat')):
            names.append(child.name)
    return names


def _split_header(text: str) -> tuple[str, str]:
    for index, line in enumerate(text.splitlines(keepends=True)):
        if line.strip().lower().startswith('@data'):
            offset = text.index(line) + len(line)
            return text[:offset], text[offset:]
    raise ValueError('KEEL file has no @data section')


def _parse_header(header: str):
    names: list[str] = []
    categories: dict[str, list[str]] = {}
    inputs = outputs = None
    for raw in header.splitlines():
        line = raw.strip()
        lowered = line.lower()
        if lowered.startswith('@attribute'):
            match = _ATTRIBUTE.match(line)
            if match is None:
                raise ValueError(f'Unreadable attribute declaration: {line!r}')
            name, spec = match.group(1).strip('\'"'), match.group(2).strip()
            names.append(name)
            if spec.startswith('{'):
                body = spec[1:spec.rindex('}')] if '}' in spec else spec[1:]
                categories[name] = [value.strip().strip('\'"') for value in body.split(',')]
        elif lowered.startswith('@inputs'):
            inputs = [value.strip() for value in line.split(None, 1)[1].split(',')]
        elif lowered.startswith('@output'):  # KEEL writes @output and @outputs.
            outputs = [value.strip() for value in line.split(None, 1)[1].split(',')]
    if not names:
        raise ValueError('KEEL file declares no attributes')
    if inputs is None:
        inputs = names[:-1]
    if outputs is None:
        outputs = names[-1:]
    if len(outputs) != 1:
        raise ValueError(f'Expected a single output attribute, got {outputs}')
    return names, categories, inputs, outputs[0]


def load_dataset(name: str, root: str | os.PathLike | None = None) -> KeelDataset:
    """Parse one KEEL classification problem into arrays.

    Nominal attributes are mapped to the integer codes of their declared order
    and flagged in ``categorical_mask``. Labels are mapped to ``0..k-1`` in the
    sorted order of their string form. Rows holding a KEEL missing marker are
    dropped and counted, so the row count stays explicit in the results.
    """
    path = dataset_path(name, root)
    header, body = _split_header(path.read_text())
    names, categories, inputs, target = _parse_header(header)
    position = {attribute: index for index, attribute in enumerate(names)}
    for attribute in [*inputs, target]:
        if attribute not in position:
            raise ValueError(f'{name}: @inputs/@outputs names an undeclared attribute {attribute!r}')

    rows, dropped = [], 0
    width = len(names)
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('%') or stripped.startswith('@'):
            continue
        values = [value.strip() for value in stripped.split(',')]
        if len(values) != width:
            raise ValueError(f'{name}: expected {width} values, found {len(values)}: {stripped[:60]!r}')
        if any(value == _MISSING for value in values):
            dropped += 1
            continue
        rows.append(values)
    if not rows:
        raise ValueError(f'{name}: no usable rows')

    X = np.empty((len(rows), len(inputs)), dtype=float)
    mask = np.zeros(len(inputs), dtype=int)
    for column, attribute in enumerate(inputs):
        index = position[attribute]
        raw = [row[index] for row in rows]
        if attribute in categories:
            codes = {value: code for code, value in enumerate(categories[attribute])}
            unseen = sorted({value for value in raw if value not in codes})
            for value in unseen:  # Tolerate levels the header forgot to declare.
                codes[value] = len(codes)
            X[:, column] = [codes[value] for value in raw]
            mask[column] = 1
        else:
            X[:, column] = [float(value) for value in raw]

    labels = [row[position[target]] for row in rows]
    class_names = sorted(set(labels))
    lookup = {value: code for code, value in enumerate(class_names)}
    y = np.array([lookup[value] for value in labels], dtype=int)
    return KeelDataset(name=name, X=X, y=y, feature_names=list(inputs),
                       class_names=class_names, categorical_mask=mask,
                       dropped_rows=dropped)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Summarise the local KEEL collection.')
    parser.add_argument('--root', default=None)
    parser.add_argument('datasets', nargs='*', help='Defaults to every dataset found.')
    options = parser.parse_args()
    wanted = options.datasets or available_datasets(options.root)
    print(f'{"dataset":20s} {"rows":>7s} {"feat":>5s} {"cat":>4s} {"cls":>4s} {"dropped":>8s}')
    for dataset_name in wanted:
        data = load_dataset(dataset_name, options.root)
        print(f'{data.name:20s} {data.n_samples:7d} {data.n_features:5d} '
              f'{data.n_categorical:4d} {data.n_classes:4d} {data.dropped_rows:8d}')
