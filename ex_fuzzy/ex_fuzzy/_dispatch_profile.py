"""Optional stored calibration for the population-evaluation route choice.

The scalar and batched evaluators produce the same objective bit for bit, so
choosing between them is purely a speed question.  By default a fit answers it
with a short runtime probe, which costs one generation of the losing route.  A
user who would rather not pay that on every fit can run the offline campaign in
``benchmarks/calibrate_population_dispatch.py`` once and leave a profile behind;
fits then look the answer up instead of measuring it.

Nothing here changes results.  A profile that is missing, unreadable, recorded
on different hardware or too far from the current workload simply yields no
answer, and the fit probes as usual.
"""
import json
import os
import platform
from typing import Optional

import numpy as np


PROFILE_ENV = 'EX_FUZZY_DISPATCH_PROFILE'
FORMAT_VERSION = 1

#: Largest per-dimension ratio between a query and a calibrated point that still
#: counts as the same workload.  Cost varies smoothly in these dimensions, so a
#: factor of two either way stays on the correct side of a route boundary far
#: more often than it does not; anything further away is left to the probe.
NEIGHBOUR_TOLERANCE = 2.0

_DIMENSIONS = ('samples', 'rules', 'features', 'population')


def default_path() -> str:
    """Where a calibration profile is read from and written to."""
    override = os.environ.get(PROFILE_ENV)
    if override:
        return override
    cache = os.environ.get('XDG_CACHE_HOME') or os.path.join(
        os.path.expanduser('~'), '.cache')
    return os.path.join(cache, 'ex-fuzzy', 'dispatch_profile.json')


def fingerprint() -> dict:
    """Identify the machine a calibration was measured on.

    A profile copied to different hardware describes the wrong cost balance, so
    fits ignore it rather than trusting a measurement that was never taken here.
    """
    return {
        'machine': platform.machine(),
        'python': '.'.join(platform.python_version_tuple()[:2]),
        'numpy': '.'.join(np.__version__.split('.')[:2]),
    }


def save(entries: list, path: Optional[str] = None) -> str:
    """Write a calibration profile, creating its directory if needed."""
    path = path or default_path()
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        'version': FORMAT_VERSION,
        'fingerprint': fingerprint(),
        'entries': entries,
    }
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def load(path: Optional[str] = None) -> Optional[list]:
    """Read a calibration profile, or None when there is no usable one."""
    path = path or default_path()
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    if (not isinstance(payload, dict)
            or payload.get('version') != FORMAT_VERSION
            or payload.get('fingerprint') != fingerprint()):
        return None
    entries = payload.get('entries')
    if not isinstance(entries, list) or not entries:
        return None
    return entries


def lookup(entries: list, samples: int, rules: int, features: int,
           population: int) -> Optional[str]:
    """Return ``'batch'``/``'scalar'`` for the nearest calibrated workload.

    Distance is measured on log ratios so that every dimension counts
    proportionally rather than absolutely.  When the closest calibrated point is
    further than :data:`NEIGHBOUR_TOLERANCE` in any dimension the profile has
    nothing to say about this workload and the caller falls back to probing.
    """
    query = {'samples': samples, 'rules': rules, 'features': features,
             'population': population}
    if any(value <= 0 for value in query.values()):
        return None
    limit = np.log(NEIGHBOUR_TOLERANCE)
    best = None
    best_distance = None
    for entry in entries:
        try:
            offsets = [abs(np.log(float(entry[name]) / query[name]))
                       for name in _DIMENSIONS]
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        if max(offsets) > limit:
            continue
        distance = sum(offset * offset for offset in offsets)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best = entry.get('decision')
    return best if best in ('batch', 'scalar') else None


def decision_for(samples: int, rules: int, features: int, population: int,
                 path: Optional[str] = None) -> Optional[str]:
    """Look a workload up in the stored profile, or None to probe instead."""
    entries = load(path)
    if entries is None:
        return None
    return lookup(entries, samples, rules, features, population)
