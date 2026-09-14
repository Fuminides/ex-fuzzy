"""
Fuzzy Cognitive Maps for Ex-Fuzzy Library

This module implements Fuzzy Cognitive Maps (FCMs), a soft computing technique that
combines fuzzy logic with cognitive mapping for modeling complex systems with
causal relationships. FCMs are particularly useful for decision making, scenario
analysis, and understanding dynamic system behavior.

Main Components:
    - FuzzyCognitiveMap: Core FCM class for creation and simulation
    - Dynamic simulation: Iterative state evolution with trivalent thresholding
    - Analysis functions: Fixed point and limit cycle identification

Key Features:
    - Support for weighted causal relationships between concepts
    - States thresholded into the {-1, 0, 1} values
    - Periodic behavior and limit cycle identification

Theoretical Background:
    Fuzzy Cognitive Maps were originally developed by Bart Kosko and extend
    traditional cognitive maps by incorporating fuzzy logic principles.
    They represent concepts as nodes and causal relationships as weighted
    directed edges, enabling the modeling of complex feedback systems.
"""
from __future__ import annotations
from typing import Union

import pandas as pd
import numpy as np


def _threshold_modules(connections: Union[np.array, pd.DataFrame], threshold) -> Union[np.array, pd.DataFrame]:
    '''Thresholds the connections matrix to the {-1, 0, 1} values.'''
    return np.sign(connections) * (np.abs(connections) > threshold)


def _look_attractors(states: list[np.array]) -> tuple[bool, list[np.array]]:
    '''
    Checks if all the states in the list are the same.

    :param states: list of states visited in consecutive steps.
    :return: (True, [state]) if the list is a fixed point, (False, []) otherwise.
    '''
    first = np.asarray(states[0])
    if all(np.array_equal(first, state) for state in states[1:]):
        return True, [first]

    return False, []


def _look_periods(states: list[np.array], min_period_len: int=2, max_period_len: int=None) -> list[np.array]:
    '''
    Looks for the shortest period in the states list.

    :param states: list of states visited in consecutive steps.
    :param min_period_len: shortest period to consider.
    :param max_period_len: longest period to consider. Half the list length if None.
    :return: the states of one period if found, None otherwise.
    '''
    longest = len(states) // 2
    if max_period_len is not None:
        longest = min(longest, max_period_len)

    for period_len in range(min_period_len, longest + 1):
        if all(np.array_equal(states[i], states[i + period_len]) for i in range(len(states) - period_len)):
            return list(states[:period_len])

    return None


def look_pattern_states(fcm: FuzzyCognitiveMap, sim_steps: int, pattern_len: int=50, max_period_size: int=50) -> list[np.array]:
    '''Looks for the pattern states of the FCM when simulation is prolongued.

    :param fcm: FuzzyCognitiveMap. The FCM to look for the attractor states.
    :param sim_steps: int. The number of steps simulated before looking for a pattern.
    :param pattern_len: int. The number of steps recorded to look for a pattern.
    :param max_period_size: int. The longest limit cycle to look for.
    :returns: list of np.array. The states of the fixed point or limit cycle. None if none was found.
    '''
    for _ in range(sim_steps):
        fcm.step()

    steps = [np.asarray(fcm.step()) for _ in range(pattern_len)]

    satisfactory, period = _look_attractors(steps)
    if not satisfactory:
        period = _look_periods(steps, min_period_len=2, max_period_len=max_period_size)

    return period


def study_attractors_FCM(fcm: FuzzyCognitiveMap, max_steps: int, random_inits: int=10) -> dict[tuple, list[np.array]]:
    '''Looks for the attractor states of the FCM when simulation is prolongued.

    :param fcm: FuzzyCognitiveMap. The FCM to look for the attractor states.
    :param max_steps: int. The number of steps simulated before looking for a pattern.
    :param random_inits: int. The number of random initial states to try.
    :returns: dict mapping each initial state (as a tuple) to its attractor states. None if none were found.
    '''
    attractors = {}
    for _ in range(random_inits):
        init_state = np.random.randint(0, 2, fcm.connections.shape[0])
        fcm.set_state(init_state)
        attractors[tuple(init_state)] = look_pattern_states(fcm, max_steps)

    return attractors


def attractors_report(attractors: dict[tuple, list[np.array]]) -> tuple[dict, dict]:
    '''
    Computes the frequency of each attractor and of each attractor state.

    :param attractors: dict. The attractors found, as returned by study_attractors_FCM.
    :return: a dict with the frequency of each pattern ('Chaotic' when none was found)
        and a dict with the frequency of each state among the patterns found.
    '''
    pattern_dict = {}

    for _, attractor in attractors.items():
        if attractor is None:
            pattern_dict['Chaotic'] = pattern_dict.get('Chaotic', 0) + 1 / len(attractors)
        else:
            pattern_dict[str(attractor)] = pattern_dict.get(str(attractor), 0) + 1 / len(attractors)

    list_states = []
    for _, attractor in attractors.items():
        if attractor is not None:
            for state in attractor:
                list_states += [str(state)]

    state_dict = {}
    for state in list_states:
        state_dict[state] = state_dict.get(state, 0) + 1 / len(list_states)

    return pattern_dict, state_dict


class FuzzyCognitiveMap:

    def __init__(self, connections: Union[np.array, pd.DataFrame], threshold: float=0.5) -> None:
        '''
        Creates a fuzzy cognitive map.

        :param connections: np.array or pd.DataFrame. A square matrix with the connections between the concepts.
        :param threshold: float, optional. When simulating steps the state
        will be trimmed using these threhold into the {0, 1, -1} values.
        The default is 0.5.
        '''
        self.connections = connections
        self.state = np.zeros(connections.shape[0])
        self.threshold = threshold


    def var_names(self) -> list[str]:
        '''Returns the names of the variables.'''
        try:
            return list(self.connections.columns)
        except AttributeError:
            return None


    def _format_state(self) -> Union[np.array, pd.Series]:
        '''Returns the state, labelled with the variable names when they are known.'''
        names = self.var_names()
        if names is None:
            return self.state

        return pd.Series(self.state, index=names)


    def step(self) -> Union[np.array, pd.Series]:
        '''Performs a step in the FCM given the actual state.'''
        self.state = np.asarray(_threshold_modules(self.state @ np.asarray(self.connections), self.threshold))

        return self._format_state()


    def simulate(self, steps: int) -> Union[np.array, pd.Series]:
        '''
        Simulates the FCM for a number of steps.

        :param steps: int. The number of steps to simulate.
        '''
        for _ in range(steps):
            self.step()

        return self._format_state()


    def add(self, other: FuzzyCognitiveMap) -> None:
        '''Adds the connections of other FCM to the actual FCM.'''
        self.connections = self.connections + other.connections


    def set_state(self, state: Union[np.array, pd.Series]) -> None:
        '''Sets the state of the FCM.'''
        self.state = np.asarray(state)


    def set_and_step(self, state: Union[np.array, pd.Series]) -> Union[np.array, pd.Series]:
        '''Sets the state of the FCM and performs a step.'''
        self.set_state(state)
        return self.step()


    def set_and_simulate(self, state: Union[np.array, pd.Series], steps: int) -> Union[np.array, pd.Series]:
        '''Sets the state of the FCM and performs a simulation.'''
        self.set_state(state)
        return self.simulate(steps)


    def clear_state(self) -> None:
        '''Clears the state of the FCM.'''
        self.state = np.zeros(self.connections.shape[0])


    def __add__(self, other: FuzzyCognitiveMap) -> FuzzyCognitiveMap:
        '''Creates a new FCM that is the addition of the two different connection matrix.'''
        return FuzzyCognitiveMap(self.connections + other.connections, self.threshold)
