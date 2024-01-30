from __future__ import annotations
'''
Module to use fuzzy cognitive maps.

The module contains the class FuzzyCognitiveMap, which is used to create and
use fuzzy cognitive maps. You can also plot them, or simulate the dynamics of
the FCM.

For the original papers about FCM, see the works by Bart Kosko.

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from . import maintenance as mnt
except ImportError:

    import maintenance as mnt


def _threshold_modules(connections: np.array | pd.DataFrame, threshold) -> np.array | pd.DataFrame:
    '''Thresholds the connections matrix to the {-1, 0, 1} values.'''
    return np.abs(connections) > threshold * np.sign(connections)


def __look_periods(states: list[np.array], min_period_len=2) -> list[np.array]:
    '''Looks for periods in the states list. Returns the period if found, None otherwise.'''
    max_period_len = int(len(states) / 2)

    for period_len in np.arange(max_period_len, min_period_len, -1):

        for i in range(len(states)):
            candidate = states[i:i+period_len]
            next_candidate = states[i+period_len:i+period_len*2]
            if len(next_candidate) < min_period_len:
                break
            if candidate != next_candidate:
                break
        
        return candidate


    return None


def __look_attractors(states: list[np.array]) -> [bool, np.array]:
    '''Checks if all the states in the list are the same'''
    attractors = []
    for state in states:
        if not state in attractors:
            attractors.append(state)
        else:
            return False, []
    
    return True, attractors[0]
    

def look_pattern_states(fcm: FuzzyCognitiveMap, sim_steps: int, pattern_len: 50, max_period_size: 50) -> [np.array]:
    '''Looks for the pattern states of the FCM when simulation is prolongued.
    
    
    :param fcm : FuzzyCognitiveMap. The FCM to look for the attractor states.
    :param max_steps: int. The maximum number of steps to look for the attractor states.
    :param random_inits : int
    :returns: list of np.array. The attractor states found. None if none were found
    '''
    for _ in range(sim_steps):
        state = fcm.step()
    
    steps = []
    for _ in range(pattern_len):
        state = fcm.step()
        steps.append(state)

    satisfactory, period = __look_attractors(steps)
    if not satisfactory:
        period = __look_periods(steps, min_period_len=2)
        
    return period


def study_attractors_FCM(fcm: FuzzyCognitiveMap, max_steps: int, random_inits: int=10) -> [np.array]:
    '''Looks for the attractor states of the FCM when simulation is prolongued.
    
    
    :param fcm : FuzzyCognitiveMap. The FCM to look for the attractor states.
    :param max_steps: int. The maximum number of steps to look for the attractor states.
    :param random_inits : int
    :returns: list of np.array. The attractor states found. None if none were found
    '''
    if mnt.save_usage_flag:
        mnt.usage_data[mnt.usage_categories.FuzzyCognitiveMaps]['fcm_report'] += 1

    attractors = {}
    gen_random_state = lambda : np.random.randint(0, 2, fcm.connections.shape[0])
    for _ in range(random_inits):
        init_state = gen_random_state()
        fcm.set_state(init_state)
        attractors[init_state] = look_pattern_states(fcm, max_steps)
    
    return attractors


def attractors_report(attractors: dict[np.array, np.array]) -> None:
    '''
    Prints a report of the attractors found.
    
    :param attractors: dict[np.array, np.array]. The attractors found.
    '''
    if mnt.save_usage_flag:
            mnt.usage_data[mnt.usage_categories.FuzzyCognitiveMaps]['fcm_report'] += 1
    pattern_dict = {}

    for _, attractor in attractors.items():
        if attractor is None:
            pattern_dict['Chaotic'] = pattern_dict.get('Chaotic', 0) + 1 / len(attractors)
        else:
            pattern_dict[str(attractor)] = pattern_dict.get(str(attractor), 0) + 1 / len(attractors)

    state_dict = {}
    list_states = []
    for _, attractor in attractors.items():
        if attractor is not None:
            for state in attractor:
                list_states += [str(state)]

    for state in list_states:
        state_dict[state] = state_dict.get(state, 0) + 1 / len(list_states)

    return pattern_dict, state_dict


class FuzzyCognitiveMap:

    def __init__(self, connections: np.array | pd.DataFrame, threshold:int=0.5) -> None:
        '''
        Creates a fuzzy cognitive map.
        
        
        :param connections: np.array | pd.DataFrame. A square matrix with the connections between the concepts.
        :param threshold: int, optional. When simulating steps the state
        will be trimmed using these threhold into the {0, 1, -1} values.
        The default is 0.5.
        '''
        if mnt.save_usage_flag:
            mnt.usage_data[mnt.usage_categories.FuzzyCognitiveMaps]['fcm_create'] += 1

        self.connections = connections
        self.state = np.zeros(connections.shape[0])
        self.threshold = threshold
    

    def var_names(self) -> list[str]:
        '''Returns the names of the variables.'''
        try:
            return list(self.connections.columns)
        except AttributeError:
            return None
    

    def step(self) -> np.array | pd.DataFrame:
        '''Performs a step in the FCM given the actual state.'''
        self.state = _threshold_modules(self.state @ self.connections, self.threshold)
    
        if isinstance(self.state, pd.DataFrame):
            return pd.DataFrame(self.state, columns=self.var_names())
        else:
            return self.state


    def simulate(self, steps: int) -> np.array | pd.DataFrame:
        '''
        Simulates the FCM for a number of steps.
        
        :param steps: int. The number of steps to simulate.
        '''
        for _ in range(steps):
            fstep = self.step()
        
        return fstep
    

    def add(self, other: FuzzyCognitiveMap) -> None:
        '''Adds the connections of other FCM to the actual FCM.'''
        self.connections = self.connections + other.connections


    def set_state(self, state: np.array | pd.DataFrame) -> None:
        '''Sets the state of the FCM.'''
        try:
            self.state = state.values
        except AttributeError:
            self.state = state


    def set_and_step(self, state: np.array | pd.DataFrame) -> np.array | pd.DataFrame:
        '''Sets the state of the FCM and performs a step.'''
        self.set_state(state)
        return self.step()
    

    def set_and_simulate(self, state: np.array | pd.DataFrame, steps: int) -> np.array | pd.DataFrame:
        '''Sets the state of the FCM and performs a simulation.'''
        self.set_state(state)
        return self.simulate(steps)


    def clear_state(self) -> None:
        '''Clears the state of the FCM.'''
        self.state = np.zeros(self.connections.shape[0])


    def __add__(self, other: FuzzyCognitiveMap) -> FuzzyCognitiveMap:
        '''Creates a new FCM that is the addition of the two different connection matrix.'''
        return FuzzyCognitiveMap(self.connections + other.connections)
