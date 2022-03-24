import numpy as np

import time
from copy import deepcopy

from typing import NamedTuple, List


class KineticTheorySimulationParameters(NamedTuple):
    peclet_number: float
    time_step: float
    simulation_iteration_number: int
    initial_condition: List[float]


def kinetic_theory_simulation(params: KineticTheorySimulationParameters):
    # set parameters
    peclet_number = params.peclet_number
    total_iteration = params.simulation_iteration_number
    space_dimension = 3
    initial_condition = params.initial_condition

    # set r result array
    r = np.zeros(shape=(space_dimension, total_iteration + 1))

    # set r1 initial value
    for r1_idx in range(space_dimension):
        # fill in initial value
        r[r1_idx, 0] = deepcopy(initial_condition[r1_idx])

    # run simulations
    dt = params.time_step
    for iteration in range(1, total_iteration + 1):
        t_begin = time.time()

        # get r_value (before)
        r1_before = r[0, iteration - 1]
        r2_before = r[1, iteration - 1]
        r3_before = r[2, iteration - 1]

        # calculate resistivity
        # TODO assumed L = ls
        r_absolute = np.sqrt(np.square(r1_before) + np.square(r2_before) + np.square(r3_before))
        resistivity = 1.6 * r_absolute + 1
        spring_force = (4 * r_absolute + 1/(1 - r_absolute)**2 - 1) / r_absolute / 6

        # update r1
        r[0, iteration] = r1_before + \
            (2 * peclet_number * r1_before - 1/resistivity * (spring_force * r1_before - 1)) * dt

        # update r2
        r[1, iteration] = r2_before + \
            (-2 * peclet_number * r2_before - 1/resistivity * (spring_force * r2_before - 1)) * dt

        # update r3
        r[2, iteration] = r3_before + \
            -1/resistivity * (spring_force * r3_before - 1) * dt

        t_end = time.time()
        print("Iteration %d took %.2f seconds" % (iteration, (t_end - t_begin)))

    # get result
    r1_result = r[0, total_iteration]
    r2_result = r[1, total_iteration]
    r3_result = r[2, total_iteration]

    return r1_result, r2_result, r3_result


if __name__ == '__main__':
    parameters = KineticTheorySimulationParameters(
        peclet_number=1.0,
        time_step=0.1,
        simulation_iteration_number=5,
        initial_condition=[0.25, 0.0, 0.0],
    )
    # answer maybe [0.78, 0.096, 0.171]
    result = kinetic_theory_simulation(params=parameters)
    # print
    for idx, value in enumerate(result):
        print("R%d square ensemble average value is %.3f" % (idx + 1, value))
