import numpy as np

import time
from copy import deepcopy

from typing import NamedTuple, List

import matplotlib.pyplot as plt


class BrownianDynamicsSimulationParameters(NamedTuple):
    peclet_number: float
    time_step: float
    simulation_iteration_number: int
    initial_r1: List[float]
    ensemble_number_per_each_dumbbell: int


def brownian_dynamics_simulation(params: BrownianDynamicsSimulationParameters):
    # set parameters
    peclet_number = params.peclet_number
    total_iteration = params.simulation_iteration_number
    ensemble_number = params.ensemble_number_per_each_dumbbell
    space_dimension = 3
    initial_r1 = params.initial_r1
    r1_initial_size = len(initial_r1)

    # set r result array
    r = np.zeros(shape=(space_dimension, total_iteration + 1, r1_initial_size, ensemble_number))

    # set r1 initial value
    for r1_idx in range(r1_initial_size):
        # fill in initial value
        r[0, 0, r1_idx, :] = deepcopy(initial_r1[r1_idx])

    # run simulations
    dt = params.time_step
    for iteration in range(1, total_iteration + 1):
        t_begin = time.time()
        for ensemble_idx in range(ensemble_number):
            # get normal vectors
            normal_y_1, normal_y_2, normal_y_3 = np.random.multivariate_normal(
                mean=[0.0, 0.0, 0.0],
                cov=np.diag([1.0, 1.0, 1.0])
            )
            normal_z_1, normal_z_2, normal_z_3 = np.random.multivariate_normal(
                mean=[0.0, 0.0, 0.0],
                cov=np.diag([1.0, 1.0, 1.0])
            )
            # get r_value (before)
            r1_before = r[0, iteration - 1, :, ensemble_idx]
            r2_before = r[1, iteration - 1, :, ensemble_idx]
            r3_before = r[2, iteration - 1, :, ensemble_idx]

            # calculate resistivity
            # TODO assumed L = ls
            r_absolute = np.sqrt(np.square(r1_before) + np.square(r2_before) + np.square(r3_before))
            resistivity = 1.6 * r_absolute + 1.0
            spring_force = (4 * r_absolute + 1/(1 - r_absolute)**2 - 1) / r_absolute / 6

            # update r1
            r[0, iteration, :, ensemble_idx] = r1_before + \
                (peclet_number * r1_before - 0.5/resistivity * spring_force * r1_before) * dt \
                + np.sqrt(0.5 * dt/resistivity) * (normal_y_1 - normal_z_1)
            # # corrector-predictor
            # r[0, iteration, :, ensemble_idx] = r1_before + \
            #     (peclet_number * r1_before - 0.5/resistivity * spring_force * r1_before) * dt *0.5 \
            #     + np.sqrt(0.5 * dt/resistivity) * (normal_y_1 - normal_z_1)

            # update r2
            r[1, iteration, :, ensemble_idx] = r2_before + \
                (-peclet_number * r2_before - 0.5/resistivity * spring_force * r2_before) * dt \
                + np.sqrt(0.5 * dt/resistivity) * (normal_y_2 - normal_z_2)

            # update r3
            r[2, iteration, :, ensemble_idx] = r3_before + \
                (-0.5/resistivity * spring_force * r3_before) * dt \
                + np.sqrt(0.5 * dt/resistivity) * (normal_y_3 - normal_z_3)

        t_end = time.time()
        print("Iteration %d took %.2f seconds" % (iteration, (t_end - t_begin)), flush=True)

    plt.figure(figsize=(8, 8))
    t_array = np.array([num * dt for num in range(total_iteration + 1)])
    plt.plot(t_array, r[0, :, 0, 0], 'r-', label='R1')
    plt.plot(t_array, r[1, :, 0, 0], 'g-', label='R2')
    plt.plot(t_array, r[2, :, 0, 0], 'b-', label='R3')

    plt.xlabel("time", fontsize=14)
    plt.ylabel("spring vector", fontsize=14)
    plt.title("BD simulation", fontsize=14)

    plt.legend(fontsize=14)
    plt.savefig("time_step_%s_iter_%s" % ('1e-5', '100000'))
    plt.show()
    plt.close()

    # get result
    r1_result = r[0, total_iteration]
    r2_result = r[1, total_iteration]
    r3_result = r[2, total_iteration]

    # ensemble average
    r1_square_average = np.average(np.square(r1_result))
    r2_square_average = np.average(np.square(r2_result))
    r3_square_average = np.average(np.square(r3_result))

    return r1_square_average, r2_square_average, r3_square_average


if __name__ == '__main__':
    parameters = BrownianDynamicsSimulationParameters(
        peclet_number=1.0,
        time_step=1e-3,
        simulation_iteration_number=1000,
        initial_r1=list(np.arange(start=0.5, stop=0.55, step=0.1)),
        ensemble_number_per_each_dumbbell=50
    )
    # At least require time step -> 1e-5
    # answer maybe [0.275, 0.0058, 0.0056]
    result = brownian_dynamics_simulation(params=parameters)
    # print
    for idx, value in enumerate(result):
        print("R%d square ensemble average value is %.7f" % (idx + 1, value), flush=True)
