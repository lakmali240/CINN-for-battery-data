from dataset import load_battery, find_regen_cycles
from filtering import ParticleFilter
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import time

logging.basicConfig(level=logging.INFO, format="%(levelname)-5s -- %(message)s")
logger = logging.getLogger("pf")


class CustomModel:
    def __init__(self):
        self.__k = 0
        # without regen
        # self.initial_ranges = [(0.95, 1.05), (0.5, 1), (1.5, 3.5)]
        # with regen
        self.initial_ranges = [(0.95, 1.05), (0.5, 1), (0.25, 2.25)]
        self.__noise_var = np.array([1e-4, 1e-4, 1e-4])
        self.particle_shape = (3,)
        self.decay = 1  # 0.99

    @property
    def noise_var(self):
        return self.__noise_var * (self.decay ** self.__k)

    def __call__(self, particles):
        self.__k += 1
        delta = (particles[:, 1] * (particles[:, 2]) * self.__k) / (
            1e4 + (self.__k * particles[:, 2]) ** 2
        )
        particles[:, 0] = particles[:, 0] - delta

        return particles


class RegenModel:
    def __init__(self, poisson_rate, distr, num_particles):
        self.__poisson_rate = poisson_rate
        self.__poisson_period = round(1.0 / poisson_rate)
        self.__distr = distr
        self.offsets = np.ones(num_particles) * np.floor(
            np.random.random() * self.__poisson_period
        )

    def __call__(self, particles, k):
        for i in range(particles.shape[0]):
            if k % self.__poisson_period == self.offsets[i]:
                sample = self.__distr["distr"].rvs(*self.__distr["params"])
                particles[i, 0] += sample
        return particles


def meas_equation(particles):
    return particles


def run_filter(data, pf, state_model, regen_model, predict_after, total_sim_cycles):
    pf.init_prior_particles(*state_model.initial_ranges)
    total_cycles = len(data)
    history = np.zeros(
        (total_sim_cycles, pf.particles.shape[0], len(state_model.initial_ranges))
    )
    weight_history = np.zeros((total_sim_cycles, pf.particles.shape[0]))
    rul = np.zeros(pf.particles.shape[0])
    k = 0
    while k < total_sim_cycles:
        # for k in range(total_sim_cycles):
        history[k] = pf.particles
        weight_history[k] = pf.weights
        if k < predict_after:
            meas = data[k]
            pf.update(np.array([meas]), states=(0,))
        else:
            pf.state_noise_var = state_model.noise_var
            pf.predict(freeze_states=range(pf.particles.shape[-1]))
            if regen_model:
                pf.particles = regen_model(pf.particles, k)
        k += 1

    predicted = np.sum(weight_history * history[:, :, 0], axis=1)
    predicted = np.median(history[:, :, 0], axis=1)
    # check RMSE over tracking portion only
    rmse = np.sqrt(
        np.mean((predicted[:total_sim_cycles] - data[:total_sim_cycles]) ** 2)
    )

    rul = np.argmax(predicted <= 0.7)
    rul = rul if rul > 0 else np.nan

    return history, weight_history, rmse, rul


def run_pf_algorithm(args):
    logger.info(f"Loading data from {args.battery_path}")
    data = load_battery(args.battery_path).astype(np.float64)

    logger.info(f"Finding regen cycles before cycle {args.predict_after}")
    cycles = np.array(find_regen_cycles(data, threshold=0.01))
    cycles = cycles[np.where(cycles <= args.predict_after)]

    logger.info(f"Found {len(cycles)} regeneration events")

    cpp_lambda = len(cycles) / args.predict_after
    logger.info(f"Estimating CPP lambda = {cpp_lambda:.3f}")

    logger.info(f"Fitting a distribution to magnitudes")
    mags = data[cycles] - data[cycles - 1]
    distr_function = scipy.stats.gamma
    params = distr_function.fit(mags)
    best_dist = {"distr": distr_function, "params": params}

    logger.info(f"Using {args.particles} particles")
    logger.info(f"Switching to prognosis after {args.predict_after} cycles")

    best_rmse = np.inf
    best_history = None
    all_histories = []
    rmse_history = np.zeros(args.runs)
    rul_history = np.zeros(args.runs)

    tic = time.time()
    for i in range(args.runs):
        state_model = CustomModel()
        regen_model = (
            RegenModel(
                poisson_rate=cpp_lambda, distr=best_dist, num_particles=args.particles
            )
            if not args.no_regen
            else None
        )

        pf = ParticleFilter(
            state_func=state_model,
            state_noise_var=np.array(state_model.noise_var),
            meas_func=meas_equation,
            meas_noise_var=args.meas_var,
            num_particles=args.particles,
            shape=(len(state_model.initial_ranges),),
        )

        history, weight_history, rmse, rul = run_filter(
            data,
            pf,
            state_model,
            regen_model,
            args.predict_after,
            total_sim_cycles=args.predict_after,  # only run tracking phase
        )
        rul_history[i] = rul
        all_histories.append((history, rmse))

        logger.info(f"End of run {i+1} -- RMSE = {rmse:.4f} -- RUL = {rul:.1f}")
        if rmse < best_rmse:
            logger.info("Saving as best")
            best_history = history.copy()
            best_rmse = rmse

        rmse_history[i] = rmse

    toc = time.time()
    return toc - tic
    # rul_history = rul_history[np.where(np.invert(np.isnan(rul_history)))]
    # logger.info(
    #     f"Done. Mean Run RMSE = {np.mean(rmse_history):.4f} \u00B1 {np.std(rmse_history):.4f} -- RUL = {np.mean(rul_history):.1f} \u00B1 {np.std(rul_history):.1f}"
    # )

    # all_histories = sorted(all_histories, key=lambda a: a[1])
    # median = all_histories[len(all_histories) // 2]
    # best_history = median[0]


def main(args):
    print(f"PARTICLE FILTERING")
    print(args)
    times = []
    for _ in range(args.timing_trials):
        times.append(run_pf_algorithm(args))
        print(f"PF Time: {times[-1]:0.2f} s")

    times = np.array(times)
    print(f"Mean: {np.mean(times)}, Std: {np.std(times)}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("battery_path", type=str, help="path to battery data")
    parser.add_argument(
        "--particles", type=int, help="number of particles", required=True
    )
    parser.add_argument(
        "--predict-after",
        type=int,
        help="switch to prognosis after this many cycles",
        required=True,
    )
    parser.add_argument("--runs", type=int, default=1, help="# of Monte Carlo runs")
    parser.add_argument(
        "--no-regen",
        action="store_true",
        default=False,
        help="suppress regeneration simulation",
    )
    parser.add_argument(
        "--meas-var", type=float, required=True, help="measurement noise variance"
    )
    parser.add_argument(
        "--timing-trials",
        type=int,
        default=1,
        help="number of trials for estimating runtime",
    )
    main(parser.parse_args())