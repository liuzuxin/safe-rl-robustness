import os
import os.path as osp

import ray
from ray import tune

from rsrl.util.run_util import setup_eval_configs
from rsrl.evaluator import Evaluator

CONFIG_DIR = osp.join(osp.dirname(osp.realpath(__file__)), "config")


EXP_CONFIG = dict(
    load_dir=None,
    noise_scale=None,
    itr=None,
    device="cpu",
    threads=4,
)


def gen_data_dir_name(load_dir, env, noise=False):
    load_dirs = []
    for root, _, files in os.walk(load_dir):
        # print(root, files)
        if "eval" in root or env not in root:
            continue
        if "config.json" in files:
            load_dirs.append(root)
    return load_dirs


def trainable(config):
    load_dir = config["load_dir"]
    model_path, new_config = setup_eval_configs(load_dir, config["itr"])

    # update config
    new_config["evaluate_episode_num"] = 1
    new_config["epochs"] = config["epochs"]
    if config["itr"] is None:
        new_config["data_dir"] += "_eval_optimal"
    else:
        new_config["data_dir"] += "_eval_last"

    if config["noise_scale"] is not None:
        config["noise_scale"] = 0.025
        new_config["exp_name"] += '_noise_' + str(config["noise_scale"])

    new_config["eval_attackers"] = ["amad", "mad", "max_cost", "max_reward", "uniform"]
    evaluator = Evaluator(**new_config, config_dict=new_config)
    evaluator.eval(model_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_dir', '-d', nargs='+', default=[])
    parser.add_argument('--env', '-e', type=str, default="CarCircle")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('--optimal', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=0)

    parser.add_argument('--cpus',
                        '-c',
                        type=int,
                        default=4,
                        help='maximum cpu resources for ray')
    parser.add_argument('--threads',
                        '-t',
                        type=int,
                        default=1,
                        help='maximum threads resources per trial')

    args = parser.parse_args()

    ray.init(num_cpus=args.cpus)
    
    EXP_CONFIG["threads"] = args.threads
    EXP_CONFIG["epochs"] = args.epochs
    load_dir = args.load_dir[0]
    load_dirs = gen_data_dir_name(load_dir, args.env, args.noise)
    EXP_CONFIG["load_dir"] = tune.grid_search(load_dirs)
    if not args.optimal:
        EXP_CONFIG["itr"] = args.itr

    if args.noise:
        noise = [i*0.015 for i in range(11)]
        EXP_CONFIG["noise_scale"] = tune.grid_search(noise)

    evaluation_spec = tune.Experiment(
        "eval",
        trainable,
        config=EXP_CONFIG,
        resources_per_trial={
            "cpu": args.threads,
            "gpu": 0
        },
    )

    tune.run_experiments(evaluation_spec)
