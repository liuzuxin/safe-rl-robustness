import os.path as osp

import ray
from ray import tune

from rsrl.util.run_util import load_config
from rsrl.runner import Runner

CONFIG_DIR = osp.join(osp.dirname(osp.realpath(__file__)), "config")


def get_cfg_value(config, key):
    if key in config:
        value = config[key]
        if isinstance(value, list):
            suffix = ""
            for i in value:
                suffix += str(i)
            return suffix
        return str(value)
    for k in config.keys():
        if isinstance(config[k], dict):
            res = get_cfg_value(config[k], key)
            if res is not None:
                return res
    return "None"


def gen_exp_name(config: dict):
    name = config["policy_name"]
    for k in EXP_NAME_KEYS:
        name += '_' + EXP_NAME_KEYS[k] + '_' + get_cfg_value(config, k)
    return name


def gen_data_dir_name(config: dict):
    name = config["env_name"] if "env_name" in config else config["env_cfg"]["env_name"]
    for k in DATA_DIR_KEYS:
        name += '_' + DATA_DIR_KEYS[k] + '_' + get_cfg_value(config, k)
    return name + DATA_DIR_SUFFIX


def trial_name_creator(trial):
    config = trial.config
    name = config["env_name"]
    for k in DATA_DIR_KEYS:
        name += '_' + DATA_DIR_KEYS[k] + '_' + get_cfg_value(config, k)
    return name + DATA_DIR_SUFFIX + '_' + config["policy_name"]


def update_config(default_config, target_config):
    # replace the default config with search configs
    for k, v in default_config.items():
        if isinstance(v, dict):
            update_config(v, target_config)
        if k in target_config:
            default_config[k] = target_config[k]


def skip_exp(config):
    '''
    determine if we should skip this exp
    '''
    for skip in SKIP_EXP_CONFIG:
        state = True if len(skip) > 0 else False
        for k in skip:
            state = state and (skip[k] == config[k])
        if state:
            return True
    return False


def trainable(config):
    if skip_exp(config):
        '''
        Skip this exp if it satisfies some criterion
        '''
        return False


    config["exp_name"] = gen_exp_name(config)
    config["data_dir"] = gen_data_dir_name(config)

    policy_name = config["policy_name"]
    if "ppo" in policy_name:
        config_path = osp.join(CONFIG_DIR, "config_robust_ppo.yaml")
    elif "focops" in policy_name:
        config_path = osp.join(CONFIG_DIR, "config_robust_focops.yaml")
    else:
        raise NotImplementedError
    default_config = load_config(config_path)

    # replace the default config with search configs
    update_config(default_config, config)

    runner = Runner(**default_config, config_dict=default_config)
    runner.train()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        '-e',
                        type=str,
                        help='experiment env name, button, circle, goal, or push')
    parser.add_argument('--cpus',
                        '-c',
                        type=int,
                        default=4,
                        help='maximum cpu resources for ray')
    parser.add_argument('--threads',
                        '-t',
                        type=int,
                        default=4,
                        help='maximum cpu threads resources per trial')
    parser.add_argument('--gpus',
                        '-g',
                        type=int,
                        default=0,
                        help='maximum gpu resources for ray')
    parser.add_argument('--fracgpu',
                        '-f',
                        type=float,
                        default=0,
                        help='maximum Fractional GPUs resources per trial')

    args = parser.parse_args()

    ray.init(num_cpus=args.cpus, num_gpus=args.gpus)

    env = args.env.lower()
    # robust ppo
    if env == 'cc_ppo':
        from script.config.cc_ppo import *
    elif env == 'ac_ppo':
        from script.config.ac_ppo import *
    elif env == 'dc_ppo':
        from script.config.dc_ppo import *
    elif env == 'cr_ppo':
        from script.config.cr_ppo import *
    elif env == 'ar_ppo':
        from script.config.ar_ppo import *
    elif env == 'dr_ppo':
        from script.config.dr_ppo import *
    # robust focops
    elif env == 'cc_focops':
        from script.config.cc_focops import *
    elif env == 'cr_focops':
        from script.config.cr_focops import *
    
    EXP_CONFIG["threads"] = args.threads
    EXP_CONFIG["device"] = "cpu"
    if args.gpus > 0:
        EXP_CONFIG["device"] = "gpu"

    experiment_spec = tune.Experiment(
        args.env,
        trainable,
        config=EXP_CONFIG,
        resources_per_trial={
            "cpu": args.threads,
            "gpu": args.fracgpu
        },
        trial_name_creator=trial_name_creator,
    )

    tune.run_experiments(experiment_spec)