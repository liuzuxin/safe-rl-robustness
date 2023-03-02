import os.path as osp

from rsrl.runner import Runner
from rsrl.util.run_util import load_config

CONFIG_DIR = osp.join(osp.dirname(osp.realpath(__file__)), "config")
EXP_NAME_KEYS = {"epochs": "epoch", "rs_mode": "train_mode"}
DATA_DIR_KEYS = {"cost_limit": "cost"}


def get_cfg_value(config, key):
    if key in config:
        return str(config[key])
    for k in config.keys():
        if isinstance(config[k], dict):
            res = get_cfg_value(config[k], key)
            if res is not "None":
                return res
    return "None"


def gen_exp_name(config: dict, suffix=None):
    suffix = "" if suffix is None else "_" + suffix
    name = config["policy_name"]
    for k in EXP_NAME_KEYS:
        name += '_' + EXP_NAME_KEYS[k] + '_' + get_cfg_value(config, k)
    return name + suffix


def gen_data_dir_name(config: dict):
    name = config["env_cfg"]["env_name"]
    for k in DATA_DIR_KEYS:
        name += '_' + DATA_DIR_KEYS[k] + '_' + get_cfg_value(config, k)
    return name


def update_config(default_config, target_config):
    # replace the default config with search configs
    for k, v in default_config.items():
        if isinstance(v, dict):
            update_config(v, target_config)
        if k in target_config:
            default_config[k] = target_config[k]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '-e', type=str, default='SafetyCarCircle-v0')
    parser.add_argument('--policy', '-p', type=str, default='robust_ppo')
    parser.add_argument('--rs_mode', '-m', type=str, default='vanilla')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--suffix', '--id', type=str, default=None)
    args = parser.parse_args()

    args_dict = vars(args)

    if "ppo" in args.policy:
        args_dict["policy_name"] = "robust_ppo_lag"
        config_path = osp.join(CONFIG_DIR, "config_robust_ppo.yaml")
    elif "focops" in args.policy:
        args_dict["policy_name"] = "robust_focops"
        config_path = osp.join(CONFIG_DIR, "config_robust_focops.yaml")
    else:
        raise NotImplementedError

    config = load_config(config_path)
    update_config(config, args_dict)

    config["exp_name"] = gen_exp_name(config, args.suffix)
    config["data_dir"] = gen_data_dir_name(config)

    runner = Runner(**config, config_dict=config)
    runner.train()