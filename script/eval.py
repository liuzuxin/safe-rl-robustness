from rsrl.evaluator import Evaluator
from rsrl.util.run_util import setup_eval_configs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '-e', type=str, default='SafetyCarCircle-v0')
    parser.add_argument('--policy', '-p', type=str, default='robust_ppo')
    parser.add_argument('--pretrain_dir', '-pre', type=str, default=None)

    parser.add_argument('--load_dir', '-d', type=str, default=None)
    parser.add_argument('--itr', '-i', type=int, default=None)

    parser.add_argument('--mode', '-m', type=str, default='eval')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--suffix', '--id', type=str, default=None)
    parser.add_argument('--no_render', action="store_true")
    parser.add_argument('--sleep', type=float, default=0.003)
    args = parser.parse_args()

    assert args.load_dir is not None, "The load_path parameter has not been specified!!!"
    model_path, config = setup_eval_configs(args.load_dir, args.itr)

    # update config
    config["evaluate_episode_num"] = 1
    config["epochs"] = args.epochs
    config["data_dir"] += "_eval"
    config["eval_attackers"] = ["amad", "max_cost", "max_reward", "uniform", "mad"]

    evaluator = Evaluator(**config, config_dict=config)
    evaluator.eval(model_path)