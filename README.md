On the Robustness of Safe Reinforcement Learning under Observational Pertubrations
==================================

This project provides the open source implementation of the robust safe RL introduced in the ICLR 2023 paper: "On the Robustness of Safe Reinforcement Learning under Observational Pertubrations" [(Liu, et al. 2023)](https://arxiv.org/abs/2205.14691). 

Safe RL trains a policy to maximize the reward while satisfying constraints.
While prior works focus on the performance optimality, we find that the optimal solutions of many safe RL problems are not robust and safe against carefully designed observational perturbations.
We propose two adversarial attacks - one maximizes the cost and the other maximizes the reward. 
One interesting and counter-intuitive finding is that the maximum reward attack is strong, as it can both induce unsafe behaviors and make the attack stealthy by maintaining the reward.
We further propose a defense method based on adversarial training, which can make the agent stay safe under attacks.
Video demos are available at the [project webpage](https://sites.google.com/view/robustsaferl/home).

If you find this code useful, consider to cite:
```bibtex
@article{liu2022robustness,
  title={On the robustness of safe reinforcement learning under observational perturbations},
  author={Liu, Zuxin and Guo, Zijian and Cen, Zhepeng and Zhang, Huan and Tan, Jie and Li, Bo and Zhao, Ding},
  journal={arXiv preprint arXiv:2205.14691},
  year={2022}
}
```

## Table of Contents

- [Environment setup](#environment-setup)
    - [System requirements](#system-requirements)
    - [Installation](#installation)
- [How to run experiments](#how-to-run-experiments)
- [Pretrained weights](#pretrained-weights)
- [Acknowledgments](#acknowledgments)

The structure of this repo is as follows:
```
Robust safe RL libraries
├── rsrl  # package folder
│   ├── policy # core algorithm implementation
│   ├── ├── model # stores the actor critic model architecture
│   ├── ├── policy_name # algorithms implementation
│   ├── util # logger and pytorch utils
│   ├── runner.py # training logic of the algorithms
│   ├── evaluator.py # evaluation logic of trained agents
├── script  # stores the training scripts.
│   ├── config # stores some configs of the env and policy
│   ├── run.py # launch a single experiment
│   ├── experiment.py # launch multiple experiments in parallel with ray
│   ├── eval.py # evaluate script of trained agents
├── data # stores experiment results
```

## Environment setup
### System requirements
- The repo is tested in Ubuntu 20.04 and should be fine with Ubuntu 18.04
- We recommend to use [Anaconda3](https://docs.anaconda.com/anaconda/install/) for python env management

### Installation
1.  **Activate a python 3.7+ virtual anaconda env**, then install the `bullet_safety_gym` simulation environment:
```
cd envs/Bullet-Safety-Gym
pip install -e .
cd ../..
```

2. After switching back to the repo root folder, install the dependencies that are listed in `requirement.txt` and the `rsrl` library:
```
pip install -r requirement.txt
pip install -e .
```

3. Then install `pytorch` based on your system configurations, see [instructions here](https://pytorch.org/get-started/locally/). 
For example, installing a cpu-only version `pytorch` via [Anaconda3](https://docs.anaconda.com/anaconda/install/) by the following command:
```
conda install pytorch cpuonly -c pytorch
```

4. The MAD attacker requires pysgmcmc library for optimization. Install it by:
```
pip install git+https://github.com/MFreidank/pysgmcmc@pytorch
```

## How to run experiments

To run a single experiment:
```
python script/run.py --rs_mode vanilla --policy robust_ppo
```

To run multiple experiments in parallel:
```
python script/experiment.py -e experiment_name 
```


To evaluate a trained model, run:
```
python script/eval.py -d path_to_model
```

To evaluate multiple trained model in parallel:
```
python script/evaluation.py -d path_to_model -e env_name
```

The complete hyper-parameters can be found in `script/config/config_robust_ppo.yaml`. 

In particular, PPO-Lagrangian has different robust training modes, which are specified by the `rs_mode` parameter. We detail the modes in the following table.

| Algorithm |   PPOL  | ADV-PPOL(MC) | ADV-PPOL(MR) | PPOL-random | SA-PPOL | SA-PPOL(MC) | SA-PPOL(MR) |
|:---------:|:-------:|:------------:|:------------:|:-----------:|:-------:|:-----------:|:-----------:|
|    Mode   | vanilla |   max_cost   |  max_reward  |   uniform   |    kl   |     klmc    |     klmr    |

- The proposed adversarial training methods correspond to the `max_cost, max_reward` modes.
- For SA-PPOL series, the modes are `kl, klmc, klmr`. The SA-PPOL with the original MAD attacker is the `kl` mode, the SA-PPOL method with the MC and MR attackers are `klmc` and `klmr` respectively. 
- Note that FOCOPS also supports the adversarail training modes `max_cost, max_reward` and `uniform, vanilla`.

## Pretrained weights

The pretrained weights are available at [here](https://drive.google.com/drive/folders/1wgy5DuNEBBUGx_SfmFme_u9i0JqpqMZe?usp=share_link).


## Acknowledgments
Part of the code is based on several public repos:
* https://github.com/SvenGronauer/Bullet-Safety-Gym, note that our BulletSafetyGym is modified based on the original one. The major modification is the simulation step where we increase it to reduce the total training time without sacrifacing too much accuracy. 
* https://github.com/openai/spinningup
