from copy import deepcopy
from tqdm import tqdm
import gym
import numpy as np
import torch
import torch.nn as nn
from rsrl.policy.onpolicy_base import OnPolicyBase
from rsrl.policy.model.mlp_ac import EnsembleQCritic
from rsrl.util.logger import EpochLogger
from rsrl.util.torch_util import to_device, to_ndarray, to_tensor
from rsrl.policy.adversary.adv_random import AdvUniform
from rsrl.policy.adversary.adv_critic import AdvCriticPPO
from rsrl.policy.adversary.adv_mad import AdvMadPPO
from rsrl.policy.adversary.adv_amad import AdvAmadPPO
from rsrl.policy.adversary.adv_base import Adv
from torch.optim import Adam


class RobustFOCOPS(OnPolicyBase):
    attacker_cls = dict(uniform=AdvUniform,
                        mad=AdvMadPPO,
                        amad=AdvAmadPPO,
                        max_reward=AdvCriticPPO,
                        min_reward=AdvCriticPPO,
                        max_cost=AdvCriticPPO)

    def __init__(self, env: gym.Env, 
                 logger: EpochLogger, 
                 env_cfg: dict, 
                 adv_cfg: dict,
                 train_actor_iters=80, 
                 kl_coef=1, 
                 nu=0,
                 nu_lr=0.01, 
                 nu_max=2.0, 
                 l2_reg=0.001, 
                 tem_lam=1.5, 
                 eta=0.02, 
                 delta=0.02, 
                 **kwargs) -> None:
        super().__init__(env, logger, env_cfg, adv_cfg, **kwargs)

        self.train_actor_iters = train_actor_iters
        self.kl_coef = kl_coef
        # Cost coefficient
        self.nu = nu
        # Cost coefficient learning rate
        self.nu_lr = nu_lr
        # Maximum cost coefficient
        self.nu_max = nu_max
        # L2 Regularization Rate
        self.l2_reg = l2_reg
        # Inverse temperature lambda
        self.tem_lam = tem_lam
        # KL bound for indicator function
        self.eta = eta
        # KL bound
        self.delta = delta

        self.policy_loss = self.vanilla_policy_loss

        # Set up model saving
        self.save_model()

    def vanilla_policy_loss(self, obs, act, logp_old, advantage, cost_advantage, *args,
                            **kwargs):
        pi, _, logp = self.actor_forward(obs, act)
        ratio = torch.exp(logp - logp_old)
        approx_kl = (logp_old - logp).mean().item()
        loss_pi = (approx_kl - (1 / self.lam) * ratio *
                   (advantage - self.nu * cost_advantage)) * (approx_kl <= self.eta)
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, LossVallina=to_ndarray(loss_pi))
        return loss_pi.mean(), pi_info, pi

    def _update_actor(self, data):
        '''
        Update the actor network
        '''
        # update nu
        avg_cost = data['cost_mean'].item()
        self.nu += self.nu_lr * (avg_cost - self.cost_limit)
        if self.nu < 0:
            self.nu = 0
        elif self.nu > self.nu_max:
            self.nu = self.nu_max
        self.logger.store(nu=self.nu, AvgCost=avg_cost)

        obs, act, logp_old = data['obs'], data['act'], data['logp']
        adv, cadv = data['adv'], data['cost_adv']

        pi_l_old, pi_info_old, _ = self.policy_loss(obs, act, logp_old, adv, cadv)

        for i in range(self.train_actor_iters):
            self.actor_optimizer.zero_grad()
            loss_pi, pi_info, pi = self.policy_loss(obs, act, logp_old, adv, cadv)
            loss_pi.backward()
            self.actor_optimizer.step()

        self.logger.store(LossPi=to_ndarray(pi_l_old),
                          CostLimit=self.cost_limit,
                          DeltaLossPi=(to_ndarray(loss_pi) - to_ndarray(pi_l_old)),
                          QcThres=self.cost_limit,
                          QcRet=torch.mean(data["cost_ret"]).item(),
                          **pi_info)
