import gym
import numpy as np
import torch
from rsrl.policy import LagrangianPIDController
from rsrl.policy.adversary.adv_base import Adv
from rsrl.policy.onpolicy_base import OnPolicyBase
from rsrl.util.logger import EpochLogger
from rsrl.util.torch_util import to_device, to_ndarray, to_tensor
from torch.nn.utils import clip_grad_norm_


class RobustPPOLagrangian(OnPolicyBase):

    def __init__(self, env: gym.Env, 
                 logger: EpochLogger, 
                 env_cfg: dict, 
                 adv_cfg: dict,
                 KP=0.1, 
                 KI=0.003, 
                 KD=0.001, 
                 per_state=False, 
                 clip_ratio=0.2, 
                 clip_ratio_adv=0.8, 
                 target_kl=0.01,
                 train_actor_iters=80, 
                 kl_coef=1, 
                 **kwargs) -> None:
        r'''
        Proximal Policy Optimization (PPO) with Lagrangian multiplier
        '''
        super().__init__(env, logger, env_cfg, adv_cfg, **kwargs)

        self.clip_ratio = clip_ratio
        self.clip_ratio_adv = clip_ratio_adv
        self.target_kl = target_kl
        self.train_actor_iters = train_actor_iters

        self.kl_coef = kl_coef

        self.controller = LagrangianPIDController(KP, KI, KD, self.cost_limit, per_state)

        self.controller_adv = LagrangianPIDController(KP, KI, KD, self.cost_limit,
                                                      per_state)

        self.policy_loss = self.vanilla_policy_loss
        if "kl" in self.mode:
            self._init_sappo_loss()

        # Set up model saving
        self.save_model()

    def _init_sappo_loss(self):
        ############## for SA-PPOL mode #################
        if self.mode == "klmc":
            cfg = self.adv_cfg["max_cost_cfg"]
            self.adversary = self.attacker_cls["max_cost"](self.obs_dim, **cfg)
        elif self.mode == "klmr":
            cfg = self.adv_cfg["max_reward_cfg"]
            self.adversary = self.attacker_cls["max_reward"](self.obs_dim, **cfg)
        elif self.mode == "kl":
            cfg = self.adv_cfg["mad_cfg"]
            self.adversary = self.attacker_cls["mad"](self.obs_dim, **cfg)
        self.policy_loss = self.kl_regularized_policy_loss

    def get_obs_adv(self, adv: Adv, obs: torch.Tensor):
        epsilon = adv.attack_batch(self, obs, self.noise_scale)
        obs_adv = (epsilon + obs).detach()
        return obs_adv

    def vanilla_policy_loss(self, obs, act, logp_old, advantage, cost_advantage,
                            multiplier, *args, **kwargs):
        pi, _, logp = self.actor_forward(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio,
                               1 + self.clip_ratio) * advantage

        qc_penalty = (ratio * cost_advantage * multiplier).mean()
        loss_vallina = -(torch.min(ratio * advantage, clip_adv)).mean()
        loss_pi = loss_vallina + qc_penalty
        loss_pi /= 1 + multiplier
        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()

        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(KL=approx_kl,
                       Entropy=ent,
                       ClipFrac=clipfrac,
                       LossQcPenalty=to_ndarray(qc_penalty),
                       LossVallina=to_ndarray(loss_vallina))

        return loss_pi, pi_info, pi

    def kl_regularized_policy_loss(self, obs, act, logp_old, advantage, cost_advantage,
                                   multiplier, *args, **kwargs):
        loss_pi, pi_info, pi = self.vanilla_policy_loss(obs, act, logp_old, advantage,
                                                        cost_advantage, multiplier)
        with torch.no_grad():
            _, a_targ, _ = self.actor_forward(obs, deterministic=True)

        pi_adv, a_adv, _ = self.actor_forward(self.obs_adv, deterministic=True)
        kl_adv = ((a_targ.detach() - a_adv)**2).sum(axis=-1)
        kl_regularizer = torch.mean(kl_adv) * self.kl_coef

        loss_pi += kl_regularizer
        pi_info["LossKLAdv"] = to_ndarray(kl_regularizer)
        return loss_pi, pi_info, pi

    def _update_obs_adv(self, obs):
        if "kl" in self.mode:
            self.obs_adv = self.get_obs_adv(self.adversary, obs)

    def _update_actor(self, data):
        '''
        Update the actor network
        '''
        obs, act, advantage, logp_old = data['obs'], data['act'], data['adv'], data[
            'logp']
        cost_advantage = data["cost_adv"]
        ep_cost = data["ep_cost"]

        # detach is very important here!
        # Otherwise the gradient will backprop through the multiplier.
        multiplier = self.controller.control(ep_cost).detach()
        self._update_obs_adv(obs)
        pi_l_old, pi_info_old, _ = self.policy_loss(obs, act, logp_old, advantage,
                                                    cost_advantage, multiplier)

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_actor_iters):
            self.actor_optimizer.zero_grad()

            loss_pi, pi_info, pi = self.policy_loss(obs, act, logp_old, advantage,
                                                    cost_advantage, multiplier)
            if i == 0 and pi_info['KL'] >= 1e-7:
                print("**" * 20)
                print("1st kl: ", pi_info['KL'])
            if pi_info['KL'] > 1.5 * self.target_kl:
                self.logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break

            loss_pi.backward()
            clip_grad_norm_(self.actor.parameters(), 0.02)
            self.actor_optimizer.step()

        self.logger.store(StopIter=i,
                          LossPi=to_ndarray(pi_l_old),
                          Lagrangian=to_ndarray(multiplier),
                          QcRet=torch.mean(data["cost_ret"]).item(),
                          **pi_info)
