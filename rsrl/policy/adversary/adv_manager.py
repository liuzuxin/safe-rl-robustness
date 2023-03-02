import numpy as np

from rsrl.policy.adversary.adv_random import AdvUniform
from rsrl.policy.adversary.adv_critic import AdvCritic, AdvCriticPPO
from rsrl.policy.adversary.adv_mad import AdvMad, AdvMadPPO
from rsrl.policy.adversary.adv_amad import AdvAmad, AdvAmadPPO


class AdvManager:
    def __init__(self, obs_dim, adv_cfg, ppo=False) -> None:
        super().__init__()
        self.adv_cfg = adv_cfg
        self.obs_dim = obs_dim
        self.noise_scale = self.adv_cfg["noise_scale"]
        self.attack_freq = self.adv_cfg["attack_freq"]

        mad_cfg = self.adv_cfg["mad_cfg"]
        amad_cfg = self.adv_cfg["amad_cfg"]
        max_cost_cfg = self.adv_cfg["max_cost_cfg"]
        max_reward_cfg = self.adv_cfg["max_reward_cfg"]
        min_reward_cfg = self.adv_cfg["min_reward_cfg"]

        mad = AdvMadPPO(self.obs_dim, **mad_cfg) if ppo else AdvMad(
            self.obs_dim, **mad_cfg)
        amad = AdvAmadPPO(self.obs_dim, **amad_cfg) if ppo else AdvAmad(
            self.obs_dim, **amad_cfg)
        AdvCriticCls = AdvCriticPPO if ppo else AdvCritic
        max_cost = AdvCriticCls(self.obs_dim, **max_cost_cfg)
        max_reward = AdvCriticCls(self.obs_dim, **max_reward_cfg)
        min_reward = AdvCriticCls(self.obs_dim, **min_reward_cfg)
        uniform = AdvUniform(self.obs_dim)

        self.adv = {
            "mad": mad,
            "amad": amad,
            "max_cost": max_cost,
            "max_reward": max_reward,
            "min_reward": min_reward,
            "uniform": uniform,
        }

    def get_adv(self, name):
        return self.adv[name]

    def set_amad_thres(self, qc_list):
        for adv_name in ["amad", "max_cost", "max_reward"]:
            adv = self.get_adv(adv_name)
            adv.set_thres(qc_list)