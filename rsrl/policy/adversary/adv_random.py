import time

import numpy as np
import torch
from rsrl.policy.adversary.adv_base import Adv
from rsrl.util.torch_util import to_ndarray, to_tensor, to_device


class AdvRandom(Adv):
    def __init__(self, obs_dim, **kwarg) -> None:
        super().__init__(**kwarg)
        self.id = 'AdvRandom'
        self.obs_dim = obs_dim

    def attack_batch(self, policy, obs, bound: float):
        '''
        Used in training.
        Given a batch of obs data and the bound, return the perturbation
        @param obs, tensor, [batch, obs_dim]
        @return epsilon, tensor, [batch, obs_dim]
        '''
        batch = obs.shape[0]
        epsilon = torch.distributions.uniform.Uniform(-bound, bound).sample(
            [batch, self.obs_dim])
        return to_device(epsilon)

    def attack_at_eval(self, policy, obs, bound: float):
        '''
        Used in evaluation.
        Given a single obs data and the bound, return the perturbation
        @param obs, ndarray, [obs_dim]
        @return epsilon, ndarray, [obs_dim]
        '''
        obs = to_tensor(obs).reshape(1, -1)
        epsilon = self.attack_batch(policy, obs, bound)
        return to_ndarray(torch.squeeze(epsilon))


class AdvUniform(Adv):
    def __init__(self, obs_dim, **kwarg) -> None:
        super().__init__(**kwarg)
        self.id = 'AdvUniform'
        self.obs_dim = obs_dim

    def attack_batch(self, policy, obs, bound: float):
        '''
        Used in training.
        Given a batch of obs data and the bound, return the perturbation
        @param obs, tensor, [batch, obs_dim]
        @return epsilon, tensor, [batch, obs_dim]
        '''
        batch = obs.shape[0]
        epsilon = torch.distributions.uniform.Uniform(-bound, bound).sample(
            [batch, self.obs_dim])
        return to_device(epsilon)

    def attack_at_eval(self, policy, obs, bound: float):
        '''
        Used in evaluation.
        Given a single obs data and the bound, return the perturbation
        @param obs, ndarray, [obs_dim]
        @return epsilon, ndarray, [obs_dim]
        '''
        obs = to_tensor(obs).reshape(1, -1)
        epsilon = self.attack_batch(policy, obs, bound)
        return to_ndarray(torch.squeeze(epsilon))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    adv_uniform = AdvUniform(2)

    obs = np.zeros(1000)

    eps = adv_uniform.attack_batch(None, obs, 0.005)

    eps = to_ndarray(eps)

    print(eps)

    plt.scatter(eps[:, 0], eps[:, 1])
    plt.show()
