import time

import numpy as np
import torch
from rsrl.policy.adversary.adv_mad import AdvMad
from rsrl.util.torch_util import to_ndarray, to_tensor
from torch.distributions.kl import kl_divergence


class AdvAmad(AdvMad):

    def __init__(self, obs_dim, attack_fraction, **kwarg) -> None:
        super().__init__(obs_dim, **kwarg)
        self.id = 'AdvAmad'
        self.attack_fraction = attack_fraction

    def set_thres(self, qc_list):
        qc_list.sort()
        qc_num = len(qc_list)
        qc_thres_index = int((1 - self.attack_fraction) * qc_num)
        self.qc_thres = qc_list[qc_thres_index]

    def attack_at_eval(self, policy, obs: np.ndarray, bound: float):
        '''
        Used in evaluation.
        Given a single obs data and the bound, return the perturbation
        @param obs, ndarray, [obs_dim]
        @return epsilon, ndarray, [obs_dim]
        '''
        obs = to_tensor(obs).reshape(1, -1)
        mean, A = policy.actor.forward(obs)
        qc, _ = policy.critic_forward(policy.qc, obs, mean)

        if torch.squeeze(qc).item() < self.qc_thres:
            return 0

        epsilon = self.attack_batch(policy, obs, bound)
        return to_ndarray(torch.squeeze(epsilon))


class AdvAmadPPO(AdvAmad):

    def __init__(self, obs_dim, **kwarg) -> None:
        super().__init__(obs_dim, **kwarg)

    def attack_batch(self, policy, obs: torch.Tensor, bound: float):
        '''
        Used in training.
        Given a batch of obs data and the bound, return the perturbation
        @param obs, tensor, [batch, obs_dim]
        @return epsilon, tensor, [batch, obs_dim]
        '''
        batch_size = obs.shape[0]
        if self.batch is None or self.batch != batch_size:
            self.setup_batch_size(batch_size)
        assert obs.shape[1] == self.obs_dim, "obs shape 1: %d, self.obs_dim: %d" % (
            obs.shape[1], self.obs_dim)

        with torch.no_grad():
            pi, a, _ = policy.actor_forward(obs)  # (K,)

        # init the epsilon data
        # self.epsilon.data = to_tensor(np.zeros((self.batch, self.obs_dim)))

        # early stop:
        self.epsilon_prev = self.epsilon.data.detach().clone()
        loss_prev = 0

        for iteration in range(self.max_iterations):
            obs_adv = self.epsilon + obs
            pi_adv, _, _ = policy.actor_forward(obs_adv)
            self.optimizer.zero_grad()

            loss = -torch.mean(kl_divergence(pi, pi_adv))

            loss_np = loss.item()

            loss.backward()
            # clip_grad_norm_(self.epsilon, bound)
            self.optimizer.step()
            # clip the perturbation within the bound
            with torch.no_grad():
                self.epsilon.clamp_(-bound, bound)

            # early stop
            epsilon_diff = self.epsilon.data.detach() - self.epsilon_prev
            epsilon_diff_max = torch.abs(epsilon_diff).max().item()
            self.epsilon_prev = self.epsilon.data.detach()

            if np.abs(loss_np -
                      loss_prev) < self.tol and epsilon_diff_max < self.eps_tol:
                break
            loss_prev = loss_np
        self.store_info(loss=loss_np, iteration=iteration, loss_prev=loss_prev)
        return self.epsilon_prev

    def attack_at_eval(self, policy, obs: np.ndarray, bound: float):
        '''
        Used in evaluation.
        Given a single obs data and the bound, return the perturbation
        @param obs, ndarray, [obs_dim]
        @return epsilon, ndarray, [obs_dim]
        '''
        obs = to_tensor(obs).reshape(1, -1)
        pi, a, _ = policy.actor.forward(obs, deterministic=True)
        qc, _ = policy.q_critic_forward(policy.qc2, obs, a)

        if torch.squeeze(qc).item() < self.qc_thres:
            return 0

        epsilon = self.attack_batch(policy, obs, bound)
        return to_ndarray(torch.squeeze(epsilon))
