import numpy as np
import torch
from rsrl.policy.adversary.adv_base import Adv
from rsrl.util.torch_util import to_ndarray, to_tensor
from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence
from torch.optim import Adam


class AdvCritic(Adv):

    def __init__(self,
                 obs_dim,
                 batch=None,
                 lr_adv=0.1,
                 max_iterations=20,
                 tol=1e-4,
                 eps_tol=1e-4,
                 reward_weight=0.1,
                 cost_weight=0.9,
                 attack_fraction=1,
                 **kwarg) -> None:
        super().__init__(**kwarg)
        if reward_weight == 0:
            self.id = 'AdvMaxCost'
        elif cost_weight == 0:
            self.id = 'AdvMaxReward'
        else:
            self.id = 'Adv' + 'Cost' + str(cost_weight) + 'Reward' + str(reward_weight)
        if reward_weight < 0:
            self.id = 'AdvMinReward'

        self.obs_dim = obs_dim
        self.max_iterations = max_iterations
        self.tol = tol
        self.eps_tol = eps_tol
        self.reward_weight = reward_weight
        self.cost_weight = cost_weight
        self.lr_adv = lr_adv
        self.batch = batch
        self.attack_fraction = attack_fraction
        self.qc_thres = -np.inf

        if self.batch is not None:
            self.setup_batch_size(self.batch)

        self.info = {}

    def set_thres(self, qc_list):
        qc_list.sort()
        qc_num = len(qc_list)
        qc_thres_index = int((1 - self.attack_fraction) * qc_num)
        self.qc_thres = qc_list[qc_thres_index]

    def store_info(self, **kwarg):
        self.info = locals()["kwarg"]

    def get_info(self):
        return self.info

    def setup_batch_size(self, batch):
        self.batch = batch
        self.epsilon = torch.nn.Parameter(to_tensor(np.zeros(
            (self.batch, self.obs_dim))),
                                          requires_grad=True)
        self.optimizer = Adam([self.epsilon], lr=self.lr_adv)

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

        # init the epsilon data
        # self.epsilon.data = to_tensor(np.zeros((self.batch, self.obs_dim)))

        # early stop:
        self.epsilon_prev = self.epsilon.data.detach().clone()
        loss_prev = 0

        for iteration in range(self.max_iterations):
            obs_adv = self.epsilon + obs
            mean, A = policy.actor.forward(obs_adv)  # (batch, action space)
            # mean, A = policy.actor.forward(obs_adv,
            #                                deterministic=True)  # (batch, action space)
            qr, _ = policy.critic_forward(policy.critic, obs, mean)

            qc, _ = policy.critic_forward(policy.qc, obs, mean)

            self.optimizer.zero_grad()

            loss = torch.mean(-self.reward_weight * qr - self.cost_weight * qc)

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

        with torch.no_grad():
            b_mean, b_A = policy.actor.forward(obs)  # (K,)
            b = MultivariateNormal(b_mean, scale_tril=b_A)  # (K,)
            obs_adv = self.epsilon + obs
            mean, A = policy.actor.forward(obs_adv)
            b_adv = MultivariateNormal(loc=mean, scale_tril=A)  # (K,)
            kl = -torch.mean(kl_divergence(b, b_adv))

        self.store_info(loss=kl.item(), iteration=iteration, loss_prev=loss_np)
        return self.epsilon

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


class AdvCriticPPO(AdvCritic):

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

        # init the epsilon data
        # self.epsilon.data = to_tensor(np.zeros((self.batch, self.obs_dim)))

        # early stop:
        self.epsilon_prev = self.epsilon.data.detach().clone()
        loss_prev = 0

        for iteration in range(self.max_iterations):
            obs_adv = self.epsilon + obs
            pi, a, _ = policy.actor_forward(obs_adv,
                                            deterministic=True)  # (batch, action space)
            qr, _ = policy.q_critic_forward(policy.critic2, obs, a)

            qc, _ = policy.q_critic_forward(policy.qc2, obs, a)

            self.optimizer.zero_grad()

            loss = torch.mean(-self.reward_weight * qr - self.cost_weight * qc)

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

        with torch.no_grad():
            pi, a, _ = policy.actor_forward(obs)  # (K,)
            obs_adv = self.epsilon + obs
            pi_adv, a_adv, _ = policy.actor_forward(obs_adv)
            kl = -torch.mean(kl_divergence(pi, pi_adv))

        self.store_info(loss=kl.item(), iteration=iteration, loss_prev=loss_np)
        return self.epsilon

    def attack_at_eval(self, policy, obs: np.ndarray, bound: float):
        obs = to_tensor(obs).reshape(1, -1)
        pi, a, _ = policy.actor.forward(obs, deterministic=True)
        qc, _ = policy.q_critic_forward(policy.qc2, obs, a)

        if torch.squeeze(qc).item() < self.qc_thres:
            return 0

        epsilon = self.attack_batch(policy, obs, bound)
        return to_ndarray(torch.squeeze(epsilon))
