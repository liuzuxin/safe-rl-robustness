from abc import ABC, abstractclassmethod
from copy import deepcopy

import gym
import numpy as np
import torch
import torch.nn as nn
from cpprb import ReplayBuffer
from rsrl.policy.adversary.adv_amad import AdvAmadPPO
from rsrl.policy.adversary.adv_base import Adv
from rsrl.policy.adversary.adv_critic import AdvCriticPPO
from rsrl.policy.adversary.adv_mad import AdvMadPPO
from rsrl.policy.adversary.adv_random import AdvUniform
from rsrl.policy.buffer import OnPolicyBuffer
from rsrl.policy.model.mlp_ac import (EnsembleQCritic, MLPCategoricalActor,
                                      MLPGaussianActor, mlp)
from rsrl.util.logger import EpochLogger
from rsrl.util.torch_util import to_device, to_ndarray, to_tensor
from torch.optim import Adam
from tqdm import tqdm


class OnPolicyBase(ABC):
    attacker_cls = dict(uniform=AdvUniform,
                        mad=AdvMadPPO,
                        amad=AdvAmadPPO,
                        max_reward=AdvCriticPPO,
                        min_reward=AdvCriticPPO,
                        max_cost=AdvCriticPPO)

    def __init__(self,
                 env: gym.Env,
                 logger: EpochLogger,
                 env_cfg: dict,
                 adv_cfg: dict,
                 actor_lr=0.0003,
                 critic_lr=0.001,
                 gamma=0.99,
                 polyak=0.995,
                 batch_size=300,
                 train_critic_iters=80,
                 hidden_sizes=[128, 128],
                 buffer_size=100000,
                 interact_steps=15000,
                 lam=0.97,
                 decay_epoch=150,
                 start_epoch=20,
                 episode_rerun_num=40,
                 rs_mode="vanilla",
                 **kwarg) -> None:

        super().__init__()
        self.env = env
        self.logger = logger
        self.env_cfg = env_cfg
        self.gamma = gamma
        self.polyak = polyak
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.hidden_sizes = hidden_sizes
        self.train_critic_iters = train_critic_iters
        self.episode_rerun_num = episode_rerun_num

        self.cost_limit = self.env_cfg["cost_limit"]
        self.cost_normalizer = self.env_cfg["cost_normalizer"]
        self.interact_steps = interact_steps
        self.lam = lam

        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        # Action limit for normalization: critically, assumes all dimensions share the same bound!
        self.act_lim = self.env.action_space.high[0]

        self.timeout_steps = self.env_cfg["timeout_steps"]

        self._init_actor()
        self._init_critic()
        self._init_cost_critic()
        self._init_buffer()

        #############################################################################
        ############################ for attacker usage #############################
        #############################################################################
        # the Q critics are used for attacking purpose
        self._init_q_critic_for_attackers()
        self.adv_cfg = adv_cfg
        self.mode = rs_mode

        if self.mode in self.attacker_cls:
            self.apply_adv_in_training = True
            cfg = self.adv_cfg[self.mode + "_cfg"]
            adv_cls = self.attacker_cls[self.mode]
            self.adversary = adv_cls(self.obs_dim, **cfg)
            if self.mode == "amad":
                self.adversary.set_thres([0])
        else:
            self.apply_adv_in_training = False

        self.noise_scale = self.adv_cfg["noise_scale"]
        self.noise_scale_schedule = 0
        self.decay_epoch = decay_epoch
        self.start_epoch = start_epoch
        self.decay_func = lambda x: self.noise_scale - self.noise_scale * np.exp(
            -5. * x / self.decay_epoch)

    def _init_actor(self, actor_state_dict=None, actor_optimizer_state_dict=None):
        if isinstance(self.env.action_space, gym.spaces.Box):
            actor = MLPGaussianActor(self.obs_dim, self.act_dim, -self.act_lim,
                                     self.act_lim, self.hidden_sizes, nn.ReLU)
        elif isinstance(self.env.action_space, gym.spaces.Discrete):
            actor = MLPCategoricalActor(self.obs_dim, self.env.action_space.n,
                                        self.hidden_sizes, nn.ReLU)
        if actor_state_dict is not None:
            actor.load_state_dict(actor_state_dict)
        self.actor = to_device(actor)

        # Set up optimizers for policy and value function
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
        if actor_optimizer_state_dict is not None:
            self.actor_optimizer.load_state_dict(actor_optimizer_state_dict)

    def _init_critic(self, critic_state_dict=None, critic_optimzer_state_dict=None):
        critic = mlp([self.obs_dim] + list(self.hidden_sizes) + [1], nn.ReLU)
        if critic_state_dict is not None:
            critic.load_state_dict(critic_state_dict)
        self.critic = to_device(critic)
        self.critic_targ = deepcopy(self.critic)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.critic_targ.parameters():
            p.requires_grad = False
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        if critic_optimzer_state_dict is not None:
            self.critic_optimizer.load_state_dict(critic_optimzer_state_dict)

    def _init_cost_critic(self, qc_state_dict=None, qc_optimizer_state_dict=None):
        # init safety critic
        qc = mlp([self.obs_dim] + list(self.hidden_sizes) + [1], nn.ReLU)
        if qc_state_dict is not None:
            qc.load_state_dict(qc_state_dict)
        self.qc = to_device(qc)
        self.qc_targ = deepcopy(self.qc)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.qc_targ.parameters():
            p.requires_grad = False

        self.qc_optimizer = Adam(self.qc.parameters(), lr=self.critic_lr)
        if qc_optimizer_state_dict is not None:
            self.qc_optimizer.load_state_dict(qc_optimizer_state_dict)

    def _init_q_critic_for_attackers(self,
                                     critic2_state_dict=None,
                                     critic2_optimizer_state_dict=None,
                                     qc2_state_dict=None,
                                     qc2_optimizer_state_dict=None):
        # off-policy Q network; for attacker usage
        critic2 = EnsembleQCritic(self.obs_dim,
                                  self.act_dim,
                                  self.hidden_sizes,
                                  nn.ReLU,
                                  num_q=2)
        qc2 = EnsembleQCritic(self.obs_dim,
                              self.act_dim,
                              self.hidden_sizes,
                              nn.ReLU,
                              num_q=1)
        if critic2_state_dict is not None:
            critic2.load_state_dict(critic2_state_dict)
        if qc2_state_dict is not None:
            qc2.load_state_dict(qc2_state_dict)

        self.critic2 = to_device(critic2)
        self.critic2_targ = deepcopy(self.critic2)

        self.qc2 = to_device(qc2)
        self.qc2_targ = deepcopy(self.qc2)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.critic2_targ.parameters():
            p.requires_grad = False
        for p in self.qc2_targ.parameters():
            p.requires_grad = False

        # Set up optimizers for safety critic
        self.qc2_optimizer = Adam(self.qc2.parameters(), lr=self.critic_lr)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=self.critic_lr)
        if critic2_optimizer_state_dict is not None:
            self.critic2_optimizer.load_state_dict(critic2_optimizer_state_dict)
        if qc2_optimizer_state_dict is not None:
            self.qc2_optimizer.load_state_dict(qc2_optimizer_state_dict)

    def _init_buffer(self):
        env_dict = {
            'act': dict(dtype=np.float32, shape=self.act_dim),
            'done': dict(dtype=np.float32),
            "obs": dict(dtype=np.float32, shape=self.obs_dim),
            "obs2": dict(dtype=np.float32, shape=self.obs_dim),
            'rew': dict(dtype=np.float32),
            'cost': dict(dtype=np.float32)
        }
        self.q_critic_buffer = ReplayBuffer(self.buffer_size, env_dict)
        self.buffer = OnPolicyBuffer(self.obs_dim, self.act_dim, self.interact_steps + 1,
                                     self.gamma, self.lam)

    def train_one_epoch(self, warmup=False, verbose=False):
        '''Train one epoch by collecting the data'''
        self.logger.store(CostLimit=self.cost_limit, tab="worker")
        epoch_steps = 0

        if warmup and verbose:
            print("*** Warming up begin ***")

        steps = self.collect_data(warmup=warmup)
        epoch_steps += steps

        training_range = range(self.episode_rerun_num)
        if verbose:
            training_range = tqdm(training_range,
                                  desc='Training steps: ',
                                  position=1,
                                  leave=False)
        # training the Q critics for attacking purpose
        for i in training_range:
            off_policy_data = self.get_sample_for_q_critic()
            self.train_Q_network(off_policy_data)

        data = self.get_sample()
        self.learn_on_batch(data)
        return epoch_steps

    def learn_on_batch(self, data: dict):
        self._update_actor(data)
        LossV, DeltaLossV = self._update_value_critic(self.critic, data["obs"],
                                                      data["ret"], self.critic_optimizer)
        LossVC, DeltaLossVQC = self._update_value_critic(self.qc, data["obs"],
                                                         data["cost_ret"],
                                                         self.qc_optimizer)
        # Log safety critic update info
        self.logger.store(LossVC=LossVC, LossV=LossV)

    @abstractclassmethod
    def _update_actor(self, data):
        '''Train the policy model with a batch of data'''
        raise NotImplementedError

    def get_sample(self):
        data = self.buffer.get()
        self.buffer.clear()
        data["ep_cost"] = to_tensor(np.mean(self.cost_list))
        return data

    def get_sample_for_q_critic(self):
        data = to_tensor(self.q_critic_buffer.sample(self.batch_size))
        data["rew"] = torch.squeeze(data["rew"])
        data["done"] = torch.squeeze(data["done"])
        if "cost" in data:
            data["cost"] = torch.squeeze(data["cost"])
        return data

    def get_cost_critic_value(self, obs):
        obs = to_tensor(obs).reshape(1, -1)
        with torch.no_grad():
            v = self.value_critic_forward(self.qc, obs)
        return np.squeeze(to_ndarray(v))

    def clear_buffer(self):
        self.q_critic_buffer.clear()

    def actor_forward(self, obs, act=None, deterministic=False):
        r''' 
        Return action distribution and action log prob [optional].
        @param obs, [tensor], (batch, obs_dim)
        @param act, [tensor], (batch, act_dim). If None, log prob is None
        @return pi, [torch distribution], (batch,)
        @return a, [torch distribution], (batch, act_dim)
        @return logp, [tensor], (batch,)
        '''
        pi, a, logp = self.actor(obs, act, deterministic)
        return pi, a, logp

    def q_critic_forward(self, critic, obs, act):
        # return the minimum q values and the list of all q_values
        return critic.predict(obs, act)

    def value_critic_forward(self, critic, obs):
        # Critical to ensure value has the right shape.
        # Without this, the training stability will be greatly affected!
        # For instance, shape [3] - shape[3,1] = shape [3, 3] instead of shape [3]
        return torch.squeeze(critic(obs), -1)

    def act(self, obs, deterministic=False, with_logprob=False):
        '''
        Given a single obs, return the action, value, logp.
        This API is used to interact with the env.

        @param obs, 1d ndarray
        @param eval, evaluation mode
        @return act, value, logp, 1d ndarray
        '''
        obs = to_tensor(obs).reshape(1, -1)
        with torch.no_grad():
            _, a, logp_a = self.actor_forward(obs, deterministic=deterministic)
            v = self.value_critic_forward(self.critic, obs)
        # squeeze them to the right shape
        a, v, logp_a = np.squeeze(to_ndarray(a),
                                  axis=0), np.squeeze(to_ndarray(v)), np.squeeze(
                                      to_ndarray(logp_a))
        return a, v, logp_a

    def _post_epoch_process(self, epoch):
        self.epoch = epoch
        if self.epoch < self.start_epoch:
            self.noise_scale_schedule = 0
        else:
            self.noise_scale_schedule = self.decay_func(epoch - self.start_epoch)

    def _pre_epoch_process(self, epoch, **kwarg):
        pass

    def _process_obs(self, obs):
        epsilon = 0
        if self.apply_adv_in_training and self.noise_scale_schedule > 0:
            epsilon = self.adversary.attack_at_eval(self, obs, self.noise_scale_schedule)
        self.logger.store(tab="worker", noise=self.noise_scale_schedule)
        return obs + epsilon

    def collect_data(self, warmup=False):
        '''Interact with the environment to collect data'''
        self.cost_list = []
        obs, ep_reward, ep_len, ep_cost = self.env.reset(), 0, 0, 0
        # epoch_num = 0
        for i in range(self.interact_steps):
            obs = self._process_obs(obs)
            action, value, log_prob = self.act(obs)
            obs_next, reward, done, info = self.env.step(action)

            cost_value = self.get_cost_critic_value(obs)

            if done and "TimeLimit.truncated" in info:
                done = False
                timeout_env = True
            else:
                timeout_env = False

            cost = info["cost"] if "cost" in info else 0
            self.buffer.store(obs, np.squeeze(action), reward, value, log_prob, done,
                              cost, cost_value)
            self.q_critic_buffer.add(obs=obs,
                                     act=np.squeeze(action),
                                     rew=reward,
                                     obs2=obs_next,
                                     done=done,
                                     cost=cost / self.cost_normalizer)
            self.logger.store(VVals=value, CostVVals=cost_value, tab="learner")
            ep_reward += reward
            ep_cost += cost
            ep_len += 1
            obs = obs_next

            timeout = ep_len == self.timeout_steps - 1 or i == self.interact_steps - 1 or timeout_env and not done
            terminal = done or timeout
            if terminal:
                if timeout:
                    _, value, _ = self.act(obs)
                    cost_value = self.get_cost_critic_value(obs)
                else:
                    value, cost_value = 0, 0
                self.buffer.finish_path(value, cost_value)
                if i < self.interact_steps - 1:
                    self.logger.store(EpRet=ep_reward,
                                      EpLen=ep_len,
                                      EpCost=ep_cost,
                                      tab="worker")

                obs = self.env.reset()
                self.cost_list.append(ep_cost)
                ep_reward, ep_cost, ep_len = 0, 0, 0

        return self.interact_steps

    def train_Q_network(self, data: dict):
        self._update_q_critic(data)
        self._update_qc_critic(data)
        self._polyak_update_target(self.critic2, self.critic2_targ)
        self._polyak_update_target(self.qc2, self.qc2_targ)

    def get_risk_estimation(self, obs):
        '''Given an obs array (obs_dim), output a risk (qc) value, and the action'''
        obs = to_tensor(obs).reshape(1, -1)
        with torch.no_grad():
            _, a, logp_a = self.actor_forward(obs, deterministic=True)
            # vc = self.value_critic_forward(self.qc, obs)
            qc, _ = self.q_critic_forward(self.qc2, obs, a)
        return torch.squeeze(qc).item(), np.squeeze(to_ndarray(a), axis=0)

    def _update_value_critic(self, critic, obs, ret, critic_optimizer):
        '''Update the value critic network'''
        obs, ret = to_tensor(obs), to_tensor(ret)

        def critic_loss():
            ret_pred = self.value_critic_forward(critic, obs)
            return ((ret_pred - ret)**2).mean()

        loss_old = critic_loss().item()

        # Value function learning
        for i in range(self.train_critic_iters):
            critic_optimizer.zero_grad()
            loss_critic = critic_loss()
            loss_critic.backward()
            critic_optimizer.step()

        return loss_old, to_ndarray(loss_critic) - loss_old

    def _update_q_critic(self, data):
        '''Update the critic network'''

        def critic_loss():
            obs, act, reward, done = to_tensor(data['obs']), to_tensor(
                data['act']), to_tensor(data['rew']), to_tensor(data['done'])
            obs_next = to_tensor(data['obs2'])
            _, q_list = self.q_critic_forward(self.critic2, obs, act)
            # Bellman backup for Q functions
            with torch.no_grad():
                # Target actions come from *current* policy
                pi_dist, act_next, _ = self.actor_forward(obs_next, deterministic=False)
                # Target Q-values
                q_pi_targ, _ = self.q_critic_forward(self.critic2_targ, obs_next,
                                                     act_next)
                backup = reward + self.gamma * (1 - done) * q_pi_targ
            # MSE loss against Bellman backup
            loss_q = self.critic2.loss(backup, q_list)
            return loss_q

        self.critic2_optimizer.zero_grad()
        loss_critic = critic_loss()
        loss_critic.backward()
        self.critic2_optimizer.step()
        self.logger.store(LossQ=loss_critic.item())

    def _update_qc_critic(self, data):
        '''Update the qc network'''

        def critic_loss():
            obs, act, reward, done = to_tensor(data['obs']), to_tensor(
                data['act']), to_tensor(data['cost']), to_tensor(data['done'])
            obs_next = to_tensor(data['obs2'])
            _, q_list = self.q_critic_forward(self.qc2, obs, act)
            # Bellman backup for Q functions
            with torch.no_grad():
                # Target actions come from *current* policy
                pi_dist, act_next, _ = self.actor_forward(obs_next, deterministic=False)
                # Target Q-values
                q_pi_targ, _ = self.q_critic_forward(self.qc2_targ, obs_next, act_next)
                backup = reward + self.gamma * (1 - done) * q_pi_targ
            # MSE loss against Bellman backup
            loss_q = self.qc2.loss(backup, q_list)
            return loss_q

        self.qc2_optimizer.zero_grad()
        loss_qc = critic_loss()
        loss_qc.backward()
        self.qc2_optimizer.step()
        self.logger.store(LossQC=loss_qc.item())

    def save_model(self):
        actor, actor_optimizer = self.actor.state_dict(
        ), self.actor_optimizer.state_dict()
        critic, critic_optimizer = self.critic.state_dict(
        ), self.critic_optimizer.state_dict()
        critic2, critic2_optimizer = self.critic2.state_dict(
        ), self.critic2_optimizer.state_dict()
        qc2, qc2_optimizer = self.qc2.state_dict(), self.qc2_optimizer.state_dict()
        model = {
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "critic": critic,
            "critic_optimizer": critic_optimizer,
            "critic2": critic2,
            "critic2_optimizer": critic2_optimizer,
            "qc2": qc2,
            "qc2_optimizer": qc2_optimizer
        }

        qc, qc_optimizer = self.qc.state_dict(), self.qc_optimizer.state_dict()
        model["qc"] = qc
        model["qc_optimizer"] = qc_optimizer
        self.logger.setup_pytorch_saver(model)

    def load_model(self, path):
        model = torch.load(path)
        assert type(model) is dict, "Unknown model type: {%s}" % {type(model)}
        actor, actor_optimizer = model["actor"], model["actor_optimizer"]
        critic, critic_optimizer = model["critic"], model["critic_optimizer"]
        qc, qc_optimizer = model["qc"], model["qc_optimizer"]
        critic2, critic2_optimizer = model["critic2"], model["critic2_optimizer"]
        qc2, qc2_optimizer = model["qc2"], model["qc2_optimizer"]
        self._init_actor(actor, actor_optimizer)
        self._init_critic(critic, critic_optimizer)
        self._init_cost_critic(qc, qc_optimizer)
        self._init_q_critic_for_attackers(critic2, critic2_optimizer, qc2, qc2_optimizer)

    # def save_model(self):
    #     '''
    #     Save the model to dir
    #     '''
    #     actor, critic = self.actor.state_dict(), self.critic.state_dict()
    #     actor_optimzer, critic_optimzer = self.actor_optimizer.state_dict(), \
    #                                       self.critic_optimizer.state_dict()
    #     model = {
    #         "actor": actor,
    #         "critic": critic,
    #         "actor_optimzer": actor_optimzer,
    #         "critic_optimzer": critic_optimzer
    #     }
    #     if self.safe_rl:
    #         qc = self.qc.state_dict()
    #         qc_optimizer = self.qc_optimizer.state_dict()
    #         model["qc"] = qc
    #         model["qc_optimizer"] = qc_optimizer
    #     self.logger.setup_pytorch_saver(model)

    # def load_model(self, path):
    #     '''
    #     Load the model from dir
    #     '''
    #     model = torch.load(path)
    #     actor, actor_optimizer = model["actor"], model["actor_optimzer"]
    #     critic, critic_optimizer = model["critic"], model["critic_optimzer"]
    #     self._init_actor(actor, actor_optimizer)
    #     self._init_critic(critic, critic_optimizer)
    #     qc, qc_optimizer = model["qc"], model["qc_optimizer"]
    #     self._init_cost_critic(qc, qc_optimizer)

    @torch.no_grad()
    def _polyak_update_target(self, model, model_targ):
        '''Update target networks by polyak averaging.'''
        for p, p_targ in zip(model.parameters(), model_targ.parameters()):
            p_targ.data.mul_(self.polyak)
            p_targ.data.add_((1 - self.polyak) * p.data)
