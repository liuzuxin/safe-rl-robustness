from abc import ABC, abstractclassmethod
import numpy as np
import torch


class Adv(ABC):
    def __init__(self, **kwarg) -> None:
        super().__init__()

    @abstractclassmethod
    def attack_at_eval(self, obs: np.ndarray, bound: float):
        '''
        Used in evaluation.
        Given a single obs data and the bound, return the perturbation
        @param obs, ndarray, [obs_dim]
        @return epsilon, ndarray, [obs_dim]
        '''
        raise NotImplementedError

    @abstractclassmethod
    def attack_batch(self, obs: torch.Tensor, bound: float) -> torch.Tensor:
        '''
        Used in training.
        Given a batch of obs data and the bound, return the perturbation
        @param obs, tensor, [batch, obs_dim]
        @return epsilon, tensor, [batch, obs_dim]
        '''
        raise NotImplementedError