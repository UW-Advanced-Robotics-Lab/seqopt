from abc import ABC
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
from stable_baselines3.common.policies import BaseModel, BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, create_mlp
import torch as th
from torch import nn

from seqopt.utils.sb3_utils import Schedule


class ValueCritic(BaseModel, ABC):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 lr_schedule: Schedule,
                 net_arch: Optional[List[int]] = None,
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None
                 ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        # For a value critic, the output "action" is a single value that is unbounded
        # We define the 'action' space accordingly
        action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,))

        super(ValueCritic, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        if net_arch is None:
            if features_extractor_class == FlattenExtractor:
                net_arch = [64, 64]
            else:
                net_arch = []

        self.net_arch = net_arch
        self.activation_fn = activation_fn

        self.features_extractor = features_extractor_class(self.observation_space,
                                                           **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim

        # Build the network
        self.v_net = None
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        # Build the feedfoward policy ending with no activation after the final output
        net = create_mlp(self.features_dim,
                         output_dim=1,
                         net_arch=self.net_arch,
                         activation_fn=self.activation_fn,
                         squash_output=False)

        # Create the network from the list of layers
        self.v_net = nn.Sequential(*net).to(self.device)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(),
                                              lr=lr_schedule(1),
                                              **self.optimizer_kwargs)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)
        return self.v_net(features)


class ContinuousCritic(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    It represents the action-state value function (Q-value function).
    Compared to A2C/PPO critics, this one represents the Q-value
    and takes the continuous action as input. It is concatenated with the state
    and then fed to the network which outputs a single value: Q(s, a).
    For more recent algorithms like SAC/TD3, multiple networks
    are created to give different estimates.
    By default, it creates two critic networks used to reduce overestimation
    thanks to clipped Q-learning (cf TD3 paper).
    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: Optional[gym.spaces.Space],
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if self.action_space is not None:
            action_dim = get_action_dim(self.action_space)
            self.has_action = True
        else:
            action_dim = 0
            self.has_action = False

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            q_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
            q_net = nn.Sequential(*q_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: Optional[th.Tensor] = None) -> Tuple[th.Tensor, ...]:
        assert actions is not None or not self.has_action, "Actions must be specified if policy has action space!"
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        if self.has_action:
            qvalue_input = th.cat([features, actions], dim=1)
        else:
            qvalue_input = features
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: Optional[th.Tensor] = None) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        assert actions is not None or not self.has_action, "Actions must be specified if policy has action space!"

        with th.no_grad():
            features = self.extract_features(obs)
        if self.has_action:
            return self.q_networks[0](th.cat([features, actions], dim=1))
        else:
            return self.q_networks[0](features)


class TerminatorPolicy(BasePolicy, ABC):
    def __init__(self,
                 observation_space: gym.spaces.Space,
                 lr_schedule: Schedule,
                 net_arch: Optional[List[int]] = None,
                 activation_fn: Type[nn.Module] = nn.Tanh,
                 squash_output: bool = False,
                 features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 normalize_images: bool = True,
                 optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 use_boltzmann = False
                 ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        # For a termination policy, the output "action" is a probability in the range [0,1]
        # We define the 'action' space accordingly
        action_space = gym.spaces.Box(low=0., high=1., shape=(1,))

        super(TerminatorPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )

        if net_arch is None:
            if features_extractor_class == FlattenExtractor:
                net_arch = [64, 64]
            else:
                net_arch = []

        self.net_arch = net_arch
        self.activation_fn = activation_fn

        self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim

        self.normalize_images = normalize_images

        # Create 2 outputs after the final layer that determine the "value" of terminating vs not terminating
        # We can use a softmax layer to calculate the probability of each and sample from it to determine if we
        # should terminate or not
        # The alternative if use_boltzmann = False, is to just use a sigmoid layer to return a single value for the
        # output that lies between [0.0, 1.0]. However, sigmoids can result in having vanishing gradients and
        # become hard to adapt to new data once they get into this regime
        self.use_boltzmann = use_boltzmann

        # Build the network
        self.net = []
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        # Build the feedfoward policy ending with a sigmoid layer
        net = []

        last_hidden_layer_size = self.features_dim
        for hidden_layer_idx in range(len(self.net_arch)):
            net.append(nn.Linear(last_hidden_layer_size, self.net_arch[hidden_layer_idx]))
            net.append(self.activation_fn())
            last_hidden_layer_size = self.net_arch[hidden_layer_idx]

        # Output
        if self.use_boltzmann:
            net.append(nn.Linear(last_hidden_layer_size, 2))
        else:
            # Add the final linear layer with an output of 1 element, followed by a sigmoid activation
            net.append(nn.Linear(last_hidden_layer_size, 1))
            net.append(nn.Sigmoid())

        # Create the network from the list of layers
        self.net = nn.Sequential(*net).to(self.device)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _predict(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        if self.use_boltzmann:
            beta_prob = th.nn.functional.softmax(self.net(obs), dim=1)[..., -1]
        else:
            beta_prob = self.net(obs)
        if deterministic:
            terminate = (beta_prob > 0.5)
        else:
            terminate = (np.random.uniform() < beta_prob)
        return terminate

    def forward(self,
                obs: th.Tensor,
                deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        if self.use_boltzmann:
            beta_prob = th.index_select(th.nn.functional.softmax(self.net(obs), dim=1),
                                        dim=-1,
                                        index=th.LongTensor([1]))
        else:
            beta_prob = self.net(obs)

        if deterministic:
            terminate = th.gt(beta_prob, 0.5)
        else:
            terminate = th.gt(beta_prob, np.random.uniform())

        # We don't allow beta prob to be too close to 0 or 1 (since the log_prob goes to infinity)
        beta_prob = th.where(beta_prob < 1e-6, 1e-6, beta_prob.double()).float()
        beta_prob = th.where(beta_prob > (1.0 - 1e-6), 1.0 - 1e-6, beta_prob.double()).float()

        # Calculate the log probability and entropy of the termination probability
        log_prob = th.log(beta_prob)

        return terminate, beta_prob, log_prob

    def evaluate(self,
                 obs: th.Tensor) -> [th.Tensor, th.Tensor, th.Tensor]:
        # Calculate the termination probability for the given state
        if self.use_boltzmann:
            beta_prob = th.nn.functional.softmax(self.net(obs), dim=1)[..., -1]
        else:
            beta_prob = self.net(obs)

        # Don't let the probability get too close to 0 or 1, otherwise its log will become unbounded
        beta_prob = th.where(beta_prob < 1e-6, 1e-6, beta_prob.double()).float()
        beta_prob = th.where(beta_prob > (1.0 - 1e-6), 1.0 - 1e-6, beta_prob.double()).float()

        # Calculate log probability and entropy
        log_prob = th.log(beta_prob)
        # entropy = th.where(beta_prob == 1.0,
        #                    0.0,
        #                    -(beta_prob * log_prob + (1 - beta_prob) * th.log(1 - beta_prob)).double()).float()
        entropy = -(beta_prob * log_prob + (1 - beta_prob) * th.log(1 - beta_prob))

        return beta_prob, log_prob, entropy
