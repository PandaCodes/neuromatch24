from tonic.torch import models, normalizers
import torch
import torch.nn as nn

from models.sensor import Sensor
from models.SwimmerActor import SwimmerActor
from models.SwimmerModule import SwimmerModule

def ppo_swimmer_model(
    n_joints=5,
    action_noise=0.1,
    critic_sizes=(64, 64),
    critic_activation=nn.Tanh,
    **swimmer_kwargs,
):
    return models.ActorCritic(
        actor=SwimmerActor(
            swimmer=SwimmerModule(
                n_joints=n_joints,
                **swimmer_kwargs,
                #include_turn_control=True,
            ),
            distribution=lambda x: torch.distributions.normal.Normal(x, action_noise),
        ),
        critic=models.Critic(
            encoder=models.ObservationEncoder(),
            torso=models.MLP(critic_sizes, critic_activation),
            head=models.ValueHead(),
        ),
        observation_normalizer=normalizers.MeanStd(),
    )


def d4pg_swimmer_model(
    n_joints=5,
    critic_sizes=(256, 256),
    critic_activation=nn.ReLU,
    **swimmer_kwargs,
):
    return models.ActorCriticWithTargets(
        actor=SwimmerActor(
            swimmer=SwimmerModule(n_joints=n_joints, **swimmer_kwargs),
        ),
        critic=models.Critic(
            encoder=models.ObservationActionEncoder(),
            torso=models.MLP(critic_sizes, critic_activation),
            # These values are for the control suite with 0.99 discount.
            head=models.DistributionalValueHead(-150.0, 150.0, 51),
        ),
        observation_normalizer=normalizers.MeanStd(),
    )


def ppo_controller_swimmer_model(
    n_joints=5,
    action_noise=0.1,
    critic_sizes=(64, 64),
    critic_activation=nn.Tanh,
    controller_args={},
    **swimmer_kwargs,
):
    _controller_args: dict = {
        "input_dim":18,
        "hidden_dim": 128,
        **controller_args,
    }
    return models.ActorCritic(
        actor=SwimmerActor(
            controller=Sensor(**_controller_args),
            swimmer=SwimmerModule(
                n_joints=n_joints,
                **swimmer_kwargs,
                include_speed_control=True,
                include_turn_control=True,
            ),
            distribution=lambda x: torch.distributions.normal.Normal(x, action_noise),
        ),
        critic=models.Critic(
            encoder=models.ObservationEncoder(),
            torso=models.MLP(critic_sizes, critic_activation),
            head=models.ValueHead(),
        ),
        observation_normalizer=normalizers.MeanStd(),
    )

def ppo_mlp_model(
    actor_sizes=(64, 64),
    actor_activation=torch.nn.Tanh,
    critic_sizes=(64, 64),
    critic_activation=torch.nn.Tanh,
):
    """
    Constructs an ActorCritic model with specified architectures for the actor and critic networks.

    Parameters:
    - actor_sizes (tuple): Sizes of the layers in the actor MLP.
    - actor_activation (torch activation): Activation function used in the actor MLP.
    - critic_sizes (tuple): Sizes of the layers in the critic MLP.
    - critic_activation (torch activation): Activation function used in the critic MLP.

    Returns:
    - models.ActorCritic: An ActorCritic model comprising an actor and a critic with MLP torsos,
      equipped with a Gaussian policy head for the actor and a value head for the critic,
      along with observation normalization.
    """

    return models.ActorCritic(
        actor=models.Actor(
            encoder=models.ObservationEncoder(),
            torso=models.MLP(actor_sizes, actor_activation),
            head=models.DetachedScaleGaussianPolicyHead(),
        ),
        critic=models.Critic(
            encoder=models.ObservationEncoder(),
            torso=models.MLP(critic_sizes, critic_activation),
            head=models.ValueHead(),
        ),
        observation_normalizer=normalizers.MeanStd(),
    )