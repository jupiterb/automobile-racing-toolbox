from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
import tensorflow as tf
from gym import spaces

from racing_toolbox.trainer.config import ModelConfig


class KerasMLP(TFModelV2):
    """Custom model for policy gradient algorithms."""

    __activations_mapping = {
        "relu": tf.nn.relu,
        "tanh": tf.nn.tanh,
        "sigmoid": tf.nn.sigmoid,
    }

    def __init__(
        self,
        obs_space: spaces.Box,
        action_space: spaces.Discrete,
        model_config: ModelConfig,
        name: str,
    ):
        assert len(obs_space.shape) == 1, "Invalid observation shape for MLP net"
        num_outputs = action_space.n

        super(KerasMLP, self).__init__(
            obs_space, action_space, num_outputs, model_config.dict(), name
        )

        hidden_sizes = model_config.hiddens
        if isinstance(activations := model_config.activations, list):
            hidden_activations = activations
        else:
            hidden_activations = [activations] * len(hidden_sizes)

        self.inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")

        hidden = self.inputs
        for size_, activation in zip(hidden_sizes, hidden_activations):
            activation_fun = self.__activations_mapping[activation]
            hidden = tf.keras.layers.Dense(
                size_,
                activation=activation_fun,
                kernel_initializer=normc_initializer(0.02),
            )(hidden)

        self.outputs = tf.keras.layers.Dense(
            num_outputs,
            activation=None,
            kernel_initializer=normc_initializer(0.02),
        )(hidden)

        self._base_model = tf.keras.Model(self.inputs, self.outputs)

    @property
    def model(self) -> tf.keras.Model:
        return self._base_model

    def forward(self, input_dict, state, seq_lens):
        model_out = self.model(input_dict["obs"])
        return model_out, state
