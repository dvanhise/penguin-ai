import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Concatenate, Reshape, Input, BatchNormalization, Activation, Dropout
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop

from episode import Episode

from itertools import chain

# MonteCarlo tree search with policy predictions at each state of probability of game win
#  Goes x moves deep in tree search trying to maximize policy prediction (+ exploration)
#  After game completes, policy at each chosen move is updated
LEARNING_RATE = .0001
ENTROPY_RATE = np.float32(.001)
GRADIENT_CLIP_MAX = 200.


class PenguinAgent:
    def __init__(self):
        self.model = self.build_model()
        self.opt = RMSprop(lr=LEARNING_RATE)
        self.recorder = [Episode()]

    def build_input(self, state):
        map_input = np.zeros((5, 11, 8), dtype=np.float32)
        map_input[0] = state['fish']
        map_input[1] = state['penguins']
        map_input[2] = np.full((11, 8), state['score'][0])
        map_input[3] = np.full((11, 8), state['score'][1])
        map_input[4] = np.full((11, 8), np.float32(state['phase']))

        map_input = np.moveaxis(map_input, -1, 0)  # Change to channels last
        map_input = np.reshape(map_input, (1, 11, 8, 5))
        return map_input

    def step(self, state, player, training=True):
        map_input = self.build_input(state)
        policy, value = self.model([map_input])
        policy = np.squeeze(policy)

        target = None
        destination = None
        mask = np.zeros((11, 8, 2), dtype=np.float32)
        if state['phase'] == 0:
            choices = np.zeros(len(state['placements']))
            for ndx, tile in enumerate(state['placements']):
                mask[tile[0]][tile[1]][0] = 1.0
                choices[ndx] = policy[tile[0]][tile[1]][0]

            choices = K.exp(choices) / (K.sum(K.exp(choices)))
            if training:
                target_ndx = np.random.choice(len(state['placements']), p=choices)
            else:
                target_ndx = np.argmax(choices)
            target = state['placements'][target_ndx]
            destination = None
        elif state['phase'] == 1:
            # TODO: MCTS

            choices = np.zeros(len(state['moves'].keys()))
            options = list(state['moves'].keys())
            for ndx, tile in enumerate(options):
                mask[tile[0]][tile[1]][0] = 1.0
                choices[ndx] = policy[tile[0]][tile[1]][0]

            choices = K.exp(choices) / (K.sum(K.exp(choices)))
            if training:
                target_ndx = np.random.choice(len(options), p=choices)
            else:
                target_ndx = np.argmax(choices)
            target = options[target_ndx]

            choices = np.zeros(len(state['moves'][target]))
            for ndx, tile in enumerate(state['moves'][target]):
                mask[tile[0]][tile[1]][1] = 1.0
                choices[ndx] = policy[tile[0]][tile[1]][1]

            choices = K.exp(choices) / (K.sum(K.exp(choices)))
            if training:
                destination_ndx = np.random.choice(len(state['moves'][target]), p=choices)
            else:
                destination_ndx = np.argmax(choices)
            destination = state['moves'][target][destination_ndx]

        self.recorder[-1].save_step(map_input, value, policy, mask, player, target, destination)

        return target, destination

    def step_end(self, rewards):
        self.recorder[-1].set_rewards(0, rewards[0])
        self.recorder[-1].set_rewards(1, rewards[1])
        self.recorder.append(Episode())

    def train(self):
        loss = np.array([0., 0., 0.])
        loss += self._train(
            np.concatenate([ep.map_input[:ep.current_step] for ep in self.recorder]),
            np.concatenate([ep.reward[:ep.current_step] for ep in self.recorder]),
            np.concatenate([ep.policy_mask[:ep.current_step] for ep in self.recorder]),
            np.concatenate([ep.policy_one_hot[:ep.current_step] for ep in self.recorder])
        )
        self.recorder = [Episode()]  # Clear recorder after training
        return loss

    def _train(self, map_input, reward, policy_mask, policy_one_hot):
        _entropy = _policy_loss = _value_loss = 0.

        policy_mask = policy_mask.astype('float32')
        with tf.GradientTape() as tape:
            policy, value = self.model(map_input)
            value = K.squeeze(value, axis=1)
            policy = K.exp(policy) / (K.sum(K.exp(policy)))

            value_loss = .5 * K.square(reward - value)
            # Should I use policy * policy_mask here?
            entropy = -K.sum(policy * K.log(policy + 1e-10), axis=[1, 2, 3])

            log_prob = K.log(K.sum(policy * policy_one_hot, axis=[1, 2, 3]) + 1e-10)
            advantage = reward - K.stop_gradient(value)

            policy_loss = -log_prob * advantage - entropy * ENTROPY_RATE

            total_loss = policy_loss + value_loss

            _entropy = K.mean(entropy)
            _policy_loss = K.mean(K.abs(policy_loss))
            _value_loss = K.mean(value_loss)

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, GRADIENT_CLIP_MAX)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))

        return [float(_value_loss), float(_policy_loss), float(_entropy)]

    def build_model(self):
        K.set_floatx('float32')
        map_input = Input(shape=(11, 8, 5), dtype='float32')

        core = Conv2D(10, 3, strides=1, padding='same', input_shape=(11, 8, 5))(map_input)
        core = BatchNormalization()(core)
        core = Activation('relu')(core)
        core = Conv2D(10, 3, strides=1, padding='same')(core)
        core = BatchNormalization()(core)
        core = Activation('relu')(core)
        core = Conv2D(10, 3, strides=1, padding='same')(core)
        core = BatchNormalization()(core)
        core = Activation('relu')(core)

        policy = Conv2D(4, 3, strides=1, padding='same', activation='relu')(core)
        policy = Conv2D(2, 3, strides=1, padding='same')(policy)  # 11 x 8 x 2

        value = Flatten()(core)
        value = Dense(20, use_bias=True)(value)
        value = Dense(1)(value)

        model = Model([map_input], [policy, value])
        return model

    def strip_reshape(self, arr):
        return np.reshape(arr, tuple(s for s in arr.shape if s > 1))
