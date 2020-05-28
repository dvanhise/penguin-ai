import numpy as np
import matplotlib.pyplot as plt

MAX_STEPS = 60


class Episode:
    def __init__(self):
        self.current_step = 0

        self.map_input = np.zeros((MAX_STEPS, 11, 8, 5))
        self.value = np.zeros((MAX_STEPS,))
        self.policy = np.zeros((MAX_STEPS, 11, 8, 2))

        self.policy_one_hot = np.zeros((MAX_STEPS, 11, 8, 2))
        self.reward = np.zeros((MAX_STEPS,))
        self.player = [.5]*MAX_STEPS
        self.policy_mask = np.zeros((MAX_STEPS, 11, 8, 2))

    def save_step(self, map_input, value, policy, policy_mask, player, target, destination):
        self.map_input[self.current_step] = map_input
        self.value[self.current_step] = value
        self.policy[self.current_step] = policy

        if target:
            self.policy_one_hot[self.current_step][target[0]][target[1]][0] = 1.0
        if destination:
            self.policy_one_hot[self.current_step][destination[0]][destination[1]][1] = 1.0
        self.player[self.current_step] = player
        self.policy_mask[self.current_step] = policy_mask

        self.current_step += 1

    def set_rewards(self, player, reward):
        for ndx, p in enumerate(self.player[:self.current_step]):
            if p == player:
                self.reward[ndx] = reward
