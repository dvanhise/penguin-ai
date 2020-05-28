from game import HtmfGame
from agent import PenguinAgent
from human_agent import HumanAgent

import os
import time
from itertools import count
import argparse

parser = argparse.ArgumentParser(description='Train and play Hey, That\'s My Fish against an AI')
parser.add_argument('--mode', default='train', nargs='?', choices=['train', 'play', 'analyze'])
parser.add_argument('--p1', default=False, action='store_const', const=True)  # Does nothing
parser.add_argument('--p2', default=False, action='store_const', const=True)


# AI plays itself to train
def train():
    agent = load_agent('agent')
    adversary = PenguinAgent()
    game = HtmfGame()

    for episode in count(1):
        t1 = time.time()
        print('Starting self play...')
        for i in range(3):
            game_loop(game, agent, agent)
        print('Episode %d complete in %.2fs' % (episode, time.time() - t1))

        adversary.model.set_weights(agent.model.get_weights())
        t2 = time.time()
        loss = agent.train()
        print('Training complete in %.2fs.  val_loss: %.3f,  policy_loss: %.3f, entropy: %.3f' %
              ((time.time() - t2), loss[0], loss[1], loss[2]))

        print('Begin testing against previous version...')
        record = {'win': 0, 'lose': 0, 'draw': 0}
        t3 = time.time()
        for i in range(50):
            result = game_loop(game, agent, adversary, training=False)
            record[result] += 1
        print('Testing complete in %.2fs' % (time.time() - t3))
        print('Testing record {win}-{lose}-{draw}'.format(**record))

        if record['win'] <= record['lose']:
            print('Reverting to last model state')
            agent.model.set_weights(adversary.model.get_weights())
        else:
            print('Saving updated model')
            agent.model.save_weights('./save/agent.tf')


# Play one game against the trained AI
def play(first_player=True):
    print('First player: %s' % first_player)
    agent = load_agent('agent')
    human = HumanAgent()
    game = HtmfGame()
    if first_player:
        result = game_loop(game, human, agent, render=[0])
    else:
        result = game_loop(game, agent, human, render=[1])
    print('You %s: score %s' % (result, str(game.get_score())))


# Attempts to load a model from file with the given name, randomly initializes it otherwise
def load_agent(name):
    agent = PenguinAgent()
    if os.path.exists('./save/%s.tf.index' % name):
        print('Loading model weights from file')
        agent.model.load_weights('./save/%s.tf' % name)
    else:
        print('Could not find saved weights, randomly initilizaing model')
    return agent


def game_loop(game, agent1, agent2, training=True, render=None):
    """
    :param game:
    :param agent1: First player
    :param agent2: Second player
    :param training: Whether to run the agent steps in training mode
    :param render: A list of the players (humans) for which the map should be rendered
    :return: The game result for agent1
    """

    a = [agent1, agent2]
    game.reset()
    current_player = 0
    phase = 0
    for turn in count():
        if render and current_player in render:
            game.render_map()

        state = game.get_state()
        target, destination = a[current_player].step(state, current_player, training=training)
        if phase == 0:
            phase, current_player = game.place(target)
        elif phase == 1:
            phase, current_player = game.move(target, destination)

        # Game over
        if phase == 2:
            agent1.step_end([game.get_reward(0), game.get_reward(1)])
            agent2.step_end([game.get_reward(0), game.get_reward(1)])
            break

    # Returns result for first agent
    if game.get_reward(0) > game.get_reward(1):
        return 'win'
    elif game.get_reward(0) < game.get_reward(1):
        return 'lose'
    else:
        return 'draw'


if __name__ == '__main__':
    args = parser.parse_args()
    if args.mode in ['train', 'analyze']:
        train()
    elif args.mode == 'play':
        play(not args.p2)
