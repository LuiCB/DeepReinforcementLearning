import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym
from keras import backend as K

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NUM_ACTIONS = 4
STATE_DIM = 8


def get_actor_loss():
    def custom_actor_loss(y_true, y_predict):
        return -K.sum(y_true * K.log(y_predict), axis=-1)
    return custom_actor_loss


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr):
        self.model = model

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.
        actor_optimizer = keras.optimizers.Adam(lr=lr)
        self.model.compile(loss=get_actor_loss(), optimizer=actor_optimizer)

    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        states, actions, rewards = self.generate_episode(env)
        T = len(states)
        t = T - 1
        G = np.zeros((T, 1))
        action_input = np.zeros(T)
        state_input = np.zeros((T, STATE_DIM))
        while t >= 0:
            multi = 1
            state_input[t] = states[t]
            action_input[t] = actions[t]
            for i in range(t, T):
                G[t][0] += multi * rewards[i]
                multi *= gamma
            t -=1
        y_true = keras.utils.to_categorical(action_input, num_classes=NUM_ACTIONS)
        y_true = G * y_true
        loss = self.model.train_on_batch(state_input, y_true / 100.0)

        return loss, sum(rewards)


    def generate_episode(self, env, render=False):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        states = []
        actions = []
        rewards = []
        state = env.reset()
        done = False
        total_rewards = 0
        while not done:
            if render:
                env.render()
            policy = self.model.predict(state.reshape((1, STATE_DIM))).reshape(NUM_ACTIONS)
            action = self.multinomial_sample(policy)
            observation, reward, done, _ = env.step(action)
            total_rewards += reward
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = observation

        return states, actions, rewards

    def multinomial_sample(self, policy):
        num_classes = policy.shape[0]
        thresholds = np.zeros(num_classes + 1)
        for i in range(num_classes):
            thresholds[i+1] = thresholds[i] + policy[i]
        rand = np.random.uniform()
        for i in range(num_classes):
            if rand >= thresholds[i] and rand <= thresholds[i+1]:
                return i
        return num_classes - 1

    def save_model(self):
        with open("reinforce_model/model.json", "w") as json_file:
            json_file.write(self.model.to_json())
        self.model.save_weights("reinforce_model/model_weights.h5")



def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    model_config_path = args.model_config_path
    num_episodes = args.num_episodes
    lr = args.lr
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')

    # Load the policy model from file.
    with open(model_config_path, 'r') as f:
        model = keras.models.model_from_json(f.read())

    # TODO: Train the model using REINFORCE and plot the learning curve.
    reinforce = Reinforce(model, lr)
    test_reward_means = []
    test_reward_stds = []
    for i in range(num_episodes):
        loss, total_rewards = reinforce.train(env, gamma=0.99)
        if i % 500 == 0:
            num_test = 100
            test_reward = np.zeros(num_test)
            reinforce.generate_episode(env, render=True)
            for j in range(num_test):
                _, _, rewards = reinforce.generate_episode(env, render=False)
                test_reward[j] = sum(rewards)
            test_reward_means.append(test_reward.mean())
            test_reward_stds.append(np.std(test_reward))
            print('episode ' + str(i) + ': ' + str(test_reward.mean()) + ' ' + str(np.std(test_reward)))
    result = np.array([test_reward_means, test_reward_stds])
    np.savetxt('reinforce_model/test.out', result)
    reinforce.save_model()


if __name__ == '__main__':
    main(sys.argv)
