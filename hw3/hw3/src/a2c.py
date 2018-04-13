import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reinforce import Reinforce

NUM_ACTIONS = 4
STATE_DIM = 8


def get_actor_loss():
    def custom_actor_loss(y_true, y_predict):
        return -K.sum(y_true * K.log(y_predict), axis=-1)
    return custom_actor_loss



class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self, model, lr, critic_model, critic_lr, n=20):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - n: The value of N in N-step A2C.
        self.model = model
        self.critic_model = critic_model
        self.n = n

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.
        actor_optimizer = keras.optimizers.Adam(lr=lr)
        self.model.compile(loss=get_actor_loss(), optimizer=actor_optimizer)
        critic_optimizer = keras.optimizers.Adam(lr=critic_lr)
        self.critic_model.compile(loss=keras.losses.mean_squared_error, optimizer=critic_optimizer)

    def train(self, env, gamma=1.0, update_actor=False):
        # Trains the model on a single episode using A2C.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        states, actions, rewards = self.generate_episode(env)
        reward = [0.01 * i for i in rewards]
        R = np.zeros((len(reward), 1))
        state = np.zeros((len(reward), STATE_DIM))
        action = np.zeros(len(reward))
        V = self.critic_model.predict(state)
        # print(np.max(V))
        for i in range(R.shape[0]):
            multi = 1
            state[i] = states[i]
            action[i] = actions[i]
            for j in range(self.n):
                r = 0 if (i + j >= R.shape[0]) else reward[i+j]
                R[i, 0] += multi * r
                multi *= gamma
            v = 0 if (i + self.n >= R.shape[0]) else V[i+self.n, 0]
            R[i, 0] += multi * v
        '''
        multi = 1
        actor_R = np.copy(R)
        for i in range(R.shape[0]):
            actor_R[i, 0] *= multi
            multi *= gamma
        '''
        # train actor model
        actor_loss = 0
        if update_actor:
            y_true = keras.utils.to_categorical(action, num_classes=NUM_ACTIONS)
            y_true = (R - V) * y_true
            actor_loss = self.model.train_on_batch(state, y_true)
        # train critic model
        critic_loss = self.critic_model.train_on_batch(state, R)
        total_rewards = sum(rewards)
        return actor_loss, critic_loss, total_rewards

    def save_model(self):
        with open("4level-32dim-model/" + str(self.n) + "actor_model.json", "w") as json_file:
            json_file.write(self.model.to_json())
        self.model.save_weights("4level-32dim-model/" + str(self.n) + "actor_model_weights.h5")

        with open("4level-32dim-model/" + str(self.n) + "critic_model.json", "w") as json_file:
            json_file.write(self.critic_model.to_json())
        self.critic_model.save_weights("4level-32dim-model/" + str(self.n) + "critic_model_weights.h5")

        print('model saved.')

    def generate_episode(self, env, render=False):
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

        '''
        state = env.reset()
        done = False
        episode = []
        total_rewards = 0
        while not done:
            if render:
                env.render()
            policy = self.model.predict(state.reshape((1, STATE_DIM))).reshape(NUM_ACTIONS)
            action = self.multinomial_sample(policy)
            observation, reward, done, _ = env.step(action)
            total_rewards += reward
            episode.append((state, action, reward))
            state = observation
        return episode, total_rewards
        '''

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


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config-path', dest='model_config_path',
                        type=str, default='LunarLander-v2-config.json',
                        help="Path to the actor model config file.")
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")
    parser.add_argument('--use-saved-model', dest='use_saved_model', type=bool,
                        default=False, help="If use saved models.")
    parser.add_argument('--save-model', dest='save_model', type=bool,
                        default=False, help="If save models.")

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
    critic_lr = args.critic_lr
    n = args.n
    render = args.render
    use_saved_model = args.use_saved_model
    save_model = args.save_model

    # Create the environment.
    env = gym.make('LunarLander-v2')

    # Load the actor model from file.
    if not use_saved_model:

        with open(model_config_path, 'r') as f:
            model = keras.models.model_from_json(f.read())
        '''
        model = Sequential()
        model.add(Dense(16, input_dim=STATE_DIM, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(16, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(16, kernel_initializer='uniform', activation='relu'))
        model.add(Dense(2, kernel_initializer='uniform', activation='softmax'))
        '''

        # Define critic model
        critic_model = Sequential()
        critic_model.add(Dense(32, input_dim=STATE_DIM, kernel_initializer='VarianceScaling', activation='relu', use_bias=True))
        critic_model.add(Dense(64, kernel_initializer='VarianceScaling', activation='relu', use_bias=True))
        critic_model.add(Dense(64, kernel_initializer='VarianceScaling', activation='relu', use_bias=True))
        critic_model.add(Dense(32, kernel_initializer='VarianceScaling', activation='relu', use_bias=True))
        critic_model.add(Dense(1, kernel_initializer='VarianceScaling', use_bias=True))
    else:
        with open("4level-32dim-model/" + str(n) + "actor_model.json", 'r') as f1:
            model = keras.models.model_from_json(f1.read())
        model.load_weights("4level-32dim-model/" + str(n) + "actor_model_weights.h5")
        with open("4level-32dim-model/" + str(n) + "critic_model.json", 'r') as f2:
            critic_model = keras.models.model_from_json(f2.read())
        critic_model.load_weights("4level-32dim-model/" + str(n) + "critic_model_weights.h5")
        print('use saved model.')

    # TODO: Train the model using A2C and plot the learning curves.
    a2c = A2C(model, lr, critic_model, critic_lr, n)
    actor_losses = []
    critic_losses = []
    test_reward_means = []
    test_reward_stds = []
    for i in range(num_episodes):
        actor_loss, critic_loss, total_rewards = a2c.train(env, gamma=0.99, update_actor=True)
        # print('episode: ' + str(i), actor_loss, critic_loss, total_rewards)
        if i % 500 == 0:
            print(actor_loss, critic_loss)
            num_test = 100
            test_reward = np.zeros(num_test)
            a2c.generate_episode(env, render=True)
            for j in range(num_test):
                _, _, reward = a2c.generate_episode(env, render=False)
                test_reward[j] = sum(reward)
            test_reward_means.append(test_reward.mean())
            test_reward_stds.append(np.std(test_reward))
            print('episode ' + str(i) + ': ' + str(test_reward.mean()) + ' ' + str(np.std(test_reward)))
    result = np.array([test_reward_means, test_reward_stds])
    curr_time = time.time()
    np.savetxt('4level-32dim-model/' + str(n) + 'test.out', result)
    if save_model:
        a2c.save_model()


if __name__ == '__main__':
    main(sys.argv)
