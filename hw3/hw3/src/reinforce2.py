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



def get_actor_loss():
    def custom_actor_loss(y_true, y_predict):
        # idx = np.argmax(y_true)
        # print y_true[idx], y_predict[idx]
        return -K.mean(y_true * K.log(y_predict))
    return custom_actor_loss


def multinomial_sample(policy):
    num_classes = policy.shape[0]
    thresholds = np.zeros(num_classes + 1)
    for i in range(num_classes):
        thresholds[i+1] = thresholds[i] + policy[i]
    rand = np.random.uniform()
    for i in range(num_classes):
        if rand >= thresholds[i] and rand <= thresholds[i+1]:
            return i
    return num_classes - 1
    


def collectData(model, env, episodeNum=100):
    expertTraj = {}
    expertTraj['name'] = str(episodeNum) + " trajectories"
    expertTraj["states"] = []
    expertTraj["actions"] = []
    expertTraj["rewards"] = []
    while episodeNum >= 1:
        episodeNum -= 1
        states, actions, rewards = Reinforce.generate_episode(model, env)
        expertTraj["states"].append(states)
        expertTraj["actions"].append(actions)
        expertTraj["rewards"].append(rewards)
    return expertTraj


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, model, lr):
        self.model = model

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.
        opt = keras.optimizers.Adam(lr=lr)
        # self.model.compile(optimizer= opt, loss="categorical_crossentropy", metrics=['accuracy'])
        self.model.compile(optimizer= opt, loss=get_actor_loss(), metrics=['accuracy'])

    def _accumG(self, Gs, y, gamma):
        # print Gs
        y[-1] *= Gs[-1]
        for i in range(Gs.size-2, -1, -1):
            Gs[i] += Gs[i+1] * gamma
            y[i] *= Gs[i]
        return y


    def run_model(self, env, render=False):
        # Generates an episode by running the cloned policy on the given env.
        return self.generate_episode(self.model, env, render)



    def train(self, env, gamma=0.9):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        
        # pseudo-code:
        # 1: sample an episode from the policy, record states, actions and rewards
        # 2: compute G_t as y
        # 
        loss = 0
        acc = 0
        num_episodes = 1        
        # X = X.reshape(-1, sample["states"][0][0].size)
        # y = y.reshape((-1, sample["actions"][0][0].size))
        # do batch
        print "do batch"
        epochs = 1000
        
        while epochs > 0:
            epochs -= 1
            states, actions, rewards = self.generate_episode(self.model, env, render=False)
            states = np.array(states)
            actions = np.array(actions)
            Gs = sum(rewards)
            rewards = np.array(rewards)
            T = len(states)
            t = T - 1
            G = np.zeros((T, 1))
            
            while t >= 0:
                multi = 1
                # state_input[t] = states[t]
                # action_input[t] = actions[t]
                for i in range(t, T):
                    G[t] += multi * rewards[i]
                    multi *= gamma
                t -=1
            # print states.shape, actions.shape, rewards.shape
            # G=G.flatten()
            
            action = np.zeros(actions.shape[0])
            for i in range(action.shape[0]):
                print actions[i]
                action[i] = np.argmax(actions[i])
            y_true = keras.utils.to_categorical(action, num_classes=4)
            print actions == y_true
            y = self._accumG(rewards, actions, gamma)

            y_true = G * y_true
            print y, y_true
            break
            loss = self.model.train_on_batch(states, y)
            if epochs % 100 == 0:
                print epochs, "loss ", loss, "rewards: ", rewards[0], Gs

        # sample = collectData(self.model, env, num_episodes)
        # X = np.array(sample["states"][0])
        # y = np.array(sample["actions"][0])
        # G = np.array(sample["rewards"][0])
        # y = self._accumG(G, y, gamma)
        # loss, acc = self.model.evaluate(x=X, y=y, batch_size=X.shape[0])
    
        return loss, acc

    
    def save_model(self):
        with open("model/REINFORCE.json", "w") as json_file:
            json_file.write(self.model.to_json())
        self.model.save_weights("model/REINFORCE_weights.h5")
        print('model saved.')


    # @staticmethod
    def generate_episode(self, model, env, render=False):
        # Generates an episode by running the given model on the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        states = []
        actions = []
        rewards = []
        state = env.reset()
        if render:
            env.render()
        while True:
            states.append(state)
            action = model.predict(state.reshape(1, state.size)).flatten()
            act = multinomial_sample(action)
            action = np.zeros(action.shape)
            action[act] += 1
            state, reward, isTerminal, debugInfo = env.step(act)
            if render:
                env.render()
                # print reward
            
            actions.append(action)
            rewards.append(reward)
            if isTerminal:
                break
            
        return states, actions, rewards


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
    lr = 0.001
    reinforce = Reinforce(model, lr)
    reinforce.train(env)
    a = 10
    while a > 0:
        a -= 1
        reinforce.run_model(env, render=True)
    # TODO: Train the model using REINFORCE and plot the learning curve.


if __name__ == '__main__':
    main(sys.argv)
