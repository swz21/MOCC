# Copyright 2019 Nathan Jay and Noga Rotman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gym
import network_sim
import tensorflow as tf


from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines import PPO1
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from common.simple_arg_parse import arg_or_default
import argparse

parser = argparse.ArgumentParser(description='Train a PCC agent using PPO1.')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor.')
parser.add_argument('--arch', type=str, default="64,32", help='Network architecture.')
parser.add_argument('--weights', type=float, nargs='+', default=[0.6, 0.3, 0.1], help='Weights for the reward function.')
parser.add_argument('--model_dir', type=str, default="/tmp/pcc_saved_models/model_000/", help='Directory to save the model.')
parser.add_argument('--ckpt_dir', type=str, default="./data_000/pcc_model_0.ckpt", help='Directory to save ckpt.')
args = parser.parse_args()

# arch_str = arg_or_default("--arch", default="64,32")
arch_str = args.arch
if arch_str == "":
    arch = []
else:
    arch = [int(layer_width) for layer_width in arch_str.split(",")]
print("Architecture is: %s" % str(arch))

training_sess = None

class MyMlpPolicy(FeedForwardPolicy):

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        _kwargs['act_fun'] = tf.nn.tanh
        super(MyMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse, net_arch=[{"pi":arch, "vf":arch}],
                                        feature_extraction="mlp", **_kwargs)
        global training_sess
        training_sess = sess

env = gym.make('PccNs-v0', weights = args.weights)
#env = gym.make('CartPole-v0')

# gamma = arg_or_default("--gamma", default=0.99)
gamma = args.gamma
print("gamma = %f" % gamma)
model = PPO1(MyMlpPolicy, env, verbose=1, schedule='constant', timesteps_per_actorbatch=8192, optim_stepsize = 0.001, optim_batchsize=2048, gamma=gamma)

# tf.saved_model.load(training_sess, [tag_constants.SERVING], "/tmp/pcc_saved_models/model_B/")

for i in range(0, 6):
    with model.graph.as_default():                                                                   
        saver = tf.train.Saver()
        if i == 0:
            saver.restore(training_sess, "./pcc_model.ckpt")
        # saver.save(training_sess, "./pcc_model_%d.ckpt" % i)
        saver.save(training_sess, args.ckpt_dir + "_%d.ckpt" % i)
    model.learn(total_timesteps=(1600 * 410))

with model.graph.as_default():                                                                   
        saver = tf.train.Saver()
        saver.save(training_sess, args.ckpt_dir + ".ckpt")

##
#   Save the model to the location specified below.
##
# default_export_dir = "/tmp/pcc_saved_models/model_118/"
# export_dir = arg_or_default("--model-dir", default=default_export_dir)
export_dir = args.model_dir
with model.graph.as_default():

    pol = model.policy_pi#act_model

    obs_ph = pol.obs_ph
    act = pol.deterministic_action
    sampled_act = pol.action

    obs_input = tf.saved_model.utils.build_tensor_info(obs_ph)
    outputs_tensor_info = tf.saved_model.utils.build_tensor_info(act)
    stochastic_act_tensor_info = tf.saved_model.utils.build_tensor_info(sampled_act)
    signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={"ob":obs_input},
        outputs={"act":outputs_tensor_info, "stochastic_act":stochastic_act_tensor_info},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    #"""
    signature_map = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                     signature}

    model_builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    model_builder.add_meta_graph_and_variables(model.sess,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map=signature_map,
        clear_devices=True)
    model_builder.save(as_text=True)
