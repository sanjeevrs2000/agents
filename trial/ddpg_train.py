import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tf_agents
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.drivers import dynamic_step_driver
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import wrappers
from tf_agents.policies import random_tf_policy
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers.schedules import ExponentialDecay, PolynomialDecay
# from tf_agents.networks import sequential
from environment import ship_environment
import os

env = wrappers.TimeLimit(ship_environment(), duration=160)
tf_env = tf_py_environment.TFPyEnvironment(env)

lr_schedule1=ExponentialDecay(initial_learning_rate=0.002, decay_steps=60000, decay_rate=0.9)
lr_schedule2 = ExponentialDecay(initial_learning_rate=0.0003, decay_steps=60000,decay_rate=0.9)

ac_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule2)

cr_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule1)

critic_joint_layers=(250,250)
critic_obs_layer=(32,32,)
critic_act_layer=(16,16,)
actor_layers=(250,250)

# dense_layers_1=Dense(actor_layers[0],activation=tf.keras.activations.relu,kernel_initializer=tf.keras.initializers.GlorotNormal())
# dense_layers_2=Dense(actor_layers[1],activation=tf.keras.activations.relu,kernel_initializer=tf.keras.initializers.GlorotNormal())
# action_layer=Dense(1,activation=tf.keras.activations.tanh,kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.001,maxval=0.001))
# net = sequential.Sequential([dense_layers_1,dense_layers_2,action_layer],input_spec=tf_env.observation_spec())

actor_net = actor_network.ActorNetwork(tf_env.observation_spec(), tf_env.action_spec(),
                                       fc_layer_params=actor_layers,activation_fn=tf.keras.activations.relu,
                                       kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                       last_kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                        # last_kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.003,maxval=0.003)
                                       )

critic_net = critic_network.CriticNetwork((tf_env.observation_spec(), tf_env.action_spec()),
                                        observation_fc_layer_params=critic_obs_layer,action_fc_layer_params=critic_act_layer,
                                        joint_fc_layer_params=critic_joint_layers,activation_fn=tf.keras.activations.relu,
                                        kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                        last_kernel_initializer=tf.keras.initializers.GlorotNormal()
                                        )

train_step_counter = tf.Variable(0)

agent = ddpg_agent.DdpgAgent(
    tf_env.time_step_spec(),
    tf_env.action_spec(),
    actor_network=actor_net,
    critic_network=critic_net,
    actor_optimizer=ac_optimizer,
    critic_optimizer=cr_optimizer,
    gamma=0.95,
    spatial_similarity_coef=0.001,
    temporal_similarity_coef=0.001,
    # ou_stddev=0.15,
    # ou_damping =1,
    target_update_tau = 0.01,
    target_update_period =5,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()

episodes=7001
replay_buffer_max_length = 100000
sample_batchsize=128
init_collect_iterations=10
returns,losses=[],[]

# print(tf_env.batch_size, "batch size")

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=replay_buffer_max_length)

# Dataset generates trajectories with shape [Bx2x...]
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=sample_batchsize,
    num_steps=2).prefetch(3)

def collect_init_data(environment, policy, buffer):
    time_step = environment.reset()
    episode_return = 0
    while not np.equal(time_step.step_type, 2):

        action_step = policy.action(time_step)
        # print(action_step.action)
        next_time_step = environment.step(action_step.action)
        episode_return += next_time_step.reward.numpy()[0]
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        buffer.add_batch(traj)
        time_step=next_time_step

    return episode_return

random_policy = random_tf_policy.RandomTFPolicy(tf_env.time_step_spec(),tf_env.action_spec())

# collecting samples with random policy to populate buffer before training
for __ in range(init_collect_iterations):
    collect_init_data(tf_env, random_policy, replay_buffer)

timestep_counter=0
ep_counter=0

#collect data through train policy with random noise and train
def collect_data(environment, policy, buffer):
    global timestep_counter
    time_step = environment.reset()
    episode_return = 0
    while not np.equal(time_step.step_type, 2):

        action_step = policy.action(time_step)

        #adding custom noise from random gaussian process with reducing std
        action=action_step.action
        std=0.15*(1-ep_counter/9000)
        action=action+np.random.normal(loc=[0],scale=[std])
        if action>35*(np.pi/180):
            action = tf.constant([[35*(np.pi/180)]], shape=(1,1,), dtype=np.float32, name='action')
        elif action < -35*(np.pi/180):
            action = tf.constant([[-35*(np.pi/180)]], shape=(1,1,), dtype=np.float32, name='action')
        action_step = tf_agents.trajectories.policy_step.PolicyStep(action)

        next_time_step = environment.step(action_step.action)
        episode_return += next_time_step.reward.numpy()[0]
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        buffer.add_batch(traj)
        time_step=next_time_step

        # if timestep_counter%5==0:
        experience, unused_info = next(iterator)
        agent.train(experience)
            # print(timestep_counter,agent.train_step_counter)

        timestep_counter += 1
    return episode_return

# (Optional) Optimize by wrapping some of the code in a graph using
# TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)
iterator = iter(dataset)

while ep_counter<episodes:
    ep_return=collect_data(tf_env, agent.policy, replay_buffer)
    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    ep_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()
    print('episode = {0}: loss = {1}  return = {2}'.format(ep_counter, ep_loss,ep_return))
    returns.append(ep_return)
    losses.append(ep_loss)

    # save model at intermediate intervals
    if ep_counter % 1000 == 0 and ep_counter >= 3000:
        mname='ddpg_mar9_gamma85_'+str(ep_counter)
        policy_dir = os.path.join('saved_agents',mname)
        tf_policy_saver = policy_saver.PolicySaver(agent.policy)
        tf_policy_saver.save(policy_dir)

        # checkpoint_dir = 'saved_agents/ddpg_jan11_'
        # train_checkpointer = common.Checkpointer(
        #     ckpt_dir=checkpoint_dir,
        #     max_to_keep=1,
        #     agent=agent,
        #     policy=agent.policy,
        #     replay_buffer=replay_buffer,
        # )
        # train_checkpointer.save(agent.train_step_counter.numpy())
    ep_counter+=1

interval = 100
avg_losses, avg_returns = [], []

for i in range(len(losses) - interval):
    avg_returns.append(sum(returns[i:i + interval]) / interval)
    avg_losses.append(sum(losses[i:i + interval]) / interval)

plt.figure()
plt.title("Training Returns vs. Episodes")
plt.ylabel("Returns")
plt.plot(avg_returns)
plt.xlabel("Episodes")
plt.grid()
plt.savefig("plots/ddpg_mar9_gamma85_return",dpi=600)
plt.show()

plt.figure()
plt.title("Train-loss vs. Episodes")
plt.xlabel("Episodes")
plt.ylabel("Loss")
plt.plot(avg_losses)
plt.grid()
plt.savefig("plots/ddpg_mar9_gamma85_loss",dpi=600)
plt.show()
