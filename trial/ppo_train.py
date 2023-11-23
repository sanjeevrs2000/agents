import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import tf_agents
from tf_agents.environments import tf_py_environment
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.environments import wrappers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers.schedules import ExponentialDecay, PolynomialDecay
# from tf_agents.networks import sequential
from environment import ship_environment
import os

env = wrappers.TimeLimit(ship_environment(), duration=160)
tf_env = tf_py_environment.TFPyEnvironment(env)

actor_fc_layers=(128,128)
value_fc_layers=(128,128)

actor_net = actor_distribution_network.ActorDistributionNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    fc_layer_params=actor_fc_layers,
    activation_fn=tf.keras.activations.tanh)

value_net = value_network.ValueNetwork(
    tf_env.observation_spec(),
    fc_layer_params=value_fc_layers,
    activation_fn=tf.keras.activations.tanh)

agent = ppo_clip_agent.PPOClipAgent(
    time_step_spec=tf_env.time_step_spec(),
    action_spec=tf_env.action_spec(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                       # ExponentialDecay(initial_learning_rate=0.001,decay_rate=0.8,decay_steps=300)),
    actor_net = actor_net,
    value_net = value_net,
    greedy_eval = True,
    importance_ratio_clipping = 0.2,
    lambda_value = 0.95,
    discount_factor = 0.95,
    entropy_regularization = 0.01,
    num_epochs = 10,
    spatial_similarity_coef=0,
    temporal_similarity_coef=0.001,
    use_gae= True,
    use_td_lambda_return= False,
    normalize_rewards = True,
    normalize_observations= True,
    debug_summaries = False,
    # train_step_counter = tf.Variable(0)
)

agent.initialize()

replay_buffer_max_length = 8000
# sample_batchsize=100

timestep_counter=0
ep_counter=0


def collect_data(environment, policy, buffer):
    global timestep_counter
    time_step = environment.reset()
    episode_return = 0
    while not np.equal(time_step.step_type, 2):

        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        episode_return += next_time_step.reward.numpy()[0]
        traj = trajectory.from_transition(time_step, action_step, next_time_step)
        buffer.add_batch(traj)
        time_step=next_time_step
        timestep_counter += 1

    return episode_return


episodes_per_iteration=3
num_iterations=100
avg_losses, avg_returns = [], []

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=replay_buffer_max_length)



for i in range(num_iterations):

    replay_buffer.clear()
    returns = []

    print('ITERATION NO: ',i+1)
    # collecting samples by running a number of episodes with same policy
    for ep in range(episodes_per_iteration):
        ep_return=collect_data(tf_env, agent.collect_policy, replay_buffer)
        returns.append(ep_return)
        print('Episode {}: returns={}'.format(ep+1,ep_return))

    avg_returns.append(np.mean(returns))
    print('ITERATION NO:{} END; AVG RETURNS={}'.format (i + 1,np.mean(avg_returns[-1])))

    # updating policy
    experience=replay_buffer.gather_all()
    agent.train(experience)
    # print(agent.train_step_counter)

# saving model with agent's policy
mname='ppo_test_'
policy_dir = os.path.join('saved_agents',mname)
tf_policy_saver = policy_saver.PolicySaver(agent.policy)
tf_policy_saver.save(policy_dir)


plt.figure()
plt.title("Training Returns vs. Iterations")
plt.ylabel("Avg Returns")
plt.plot(avg_returns)
plt.xlabel("Iterations")
plt.grid()
plt.savefig("plots/ppo_test_return",dpi=600)
plt.show()
