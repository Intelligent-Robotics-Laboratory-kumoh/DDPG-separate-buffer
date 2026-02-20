#!/usr/bin/env python
# An implementation of UAV DRL (DDPG + PER) - TF1 minimal BN patch + PER stabilization

import rospy
from drl_uav.srv import getEnvinfo, getEnvinfoResponse
from drl_uav.srv import getEnvinfo_att, getEnvinfo_attResponse

import random, numpy, math, time, sys, os, csv
from datetime import datetime
import numpy as np
import tensorflow as tf

tf.debugging.set_log_device_placement(True)
tf.compat.v1.disable_eager_execution()

# ======================
# Hyper-parameters (noise)
# ======================
mu = 0
theta = 0.15
sigma = 0.2

# ======================
# PER SumTree
# ======================
class SumTree:
    """우선순위를 효율적으로 저장하고 샘플링하기 위한 자료구조"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        self.n_entries = min(self.n_entries + 1, self.capacity)

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]


class PERMemory:
    """Prioritized Experience Replay 관리 클래스 (안정화 패치 포함)"""
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.epsilon = 0.01
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001
        self.abs_err_upper = 1.0

    def store(self, transition):
        # 최대 우선순위로 넣어 1회 이상 샘플링 보장
        leaf_priorities = self.tree.tree[-self.tree.capacity:]
        max_p = np.max(leaf_priorities)
        if max_p <= 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)

    def can_sample(self, batch_size):
        return self.tree.n_entries >= batch_size and self.tree.total_p > 0

    def sample(self, n):
        batch_idx, batch_memory, is_weights = [], [], []

        total_p = self.tree.total_p
        if total_p <= 0:
            return [], [], np.ones((0, 1), dtype=np.float32)

        segment = total_p / n
        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        # ✅ 안정화 1: min_prob 계산 시 priority==0(빈 leaf) 제외
        leaf_p = self.tree.tree[-self.tree.capacity:]
        nonzero = leaf_p[leaf_p > 0]
        if nonzero.size == 0:
            min_prob = 1.0 / self.tree.capacity
        else:
            min_prob = np.min(nonzero) / total_p
        min_prob = max(min_prob, 1e-12)

        for i in range(n):
            a, b = segment * i, segment * (i + 1)
            v = random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)

            prob = p / total_p
            prob = max(prob, 1e-12)

            w = np.power(prob / min_prob, -self.beta)
            is_weights.append(w)
            batch_idx.append(idx)
            batch_memory.append(data)

        is_weights = np.array(is_weights, dtype=np.float32).reshape(-1, 1)

        # ✅ 안정화 2: IS weight 정규화(최대=1)
        is_weights /= (np.max(is_weights) + 1e-12)

        return batch_idx, batch_memory, is_weights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors = abs_errors + self.epsilon
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


# ======================
# OU Noise
# ======================
class OU_noise:
    def __init__(self, action_size):
        self.action_size = action_size
        self.reset()

    def reset(self):
        self.X = numpy.ones(self.action_size) * mu

    def sample(self):
        dx = theta * (mu - self.X) + sigma * numpy.random.randn(len(self.X))
        self.X += dx
        return self.X


# ======================
# Actor / Critic (BN minimal patch)
# ======================
class Actor:
    def __init__(self, state_size, action_size, name):
        self.name = name
        with tf.compat.v1.variable_scope(name):
            self.state = tf.compat.v1.placeholder(tf.float32, [None, state_size], name="state")
            self.is_training = tf.compat.v1.placeholder_with_default(False, shape=(), name="is_training")

            fc1 = tf.compat.v1.layers.dense(self.state, 900, activation=tf.nn.elu, name="fc1")
            bn1 = tf.compat.v1.layers.batch_normalization(fc1, training=self.is_training, name="bn1")

            fc2 = tf.compat.v1.layers.dense(bn1, 800, activation=tf.nn.elu, name="fc2")
            bn2 = tf.compat.v1.layers.batch_normalization(fc2, training=self.is_training, name="bn2")

            fc3 = tf.compat.v1.layers.dense(bn2, 700, activation=tf.nn.elu, name="fc3")
            bn3 = tf.compat.v1.layers.batch_normalization(fc3, training=self.is_training, name="bn3")

            self.action = tf.compat.v1.layers.dense(bn3, action_size, activation=tf.nn.tanh, name="action")

        self.trainable_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=name)


class Critic:
    def __init__(self, state_size, action_size, name):
        self.name = name
        with tf.compat.v1.variable_scope(name):
            self.state = tf.compat.v1.placeholder(tf.float32, [None, state_size], name="state")
            self.action_size_placeholder = tf.compat.v1.placeholder(tf.float32, [None, action_size], name="action_in")
            self.is_training = tf.compat.v1.placeholder_with_default(False, shape=(), name="is_training")

            concat = tf.concat([self.state, self.action_size_placeholder], axis=-1, name="concat")

            fc1 = tf.compat.v1.layers.dense(concat, 900, activation=tf.nn.elu, name="fc1")
            bn1 = tf.compat.v1.layers.batch_normalization(fc1, training=self.is_training, name="bn1")

            fc2 = tf.compat.v1.layers.dense(bn1, 800, activation=tf.nn.elu, name="fc2")
            bn2 = tf.compat.v1.layers.batch_normalization(fc2, training=self.is_training, name="bn2")

            fc3 = tf.compat.v1.layers.dense(bn2, 700, activation=tf.nn.elu, name="fc3")

            self.predict_q = tf.compat.v1.layers.dense(fc3, 1, activation=None, name="q")

        self.trainable_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=name)


# ======================
# Agent (DDPG + PER)
# ======================
class Agent:
    def __init__(self, num_state, num_action):
        self.num_state = num_state
        self.num_action = num_action

        # Hyper-params
        self.MEMORY_CAPACITY = 100000
        self.GAMMA = 0.99
        self.MEMORY_BATCH_SIZE = 64
        self.tau = 0.001
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.epsilon = 1.0
        self.EPSILON_DECAY = 0.9995
        self.MIN_EPSILON = 0.01

        # PER memory
        self.memory = PERMemory(capacity=self.MEMORY_CAPACITY)

        # Networks
        self.actor_main = Actor(num_state, num_action, "main_actor")
        self.actor_target = Actor(num_state, num_action, "target_actor")
        self.critic_main = Critic(num_state, num_action, "main_critic")
        self.critic_target = Critic(num_state, num_action, "target_critic")

        # Placeholders
        self.target_q = tf.compat.v1.placeholder(tf.float32, [None, 1], name="target_q")
        self.is_weights = tf.compat.v1.placeholder(tf.float32, [None, 1], name="is_weights")

        # Critic loss (PER IS-weights)
        self.td_error = tf.abs(self.target_q - self.critic_main.predict_q)
        critic_loss = tf.reduce_mean(self.is_weights * tf.square(self.td_error))

        # ✅ BN update ops included (main_critic)
        critic_update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, scope="main_critic")
        with tf.control_dependencies(critic_update_ops):
            self.train_critic = tf.compat.v1.train.AdamOptimizer(self.critic_lr).minimize(
                critic_loss, var_list=self.critic_main.trainable_var
            )

        # Actor update (policy gradient)
        action_grad = tf.gradients(self.critic_main.predict_q, self.critic_main.action_size_placeholder)[0]
        policy_grad = tf.gradients(self.actor_main.action, self.actor_main.trainable_var, -action_grad)

        # ✅ BN update ops included (main_actor)
        actor_update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, scope="main_actor")
        with tf.control_dependencies(actor_update_ops):
            self.train_actor = tf.compat.v1.train.AdamOptimizer(self.actor_lr).apply_gradients(
                zip(policy_grad, self.actor_main.trainable_var)
            )

        # Session
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.saver = tf.compat.v1.train.Saver(max_to_keep=20)
        self.Noise = OU_noise(num_action)

        # Target update ops (soft update)
        self.update_target_ops = []
        for i in range(len(self.actor_main.trainable_var)):
            self.update_target_ops.append(
                self.actor_target.trainable_var[i].assign(
                    self.tau * self.actor_main.trainable_var[i] + (1 - self.tau) * self.actor_target.trainable_var[i]
                )
            )
        for i in range(len(self.critic_main.trainable_var)):
            self.update_target_ops.append(
                self.critic_target.trainable_var[i].assign(
                    self.tau * self.critic_main.trainable_var[i] + (1 - self.tau) * self.critic_target.trainable_var[i]
                )
            )

    def act(self, state):
        # eval mode for BN
        action = self.sess.run(self.actor_main.action, feed_dict={
            self.actor_main.state: [state],
            self.actor_main.is_training: False
        })[0]
        noise = self.Noise.sample() * self.epsilon
        return np.clip(action + noise, -1, 1)

    def observe(self, state, action, reward, next_state, done):
        self.memory.store((state, action, reward, next_state, done))
        if self.epsilon > self.MIN_EPSILON:
            self.epsilon *= self.EPSILON_DECAY

    def replay(self):
        if not self.memory.can_sample(self.MEMORY_BATCH_SIZE):
            return

        tree_idx, mini_batch, is_weights = self.memory.sample(self.MEMORY_BATCH_SIZE)
        if len(mini_batch) < self.MEMORY_BATCH_SIZE:
            return

        s_batch = np.array([x[0] for x in mini_batch])
        a_batch = np.array([x[1] for x in mini_batch])
        r_batch = np.array([x[2] for x in mini_batch]).reshape(-1, 1)
        next_s_batch = np.array([x[3] for x in mini_batch])
        done_batch = np.array([x[4] for x in mini_batch]).astype(int).reshape(-1, 1)

        # Target Q
        target_actions = self.sess.run(self.actor_target.action, feed_dict={
            self.actor_target.state: next_s_batch,
            self.actor_target.is_training: False
        })
        target_q_values = self.sess.run(self.critic_target.predict_q, feed_dict={
            self.critic_target.state: next_s_batch,
            self.critic_target.action_size_placeholder: target_actions,
            self.critic_target.is_training: False
        })
        y_target = r_batch + self.GAMMA * (1 - done_batch) * target_q_values

        # Critic train (train mode)
        _, abs_errors = self.sess.run([self.train_critic, self.td_error], feed_dict={
            self.critic_main.state: s_batch,
            self.critic_main.action_size_placeholder: a_batch,
            self.critic_main.is_training: True,
            self.target_q: y_target,
            self.is_weights: is_weights
        })

        # PER priority update
        self.memory.batch_update(tree_idx, abs_errors.flatten())

        # Actor train
        mu_batch = self.sess.run(self.actor_main.action, feed_dict={
            self.actor_main.state: s_batch,
            self.actor_main.is_training: False
        })
        self.sess.run(self.train_actor, feed_dict={
            self.actor_main.state: s_batch,
            self.actor_main.is_training: True,
            self.critic_main.state: s_batch,
            self.critic_main.action_size_placeholder: mu_batch,
            self.critic_main.is_training: False
        })

        # Soft update
        self.sess.run(self.update_target_ops)

    def save_models(self, path, filename):
        if not os.path.exists(path):
            os.makedirs(path)
        save_path = os.path.join(path, filename)
        self.saver.save(self.sess, save_path)
        print("--- Model saved to {} ---".format(save_path))


# ======================
# Globals
# ======================
global episode_R

def get_Envinfo_callback(data):
    global Envinfo, env_action, agent, state, next_state, episode_R, update_start, landingstate

    start_time = time.time()

    Envinfo.rel_x = data.rel_x
    Envinfo.rel_y = data.rel_y
    Envinfo.rel_z = data.rel_z
    Envinfo.rel_pose_xdot = data.rel_pose_xdot
    Envinfo.rel_pose_ydot = data.rel_pose_ydot
    Envinfo.rel_pose_zdot = data.rel_pose_zdot
    Envinfo.episode = data.episode
    Envinfo.step = data.step
    Envinfo.reward = data.reward
    Envinfo.yaw = data.yaw
    Envinfo.yaw_speed = data.yaw_speed
    Envinfo.done = data.done
    Envinfo.reset_error = data.reset_error
    Envinfo.mark_recogn = data.mark_recogn

    if Envinfo.reset_error == False:
        if Envinfo.step != 0:
            next_state = interact()
            agent.observe(state, env_action, Envinfo.reward, next_state, Envinfo.done)
            state = next_state
        else:
            state = interact()

        env_action = agent.act(state)
        update_start = True
        episode_R += Envinfo.reward
    else:
        env_action[0] = 0
        env_action[1] = 0
        env_action[2] = 0
        episode_R = 0
        update_start = False

    print("total_time : ", time.time() - start_time)
    return getEnvinfoResponse(uav_vx=1 * env_action[0], uav_vy=1 * env_action[1], uav_vz=1 * env_action[2])


def interact():
    global Envinfo, done

    MAXUAVPOSE_Z = 8.0
    MINUAVPOSE_Z = 1.25
    camera_fov = 80 * math.pi / 180

    normalized_rel_x = Envinfo.rel_x * (4/3) / (MAXUAVPOSE_Z * math.tan(camera_fov / 2))
    normalized_rel_y = Envinfo.rel_y * (4/3) / (MAXUAVPOSE_Z * math.tan(camera_fov / 2))
    normalized_rel_z = (Envinfo.rel_z - (MAXUAVPOSE_Z - MINUAVPOSE_Z) / 2) / (MAXUAVPOSE_Z - MINUAVPOSE_Z)

    normalized_rel_pose_xdot = Envinfo.rel_pose_xdot / (MAXUAVPOSE_Z * math.tan(camera_fov/2) * 2)
    normalized_rel_pose_ydot = Envinfo.rel_pose_ydot / (MAXUAVPOSE_Z * math.tan(camera_fov/2) * 2)
    normalized_rel_pose_zdot = Envinfo.rel_pose_zdot / (MAXUAVPOSE_Z - MINUAVPOSE_Z)

    normalized_yaw = Envinfo.yaw / math.pi

    state_ = numpy.array((
        normalized_rel_x,
        normalized_rel_y,
        normalized_rel_z,
        normalized_rel_pose_xdot,
        normalized_rel_pose_ydot,
        normalized_rel_pose_zdot,
        normalized_yaw
    ))

    return state_


def main_loop():
    global Envinfo, update_start, env_action, state, next_state, agent, episode_R, landingstate

    base_path = '/home/baek/ddpg_result/ddpg_per'
    start_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = os.path.join(base_path, start_time_str)
    models_dir = os.path.join(session_dir, 'models')

    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
        os.makedirs(models_dir)
        print("Created session directory: {}".format(session_dir))

    output_file_name = os.path.join(session_dir, 'vel_result_output.csv')
    with open(output_file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Step', 'Total_Reward', 'Landing_State'])

    print("Logging to: {}".format(output_file_name))

    rospy.init_node('ddpg_agent', anonymous=True)
    rospy.Service('exchange_state_action', getEnvinfo, get_Envinfo_callback)

    done = 0
    num_state = 7
    num_action = 3

    agent = Agent(num_state, num_action)

    episode_R = 0
    landingstate = 0

    Envinfo = getEnvinfo()
    Envinfo.step = 0
    Envinfo.done = False
    env_action = numpy.zeros((num_action))

    writed = False
    update_start = False
    r = rospy.Rate(30)
    max_reward = -np.inf

    while not rospy.is_shutdown():
        if Envinfo.step != 0 and update_start == True:
            agent.replay()
            update_start = False

        if Envinfo.done and Envinfo.reset_error == False and Envinfo.step > 1:
            if abs(Envinfo.rel_x) <= 0.75 and abs(Envinfo.rel_y) <= 0.75 and abs(Envinfo.rel_z) <= 1:
                landingstate = 1
            else:
                landingstate = 0

            if writed:
                if (Envinfo.episode + 1) % 10 == 0:
                    agent.save_models(models_dir, 'ep_{}.ckpt'.format(Envinfo.episode + 1))

                if episode_R > max_reward:
                    max_reward = episode_R
                    agent.save_models(models_dir, 'best_model.ckpt')
                    print("New Best Reward: {:.4f}! Model updated.".format(max_reward))

                print("-----------------------------")
                print("Total reward:", episode_R)

                with open(output_file_name, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        Envinfo.episode + 1,
                        Envinfo.step,
                        "{:.4f}".format(episode_R),
                        landingstate
                    ])

                print("episode : ", Envinfo.episode)
                print("total step :", Envinfo.step)
                print("-----------------------------")
                episode_R = 0.0
                writed = False

        elif Envinfo.done and Envinfo.reset_error == False and Envinfo.step == 1:
            episode_R = 0
            writed = True
        elif Envinfo.done and Envinfo.reset_error == True:
            episode_R = 0
            writed = True
        else:
            writed = True


if __name__ == '__main__':
    try:
        main_loop()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        print('Interrupted')
        sys.exit(0)
