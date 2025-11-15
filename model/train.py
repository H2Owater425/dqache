# %%
from numpy import empty
from numpy.random import choice

class ReplayBuffer:
	def __init__(self, capacity):
		self.buffer = empty(capacity, object)
		self.capacity = capacity
		self.position = 0
		self.size = 0

	def append(self, state, reward, next_state):
		self.buffer[self.position] = (state, reward, next_state)
		self.position = (self.position + 1) % self.capacity

		if self.size < self.capacity:
			self.size += 1

	def sample(self, batch_size):
		return self.buffer[choice(self.size, batch_size, False)]

# %%
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.activations import leaky_relu, linear
from keras.losses import mean_squared_error
from numpy import array, zeros, vstack

class DeepQNetworkAgent:
	@staticmethod
	def create_model(feature_count, learning_rate):
		model = Sequential([
			Input((feature_count,)),
			Dense(64, leaky_relu),
			Dense(32, leaky_relu),
			Dense(1, linear),
		])
		
		model.compile(Adam(learning_rate), mean_squared_error)

		return model

	def __init__(self, feature_count, learning_rate, gamma, buffer_size, batch_size):
		self.feature_count = feature_count
		self.gamma = gamma
		self.batch_size = batch_size

		self.policy_model = self.create_model(feature_count, learning_rate)
		self.target_model = self.create_model(feature_count, learning_rate)

		self.sync_target_model()

		self.replay_buffer = ReplayBuffer(buffer_size)

	def sync_target_model(self):
		self.target_model.set_weights(self.policy_model.get_weights())

	def get_scores(self, features):
		return self.policy_model(features, training=False).numpy().flatten()

	def store_experience(self, *arguments):
		self.replay_buffer.append(*arguments)

	def train_from_replay(self):
		if self.replay_buffer.size < self.batch_size:
			return
		
		states, rewards, next_states = zip(*self.replay_buffer.sample(self.batch_size))

		non_terminal_mask = array([s is not None for s in next_states])
		non_terminal_next_states = vstack([s for s in next_states if s is not None])

		target_q_values = zeros(self.batch_size)
		
		if non_terminal_next_states.shape[0] > 0:
			target_q_values[non_terminal_mask] = self.target_model(non_terminal_next_states, training=False).numpy().flatten()

		self.policy_model.train_on_batch(vstack(states), array(rewards) + (self.gamma * target_q_values))

# %%
from numpy import log1p, argmin

class Environment:
	def __init__(self, capacity, agent, data):
		self.capacity = capacity
		self.agent = agent
		self.data = data.to_dict('records')
		
		# {id: [size, last_access_time, frequency]}
		self.caches = {}
		self.current_time = 0
		
		self.hit_count = 0
		self.miss_count = 0

	def get_features(self, id):
		return array([log1p(self.current_time - self.caches[id][1] if self.current_time >= self.caches[id][1] else 0), log1p(self.caches[id][2]), log1p(self.caches[id][0]), log1p(self.capacity)])

	def iterate(self):
		for row in self.data:
			self.current_time = row['c_time']

			if row['filename'] in self.caches:
				self.hit_count += 1
				previous_features = self.get_features(row['filename'])

				self.caches[row['filename']][1] = self.current_time
				self.caches[row['filename']][2] += 1
				
				if row['op_type'] == 'WRITE':
					self.caches[row['filename']][0] = row['request_io_size_bytes']

				self.agent.store_experience(previous_features, 1, self.get_features(row['filename']))

				yield 1
				continue

			self.miss_count += 1

			if len(self.caches) >= self.capacity:
				ids = list(self.caches.keys())
				features = array([self.get_features(id) for id in ids])
				deleted_index = argmin(self.agent.get_scores(features))

				del self.caches[ids[deleted_index]]
				
				self.agent.store_experience(features[deleted_index], 0, None)

			self.caches[row['filename']] = [row['request_io_size_bytes'], self.current_time, 1]

			yield 0

# %%
from numpy import array_split
from pandas import read_csv

def load_datas(path, count):
	data = read_csv(path)

	data.dropna(inplace=True)

	data = data[data['request_io_size_bytes'] != 0][['filename', 'c_time', 'op_type', 'request_io_size_bytes']]

	return [data.iloc[indices] for indices in array_split(range(len(data)), count)]

# %%
from time import time
from math import trunc

def unix_epoch():
	return trunc(time())

# %%
FEATURE_COUNT = 4
LEARNING_RATE = 0.001
GAMMA = 0.95
REPLAY_BUFFER_SIZE = 1048576
BATCH_SIZE = 128

MIN_CACHE_CAPACITY = 64
MAX_CACHE_CAPACITY = 256

TARGET_UPDATE_FREQUENCY = 16384
SPLIT_COUNT = 32

FOLDER_COUNT = 32
FILE_COUNT = 16
CHUNK_COUNT = 8

# %%
from os import listdir
from posixpath import join
from random import randint
from matplotlib.pyplot import subplots, close
from collections import deque
from random import sample

agent = DeepQNetworkAgent(
	feature_count=FEATURE_COUNT,
	learning_rate=LEARNING_RATE,
	gamma=GAMMA,
	buffer_size=REPLAY_BUFFER_SIZE,
	batch_size=BATCH_SIZE
)

training_step_counter = 1
best_hit_score = -1.0

history_hit_rates = deque(maxlen=32)
history_hit_scores = deque(maxlen=32)
history_hit_counts = deque(maxlen=32)
history_miss_counts = deque(maxlen=32)
history_capacities = deque(maxlen=32)

with open(f'logs/{unix_epoch()}.log', 'w') as output:
	for i, folder in enumerate(sample(listdir('data'), FOLDER_COUNT)):
		for j, file in enumerate(sample(listdir(join('data', folder)), FILE_COUNT)):
			for k, data in enumerate(sample(load_datas(join('data', folder, file), SPLIT_COUNT), CHUNK_COUNT)):
				capacity = randint(MIN_CACHE_CAPACITY, MAX_CACHE_CAPACITY)
				environment = Environment(capacity, agent, data)

				output.write(f'chunk {k + 1}/{CHUNK_COUNT} in {file} {j + 1}/{FILE_COUNT} in {folder} {i + 1}/{FOLDER_COUNT} (capacity: {capacity})\n')

				for _ in environment.iterate():
					if agent.replay_buffer.size > BATCH_SIZE:
						agent.train_from_replay()
						training_step_counter += 1

						if training_step_counter == TARGET_UPDATE_FREQUENCY:
							training_step_counter = 1
							now = unix_epoch()

							agent.sync_target_model()
							agent.policy_model.save(f'saves/{now}.keras')

							output.write(f"saved (period) at {now}\n")

				total_count = environment.hit_count + environment.miss_count
				now = unix_epoch()

				if total_count > 0:
					hit_rate = environment.hit_count / total_count * 100
					hit_score = hit_rate / log1p(capacity)

					output.write(f"finished at {now}\nhit count: {environment.hit_count}\nmiss count: {environment.miss_count}\nhit rate: {hit_rate:.2f}%\nhit score: {hit_score:.4f}\n")

					history_hit_rates.append(hit_rate)
					history_hit_scores.append(hit_score)
					history_hit_counts.append(environment.hit_count)
					history_miss_counts.append(environment.miss_count)
					history_capacities.append(capacity)

					if hit_score > best_hit_score:
						best_hit_score = hit_score
						agent.policy_model.save(f'saves/{now}.keras')
						
						output.write("saved (best)\n")

				output.write(f"best hit score: {best_hit_score:.4f}\n")
				output.flush()

				if (j + 1) % 4 == 0 and k + 1 == CHUNK_COUNT:
					fig, (ax1, ax2) = subplots(2, 1, figsize=(6, 5), sharex=True)

					chunks = range(1, len(history_hit_rates) + 1)

					color = 'tab:blue'
					ax1.set_xlabel('chunk')
					ax1.set_ylabel('hit score', color=color)
					ax1.plot(chunks, history_hit_scores, color=color, marker='o', linestyle='-', label='hit score')
					ax1.tick_params(axis='y', labelcolor=color)
					ax1.set_title('hit score, capacity')
					ax1.grid(True)
					
					ax1b = ax1.twinx()
					color = 'tab:green'
					ax1b.set_ylabel('capacity', color=color)
					ax1b.plot(chunks, history_capacities, color=color, linestyle='--', marker='x', label='capacity')
					ax1b.tick_params(axis='y', labelcolor=color)

					ax2.set_xlabel('chunk')
					ax2.set_ylabel('count')
					ax2.plot(chunks, history_hit_counts, color='tab:green', marker='o', label='hit')
					ax2.plot(chunks, history_miss_counts, color='tab:red', marker='o', label='miss')
					ax2.set_title('hit / miss count, hit rate')
					ax2.grid(True)
					ax2.legend()

					ax2b = ax2.twinx()
					color = 'tab:blue'
					ax2b.set_ylabel('hit rate %', color=color)
					ax2b.plot(chunks, history_hit_rates, color=color, linestyle='--', marker='x')
					ax2b.tick_params(axis='y', labelcolor=color)

					fig.suptitle(f'{now}')
					fig.tight_layout()
					fig.savefig(f'figures/{now}.svg')

					close(fig)

	output.write('saved (last)\n')
	agent.policy_model.save(f'saves/{unix_epoch()}.keras')