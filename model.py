import os
import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from collections import deque

class Neural_Network(nn.Module):
    '''
        神经网络部分，采用多头网络输出、3DQN结构
    '''
    def __init__(self, input_dim, action_A1_dim, action_A2_dim):
        super(Neural_Network, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)

        self.val_layer_A1 = nn.Linear(64, 1)
        self.adv_layer_A1 = nn.Linear(64, action_A1_dim)

        self.val_layer_A2 = nn.Linear(64, 1)
        self.adv_layer_A2 = nn.Linear(64, action_A2_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        # Dueling Network
        val_A1 = self.val_layer_A1(x)
        adv_A1 = self.adv_layer_A1(x)
        q_values_A1 = val_A1 + adv_A1 - adv_A1.mean(dim=1, keepdim=True)

        val_A2 = self.val_layer_A2(x)
        adv_A2 = self.adv_layer_A2(x)
        q_values_A2 = val_A2 + adv_A2 - adv_A2.mean(dim=1, keepdim=True)

        return q_values_A1, q_values_A2

class TrainingModel(object):
    '''
        算法部分
    '''
    def __init__(self, state_dim, phase_dim, duration_dim, batch_size, lr, gamma, size_max):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._signal_id = None
        self._state_dim = state_dim
        self._phase_dim = phase_dim
        self._duration_dim = duration_dim
        self._batch_size = batch_size
        self._lr = lr
        self._gamma = gamma


        # 初始化神经网络
        self.q_network = Neural_Network(state_dim, phase_dim, duration_dim).to(self._device)
        self.target_network = Neural_Network(state_dim, phase_dim, duration_dim).to(self._device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=size_max)  # 经验回放池
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_decay = 0.995  # 衰减系数
        self.epsilon_min = 0.1

        # 统计指标
        self._loss_store = []

    def predict(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            phase = np.random.randint(0, self._phase_dim)
            duration = np.random.randint(0, self._duration_dim)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self._device)
            with torch.no_grad():
                q_values_A1, q_values_A2 = self.q_network(state_tensor)
                phase = torch.argmax(q_values_A1, dim=1).item()
                duration = torch.argmax(q_values_A2, dim=1).item()
        return phase, duration

    def learn(self):
        # 未到达采样阈值不进行训练
        if len(self.memory) < self._batch_size:
            return
        # 从经验回放池中采样，放回采样
        batch = random.choices(self.memory, k=self._batch_size)
        states, phases, durations, rewards, next_states = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self._device)
        phases = torch.tensor(phases, dtype=torch.long).unsqueeze(1).to(self._device)
        durations = torch.tensor(durations, dtype=torch.long).unsqueeze(1).to(self._device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self._device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self._device)
        
        # 计算 Q 值
        q_values_A1, q_values_A2 = self.q_network(states)
        q_values_A1 = q_values_A1.gather(1, phases)
        q_values_A2 = q_values_A2.gather(1, durations)

        # 计算目标 Q 值
        with torch.no_grad():
            next_q_values_A1, next_q_values_A2 = self.target_network(next_states)
            max_next_q_values_A1 = next_q_values_A1.max(dim=1, keepdim=True)[0]
            max_next_q_values_A2 = next_q_values_A2.max(dim=1, keepdim=True)[0]
            target_q_values_A1 = rewards + self._gamma * max_next_q_values_A1
            target_q_values_A2 = rewards + self._gamma * max_next_q_values_A2

        # 计算损失
        loss_A1 = self.loss_fn(q_values_A1, target_q_values_A1)
        loss_A2 = self.loss_fn(q_values_A2, target_q_values_A2)
        loss = loss_A1 + loss_A2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._loss_store.append(loss.item())

    def sample_processor(self, old_state, old_phase, old_duration, reward, current_state):
        self.memory.append((old_state, old_phase, old_duration, reward, current_state))

    def update_epsilon(self, episode, total_episodes):
        self.epsilon = 1 - episode / total_episodes
        # self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, path, episode):
        pt_path = os.path.join(path, 'pt', '')
        os.makedirs(pt_path, exist_ok=True)
        torch.save(self.q_network, os.path.join(pt_path, 'trained_model_' + self._signal_id + '_' + str(episode) + '.pt'))

    def load_model(self, model_folder_path, model_episode, signal_id):
        model_file_path = os.path.join(model_folder_path, 'trained_model_' + signal_id + '_' + str(model_episode) + '.pt')

        if os.path.isfile(model_file_path):
            loaded_model = torch.load(model_file_path, map_location=self._device)
            return loaded_model
        else:
            sys.exit("Model number not found")
    
    @property
    def loss_store(self):
        return self._loss_store