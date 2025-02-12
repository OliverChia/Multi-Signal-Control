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
    def __init__(self, input_dim, action_dim):
        super(Neural_Network, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)

        self.val_layer = nn.Linear(64, 1)
        self.adv_layer = nn.Linear(64, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        # Dueling Network
        val = self.val_layer(x)
        adv = self.adv_layer(x)
        q_values = val + adv - adv.mean(dim=1, keepdim=True)

        return q_values

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
        self.phase_q_network = Neural_Network(state_dim, phase_dim).to(self._device)
        self.phase_target_network = Neural_Network(state_dim, phase_dim).to(self._device)
        self.phase_target_network.load_state_dict(self.phase_q_network.state_dict())
        self.phase_target_network.eval()

        self.phase_optimizer = optim.Adam(self.phase_q_network.parameters(), lr=lr)
        self.phase_loss = nn.MSELoss()

        self.duration_q_network = Neural_Network(state_dim, duration_dim).to(self._device)
        self.duration_target_network = Neural_Network(state_dim, duration_dim).to(self._device)
        self.duration_target_network.load_state_dict(self.duration_q_network.state_dict())
        self.duration_target_network.eval()

        self.duration_optimizer = optim.Adam(self.duration_q_network.parameters(), lr=lr)
        self.duration_loss = nn.MSELoss()

        self.memory = deque(maxlen=size_max)  # 经验回放池
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_decay = 0.995  # 衰减系数
        self.epsilon_min = 0.1

        # 统计指标
        self._phase_loss_store = []
        self._duration_loss_store = []

    def predict(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            phase = np.random.randint(0, self._phase_dim)
            duration = np.random.randint(0, self._duration_dim)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self._device)
            with torch.no_grad():
                phase_q_values = self.phase_q_network(state_tensor)
                duration_q_values = self.duration_q_network(state_tensor)
                phase = torch.argmax(phase_q_values, dim=1).item()
                duration = torch.argmax(duration_q_values, dim=1).item()
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
        phase_q_values = self.phase_q_network(states)
        duration_q_values = self.duration_q_network(states)
        phase_q_values = phase_q_values.gather(1, phases)
        duration_q_values = duration_q_values.gather(1, durations)

        # 计算目标 Q 值
        with torch.no_grad():
            next_phase_q_values = self.phase_target_network(next_states)
            next_duration_q_values = self.duration_target_network(next_states)
            max_next_phase_q_values = next_phase_q_values.max(dim=1, keepdim=True)[0]
            max_next_duration_q_values = next_duration_q_values.max(dim=1, keepdim=True)[0]
            phase_target_q_values = rewards + self._gamma * max_next_phase_q_values
            duration_target_q_values = rewards + self._gamma * max_next_duration_q_values

        # 计算损失
        phase_loss = self.phase_loss(phase_q_values, phase_target_q_values)
        duration_loss = self.duration_loss(duration_q_values, duration_target_q_values)

        self.phase_optimizer.zero_grad()
        phase_loss.backward()
        self.duration_optimizer.zero_grad()
        duration_loss.backward()

        self.phase_optimizer.step()
        self.duration_optimizer.step()
        self._phase_loss_store.append(phase_loss.item())
        self._duration_loss_store.append(duration_loss.item())

    def sample_processor(self, old_state, old_phase, old_duration, reward, current_state):
        self.memory.append((old_state, old_phase, old_duration, reward, current_state))

    def update_epsilon(self, episode, total_episodes):
        self.epsilon = 1 - episode / total_episodes
        # self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_network(self):
        self.phase_target_network.load_state_dict(self.phase_q_network.state_dict())
        self.duration_target_network.load_state_dict(self.duration_q_network.state_dict())

    def save_model(self, path, episode):
        pt_path = os.path.join(path, 'pt', '')
        os.makedirs(pt_path, exist_ok=True)

        save_path = os.path.join(pt_path, 'trained_model_' + self._signal_id + '_' + str(episode) + '.pt')
        torch.save({
            'phase_q_network_state_dict': self.phase_q_network.state_dict(),
            'duration_q_network_state_dict': self.duration_q_network.state_dict(),
        }, save_path)
    
    @property
    def phase_loss_store(self):
        return self._phase_loss_store
    
    @property
    def duration_loss_store(self):
        return self._duration_loss_store
    
class TestModel(nn.Module):
    def __init__(self, signal_id, model_file_path, state_dim, phase_dim, duration_dim):
        super().__init__()
        self._device = 'cpu'
        self._signal_id = signal_id
        self._state_dim = state_dim
        self._phase_dim = phase_dim
        self._duration_dim = duration_dim
        self.phase_q_network = Neural_Network(state_dim, phase_dim).to(self._device)
        self.duration_q_network = Neural_Network(state_dim, duration_dim).to(self._device)
        self.load_model(model_file_path)

    def load_model(self, model_file_path):
        if os.path.isfile(model_file_path):
            checkpoint = torch.load(model_file_path, map_location=self._device)
            self.phase_q_network.load_state_dict(checkpoint['phase_q_network_state_dict'])
            self.duration_q_network.load_state_dict(checkpoint['duration_q_network_state_dict'])
        else:
            sys.exit("Model number not found")

    def predict(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self._device)
        with torch.no_grad():
            phase_q_values = self.phase_q_network(state_tensor)
            duration_q_values = self.duration_q_network(state_tensor)
            phase = torch.argmax(phase_q_values, dim=1).item()
            duration = torch.argmax(duration_q_values, dim=1).item()
        return phase, duration
    