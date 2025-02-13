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

        self.val_layer = nn.Linear(128, 1)
        self.adv_layer = nn.Linear(128, action_dim)

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

    def predict(self, state, signal_idx, phase_joint_q):
        if np.random.uniform(0, 1) < self.epsilon:
            phase = np.random.randint(0, self._phase_dim)
            duration = np.random.randint(0, self._duration_dim)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self._device)
            with torch.no_grad():
                phase_q_values = self.phase_q_network(state_tensor)
                duration_q_values = self.duration_q_network(state_tensor)

                phase_joint_q[signal_idx] = phase_q_values[0].numpy()
                new_strategies = self.compute_nash_equilibrium(phase_joint_q)

                phase = torch.argmax(torch.tensor(new_strategies[signal_idx]), dim=0).item()
                duration = torch.argmax(duration_q_values, dim=1).item()
        return phase, duration, phase_q_values[0].numpy()

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

    def compute_nash_equilibrium(self, payoff_matrices, num_iterations=1000, tolerance=1e-6):
        """
        计算 n 个智能体的纳什均衡策略（近似）。
        
        参数：
        - payoff_matrices: 长度为 n 的列表，其中第 i 个元素是智能体 i 的收益矩阵，形状为 (A1, A2, ..., An)。
        - num_iterations: 最大迭代次数。
        - tolerance: 策略变化的容忍度，用于判断收敛。

        返回：
        - strategies: 长度为 n 的列表，其中每个元素是对应智能体的混合策略。
        """
        num_agents = len(payoff_matrices)
        action_spaces = [matrix.shape for matrix in payoff_matrices]

        # 初始化所有智能体的混合策略为均匀分布
        strategies = [np.ones(action_space) / action_space for action_space in action_spaces]

        for iteration in range(num_iterations):
            prev_strategies = [strategy.copy() for strategy in strategies]

            for i in range(num_agents):
                # 构造联合策略
                joint_distribution = strategies[0]
                for strategy in strategies[1:]:
                    joint_distribution = np.tensordot(joint_distribution, strategy, axes=0)

                # 计算当前联合策略下的期望收益
                expected_payoff = np.sum(payoff_matrices[i] * joint_distribution)

                # 计算最佳响应策略
                best_response = np.zeros_like(strategies[i])
                best_action = np.argmax(expected_payoff)
                best_response[best_action] = 1.0

                # 更新智能体 i 的策略为当前策略与最佳响应的加权和
                strategies[i] = 0.5 * strategies[i] + 0.5 * best_response

            # 检查策略是否收敛
            max_change = max(np.max(np.abs(prev_strategy - strategy)) for prev_strategy, strategy in zip(prev_strategies, strategies))
            if max_change < tolerance:
                break

        return strategies

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

    def predict(self, state, signal_idx, phase_joint_q):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self._device)
        with torch.no_grad():
            phase_q_values = self.phase_q_network(state_tensor)
            duration_q_values = self.duration_q_network(state_tensor)

            phase_joint_q[signal_idx] = phase_q_values[0].numpy()
            new_strategies = self.compute_nash_equilibrium(phase_joint_q)

            phase = torch.argmax(torch.tensor(new_strategies[signal_idx]), dim=0).item()
            duration = torch.argmax(duration_q_values, dim=1).item()
        return phase, duration, phase_q_values[0].numpy()
    
    def compute_nash_equilibrium(self, payoff_matrices, num_iterations=1000, tolerance=1e-6):
        """
        计算 n 个智能体的纳什均衡策略（近似）。
        
        参数：
        - payoff_matrices: 长度为 n 的列表，其中第 i 个元素是智能体 i 的收益矩阵，形状为 (A1, A2, ..., An)。
        - num_iterations: 最大迭代次数。
        - tolerance: 策略变化的容忍度，用于判断收敛。

        返回：
        - strategies: 长度为 n 的列表，其中每个元素是对应智能体的混合策略。
        """
        num_agents = len(payoff_matrices)
        action_spaces = [matrix.shape for matrix in payoff_matrices]

        # 初始化所有智能体的混合策略为均匀分布
        strategies = [np.ones(action_space) / action_space for action_space in action_spaces]

        for iteration in range(num_iterations):
            prev_strategies = [strategy.copy() for strategy in strategies]

            for i in range(num_agents):
                # 构造联合策略
                joint_distribution = strategies[0]
                for strategy in strategies[1:]:
                    joint_distribution = np.tensordot(joint_distribution, strategy, axes=0)

                # 计算当前联合策略下的期望收益
                expected_payoff = np.sum(payoff_matrices[i] * joint_distribution)

                # 计算最佳响应策略
                best_response = np.zeros_like(strategies[i])
                best_action = np.argmax(expected_payoff)
                best_response[best_action] = 1.0

                # 更新智能体 i 的策略为当前策略与最佳响应的加权和
                strategies[i] = 0.5 * strategies[i] + 0.5 * best_response

            # 检查策略是否收敛
            max_change = max(np.max(np.abs(prev_strategy - strategy)) for prev_strategy, strategy in zip(prev_strategies, strategies))
            if max_change < tolerance:
                break

        return strategies
    