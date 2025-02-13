import traci
import numpy as np
import math
import timeit

# 交通信号灯的相位定义
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7

class Simulation(object):
    '''
        环境交互部分, 调用learner进行预测, 应用于环境后收集信息用于训练等
    '''
    def __init__(self, Models, TrafficGen, sumo_cmd, isTrain, training_epochs, max_steps, yellow_duration, green_min, grids_length, area_length):
        self._Models = Models
        self._TrafficGen = TrafficGen
        self._sumo_cmd = sumo_cmd
        self._isTrain = isTrain
        self._training_epochs = training_epochs
        self._max_steps = max_steps
        self._yellow_duration = yellow_duration
        self._green_min = green_min

        # 定义状态量维度
        self._lane_num = 16
        self._lane_length = 750
        self._grids_length = grids_length
        self._area_length = area_length

    def run(self, episode):
        # 统计仿真用时
        start_time = timeit.default_timer()
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # 初始化关键要素
        current_step = 0
        signal_ids = traci.trafficlight.getIDList()

        if self._isTrain:
            rl_signals = self._init_agents_id(signal_ids)
        else:
            rl_signals = {agent._signal_id: agent for agent in self._Models}
        rl_signals_idx = {key: index for index, key in enumerate(rl_signals.keys())}

        if episode == 0:
            self._init_statistics(rl_signals)

        # 强化学习生成样本储存的变量
        old_state = {signal_id: [] for signal_id in rl_signals}
        old_phase = {signal_id: 0 for signal_id in rl_signals}
        old_duration = {signal_id: 0 for signal_id in rl_signals}   # 记录强化学习信号灯下次持续时间，防止无车时与固定配时信号灯冲突
        phase_joint_q = [np.ones(agent._phase_dim) / agent._phase_dim for agent in rl_signals.values()]     # 异步决策，按rl_signals中的顺序储存上一步策略

        # 环境交互时的判断变量
        current_phase = {signal_id: 0 for signal_id in signal_ids}
        next_phase = {signal_id: 0 for signal_id in signal_ids}
        next_duration = {signal_id: 0 for signal_id in signal_ids}  # 记录所有信号灯下次持续时间
        next_update_step = {signal_id: 0 for signal_id in signal_ids}

        # 全局统计变量
        self._previous_edge = {}    # 记录每辆车的上一条边，用于清空车辆等待时间
        self._waiting_times = {}    # 统计每辆车的等待时间，车辆切换edge时清空
        self._old_total_wait = {signal_id: 0 for signal_id in rl_signals}
        self._sum_reward = {signal_id: 0 for signal_id in rl_signals}
        self._sum_queue_length = []
        self._sum_travel_time = []
        self._sum_vehicle_speed = []

        # 异步更新信号策略
        while current_step < self._max_steps:
            if (current_step+1) % 300 == 0:
                print(f"-Current step: {current_step+1} -Vehicle Count: {traci.vehicle.getIDCount()} -Total Cost time: {round(timeit.default_timer() - start_time, 1)}s")
            # 各交叉口信号独立更新
            for signal_id in signal_ids:
                if current_step >= next_update_step[signal_id]:
                    # 黄灯相位即将结束，执行上次决策的动作（由于黄灯相位的存在，实际上为延迟动作）
                    if traci.trafficlight.getPhase(signal_id) % 2 != 0:
                        self._set_green(signal_id, next_phase[signal_id])
                        current_phase[signal_id] = next_phase[signal_id]
                        next_update_step[signal_id] = current_step + next_duration[signal_id] + self._green_min
                    else:
                        # 固定配时信号交叉口/强化学习交叉口内无车辆
                        if signal_id not in rl_signals: # or self._veh_not_exist: 无车辆时可能非强化学习给出动作会被记录到样本中，或两次强化学习决策间隔一次非强化学习决策
                            next_phase[signal_id] = (current_phase[signal_id] + 1) % 4
                            next_duration[signal_id] = 10
                            self._set_yellow(signal_id, current_phase[signal_id])
                            next_update_step[signal_id] = current_step + self._yellow_duration
                            continue

                        # 强化学习信号交叉口
                        agent = rl_signals[signal_id]
                        current_state = self._get_state(signal_id)
                        reward = self._get_reward(signal_id)

                        # 储存样本
                        if current_step != 0 and self._isTrain:
                            agent.sample_processor(
                                old_state[signal_id], old_phase[signal_id], old_duration[signal_id], reward, current_state)

                        signal_idx = rl_signals_idx[signal_id]
                        next_phase[signal_id], next_duration[signal_id], single_q = agent.predict(current_state, signal_idx, phase_joint_q)
                        phase_joint_q[signal_idx] = single_q
                        old_phase[signal_id] = next_phase[signal_id]
                        old_duration[signal_id] = next_duration[signal_id]
                        old_state[signal_id] = current_state

                        # 当前相位与决策出的相位不同时，插入固定时长的黄灯相位
                        if current_phase[signal_id] != next_phase[signal_id]:
                            self._set_yellow(signal_id, current_phase[signal_id])
                            next_update_step[signal_id] = current_step + self._yellow_duration
                        else: # 连续两次决策相位相同，不插入黄灯相位，直接延长该相位持续时间
                            next_update_step[signal_id] = current_step + next_duration[signal_id] + self._green_min
                        
                        # 统计累加奖励
                        if reward < 0:
                            self._sum_reward[signal_id] += reward

            # 每帧需要更新的内容
            self._update_waiting_times()
            self._update_statistics(signal_ids)
            traci.simulationStep()
            current_step += 1

        # 训练模式下进行模型训练，并保存每个episode的评价指标
        if self._isTrain:
            for agent in self._Models:
                signal_id = agent._signal_id
                print("Signal id:", signal_id,"Total reward:", self._sum_reward[signal_id], "- Epsilon:", round(agent.epsilon, 2))
            traci.close()
            simulation_time = round(timeit.default_timer() - start_time, 1)

            print("Training...")
            start_time = timeit.default_timer()
            for agent in self._Models:
                for _ in range(self._training_epochs):
                    agent.learn()
            for signal_id in rl_signals:
                self._save_reward_store(signal_id)
            self._save_statistics()

            training_time = round(timeit.default_timer() - start_time, 1)
            return simulation_time, training_time
        else:
            traci.close()
            simulation_time = round(timeit.default_timer() - start_time, 1)
            return simulation_time

    def _get_state(self, signal_id):
        position = np.zeros([self._lane_num, round(self._area_length / self._grids_length)])
        speed = np.zeros([self._lane_num, round(self._area_length / self._grids_length)])
        accel = np.zeros([self._lane_num, round(self._area_length / self._grids_length)])

        in_lanes = self._get_in_lanes(signal_id)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = self._lane_length - lane_pos

            if lane_id in in_lanes:
                lane_group = in_lanes[lane_id]
            else:
                continue
            lane_cell = math.floor(lane_pos / self._grids_length)
            if lane_cell < self._area_length / self._grids_length:
                position[lane_group][lane_cell] = 1
                speed[lane_group][lane_cell] = traci.vehicle.getSpeed(car_id)
                accel[lane_group][lane_cell] = traci.vehicle.getAcceleration(car_id)

        speed = speed / (np.max(speed) + 1e-7)
        accel = accel / (np.max(accel) + 1e-7)
        
        # state = np.concatenate(position, axis=0)
        state = np.stack([position, speed, accel], axis=0).flatten()
        return state

    def _get_reward(self, signal_id):
        in_lanes = self._get_in_lanes(signal_id)
        current_total_wait = self._collect_waiting_times(in_lanes)

        reward = self._old_total_wait[signal_id] - current_total_wait
        self._old_total_wait[signal_id] = current_total_wait
        return reward

    def _collect_waiting_times(self, in_lanes):
        total_waiting_time = 0
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            if traci.vehicle.getLaneID(car_id) in in_lanes:
                if car_id not in self._waiting_times:
                    continue
                elif len(self._waiting_times[car_id]) == 1:
                    wait_time = self._waiting_times[car_id][0]
                else:
                    wait_time = self._waiting_times[car_id][-1] - self._waiting_times[car_id][-2]
                total_waiting_time += wait_time
        return total_waiting_time

    def _update_waiting_times(self):
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            edge_id = traci.vehicle.getRoadID(car_id)
            if edge_id == "" or edge_id.startswith(":"):
                # 车辆位于交叉口或已到达目的地
                continue
            if car_id not in self._waiting_times:
                # 车辆第一次出现时，初始化等待时间及车辆所在的edge
                self._waiting_times[car_id] = []
                self._waiting_times[car_id].append(wait_time)
                self._previous_edge[car_id] = edge_id
            elif self._previous_edge[car_id] != edge_id:
                # 车辆切换edge时，记录在新edge上的总等待时间
                self._waiting_times[car_id].append(wait_time)
                self._previous_edge[car_id] = edge_id
            else:   # 未切换edge时，更新最新的总等待时间
                self._waiting_times[car_id][-1] = wait_time
            
    def _set_green(self, signal_id, action_number):
        if action_number == 0:
            traci.trafficlight.setPhase(signal_id, PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase(signal_id, PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase(signal_id, PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase(signal_id, PHASE_EWL_GREEN)

    def _set_yellow(self, signal_id, old_action):
        yellow_phase_code = old_action * 2 + 1
        traci.trafficlight.setPhase(signal_id, yellow_phase_code)

    def _veh_not_exist(self, signal_id):
        lane_idx = set(traci.trafficlight.getControlledLanes(signal_id))
        for lane_id in lane_idx:
            #if traci.edge.getTo(traci.lane.getEdgeID(lane_id)) == signal_id:
            if traci.lane.getLastStepVehicleNumber(lane_id) > 0:
                return False
        return True

    def _init_agents_id(self, signal_ids):
        np.random.seed(0)   # 固定随机种子，控制由RL控制的交叉路口位置，保证实验可复现
        sampled_ids = np.random.choice(signal_ids, len(self._Models), replace=False)    # 不放回抽取
        for agent, signal_id in zip(self._Models, sampled_ids):
            agent._signal_id = signal_id
        np.random.seed()    # 恢复随机种子
        return {agent._signal_id: agent for agent in self._Models}
    
    def _get_in_lanes(self, signal_id):
        in_lanes = {}
        cnt = 0
        lane_idx = set(traci.trafficlight.getControlledLanes(signal_id))
        for lane_id in lane_idx:
            in_lanes[lane_id] = cnt
            cnt += 1
        return in_lanes

    def _init_statistics(self, rl_signals):
        # 全局统计变量，储存每个episode的训练指标
        self._reward_store = {signal_id: [] for signal_id in rl_signals}    # reward收敛数据统计范围为单独路口
        self._avg_queue_length_store = []   # 评价指标统计范围为整个路网
        self._avg_travel_time_store = []
        self._avg_vehicle_speed_store = []

    def _update_statistics(self, signal_ids):
        # 记录每个episode内训练指标随step变化的数据
        queue_length = 0
        travel_time = 0
        vehicle_speed = 0
        cnt = 0
        for signal_id in signal_ids:
            lane_idx = set(traci.trafficlight.getControlledLanes(signal_id))
            for lane_id in lane_idx:
                queue_length += traci.lane.getLastStepHaltingNumber(lane_id)
                travel_time += traci.lane.getTraveltime(lane_id)
                vehicle_speed += traci.lane.getLastStepMeanSpeed(lane_id)
                cnt += 1
        self._sum_queue_length.append(queue_length)
        self._sum_travel_time.append(travel_time/cnt)
        self._sum_vehicle_speed.append(vehicle_speed/cnt)

    def _save_reward_store(self, signal_id):
        # 保存每个episode的total_reward
        self._reward_store[signal_id].append(self._sum_reward[signal_id])

    def _save_statistics(self):
        # 保存每个episode的评价指标
        self._avg_queue_length_store.append(sum(self._sum_queue_length) / self._max_steps)
        self._avg_travel_time_store.append(sum(self._sum_travel_time) / self._max_steps)
        self._avg_vehicle_speed_store.append(sum(self._sum_vehicle_speed) / self._max_steps)

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store
    
    @property
    def avg_travel_time_store(self):
        return self._avg_travel_time_store
    
    @property
    def avg_vehicle_speed_store(self):
        return self._avg_vehicle_speed_store
    
    @property
    def queue_length_episode(self):
        return self._sum_queue_length
    
    @property
    def travel_time_episode(self):
        return self._sum_travel_time
    
    @property
    def vehicle_speed_episode(self):
        return self._sum_vehicle_speed
