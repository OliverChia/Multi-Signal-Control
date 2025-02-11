import os
import datetime
import model
from shutil import copyfile
from simulation import Simulation
from generator import TrafficGenerator
from visualization import Visualization
from utils import import_configuration, set_sumo, set_train_path, set_test_path, set_plot_path

if __name__ == '__main__':
    '''
        主函数部分, 用于启动train/test, 并收集数据进行绘图
    '''
    config = import_configuration(config_file='config.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    
    if config['isTrain']:
        models_path = set_train_path(config['models_path'])
        metrics_path = set_train_path(config['metrics_path'])

        # 初始化类
        TrafficGen = TrafficGenerator(
            config['max_steps'],
            config['n_cars_generated']
        )

        Models = [
            model.TrainingModel(
                config["state_dim"],
                config["phase_dim"],
                config["duration_dim"],
                config["batch_size"],
                config["learning_rate"],
                config['gamma'],
                config["size_max"]
            )
            for _ in range(config["num_agents"])
        ]

        Simulation = Simulation(
            Models,
            TrafficGen,
            sumo_cmd,
            config['isTrain'],
            config['training_epochs'],
            config['max_steps'],
            config['yellow_duration'],
            config['green_min'],
            config['grids_length'],
            config['area_length']
        )

        # 每个智能体收敛曲线各绘制一张图
        Curve = Visualization(
            models_path,
            dpi=300
        )

        Metrics = Visualization(
            metrics_path,
            dpi=300
        )

        timestamp_start = datetime.datetime.now()
        episode = 0
        while episode < config['total_episode']:
            print('\n-----Episode', str(episode + 1), 'of', str(config['total_episode']))

            # 更新epsilon
            [agent.update_epsilon(episode, config['total_episode']) for agent in Models]

            # 更新目标网络
            if config['update_target_episodes'] > 0 and episode % config['update_target_episodes'] == 0:
                [agent.update_target_network() for agent in Models]

            # 仿真模拟及训练
            simulation_time, training_time = Simulation.run(episode)
            print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:',
                  round(simulation_time + training_time, 1), 's')
            
            # 数据绘图及模型保存
            if (episode + 1) % 100 == 0:
                [Curve.save_data_and_plot(data=Simulation.reward_store[agent._signal_id], filename=agent._signal_id + '_reward_', xlabel='Episode', ylabel='Cumulative reward', episode=str(episode + 1)) for agent in Models]
                [Curve.save_data_and_plot(data=agent.loss_store, filename=agent._signal_id + '_loss_', xlabel='Episode', ylabel='Cumulative loss', episode=str(episode + 1)) for agent in Models]
                Metrics.save_data_and_plot(data=Simulation.avg_queue_length_store, filename='queue_', xlabel='Episode', ylabel='Average queue length (vehicles)', episode=str(episode + 1))
                Metrics.save_data_and_plot(data=Simulation.avg_travel_time_store, filename='travel_', xlabel='Episode', ylabel='Average travel time (seconds)', episode=str(episode + 1))
                Metrics.save_data_and_plot(data=Simulation.avg_vehicle_speed_store, filename='speed_', xlabel='Episode', ylabel='Average vehicle speed (m/s)', episode=str(episode + 1))

                [agent.save_model(models_path, episode + 1) for agent in Models]
            episode += 1
        
        print("\n----- Start time:", timestamp_start)
        print("----- End time:", datetime.datetime.now())
        print("----- Session info saved at:", models_path)

    else:
        models_path = set_test_path(config['models_path'], config['model_to_test'], config['model_episode'])
        plot_path = set_plot_path(config['metrics_path'], config['model_to_test'])

        # 初始化类
        TrafficGen = TrafficGenerator(
            config['max_steps'],
            config['n_cars_generated']
        )

        Models = [
            model.TestModel(
                signal_id,
                models_path[signal_id],
                config["phase_dim"],
                config["duration_dim"]
            )
            for signal_id in models_path
        ]

        Simulation = Simulation(
            Models,
            TrafficGen,
            sumo_cmd,
            config['isTrain'],
            config['training_epochs'],
            config['max_steps'],
            config['yellow_duration'],
            config['green_min'],
            config['grids_length'],
            config['area_length']
        )

        print('\n----- Test episode')
        simulation_time = Simulation.run(config['episode_seed'])  # run the simulation
        print('Simulation time:', simulation_time, 's')

        print("----- Testing info of action saved at:", plot_path)

        copyfile(src='config.ini', dst=os.path.join(plot_path, 'config.ini'))

        # 绘制图表
        Visualization.save_data_and_plot(data=Simulation.queue_length_episode, filename='queue', xlabel='Step', ylabel='Queue length (vehicles)')
        Visualization.save_data_and_plot(data=Simulation.travel_time_episode, filename='travel', xlabel='Step', ylabel='Travel time (s)')
        Visualization.save_data_and_plot(data=Simulation.vehicle_speed_episode, filename='speed', xlabel='Step', ylabel='Vehicle speed (m/s)')

