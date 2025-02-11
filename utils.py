import configparser
from sumolib import checkBinary
import re
import os
import sys

def import_configuration(config_file):
    content = configparser.ConfigParser()
    content.read(config_file, encoding='utf-8')
    config = {}
    config['gui'] = content['simulation'].getboolean('gui')
    config['total_episode'] = content['simulation'].getint('total_episode')
    config['max_steps'] = content['simulation'].getint('max_steps')
    config['n_cars_generated'] = content['simulation'].getint('n_cars_generated')
    config['yellow_duration'] = content['simulation'].getint('yellow_duration')
    config['green_min'] = content['simulation'].getint('green_min')
    config['training_epochs'] = content['simulation'].getint('training_epochs')
    config['episode_seed'] = content['simulation'].getint('episode_seed')
    config['grids_length'] = content['simulation'].getint('grids_length')
    config['area_length'] = content['simulation'].getint('area_length')

    config['isTrain'] = content['model'].getboolean('isTrain')
    config['batch_size'] = content['model'].getint('batch_size')
    config['learning_rate'] = content['model'].getfloat('learning_rate')
    config['size_max'] = content['model'].getint('memory_size_max')

    config['num_agents'] = content['agent'].getint('num_agents')
    config['state_dim'] = content['agent'].getint('state_dim')
    config['phase_dim'] = content['agent'].getint('phase_dim')
    config['duration_dim'] = content['agent'].getint('duration_dim')
    config['gamma'] = content['agent'].getfloat('gamma')
    config['update_target_episodes'] = content['agent'].getint('update_target_episodes')

    config['models_path'] = content['dir']['models_path']
    config['metrics_path'] = content['dir']['metrics_path']
    config['model_to_test'] = content['dir']['model_to_test']
    config['model_episode'] = content['dir']['model_episode']
    config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']
    return config


def set_sumo(gui, sumocfg_file_name, max_steps):
    """
    Configure various parameters of SUMO
    """
    # sumo things - we need to import python modules from the $SUMO_HOME/tools directory
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # setting the cmd mode or the visual mode
    if gui == False:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # setting the cmd command to run sumo at simulation time
    sumo_cmd = [sumoBinary, "-c", os.path.join('intersection', sumocfg_file_name), "--no-step-log", "true",
                "--waiting-time-memory", str(max_steps)]

    return sumo_cmd


def set_train_path(models_path_name):
    """
    Create a new model path with an incremental integer, also considering previously created model paths
    """
    models_path = os.path.join(os.getcwd(), models_path_name, '')
    os.makedirs(os.path.dirname(models_path), exist_ok=True)

    dir_content = os.listdir(models_path)
    if dir_content:
        previous_versions = [int(name.split("_")[1]) for name in dir_content]
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'

    data_path = os.path.join(models_path, 'experiment_' + new_version, '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return data_path


def set_test_path(models_path_name, experiment_n, model_episode):
    path = os.path.join(os.getcwd(), models_path_name, 'experiment_' + str(experiment_n), '')
    pt_path = os.path.join(path, 'pt', '')
    all_files = os.listdir(pt_path)
    
    pattern = re.compile(rf"trained_model_(J\d+#\d+)_{str(model_episode)}\.pt$")

    result_dict = {}

    for f in all_files:
        match = pattern.match(f)
        if match:
            intersection_id = match.group(1)  # 提取交叉口 ID
            file_path = os.path.join(pt_path, f)  # 获取完整文件路径
            result_dict[intersection_id] = file_path  # 存入字典

    return result_dict


def set_plot_path(metrics_path_name, experiment_n):
    """
    Returns a model path that identifies the model number provided as argument and a newly created 'test' path
    """
    model_folder_path = os.path.join(os.getcwd(), metrics_path_name, 'experiment_' + str(experiment_n), '')

    if os.path.isdir(model_folder_path):
        plot_path = os.path.join(model_folder_path, 'test', '')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        return plot_path
    else:
        sys.exit('The model number specified does not exist in the metrics folder')