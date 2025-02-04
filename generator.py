import os
import xml.etree.ElementTree as ET
import numpy as np
import math
import subprocess

class TrafficGenerator(object):
    '''
        生成器部分, 用于生成车流数据
    '''
    def __init__(self, max_steps, n_cars_generated):
        self.simulation_time = max_steps  # 仿真时长（秒）
        self.vehicle_count = n_cars_generated     # 车辆总数
        self.shape_factor = 2.0       # 威布尔分布形状参数
        self.scale_factor = 1200      # 威布尔分布尺度参数（调整流量峰值时间）

    def generate_routefile(self, seed):
        # 使生成的路由文件可复现
        np.random.seed(seed)

        # 读取 SUMO 路网文件
        net_file = "intersection/environment.net.xml"
        tree = ET.parse(net_file)
        root = tree.getroot()

        # 找到所有外部边界道路（可以作为起点和终点的道路）
        in_edges = [edge.get("id") for edge in root.findall("edge") if edge.get("type") == "in"]
        out_edges = [edge.get("id") for edge in root.findall("edge") if edge.get("type") == "out"]

        # 生成车辆出发时间，按照威布尔分布
        timings = np.random.weibull(self.shape_factor, self.vehicle_count) * self.scale_factor
        timings.sort()  # 确保时间有序
        car_gen_steps = []  # 将车辆出发时间限制在仿真时间内
        min_old = math.floor(timings[0])    #向下取整
        max_old = math.ceil(timings[-1])    #向上取整
        min_new = 0
        max_new = self.simulation_time
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((value - min_old) / (max_old - min_old) * (max_new - min_new) + min_new))
        car_gen_steps = np.rint(car_gen_steps)

        # 生成车辆路径
        vehicles = []
        for i in range(self.vehicle_count):
            # 随机选择起点和终点
            start_edge = np.random.choice(in_edges)
            end_edge = np.random.choice(out_edges)

            # 生成车辆信息
            vehicle = {
                "id": f"veh_{i}",
                "depart": car_gen_steps[i],
                "from": start_edge,
                "to": end_edge,
            }
            vehicles.append(vehicle)

        # 生成 .trips.xml 文件
        output_dir = "intersection"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "episode_trips.trips.xml")

        with open(output_file, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<routes>\n')

            # 定义车辆类型
            f.write('    <vType accel="1.0" decel="4.5" id="standard_car" length="5" minGap="2.5" maxSpeed="25" sigma="0.5"/>\n')

            # 生成路线
            for vehicle in vehicles:
                f.write(f'    <trip id="{vehicle["id"]}" depart="{vehicle["depart"]}" from="{vehicle["from"]}" to="{vehicle["to"]}" />\n')


            f.write('</routes>\n')
        
        # 将 .trips.xml 文件转换为 .rou.xml 文件
        sumo_cmd = f"duarouter --route-files {output_dir}\episode_trips.trips.xml --net-file {output_dir}\environment.net.xml --output-file {output_dir}\episode_routes.rou.xml"
        subprocess.run(sumo_cmd, shell=True)

if __name__ == "__main__":
    generator = TrafficGenerator(max_steps=3600, n_cars_generated=2000)
    generator.generate_routefile(seed=0)