import rosbag
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R

class DataAligner:
    def __init__(self, folder_name, config_A, config_B):
        self.folder_name = folder_name
        self.bag_filename = os.path.join(folder_name, 'align.bag')
        self.topic_A = config_A['topic']
        self.topic_B = config_B['topic']
        self.extract_A = config_A['extract']
        self.extract_B = config_B['extract']
        self.name_A = config_A['name']
        self.name_B = config_B['name']
        self.file_suffix = f'_{self.name_A.replace(" ", "_")}_{self.name_B.replace(" ", "_")}'
        self.data_A = []
        self.data_B = []
        self.dt_values = np.linspace(0.00, 1.00, 100)
        self.std_devs = []
        self.load_data()

    def load_data(self):
        bag = rosbag.Bag(self.bag_filename)
        for topic, msg, t in bag.read_messages(topics=[self.topic_A]):
            value = self.extract_A(msg)
            if value is not None:
                self.data_A.append([t.to_sec(), value])
        for topic, msg, t in bag.read_messages(topics=[self.topic_B]):
            value = self.extract_B(msg)
            if value is not None:
                self.data_B.append([t.to_sec(), value])
        self.data_A = np.array(self.data_A)
        self.data_B = np.array(self.data_B)
        bag.close()

    def calculate_residuals(self):
        residuals = []
        self.std_devs = []
        for dt in self.dt_values:
            times_shifted = self.data_A[:, 0] - dt
            residual = []
            for i, time in enumerate(times_shifted):
                index_B = np.searchsorted(self.data_B[:, 0], time, side='left')
                if index_B < len(self.data_B):
                    residual.append(self.data_A[i, 1] + self.data_B[index_B, 1])
            residuals.append((times_shifted[:len(residual)], residual))
            self.std_devs.append(np.std(residual))
        return residuals

    def plot_residuals(self, residuals):
        plt.figure(figsize=(12, 6))
        
        num_representative_lines = 10
        step = max(1, len(self.dt_values) // num_representative_lines)
        for i in range(0, len(self.dt_values), step):
            times_shifted, residual = residuals[i]
            plt.plot(times_shifted, residual, label=f'dt = {self.dt_values[i]:.2f}')
        
        min_std_dev_index = np.argmin(self.std_devs)
        optimal_dt = self.dt_values[min_std_dev_index]
        optimal_residuals = residuals[min_std_dev_index]
        plt.plot(optimal_residuals[0], optimal_residuals[1], 'r-', linewidth=2, label=f'Optimal dt = {optimal_dt:.2f}')
        plt.xlabel('Shifted Time (s)')
        plt.ylabel('Residuals')
        plt.title('Residuals for Different Time Shifts')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_name, f'residuals_vs_time_shift{self.file_suffix}.png'))
        plt.close()
        print(f'Optimal dt: {optimal_dt:.2f} seconds')

    def plot_angles_combined(self, optimal_dt):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data_A[:, 0], self.data_A[:, 1], '*', label=self.name_A, color='blue')
        plt.plot(self.data_B[:, 0], self.data_B[:, 1], '*', label=self.name_B, color='orange')
        plt.xlabel('Time (s)')
        plt.ylabel('Values')
        plt.title(f'{self.name_A} and {self.name_B} over Time')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_name, f'angles_plot_combined{self.file_suffix}.png'))
        plt.close()

        plt.figure(figsize=(12, 6))
        times_shifted = self.data_A[:, 0] - optimal_dt
        plt.plot(self.data_A[:, 0], self.data_A[:, 1], '*', label=f'{self.name_A} (original)', color='blue', alpha=0.5)
        plt.plot(times_shifted, self.data_A[:, 1], '*', label=f'{self.name_A} (shifted)', color='red')
        plt.plot(self.data_B[:, 0], self.data_B[:, 1], '*', label=self.name_B, color='orange')
        plt.xlabel('Time (s)')
        plt.ylabel('Values')
        plt.title(f'{self.name_A} and {self.name_B} over Time (Shifted by {optimal_dt:.2f}s)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_name, f'angles_plot_combined_delayed{self.file_suffix}.png'))
        plt.close()

    def plot_std_devs(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.dt_values, self.std_devs, 'b-', label='Standard Deviation of Residuals')
        plt.xlabel('dt (s)')
        plt.ylabel('Standard Deviation')
        plt.title('Standard Deviation of Residuals vs. dt')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_name, f'std_dev_vs_dt{self.file_suffix}.png'))
        plt.close()

    def run(self):
        residuals = self.calculate_residuals()
        self.plot_residuals(residuals)
        optimal_dt = self.dt_values[np.argmin(self.std_devs)]
        self.plot_angles_combined(optimal_dt)
        self.plot_std_devs()

folder_list = ['2024-07-02-19-28-30', '2024-07-04-11-08-36', '2024-07-04-14-19-51']

datas = {
    'ca': {
        'name': 'Camera Azimuth',
        'topic': '/spirecv/aruco_detection',
        'extract': lambda msg: msg.fov_x / 2 * msg.targets[0].cx if len(msg.targets) > 0 else None
    },
    'ce': {
        'name': 'Camera Elevation',
        'topic': '/spirecv/aruco_detection',
        'extract': lambda msg: msg.fov_y / 2 * msg.targets[0].cy if len(msg.targets) > 0 else None
    },
    'gy': {
        'name': 'Gimbal Yaw',
        'topic': '/suav/dji_osdk_ros/gimbal_angle',
        'extract': lambda msg: msg.vector.z
    },
    'gp': {
        'name': 'Gimbal Pitch',
        'topic': '/suav/dji_osdk_ros/gimbal_angle',
        'extract': lambda msg: -msg.vector.x
    },
    'ey': {
        'name': 'Euler Yaw',
        'topic': '/suav/dji_osdk_ros/imu',
        'extract': lambda msg: R.from_quat([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]).as_euler('zyx', degrees=True)[0]
    }
}

runs = [
    # {
    #     'folder_name': '2024-07-02-19-28-30',
    #     'configs': [
    #         ('ca', 'gy'),
    #         ('ce', 'gp')
    #     ]
    # },
    # {
    #     'folder_name': '2024-07-04-11-08-36',
    #     'configs': [
    #         ('ca', 'gy'),
    #         ('ce', 'gp')
    #     ]
    # },
    # {
    #     'folder_name': '2024-07-04-14-19-51',
    #     'configs': [
    #         ('ca', 'gy')
    #     ]
    # },
    {
        'folder_name': '2024-07-04-14-48-31',
        'configs': [
            ('ey', 'gy')
        ]
    }
]

for run in runs:
    for config in run['configs']:
        aligner = DataAligner(run['folder_name'], datas[config[0]], datas[config[1]])
        aligner.run()