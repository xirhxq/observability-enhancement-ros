import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot3dTrajectoryENU():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if allTargetPositions:
        firstTargetPosition = np.array(allTargetPositions[0])
        ax.plot(firstTargetPosition[:, 0], firstTargetPosition[:, 1], firstTargetPosition[:, 2], 
                marker='*', linestyle='none', color='red', label='Target Position')
        
    for testNum, mePositions in enumerate(allMePositions):
        mePositions = np.array(mePositions)
        ax.plot(mePositions[:, 0], mePositions[:, 1], mePositions[:, 2], linewidth=2.0,
                label=f'test {testNum + 1}')
        
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_box_aspect([1,1,1])
    ax.view_init(elev=30, azim=180)
    ax.set_title('ENU position')
    ax.legend(loc='upper right')
    
    output_image_path = os.path.join(root_dir, '3d_trajectory.png')
    plt.savefig(output_image_path, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D trajectory plot at {output_image_path}")

def plotTargetPositionErrorENU():
    fig, ax = plt.subplots(3, 1, figsize=(8, 10))
    for testNum, targetPosition in enumerate(allTargetPositions):
        time = np.array(allTimes[testNum])
        ekfState = np.array(allEkfStates[testNum])
        targetPosition = np.array(targetPosition)
        targetPositionError = np.abs(ekfState[:, :3] - targetPosition)
        ax[0].plot(time[:], targetPositionError[:, 0], linewidth=2.0,
                label=f'test {testNum + 1}')
        ax[1].plot(time[:], targetPositionError[:, 1], linewidth=2.0,
                label=f'test {testNum + 1}')
        ax[2].plot(time[:], targetPositionError[:, 2], linewidth=2.0,
                label=f'test {testNum + 1}')
    
    for num,direction in enumerate(['East position', 'North position', 'Up position']):
        ax[num].set_xlabel('Time (s)')
        ax[num].set_ylabel(direction +' estimated error (m)')
        ax[num].legend(loc='upper right')
    
    ax[0].set_title('ENU target position estimated error')
    output_image_path = os.path.join(root_dir, 'target_position_error.png')
    plt.savefig(output_image_path, bbox_inches='tight')
    plt.close()
    print(f"Saved target_position_error plot at {output_image_path}")

def plotTargetVelocityErrorENU():
    fig, ax = plt.subplots(3, 1, figsize=(8, 10))
    for testNum, targetVelocity in enumerate(allTargetVelocitys):
        time = np.array(allTimes[testNum])
        ekfState = np.array(allEkfStates[testNum])
        targetVelocity = np.array(targetVelocity)
        targettargetVelocityError = np.abs(ekfState[:, 3:] - targetVelocity)
        ax[0].plot(time[:], targettargetVelocityError[:, 0], linewidth=2.0,
                label=f'test {testNum + 1}')
        ax[1].plot(time[:], targettargetVelocityError[:, 1], linewidth=2.0,
                label=f'test {testNum + 1}')
        ax[2].plot(time[:], targettargetVelocityError[:, 2], linewidth=2.0,
                label=f'test {testNum + 1}')
    
    for num,direction in enumerate(['East velocity', 'North velocity', 'Up velocity']):
        ax[num].set_xlabel('Time (s)')
        ax[num].set_ylabel(direction +' estimated error (m/s)')
        ax[num].legend(loc='upper right')
    
    ax[0].set_title('ENU target velocity estimated error')
    output_image_path = os.path.join(root_dir, 'target_velocity_error.png')
    plt.savefig(output_image_path, bbox_inches='tight')
    plt.close()
    print(f"Saved target_velocity_error plot at {output_image_path}")

def plotMeasurementUseAndLosAngle():
    fig, ax = plt.subplots(2, 1, figsize=(8, 10))
    for testNum, targetPosition in enumerate(allTargetPositions):
        time = np.array(allTimes[testNum])
        targetPosition = np.array(targetPosition)
        mePosition = np.array(allMePositions[testNum])
        measurementUse = np.array(allMeasurementUse[testNum])
        timeDelay = np.array(allTimeDelay[testNum])
        timeStep = np.array(allTimeStep[testNum])
        num_data_points = len(mePosition)
        relativePosition = np.zeros((3, num_data_points))
        losAngle = np.zeros((2, num_data_points))
        for i in range(num_data_points):
            relativePosition[:, i] = targetPosition[i, :] - mePosition[i, :]
            losAngle[:, i] = np.array([
                np.arctan2(relativePosition[2, i], np.sqrt(relativePosition[0, i] ** 2 + relativePosition[1, i] ** 2)),
                np.arctan2(relativePosition[1, i], relativePosition[0, i])
            ])
        for num, direction in enumerate(['elevation angle', 'azimuth angle']):
            ax[num].plot(time[int(timeDelay / timeStep):], np.rad2deg(measurementUse[:, num]), color = colorsUse[testNum], linestyle='-', linewidth=2.0,
                        label=f'test {testNum + 1}:'+'Measurement of ' + direction)
            ax[num].plot(time, np.rad2deg(losAngle[num, :]), color = colorsUse[testNum], linestyle='--', linewidth=2.0,
                        label=f'test {testNum + 1}:'+'Real ' + direction)
            ax[num].set_xlabel('Time (s)')
            ax[num].set_ylabel('Angle (deg)')
            ax[num].legend(loc='lower left')

    output_image_path = os.path.join(root_dir, 'measurement_use_and_los_angle.png')
    plt.savefig(output_image_path, bbox_inches='tight')
    plt.close()
    print(f"Saved target_position_error plot at {output_image_path}")

script_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = script_dir

allTimes = []
allTimeDelay = []
allTimeStep = []
allMePositions = []
allTargetPositions = []
allTargetVelocitys = []
allEkfStates = []
allMeasurementUse = []
colorsUse = plt.rcParams['axes.prop_cycle'].by_key()['color']
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file == 'data.pkl':
            file_path = os.path.join(subdir, file)
            with open(file_path, 'rb') as f:
                data_list = pickle.load(f)
                times = []
                mePositions = []
                targetPositions = []
                targetVelocitys = []
                ekfStates = []
                measurementUse = []
                timeDelay = []
                timeStep = []
                for data in data_list:
                    times.append(data['t'])
                    timeDelay.append(data['timeDelay'])
                    timeStep.append(data['tStep'])
                    mePositions.append(data['mePositionENU'])
                    targetPositions.append(data['targetPosition'])
                    targetVelocitys.append(data['targetVelocity'])
                    ekfStates.append(data['ekfState'])
                    if not len(data['measurementUse']) == 0:
                        if data['measurementUse'][1] < 0:
                            data['measurementUse'][1] += 2 * np.pi
                        measurementUse.append(data['measurementUse'])
                allTimes.append(times)
                allTimeDelay.append(timeDelay[0])
                allTimeStep.append(timeStep[0])
                allMePositions.append(mePositions)
                allTargetPositions.append(targetPositions)
                allTargetVelocitys.append(targetVelocitys)
                allEkfStates.append(ekfStates)
                allMeasurementUse.append(measurementUse)

plotMeasurementUseAndLosAngle()    
plot3dTrajectoryENU()
plotTargetPositionErrorENU()
plotTargetVelocityErrorENU()

