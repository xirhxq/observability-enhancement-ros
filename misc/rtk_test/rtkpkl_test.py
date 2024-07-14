import pickle
import numpy as np
import matplotlib.pyplot as plt

def localOffsetFromGpsOffset(target, origin):
    # la lo al
    deltaLat = target[0] - origin[0]
    deltaLon = target[1] - origin[1]
    C_EARTH = 6378137.0
    deltaENU = [np.deg2rad(deltaLon) * C_EARTH * np.cos(np.deg2rad(target[0])), 
                np.deg2rad(deltaLat) * C_EARTH, 
                target[2] - origin[2]]
    return deltaENU

def main():
    with open('data.pkl', 'rb') as file:
        data = pickle.load(file)

    for key, value in data.items():
        if key == 'origin':
            origin = value
        elif key == 'localPosition':
            localPosition = value
        elif key == 'gpsPosition':
            gpsPosition = value
        elif key == 'rtkPosition':
            rtkPosition = value

    gps2locPos = []
    rtk2locPos = []

    for i in range(len(gpsPosition)):
        gps2locPos.append(localOffsetFromGpsOffset(gpsPosition[i], origin))
    for i in range(len(rtkPosition)):
        rtk2locPos.append(localOffsetFromGpsOffset(rtkPosition[i], origin))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(np.array(localPosition)[0, :], np.array(localPosition)[1, :], np.array(localPosition)[2, :], color='r', label='locPos')
    ax.plot(np.array(gps2locPos)[0, :], np.array(gps2locPos)[1, :], np.array(gps2locPos)[2, :], color='g', label='GPS')
    ax.plot(np.array(rtk2locPos)[0, :], np.array(rtk2locPos)[1, :], np.array(rtk2locPos)[2, :], color='b', label='RTK')
    ax.set_xlabel('$X$ (m)')
    ax.set_ylabel('$Y$ (m)')
    ax.set_zlabel('$Z$ (m)')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()