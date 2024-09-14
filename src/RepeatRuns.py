#!/usr/bin/env python3
import datetime
from TestRun import SingleRun
import rospy
import pickle
import os
from PlotRepeatRuns import PlotRepeatRuns

class RepeatRuns:
    def __init__(self):
        self.repeatTime = 2 
        self.guidanceLaw = None
        self.timeStr = self.getCurrentTimeStr()
        self.folderName = None
        self.fileName = None

    def getCurrentTimeStr(self):
        return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    def run(self):
        for num in range(self.repeatTime):
            sr = SingleRun(runType = 'Repeat')
            self.guidanceLaw = sr.guidanceLawName
            self.folderName = os.path.join(
            sr.packagePath, 
            'data',
            'dataRepeat',
            self.timeStr)
            sr.folderName =  self.folderName
            sr.run()
            self.saveLog(sr)
            del sr
        rospy.signal_shutdown('Shutting down')

    def saveLog(self, sr):
        os.makedirs(self.folderName, exist_ok=True)
        self.fileName = os.path.join(self.folderName, sr.timeStr, 'data.pkl')
        with open(self.fileName, "wb") as file:
            pickle.dump(sr.data, file)

        print(f"Data saved to {self.fileName}")


if __name__ == '__main__':
    rr = RepeatRuns()
    rr.run()
    print(f"rr.folderName = {rr.folderName}")
    pmr = PlotRepeatRuns(baseDir = rr.folderName, guidanceLaw = rr.guidanceLaw, repeatTime = rr.repeatTime)
    pmr.loadData()
    pmr.plotAll()
