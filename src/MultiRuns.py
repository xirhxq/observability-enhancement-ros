#!/usr/bin/env python3
import datetime
from TestRun import SingleRun
import rospy
import pickle
import os
from PlotMultiRuns import PlotMultiRuns

class MultiRuns:
    def __init__(self):
        self.guidanceLaws = ["PN_test", "OEHG_test"]
        self.timeStr = self.getCurrentTimeStr()
        self.folderName = None
        self.fileName = None

    def getCurrentTimeStr(self):
        return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    def run(self):
        for guidanceLaw in self.guidanceLaws:
            sr = SingleRun(runType = 'Multi')
            sr.guidanceLawName = guidanceLaw
            sr.timeStr = self.timeStr
            sr.folderName =  self.folderName = os.path.join(sr.packagePath, 'data', 'dataMulti', self.timeStr)
            sr.run()

            self.saveLog(sr)
            del sr
        rospy.signal_shutdown('Shutting down')

    def saveLog(self, sr):
        os.makedirs(os.path.join(self.folderName, sr.guidanceLawName), exist_ok=True)
        self.fileName = os.path.join(self.folderName, sr.guidanceLawName, 'data.pkl')
        with open(self.fileName, "wb") as file:
            pickle.dump(sr.data, file)

        print(f"Data saved to {self.fileName}")


if __name__ == '__main__':
    mr = MultiRuns()
    mr.run()

    pmr = PlotMultiRuns(baseDir = mr.folderName)
    pmr.findGuidanceLaws()
    pmr.loadData()
    pmr.plotAll()
