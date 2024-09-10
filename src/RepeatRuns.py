import datetime
from TestRun import SingleRun
import rospy
import pickle
import os
from PlotRepeatRuns import PlotRepeatRuns

class RepeatRuns:
    def __init__(self):
        self.repeatTime = 5 
        self.guidanceLaw = None
        self.timeStr = self.getCurrentTimeStr()
        self.folderName = None
        self.fileName = None

    def getCurrentTimeStr(self):
        return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    def run(self):
        for num in range(self.repeatTime):
            sr = SingleRun()
            self.guidanceLaw = sr.guidanceLawName
            sr.folderName =  self.folderName = os.path.join(
            sr.packagePath, 
            'dataRepeat'
        )
            sr.run()
            rospy.signal_shutdown('Shutting down')
            sr.spinThread.join()

            self.saveLog(sr)
            del sr

    def saveLog(self, sr):
        os.makedirs(os.path.join(self.folderName, self.timeStr), exist_ok=True)
        self.fileName = os.path.join(self.folderName, self.timeStr, sr.timeStr, 'data.pkl')
        with open(self.fileName, "wb") as file:
            pickle.dump(sr.data, file)

        print(f"Data saved to {self.fileName}")


if __name__ == '__main__':
    rr = RepeatRuns()
    rr.run()

    pmr = PlotRepeatRuns(baseDir = rr.folderName, guidanceLaw = rr.guidanceLaw)
    pmr.loadData()
    pmr.plotAll()
