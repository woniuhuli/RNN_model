from RNN_model import *
from ReadTrainData import *
import time


def main():
    t0 = time.time()

    trainPath = 'TrainData_2015.1.1_2015.2.19.txt'
    testPath = 'input_5flavors_cpu_7days.txt'
    trainData = ReadTrainData(trainPath)
    physicalServer, flavorToPredict, flavorDict, target, daysToPredict = ReadInputData(testPath)

    DataX,DataY = getDataSet(trainData,15,flavorToPredict,daysToPredict,7)

    RNN = RNN_network(15,5,len(DataY[0]))

    RNN.train(20,DataX,DataY,0.5,5)

    result = RNN.predict(DataX[-5:])
    print result






if __name__ == "__main__":
    main()
