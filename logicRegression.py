from numpy import *
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import time

def loadDataSet():
    filename = "/opt/LR/testSet.txt"
    dataMat = []
    labelMat = []
    with open(filename) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))
        return dataMat, labelMat

def sigmoid(inX):
    return 1.0 / ( 1 +exp(-inX) )


def gradAscent(dataMat, labelMat):
    dataMatrix = mat(dataMat)
    classLabels = mat(labelMat).transpose()
    m,n = shape(dataMatrix)
    print ("dataMatrix = ",dataMatrix)
    print ("n=",n)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (classLabels - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

def stocGradAscent(dataMat, labelMat):
    dataMatrix = mat(dataMat)
    classLabels = labelMat
    m,n = shape(dataMatrix)
    alpha = 0.01
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        for i in range(m):
            h = sigmoid(sum(dataMatrix[i] * weights))
            error =h - classLabels[i]
            weights = weights - alpha * error * dataMatrix[i].transpose()
    return weights

def stocGradAscentAlpha(dataMat, labelMat):
    dataMatrix = mat(dataMat)
    classLabels = labelMat
    m,n = shape(dataMatrix)
    weights = ones((n,1))
    maxCycles = 500
    for j in range(maxCycles):
        dataIndex = [i for i in range(m)]
        for i in range(m):
            alpha = 1 / (1+j+i) + 0.0001
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex].transpose()
            del(dataIndex[randIndex])
    print ("weights = ",weights)
    return weights

def plotBestFit(weights):
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1, s=30, c = 'red', marker ='s')
    ax.scatter(xcord2,ycord2, s=30, c = 'green')
    x = arange(-3.0,3.0, 0.1)
    y = (-weights[0] - weights[1] * x)/ weights[2]
    ax.plot(x,y)
    plt.xlabel("X1")
    plt.xlabel("X2")
    plt.show()
    plt.savefig("GD.png")


def main():
    dataMat, labelMat = loadDataSet()
    weights = stocGradAscentAlpha(dataMat, labelMat).getA()
    print ("weights = ",weights)
    plotBestFit(weights)

if  __name__ == '__main__':
    main()
