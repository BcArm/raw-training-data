import scipy.io
import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from sklearn.decomposition import PCA
import random
import pickle

samplingRate = 128
window = 128
channel = 11
nChannels = 14
nflash=13
nLabels=12
epochs = 30
screen=[ ['A','B','C','D','E','F'],
     ['G','H','I','J','K','L'],
     ['M','N','O','P','Q','R'],
     ['S','T','U','V','W','X'],
     ['Y','Z','1','2','3','4'],
     ['5','6','7','8','9','0'] ]
def loadData():


    binarytarget = np.zeros((12, 1))
    target = np.empty((0, 1))
    data = np.empty((0, 15, 128, 14))
    samplesData = np.zeros((12, 15, 128, 14))
    
    # open files
    finFlashing = open("flashing.txt", "r")
    finStimCode = open("stimulusCode.txt", "r")
    finStimType = open("stimulusType.txt", "r")
    finSampData = open("samplesData.txt", "r")
    # copy files
    isFlashing = list(map(int, finFlashing.readlines()))
    stimCode = list(map(int, finStimCode.readlines()))
    stimType = list(map(int, finStimType.readlines()))
    sampData = finSampData.readlines()
    for i in range(len(sampData)):
        sampData[i] = map(float, sampData[i].split())
    # close files
    finFlashing.close()
    finStimCode.close()
    finStimType.close()
    finSampData.close()

    # parse data
    ind = 1
    for epoch in range(30):
        rowColCnt = [0] * 12
        while (min(rowColCnt) < 15):
            lst = isFlashing[ind - 1]
            cur = isFlashing[ind]
            # last moment of flashing
            if (lst == 1 and cur == 0):
                rowcol = stimCode[ind - 1] - 1
                typ = stimType[ind - 1]
                binarytarget[rowcol] = typ
                L = []
                for i in range(ind - nflash, min(ind + window - nflash, len(sampData))):
                    L.append(list(sampData[i]))
                while (len(L) < 128):
                    #print epoch
                    L.append([0] * 14)
                samplesData[rowcol, rowColCnt[rowcol], :, :] = np.array(list(L))
                rowColCnt[rowcol] += 1
            ind += 1
        target = np.append(target, binarytarget, axis = 0)
        data = np.concatenate((data, samplesData), axis = 0)
    return (data, target)

def PCA_Transform(X):
    pca = PCA(n_components = X.shape[0])
    pca.fit(X)
    return pca.transform(X)


def work(X, Y, iters, PCA_on):
    # load true characters
    finTrueLabels = open("trueChars.txt", "r")
    R = finTrueLabels.readline().split()[0]
    print(R)
    finTrueLabels.close()
    S = [1] * 24 + [0] * 6

    if (PCA_on):
        X = PCA_Transform(X)
    good = 0
    bad = 0
    for it in range(iters):
        random.shuffle(S)
        XX = []
        YY = []
        for i in range(30):
            if (S[i] == 1):
                for j in range(12):
                    XX.append(list(X[i * 12 + j]))
                    YY.append(list(Y[i * 12 + j])[0])

##        model=linear_model.LogisticRegression(penalty='l2',C=0.12)
##        model.fit(XX,YY)
        
##        model = linear_model.logistic(penalty='l2')
##        model.fit(XX, YY)
##
        model = svm.LinearSVC(C = 1.0)
        model.fit(XX, YY)
        
##        model.class_weight = "balanced"
        
        for i in range(30):
            if (S[i] == 0):
                target = R[i]
                dsamplesData=[]
                for j in range(12):
                    dsamplesData.append(list(X[i * 12 + j]))
                score = model.decision_function(dsamplesData)
                bestcol=np.argmax(score[0:6])
                bestrow=np.argmax(score[6:12])
                result = screen[bestcol][bestrow]
                if (result == target):
                    good += 1
                else:
                    bad += 1
    print("Accuracy: " + str(good * 1.0 / (good + bad)))
                
(data, target) = loadData()
Y=np.reshape(target,(nLabels*epochs,1))
data=np.mean(data,axis=1)
X=np.reshape(data,(nLabels*epochs,window*nChannels),order="F")

work(X, Y, 100, True)
