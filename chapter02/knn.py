import numpy as np
import operator
import matplotlib.pyplot as plot

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX,dataset,labels,k):
    datasetsize=dataset.shape[0]
    diffmat=np.tile(inX,(datasetsize,1))-dataset
    distances=(diffmat**2).sum(axis=1)**0.5
    sorted_distindicies=distances.argsort()
    classcount={}
    for i in range(k):
        voteilabel=labels[sorted_distindicies[i]]
        classcount[voteilabel]=classcount.get(voteilabel,0)+1
    sortedclasscount=sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedclasscount[0][0]

def file2matrix(filename):
    with open(filename) as f:
        numberoflines=len(f.readlines())
    returnmat=np.zeros((numberoflines,3))
    classlabelvector=[]
    with open(filename) as f:
        filelines=f.readlines()
    index=0
    for line in filelines:
        line = line.strip()
        listfromline=line.split('\t')
        returnmat[index,:]=listfromline[0:3]
        classlabelvector.append(int(listfromline[-1]))
        index+=1
    return returnmat,classlabelvector

def autonorm(dataset):
    minvals=dataset.min(0)
    maxvals=dataset.max(0)
    ranges=maxvals-minvals
    normdata=np.zeros(np.shape(dataset))
    normdata=dataset-np.tile((minvals,(dataset.shape[0],1)))
    normdata/=np.tile((ranges,(dataset.shape[0],1)))
    return normdata,ranges,minvals

def datingclasstest():
    ratio=0.5
    dataingdatamat,datinglables=file2matrix('datingTestSet2.txt')
    normmat,ranges,minval=autonorm(dataingdatamat)
    m=normmat.shape[0]
    numoftestvecs=int(ratio*m)
    errorcount=0
    for i in range(numoftestvecs):
        classifierresult=classify0(normmat[i,:],normmat[numoftestvecs:m,:],datinglables[numoftestvecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierresult,datinglables[i]))


# dataset,lables=createDataSet()
# classify0([0,0],dataset,lables,3)
# plot.scatter(dataset[:,0],dataset[:,1])
# plot.show()