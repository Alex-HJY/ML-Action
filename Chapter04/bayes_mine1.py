import numpy as np


def loadDataSet():
    """
    创建数据集
    :return: 单词列表postingList, 所属类别classVec
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],  # [0,0,1,1,1......]
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocablist(dataset):
    """
    获取单词集合
    :param dataset:
    :return: 单词集合
    """
    vocabset = set([])
    for doc in dataset:
        vocabset = vocabset | set(doc)
    return vocabset


def setofword2vec(vocablist, inputset):
    returnvec = [0] * len(vocablist)
    vocablist = list(vocablist)
    vocablist = sorted(vocablist)
    for word in inputset:
        if word in vocablist:
            returnvec[vocablist.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnvec


def trainnb0(trainmatrix, traincategory):
    num_traindocs = len(trainmatrix)
    num_words = len(trainmatrix[0])
    pAbusive = sum(traincategory) / num_traindocs
    p0num = np.ones(num_words)
    p1num = np.ones(num_words)
    p0denom = 2
    p1denom = 2
    for i in range(num_traindocs):
        if traincategory[i] == 1:
            p1num += trainmatrix[i]
            p1denom += sum(trainmatrix[i])
        else:
            p0num += trainmatrix[i]
            p0denom += sum(trainmatrix[i])
    p1vect = np.log(p1num / p1denom)
    p0vect = np.log(p0num / p0denom)
    return p0vect, p1vect, pAbusive


def classifynb(vec2classify, p0vec, p1vec, pClass1):
    p1 = sum(vec2classify * p1vec) + np.log(pClass1)
    p0 = sum(vec2classify * p0vec) + np.log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def testnb():
    listoposts, listclasses = loadDataSet()
    myvocablist = createVocablist(listoposts)
    trainmat = []
    for doc in listoposts:
        trainmat.append(setofword2vec(myvocablist, doc))
    p0v, p1v, pclass = trainnb0(np.array(trainmat), np.array(listclasses))
    testvec = ['stupid']
    thisdoc = setofword2vec(myvocablist, testvec)
    print(classifynb(thisdoc, p0v, p1v, pclass))


testnb()
