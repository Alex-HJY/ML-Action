import sklearn as sk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import operator
import sys
import kNN

def create_DataSet():
    group=np.array([[1,1.1],[1,1],[0,0],[0,0.1]])
    lables=['A','A','B','B']
    return group,lables

print(sys.path)
