import numpy as np
#Each sample of our dataset consists of 6 signals, and each signal has 501 samples.
SIGNAL_NUM = 6
SAMPLE_NUM = 501
def minmaxscaler(mat):
    l=[]
    for i in range(len(mat)):
        Min = mat[i,:].min()
        Max = mat[i,:].max()
        for k in mat[i,:]:
            l.append((k-Min)/(Max-Min))
    return(np.array(l).reshape(mat.shape))

def MVPE(mvts):
    ScaledMvts = minmaxscaler(mvts)
    permutations = np.array(list(it.permutations(range(6))))
    patmat = permutations.dot(ScaledMvts) #pattern matrix
    Plist = patmat.argmax(0) #list consist of pattern for each column
    Pdict = dict()
    for i in Plist:
        if i in Pdict:
            Pdict[i] += 1
        else:
            Pdict[i] = 1
    for i in Pdict:
        Pdict[i] = Pdict[i]/SAMPLE_NUM
    MvPE=0
    for i in Pdict:
        MvPE += -Pdict[i]*np.log(Pdict[i])
    MvPEN = MvPE/(np.log(np.factorial(SIGNAL_NUM)))
    return MvPEN