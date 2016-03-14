import math
import sys
from scipy.io import arff

def initial():
    global traindata,trainmeta,attr,row,col,testdata,testmeta,trow,tcol
    traindata, trainmeta = arff.loadarff(sys.argv[1])
    attr = trainmeta._attrnames
    row = len(traindata)
    col = len(traindata[0])
    testdata, testmeta = arff.loadarff(sys.argv[2])
    trow = len(testdata)
    tcol = len(testdata[0])
    return sys.argv[3] == 'n'

def mutualI(a,b,traindata):
    if a == b:
        return 0
    res = 0
    for label in trainmeta[attr[col-1]][1]:
        for ai in trainmeta[attr[a]][1]:
           for bi in trainmeta[attr[b]][1]:
                joint_num = 0
                label_num = 0
                a_num = 0
                b_num = 0
                for user in traindata:
                    if (ai,bi,label) == (user[a].decode("utf-8"),user[b].decode("utf-8"),user[col-1].decode("utf-8")):
                        joint_num += 1
                    if label == user[col-1].decode("utf-8"):
                        label_num +=1
                    if (ai,label) == (user[a].decode("utf-8"),user[col-1].decode("utf-8")):
                        a_num += 1
                    if (bi,label) == (user[b].decode("utf-8"),user[col-1].decode("utf-8")):
                        b_num += 1
                p_ab = (joint_num+1) / (label_num + len(trainmeta[attr[a]][1])*len(trainmeta[attr[b]][1]))
                p_a = (a_num + 1) / (label_num+len(trainmeta[attr[a]][1]))
                p_b = (b_num+1) / (label_num + len(trainmeta[attr[b]][1]))
                inf = (joint_num+1) / (len(traindata)+len(trainmeta[attr[col-1]][1])*len(trainmeta[attr[a]][1])*
                                      len(trainmeta[attr[b]][1]))*math.log(p_ab/(p_a*p_b),2)
                res += inf
    return res


def edge_weight(traindata):
    adjlist = []
    for i in range (col-1):
        row_edge = []
        for j in range(col -1 ):
            row_edge.append(mutualI(i,j,traindata))
        adjlist.append(row_edge)
    return adjlist

def generatetree(traindata):
    adjlist = edge_weight(traindata)
    # now we are using prim to build a tree
    v = []
    e = [[] for x in range(col-1)]
    v.append(0)
    while len(v) < col-1:
        max = 0
        max_index = -1
        max_initial = -1
        for i in v:
            for j in range(col-1):
                if j not in v:
                    if adjlist[i][j] > max:
                        max = adjlist[i][j]
                        max_index = j
                        max_initial = i
        v.append(max_index)
        e[max_index].append(max_initial)
    return e

def cond_prob(tree,feature_index,testuser,label,traindata):
    testfeature = [testuser[x].decode("utf-8") for x in tree[feature_index] ]
    testfeature.append(label)
    nom = 0
    denom = 0
    for user in traindata:
        trainfeature = [user[x].decode("utf-8") for x in tree[feature_index]]
        trainfeature.append(user[col-1].decode("utf-8"))
        if trainfeature == testfeature:
            denom+=1
            if testuser[feature_index] == user[feature_index]:
                nom+=1
    prob = (nom+1)/(len(trainmeta[attr[feature_index]][1])+denom)
    return prob

def calprob_tan(tree,testuser,traindata):
    problist = []
    for label in trainmeta[attr[col-1]][1]:
        num = 0
        for user in traindata:
            if user[col-1].decode("utf-8") == label:
                num+=1
        prob = (num+1)/(len(traindata)+len(trainmeta[attr[col-1]][1]))
        for i in range(col-1):
            prob *= cond_prob(tree,i,testuser,label,traindata)
        problist.append(prob)
    return problist

def bayes():
    v = []
    e = [[] for x in range(col-1)]
    for i in range(col-1):
        print (attr[i])
    print("\n")
    right_label = 0
    for user in testdata:
        list = calprob_tan(e,user,traindata)
        summ = sum(list)
        predict = trainmeta[attr[col-1]][1][list.index(max(list))]
        print (predict,user[col-1].decode("utf-8"),round(max(list)/summ,12))
        if(predict == user[col-1].decode("utf-8")):
            right_label+=1
    print ('\n')
    print (right_label)



def tan():
    tree = generatetree(traindata)
    for i in range(col-1):
        print (attr[i], " ".join([attr[j] for j in tree[i]]),"class")
    print("\n")
    right_label = 0
    for user in testdata:
        list = calprob_tan(tree,user,traindata)
        summ = sum(list)
        predict = trainmeta[attr[col-1]][1][list.index(max(list))]
        print (predict,user[col-1].decode("utf-8"),round(max(list)/summ,12))
        if(predict == user[col-1].decode("utf-8")):
            right_label+=1

    print ('\n')
    print (right_label)



if __name__ == "__main__":
    if initial() == True:
        bayes()
    else:
        tan()