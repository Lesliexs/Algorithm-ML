from numpy import *
import numpy as np
import operator
def createDataSet():
    group= array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

# inX是需要分类的向量，dataSet是训练样本集，labels是标签，k是前K个距离最小元素
# 该函数做欧式距离计算
def classify0(inX,dataSet,labels,k):
    # 选取dataSet样本集的行数，因为样本是矩阵，只有两维，所以shape[0]是指的行数，
    dataSetSize=dataSet.shape[0]
    # dateSetSize=4 所以tile（inX，（4，1））的意思是让inx数组扩张成4*1的矩阵，这样跟后面的dateSet格式相同，可以做减法
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    # 将每个训练样本与需要分类的向量相减后的值取平方值
    sqDiffMat=diffMat**2
    # 这里的sum(axis=1)比较有意思，将矩阵的每一行向量相加
    sqDistance=sqDiffMat.sum(axis=1)
    # 相加之后求平均，得到欧式距离
    distance=sqDistance**0.5
    # 返回距离数组从小到大的索引值
    sortedDistIndicies=distance.argsort()
    # 创建一个列表
    classCount={}
    for i in range(k):
        # 找出从小到达索引值对应的标签
        voteIlabel=labels[sortedDistIndicies[i]]
        # 计算前K个数据中的标签所对应的个数
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    # 按照标签个数升序排列，取出第一个最多的标签,classCount.item()返回的是一个列表，里面包含的多个元祖，按照元祖的第二个值排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
# 不能分开赋值group labels 因为函数是将第一个值赋给group，如果在赋给labels的话，group又赋给了labels

# group=createDataSet()
# label=createDataSet()
# 这样就将第一个返回值给了group，第二个返回值给了labels，刚好对应

# def file2matrix(filename):
#     fr = open(filename)
#
#     arrayOLines=fr.readline()
#     # print(arrayOLines)
#     numberOfLines=len(arrayOLines)
#     returnMat = np.zeros((numberOfLines,3))
#     classLabelVector=[numberOfLines,3]
#     n=1
#     index=0
#     list=[]
#     while n<numberOfLines:
#
#         arrayOLines = fr.readline(n)
#         list1=arrayOLines.split('\t')
#         n+=1
#         # print(list1)
#         for i in list1:
#             m=float(i)
#         # print(m
#             list.append(m)
#     # print(list)
#         returnMat[index,:]=list1[0:3]
#         index+=1
#         print(returnMat)
    # for line in arrayOLines:
    #     line=line.strip()
    #     if line=='\t':
    #         returnMat[index, :] = (listFromLine[0:1])
    #         classLabelVector.append(int(listFromLine[-1]))
    #         continue
    #     else:
    #         list.append(line)
        # if index==3
        # returnMat[index,:]=(listFromLine[0:1])
        # classLabelVector.append(int(listFromLine[-1]))
        # index+=1
    # return returnMat,classLabelVector

# 将文本记录转换成Numpy的解析程序
def file2matrix(filename):
    fr = open(filename)
    # 按行读取文件
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        # 去除首尾的空格
        line = line.strip()
        # 每次遇到一个't'就把这一部分赋值给这个元素
        listFromLine = line.split('\t')
        # 将每一行的前三个元素依次赋值给预留矩阵空间
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

# 画散点图
def draw():
    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax=fig.add_subplot(111)
    # datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15*array(datingLabels),array(datingLabels))
    plt.show()

# def draw():
#     import matplotlib
#     import matplotlib.pyplot as plt
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
#     ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
#     #ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15,0*array(datingLabels))
#     plt.show()

# 归一化
def autoNormal(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    # 以dataSet的大小创建一个全0的矩阵
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

# 分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio=0.10
    normMat, ranges, minVals = autoNormal(datingDataMat)
    m=normMat.shape[0]
    # normMat, ranges, minVals=autoNormal(datingDataMat)
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print('the classifier came back with %d,the real answer is:%d' %(classifierResult,datingLabels[i]))
        if classifierResult !=datingLabels[i]:
            errorCount+=1.0
    print('the total error rata is %f' %(errorCount/float(numTestVecs)))

# def datingClassTest():
#     hoRatio = 0.10
#     datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
#     normMat,ranges,minVals = autoNormal(datingDataMat)
#     m = np.size(normMat,axis=0)
#     numTestVecs = int(hoRatio*m)
#     errorCount = 0.0
#     for i in range(numTestVecs):
#         classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
#         print("The classifier came back with:%d,the real answer is:%d" % (classifierResult,datingLabels[i]))
#         if classifierResult != datingLabels[i]:
#             errorCount += 1.0
#     print("the total error rate is:%f" % (errorCount/float(numTestVecs)))

# 约会网站预测函数
def classifyPerson():
    resultList=['not at all','in small dose','in large dose']
    percentTats=float(input('percenttage of time spent playing video games?'))
    ffmiles=float(input('frequent filer miles earned per year?'))
    iceCream=float(input('liter of iceCream consuned per years'))
    normMat,ranges,minVals=autoNormal(datingDataMat)
    inArr=array([percentTats,ffmiles,iceCream])
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print('you will probably like this person:',resultList[classifierResult-1])

# 手写识别系统 准备数据：将图像转换成测试向量
def img2Vector(filename):
    returnVector=zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVector[0,i*32+j]=lineStr[j]
        return returnVector




if __name__=='__main__':

    group,labels=createDataSet()
    print(classify0([3,0],group,labels,3))
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    print(datingDataMat)
    print(datingLabels[0:20])
    draw()
    print(autoNormal(datingDataMat))
    print(datingClassTest())
    print(classifyPerson())


