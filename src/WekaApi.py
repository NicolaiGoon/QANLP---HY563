def createWeka(path,X,Y,classes):
    '''
    exports a weka file for Machine Learning
    '''
    assert(len(X) == len(Y))
    f = open(path+ ".arff", "w")
    f.write("@relation category\n")
    vectorsSize=len(X[0])
    for i in range(vectorsSize):
        f.write("@attribute att"+str(i)+" numeric\n")
    f.write("@attribute category {"+classes+"}\n\n@data\n")
        
    for i in range(len(X)):
        for j in range(vectorsSize):
            f.write(str(X[i][j])+",")
        f.write(Y[i]+'\n')
    f.close()
