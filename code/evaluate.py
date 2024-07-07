import numpy as np
import json

# recursive function to predict risk value
def predictRisk(tree, data_interpret, testValues):
    for attr in data_interpret: # for every attribute
        if(attr[0] == tree[0]): # if the node is node from the tree
            index = data_interpret.index(attr) # store index of the attribute
            value = testValues[index] # store value of the test
            # for each element if the tree
            for ele in tree[1]:
                if(int(ele) == value):
                    # if leaf node return the value
                    if(tree[1][ele] == 1):
                        return 1
                    if(tree[1][ele] == 2):
                        return 2
                    # recursive call to next node
                    return predictRisk(tree[1][ele], data_interpret, testValues)

# main function: takes the name of the file where the tree is stored
def main(fname):
    # loading tree from file
    with open('../data/' + fname) as f:
        tree = json.load(f)
    # loading dataDesc.txt
    with open('../data/dataDesc.txt') as f:
        data_interpret = json.load(f)
    # loading the test set
    file = open('../data/test.txt', "r")
    testSet = []
    # appending values to test set matrix
    for x in file.readlines():
        row = x.split(' ')
        intRow = [int(ele) for ele in row]
        testSet.append(intRow)

    correctPredictions = 0 # number of correct prediction
    wrongPredictions = 0 # number of wrong prediction
    # for each test set
    for test in range(len(testSet[0])):
        # create the test set
        testValues = []
        for i in range(6):
            testValues.append(testSet[i][test])
        # predict the risk
        predicted_Risk = predictRisk(tree.copy(), data_interpret, testValues)
        # if predicted correctly
        if(testSet[0][test] == predicted_Risk):
            correctPredictions +=1 # increase correct predictions
        else:
            wrongPredictions +=1 # increase wrong predictions
    # calculating accuracy in percentage using the correct and wrong predictions
    accuracy_percent = (correctPredictions/(correctPredictions+wrongPredictions))*100
    # return the accuracy
    return accuracy_percent


