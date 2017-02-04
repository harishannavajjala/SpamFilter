# NOTE: All our team mates have some kind of decission tree implementation in the past and this code and ideas are inspired
# from our previous experiences
import math
import time
import random

class Decision_node:								# class to represent each node in the tree
    def __init__(self, results=None,depthLevel = 1,col=None,values=None,children=[],hasChildren = False):  #initialize each node in the decision tree
        self.results = results          # a list of lists to store the resulting rows
        self.col = col                  # a variable to store the column value of the attribute to be split on
        self.children = children        # a list containing the children to each node
        self.depthLevel = depthLevel    # height till which the tree has to be constructed
        self.isLeaf = False             # a variable to test if the node is leaf
        self.parent = None              # a variable to keep track of the parent of the current node
        self.classDist = None           # a variable to give out the class distribution of a particular node
        self.colValues = None

    #This method splits given results set based on a feature
    def splitData(self):
        resultSet = {} # a dictionary to store the rows associated to each attribute value
        for result in self.results:
            if result[self.col] not in resultSet:
                resultSet[result[self.col]] = [result]
            else:
                resultSet[result[self.col]].append(result)
        return resultSet

    #This method sets the leaf nodes of decission tree to some class : (0,1)
    def setClassDist(self): # a method to store the class distribution for a node based on majority values
        results = self.results

        count0 = 0
        count1 = 0
        for result in results:
            if (result[-1] == 0):
                count0 += 1
            elif (result[-1] == 1):
                count1 += 1

        if (count0 > count1):
            self.classDist = 0
        else:
            self.classDist = 1

    #This method classifies a given test record to either : (0/1)
    def classify(self,testRecord):
        if(self.isLeaf):
            return self.classDist
        else:
            col = self.col
            for child in self.children:
                if(child.results[0][col] == testRecord[col]):
                    return child.classify(testRecord)

    #This method organizes the decission tree
    def deleteExtraChildren(self):
        result = []
        for i in range(len(self.children)):
            if(self.children[i].parent == self):
                result.append(self.children[i])
        self.children = result

def entropy(results):       #a function to calculate the entropy of a particular dataset
    entropy_value = 0.0
    rows_length = len(results)
    counted_dict = class_attrib_value_count(results)
    for value in counted_dict.keys():
        p = float(counted_dict[value])/rows_length
        if p<=0:
            p=1
        else:
            entropy_value -= (p * math.log(p,2))
    return entropy_value


def class_attrib_value_count(results):  # a function to give out the existing class distributions of a given dataset.
    count_dict = {}   # a dictionary to maintain count of each attribute value
    for row in results:
        value = row[-1]
        if value in count_dict:    # if value is already in dict, increment it
            count_dict[value] += 1
        else:
            count_dict[value] = 1   # else assign its count as zero
    #print count_dict
    return count_dict

#This method finds if a given dataset is pure or not i.e., is it all from same class - (0/1)
def isImPure(results):
    count0=0
    count1=0
    for result in results:
        if(result[-1]==0):
            count0 +=1
        elif(result[-1]==1):
            count1+=1
        if(count0>0 and count1>0):
            return True
    return False

#This method recursively builds a decission tree for a given dataset , feature list and a  depth
def buildTree(results,totalDepth,featureList,initialDepth,parent = None):
    newNode = Decision_node(results, initialDepth)
    newNode.parent = parent
    best_gain = 0
    best_attrib = None
    best_partition= None
    current_entropy = entropy(results) # find out the entropy of the new node containing the subset
    for column in featureList:
        newNode.col = column
        partitions = newNode.splitData()  # split up the node into their resulting children along with their own subsets

        new_entropy = 0.0 # set the intermediate entropy computation to zero
        for val in partitions: # loop through all the possible column values
            new_entropy = new_entropy + (entropy(partitions[val]) * (float(len(partitions[val]))/len(results)) ) # calculate the weighted entropy for that column
        information_gain = current_entropy - new_entropy
        if (information_gain > best_gain):
            best_gain = information_gain
            best_attrib = column
            best_partition = partitions

    newNode.col = best_attrib # set the column with highest information gain(best attribute) to be the splitting column
    if(newNode.depthLevel<=totalDepth and len(results)>1 and isImPure(results) and best_attrib!=None) :
        resultSet = best_partition
        newNode.colValues=resultSet.keys()
        for i in resultSet:
            x = buildTree(resultSet[i],totalDepth,featureList,initialDepth+1,newNode)
            if x.depthLevel == newNode.depthLevel+1:
                newNode.children.append(x)
    else:
        newNode.isLeaf = True
        newNode.children = []
        newNode.setClassDist()

    newNode.deleteExtraChildren()
    return newNode



#Finds the accuracy
def calculate_accuracy(incorrectly_classified,correctly_classified):

    print("\n\n\nIncorrectly classified= " + str(incorrectly_classified) + "\t\t Correctly classified= " + str(correctly_classified)+"\n")
    accuracy = float(correctly_classified) / (correctly_classified + incorrectly_classified)
    print("\nAccuracy is " + str(accuracy*100)+" %"+"\n")
    return accuracy

#This method prints the confusion matrix
def create_confusion_matrix(tp,fn,fp,tn):
    print "\nThe confusion matrix is as follows:\n"
    print "                                 Predicted"
    print "----"*20
    print "Actual"
    print "                 True Negative: "+str(tn),
    print "                 False Positive: "+str(fp)
    print "\n"
    print "                 False Negative: "+str(fn),
    print "                 True Positive: "+str(tp)
    print "----"*20



#This method prints the decission tree
def printTree(node,num=0):
    depth = 5
    if(node.isLeaf == False):
        for child in node.children:
            child.parent = None
            child.results = child.results[:2]
            val = child.results[0][node.col]
            if node.depthLevel <depth:
                for i in range(num):
                    print "\t",
                print "if(column "+str(node.col)+"=="+str(val)+"):",
            if(child.isLeaf == False):
                if node.depthLevel < depth:
                    print "\n"
                printTree(child,num+1)
            else:
                if node.depthLevel < depth:
                    print "class Distribution="+str(child.classDist)

def main(trainData,featureList):
    print("\nWelcome to the decision tree classifier implementation!")

    x = len(featureList)/2 if len(featureList)/2 < 7000 else 7000
    featureList = [random.choice(featureList) for i in range(x)]
    totalDepth = 2000
    head = buildTree(trainData, totalDepth, featureList, 1)
    head.results = head.results[:2] #Just removing some unnecessary stuff from the nodes
    printTree(head)
    return head

def testYourTree(head,testData):
    printTree(head)
    print "INSIDE TESTYOURTREE"
    incorrectly_classified = 0
    correctly_classified = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for t in testData:
        predicted = head.classify(t)
        if (predicted != t[-1]):
            incorrectly_classified += 1
        else:
            correctly_classified += 1

        if (predicted == 1 and t[-1] == 1):
            tp += 1
        elif (predicted == 0 and t[-1] == 1):
            fn += 1
        elif (predicted == 1 and t[-1] == 0):
            fp += 1
        elif (predicted == 0 and t[-1] == 0):
            tn += 1

    print "tp=",tp,"fp=",fp,"tn=",tn,"fn=",fn
    create_confusion_matrix(tp,fn,fp,tn)
    acc = calculate_accuracy(incorrectly_classified,correctly_classified)
    pass
