"""
Problem Statement:
This is a classic machine learning problem where we have to train a model for spam classification. We have used a simple
bag of words model where we represent every document in terms of just an unordered bag of words instead of taking
grammar/context into consideration.

Problem Formulation:
Every document is represented as a list of feature values in a vector. And every value of the vector corresponds to a
unique word from the training set. We have considered two types of feature values:
1. Binary :- Every value in the data point vector is either a 0 or 1 . 0 represents that the particular word didnt occur in the
 			 document and 1 represents that the particular word has occurred atleast once in that document.

2.Continous:- Every value in the vector corresponds to a unique word in the training set. And every value is a number >=0
   			   and it represents the number of times that particular word has occurred. So basically we are considering the
   			   count of word occurrences in a document as the feature type.

Algorithms implemented:
1. Naive Bayes:
     In this algorithm we need to calculate the posterior probability of P(S=1/w1,w2 and so on) . To make things easier we take the
     likelihood ratio of P(S=1/w1,w2,w3 and so on)/P(S=0/w1,w2,w3 and so on). For this we need to calculate the likelihood and prior
     probabilities.
     Assumptions made:- 1. probability of occurrence of words is considered to be independent of each other.
    			        2. Some stop words like 'the','to','from' etc are not considered into our classification as they are
							equally likely to occur in a spam or a non spam document.

    Our program implementation:
	1. Loading the data from the training files
	2. Preprocessing the data and modifying the data into necessary format.
	3. Calculate the Likelihood dictionaries : For both binary and continuous cases
	4. Calculate the prior probabilities
	5. Test on the test data using the model generated

    Problems faced:
           1. In some scenarios we could encounter a word in test data that has never occurred in the train data. So, that word
  	         would have a zero likelihood probability and thus making our product of likelihoods zero and thus making our posterior
  	         probability zero.
      	     Our fix: We have assumed a very minimalistic probability for that kind of words. This is similar to laplace smoothing.

2. Decission Tree:
    For the same problem we are building a decision tree using features as words and class label being 1 for spam and 0 for
    not spam. The whole training data has been converted into a tabular format where every row represents a document and every
    column represents a feature/word. We have taken both binary and continuous feature values just as like in the naive bayes  case.

    Our program implementation:
	1. Loading the data from the training files
	2. Preprocessing the data and modifying the data into necessary format.
	3. Build a decision tree:
 		1) We have implemented a n-ary tree => A node can have multiple children depending on the number of distinct values a feature has.
		2) At every node the feature which gives the maximum Information gain is selected for splitting.
		3)The tree is built in a depth first manner .
		4) There are few stopping conditions for the tree :
			a. When depth is exceeding maximum specified depth
			b. At a node when the data is pure => either completely belonging to spam or completely belonging
			   to not spam.
	4. Test on the test data using the model generated

Results:-
	NaiveBayes:  When we use continuous feature values the accuracy is slightly higher than when we use binary feature values.
			In continuous case If a word is occurring more number of times it would be having higher likelihood probability than another
			word which occurs very less number of times. Whereas in  the binary case both would have same probability as we are only
			concerned if it occurs or not, and not about its frequency. This could be a good reason why the continuous case is giving
			better results in Naive Bayes.

  	Decission Tree: Here binary feature values are giving slightly higher accuracy value than continuous feature values.
			        This could be because, in the continuous case as the breadth of the tree is also large along with the
			        depth of the tree we are getting a very large tree and as we have only few thousand data points
 			        a lot of leaf nodes would be having very few data points.



"""
import math
import os
import time
import pickle
from DecissionTree import *
import sys

#This method finds the words that are most associated with spam and least associated with spam
def findTheTopWords(lk):
	ftype = ["bin", "cont"]
	lkCopy = lk
	for type in ftype:
		lk = lkCopy[type][1]
		values = lk.values()
		values.sort(reverse = True)
		spamValues = values[:10]

		values = lk.values()
		values.sort()
		notSpamValues = values[:10]

		spamWords = []
		notSpamWords = []
		for word,prob in lk.items():
			if prob in spamValues and len(spamWords) < 10:
				spamWords.append(word)
			if prob in notSpamValues and len(notSpamWords) < 10:
				notSpamWords.append(word)

		if type == "bin":
			print "In case of binary feature values:"
		elif type == "cont":
			print "In case of continous feature values:"
		print "Words most associated with spam are:",spamWords
		print "Words least associated with not spam are:", notSpamWords
		print ""

#This function just removes the stop words from a file and returns a bag of words
def cleanData(lines):
	stop_words=('the','From:','To:', 'a','an','From','To',':',',','and')
	c = []
	for line in lines:
		line = line.strip()
		words = line.split(" ")
		for word in words:
			if( 1 < len(word) < 20) and word not in stop_words:
				c.append(word)

	return c

#This function reads data from the files
def read_data(folder1, folder2 ,flag, totalWordsLst=[],totalWordsDict={},wordCount={},ftype = "bin"):
	print folder1
	print folder2
	files1 = os.listdir(folder1)
	files2 = os.listdir(folder2)
	d = {folder1: [files1, 0], folder2: [files2, 1]}
	if(flag == "train"):
		totalWordsLst = []
		for folder in d:
			for fle in d[folder][0]:
				if(fle != ".DS_Store"):
					f = open(folder + "/" + fle, 'r')
					lines = f.readlines()
					f.close()
					contents = cleanData(lines)
					totalWordsLst+= contents


		totalWordsLst = tuple(set(totalWordsLst))
		totalWordsDict = {}
		wordCount= {}
		counter = 0
		for word in totalWordsLst:
			totalWordsDict[word] = counter
			wordCount[word] = 0
			counter+=1


	totalData=[]
	totalDataCont = []
	for folder in d:
		for fle in d[folder][0]:
			if (fle != ".DS_Store"):

				record = [0 for i in range(len(totalWordsLst))]
				recordCont = [0 for i in range(len(totalWordsLst))]
				f = open(folder + "/" + fle, 'r')
				lines = f.readlines()
				f.close()
				contents = cleanData(lines)
				for word in contents:
					if word in totalWordsDict:
						record[totalWordsDict[word]] += 1
						recordCont[totalWordsDict[word]] += 1
						if(flag == "train"):
							wordCount[word]+=1
				record.append(d[folder][1])
				recordCont.append(d[folder][1])
				totalData.append(record)
				totalDataCont.append(recordCont)
				#print "len of record =",len(record)

	totalData=[totalData,totalDataCont]


	return totalData,totalWordsLst,totalWordsDict,wordCount

#reads data from files in a folder and appends them to a list of strings
def read_data_NVB(datasetDirectory):
	totalWordsLst = []
	dirs = {"/spam":1,"/notspam":0}
	postProb = {"/spam":0.0,"/notspam":0.0}
	summ = 0
	for dir in dirs:
		files =	os.listdir(datasetDirectory + dir)
		print "len of files = ",len(files)
		postProb[dir]= len(files)
		summ += len(files)
		for fle in files:
			if (fle != ".DS_Store"):
				f = open(datasetDirectory + dir + "/" + fle, 'r')
				lines = f.readlines()
				f.close()
				contents = cleanData(lines)
				totalWordsLst += contents

	for x in postProb:
		if(postProb[x] == 0):
			postProb[x] = 1
		postProb[x] = float (postProb[x])/summ

	print postProb
	print "got the totalwordslst"
	totalWordstuple = tuple(set(totalWordsLst))
	totalWordsDict = {} # {"w1":0,"w2":1,.....}
	counter = 0
	for word in totalWordstuple:
		totalWordsDict[word] = counter
		counter+=1

	totalData=[]
	for dir in dirs:
		files = os.listdir(datasetDirectory + dir)
		for fle in files:
			if (fle != ".DS_Store"):
				record = [0 for i in range(len(totalWordsDict))]
				f = open(datasetDirectory + dir + "/" + fle, 'r')
				lines = f.readlines()
				f.close()
				contents = cleanData(lines)
				for word in contents:
					if word in totalWordsDict:
						record[totalWordsDict[word]] += 1
				record.append(dirs[dir])
				totalData.append(record)			
	return totalData, totalWordsDict , totalWordstuple,postProb

#This function computes the lileihood and priors
def computeLikelihoodDicts(trainData,totalWordsDict,totalWordstuple):
	print "entered computing likehiloods . This program takes 4 minutes approx to run in the training part"
	lk = { "bin"  : {0:{}, 1: {}} , \
		   "cont" : {0:{}, 1: {}} }

	total = {"bin"  : {0:0, 1:0 } , \
		     "cont" : {0:0, 1:0} 	\
			}

	print "len of trainData=",len(trainData)
	for row in trainData:
		key = row[-1]
		for j in range(len(row[:-1])):
			word = totalWordstuple[j]
			if(word in lk["cont"][key]):
				lk["cont"][key][word] += row[j]
			else:
				lk["cont"][key][word] = row[j]

			total["cont"][key] += row[j]

			if(row[j] != 0):
				if (word in lk["bin"][key]):
					lk["bin"][key][word] += 1
				else:
					lk["bin"][key][word] = 1

		total["bin"][key] += 1

	for featType in lk:
		for type in lk[featType]:
			for word in lk[featType][type] :
				if lk[featType][type][word] == 0:
					lk[featType][type][word] = 1
				lk[featType][type][word] = float (lk[featType][type][word]) / total[featType][type]
				lk[featType][type][word] = math.log(lk[featType][type][word])

	return lk


#Main SECTION STARTS


mode = sys.argv[1]
technique = sys.argv[2]
datasetDirectory = sys.argv[3]
modelFile = sys.argv[4]

time1 = time.time()

if(technique == "bayes"):
	if mode == "train":
		trainData, totalWordsDict ,totalWordstuple,postProb = read_data_NVB(datasetDirectory)
		lk = computeLikelihoodDicts(trainData,totalWordsDict,totalWordstuple)
		lk = {'lk' : lk , 'post' : postProb}
		with open(modelFile, 'wb') as handle:
			pickle.dump(lk, handle)
		findTheTopWords(lk['lk'])
	elif mode == "test":

		with open(modelFile, 'rb') as handle:
			lk = pickle.load(handle)

		post = lk['post']
		lk = lk['lk']
		findTheTopWords(lk)
		dirs = {"/spam": 1,"/notspam": 0}
		case = {"bin":"binary","cont":"continous"}

		for ftype in ["bin","cont"]:
			print "BELOW ARE THE RESULTS FOR "+case[ftype]+" CASE:"
			corr = 0
			wrong = 0
			count0 = 0
			count1 = 0

			tp,tn,fp,fn = 0,0,0,0
			for dir in dirs:
				files = os.listdir(datasetDirectory + dir)
				for fle in files:
					cls = 0
					if (fle != ".DS_Store"):
						f = open(datasetDirectory + dir + "/" + fle, 'r')
						lines = f.readlines()
						f.close()
						content = cleanData(lines)
						content = set(content)
						prob0 = 0
						prob1 = 0
						for word in content:
							if(word in lk[ftype][0] and word in lk[ftype][1]):
								prob0 += lk[ftype][0][word]
								prob1 += lk[ftype][1][word]

						prob0 += math.log(post['/notspam'])
						prob1 += math.log(post['/spam'])

						if(prob1 > prob0):
							cls = 1
						else:
							cls = 0

						if (cls == dirs[dir]):
							corr += 1
						else:
							wrong += 1


						if(cls == 1 and dirs[dir] == 1):
							tp+=1
						if (cls == 1 and dirs[dir] == 0):
							fp += 1
						if (cls == 0 and dirs[dir] == 0):
							tn += 1
						if (cls == 0 and dirs[dir] == 1):
							fn += 1

			print "corr = ", corr, "wrong =", wrong
			create_confusion_matrix(tp, fn, fp, tn)
			print "Accuracy= "+str(float(tp+tn) * 100/(tp+fp+tn+fn))+"%"

elif(technique == "dt"):
	if mode == "train":
		print "inside dt"
		testData = []
		trainData, totalWordsLst, totalWordsDict,wordCount = read_data(datasetDirectory + "/notspam", datasetDirectory + "/spam", "train")
		trainDataCont = trainData[1]
		trainData = trainData[0]

		print "after getting train data from files"
		featureList = []
		total = sum(wordCount.values())
		for word in wordCount:
			if (6 < wordCount[word] < 10000):
				featureList.append(totalWordsDict[word])

		head = main(trainData,featureList)
		print "after getting binary head"
		headCont = main(trainDataCont,featureList)
		print "after getting continous head"
		model = {"model":head,"modelCont":headCont,"totalWordsLst":totalWordsLst, "totalWordsDict":totalWordsDict }

		with open(modelFile, 'wb') as handle:
			pickle.dump(model, handle, protocol=2)
			handle.close()

	elif mode == "test":
		with open(modelFile, 'rb') as handle:
			model = pickle.load(handle)
			handle.close()

		head = model["model"]
		headCont = model["modelCont"]

		totalWordsLst = model["totalWordsLst"]
		totalWordsDict = model["totalWordsDict"]

		testData, totalWordsLst, totalWordsDict,wordCount = read_data(datasetDirectory + "/notspam", datasetDirectory + "/spam", "test",
															totalWordsLst, totalWordsDict)

		testDataCont = testData[1]
		testData = testData[0]
		print "RESULTS FOR BINARY CASE ARE BELOW"
		testYourTree(head, testData)
		print "RESULTS FOR CONTINOUS CASE ARE BELOW"
		testYourTree(headCont, testDataCont)

print "total time taken=",time.time()-time1
