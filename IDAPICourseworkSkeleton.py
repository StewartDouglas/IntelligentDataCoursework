#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from IDAPICourseworkLibrary import *
from numpy import *
import math
#
# Coursework 1 begins here
#
# Function to compute the prior distribution of the variable root from the data set
def Prior(theData, root, noStates):
    prior = zeros((noStates[root]), float) 
    # I assume the theData is a list of lists
    # Calculate an array of probabilties
    # First caluclate the number of times each data point occurs
    for row in range(theData.shape[0]):
        prior[theData[row][root]] += 1;
    # Divide by the total number of data points to get probabilities 
    for prob in range(prior.shape[0]):
        prior[prob]/=theData.shape[0]
    return prior
  
# Function to compute a CPT with parent node varP and child node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = zeros((noStates[varC], noStates[varP]), float )
    parentCount = zeros(noStates[varP])
    for row in range(theData.shape[0]):
        cPT[theData[row][varC]][theData[row][varP]]+=1
        parentCount[theData[row][varP]]+=1
    # Now normalise
    for row in range(cPT.shape[0]):
      for col in range(cPT.shape[1]):
	  if parentCount[col] != 0:
            cPT[row][col]/=parentCount[col]
    return cPT
  
# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = zeros((noStates[varRow], noStates[varCol]), float )
    for row in range(theData.shape[0]):
        jPT[theData[row][varRow]][theData[row][varCol]]+=1
    for row in range(jPT.shape[0]):
        for col in range(jPT.shape[1]):
            jPT[row][col]/=theData.shape[0]
    return jPT

# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
  # Calcualte the probably of the column/parent variable for each of its states
  probVarCol = zeros(aJPT.shape[1])
  for col in range(aJPT.shape[1]):
    for row in range(aJPT.shape[0]):
      probVarCol[col] += aJPT[row][col]
  for col in range(aJPT.shape[1]):
    for row in range(aJPT.shape[0]):
      aJPT[row][col]/=probVarCol[col]
  return aJPT

#
# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes): 
    rootPdf = zeros((naiveBayes[0].shape[0]), float)

    for i in range(rootPdf.shape[0]):
        # Initially the posterior probability is equal to the prior probability
        rootPdf[i] = naiveBayes[0][i]
        # Update the posterior probability using relevant conditionals
        for j in range(len(naiveBayes)-1):
            rootPdf[i] *= naiveBayes[j+1][theQuery[j]][i]

    # Calculate the scale factor (1/denominator)
    denominator = 0
    for i in range(rootPdf.shape[0]):
        temp = 1.0
        for j in range(len(naiveBayes)-1):
            temp *= naiveBayes[j+1][theQuery[j]][i]
        denominator += (temp*naiveBayes[0][i])
    
    # Normalise the probabilities    
    for i in range(rootPdf.shape[0]):
        rootPdf[i] /= denominator

    return rootPdf

# End of Coursework 1
#
# Coursework 2 begins here
#
# Calculate the mutual information from the joint probability table of two variables
def MutualInformation(jP):
    mi=0.0
    # First, calculate the marginals
    marginalA = zeros(jP.shape[0])
    marginalB = zeros(jP.shape[1])
    for col in range(jP.shape[1]):
         for row in range(jP.shape[0]):
             marginalB[col] += jP[row][col]
    for row in range(jP.shape[0]):
        for col in range(jP.shape[1]):
            marginalA[row] += jP[row][col] 
    # Second, calculate the mutual information
    for row in range(jP.shape[0]):
        for col in range(jP.shape[1]):
            if jP[row][col] > 0:
                 mi += jP[row][col]*math.log((jP[row][col]/(marginalA[row]*marginalB[col])),2);
    return mi
#
# construct a dependency matrix for all the variables
def DependencyMatrix(theData, noVariables, noStates):
    MIMatrix = zeros((noVariables,noVariables))
    for i in range(noVariables):
        for j in range(noVariables):
            jP = JPT(theData,i,j,noStates)
            MIMatrix[i][j] = MutualInformation(jP)
    return MIMatrix

# Function to compute an ordered list of dependencies 
def DependencyList(depMatrix):
    depList=[]
    for i in range(depMatrix.shape[0]):
        for j in range(depMatrix.shape[1]):
            if i < j:
                depList.append([depMatrix[i][j],i,j])
    depList2 = sorted(depList, key=lambda dep: dep[0], reverse = True)
    return array(depList2)
#
# Functions implementing the spanning tree algorithm
# Coursework 2 task 4

def SpanningTreeAlgorithm(depList, noVariables):
    spanningTree = []
  
    return array(spanningTree)
#
# End of coursework 2
#
# Coursework 3 begins here
#
# Function to compute a CPT with multiple parents from he data set
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT_2(theData, child, parent1, parent2, noStates):
    cPT = zeros([noStates[child],noStates[parent1],noStates[parent2]], float )
# Coursework 3 task 1 should be inserted here
   

# End of Coursework 3 task 1           
    return cPT
#
# Definition of a Bayesian Network
def ExampleBayesianNetwork(theData, noStates):
    arcList = [[0],[1],[2,0],[3,2,1],[4,3],[5,3]]
    cpt0 = Prior(theData, 0, noStates)
    cpt1 = Prior(theData, 1, noStates)
    cpt2 = CPT(theData, 2, 0, noStates)
    cpt3 = CPT_2(theData, 3, 2, 1, noStates)
    cpt4 = CPT(theData, 4, 3, noStates)
    cpt5 = CPT(theData, 5, 3, noStates)
    cptList = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5]
    return arcList, cptList
# Coursework 3 task 2 begins here

# end of coursework 3 task 2
#
# Function to calculate the MDL size of a Bayesian Network
def MDLSize(arcList, cptList, noDataPoints, noStates):
    mdlSize = 0.0
# Coursework 3 task 3 begins here


# Coursework 3 task 3 ends here 
    return mdlSize 
#
# Function to calculate the joint probability of a single data point in a Network
def JointProbability(dataPoint, arcList, cptList):
    jP = 1.0
# Coursework 3 task 4 begins here


# Coursework 3 task 4 ends here 
    return jP
#
# Function to calculate the MDLAccuracy from a data set
def MDLAccuracy(theData, arcList, cptList):
    mdlAccuracy=0
# Coursework 3 task 5 begins here


# Coursework 3 task 5 ends here 
    return mdlAccuracy
#
# End of coursework 2
#
# Coursework 3 begins here
#
def Mean(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    mean = []
    # Coursework 4 task 1 begins here



    # Coursework 4 task 1 ends here
    return array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    covar = zeros((noVariables, noVariables), float)
    # Coursework 4 task 2 begins here


    # Coursework 4 task 2 ends here
    return covar
def CreateEigenfaceFiles(theBasis):
    adummystatement = 0 #delete this when you do the coursework
    # Coursework 4 task 3 begins here

    # Coursework 4 task 3 ends here

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 4 task 4 begins here

    # Coursework 4 task 4 ends here
    return array(magnitudes)

def CreatePartialReconstructions(aBasis, aMean, componentMags):
    adummystatement = 0  #delete this when you do the coursework
    # Coursework 4 task 5 begins here

    # Coursework 4 task 5 ends here

def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 4 task 3 begins here
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending 
    # order of their eignevalues magnitudes

    
    # Coursework 4 task 6 ends here
    return array(orthoPhi)

#
# main program part for Coursework 1
#

noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)
#AppendString("results.txt","Coursework One Results by sd3112")
#AppendString("results.txt","") #blank line
#AppendString("results.txt","The prior probability of node 0:")
#prior = Prior(theData, 0, noStates)
#AppendList("results.txt", prior)
#AppendString("results.txt","The conditional probability matrix P(2|0):")
#cPT = CPT(theData, 2, 0, noStates)
#AppendArray("results.txt", cPT)
#AppendString("results.txt","The joint probability matrix P(2&0):")
#jPT = JPT(theData, 2, 0, noStates)
#AppendArray("results.txt", jPT)
#AppendString("results.txt","The conditional probability matrx P(2|0) calculated from P(2&0):")
#newCPT = JPT2CPT(jPT)
#AppendArray("results.txt", newCPT)

# Create the Bayesian network
#naiveBayes = []
#naiveBayes.append(prior)
#for i in range(5):
#    naiveBayes.append(CPT(theData,i+1,0,noStates))

#AppendString("results.txt","The results of passing state [4,0,0,0,5] into the naive network:")
#posteriorProbs1 = Query([4,0,0,0,5],naiveBayes)
#AppendList("results.txt", posteriorProbs1)

#AppendString("results.txt","The results of passing state [6,5,2,5,5] into the naive network:")
#posteriorProbs2 = Query([6,5,2,5,5],naiveBayes)
#AppendList("results.txt", posteriorProbs2)


# *********** Coursework 2 *****************

AppendString("results.txt","Coursework Two Results by sd3112")
prior = Prior(theData, 0, noStates)
jPT = JPT(theData, 2, 0, noStates)
#print DependencyMatrix(theData, noVariables, noStates)
dm =  DependencyMatrix(theData, noVariables, noStates)
AppendString("results.txt","The dependency matrix for the HepatitisC varilabes:")
AppendArray("results.txt",dm)
dl = DependencyList(dm)
AppendString("results.txt","The dependency list for the HepatitisC varilabes:")
AppendArray("results.txt",dl)
