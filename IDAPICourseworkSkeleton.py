#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from IDAPICourseworkLibrary import *
from numpy import *
from math import *
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
        if(probVarCol[col] != 0):
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

# I use Kruskal's algorithm but to find the maximal spanning tree
# rather than the more conventional minimal spanning tree. To do this
# I use the helper functions find() and union().
def SpanningTreeAlgorithm(depList, noVariables):
    spanningTree = []
    # Create a dictionary where each node points to itself
    # (initially each node is isolated)
    C = {u:u for u in range(noVariables)}
    for i in range(depList.shape[0]):
        if find(C,depList[i][1]) != find(C,depList[i][2]): 
            spanningTree.append(depList[i])
            #print depList[i]
            union(C,depList[i][1],depList[i][2])
    return array(spanningTree)

# Get the 'representative' component
def find(C,n):
    while C[n] != n:
        n = C[n]
    return n;

# Nodes n and m will become part of the same
# connected component
def union(C,n,m):
    n = find(C,n)
    m = find(C,m)
    C[n] = m

#
# End of coursework 2
#
# Coursework 3 begins here
#
# Function to compute a CPT with multiple parents from the data set
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT_2(theData, child, parent1, parent2, noStates):
    cPT = zeros([noStates[child],noStates[parent1],noStates[parent2]], float ) # 
# Coursework 3 task 1 should be inserted here
    parentCount = zeros([noStates[parent1],noStates[parent2]], float)
    # Need to calculate the probability of the child event for every parent combination   
    for row in range(theData.shape[0]):
        cPT[theData[row][child]][theData[row][parent1]][theData[row][parent2]]+=1
        parentCount[theData[row][parent1]][theData[row][parent2]]+=1
    # Now normalise
    for dim1 in range(cPT.shape[0]):
      for dim2 in range(cPT.shape[1]):
        for dim3 in range(cPT.shape[2]):
            if parentCount[dim2][dim3] != 0:
                cPT[dim1][dim2][dim3]/=parentCount[dim2][dim3]
    return cPT

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

def HepCBayesianNetwork(theData, noStates):
    arcList = [[0],[1],[2,0],[3,4],[4,1],[5,4],[6,1],[7,0,1],[8,7]]
    cpt0 = Prior(theData,0,noStates)
    cpt1 = Prior(theData,1,noStates)
    cpt2 = CPT(theData,2,0,noStates)
    cpt3 = CPT(theData,3,4,noStates)
    cpt4 = CPT(theData,4,1,noStates)
    cpt5 = CPT(theData,5,4,noStates)
    cpt6 = CPT(theData,6,1,noStates)
    cpt7 = CPT_2(theData,7,0,1,noStates)
    cpt8 = CPT(theData,8,7,noStates)
    cptList = [cpt0,cpt1,cpt2,cpt3,cpt4,cpt5,cpt6,cpt7,cpt8]
    return arcList, cptList

# end of coursework 3 task 2
#
# Function to calculate the MDL size of a Bayesian Network
def MDLSize(arcList, cptList, noDataPoints, noStates):
    mdlSize = 0.0
# Coursework 3 task 3 begins here
    for i in range(len(arcList)):
        if len(arcList[i]) == 1:
            mdlSize += noStates[i]-1
        elif len(arcList[i]) == 2:
            mdlSize += (noStates[i]-1)*(noStates[arcList[i][1]])
        else:
            mdlSize += (noStates[i]-1)*(noStates[arcList[i][1]])*(noStates[arcList[i][2]])
    mdlSize *= log2(noDataPoints)/2
# Coursework 3 task 3 ends here 
    return mdlSize 
#
# Function to calculate the joint probability of a single data point in a Network
def JointProbability(dataPoint, arcList, cptList):
    jP = 1.0
# Coursework 3 task 4 begins here
    for i in range(len(cptList)):
        if len(arcList[i]) == 1: 
            jP *= cptList[i][dataPoint[arcList[i][0]]]
        elif len(arcList[i]) == 2:
            jP *= cptList[i][dataPoint[i]][dataPoint[arcList[i][1]]]
        else:
            jP *= cptList[i][dataPoint[i]][dataPoint[arcList[i][1]]][dataPoint[arcList[i][2]]]
# Coursework 3 task 4 ends here 
    return jP
#
# Function to calculate the MDLAccuracy from a data set
def MDLAccuracy(theData, arcList, cptList):
    mdlAccuracy=0
# Coursework 3 task 5 begins here
    for row in range(theData.shape[0]):
        jp = JointProbability(theData[row],arcList,cptList)
        if jp != 0:
            mdlAccuracy += log2(jp)    
# Coursework 3 task 5 ends here 
    return mdlAccuracy

# I assume that graph supplied to this function is a spanning tree
def BestScoringNetwork(theData, arcList, cptList, noStates):
    
    minScore = 0
    minArcList = []
    for i in range(len(arcList)):
        if len(arcList[i]) == 2:
            tempArcList = list(arcList)
            tempCptList = list(cptList)
            tempArcList[i] = [ arcList[i][0] ]
            tempCptList[i] = Prior(theData,arcList[i][0],noStates)
            mdlSize = MDLSize(tempArcList, tempCptList, theData.shape[0], noStates)
            mdlAcc  = MDLAccuracy(theData, tempArcList, tempCptList)
            score = mdlSize - mdlAcc
            if minScore == 0 or score < minScore:
                minScore = score
                minArcList = tempArcList
        if len(arcList[i]) == 3:
            for k in [1,2]:
            # We should not delete the 0th entry in the sub list
            # as this would violate the structure of cptList
                tempArcList = list(arcList)
                tempCptList = list(cptList)
                tempArcList[i] = [ arcList[i][0], arcList[i][k] ]
                tempCptList[i] = CPT(theData,arcList[i][0],arcList[i][k],noStates)
                mdlSize = MDLSize(tempArcList, tempCptList, theData.shape[0], noStates)
                mdlAcc  = MDLAccuracy(theData, tempArcList, tempCptList)
                score = mdlSize - mdlAcc
                if minScore == 0 or score < minScore:
                    minScore = score
                    minArcList = tempArcList
    
    return minScore, minArcList

#
# End of coursework 3
#
# Coursework 4 begins here
#
def Mean(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    mean = []
    # Coursework 4 task 1 begins here
    noDataPoints = theData.shape[0]
    for i in range(noVariables):
        sum = 0.0
        for j in range(noDataPoints):
            sum += theData[j][i]
        mean.append(sum/noDataPoints)
    # Coursework 4 task 1 ends here
    return array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    covar = zeros((noVariables, noVariables), float)
    # Coursework 4 task 2 begins here
    noDataPoints = theData.shape[0]
    mean = Mean(realData)
    for i in range(noVariables):
        for j in range(noVariables):
            sum = 0
            for k in range(noDataPoints):
                sum+= (realData[k][i]-mean[i])*(realData[k][j]-mean[j])
            covar[i][j] = sum/(noDataPoints - 1)       
    # Coursework 4 task 2 ends here
    return covar

def CreateEigenfaceFiles(): #theBasis
    # Coursework 4 task 3 begins here
    eigFaces = ReadEigenfaceBasis()
    for i in range(10):
        filename = "PrincipleComponent" + str(i) + ".jpg"
        SaveEigenface(eigFaces[i],filename)


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

# *********** Coursework 1 *****************

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

#AppendString("results.txt","Coursework Two Results by sd3112")
#jPT = JPT(theData, 2, 0, noStates)
#print DependencyMatrix(theData, noVariables, noStates)
#dm =  DependencyMatrix(theData, noVariables, noStates)
#AppendString("results.txt","The dependency matrix for the HepatitisC data set:")
#AppendArray("results.txt",dm)
#dl = DependencyList(dm)
#AppendString("results.txt","The dependency list for the HepatitisC data set:")
#AppendArray("results.txt",dl)
#mst = SpanningTreeAlgorithm(dl,noVariables)
#AppendString("results.txt","Maximally Weighted Spanning Tree for the HepatitisC data set:")
#AppendArray("results.txt",mst)

# *********** Coursework 3 *****************

#AppendString("results.txt","Coursework Three Results by sd3112 \n")

#arcList, cptList = HepCBayesianNetwork(theData,noStates)

#mdlSize = MDLSize(arcList, cptList, noDataPoints, noStates)
#output = "MDLSize     = " + str(mdlSize)
#print output
#AppendString("results.txt", output)

#mdlAccuracy = MDLAccuracy(theData, arcList, cptList)
#output = "MDLAccuracy = " + str(mdlAccuracy)
#print output
#AppendString("results.txt", output)

#mdlScore = mdlSize - mdlAccuracy
#output = "MDLScore    = " + str(mdlScore) + "\n"
#print output
#AppendString("results.txt", output)

#bestScore, bestNet = BestScoringNetwork(theData, arcList, cptList, noStates)
#output1 = "Best Score  = " + str(bestScore)
#output2 = "The best score in the case of the Hep C data came from: \n " + str(bestNet)
#print output1
#print output2
#AppendString("results.txt",output1)
#AppendString("results.txt",output2)

# *********** Coursework 4 *****************

AppendString("results.txt","Coursework Four Results by sd3112 \n")
#print theData
#print Mean(theData)
#print Covariance(theData)
eig = ReadEigenfaceBasis()
# 10 * 10304 matrix
print "ReadEigenfaceBasis() returns an array of size " 
print eig.shape[0]
print "rows by "
print eig.shape[1]
print "columns"
CreateEigenfaceFiles()
#AppendArray("results.txt",eig)
