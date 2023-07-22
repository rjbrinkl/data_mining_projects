import pandas as pd
#import datetime
#from sklearn import svm
#from sklearn.metrics import classification_report
#import math
import pickle
#import csv
import math
import numpy




#The test.py reads test.csv which has the N x 24 matrix 



def timeToHighestGlucose(data, row):
    #time from start of meal to highest glucose level (in seconds)

    #handles both meal and no meal
    numColumns = len(data[0])
    
    #assume first data is highest due to meal intake; wait until next highest
    columnStart = 1
    if numColumns == 30:
        columnStart = 7
    
    #initially equal to 0
    highestGlucose = 0.0
    highestTime = 0.0
    
    for col in range(columnStart, numColumns):
        if float(data[row][col]) > highestGlucose:
            highestGlucose = float(data[row][col])
            highestTime = col
    
    #convert from 5 minute intervals to minutes, then to seconds (assume first record is 0)
    difference = (highestTime) * 5 * 60
    return difference
    
def differenceGlucose(data, row):
    numColumns = len(data[0])
    
    columnStart = 1
    if numColumns == 30:
        columnStart = 0
        
    highestGlucose = 0.0
    startGlucose = float(data[row][columnStart - 1])
    result = 0.0
    
    for col in range(columnStart, numColumns):
        if float(data[row][col]) > highestGlucose:
            highestGlucose = float(data[row][col])

    if highestGlucose > startGlucose:
        result = (highestGlucose - startGlucose)
    
    return result
    
def normalizedDifference(data, row):
    #normalized difference of start of meal glucose level to highest glucose level
    
    #handles both meal and no meal
    numColumns = len(data[0])
    
    #assume first data is highest due to meal intake; wait until next highest
    
    #initially equal to 0
    highestGlucose = float(data[row][0])
    startGlucose = highestGlucose
    result = 0.0
    
    for col in range(1, numColumns):
        if float(data[row][col]) > highestGlucose:
            highestGlucose = float(data[row][col])

    if highestGlucose < startGlucose:
        result = 0.0
    else:
        result = (highestGlucose - startGlucose)/startGlucose
    
    return result
    
#returns the max distance between two pieces of CGMData
def differentiation(data, row):
    
    temp = data[row]
    for i in range(len(data[row])):
        if math.isnan(float(temp[i])):
            continue
        temp[i] = int(float(temp[i]))
    
    diff = numpy.diff(temp)
    
    return (max(diff))
    
#returns the max distance after a double differentiation
def doubleDifferentiation(data, row):
    temp = data[row]
    for i in range(len(data[row])):
        if math.isnan(float(temp[i])):
            continue
        temp[i] = int(float(temp[i]))
        
    doubleDiff = numpy.diff(temp, n = 2)
    
    return (max(doubleDiff))
    
def fastFourier(data, row):
    temp = data[row]
    for i in range(len(data[row])):
        if math.isnan(float(temp[i])):
            continue
        temp[i] = int(float(temp[i]))
        
    fft = numpy.fft.rfft(temp)
    fftABS = abs(fft)
    fftFREQ = numpy.fft.rfftfreq(len(temp))

    fftABSList = fftABS.tolist()
    fftABSList2 = fftABS.tolist()

    fftABSList.remove(max(fftABSList))
    
    pf1 = max(fftABSList)
    
    if math.isnan(pf1):
        return ([numpy.nan, numpy.nan, numpy.nan, numpy.nan])
    
    f1 = fftFREQ[fftABSList2.index(pf1)]

    fftABSList.remove(pf1)
    
    pf2 = max(fftABSList)
    f2 = fftFREQ[fftABSList2.index(pf2)]

    results = [pf1, f1, pf2, f2]
    return (results)
    
def standardDeviation(data, row):
    temp = data[row]
    for i in range(len(data[row])):
        if math.isnan(float(temp[i])):
            continue
        temp[i] = int(float(temp[i]))
        
    return numpy.std(temp)
    
def maxGradient(data, row):
    temp = data[row]
    sumCalc = 0.0
    for i in range(len(data[row])):
        if math.isnan(float(temp[i])):
            continue
        temp[i] = int(float(temp[i]))
        
    grad = numpy.gradient(temp)
    maxGrad = max(abs(grad))

    return (maxGrad)

#extracts features, 
def extractFeatures(mealData):
    
    #P = length of meal data matrix
    mealCount = len(mealData)

    mealFeatureMatrix = []
    
    for row in range(mealCount):
        
        #1 x F matrix containing all of the features (F = number of features used (5-8))
        mealFeatures = []
        
        #time from start of meal to highest glucose level (in seconds)
        #mealFeatures.append(timeToHighestGlucose(mealData, row))
        
        #normalized difference of start of meal glucose level to highest glucose level
        mealFeatures.append(normalizedDifference(mealData, row))
        
        #mealFeatures.append(differenceGlucose(mealData, row))

        mealFeatures.append(differentiation(mealData, row))
        
        mealFeatures.append(doubleDifferentiation(mealData, row))
        
        mealFeatures.append(maxGradient(mealData, row))
        
        # fastF = fastFourier(mealData, row)
        # for i in range(len(fastF)):
            # mealFeatures.append(fastF[i])
        
        mealFeatureMatrix.append(mealFeatures)
    
    return mealFeatureMatrix





#and outputs a Result.csv file which has N x 1 vector of 1s and 0s, where 1 denotes meal, 0 denotes no meal.



def main():
    
    namelist = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
    CGMDataIn = pd.read_csv('./test.csv', names = namelist)
    
    # print(CGMDataIn)
    
    cgmdata = []
    for i, row in CGMDataIn.iterrows():
        data = [row['0'], row['1'], row['2'], row['3'], row['4'], row['5'], row['6'], row['7'], row['8'], 
        row['9'],row['10'],row['11'],row['12'],row['13'],row['14'],row['15'],row['16'],row['17'],row['18'],row['19'],
        row['20'],row['21'],row['22'],row['23']]
        cgmdata.append(data)
        
    # print(cgmdata)
    # print(cgmdata[0])
    # print(cgmdata[0][0])
    # with open('./test.csv', newline='') as file:
        # read = csv.reader(file)
        # CGMDataIn = list(read)

    test = 0
    if test != 1:
        featureMatrix = extractFeatures(cgmdata)
    else:
        featureMatrix = cgmdata
    # print(len(cgmdata))
    
    #print(featureMatrix)
    model = pickle.load(open('./model.sav', 'rb'))

    predictedLabel = model.predict(featureMatrix)
    
    #print(predictedLabel)
    
    results = pd.DataFrame(predictedLabel)
    #print(results)

    results.to_csv(r'./Result.csv', index = False, header = False)    


main()