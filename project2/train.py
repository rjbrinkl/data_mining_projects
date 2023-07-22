import pandas as pd
#import datetime
from sklearn import svm
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import math
import pickle
import numpy

def extractData(CGMPath, InsulinPath):

#The train.py reads CGMData.csv, CGM_patient2.csv and InsulinData.csv, Insulin_patient2.csv, 

    CGMDataIn = pd.read_csv(CGMPath, sep = ',', dtype = 'unicode', parse_dates = [['Date', 'Time']])
    CGMDataIn_descending = CGMDataIn.sort_index(ascending = False)
    
    InsulinDataIn = pd.read_csv(InsulinPath, sep = ',', dtype = 'unicode', parse_dates = [['Date', 'Time']])
    InsulinDataIn_descending = InsulinDataIn.sort_index(ascending = False)

#extracts meal and no-meal data, 
    
    #get only the columns we need from CGMData
    CGMDataIn_descending['Date'] = CGMDataIn_descending['Date_Time'].dt.date
    CGMDataIn_reduced = CGMDataIn_descending[['Index', 'Date_Time', 'Date', 'Sensor Glucose (mg/dL)']]
    #print(CGMDataIn_reduced)
    
    #filter out the list of rows with actual values in column Y of InsulinData(not NaN and > 0)
    InsulinDataIn_descending['Date'] = InsulinDataIn_descending['Date_Time'].dt.date
    InsulinDataIn_cleaned = InsulinDataIn_descending[['Index', 'Date_Time', 'Date', 'BWZ Carb Input (grams)']]
    InsulinDataIn_cleaned_notnull = InsulinDataIn_cleaned[(InsulinDataIn_cleaned['BWZ Carb Input (grams)'].notnull()) & (InsulinDataIn_cleaned['BWZ Carb Input (grams)'] > str(0))]
    #print("not equal to NaN or 0\n")
    #print(InsulinDataIn_cleaned_notnull)
    
    #get the list of unique days from the insulin data
    InsulinUniqueDays = InsulinDataIn_cleaned_notnull.Date.unique()

    #GET THE MEAL DATA
    ##################
    mealMatrix = []
    mealMatrixTimes = []
    noMealMatrix = []
    noMealMatrixTimes = []
    nomealincorrectLengthCount = 0
    mealIncorrectLengthCount = 0
    #loop through each of the unique days
    for day in InsulinUniqueDays:
    
        #get the list of meal times for the given day
        tempTimes = InsulinDataIn_cleaned_notnull[InsulinDataIn_cleaned_notnull['Date'] == day]
        times = tempTimes['Date_Time']
    
        #loop through all of the meal times for a given day
        for tm in times:
            #print("current timestamp under test: " + str(tm) + "\n")
            
            #check if the time is a valid meal time ##NEED TO DETERMINE IF MORE NECESSARY FOR THE 3RD BULLET
            #ALSO DO I NEED TO CHECK EXPLICITLY
            temp_len = (len(InsulinDataIn_cleaned_notnull[(InsulinDataIn_cleaned_notnull['Date'] == day) & ((InsulinDataIn_cleaned_notnull['Date_Time'] > (tm + pd.Timedelta('0 hours'))) 
            & (InsulinDataIn_cleaned_notnull['Date_Time'] < (tm + pd.Timedelta('2 hours'))))]))
            
            #if no other meal times were within the range, do something
            if temp_len == 0:
            
                #NEED TO ADD SOMETHING HERE TO USE THIS TIME IN THE CGM DATA TO GATHER THE MEAL DATA
                #print("there are no records in the range tm - tm+2hours for this record:" + str(tm) + "\n")
                
                #get the list of times after the meal time in CGMData
                CGMData_after_meal = CGMDataIn_reduced[(CGMDataIn_reduced['Date'] == day) & (CGMDataIn_reduced['Date_Time'] >= tm)]
                lenNextCGMTime = len(CGMData_after_meal)
                
                #check if the next reading after the meal time is not on the same day
                if lenNextCGMTime == 0:
                
                    #print("first CGMData time after current time not on same day")
                    nextday = pd.to_datetime(day)
                    oneday = pd.to_timedelta(pd.np.ceil(1), unit="D")
                    nextday = nextday + oneday
                    #get readings on the next day
                    CGMData_after_meal = CGMDataIn_reduced[(CGMDataIn_reduced['Date_Time'] >= (str(nextday) + ' 00:00:00'))]

                #get reading where we will start the meal data
                firstCGMDateTimeAfterMeal = CGMData_after_meal.iloc[0]['Date_Time']
                #print("first CGMData time after current time: " + str(firstCGMDateTimeAfterMeal))
                #print("\n")
                
                #should produce 31 rows of data (may be less if missing rows or times are off) (need to do something if less)
                CGMMealData = CGMDataIn_reduced[(CGMDataIn_reduced['Date_Time'] >= (firstCGMDateTimeAfterMeal - pd.Timedelta('31 minutes')))
                & (CGMDataIn_reduced['Date_Time'] < (firstCGMDateTimeAfterMeal + pd.Timedelta('2 hours 1 minute')))]
                
                lenCGMMealData = len(CGMMealData)
                test_CGMMealData = CGMMealData[0:(lenCGMMealData - 1)]
                lenCGMMealData = len(test_CGMMealData)
                
                cleanedMealData = test_CGMMealData[test_CGMMealData['Sensor Glucose (mg/dL)'].notnull()]
                lengthCleanedMealData = len(cleanedMealData)
                
                #print("length of cleaned out meal data (no NaN) is: " + str(lengthCleanedMealData) + "\n")
                
                #need to do something with NaN values in CGMData?
                
                #print(CGMMealData)
                
                #check if meal data received has length over 24 (missing 6 or less records)###########################
                if lengthCleanedMealData == 30:
                
                    #add meal data to dictionary to be added to a dataframe after all has been collected
                    mealList = cleanedMealData['Sensor Glucose (mg/dL)'].tolist()
                    timeList = cleanedMealData['Date_Time'].tolist()
                    mealMatrix.append(mealList)
                    mealMatrixTimes.append(timeList)
                    
                else:
                    mealIncorrectLengthCount += 1
                
                #NO MEAL DATA COLLECTION
                ########################
                startofNoMeal = CGMMealData.iloc[-1]['Date_Time']
                #print("start of no meal data from CGMData: " + str(startofNoMeal))
                
                #look in list of all times for first record after the start of no meal
                InsulinData_time_after_meal = InsulinDataIn_cleaned[InsulinDataIn_cleaned['Date_Time'] >= startofNoMeal]
                InsulinData_startofNoMeal = InsulinData_time_after_meal.iloc[0]['Date_Time']
                #print("first InsulinData that marks start of no meal: " + str(InsulinData_startofNoMeal))
                
                #look in list of all meal times for next meal
                test_MealsInNoMealRange = InsulinDataIn_cleaned_notnull[InsulinDataIn_cleaned_notnull['Date_Time'] >= InsulinData_startofNoMeal]
                
                #check if at end of dataset (no more meals to come)
                if len(test_MealsInNoMealRange) > 0:
                    InsulinData_nextMeal = test_MealsInNoMealRange.iloc[0]['Date_Time']
                    #print("next meal after start of no meal: " + str(InsulinData_nextMeal))
                    
                    hoursBetweenNoMealStartandNextMeal = int(pd.Timedelta(InsulinData_nextMeal 
                    - InsulinData_startofNoMeal).total_seconds() / 3600.0)
                    
                else:
                    #no more meals
                    lastInsulinTime = InsulinDataIn_cleaned.iloc[-1]['Date_Time']
                    
                    #get the number of whole hours between the no meal start and next meal
                    hoursBetweenNoMealStartandNextMeal = int(pd.Timedelta(lastInsulinTime 
                    - InsulinData_startofNoMeal).total_seconds() / 3600.0)
                    
                #get the number of whole hours between the no meal start and next meal

                #print("number of hours between start of no meal and next meal time: " + str(hoursBetweenNoMealStartandNextMeal))
                
                #hours / 2 is the number of no meal sequences possible until the next meal time
                numberOfNoMealSequences = int(hoursBetweenNoMealStartandNextMeal / 2)
                #print("number of no meal sequences possible: " + str(numberOfNoMealSequences))
                
                firstCGMNoMealTime = CGMDataIn_reduced[CGMDataIn_reduced['Date_Time'] >= InsulinData_startofNoMeal]
                firstCGMNoMealTime = firstCGMNoMealTime.iloc[0]['Date_Time']
                #print("start of no meal in CGMData: " + str(firstCGMNoMealTime))
                
                if numberOfNoMealSequences > 0:
                
                    for sequence in range(numberOfNoMealSequences):
                    
                        #print("iteration #" + str(sequence) + " of no meal data starting at: " + str(firstCGMNoMealTime))
                        
                        noMealData = CGMDataIn_reduced[(CGMDataIn_reduced['Date_Time'] >= firstCGMNoMealTime) 
                        & (CGMDataIn_reduced['Date_Time'] < (firstCGMNoMealTime + pd.Timedelta('2 hours')))]
                        
                        if len(noMealData) != 24:
                            nomealincorrectLengthCount += 1
                        else:
                            #interpolate the missing values in the no meal data
                            conversion = {'Sensor Glucose (mg/dL)': float}
                            noMealData = noMealData.astype(conversion)
                            noMealData = noMealData.interpolate(method = 'linear', limit_direction = 'both')
                            #add data to list
                            mealData = noMealData['Sensor Glucose (mg/dL)'].tolist()
                            mealTimes = noMealData['Date_Time'].tolist()
                            noMealMatrix.append(mealData)
                            noMealMatrixTimes.append(mealTimes)
                        
                        if (sequence != numberOfNoMealSequences - 1):
                            lastRecord = noMealData.iloc[-1]['Date_Time']
                            #print("sequence: " + str(sequence) + " number of no meal sequences: " + str(numberOfNoMealSequences))
                            #print("last record: " + str(lastRecord))
                            nextNoMeal = CGMDataIn_reduced[CGMDataIn_reduced['Date_Time'] > lastRecord]
                            #print("nextNoMeal: " + str(nextNoMeal))
                            if (len(nextNoMeal) > 0):
                                firstCGMNoMealTime = nextNoMeal.iloc[0]['Date_Time']
                
    return mealMatrix, mealMatrixTimes, noMealMatrix, noMealMatrixTimes

def timeToHighestGlucose(data, times, row):
    #time from start of meal to highest glucose level (in seconds)

    #handles both meal and no meal
    numColumns = len(data[0])
    
    #assume first data is highest due to meal intake; wait until next highest
    columnStart = 1
    if numColumns == 30:
        columnStart = 7
    
    #initially equal to 0
    highestGlucose = 0.0
    highestTime = times[row][columnStart - 1]
    
    for col in range(columnStart, numColumns):
        if float(data[row][col]) > highestGlucose:
            highestGlucose = float(data[row][col])
            highestTime = times[row][col]

    return (highestTime - times[row][columnStart - 1]).total_seconds()
    
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
    columnStart = 1
    if numColumns == 30:
        columnStart = 7
    
    #initially equal to 0
    highestGlucose = 0.0
    startGlucose = float(data[row][columnStart - 1])
    result = 0.0
    
    for col in range(columnStart, numColumns):
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
        
    std = numpy.std(temp, dtype = numpy.float32)

    return std
    
def average(data, row):
    temp = data[row]
    sumCalc = 0.0
    for i in range(len(data[row])):
        if math.isnan(float(temp[i])):
            continue
        sumCalc += int(float(temp[i]))
        
    avg = sumCalc / len(data[row])
    return avg

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
def extractFeatures(mealData, mealTimes, noMealData, noMealTimes):
    
    #P = length of meal data matrix
    mealCount = len(mealData)
    #Q = length of no meal data matrix
    noMealCount = len(noMealData)
    
    #P x F feature matrix
    mealFeatureMatrix = []
    #Q x F feature matrix
    noMealFeatureMatrix = []
    
    
    for row in range(mealCount):
        
        #1 x F matrix containing all of the features (F = number of features used (5-8))
        mealFeatures = []
        
        #time from start of meal to highest glucose level (in seconds)
        #mealFeatures.append(timeToHighestGlucose(mealData, mealTimes, row))
        #normalized difference of start of meal glucose level to highest glucose level
        mealFeatures.append(normalizedDifference(mealData, row))
        
        #mealFeatures.append(differenceGlucose(mealData, row))
        
        mealFeatures.append(differentiation(mealData, row))
        
        mealFeatures.append(doubleDifferentiation(mealData, row))
        
        #mealFeatures.append(standardDeviation(mealData, row))
        
        #mealFeatures.append(average(mealData, row))
        
        mealFeatures.append(maxGradient(mealData, row))
        
        # fastF = fastFourier(mealData, row)
        # for i in range(len(fastF)):
            # mealFeatures.append(fastF[i])
        
        skip = 0
        for i in range(len(mealFeatures)):
            if math.isnan(mealFeatures[i]):
                skip = 1
            
        if skip == 1:
            continue
        
        mealFeatureMatrix.append(mealFeatures)
    
    print ("meal feature extraction completed")
    
    for row in range(noMealCount):

        #1 x F matrix containing all of the features (F = number of features used (5-8))
        noMealFeatures = []
        
        #time from start of meal to highest glucose level (in seconds)
        #noMealFeatures.append(timeToHighestGlucose(noMealData, noMealTimes, row))
        #normalized difference of start of meal glucose level to highest glucose level
        noMealFeatures.append(normalizedDifference(noMealData, row))
        
        #noMealFeatures.append(differenceGlucose(noMealData, row))
        
        noMealFeatures.append(differentiation(noMealData, row))
        
        noMealFeatures.append(doubleDifferentiation(noMealData, row))
        
        #noMealFeatures.append(standardDeviation(noMealData, row))
        
        #noMealFeatures.append(average(noMealData, row))
        
        noMealFeatures.append(maxGradient(noMealData, row))
        
        # fastF = fastFourier(noMealData, row)
        # for i in range(len(fastF)):
            # noMealFeatures.append(fastF[i])
        
        skip = 0
        for i in range(len(noMealFeatures)):
            if math.isnan(noMealFeatures[i]):
                skip = 1
            
        if skip == 1:
            continue

        noMealFeatureMatrix.append(noMealFeatures)

    print ("feature extraction finished")

    return mealFeatureMatrix, noMealFeatureMatrix


#trains your machine to recognize meal and no-meal classes, 








#stores the machine in a pickle file (Python API pickle)




#main method execution
def main():

    #get meal and no meal data from the two datasets
    mealData1, mealData1Times, noMealData1, noMealData1Times = extractData('./CGMData.csv', './InsulinData.csv')
    mealData2, mealData2Times, noMealData2, noMealData2Times = extractData('./CGM_patient2.csv', './Insulin_patient2.csv')
    
    #concatenate the two datasets into one
    mealData = mealData1 + mealData2
    noMealData = noMealData1 + noMealData2
    mealTimes = mealData1Times + mealData2Times
    noMealTimes = noMealData1Times + noMealData2Times

    
    #get the feature matrix of both meal and no meal
    mealFeature, noMealFeature = extractFeatures(mealData, mealTimes, noMealData, noMealTimes)
    
    #get length of meal and no meal data to create class label matrix
    mealLength = len(mealFeature)      # = P
    noMealLength = len(noMealFeature)  # = Q
    
    print("meal length" + str(mealLength))
    print("no meal length" + str(noMealLength))
    
    #create class label matrix with P 1's followed by Q 0's
    mealLabel = []
    noMealLabel = []
    for i in range(mealLength):
        mealLabel.append(1)
    for i in range(noMealLength):
        noMealLabel.append(0)
    classLabel = mealLabel + noMealLabel

    #split data for meal: mealTrain is first 80%, mealTest is last 20%
    mealSplitLength = int(0.8 * (int(mealLength)))
    mealTrain = mealFeature[:mealSplitLength]
    mealTest = mealFeature[mealSplitLength:]
    
    #split data for noMeal: noMealTrain is first 80%, noMealTest is last 20%
    noMealSplitLength = int(0.8 * (int(noMealLength)))
    noMealTrain = noMealFeature[:noMealSplitLength]
    noMealTest = noMealFeature[noMealSplitLength:]
    
    #split data for class label for both meal and noMeal: 
    featureSplitLength = int(0.8 * (int(mealLength)))
    mealFeatureTrain = mealLabel[:featureSplitLength]
    mealFeatureTest = mealLabel[featureSplitLength:]
    
    featureSplitLength = int(0.8 * (int(noMealLength)))
    noMealFeatureTrain = noMealLabel[:featureSplitLength]
    noMealFeatureTest = noMealLabel[featureSplitLength:]
    
    #collect train and test sets
    trainSet = mealTrain + noMealTrain
    testSet = mealTest + noMealTest
    labelTrainSet = mealFeatureTrain + noMealFeatureTrain
    labelTestSet = mealFeatureTest + noMealFeatureTest
    
    SVM = svm.SVC(gamma = 'scale')
    model = SVM.fit(trainSet, labelTrainSet)
    predictedLabel = model.predict(testSet)
    
    # dTree = tree.DecisionTreeClassifier(criterion = 'entropy')
    # model = dTree.fit(trainSet, labelTrainSet)
    # predictedLabel = model.predict(testSet)
    
    #print(predictedLabel)
    
    print(classification_report(labelTestSet, predictedLabel, labels = [1, 0]))
    
    pickle.dump(model, open('./model.sav', 'wb'))
    
main()