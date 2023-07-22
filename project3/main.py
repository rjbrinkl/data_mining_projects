import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
from matplotlib import pyplot as plt
from sklearn import metrics
import math
import numpy

def extractData():

    #read in CGMData and InsulinData
    CGMDataIn = pd.read_csv('./CGMData.csv', sep = ',', dtype = 'unicode', parse_dates = [['Date', 'Time']])
    CGMDataIn_descending = CGMDataIn.sort_index(ascending = False)
    
    InsulinDataIn = pd.read_csv('./InsulinData.csv', sep = ',', dtype = 'unicode', parse_dates = [['Date', 'Time']])
    InsulinDataIn_descending = InsulinDataIn.sort_index(ascending = False)

    #extract specific day and sort in order of date
    CGMDataIn_descending['Date'] = CGMDataIn_descending['Date_Time'].dt.date
    CGMDataIn_reduced = CGMDataIn_descending[['Index', 'Date_Time', 'Date', 'Sensor Glucose (mg/dL)']]
    
    #extract specific day, sort in order of date
    InsulinDataIn_descending['Date'] = InsulinDataIn_descending['Date_Time'].dt.date
    InsulinDataIn_cleaned = InsulinDataIn_descending[['Index', 'Date_Time', 'Date', 'BWZ Carb Input (grams)']]
    #remove null and 0 entries
    InsulinDataIn_cleaned_notnull = InsulinDataIn_cleaned[(InsulinDataIn_cleaned['BWZ Carb Input (grams)'].notnull())]
    InsulinDataIn_cleaned_notnull = InsulinDataIn_cleaned_notnull.astype({"BWZ Carb Input (grams)": float})
    InsulinDataIn_cleaned_notnull = InsulinDataIn_cleaned_notnull[(InsulinDataIn_cleaned_notnull['BWZ Carb Input (grams)'] > 0.0)]

    #get max meal intake
    maxI = InsulinDataIn_cleaned_notnull['BWZ Carb Input (grams)'].max()

    #get min meal intake
    minI = InsulinDataIn_cleaned_notnull['BWZ Carb Input (grams)'].min()
    
    #calculate number of bins
    binSize = 20
    numBins = int((maxI - minI) / 20)
    
    #bins become min + (20 * i) -> min + (20 * i+1) - 1
    bins = []
    for i in range(numBins):
        bins.append(minI + (20*i)-1)
    bins.append(maxI)
    
    #put the meal intake data into bins
    InsulinBinned = InsulinDataIn_cleaned_notnull
    binLabels = [0,1,2,3,4,5]
    InsulinBinned['bins'] = pd.cut(InsulinBinned['BWZ Carb Input (grams)'], bins=bins, labels=binLabels)

    #get the list of unique days from the insulin data
    InsulinUniqueDays = InsulinBinned.Date.unique()

    #GET THE MEAL DATA
    ##################
    mealMatrix = []
    mealMatrixTimes = []
    insulinMatrix = []
    insulinBinMatrix = []
    mealIncorrectLengthCount = 0
    #loop through each of the unique days
    for day in InsulinUniqueDays:
    
        #get the list of meal times for the given day
        tempTimes = InsulinBinned[InsulinBinned['Date'] == day]
        times = tempTimes['Date_Time']
    
        #loop through all of the meal times for a given day
        for tm in times:
            
            #check if the time is a valid meal time
            temp_len = (len(InsulinBinned[(InsulinBinned['Date'] == day) & ((InsulinBinned['Date_Time'] > (tm + pd.Timedelta('0 hours'))) 
            & (InsulinBinned['Date_Time'] < (tm + pd.Timedelta('2 hours'))))]))
            
            #if no other meal times were within the range, do something
            if temp_len == 0:
                
                #get the list of times after the meal time in CGMData
                CGMData_after_meal = CGMDataIn_reduced[(CGMDataIn_reduced['Date'] == day) & (CGMDataIn_reduced['Date_Time'] >= tm)]
                lenNextCGMTime = len(CGMData_after_meal)
                
                #check if the next reading after the meal time is not on the same day
                if lenNextCGMTime == 0:
                    nextday = pd.to_datetime(day)
                    oneday = pd.to_timedelta(pd.np.ceil(1), unit="D")
                    nextday = nextday + oneday
                    #get readings on the next day
                    CGMData_after_meal = CGMDataIn_reduced[(CGMDataIn_reduced['Date_Time'] >= (str(nextday) + ' 00:00:00'))]

                #get reading where we will start the meal data
                firstCGMDateTimeAfterMeal = CGMData_after_meal.iloc[0]['Date_Time']

                #should produce 31 rows of data (may be less if missing rows or times are off) (need to do something if less)
                CGMMealData = CGMDataIn_reduced[(CGMDataIn_reduced['Date_Time'] >= (firstCGMDateTimeAfterMeal - pd.Timedelta('31 minutes')))
                & (CGMDataIn_reduced['Date_Time'] < (firstCGMDateTimeAfterMeal + pd.Timedelta('2 hours 1 minute')))]
                
                lenCGMMealData = len(CGMMealData)
                test_CGMMealData = CGMMealData[0:(lenCGMMealData - 1)]
                lenCGMMealData = len(test_CGMMealData)
                
                cleanedMealData = test_CGMMealData[test_CGMMealData['Sensor Glucose (mg/dL)'].notnull()]
                lengthCleanedMealData = len(cleanedMealData)

                #need to do something with NaN values in CGMData?
                
                #check if meal data received has length over 24 (missing 6 or less records)
                if lengthCleanedMealData == 30:
                
                    #add meal data to dictionary to be added to a dataframe after all has been collected
                    mealList = cleanedMealData['Sensor Glucose (mg/dL)'].tolist()
                    timeList = cleanedMealData['Date_Time'].tolist()
                    mealMatrix.append(mealList)
                    mealMatrixTimes.append(timeList)
                    
                    #add associated meal intake amount to list
                    InsulinMeal = InsulinBinned[(InsulinBinned['Date'] == day) & (InsulinBinned['Date_Time'] == tm)]
                    mealAmount = InsulinMeal.iloc[0]['BWZ Carb Input (grams)']
                    insulinMatrix.append(mealAmount)
                    #add associated insulin bin to the list
                    insulinBin = InsulinMeal.iloc[0]['bins']
                    insulinBinMatrix.append(insulinBin)
                    

                    
                else:
                    mealIncorrectLengthCount += 1
                    
                    
    return mealMatrix, mealMatrixTimes, insulinMatrix, insulinBinMatrix, numBins

def timeToHighestGlucose(data, times, row):
    #time from start of meal to highest glucose level (in seconds)

    #handles both meal and no meal
    numColumns = len(data[0])
    
    #assume first data is highest due to meal intake; wait until next highest
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

def maxminDiff(data, row):
    numColumns = len(data[0])
    
    columnStart = 0
        
    highestGlucose = 0.0
    lowestGlucose = float(data[row][columnStart])
    result = 0.0
    
    for col in range(columnStart, numColumns):
        if float(data[row][col]) > highestGlucose:
            highestGlucose = float(data[row][col])
        if float(data[row][col]) < lowestGlucose:
            lowestGlucose = float(data[row][col])

    if highestGlucose > lowestGlucose:
        result = (highestGlucose - lowestGlucose)
    else:
        result = 0
    
    return result
    
def maxminDiffNorm(data, row):
    numColumns = len(data[0])
    
    columnStart = 0
        
    highestGlucose = 0.0
    lowestGlucose = float(data[row][columnStart])
    result = 0.0
    
    for col in range(columnStart, numColumns):
        if float(data[row][col]) > highestGlucose:
            highestGlucose = float(data[row][col])
        if float(data[row][col]) < lowestGlucose:
            lowestGlucose = float(data[row][col])

    if highestGlucose > lowestGlucose and lowestGlucose > 0:
        result = (highestGlucose - lowestGlucose)/lowestGlucose
    else:
        result = 0
    
    return result

#extracts features, 
def extractFeatures(mealData, mealTimes):
    
    #P = length of meal data matrix
    mealCount = len(mealData)
    
    #P x F feature matrix
    mealFeatureMatrix = []

    for row in range(mealCount):
        
        #1 x F matrix containing all of the features (F = number of features used (5-8))
        mealFeatures = []
        
        #mealFeatures.append(timeToHighestGlucose(mealData, mealTimes, row))

        mealFeatures.append(normalizedDifference(mealData, row))####
        
        #mealFeatures.append(differenceGlucose(mealData, row))
        
        mealFeatures.append(differentiation(mealData, row))####
        
        mealFeatures.append(doubleDifferentiation(mealData, row))####
        
        mealFeatures.append(maxGradient(mealData, row))####
        
        #mealFeatures.append(maxminDiff(mealData, row))##
        
        #mealFeatures.append(maxminDiffNorm(mealData, row))##
        
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
    
    #print ("meal feature extraction completed")

    return mealFeatureMatrix

def dbscanSSEAddVals(cluster, values):
    if values[0] is not None:
        cluster[0] += values[0]
    if values[1] is not None:
        cluster[1] += values[1]
    if values[2] is not None:
        cluster[2] += values[2] #change depending on number of features used (number of values = number of features)
    if values[3] is not None:
        cluster[3] += values[3]

    return cluster
    
def dbscanSSECalc(labels, mealFeature):
    cluster0 = [0, 0, 0, 0] #change depending on number of features used (number of 0s = number of features))
    cluster1 = [0, 0, 0, 0]
    cluster2 = [0, 0, 0, 0]
    cluster3 = [0, 0, 0, 0]
    cluster4 = [0, 0, 0, 0]
    cluster5 = [0, 0, 0, 0]
    cluster0Count = 0
    cluster1Count = 0
    cluster2Count = 0
    cluster3Count = 0
    cluster4Count = 0
    cluster5Count = 0
    
    for i in range(len(labels)):
        if labels[i] == 0:
            cluster0 = dbscanSSEAddVals(cluster0, mealFeature[i])
            cluster0Count += 1
        if labels[i] == 1:
            cluster1 = dbscanSSEAddVals(cluster1, mealFeature[i])
            cluster1Count += 1
        if labels[i] == 2:
            cluster2 = dbscanSSEAddVals(cluster2, mealFeature[i])
            cluster2Count += 1
        if labels[i] == 3:
            cluster3 = dbscanSSEAddVals(cluster3, mealFeature[i])
            cluster3Count += 1
        if labels[i] == 4:
            cluster4 = dbscanSSEAddVals(cluster4, mealFeature[i])
            cluster4Count += 1
        if labels[i] == 5:
            cluster5 = dbscanSSEAddVals(cluster5, mealFeature[i])
            cluster5Count += 1

    #calculate the centroid for each cluster (average)
    for i in range(len(mealFeature[0])):
        if cluster0Count != 0:
            cluster0[i] = cluster0[i]/cluster0Count
        else:
            cluster0[i] = 0
        if cluster1Count != 0:
            cluster1[i] = cluster1[i]/cluster1Count
        else:
            cluster1[i] = 0
        if cluster2Count != 0:
            cluster2[i] = cluster2[i]/cluster2Count
        else:
            cluster2[i] = 0
        if cluster3Count != 0:
            cluster3[i] = cluster3[i]/cluster3Count
        else:
            cluster3[i] = 0
        if cluster4Count != 0:
            cluster4[i] = cluster4[i]/cluster4Count
        else:
            cluster4[i] = 0
        if cluster5Count != 0:
            cluster5[i] = cluster5[i]/cluster5Count
        else:
            cluster5[i] = 0

    dbscanSSE = 0
    for i in range(len(labels)):
        if labels[i] == 0 and cluster0Count > 0:
            dbscanSSE += (numpy.linalg.norm(numpy.array(cluster0) - numpy.array(mealFeature[i])) ** 2)
        if labels[i] == 1 and cluster1Count > 0:
            dbscanSSE += (numpy.linalg.norm(numpy.array(cluster1) - numpy.array(mealFeature[i])) ** 2)
        if labels[i] == 2 and cluster2Count > 0:
            dbscanSSE += (numpy.linalg.norm(numpy.array(cluster2) - numpy.array(mealFeature[i])) ** 2)
        if labels[i] == 3 and cluster3Count > 0:
            dbscanSSE += (numpy.linalg.norm(numpy.array(cluster3) - numpy.array(mealFeature[i])) ** 2)
        if labels[i] == 4 and cluster4Count > 0:
            dbscanSSE += (numpy.linalg.norm(numpy.array(cluster4) - numpy.array(mealFeature[i])) ** 2)
        if labels[i] == 5 and cluster5Count > 0:
            dbscanSSE += (numpy.linalg.norm(numpy.array(cluster5) - numpy.array(mealFeature[i])) ** 2)
            
    return dbscanSSE
    
def entropyCalc(confMatrix):

    countOfEach = [0,0,0,0,0,0]
    totalCount = 0
    
    start = 0
    matrixLength = len(confMatrix)
    if matrixLength > 6:
        start = 1
        
    for i in range(start, matrixLength):
        if i < 7:
            countOfEach[i] = confMatrix[i][0] + confMatrix[i][1] + confMatrix[i][2] + confMatrix[i][3] + confMatrix[i][4] + confMatrix[i][5]
            totalCount += countOfEach[i]
    
    tempSeries = confMatrix
    
    #print (countOfEach)

    for i in range(start, matrixLength):
        if i < 7:
            for j in range(6):
                if countOfEach[i] > 0:
                    tempSeries[i][j] = tempSeries[i][j] / countOfEach[i]
                else:
                    tempSeries[i][j] = 0

    #print (tempSeries)
    
    eachClusterEntropy = []
    for i in range(start, matrixLength):
        if i < 7:
            eachClusterEntropy.append(entropy(tempSeries[i], base = 2))
    
    #print (eachClusterEntropy)
    
    entropySum = 0
    for i in range(6):
        entropySum += (eachClusterEntropy[i] * countOfEach[i])
        #print(entropySum)
        
    #print(totalCount)
    if totalCount > 0:
        return (entropySum / totalCount)
    else:
        return 0

def main():

    mealMatrix, mealMatrixTimes, insulinMatrix, insulinBinMatrix, numBins = extractData()
    
    #get the feature matrix of meal data
    mealFeature = extractFeatures(mealMatrix, mealMatrixTimes)
    
    #Kmeans:SSE, DBSCAN:SSE, Kmeans:Entropy, DBSCAN:Entropy, Kmeans:Purity, DBSCAN:Purity
    result = []
    
    dbscanModel = DBSCAN(eps = 3, min_samples = 4) #3/4 = my 4 features, 2.2/4 = 2 rec
    ss = StandardScaler()
    mealFeature2 = ss.fit_transform(mealFeature)
    dbscanFit = dbscanModel.fit(mealFeature)
    y_pred = dbscanFit.fit_predict(mealFeature)
    lab = dbscanFit.labels_
    #plt.scatter(lab[:,0], lab[:,1], c='black', cmap='Paired')
    plt.scatter(mealFeature2[:,0], mealFeature2[:,1], c=lab, cmap='Paired')
    plt.title("DBSCAN")
    print(dbscanFit.labels_)
    plt.show()
    
    # nn = NearestNeighbors(n_neighbors=2)
    # nnOutput = nn.fit(mealFeature)
    # nnDistances, nnIndex = nnOutput.kneighbors(mealFeature)
    
    # nnDistances = numpy.sort(nnDistances, axis=0)
    # nnDistances = nnDistances[:,1]
    # plt.plot(nnDistances)
    # plt.xlabel("Index")
    # plt.ylabel("Distance")
    # plt.show()

    
    
    kmeansModel = KMeans(n_clusters = numBins, random_state = 0)
    ss = StandardScaler()
    mealFeature = ss.fit_transform(mealFeature)
    kmeansFit = kmeansModel.fit(mealFeature)
    y_pred = kmeansFit.predict(mealFeature)
    plt.scatter(mealFeature[:,0], mealFeature[:,1], c=y_pred, cmap='Paired')
    plt.title("K-means")
    plt.show()
    


    
    #calculate Kmeans:SSE
    kmeansSSE = kmeansFit.inertia_
    #result.append(kmeansFit.inertia_)
    
    #calculate DBSCAN:SSE       ##############################need to automate this more so it doesn't rely on there being 6 clusters
    dbscanSSE = dbscanSSECalc(dbscanFit.labels_, mealFeature)
    #result.append()
    
    #calculate Kmeans:Entropy
    kmeansConfMatrix = confusion_matrix(insulinBinMatrix, kmeansFit.labels_, labels=[0,1,2,3,4,5]).tolist()
    
    # y_true = pd.Series(insulinBinMatrix, name='Actual')
    # y_pred = pd.Series(kmeansFit.labels_, name='Predicted')
    # df_confusion = pd.crosstab(y_true, y_pred)
    #print(df_confusion)
    
    kmeansEntropy = entropyCalc(kmeansConfMatrix)
    
    #result.append(kmeansEntropy)
    
    #calculate DBSCAN:Entropy
    dbscanConfMatrix = confusion_matrix(insulinBinMatrix, dbscanFit.labels_, labels=[0,1,2,3,4,5]).tolist()
    
    # y_true = pd.Series(insulinBinMatrix, name='Actual')
    # y_pred = pd.Series(dbscanFit.labels_, name='Predicted')
    # df_confusion = pd.crosstab(y_true, y_pred)
    #print(df_confusion)
    
    dbscanEntropy = entropyCalc(dbscanConfMatrix)
    
    #result.append(dbscanEntropy)        #this isn't right, need to doublecheck
    
    #calculate Kmeans:Purity
    
    kmeansConfMatrix = metrics.cluster.contingency_matrix(insulinBinMatrix, kmeansFit.labels_)
    #print (kmeansConfMatrix)
    kmeansPurity = numpy.sum(numpy.amax(kmeansConfMatrix, axis=0)) / numpy.sum(kmeansConfMatrix) 
    #print(kmeansPurity)
    
    #result.append(kmeansPurity)
    
    #calculate DBSCAN:Purity
    dbscanConfMatrix = metrics.cluster.contingency_matrix(dbscanFit.labels_, insulinBinMatrix)
    dbscanConfMatrix = dbscanConfMatrix[1:]
    #print (dbscanConfMatrix)
    dbscanPurity = numpy.sum(numpy.amax(dbscanConfMatrix, axis=1)) / numpy.sum(dbscanConfMatrix) 
    #print (dbscanPurity)
    
    #dbscanSSE = 24137
    result = [[kmeansSSE, dbscanSSE, kmeansEntropy, dbscanEntropy, kmeansPurity, dbscanPurity]]
    
    results = pd.DataFrame(result)
    #print(results)

    results.to_csv(r'./Result.csv', index = False, header = False)
    

    # nn = NearestNeighbors(n_neighbors=2)
    # nnOutput = nn.fit(mealFeature)
    # nnDistances, nnIndex = nnOutput.kneighbors(mealFeature)
    
    # nnDistances = numpy.sort(nnDistances, axis=0)
    # nnDistances = nnDistances[:,1]
    # plt.plot(nnDistances)
    # plt.show()
    

main()