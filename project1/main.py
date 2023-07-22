#generate Results.csv with format similar to Lastname_results.csv
import pandas as pd
import math
import datetime


def main():

    CGMDataIn = pd.read_csv('./CGMData.csv', sep = ',', dtype = 'unicode')
    CGMDataIn_descending = CGMDataIn.sort_values('Index', ascending = False)
    
    InsulinDataIn = pd.read_csv('./InsulinData.csv', sep = ',', dtype = 'unicode')
    InsulinDataIn_descending = InsulinDataIn.sort_values('Index', ascending = False)
    
    #the below creates a combined column for DateTime from the Date and Time columns
    CGMDataIn_descending['DateTime'] = pd.to_datetime(CGMDataIn_descending['Date'] + " " 
    + CGMDataIn_descending['Time'], format = '%m/%d/%Y %H:%M:%S')
    
    InsulinDataIn_descending['DateTime'] = pd.to_datetime(InsulinDataIn_descending['Date'] + " " 
    + InsulinDataIn_descending['Time'], format = '%m/%d/%Y %H:%M:%S')
    
    #gets the timestamp from the InsulinData where auto mode starts
    autoModeMarkInsulin = InsulinDataIn_descending[InsulinDataIn_descending.Alarm == 'AUTO MODE ACTIVE PLGM OFF']
    autoModeMarkInsulin = autoModeMarkInsulin.iloc[0]['DateTime']
    #print(autoModeMarkInsulin)
    
    #gets the timestamp from the CGM data where auto mode starts
    CGMData_manualmode_TS_list = CGMDataIn_descending[CGMDataIn_descending.DateTime >= autoModeMarkInsulin]
    autoModeStart = CGMData_manualmode_TS_list.iloc[0]['DateTime']
    #print(CGMData_manualmode_TS['DateTime'])
    #print(autoModeStart)
    
    #everything < autoModeStart is Manual Mode; everything >= autoModeStart is Auto Mode
    #Manual Mode section
    CGMData_manualMode = CGMDataIn_descending[CGMDataIn_descending.DateTime < autoModeStart]
    CGMData_manualMode = CGMData_manualMode[['DateTime', 'Sensor Glucose (mg/dL)', 'Date', 'Time']]
    #print(CGMData_manualMode)
    
    #list of unique days in the Manual Mode section
    manualModeDays = CGMData_manualMode.Date.unique()
    #print(manualModeDays)
    
    #list of entries with NaN in Manual Mode section
    manualModeNaN = CGMData_manualMode[CGMData_manualMode['Sensor Glucose (mg/dL)'].isnull()]
    #print(manualModeNaN)   #total of 362 NaN values (need to drop these)
    
    #list of unique days with NaN in the Manual Mode section
    manualModeNaNDays = manualModeNaN.Date.unique()
    #print(manualModeNaNDays)
    
    #list of entries without a day that has a NaN in the Manual Mode section
    CGMData_manualMode_cleaned = CGMData_manualMode[~CGMData_manualMode.Date.isin(manualModeNaNDays)]
    #print(CGMData_manualMode_cleaned)
    
    manualModeCleanDays = CGMData_manualMode_cleaned.Date.unique()
    #print(manualModeCleanDays)
    
    dayCount = 0
    metric1SumFull = 0
    metric2SumFull = 0
    metric3SumFull = 0
    metric4SumFull = 0
    metric5SumFull = 0
    metric6SumFull = 0
    
    metric1SumDay = 0
    metric2SumDay = 0
    metric3SumDay = 0
    metric4SumDay = 0
    metric5SumDay = 0
    metric6SumDay = 0
    
    metric1SumNight = 0
    metric2SumNight = 0
    metric3SumNight = 0
    metric4SumNight = 0
    metric5SumNight = 0
    metric6SumNight = 0
    
    #metrics calculation
    for day in manualModeCleanDays:
        #full day metrics
        metric180Full = (len(CGMData_manualMode_cleaned[CGMData_manualMode_cleaned.Date.isin([day]) & 
        (CGMData_manualMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) > 180)].index) / 288) * 100
        
        metric250Full = (len(CGMData_manualMode_cleaned[CGMData_manualMode_cleaned.Date.isin([day]) & 
        (CGMData_manualMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) > 250)].index) / 288) * 100
        
        metric70and180Full = (len(CGMData_manualMode_cleaned[CGMData_manualMode_cleaned.Date.isin([day]) & 
        (CGMData_manualMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) >= 70) & 
        (CGMData_manualMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) <= 180)].index) / 288) * 100
        
        metric70and150Full = (len(CGMData_manualMode_cleaned[CGMData_manualMode_cleaned.Date.isin([day]) & 
        (CGMData_manualMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) >= 70) & 
        (CGMData_manualMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) <= 150)].index) / 288) * 100
        
        metric70Full = (len(CGMData_manualMode_cleaned[CGMData_manualMode_cleaned.Date.isin([day]) & 
        (CGMData_manualMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) < 70)].index) / 288) * 100
        
        metric54Full = (len(CGMData_manualMode_cleaned[CGMData_manualMode_cleaned.Date.isin([day]) & 
        (CGMData_manualMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) < 54)].index) / 288) * 100
        
        metric1SumFull += metric180Full
        metric2SumFull += metric250Full
        metric3SumFull += metric70and180Full
        metric4SumFull += metric70and150Full
        metric5SumFull += metric70Full
        metric6SumFull += metric54Full
        
        daytemp = datetime.datetime.strptime(str(day), '%m/%d/%Y').strftime('%Y-%m-%d')
        
        #daytime metrics
        metric180Day = (len(CGMData_manualMode_cleaned[CGMData_manualMode_cleaned.Date.isin([day]) & 
        (CGMData_manualMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) > 180) & 
        (CGMData_manualMode_cleaned['DateTime'] >= (str(daytemp) + ' 06:00:00')) & 
        (CGMData_manualMode_cleaned['DateTime'] <= (str(daytemp) + ' 23:59:59'))].index) / 288) * 100
        
        metric250Day = (len(CGMData_manualMode_cleaned[CGMData_manualMode_cleaned.Date.isin([day]) & 
        (CGMData_manualMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) > 250) & 
        (CGMData_manualMode_cleaned['DateTime'] >= (str(daytemp) + ' 06:00:00')) & 
        (CGMData_manualMode_cleaned['DateTime'] <= (str(daytemp) + ' 23:59:59'))].index) / 288) * 100
        
        metric70and180Day = (len(CGMData_manualMode_cleaned[CGMData_manualMode_cleaned.Date.isin([day]) & 
        (CGMData_manualMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) >= 70) & 
        (CGMData_manualMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) <= 180) & 
        (CGMData_manualMode_cleaned['DateTime'] >= (str(daytemp) + ' 06:00:00')) & 
        (CGMData_manualMode_cleaned['DateTime'] <= (str(daytemp) + ' 23:59:59'))].index) / 288) * 100
        
        metric70and150Day = (len(CGMData_manualMode_cleaned[CGMData_manualMode_cleaned.Date.isin([day]) & 
        (CGMData_manualMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) >= 70) & 
        (CGMData_manualMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) <= 150) & 
        (CGMData_manualMode_cleaned['DateTime'] >= (str(daytemp) + ' 06:00:00')) & 
        (CGMData_manualMode_cleaned['DateTime'] <= (str(daytemp) + ' 23:59:59'))].index) / 288) * 100
        
        metric70Day = (len(CGMData_manualMode_cleaned[CGMData_manualMode_cleaned.Date.isin([day]) & 
        (CGMData_manualMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) < 70) & 
        (CGMData_manualMode_cleaned['DateTime'] >= (str(daytemp) + ' 06:00:00')) & 
        (CGMData_manualMode_cleaned['DateTime'] <= (str(daytemp) + ' 23:59:59'))].index) / 288) * 100
        
        metric54Day = (len(CGMData_manualMode_cleaned[CGMData_manualMode_cleaned.Date.isin([day]) & 
        (CGMData_manualMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) < 54) & 
        (CGMData_manualMode_cleaned['DateTime'] >= (str(daytemp) + ' 06:00:00')) & 
        (CGMData_manualMode_cleaned['DateTime'] <= (str(daytemp) + ' 23:59:59'))].index) / 288) * 100
        
        metric1SumDay += metric180Day
        metric2SumDay += metric250Day
        metric3SumDay += metric70and180Day
        metric4SumDay += metric70and150Day
        metric5SumDay += metric70Day
        metric6SumDay += metric54Day
        
        #night metrics
        metric180Night = (len(CGMData_manualMode_cleaned[CGMData_manualMode_cleaned.Date.isin([day]) & 
        (CGMData_manualMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) > 180) & 
        (CGMData_manualMode_cleaned['DateTime'] >= (str(daytemp) + ' 00:00:00')) & 
        (CGMData_manualMode_cleaned['DateTime'] <= (str(daytemp) + ' 06:00:00'))].index) / 288) * 100
        
        metric250Night = (len(CGMData_manualMode_cleaned[CGMData_manualMode_cleaned.Date.isin([day]) & 
        (CGMData_manualMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) > 250) & 
        (CGMData_manualMode_cleaned['DateTime'] >= (str(daytemp) + ' 00:00:00')) & 
        (CGMData_manualMode_cleaned['DateTime'] <= (str(daytemp) + ' 06:00:00'))].index) / 288) * 100
        
        metric70and180Night = (len(CGMData_manualMode_cleaned[CGMData_manualMode_cleaned.Date.isin([day]) & 
        (CGMData_manualMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) >= 70) & 
        (CGMData_manualMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) <= 180) & 
        (CGMData_manualMode_cleaned['DateTime'] >= (str(daytemp) + ' 00:00:00')) & 
        (CGMData_manualMode_cleaned['DateTime'] <= (str(daytemp) + ' 06:00:00'))].index) / 288) * 100
        
        metric70and150Night = (len(CGMData_manualMode_cleaned[CGMData_manualMode_cleaned.Date.isin([day]) & 
        (CGMData_manualMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) >= 70) & 
        (CGMData_manualMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) <= 150) & 
        (CGMData_manualMode_cleaned['DateTime'] >= (str(daytemp) + ' 00:00:00')) & 
        (CGMData_manualMode_cleaned['DateTime'] <= (str(daytemp) + ' 06:00:00'))].index) / 288) * 100
        
        metric70Night = (len(CGMData_manualMode_cleaned[CGMData_manualMode_cleaned.Date.isin([day]) & 
        (CGMData_manualMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) < 70) & 
        (CGMData_manualMode_cleaned['DateTime'] >= (str(daytemp) + ' 00:00:00')) & 
        (CGMData_manualMode_cleaned['DateTime'] <= (str(daytemp) + ' 06:00:00'))].index) / 288) * 100
        
        metric54Night = (len(CGMData_manualMode_cleaned[CGMData_manualMode_cleaned.Date.isin([day]) & 
        (CGMData_manualMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) < 54) & 
        (CGMData_manualMode_cleaned['DateTime'] >= (str(daytemp) + ' 00:00:00')) & 
        (CGMData_manualMode_cleaned['DateTime'] <= (str(daytemp) + ' 06:00:00'))].index) / 288) * 100
        
        metric1SumNight += metric180Night
        metric2SumNight += metric250Night
        metric3SumNight += metric70and180Night
        metric4SumNight += metric70and150Night
        metric5SumNight += metric70Night
        metric6SumNight += metric54Night
        
        dayCount += 1
        
    #averaging each of the metrics
    manualMetric1AvgFull = metric1SumFull / dayCount
    manualMetric2AvgFull = metric2SumFull / dayCount
    manualMetric3AvgFull = metric3SumFull / dayCount
    manualMetric4AvgFull = metric4SumFull / dayCount
    manualMetric5AvgFull = metric5SumFull / dayCount
    manualMetric6AvgFull = metric6SumFull / dayCount
    
    manualMetric1AvgDay = metric1SumDay / dayCount
    manualMetric2AvgDay = metric2SumDay / dayCount
    manualMetric3AvgDay = metric3SumDay / dayCount
    manualMetric4AvgDay = metric4SumDay / dayCount
    manualMetric5AvgDay = metric5SumDay / dayCount
    manualMetric6AvgDay = metric6SumDay / dayCount
    
    manualMetric1AvgNight = metric1SumNight / dayCount
    manualMetric2AvgNight = metric2SumNight / dayCount
    manualMetric3AvgNight = metric3SumNight / dayCount
    manualMetric4AvgNight = metric4SumNight / dayCount
    manualMetric5AvgNight = metric5SumNight / dayCount
    manualMetric6AvgNight = metric6SumNight / dayCount

    #Auto Mode section
    CGMData_autoMode = CGMDataIn_descending[CGMDataIn_descending.DateTime >= autoModeStart]
    CGMData_autoMode = CGMData_autoMode[['DateTime', 'Sensor Glucose (mg/dL)', 'Date', 'Time']]
    #print(CGMData_autoMode)
    
    #list of unique days in the Auto Mode section
    autoModeDays = CGMData_autoMode.Date.unique()
    #print(autoModeDays)
    
    #list of entries with NaN in Auto Mode section
    autoModeNaN = CGMData_autoMode[CGMData_autoMode['Sensor Glucose (mg/dL)'].isnull()]
    #print(autoModeNaN)   #total of 362 NaN values (need to drop these)
    
    #list of unique days with NaN in the Auto Mode section
    autoModeNaNDays = autoModeNaN.Date.unique()
    #print(autoModeNaNDays)
    
    #list of entries without a day that has a NaN in the Auto Mode section
    CGMData_autoMode_cleaned = CGMData_autoMode[~CGMData_autoMode.Date.isin(autoModeNaNDays)]
    #print(CGMData_autoMode_cleaned)
    
    autoModeCleanDays = CGMData_autoMode_cleaned.Date.unique()
    
    dayCount = 0
    metric1SumFull = 0
    metric2SumFull = 0
    metric3SumFull = 0
    metric4SumFull = 0
    metric5SumFull = 0
    metric6SumFull = 0
    
    metric1SumDay = 0
    metric2SumDay = 0
    metric3SumDay = 0
    metric4SumDay = 0
    metric5SumDay = 0
    metric6SumDay = 0
    
    metric1SumNight = 0
    metric2SumNight = 0
    metric3SumNight = 0
    metric4SumNight = 0
    metric5SumNight = 0
    metric6SumNight = 0
    
    #metrics calculation
    for day in autoModeCleanDays:
        #full day metrics
        metric180Full = (len(CGMData_autoMode_cleaned[CGMData_autoMode_cleaned.Date.isin([day]) & 
        (CGMData_autoMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) > 180)].index) / 288) * 100
        
        metric250Full = (len(CGMData_autoMode_cleaned[CGMData_autoMode_cleaned.Date.isin([day]) & 
        (CGMData_autoMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) > 250)].index) / 288) * 100
        
        metric70and180Full = (len(CGMData_autoMode_cleaned[CGMData_autoMode_cleaned.Date.isin([day]) & 
        (CGMData_autoMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) >= 70) & 
        (CGMData_autoMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) <= 180)].index) / 288) * 100
        
        metric70and150Full = (len(CGMData_autoMode_cleaned[CGMData_autoMode_cleaned.Date.isin([day]) & 
        (CGMData_autoMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) >= 70) & 
        (CGMData_autoMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) <= 150)].index) / 288) * 100
        
        metric70Full = (len(CGMData_autoMode_cleaned[CGMData_autoMode_cleaned.Date.isin([day]) & 
        (CGMData_autoMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) < 70)].index) / 288) * 100
        
        metric54Full = (len(CGMData_autoMode_cleaned[CGMData_autoMode_cleaned.Date.isin([day]) & 
        (CGMData_autoMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) < 54)].index) / 288) * 100
        
        metric1SumFull += metric180Full
        metric2SumFull += metric250Full
        metric3SumFull += metric70and180Full
        metric4SumFull += metric70and150Full
        metric5SumFull += metric70Full
        metric6SumFull += metric54Full
        
        daytemp = datetime.datetime.strptime(str(day), '%m/%d/%Y').strftime('%Y-%m-%d')
        
        #daytime metrics
        metric180Day = (len(CGMData_autoMode_cleaned[CGMData_autoMode_cleaned.Date.isin([day]) & 
        (CGMData_autoMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) > 180) & 
        (CGMData_autoMode_cleaned['DateTime'] >= (str(daytemp) + ' 06:00:00')) & 
        (CGMData_autoMode_cleaned['DateTime'] <= (str(daytemp) + ' 23:59:59'))].index) / 288) * 100
        
        metric250Day = (len(CGMData_autoMode_cleaned[CGMData_autoMode_cleaned.Date.isin([day]) & 
        (CGMData_autoMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) > 250) & 
        (CGMData_autoMode_cleaned['DateTime'] >= (str(daytemp) + ' 06:00:00')) & 
        (CGMData_autoMode_cleaned['DateTime'] <= (str(daytemp) + ' 23:59:59'))].index) / 288) * 100
        
        metric70and180Day = (len(CGMData_autoMode_cleaned[CGMData_autoMode_cleaned.Date.isin([day]) & 
        (CGMData_autoMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) >= 70) & 
        (CGMData_autoMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) <= 180) & 
        (CGMData_autoMode_cleaned['DateTime'] >= (str(daytemp) + ' 06:00:00')) & 
        (CGMData_autoMode_cleaned['DateTime'] <= (str(daytemp) + ' 23:59:59'))].index) / 288) * 100
        
        metric70and150Day = (len(CGMData_autoMode_cleaned[CGMData_autoMode_cleaned.Date.isin([day]) & 
        (CGMData_autoMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) >= 70) & 
        (CGMData_autoMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) <= 150) & 
        (CGMData_autoMode_cleaned['DateTime'] >= (str(daytemp) + ' 06:00:00')) & 
        (CGMData_autoMode_cleaned['DateTime'] <= (str(daytemp) + ' 23:59:59'))].index) / 288) * 100
        
        metric70Day = (len(CGMData_autoMode_cleaned[CGMData_autoMode_cleaned.Date.isin([day]) & 
        (CGMData_autoMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) < 70) & 
        (CGMData_autoMode_cleaned['DateTime'] >= (str(daytemp) + ' 06:00:00')) & 
        (CGMData_autoMode_cleaned['DateTime'] <= (str(daytemp) + ' 23:59:59'))].index) / 288) * 100
        
        metric54Day = (len(CGMData_autoMode_cleaned[CGMData_autoMode_cleaned.Date.isin([day]) & 
        (CGMData_autoMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) < 54) & 
        (CGMData_autoMode_cleaned['DateTime'] >= (str(daytemp) + ' 06:00:00')) & 
        (CGMData_autoMode_cleaned['DateTime'] <= (str(daytemp) + ' 23:59:59'))].index) / 288) * 100
        
        metric1SumDay += metric180Day
        metric2SumDay += metric250Day
        metric3SumDay += metric70and180Day
        metric4SumDay += metric70and150Day
        metric5SumDay += metric70Day
        metric6SumDay += metric54Day
        
        #night metrics
        metric180Night = (len(CGMData_autoMode_cleaned[CGMData_autoMode_cleaned.Date.isin([day]) & 
        (CGMData_autoMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) > 180) & 
        (CGMData_autoMode_cleaned['DateTime'] >= (str(daytemp) + ' 00:00:00')) & 
        (CGMData_autoMode_cleaned['DateTime'] <= (str(daytemp) + ' 06:00:00'))].index) / 288) * 100
        
        metric250Night = (len(CGMData_autoMode_cleaned[CGMData_autoMode_cleaned.Date.isin([day]) & 
        (CGMData_autoMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) > 250) & 
        (CGMData_autoMode_cleaned['DateTime'] >= (str(daytemp) + ' 00:00:00')) & 
        (CGMData_autoMode_cleaned['DateTime'] <= (str(daytemp) + ' 06:00:00'))].index) / 288) * 100
        
        metric70and180Night = (len(CGMData_autoMode_cleaned[CGMData_autoMode_cleaned.Date.isin([day]) & 
        (CGMData_autoMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) >= 70) & 
        (CGMData_autoMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) <= 180) & 
        (CGMData_autoMode_cleaned['DateTime'] >= (str(daytemp) + ' 00:00:00')) & 
        (CGMData_autoMode_cleaned['DateTime'] <= (str(daytemp) + ' 06:00:00'))].index) / 288) * 100
        
        metric70and150Night = (len(CGMData_autoMode_cleaned[CGMData_autoMode_cleaned.Date.isin([day]) & 
        (CGMData_autoMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) >= 70) & 
        (CGMData_autoMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) <= 150) & 
        (CGMData_autoMode_cleaned['DateTime'] >= (str(daytemp) + ' 00:00:00')) & 
        (CGMData_autoMode_cleaned['DateTime'] <= (str(daytemp) + ' 06:00:00'))].index) / 288) * 100
        
        metric70Night = (len(CGMData_autoMode_cleaned[CGMData_autoMode_cleaned.Date.isin([day]) & 
        (CGMData_autoMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) < 70) & 
        (CGMData_autoMode_cleaned['DateTime'] >= (str(daytemp) + ' 00:00:00')) & 
        (CGMData_autoMode_cleaned['DateTime'] <= (str(daytemp) + ' 06:00:00'))].index) / 288) * 100
        
        metric54Night = (len(CGMData_autoMode_cleaned[CGMData_autoMode_cleaned.Date.isin([day]) & 
        (CGMData_autoMode_cleaned['Sensor Glucose (mg/dL)'].astype(int) < 54) & 
        (CGMData_autoMode_cleaned['DateTime'] >= (str(daytemp) + ' 00:00:00')) & 
        (CGMData_autoMode_cleaned['DateTime'] <= (str(daytemp) + ' 06:00:00'))].index) / 288) * 100
        
        metric1SumNight += metric180Night
        metric2SumNight += metric250Night
        metric3SumNight += metric70and180Night
        metric4SumNight += metric70and150Night
        metric5SumNight += metric70Night
        metric6SumNight += metric54Night
        
        dayCount += 1
    
    #averaging each of the metrics
    autoMetric1AvgFull = metric1SumFull / dayCount
    autoMetric2AvgFull = metric2SumFull / dayCount
    autoMetric3AvgFull = metric3SumFull / dayCount
    autoMetric4AvgFull = metric4SumFull / dayCount
    autoMetric5AvgFull = metric5SumFull / dayCount
    autoMetric6AvgFull = metric6SumFull / dayCount
    
    autoMetric1AvgDay = metric1SumDay / dayCount
    autoMetric2AvgDay = metric2SumDay / dayCount
    autoMetric3AvgDay = metric3SumDay / dayCount
    autoMetric4AvgDay = metric4SumDay / dayCount
    autoMetric5AvgDay = metric5SumDay / dayCount
    autoMetric6AvgDay = metric6SumDay / dayCount
    
    autoMetric1AvgNight = metric1SumNight / dayCount
    autoMetric2AvgNight = metric2SumNight / dayCount
    autoMetric3AvgNight = metric3SumNight / dayCount
    autoMetric4AvgNight = metric4SumNight / dayCount
    autoMetric5AvgNight = metric5SumNight / dayCount
    autoMetric6AvgNight = metric6SumNight / dayCount
    
    #collect data into format for writing to file
    resultsData = [[manualMetric1AvgNight, manualMetric2AvgNight, manualMetric3AvgNight, manualMetric4AvgNight, manualMetric5AvgNight, manualMetric6AvgNight, 
    manualMetric1AvgDay, manualMetric2AvgDay, manualMetric3AvgDay, manualMetric4AvgDay, manualMetric5AvgDay, manualMetric6AvgDay, 
    manualMetric1AvgFull, manualMetric2AvgFull, manualMetric3AvgFull, manualMetric4AvgFull, manualMetric5AvgFull, manualMetric6AvgFull], 
    [autoMetric1AvgNight, autoMetric2AvgNight, autoMetric3AvgNight, autoMetric4AvgNight, autoMetric5AvgNight, autoMetric6AvgNight, 
    autoMetric1AvgDay, autoMetric2AvgDay, autoMetric3AvgDay, autoMetric4AvgDay, autoMetric5AvgDay, autoMetric6AvgDay, 
    autoMetric1AvgFull, autoMetric2AvgFull, autoMetric3AvgFull, autoMetric4AvgFull, autoMetric5AvgFull, autoMetric6AvgFull]] 
    
    #create dataframe from results
    resultsDF = pd.DataFrame(resultsData) 
    
    #write results to file without index or header
    resultsDF.to_csv(r'./Results.csv', index = False, header = False)    

main()