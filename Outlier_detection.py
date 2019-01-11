# Importing the libraries
import numpy as np
import statsmodels.api as smapi
import statsmodels.graphics as smgraphics
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Labelling_Outlier_Detection.csv')
X = dataset.iloc[:, 2].values
y = dataset.iloc[:, 7].values
dataset['Outliers']=0

#Regression Fit
regression = smapi.OLS(X, y).fit()

#Saving the model
regression.save("outlier_detector_model.dat")

#regression = smapi.load("outlier_detector_model.dat")

#Plot the graph
figure = smgraphics.regressionplots.plot_fit(regression, 0)

# Find outliers 
test = regression.outlier_test()

#Inserting the data back in dataset
myarray = np.array(list(((i,1) for i,t in enumerate(test) if t[2] < .1)))
for index, row in dataset.iterrows():
    col_num=index
    for j in myarray:
        if index==j[0]:
            dataset.at[col_num, 'Outliers']=1
            print("Outlier in row", j[0])



import json
import requests
index_name = 'labelling_task_data_dev'
index_type = 'data'
headers = {'Content-type': 'application/json'}
for index, row in dataset.iterrows():
    print(row["Label Task Id"])
    if row["Outliers"]==1:
        document = {
            "taskId": row["taskId"],
            "taskName": row["taskName"],
            "annotation": row["Annotation"],
            "imageId": row["imageId"],
            "labelTaskId":row["Label Task Id"] ,
            "duration": row["Duration"],
            "qaDuration": row["QA Duration"],
            "totalDuration": row["Total Duration"],
            "primaryExecutor": row["Primary Executor "],
            "Outliers":row["Outliers"]
            }
        data_json_string = json.dumps(document)
        url = 'https://vpc-user-metrics-5sp4wfnnztrxbhhl3ddfhwwa3y.us-east-1.es.amazonaws.com/' + index_name + '/' + index_type + '/' + document['taskId']
        #Put data
        response = requests.put(url=url, data=data_json_string, headers=headers)
        print(response.json())
#Get data
response = requests.get(url=url, headers=headers)
print(response.json())