import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

# Read the CSV file
df = pd.read_csv('creditcard.csv')


# Checking distribution of data
df.groupby(['Class']).size() 

# Only use the 'Amount' and 'V1', ..., 'V28' features
X = df.iloc[:, 1:30].values
y = df.iloc[:, 30].values

# normalize the data attributes
X = preprocessing.normalize(X)

# Define the model
model = LogisticRegression()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


    
# Fit and predict!
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# And finally: 
tn,fp,fn,tp=confusion_matrix(y_test, y_pred).ravel()
print("No fraud: "+ str(tn)+"Not fraud but reported as frauds: "+str(fp)+"Fraud but not reported as frauds: "+str(fn)+"Frauds: "+str(tp))

