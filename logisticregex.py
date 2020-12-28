#import required libs

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt


#example data for applicants admitted by a university for PhD - 48 data points
applicants = {'percent_marks': [78,75,69,71,68,73,65,69,72,74,69,61,69,71,68,77,61,58,65,54,59,62,60,55,55,57,67,66,58,65,66,64,62,66,66,68,65,67,58,59,69,82,65,74,54,75,79,63],
              'research_papers': [3,4,3,5,4,6,1,4,5,2,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5,4,3,5,1,4,1,4],
              'admitted': [1,1,0,1,0,1,0,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1,0,0,1,0,0,0,1]
              }

#create dataframe with available data
df = pd.DataFrame(applicants,columns= ['percent_marks', 'research_papers','admitted'])
print ('Original dataset\n',df)


#set the variables X and y for the model

X = df[['percent_marks', 'research_papers']]
y = df['admitted']


#Split the dataset into training set and test set of 25%

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)


#apply the logistic regression

logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,y_train)

y_pred=logistic_regression.predict(X_test)

#check the prediction for test data
print('\nTest data',X_test)
print('\nPrediction based on test data',y_pred)


#get the confusion matrix and plot it

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print('\nConfusion matrix:')
sn.heatmap(confusion_matrix, annot=True)
plt.show()


print('\nAccuracy: ',metrics.accuracy_score(y_test, y_pred))


#use the model for prediction

new_candidates = {'percent_marks': [67,78,51,79,65,81],
                  'research_papers': [3,4,2,5,1,2]
                 }

df2 = pd.DataFrame(new_candidates,columns= ['percent_marks', 'research_papers'])
print('\nInput dataset:\n',df2)

y_pred=logistic_regression.predict(df2)

print ('\nPrediction of the algorithm:',y_pred)
