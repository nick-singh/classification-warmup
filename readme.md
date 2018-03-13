# Classification Tasks 


1. Import the necessary data manipulation and visualization libraries
```
  import pandas as pd
  from matplotlib import pyplot as plt
  import seaborn as sns
  import numpy as np
```
2. Import the machine learning algorithms from sklearn
```
  from sklearn.linear_model import LogisticRegression
  from sklearn.svm import SVC, LinearSVC
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.naive_bayes import GaussianNB
  from sklearn.linear_model import Perceptron
  from sklearn.linear_model import SGDClassifier
  from sklearn.tree import DecisionTreeClassifier
```
3. Import some utility and metric classes
```
  from sklearn.metrics import confusion_matrix
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import accuracy_score
```
4. Load the data (in this case we have separate datasets for training and testing)
5. Analyze the data to get a feel of the data
6. Write a block of code or function (for both training and testing data) to:
	1. Get all the columns from the dataset
	2. Determine if the values in a column is numeric
	3. Clean all numeric columns 
	4. Get a list of all categorical columns with NAN values
7. Get a count of all the NAN values in categorical columns
8. Write a function or a block of code (for both training and testing data) to:
	1. Use the categorical columns to get the most most occurring 
	2. Create a dictionary with the column and associated most occurring category
9. Using the dict and the fillna function fill all NAN categorical values in the dataset.
10. Ensure that there are no more NAN values in the categorical columns
11. Get all the numeric columns that contain any NAN values
12. Choose an appropriate method to fill the NAN values in the numeric columns
13. Ensure that there are no more NAN values in the numeric columns
14. Get a list of all categorical columns 
15. Encode all the categories here is a good [blog post](http://pbpython.com/categorical-encoding.html)
16. Split the training data into X_train, Y_train
17. Split the test data into X_test, Y_test
18. Using the machine learning algorithms imported in 2 train and test each classifier and keep track of their accuracies. 

