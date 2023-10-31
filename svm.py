#-------------------------------------------------------------------------
# AUTHOR: Neil Patrick Reyes
# FILENAME: svm.py
# SPECIFICATION: Simulate a grid search using a combination of 4 various hyperparameter in order to
#                   determine which combination outputs the most accurate result
# FOR: CS 4210- Assignment #3
# TIME SPENT: 1.5 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

#defining the hyperparameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the training data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:,-1] #getting the last field to create the class training data and convert them to NumPy array

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the training data by using Pandas library

X_test = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df.values)[:,-1] #getting the last field to create the class testing data and convert them to NumPy array

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
highest_accuracy = 0.0

for c_val in c:
    for degree_val in degree:
        for kernel_val in kernel:
           for shape_val in decision_function_shape:

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape.
                #For instance svm.SVC(c=1, degree=1, kernel="linear", decision_function_shape = "ovo")
                clf = svm.SVC(C = c_val, degree = degree_val, kernel = kernel_val, decision_function_shape = shape_val)

                #Fit SVM to the training data
                clf.fit(X_training, y_training)

                #make the SVM prediction for each test sample and start computing its accuracy
                #hint: to iterate over two collections simultaneously, use zip()
                #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
                #to make a prediction do: clf.predict([x_testSample])
                num_correct = 0
                num_samples = len(y_test)
                for X_test_val, y_test_val in zip(X_test, y_test):
                    predict = clf.predict([X_test_val])
                    if(predict[0] == y_test_val):
                        num_correct = num_correct + 1

                #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
                #with the SVM hyperparameters. Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                accuracy = num_correct / num_samples

                if(accuracy > highest_accuracy):
                    highest_accuracy = accuracy
                    print(f"Current highest accuracy: {accuracy} at hyperparameter: c = {c_val}, degree = {degree_val}, kernel = {kernel_val}, and decision function shape = {shape_val}")