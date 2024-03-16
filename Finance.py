# Import required libraries
# Import the TensorFlow library
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from collections import Counter
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import keras_tuner as kt
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


# importing the data files and assigning labels using pandas, pandas.read_csv the csv file and applies the label names to each column  
rawIdata = pd.read_csv('train_data.csv') # training dataset
rawTdata = pd.read_csv('test_data_hidden.csv' ) # testing dataset

# printing out the input datasets properties like count, mean, standard deviation, etc. to compare each features values range  
des = rawIdata.describe().transpose() # using pandas.describe to find the input datas different properties and .transpose chnages the orientation 
print("Input features Properties:\n",des,"\n")

print("First five data points of combined file of input and output dataset:\n ",rawIdata.head())#printing the 1st five data points of new variable having input and output dataset 
#using pandas.head 

#using Seaborn.pairplot to plot the each input features and ouput variable vs each other input feature and output variable  
plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(rawIdata.corr()[['Class']].sort_values(by='Class', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Features Correlating with Class', fontdict={'fontsize':18}, pad=16)
plt.show()


# checking for null values 
print('No. of Null values in dataset =', rawIdata.isnull().sum().max())
# Checking for imbalances in target by checking total class count
NFpercent =  round(rawIdata['Class'].value_counts()[0]/len(rawIdata) * 100,2)
Fpercent =  round(rawIdata['Class'].value_counts()[1]/len(rawIdata) * 100,2)

# Plotting the target variable to check for imbalances
sns.countplot(x='Class', data=rawIdata)
plt.title('Class Count \n (Class 0 = Non-Fraud is %f percent of the dataset || Class 1 = Fraud is %f percent of the dataset' % (NFpercent, Fpercent) ,fontsize=10)
plt.xticks(range(2), ["Non-Fraud", "Fraud"])
plt.show()


# separating features and target columns for training
X =pd.read_csv('train_data.csv', usecols=list(c for c in rawIdata.columns if c != 'Class'))
Y = rawIdata['Class']
print('Shape of the input dataset for features :'  , X.shape, 'and target:',Y.shape)
print('The fist five data points of input dataset for features :\n',X.head())
print('The fist five data points of input dataset for target :\n',Y.head())

# Applying Oversampling and then Undersampling on training dataset to balance the target variable using SMOTEENN
# SMOTEENN is a hybrid method that applies SMOTE to oversample and then applies Edited Nearest Neighbours to clean the dataset
# Edited Nearest Neighbours is a method that removes any noisy majoirity class samples from the dataset
from imblearn.combine import SMOTEENN
nm3 = SMOTEENN(sampling_strategy=0.1) # using SMOTEENN to create an object nm3
x_samples,y_samples= nm3.fit_resample(X, Y) 
print(sorted(Counter(y_samples).items()))
print('Shape of the new oversmapled and undersampled dataset for features :'  , x_samples.shape, 'and target:',y_samples.shape)
print('Original dataset shape {}'.format(Counter(Y)))
print('Resampled dataset shape {}'.format(Counter(y_samples)))

# Splitting the test dataset into features and target columns
Xtest =pd.read_csv('test_data_hidden.csv', usecols= list(c for c in rawTdata.columns if c != 'Class'))
Ytest = rawTdata['Class']

# Since the mean and std deviantion for the features is drastically different we standardise the feature dataset
standardize = preprocessing.RobustScaler() # using Sklearn standarsizing function to create an object stadardize
x_samples = standardize.fit_transform(x_samples)  # Fit methods calculates the stadardization of input data set, then transform methods applies it to it.
print("The new scaled training datsset is:",x_samples)
Xtest = standardize.fit_transform(Xtest)
print("The new scaled input test datsset is:",Xtest)

# Creating 3 models for the dataset below
# Create an instance of SVM classifier
svm = SVC()
# Create an instance of Logistic Regression classifier
lr = LogisticRegression()
# Create an instance of Naive Bayes classifier
nb = GaussianNB()


# Fit the SVM and logistic regression to the scaled X data and y samples
svm.fit(x_samples, y_samples)
lr.fit(x_samples, y_samples)
nb.fit(x_samples, y_samples)

# Predict the target variable for the test data
y_pred1 = svm.predict(Xtest)
y_pred2 = lr.predict(Xtest)
y_pred3 = nb.predict(Xtest)

# Calculate the accuracy score and  classification report for the models that shows f1 score, precision, recall and support for SVM, Logistic Regression and Naive Bayes
print('The accuracy score for SVM is:',accuracy_score(Ytest,y_pred1)*100)
print('The classification report for SVM is:\n',classification_report(Ytest,y_pred1))

print('The accuracy score for Logistic Regression is:',accuracy_score(Ytest,y_pred2)*100)
print('The classification report for Logistic Regression is:\n',classification_report(Ytest,y_pred2))

print('The accuracy score for Naive Bayes is:',accuracy_score(Ytest,y_pred3)*100)
print('The classification report for Naive Bayes is:\n',classification_report(Ytest,y_pred3))

# Create a confusion matrix for SVM, Logistic Regression and Naive Bayes
cm1 = confusion_matrix(Ytest, y_pred1)
cm2 = confusion_matrix(Ytest, y_pred2)
cm3 = confusion_matrix(Ytest, y_pred3)

# Plot the confusion matrix for SVM, Logistic Regression and Naive Bayes
i=0
plt.figure(figsize=(10, 10))
for i in range(3):
    if i == 0:
        cm = cm1
        y_pred = y_pred1
        title = 'SVM '
    elif i == 1:
        cm = cm2
        y_pred = y_pred2
        title = 'Logistic Regression '
    elif i == 2:
        cm = cm3
        y_pred = y_pred3
        title = 'Naive Bayes '
    plt.subplot(2,2,i+1)
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(accuracy_score(Ytest, y_pred)*100)
    plt.title(title+all_sample_title, size = 15)
    i += 1
plt.show()


# Since the mean and std deviantion for the features is drastically different we standardise the feature dataset
standardize = preprocessing.RobustScaler() # using Sklearn standarsizing function to create an object stadardize
X = standardize.fit_transform(X)  # Fit methods calculates the stadardization of input data set, then transform methods applies it to it.
print("The new scaled input datsset is:",X)

# Create an instance of Random Forest classifier ans XGBoost
RF = RandomForestClassifier(class_weight='balanced', verbose=2, n_estimators=100, n_jobs=-1) # Class_weight is used to balance the target variable while sampling during bagging 
XGB = XGBClassifier(n_estimators=100, n_jobs=-1, learning_rate=0.01, objective='binary:logistic', verbosity=2)

# fit model for Random Forest and XGBoost
RF.fit(X, Y)
XGB.fit(X, Y)

# Predict the target variable for the test data
y_pred4 = RF.predict(Xtest)
y_pred5 = XGB.predict(Xtest)

# Calculate the accuracy score and  classification report for the models that shows f1 score, precision, recall and support for Random Forest and XGBoost
print('The accuracy score for Random Forest is:', accuracy_score(Ytest,y_pred4)*100)
print('The classification report for Random Forest is:\n',classification_report(Ytest,y_pred4))
print('The accuracy score for XGBoost is:', accuracy_score(Ytest,y_pred5)*100)
print('The classification report for XGBoost is:\n',classification_report(Ytest,y_pred5))

# Create a confusion matrix for Random Forest and XGBoost
cm4 = confusion_matrix(Ytest, y_pred4)
cm5 = confusion_matrix(Ytest, y_pred5)

# Plot the confusion matrix for Random Forest and XGBoost
i=0
plt.figure(figsize=(8, 10))
for i in range(2):
    if i == 0:
        cm = cm4
        y_pred = y_pred4
        title = 'Random Forest '
    elif i == 1:
        cm = cm5
        y_pred = y_pred5
        title = 'XGBoost '
    plt.subplot(2,1,i+1)
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(accuracy_score(Ytest, y_pred)*100)
    plt.title(title+all_sample_title, size = 15)
    i += 1
plt.show()

# Defining a custome ANN model for hyperparameter tuning. The model will have 1 input layer, 1-3 hidden layer and 1 output layer
# The model will be compiled with different hyperparameters like activation function, optimizer, learning rate, dropout and number of hidden layers
def build_model(hp):
    active_func = hp.Choice('activation', ['relu', 'tanh'])
    optimizer = hp.Choice('optimizer', ['adam', 'SGD', 'RMSprop'])
    lr = hp.Float('learning_rate', min_value=0.0001, max_value=0.1, sampling='log', step=10)
    drop = hp.Float('dropout', min_value=0.1, max_value=0.4, sampling= 'linear',step=0.1)
    inputs = tf.keras.Input(shape=[X.shape[1],])

    # create hidden layers
    dnn_units = hp.Int(f"0_units", min_value=32, max_value=512)
    dense = tf.keras.layers.Dense(units=dnn_units, activation=active_func)(inputs)
    for layer_i in range(hp.Choice("n_layers", [1,2,3,4]) - 1):
        dnn_units = hp.Int(f"{layer_i}_units", min_value=32, max_value=512)
        dense = tf.keras.layers.Dense(units=dnn_units, activation=active_func)(dense)
        dense = tf.keras.layers.Dropout(rate=drop)(dense)
    outputs = tf.keras.layers.Dense(units=1, activation='sigmoid')(dense)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # compile the model choosing the optimizer and learning rate
    if optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == "RMSprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    else:
        raise("Not supported optimizer")
        
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy',tf.keras.metrics.RootMeanSquaredError()])
    return model

# Define the tuner and build the tuner for hyperparameter tuning with either RandomSearch, Hyperband or BayesianOptimization
def build_tuner(model, hpo_method, objective, dir_name):
    if hpo_method == "RandomSearch":
        tuner = kt.RandomSearch(model, objective=objective, max_trials=10, executions_per_trial=2,
                               project_name=hpo_method, directory=dir_name)
    elif hpo_method == "Hyperband":
        tuner = kt.Hyperband(model, objective=objective, max_epochs=200, hyperband_iterations=2, 
                            project_name=hpo_method)
    elif hpo_method == "BayesianOptimization":
        tuner = kt.BayesianOptimization(model, objective=objective, max_trials=10, executions_per_trial=2,
                                       project_name=hpo_method)
    return tuner

# Define the objective and build the tuner based on either RandomSearch, Hyperband or BayesianOptimization
obj = kt.Objective('val_root_mean_squared_error', direction='min')
dir_name = "v1"
randomsearch_tuner = build_tuner(build_model, "Hyperband", obj, dir_name)

# Define the early stopping callback
es = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error',patience=20, restore_best_weights=True, verbose=1, mode='min')

# Search for the best hyperparameters
randomsearch_tuner.search(X, Y, epochs=100, verbose=2, batch_size = 1024, validation_data=(Xtest, Ytest))
print(randomsearch_tuner.results_summary())
print('The best hyperparameters are:\n',randomsearch_tuner.get_best_hyperparameters()[0].values)
best_model = build_model(randomsearch_tuner.get_best_hyperparameters(1)[0])
best_model.summary()


# Training full dataset the ANN model architecture with the best hyperparameters
best_model.fit(X, Y, epochs=100, batch_size = 1024, validation_data=(Xtest, Ytest), callbacks=es)


# Predict the target variable for the test data
y_pred6 = best_model.predict(Xtest).round()
# Calculate the accuracy score and  classification report for the models that shows f1 score, precision, recall and support 
print('The accuracy score for ANN is:',accuracy_score(Ytest,y_pred6)*100)
print('The classification report for ANN is:\n',classification_report(Ytest,y_pred6))
RMSE = tf.keras.metrics.RootMeanSquaredError()
print('The RMSE for ANN is:',RMSE(Ytest,y_pred6).numpy())

# Create a confusion matrix 
cm6 = confusion_matrix(Ytest, y_pred6)

# Plot the confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(cm6, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(accuracy_score(Ytest, y_pred6)*100)
plt.title('ANN '+all_sample_title, size = 15)
plt.show()

# Save the model
best_model.save('ANN.h5')


# Implementing Anomaly Detection
# Visualizing the distribution of the features, for sleceting the features for anomaly detection
features=['V17','V14', 'V11', 'V4', 'V15', 'V13']
plt.figure(figsize=(15,25))
for i, feat in enumerate(features):
    plt.subplot(6, 1, i+1)
    
    # Extract fraud and non-fraud data
    fraud_data = rawIdata[rawIdata['Class']==1]
    non_fraud_data = rawIdata[rawIdata['Class']==0]
    x_w = np.empty(fraud_data[feat].shape)
    x_w.fill(1/fraud_data[feat].shape[0])
    y_w = np.empty(non_fraud_data[feat].shape)
    y_w.fill(1/non_fraud_data[feat].shape[0])
    
    # Plot histograms
    plt.hist([fraud_data[feat], non_fraud_data[feat]], bins=np.linspace(-10, 10, 30),weights=[x_w, y_w], alpha=0.5, color=['r', 'b'])
    plt.legend(['fraudulent', 'non-fraudulent'], loc='best')
    plt.xlabel('')
    plt.title('Distribution of feature: ' + feat)
plt.show()
# Since the features V14 and V17 have a clear distinction between the fraudulent and non-fraudulent transactions, 
#we will use these features for anomaly detection
sns.pairplot(rawIdata[rawIdata.Class==1], vars=['V14', 'V17'], kind='reg', hue='Class')
plt.title('V14 and V17 for fraudulent transactions \n ',fontsize=10)
plt.show()

# Estimate the Gaussian distribution of the dataset
def estimateGaussian(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.cov(dataset.T)
    return mu, sigma
# Multivariate Gaussian distribution
def multivariateGaussian(dataset,mu,sigma):
    p = multivariate_normal(mean=mu, cov=sigma, allow_singular=True)
    return p.pdf(dataset)

# Selecting the features for anomaly detection
fraud = rawIdata[rawIdata['Class']==1]
Nfraud = rawIdata[rawIdata['Class']==0]
fraud = fraud[['V14','V17','Class']]
Nfraud = Nfraud[['V14','V17','Class']]
# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(Nfraud.drop(['Class'],axis=1), Nfraud['Class'] , test_size=0.4, random_state=0)

# Selecting the validation set for anomaly detection
y_fval = fraud['Class']
x_fval = fraud.drop(['Class'],axis=1)

# Concatenating the training and validation sets
train_cv = pd.concat([X_val,x_fval],axis=0)
train_cv_y = pd.concat([y_val,y_fval],axis=0)

# Calculating threshold using cross validation set
mu, sigma = estimateGaussian(X_train)
pdf = multivariateGaussian(X_train,mu,sigma)
pdf_cv = multivariateGaussian(train_cv,mu,sigma)
x_test = rawTdata[['V14','V17']]
pdf_test = multivariateGaussian(x_test,mu,sigma)

# Function to select the threshold for anomaly detection using cross validation set and f1 score for updating the threshold epsilon
def select_threshold(test_data, probs):
    best_epsilon = 0
    best_f1 = 0
    f = 0
    stepsize = (max(probs) - min(probs)) / 1000
    epsilons = np.arange(min(probs), max(probs), stepsize)
    for epsilon in np.nditer(epsilons):
        predictions = (probs < epsilon)
        f = f1_score(test_data, predictions, average='binary')
        if f > best_f1:
            best_f1 = f
            best_epsilon = epsilon

    return best_f1, best_epsilon

# Selecting the best threshold
F1score, ep = select_threshold(train_cv_y, pdf_cv)
print('Best Epsilon %f' %ep)

# Predicting the anomalies
y_test = rawTdata['Class']
predictions = (pdf_test < ep)
F1score = f1_score(y_test, predictions, average = "binary")    
print ('Best F1 Score %f' %F1score)
# The anomlay count are as follows
def findIndices(binVec):
    l = []
    for i in range(len(binVec)):
        if binVec[i] == 1:
            l.append(i)
    return l

# Counting the number of outliers 
listOfOutliers = findIndices(predictions)
count_outliers = len(listOfOutliers)
print('\n\nNumber of outliers:', count_outliers)
print('\n',listOfOutliers)

#Lets Visualize our predictions in below scatter plot 
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(x_test['V14'],x_test['V17'],marker="o", color="lightBlue")
ax.set_title('Anomalies(in red) vs Predicted Anomalies(in Green)')
for i, txt in enumerate(x_test['V14'].index):
       if y_test.loc[txt] == 1 :
            ax.annotate('*', (x_test['V14'].loc[txt],x_test['V17'].loc[txt]),fontsize=13,color='Red')
       if predictions[i] == True :
            ax.annotate('o', (x_test['V14'].loc[txt],x_test['V17'].loc[txt]),fontsize=15,color='Green')
plt.show()
