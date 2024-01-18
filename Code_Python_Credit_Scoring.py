#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import librairies 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix




# In[2]:


# Reading the CSV file into a DataFrame
data=pd.read_csv("\\Users\\2thebest/Downloads/fid2/loan big data.csv")
data


# In[3]:


# Calculating the count of missing values in each column of the DataFrame
missing_values_count=data.isnull().sum()
missing_values_count


# In[4]:


# List of columns to impute missing values
columns_to_impute = ['Gender', 'Married', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term', 'Dependents']

# Imputing missing values in specified columns with the mode value
for column in columns_to_impute:
    # Calculating the mode value for the current column
    mode_value = data[column].mode()[0]
    
    # Filling missing values in the column with the mode value
    data[column].fillna(mode_value, inplace=True)


# In[5]:


# Calculating the mean value of the 'LoanAmount' column
mean_loanamount = data['LoanAmount'].mean()

# Filling missing values in the 'LoanAmount' column with the mean value
data['LoanAmount'] = data['LoanAmount'].fillna(mean_loanamount)


# In[6]:


# Calculating the count of missing values in each column of the DataFrame
missing_values_count = data.isnull().sum()

# Displaying the count of missing values for each column
missing_values_count


# In[7]:


# Selecting the 'LoanAmount' column from the DataFrame
data1 = data[['LoanAmount']]

# Calculating the Z-scores for each value in the 'LoanAmount' column
z_scores = np.abs((data1 - data1.mean(axis=0)) / np.std(data1))

# Displaying the calculated Z-scores
z_scores


# In[8]:


# Setting the threshold for Z-scores to identify outliers
threshold = 2.0

# Finding LoanAmount outliers based on Z-scores exceeding the threshold
LoanAmount_outliers = [data1.loc[i, 'LoanAmount'] for i in range(len(data1)) if z_scores.loc[i, 'LoanAmount'] > threshold]

# Displaying the value of the first LoanAmount outlier (if any)
print('The value of the LoanAmount outlier is ' + str(LoanAmount_outliers[0]))


# In[9]:


# Setting the figure size for the plot
plt.figure(figsize=(8, 4))

# Plotting the original 'LoanAmount' data points
plt.plot(data1['LoanAmount'], marker='o', linestyle='', label='Data Points')

# Identifying and marking LoanAmount outliers with red 'x' markers
plt.scatter(
    [i for i in range(len(data1)) if z_scores.loc[i, 'LoanAmount'] > threshold],
    LoanAmount_outliers,
    color='red', label='LoanAmount_outliers', marker='x', s=100
)

# Adding labels and title to the plot
plt.xlabel('Data Point Index')
plt.ylabel('LoanAmount')
plt.title('Identifying LoanAmount Outliers Using Z-scores')

# Displaying legend for better interpretation
plt.legend()

# Displaying the plot
plt.show()


# In[10]:


# Setting the figure size for the histogram plot
plt.figure(figsize=(8, 6))

# Creating a histogram of 'LoanAmount' with 20 bins, colored in green with black edges
plt.hist(data['LoanAmount'], bins=20, color='green', edgecolor='black')

# Adding labels to the axes
plt.xlabel('Loan_Amount_Term')
plt.ylabel('LoanAmount')

# Adding a title to the histogram plot
plt.title('Loan Distribution')

# Displaying the histogram plot
plt.show()


# In[11]:


# Selecting the 'ApplicantIncome' column from the DataFrame
data2 = data[['ApplicantIncome']]

# Calculating the Z-scores for each value in the 'ApplicantIncome' column
z_scores = np.abs((data2 - data2.mean(axis=0)) / np.std(data2))

# Displaying the calculated Z-scores for 'ApplicantIncome'
z_scores


# In[12]:


# Setting the threshold for Z-scores to identify outliers
threshold = 2.0

# Finding ApplicantIncome outliers based on Z-scores exceeding the threshold
ApplicantIncome_outliers = [data2.loc[i, 'ApplicantIncome'] for i in range(len(data2)) if z_scores.loc[i, 'ApplicantIncome'] > threshold]

# Displaying the value of the first ApplicantIncome outlier (if any)
print('The value of the ApplicantIncome outlier is ' + str(ApplicantIncome_outliers[0]))


# In[13]:


# Setting the figure size for the plot
plt.figure(figsize=(8, 4))

# Plotting the original 'ApplicantIncome' data points
plt.plot(data2['ApplicantIncome'], marker='v', linestyle='', label='Data Points')

# Identifying and marking ApplicantIncome outliers with red 'x' markers
plt.scatter(
    [i for i in range(len(data2)) if z_scores.loc[i, 'ApplicantIncome'] > threshold],
    ApplicantIncome_outliers,
    color='red', label='ApplicantIncome_outliers', marker='x', s=100
)

# Adding labels and title to the plot
plt.xlabel('Data Point Index')
plt.ylabel('ApplicantIncome')
plt.title('Identifying ApplicantIncome Outliers Using Z-scores')

# Displaying legend for better interpretation
plt.legend()

# Displaying the plot
plt.show()


# In[14]:


# Assuming 'data' is your DataFrame
print(data)


# In[15]:


# Generating summary statistics for the DataFrame 'data'
summary = data.describe()

# Printing the summary statistics
print(summary)


# In[16]:


# Mapping 'Yes' to 1 and 'No' to 0 in the 'Self_Employed' column of the DataFrame 'data'
data['Self_Employed'] = data['Self_Employed'].map({'Yes': 1, 'No': 0})


# In[17]:


# Mapping 'Yes' to 1 and 'No' to 0 in the 'Married' column of the DataFrame 'data'
data['Married'] = data['Married'].map({'Yes': 1, 'No': 0})


# In[18]:


# Mapping 'Yes' to 1 and 'No' to 0 in the 'Gender' column of the DataFrame 'data'
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})


# In[19]:


# Mapping 'Yes' to 1 and 'No' to 0 in the 'Graduate' column of the DataFrame 'data'
data['Education'] = data['Education'].map({'Graduate': 1, 'Not Graduate': 0})


# In[20]:


# Mapping 'Y' to 1 and 'N' to 0 in the 'Loan_Status' column of the DataFrame 'data'
data['Loan_Status'] = data['Loan_Status'].map({'Y': 1, 'N': 0})


# In[21]:


# Assuming 'data' is your DataFrame
print(data)


# In[22]:


# Dropping the 'Loan_ID' column from the DataFrame 'data'
data = data.drop('Loan_ID', axis=1)


# In[23]:


# Displaying the updated DataFrame 'data' after dropping the 'Loan_ID' column
data


# In[24]:


# Dropping duplicate rows from the DataFrame 'data'
data = data.drop_duplicates()

# Displaying the DataFrame 'data' after removing duplicates
print(data)


# In[25]:


# Creating a new DataFrame 'data3' by selecting specific columns from 'data'
data3 = data[['Gender', 'Married', 'Education', 'Self_Employed', 'Credit_History', 'Loan_Status', 'LoanAmount', 'Loan_Amount_Term', 'ApplicantIncome']]



# In[26]:


# Calculating the correlation matrix for the selected columns in the DataFrame 'data3'
correlation_matrix_known_columns = data3.corr()

# Displaying the correlation matrix
print(correlation_matrix_known_columns)


# In[27]:


# Creating a custom color palette for the heatmap
custom_colors = sns.diverging_palette(240, 10, as_cmap=True)

# Setting the figure size for the heatmap
plt.figure(figsize=(8, 6))

# Creating a heatmap for the correlation matrix with annotations and custom colors
sns.heatmap(correlation_matrix_known_columns, annot=True, cmap=custom_colors, fmt='.2f')

# Adding a title to the heatmap
plt.title('Correlation Heatmap')

# Displaying the heatmap
plt.show()


# In[28]:


# Extracting the correlation of each feature with the target variable 'Loan_Status'
correlation_with_target = correlation_matrix_known_columns['Loan_Status']

# Displaying the correlation of each feature with 'Loan_Status'
print(correlation_with_target)


# In[29]:


# Sort the correlation values in descending order
sorted_correlation = correlation_with_target.abs().sort_values(ascending=False)

# Select the features with the highest absolute correlation values
selected_features = sorted_correlation[1:] # Exclude the target variable
selected_features= pd.DataFrame(selected_features)
print("Selected features:")
selected_features


# In[30]:


# Separate the features and the target variable
X = data3.drop('Loan_Status', axis=1)
y = data3['Loan_Status']

# Create a Random Forest classifier
rf = RandomForestClassifier()

# Fit the classifier to the data
rf.fit(X, y)

# Get feature importances
feature_importances = rf.feature_importances_

# Sort the feature importances in descending order
sorted_importances = sorted(zip(feature_importances, X.columns), reverse=True)

# Print the feature importances
print("Feature importances:")
selected_features_importance=list()
for importance, feature in sorted_importances:
    print(feature, ":", importance)
    selected_features_importance.append(feature)


# In[31]:


# Separate the features and the target variable
X = data3.drop('Loan_Status', axis=1)
y = data3['Loan_Status']

# Perform chi-squared feature selection
selector = SelectKBest(score_func=chi2, k=5)  # Select top 5 features
X_selected = selector.fit_transform(X, y)

# Get the selected feature names
selected_feature_names = X.columns[selector.get_support()]

print("Selected features:")
print(selected_feature_names)


# In[32]:


# Separate the features and the target variable
X = data3.drop('Loan_Status', axis=1)
y = data3['Loan_Status']

# Perform mutual information feature selection
selector = SelectKBest(score_func=mutual_info_classif, k=5)  # Select top 5 features
X_selected = selector.fit_transform(X, y)

# Get the selected feature names
selected_feature_names_MI = X.columns[selector.get_support()]

print("Selected features:")
print(selected_feature_names_MI)


# In[33]:


# Select the top k features based on correlation analysis
k = 5  # Number of top features to select
selected_features_corr=selected_features.index.tolist()
selected_features_corr = selected_features_corr[:k]

# Select the top k features based on feature importance
k = 5  # Number of top features to select
selected_features_importance = selected_features_importance[:k]

# Select the top k features based on univariate feature selection
k = 5  # Number of top features to select
selected_features_univariate = selected_feature_names[:k]

# Combine the selected features from different methods
selected_features = []
selected_features.extend(selected_features_corr)
selected_features.extend(selected_features_importance)
selected_features.extend(selected_features_univariate)

# Print the selected features
print("Selected features:")
print(selected_features)


# In[34]:


# Counting the occurrences of elements in the 'selected_features' iterable
element_counts = Counter(selected_features)

# Displaying the count for each unique element
for element, count in element_counts.items():
    print(f"{element}: {count}")


# In[35]:


# Converting 'selected_features' to a set to get unique elements, and then converting it back to a list
selected_features = list(set(selected_features))

# Displaying the updated 'selected_features' list with unique elements
print(selected_features)


# In[36]:


# Selecting specific features from the DataFrame 'data' to create the feature matrix 'X'
X = data[selected_features]
# Splitting the data into training and testing sets
# X_train: training feature matrix, X_test: testing feature matrix
# y_train: training target variable, y_test: testing target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[37]:


# Scaling the data using StandardScaler()
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[38]:


# Create a Random Forest classifier
rf = RandomForestClassifier()

# Train the model on the training data
rf.fit(X_train, y_train)


# In[39]:


# Initializing the Logistic Regression model
LR = LogisticRegression()

# Fitting the model with training data
LR.fit(X_train, y_train)


# In[40]:


# Create an SVM classifier
svm = SVC()

# Train the SVM model on the training data
svm.fit(X_train, y_train)


# In[41]:


# Perform cross-validation and calculate performance metrics
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
scores = cross_validate(rf, X_train, y_train, cv=5, scoring=scoring)

# Print the mean and standard deviation of each metric
print("Accuracy:", scores['test_accuracy'].mean())
print("Precision:", scores['test_precision'].mean())
print("Recall:", scores['test_recall'].mean())
print("F1-score:", scores['test_f1'].mean())
print("AUC-ROC:", scores['test_roc_auc'].mean())


# In[42]:


# Perform cross-validation and calculate performance metrics
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
scores = cross_validate(LR, X_train, y_train, cv=5, scoring=scoring)

# Print the mean and standard deviation of each metric
print("Accuracy:", scores['test_accuracy'].mean())
print("Precision:", scores['test_precision'].mean())
print("Recall:", scores['test_recall'].mean())
print("F1-score:", scores['test_f1'].mean())
print("AUC-ROC:", scores['test_roc_auc'].mean())


# In[43]:


# Perform cross-validation and calculate performance metrics
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
scores = cross_validate(svm, X, y, cv=5, scoring=scoring)

# Print the mean and standard deviation of each metric
print("Accuracy:", scores['test_accuracy'].mean())
print("Precision:", scores['test_precision'].mean())
print("Recall:", scores['test_recall'].mean())
print("F1-score:", scores['test_f1'].mean())
print("AUC-ROC:", scores['test_roc_auc'].mean())


# In[44]:


# Make predictions on the test set
y_pred = svm.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[45]:


# Make predictions on the test set
y_pred = LR.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[46]:


# Make predictions on the test set
y_pred = rf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Confusion Matrix:\n", conf_matrix)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[47]:


# Make predictions on the new data using the trained SVM classifier , the best one . 
predicted_loan_status = LR.predict(--new_data--)

# Display the predicted 'Loan_Status'
print("Predicted Loan Status for new data:")
print(predicted_loan_status)


# In[ ]:





# In[ ]:




