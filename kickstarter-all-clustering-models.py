################################################################################
############################# Supervised Learning ##############################
################################################################################
# Import libraries
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)
custom_colors = ["#ebdc78", "#63bff0", "#1984c5", "#54bebe", "#df979e", "#d7658b", "#ffd3b6", "#ee4035"]

# Import dataset
kickstarter_df = pd.read_excel('kickstarter.xlsx')
# Display the shape of the dataset
kickstarter_df.shape # (15474, 45)
# Display the names of all columns
kickstarter_df.columns
# Check for data types
kickstarter_df.info()
# Display the first 5 rows of the dataset
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(kickstarter_df.head(5))
# Descriptive statistics for numerical features    
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print (kickstarter_df.describe())

# ==============================================================================    
# Data preprocessing and exploration
# ==============================================================================
# Plot the distribution of country
country_counts = kickstarter_df['country'].value_counts()
# Sort the country_counts in descending order
country_counts = country_counts.sort_values(ascending=True)
# Plot the distribution of country with custom colors
plt.figure(figsize=(10, 8))
ax = country_counts.plot(kind='barh', color=custom_colors[1])
# Add count labels next to each bar
for index, value in enumerate(country_counts):
    ax.text(value, index, str(value), ha='left', va='center', fontsize=10, color='black')
plt.title('Distribution of Countries in Kickstarter Projects')
plt.xlabel('Count')
plt.ylabel('Country')
# Save the figure
plt.savefig('country_distribution.png', bbox_inches='tight', dpi=300)
plt.show()

# 11000 US, 1924 GB, 830 CA, 532 AU, 240 NL, 181 DE, 161 FR, 102 IT, 89 DK, 86 NZ and so on.

# ==============================================================================
# Plot a chart displaying country and currency
country_currency_counts = kickstarter_df[['country', 'currency']].value_counts().sort_values()
# Plot the figure
plt.figure(figsize=(10, 8))
ax = country_currency_counts.plot(kind='barh', color=custom_colors[1])
# Add count labels next to each bar
for index, value in enumerate(country_currency_counts):
    ax.text(value, index, f'{value} ({country_currency_counts.index[index][0]} - {country_currency_counts.index[index][1]})', ha='left', va='center', fontsize=10)
# Add title and axis names
plt.title('Distribution of Countries and Currencies in Kickstarter Projects')
plt.xlabel('Count')
plt.ylabel('Country and Currency')
# Save the figure
plt.savefig('country-currency_distribution.png', bbox_inches='tight', dpi=300)
plt.show()

# We can remove the 'currency' column, since it is the same as 'country' column
kickstarter_df = kickstarter_df.drop(columns=['currency'], axis=1) # 15474 rows x 44 columns
# Replace non-US countries in country column with 'non-US'
kickstarter_df.loc[kickstarter_df['country'] != 'US', 'country'] = 'Non-US' # 15474 rows x 44 columns
# Display unique values for the country column
kickstarter_df['country'].unique()

# ==============================================================================
# Relationship between name_len and name_len_clean, and blurb_len and blurb_len_clean
plt.figure(figsize=(12, 6))
# Scatter plot for name_len vs name_len_clean
plt.subplot(1, 2, 1)
plt.scatter(kickstarter_df['name_len'], kickstarter_df['name_len_clean'], alpha=0.5, c=custom_colors[1])
plt.title('Relationship between name_len and name_len_clean')
plt.xlabel('name_len')
plt.ylabel('name_len_clean')
# Scatter plot for blurb_len vs blurb_len_clean
plt.subplot(1, 2, 2)
plt.scatter(kickstarter_df['blurb_len'], kickstarter_df['blurb_len_clean'], alpha=0.5, c=custom_colors[1])
plt.title('Relationship between blurb_len and blurb_len_clean')
plt.xlabel('blurb_len')
plt.ylabel('blurb_len_clean')
# Print the layout
plt.tight_layout()
# Save the figure
plt.savefig('relationship-name.png', bbox_inches='tight', dpi=300)
plt.show()

# Drop name_len and blurb_len, since we already have clean versions!
kickstarter_df = kickstarter_df.drop(columns=['name_len', 'blurb_len'], axis=1) # 15474 rows x 42 columns
# Replace null values in name_len_clean with 0
kickstarter_df['name_len_clean'] = kickstarter_df['name_len_clean'].fillna(0)
# Replace null values in blurb_len_clean with 0
kickstarter_df['blurb_len_clean'] = kickstarter_df['blurb_len_clean'].fillna(0)

# ==============================================================================
# Unique values for pledged along with their counts
kickstarter_df['disable_communication'].value_counts()
# Note: Almost all projects are not disabled for communication, i.e. only 167 / 15474 = 1.08% projects are disabled.

# Check if projects with disable_communication=True have a state of 'suspended' or 'canceled'
kickstarter_df[kickstarter_df['disable_communication'] == True]['state'].value_counts() 
# All 167 have state='suspended'

# ==============================================================================
# Unique values for staff_pick along with their counts
kickstarter_df['staff_pick'].value_counts()
# Note: Only 1782 / 15474 = 11.5% of projects are staff picked.

# ==============================================================================
# Correlation between pledged and usd_pledged
kickstarter_df['pledged'].corr(kickstarter_df['usd_pledged']) # 0.989, very high
# Drop pledged, since we already have usd_pledged
kickstarter_df = kickstarter_df.drop(columns=['pledged'], axis=1) # 15474 rows x 41 columns

# ==============================================================================
# Create new column 'goal_usd' by multiplying 'goal' and 'static_usd_rate'
kickstarter_df['goal_usd'] = kickstarter_df['goal'] * kickstarter_df['static_usd_rate'] # 15474 rows x 42 columns
# Remove columns 'goal' and 'static_usd_rate'
kickstarter_df = kickstarter_df.drop(columns=['goal', 'static_usd_rate'], axis=1) # 15474 rows x 40 columns

# ==============================================================================
# Remove irrelevant columns
irrelevant_columns = ['id', 'name', 'deadline_hr', 'created_at_hr', 'launched_at_hr']
kickstarter_df = kickstarter_df.drop(columns=irrelevant_columns, axis=1) # 15474 rows x 35 columns

# Check for duplicates
kickstarter_df[kickstarter_df.duplicated()].shape # (0, 35)

# Check for missing values
kickstarter_df.isnull().sum() # 1392 missing values in 'category' column

# If category is missing, then replace with "No category"
kickstarter_df['category'] = kickstarter_df['category'].fillna('No category') # 15474 rows x 35 columns

# ==============================================================================
# Outlier detection
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
# Plotting the countplot
ax = sns.countplot(x='state', data=kickstarter_df, color=custom_colors[1])
plt.title('Countplot for State')
# Annotating each bar with its count (without decimal places)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
# Save the figure
plt.savefig('state.png', bbox_inches='tight', dpi=300)
plt.show()
# For failed: 8860, For successful: 4575, For canceled: 1872, For suspended: 167

# Drop 'canceled' and 'suspended' from state column
kickstarter_df = kickstarter_df[kickstarter_df['state'] != 'canceled'] # 13602 rows x 35 columns
kickstarter_df = kickstarter_df[kickstarter_df['state'] != 'suspended'] # 13435 rows x 35 columns

# Change 'successful' to 1 and 'failed' to 0
kickstarter_df['state'] = kickstarter_df['state'].replace(['successful', 'failed'], [1, 0])

# Remove original date columns
date_columns = ['deadline', 'created_at', 'launched_at']
kickstarter_df = kickstarter_df.drop(columns=date_columns, axis=1) # 13435 rows x 32 columns

# Remove weekday columns
weekday_columns = ['deadline_weekday', 'created_at_weekday', 'launched_at_weekday']
kickstarter_df = kickstarter_df.drop(columns=weekday_columns, axis=1) # 13435 rows x 29 columns

# ==============================================================================
# The classification task is assumed to be done at the time each project is launched. In other
# words, we execute the model to predict whether a new project is going to be successful or not, at the moment
# when the project owner submits the project. Therefore, the model should only use the predictors that are
# available at the moment when a new project is launched.

# Columns that are not available at the time of launching a project
columns_not_available = ['disable_communication', 'state_changed_at', 'staff_pick', 
                         'backers_count', 'usd_pledged', 'spotlight', 'state_changed_at_weekday', 
                         'state_changed_at_month', 'state_changed_at_day', 'state_changed_at_yr', 
                         'state_changed_at_hr', 'launch_to_state_change_days']
# Remove columns that are not available at the time of launching a project
kickstarter_df = kickstarter_df.drop(columns=columns_not_available, axis=1) # 13435 rows x 17 columns

# X predictors
X = kickstarter_df.loc[:,kickstarter_df.columns!='state'] # 13435 rows x 16 columns
# Target variable
y = kickstarter_df['state'] # 13435 rows x 1 column

# Dummify categorical variables
dummify_cols = ['category', 'country']
X = pd.get_dummies(X, columns=dummify_cols) # 13435 rows x 39 columns

# Check for new X dataset
X.columns
X.info()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(X.head(5))

# Correlation
c = X.corr()
# Display the correlation matrix
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(c)
# Plotting the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(c, cmap="rocket_r", annot=True, fmt=".0f")  # fmt=".0f" for no decimal places
plt.title('Correlation Matrix')
# Save the figure
plt.savefig('corr.png', bbox_inches='tight', dpi=300)
plt.show()

# List of highly correlated features
corr_features = []
# Set a threshold for correlation
threshold = 0.8
# Iterate through the correlation matrix and identify highly correlated features
for i in range(len(c.columns)):
    for j in range(i):
        if abs(c.iloc[i, j]) >= threshold:
            colname = c.columns[i]
            corr_features.append(colname)
# Display the highly correlated features
print("Highly Correlated Features:")
print(corr_features)
correlated_features = ['created_at_yr', 'launched_at_yr', 'country_Non-US']

# Revise X dataset
X = X.drop(columns=correlated_features, axis=1) # 13435 rows x 36 columns

# ==============================================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

# Validation set approach: Split the standardized dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 5)
# I tried different test_sizes but I found 0.33 to be the best

# Create standardized training and test sets
standardizer = StandardScaler()
X_train_std = standardizer.fit_transform(X_train)
X_test_std = standardizer.transform(X_test)

# ==============================================================================
# Model 1: Logistic regression
# ==============================================================================
# Run the model
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model_lr = lr.fit(X_train_std,y_train)
# View results
model_lr.intercept_
model_lr.coef_
# Using the model to predict the results based on the test dataset
y_test_pred_lr = model_lr.predict(X_test_std)
# Using the model to predict the probability of being classified to each category
y_test_pred_prob = model_lr.predict_proba(X_test_std)[:,1]
# Performance measures
accuracy_lr = accuracy_score(y_test, y_test_pred_lr)
precision_lr = precision_score(y_test, y_test_pred_lr)
recall_lr = recall_score(y_test, y_test_pred_lr)
f1_lr = f1_score(y_test, y_test_pred_lr)
auc_lr = roc_auc_score(y_test, y_test_pred_lr)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_test_pred_prob)
conf_matrix_lr = confusion_matrix(y_test, y_test_pred_lr)
# Print the results
print(f"Accuracy of Logistic Regression Model is: {accuracy_lr*100:.2f}%")  # 72.06%
print(f"Precision of Logistic Regression Model is: {precision_lr*100:.2f}%")  # 64.05%
print(f"Recall of Logistic Regression Model is: {recall_lr*100:.2f}%")  # 44.08%
print(f"F1 Score of Logistic Regression Model is: {f1_lr*100:.2f}%")  # 52.22%
print(f"AUC of Logistic Regression Model is: {auc_lr*100:.2f}%")  # 65.48%
print("Confusion Matrix of Logistic Regression Model is:")
print(conf_matrix_lr)

# ==============================================================================
# Model 2: K-Nearest Neighbors
# ==============================================================================
from sklearn.neighbors import KNeighborsClassifier
# The general rule of thumb to pick a starting value of k is the square root of the number of observations in the dataset
import math
k = int(math.sqrt(len(X_train_std))) # 94
# Build a model with k = 3 and using euclidean distance function
knn = KNeighborsClassifier(n_neighbors=k,p=2) # For p, 1: Manhattan, 2: Euclidean
model_knn = knn.fit(X_train_std,y_train)
# Using the model to predict the results based on the test dataset
y_test_pred_knn = model_knn.predict(X_test_std)
# Performance measures
accuracy_knn = accuracy_score(y_test, y_test_pred_knn)
precision_knn = precision_score(y_test, y_test_pred_knn)
recall_knn = recall_score(y_test, y_test_pred_knn)
f1_knn = f1_score(y_test, y_test_pred_knn)
auc_knn = roc_auc_score(y_test, y_test_pred_knn)
conf_matrix_knn = confusion_matrix(y_test, y_test_pred_knn)
# Print the results
print(f"Accuracy of K-Nearest Neighbors Model is: {accuracy_knn*100:.2f}%") # 70.64%
print(f"Precision of K-Nearest Neighbors Model is: {precision_knn*100:.2f}%") # 63.27%
print(f"Recall of K-Nearest Neighbors Model is: {recall_knn*100:.2f}%") # 36.33%
print(f"F1 Score of K-Nearest Neighbors Model is: {f1_knn*100:.2f}%") # 46.15%
print(f"AUC of K-Nearest Neighbors Model is: {auc_knn*100:.2f}%") # 62.57%
print("Confusion Matrix of K-Nearest Neighbors Model is:")
print(conf_matrix_knn)
# Choosing k
for i in range (30,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    model = knn.fit(X_train_std,y_train)
    y_test_pred = model.predict(X_test_std)
    print("Accuracy score using k-NN with ",i," neighbors = "+str(accuracy_score(y_test, y_test_pred)))
   
# ==============================================================================
# Model 3: Classification tree
# ==============================================================================
from sklearn.tree import DecisionTreeClassifier
# Build a tree model with 3 layers
ct = DecisionTreeClassifier(max_depth=3) # 3 layers
model_ct = ct.fit(X_train, y_train)  
# Make prediction and evaluate accuracy
y_test_pred_ct = model_ct.predict(X_test)  
# Performance measures
accuracy_ct = accuracy_score(y_test, y_test_pred_ct)
precision_ct = precision_score(y_test, y_test_pred_ct)
recall_ct = recall_score(y_test, y_test_pred_ct)
f1_ct = f1_score(y_test, y_test_pred_ct)
auc_ct = roc_auc_score(y_test, y_test_pred_ct)
conf_matrix_ct = confusion_matrix(y_test, y_test_pred_ct)
# Print the results
print(f"Accuracy of Classification Tree is: {accuracy_ct*100:.2f}%") # 69.51%
print(f"Precision of Classification Tree is: {precision_ct*100:.2f}%") # 55.59%
print(f"Recall of Classification Tree is: {recall_ct*100:.2f}%") # 59.57%
print(f"F1 Score of Classification Tree is: {f1_ct*100:.2f}%") # 57.51%
print(f"AUC of Classification Tree is: {auc_ct*100:.2f}%") # 67.17%
print("Confusion Matrix of Classification Tree is:")
print(conf_matrix_ct)
# Print the tree
from sklearn import tree
plt.figure(figsize=(20,20))
features = X.columns
classes = ['1','0']
tree.plot_tree(model_ct,feature_names=features,class_names=classes,filled=True)
plt.savefig('tree.png', bbox_inches='tight', dpi=300)
plt.show()

# Pruning pre-model building using cross validation for trees with different depths
from sklearn.model_selection import cross_val_score
for i in range (2,21):                                                 
    model = DecisionTreeClassifier(max_depth=i)
    scores = cross_val_score(estimator=model, X=X, y=y, cv=5)
    print(i,':',np.average(scores))
# Highest accuracy is for max_depth=5 @ 70.12%

# ==============================================================================
# Model 4: Random Forest
# ==============================================================================
from sklearn.ensemble import RandomForestClassifier
# Build the model
randomforest = RandomForestClassifier(random_state=0)
model_rf = randomforest.fit(X_train, y_train)
# Print feature importance
pd.Series(model_rf.feature_importances_,index = X.columns).sort_values(ascending = False).plot(kind = 'bar', figsize = (14,6))
# Make prediction and evaluate accuracy
y_test_pred_rf = model_rf.predict(X_test)
# Performance measures
accuracy_rf = accuracy_score(y_test, y_test_pred_rf)
precision_rf = precision_score(y_test, y_test_pred_rf)
recall_rf = recall_score(y_test, y_test_pred_rf)
f1_rf = f1_score(y_test, y_test_pred_rf)
auc_rf = roc_auc_score(y_test, y_test_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_test_pred_rf)
# Print the results
print(f"Accuracy of Random Forest Model is: {accuracy_rf*100:.2f}%") # 74.24%
print(f"Precision of Random Forest Model is: {precision_rf*100:.2f}%") # 67.40%
print(f"Recall of Random Forest Model is: {recall_rf*100:.2f}%") # 49.67%
print(f"F1 Score of Random Forest Model is: {f1_rf*100:.2f}%") # 57.20%
print(f"AUC of Random Forest Model is: {auc_rf*100:.2f}%") # 68.47%
print("Confusion Matrix of Random Forest Model is:")
print(conf_matrix_rf)
# K-fold cross validation for different numbers of features to consider at each split
for i in range (2,7):                                                                   
    model = RandomForestClassifier(random_state=0,max_features=i,n_estimators=100)
    scores = cross_val_score(estimator=model, X=X, y=y, cv=5)
    print(i,':',np.average(scores))
# Cross-validate internally using OOB observations
randomforest = RandomForestClassifier(random_state=0,oob_score=True)   
model = randomforest.fit(X, y)
model.oob_score_ # 73.22%

# ==============================================================================
# Model 5: Gradient Boosting Algorithm
# ==============================================================================
from sklearn.ensemble import GradientBoostingClassifier
# Build the model
gbt = GradientBoostingClassifier(random_state=0)                           
model_gbt = gbt.fit(X_train, y_train)
# Make prediction and evaluate accuracy
y_test_pred_gbt = model_gbt.predict(X_test)
# Performance measures
accuracy_gbt = accuracy_score(y_test, y_test_pred_gbt)
precision_gbt = precision_score(y_test, y_test_pred_gbt)
recall_gbt = recall_score(y_test, y_test_pred_gbt)
f1_gbt = f1_score(y_test, y_test_pred_gbt)
auc_gbt = roc_auc_score(y_test, y_test_pred_gbt)
conf_matrix_gbt = confusion_matrix(y_test, y_test_pred_gbt)
# Print the results
print(f"Accuracy of Gradient Boosting Model is: {accuracy_gbt*100:.2f}%") # 75.30%
print(f"Precision of Gradient Boosting Model is: {precision_gbt*100:.2f}%") # 68.45%
print(f"Recall of Gradient Boosting Model is: {recall_gbt*100:.2f}%") # 53.26%
print(f"F1 Score of Gradient Boosting Model is: {f1_gbt*100:.2f}%") # 59.90%
print(f"AUC of Gradient Boosting Model is: {auc_gbt*100:.2f}%") # 70.12%
print("Confusion Matrix of Gradient Boosting Model is:")
print(conf_matrix_gbt)
# K-fold cross-validation with different number of samples required to split
for i in range (2,10):                                                                        
    model2 = GradientBoostingClassifier(random_state=0,min_samples_split=i,n_estimators=100)
    scores = cross_val_score(estimator=model2, X=X, y=y, cv=5)
    print(i,':',np.average(scores))
    
# ==============================================================================
# Model 6: Artificial Neural Network
# ==============================================================================
from sklearn.neural_network import MLPClassifier
# Build a model
mlp = MLPClassifier(hidden_layer_sizes=(11),max_iter=1000, random_state=0)
model_ann = mlp.fit(X_train_std,y_train)
# Make prediction and evaluate the performance
y_test_pred_ann = model_ann.predict(X_test_std)
# Performance measures
accuracy_ann = accuracy_score(y_test, y_test_pred_ann)
precision_ann = precision_score(y_test, y_test_pred_ann)
recall_ann = recall_score(y_test, y_test_pred_ann)
f1_ann = f1_score(y_test, y_test_pred_ann)
auc_ann = roc_auc_score(y_test, y_test_pred_ann)
conf_matrix_ann = confusion_matrix(y_test, y_test_pred_ann)
# Print the results
print(f"Accuracy of Artificial Neural Network Model is: {accuracy_ann*100:.2f}%") # 71.06%
print(f"Precision of Artificial Neural Network Model is: {precision_ann*100:.2f}%") # 61.59%
print(f"Recall of Artificial Neural Network Model is: {recall_ann*100:.2f}%") # 43.75%
print(f"F1 Score of Artificial Neural Network Model is: {f1_ann*100:.2f}%") # 51.16%
print(f"AUC of Artificial Neural Network Model is: {auc_ann*100:.2f}%") # 64.65%
print("Confusion Matrix of Artificial Neural Network Model is:")
print(conf_matrix_ann)
# Varying the number of hidden layers
mlp2 = MLPClassifier(hidden_layer_sizes=(11,11),max_iter=1000, random_state=0)
model2 = mlp2.fit(X_train_std,y_train)
y_test_pred_2 = model2.predict(X_test_std)
accuracy_score(y_test, y_test_pred_2) # 71.54%
# Cross-validate with different size of the hidden layer
for i in range (2,21):    
    model3 = MLPClassifier(hidden_layer_sizes=(i),max_iter=1000, random_state=0)
    scores = cross_val_score(estimator=model3, X=X, y=y, cv=5)
    print(i,':',np.average(scores))

################################################################################
############################ Unsupervised Learning #############################
################################################################################
# Import libraries
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

kickstarter_df = pd.read_excel("kickstarter.xlsx")
kickstarter_df.info()

# Unique values for state
kickstarter_df['state'].unique()

# Only keep rows with successful and failed projects
kickstarter_df = kickstarter_df[(kickstarter_df['state'] == 'successful') | (kickstarter_df['state'] == 'failed')] # 13435 rows x 45 columns

# Remove name and id columns
kickstarter_df = kickstarter_df.drop(['name', 'id'], axis=1) # 13435 rows x 43 columns

# Drop name_len and blurb_len, since we already have clean versions!
kickstarter_df = kickstarter_df.drop(columns=['name_len', 'blurb_len'], axis=1) # 13435 rows x 41 columns

# Drop pledged, since we already have usd_pledged
kickstarter_df = kickstarter_df.drop(columns=['pledged'], axis=1) # 13435 rows x 40 columns

# Create new column 'goal_usd' by multiplying 'goal' and 'static_usd_rate'
kickstarter_df['goal_usd'] = kickstarter_df['goal'] * kickstarter_df['static_usd_rate'] # 13435 rows x 41 columns

# Remove columns 'goal' and 'static_usd_rate'
kickstarter_df = kickstarter_df.drop(columns=['goal', 'static_usd_rate'], axis=1) # 13435 rows x 39 columns

# Display number of missing values by column
kickstarter_df.isnull().sum() # 1254 missing values in category column

# Replace missing values in category column with "No category"
kickstarter_df['category'] = kickstarter_df['category'].fillna('No category')

# ==============================================================================
# Display unique values for country and currency columns
unique_combinations = (
    kickstarter_df.groupby(['country', 'currency'])
    .size()
    .reset_index(name='count')
    .assign(percentage=lambda x: (x['count'] / len(kickstarter_df)) * 100)
    .sort_values(by='count', ascending=False)
)

# Format the 'percentage' column as XX.XX%
unique_combinations['percentage'] = unique_combinations['percentage'].map('{:.2f}%'.format)
print(unique_combinations)

# ==============================================================================
# Replace non-US countries with 'Non-US' in country column
kickstarter_df['country'] = kickstarter_df['country'].where(kickstarter_df['country'] == 'US', 'Non-US')
print(kickstarter_df['country'].unique())

# Drop currency column
kickstarter_df = kickstarter_df.drop(columns=['currency'], axis=1) # 13435 rows x 38 columns

# Drop irrelevant columns
irrelevant_columns = ['deadline', 'created_at', 'launched_at', 'state_changed_at', 'deadline_hr', 'created_at_hr', 'launched_at_hr', 'state_changed_at_hr']

# Delete irrelevant columns
kickstarter_df = kickstarter_df.drop(columns=irrelevant_columns, axis=1) # 13435 rows x 30 columns

# Select features for clustering (excluding non-numeric and datetime columns)
numeric_features = kickstarter_df.select_dtypes(include=['int64', 'float64']).columns # 20
boolean_features = kickstarter_df.select_dtypes(include=['bool']).columns # 3
non_numeric_features = kickstarter_df.select_dtypes(include=['object']).columns # 7

kickstarter_df.info()

# ==============================================================================
# Anomaly Detection - Isolation Forest Model
# ==============================================================================
# Create a copy of the original DataFrame
kickstarter_df_copy = kickstarter_df.copy()

# Dummify boolean features and non-numeric features
kickstarter_df_copy = pd.get_dummies(kickstarter_df_copy, columns=boolean_features) # 13435 rows x 32 columns
kickstarter_df_copy = pd.get_dummies(kickstarter_df_copy, columns=non_numeric_features) # 13435 rows x 80 columns

# Standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std = scaler.fit_transform(kickstarter_df_copy)

# Isolation Forest for anomaly detection
from sklearn.ensemble import IsolationForest
iforest = IsolationForest(n_estimators=150, contamination=0.1, bootstrap=True, random_state=0)
pred = iforest.fit_predict(X_std)
score = iforest.decision_function(X_std) 
score = score * -1 + 0.5

# Identify anomalies
anomaly_index = np.where(pred == -1)[0]
anomaly_values = kickstarter_df.iloc[anomaly_index]

# Number of anomalies
print('Number of anomalies:', len(anomaly_values)) # 1344

# Remove identified anomalies from kickstarter_df
kickstarter_df = kickstarter_df.drop(index=anomaly_values.index) # 13435 rows reduced to 12091 rows

# Display the anomaly values from dataset
anomaly_values.describe()

# Display the descriptive statistics of the dataset
kickstarter_df.describe()

# ==============================================================================
# Model 1: K-Prototypes Clustering
# ==============================================================================
# The k-prototypes clustering algorithm is a partitioning-based clustering algorithm that we can use to cluster 
# datasets having attributes of numeric and categorical types. This algorithm is an ensemble of the k-means 
# clustering algorithm and the k-modes clustering algorithm.
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes

# Specify the numerical columns you want to standardize
numerical_index = [4, 5, 8, 9, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
numerical_columns = kickstarter_df.columns[numerical_index] # 25 columns

# Create a subset DataFrame with only the numerical columns
numerical_data = kickstarter_df[numerical_columns]

# Create a StandardScaler instance and fit-transform the numerical data
scaler = StandardScaler()
scaled_numerical_data = scaler.fit_transform(numerical_data)

# Replace the original numerical columns with the scaled data
kickstarter_df[numerical_columns] = scaled_numerical_data

kickstarter_array = kickstarter_df.values

# Convert numerical features to float type
kickstarter_array[:, 4] = kickstarter_array[:, 4].astype(float)
kickstarter_array[:, 5] = kickstarter_array[:, 5].astype(float)
kickstarter_array[:, 8] = kickstarter_array[:, 8].astype(float)
kickstarter_array[:, 9] = kickstarter_array[:, 9].astype(float)
kickstarter_array[:, 14:] = kickstarter_array[:, 14:].astype(float)

kickstarter_array

# ==============================================================================
# Find the optimal K using the elbow method
elbow_scores = dict()
range_of_k = range(2, 20)

for k in range_of_k:
    kproto = KPrototypes(n_clusters=k, verbose=2, max_iter=100)
    clusters = kproto.fit_predict(kickstarter_array, categorical=[0, 1, 2, 3, 6, 7, 10, 11, 12, 13])
    elbow_scores[k] = kproto.cost_

# Print the elbow scores
print(elbow_scores)

# Plot the elbow graph with custom colors
plt.plot(elbow_scores.keys(), elbow_scores.values(), color=custom_colors[0], marker='o')
plt.scatter(elbow_scores.keys(), elbow_scores.values(), color=custom_colors[0])
plt.xlabel("Values of K")
plt.ylabel("Cost")
plt.title('Elbow Method For Optimal k')

# Disable scientific notation on the y-axis
plt.ticklabel_format(style='plain', axis='y')

# Save the figure
plt.savefig('k-prototype-elbow.png', bbox_inches='tight', dpi=300)

plt.show()   

# There is a significant decrease in cost from k=2 to k=8. After that, the value of k 
# is almost constant. We can consider k=8 as the elbow point. Hence, we will select 
# 8 as the best k for k-prototypes clustering for the given dataset.

# Choose the optimal number of clusters based on the elbow graph
optimal_k = 8  # This is based on the graph

kproto = KPrototypes(n_clusters=optimal_k, verbose=2, max_iter=100)
clusters = kproto.fit_predict(kickstarter_array, categorical=[0, 1, 2, 3, 6, 7, 10, 11, 12, 13])

# Calculate distance between clusters based on numerical features
# Cost is a combination of the cost associated with numerical features (handled by K-Means) and the cost associated with categorical features (handled by K-Modes)
num_distance = kproto.cost_ # 172069.61

# Get the cluster's centroids
centroids = kproto.cluster_centroids_

# Display centroid for cluster 0. Display also the names of the columns
print(f"Centroid for cluster 0: {centroids[0]}")
print(f"Centroid for cluster 1: {centroids[1]}")
print(f"Centroid for cluster 2: {centroids[2]}")
print(f"Centroid for cluster 3: {centroids[3]}")
print(f"Centroid for cluster 4: {centroids[4]}") # Splotlight True, State Successful High Backers High Pledged
print(f"Centroid for cluster 5: {centroids[5]}")
print(f"Centroid for cluster 6: {centroids[6]}")
print(f"Centroid for cluster 7: {centroids[7]}")

# ==============================================================================
cluster_dict = []

for c in clusters:
    cluster_dict.append(c)
    
cluster_dict

# Create copy of the original DataFrame
kickstarter_df_copy = kickstarter_df.copy()
kickstarter_df_copy['cluster'] = cluster_dict

# Display the count of projects in each cluster
kickstarter_df_copy['cluster'].value_counts()

# Set display options for pandas
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Display statistics for each cluster
cluster_stats = kickstarter_df_copy.groupby('cluster').agg(['median', 'mean'])
print(cluster_stats.T)

# Extract mean statistics
mean_stats = cluster_stats.xs('mean', level=1, axis=1)

# Extract median statistics
median_stats = cluster_stats.xs('median', level=1, axis=1)

# Print mean, median, and mode separately
print("Mean Statistics:")
print(mean_stats.T)

print("\nMedian Statistics:")
print(median_stats.T)
    
# ==============================================================================
# Display statistics for each cluster
# ==============================================================================
# Set display options for pandas
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Display statistics for each cluster
for cluster_id in range(optimal_k):
    print(f"\nCluster {cluster_id} Statistics:")
    
    # Filter DataFrame for the specific cluster
    cluster_data = kickstarter_df_copy[kickstarter_df_copy['cluster'] == cluster_id]
    
    # Display statistics for the cluster
    print(cluster_data.describe().T)

# ==============================================================================
# Calculate the average values for each cluster
avg_values_by_cluster = kickstarter_df_copy.groupby('Cluster_Label').agg({
    'usd_pledged': 'mean',
    'goal_usd': 'mean',  # Adjust this to your actual column name for goal
    'backers_count': 'mean',
    'deadline_yr': 'mean',
    'created_at_yr': 'mean',
    'launched_at_yr': 'mean',
    'create_to_launch_days': 'mean',
    'launch_to_deadline_days': 'mean',
    'launch_to_state_change_days': 'mean',
}).reset_index()

# Rename columns for clarity
avg_values_by_cluster.columns = ['Cluster', 'Average Pledged', 'Average Goal (USD)', 'Average Backers Count']

# Display the DataFrame
print(avg_values_by_cluster)    

# =================================================================================
# Plot the average goal_usd for each cluster
# =================================================================================
# Create copy of the original DataFrame
kickstarter_df_copy = kickstarter_df.copy()

# Add Cluster_Label column to the DataFrame X
kickstarter_df_copy['Cluster_Label'] = clusters
kickstarter_df_copy.info()

# Calculate the average goal_usd for each cluster
avg_goal_by_cluster = kickstarter_df_copy.groupby('Cluster_Label')['goal_usd'].mean()

# Define custom colors for positive and negative values
positive_colors = ["#54bebe"]
negative_colors = ["#d7658b"]

# Plot the average goal_usd for each cluster
bar_plot = sns.barplot(x=avg_goal_by_cluster.index, y=avg_goal_by_cluster, palette=positive_colors + negative_colors)

# Add labels and title
plt.xlabel('Cluster')
plt.ylabel('Average Goal (USD)')
plt.title('Average Goal (USD) for K-Prototypes Clustering')

# Set font color to black
for text in bar_plot.texts:
    text.set_color('black')

# Add labels to the bars with color based on the value
for i, v in enumerate(avg_goal_by_cluster):
    color = positive_colors[0] if v > 0 else negative_colors[0]
    bar_plot.patches[i].set_facecolor(color)
    va = 'top' if v < 0 else 'bottom'
    plt.text(i, v, f'{v:.2f}', ha='center', va=va, fontsize=8, color='black')

# Save the figure
plt.savefig('/Users/kritikanayyar/Documents/MMA/2. Sept-Dec_Fall 2023/F23-INSY-662-Data Mining and Visualization/Individual Project/average_goal_usd_plot.png', bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

# =================================================================================
# Plot the average usd_pledged for each cluster
# =================================================================================
# Calculate the average usd_pledged for each cluster
avg_pledged_by_cluster = kickstarter_df_copy.groupby('Cluster_Label')['usd_pledged'].mean()

# Plot the average usd_pledged for each cluster
bar_plot = sns.barplot(x=avg_pledged_by_cluster.index, y=avg_pledged_by_cluster, palette=positive_colors + negative_colors)

# Add labels and title
plt.xlabel('Cluster')
plt.ylabel('Average Pledged (USD)')  # Update ylabel to reflect the change
plt.title('Average Pledged (USD) for K-Prototypes Clustering')

# Set font color to black
for text in bar_plot.texts:
    text.set_color('black')

# Add labels to the bars with color based on the value
for i, v in enumerate(avg_pledged_by_cluster):
    color = positive_colors[0] if v > 0 else negative_colors[0]
    bar_plot.patches[i].set_facecolor(color)
    va = 'top' if v < 0 else 'bottom'
    plt.text(i, v, f'{v:.2f}', ha='center', va=va, fontsize=8, color='black')

# Save the figure
plt.savefig('/Users/kritikanayyar/Documents/MMA/2. Sept-Dec_Fall 2023/F23-INSY-662-Data Mining and Visualization/Individual Project/average_pledged_usd_plot.png', bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

# =================================================================================
# Plot the average backers_count for each cluster
# =================================================================================
# Create copy of the original DataFrame
kickstarter_df_copy = kickstarter_df.copy()

# Add Cluster_Label column to the DataFrame X
kickstarter_df_copy['Cluster_Label'] = clusters
kickstarter_df_copy.info()

# Calculate the average backers_count for each cluster
avg_backers_by_cluster = kickstarter_df_copy.groupby('Cluster_Label')['backers_count'].mean()

# Define custom colors for positive and negative values
positive_colors = ["#54bebe"]
negative_colors = ["#d7658b"]

# Plot the average backers_count for each cluster
bar_plot = sns.barplot(x=avg_backers_by_cluster.index, y=avg_backers_by_cluster, palette=positive_colors + negative_colors)

# Add labels and title
plt.xlabel('Cluster')
plt.ylabel('Average Backers Count')
plt.title('Average Backers Count for K-Prototypes Clustering')

# Set font color to black
for text in bar_plot.texts:
    text.set_color('black')

# Add labels to the bars with color based on the value
for i, v in enumerate(avg_backers_by_cluster):
    color = positive_colors[0] if v > 0 else negative_colors[0]
    bar_plot.patches[i].set_facecolor(color)
    va = 'top' if v < 0 else 'bottom'
    plt.text(i, v, f'{v:.2f}', ha='center', va=va, fontsize=8, color='black')

# Save the figure
plt.savefig('/Users/kritikanayyar/Documents/MMA/2. Sept-Dec_Fall 2023/F23-INSY-662-Data Mining and Visualization/Individual Project/average_backers_count_plot.png', bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

# =================================================================================
# Convert 'state' to numeric (successful=1, failure=0)
kickstarter_df['state_numeric'] = (kickstarter_df['state'] == 'successful').astype(int)

# Create a copy of the original DataFrame
kickstarter_df_copy = kickstarter_df.copy()

# Add 'Cluster_Label' column to the DataFrame
kickstarter_df_copy['Cluster_Label'] = clusters

# Plot scatterplot for 'usd_pledged' with 'state_numeric' on x-axis and 'Cluster_Label' as hue
plt.figure(figsize=(10, 8))
sns.scatterplot(x='state_numeric', y='usd_pledged', hue='Cluster_Label', data=kickstarter_df_copy, palette=custom_colors, marker='o', alpha=0.7)

# Customize the plot
plt.title('Pledged (USD) by State (Successful vs Failure)')
plt.xlabel('State (Failure=0, Successful=1)')
plt.ylabel('Pledged (USD)')
plt.legend(title='Cluster_Label')
plt.grid(True)

# Save the figure with dpi set to 300
plt.savefig('/Users/kritikanayyar/Documents/MMA/2. Sept-Dec_Fall 2023/F23-INSY-662-Data Mining and Visualization/Individual Project/pledged_by_state_plot.png', bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

# =================================================================================
# Display pair plots for 'usd_pledged', 'goal_usd', 'backers_count', 'name_len_clean', 'blurb_len_clean', 
# 'create_to_launch_days', 'launch_to_deadline_days', 'launch_to_state_change_days' against 'state'
columns_of_interest = ['usd_pledged', 'goal_usd', 'backers_count', 'name_len_clean', 'blurb_len_clean', 
                       'create_to_launch_days', 'launch_to_deadline_days', 'launch_to_state_change_days', 'state_numeric', 'Cluster_Label']
pair_plot_data = kickstarter_df_copy[columns_of_interest]

# Plot pair plots against 'state' with custom colors
sns.pairplot(pair_plot_data, hue='Cluster_Label', palette=custom_colors, markers='o', plot_kws={'alpha': 0.7}, 
             vars=['state_numeric', 'usd_pledged', 'goal_usd', 'backers_count', 'name_len_clean', 'blurb_len_clean', 
                   'create_to_launch_days', 'launch_to_deadline_days', 'launch_to_state_change_days'], kind='scatter')

# Customize the plot
plt.suptitle('Pair Plots of with Cluster Labels', y=1.02)

# Save the figure
plt.savefig('/Users/kritikanayyar/Documents/MMA/2. Sept-Dec_Fall 2023/F23-INSY-662-Data Mining and Visualization/Individual Project/pair_plots_numeric.png', bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

# =================================================================================
# Define columns_of_interest
columns_of_interest = ['usd_pledged', 'goal_usd', 'backers_count', 'name_len_clean', 'blurb_len_clean', 
                       'create_to_launch_days', 'launch_to_deadline_days', 'launch_to_state_change_days']

# Create an empty DataFrame to store the results
result_df = pd.DataFrame(columns=['Columns_of_Interest', 'Cluster_Label', 'State_numeric_0', 'State_numeric_1'])

# Iterate over each column of interest
for col in columns_of_interest:
    # Group by Cluster_Label and state_numeric, then count the occurrences
    counts = kickstarter_df_copy.groupby(['Cluster_Label', 'state_numeric'])[col].count().unstack().reset_index()
    
    # Rename the count columns
    counts = counts.rename(columns={0: 'State_numeric_0', 1: 'State_numeric_1'})
    
    # Add the current column_of_interest to the DataFrame
    counts['Columns_of_Interest'] = col
    
    # Reorder the columns
    counts = counts[['Columns_of_Interest', 'Cluster_Label', 'State_numeric_0', 'State_numeric_1']]
    
    # Append the results to the main DataFrame
    result_df = result_df.append(counts, ignore_index=True)

# Display the resulting DataFrame
print(result_df)

result_df[:8]

# =================================================================================
# Cluster 0
# =================================================================================
# Display pair plots for 'usd_pledged', 'goal_usd', 'backers_count', 'name_len_clean', 'blurb_len_clean', 
# 'create_to_launch_days', 'launch_to_deadline_days', 'launch_to_state_change_days' against 'state'
columns_of_interest = ['usd_pledged', 'goal_usd', 'backers_count', 'name_len_clean', 'blurb_len_clean', 
                       'create_to_launch_days', 'launch_to_deadline_days', 'launch_to_state_change_days', 'state_numeric', 'Cluster_Label']
pair_plot_data = kickstarter_df_copy[columns_of_interest]

# Specify the cluster ID for which you want to create pair plots
selected_cluster_id = 0

# Filter data for the selected cluster
selected_cluster_data = pair_plot_data[pair_plot_data['Cluster_Label'] == selected_cluster_id]

# Plot pair plots against 'state' with custom colors for the selected cluster
sns.pairplot(selected_cluster_data, hue='Cluster_Label', palette=custom_colors, markers='o', plot_kws={'alpha': 0.7}, 
             vars=['state_numeric', 'usd_pledged', 'goal_usd', 'backers_count', 'name_len_clean', 'blurb_len_clean',
                   'create_to_launch_days', 'launch_to_deadline_days', 'launch_to_state_change_days'], kind='scatter')

# Customize the plot
plt.suptitle(f'Pair Plots for Cluster {selected_cluster_id} with Cluster Labels', y=1.02)

# Save the figure
plt.savefig(f'/Users/kritikanayyar/Documents/MMA/2. Sept-Dec_Fall 2023/F23-INSY-662-Data Mining and Visualization/Individual Project/pair_plots_cluster_{selected_cluster_id}.png', bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

# =================================================================================
# Display pair plots for 'deadline_month', 'state_changed_at_month', 'created_at_month', 'launched_at_month' against 'state'
columns_of_interest = ['deadline_month', 'state_changed_at_month', 'created_at_month', 'launched_at_month', 'state_numeric', 'Cluster_Label']
pair_plot_data = kickstarter_df_copy[columns_of_interest]

# Plot pair plots against 'state'
sns.pairplot(pair_plot_data, hue='Cluster_Label', palette=custom_colors, markers='o', plot_kws={'alpha': 0.7}, 
             vars=['state_numeric', 'deadline_month', 'state_changed_at_month', 'created_at_month', 'launched_at_month'], kind='scatter')

# Customize the plot
plt.suptitle('Pair Plots of with Cluster Labels', y=1.02)

# Save the figure
plt.savefig('/Users/kritikanayyar/Documents/MMA/2. Sept-Dec_Fall 2023/F23-INSY-662-Data Mining and Visualization/Individual Project/pair_plots_dates_month.png', bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

# =================================================================================
# Display pair plots for 'deadline_day', 'state_changed_at_day', 'created_at_day', 'launched_at_day' against 'state'
columns_of_interest = ['deadline_day', 'state_changed_at_day', 'created_at_day', 'launched_at_day', 'state_numeric', 'Cluster_Label']
pair_plot_data = kickstarter_df_copy[columns_of_interest]

# Plot pair plots against 'state'
sns.pairplot(pair_plot_data, hue='Cluster_Label', palette=custom_colors, markers='o', plot_kws={'alpha': 0.7}, 
             vars=['state_numeric', 'deadline_day', 'state_changed_at_day', 'created_at_day', 'launched_at_day'], kind='scatter')

# Customize the plot
plt.suptitle('Pair Plots of with Cluster Labels', y=1.02)

# Save the figure
plt.savefig('/Users/kritikanayyar/Documents/MMA/2. Sept-Dec_Fall 2023/F23-INSY-662-Data Mining and Visualization/Individual Project/pair_plots_dates_day.png', bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

# =================================================================================
# Display pair plots for 'deadline_yr', 'state_changed_at_yr', 'created_at_yr', 'launched_at_yr' against 'state'
columns_of_interest = ['deadline_yr', 'state_changed_at_yr', 'created_at_yr', 'launched_at_yr', 'state_numeric', 'Cluster_Label']
pair_plot_data = kickstarter_df_copy[columns_of_interest]

# Plot pair plots against 'state'
sns.pairplot(pair_plot_data, hue='Cluster_Label', palette=custom_colors, markers='o', plot_kws={'alpha': 0.7}, 
             vars=['state_numeric', 'deadline_yr', 'state_changed_at_yr', 'created_at_yr', 'launched_at_yr'], kind='scatter')

# Customize the plot
plt.suptitle('Pair Plots of with Cluster Labels', y=1.02)

# Save the figure
plt.savefig('/Users/kritikanayyar/Documents/MMA/2. Sept-Dec_Fall 2023/F23-INSY-662-Data Mining and Visualization/Individual Project/pair_plots_dates_year.png', bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

# =================================================================================
# Convert boolean columns to numeric (0 and 1)
boolean_columns = ['disable_communication', 'staff_pick', 'spotlight']
kickstarter_df_copy[boolean_columns] = kickstarter_df_copy[boolean_columns].astype(int)

# Select relevant columns for the pair plot
columns_of_interest = ['state_numeric', 'disable_communication', 'staff_pick', 'spotlight', 'Cluster_Label']
pair_plot_data = kickstarter_df_copy[columns_of_interest]

# Plot pair plots against 'state'
sns.pairplot(pair_plot_data, hue='Cluster_Label', palette=custom_colors, markers='o', plot_kws={'alpha': 0.7}, 
             vars=columns_of_interest[:-1], kind='scatter')

# Customize the plot
plt.suptitle('Pair Plots with Cluster Labels', y=1.02)

# Save the figure
plt.savefig('/Users/kritikanayyar/Documents/MMA/2. Sept-Dec_Fall 2023/F23-INSY-662-Data Mining and Visualization/Individual Project/pair_plots_booleans.png', bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

# ==============================================================================
# Measuring clustering performance: Silhouette method
# ==============================================================================
# It is hard to measure ‘distance’ between categorical variables.
# I was unable to run 'kprototypes.matching_dissim' and 'kprototypes.check_distance' in Python

# ==============================================================================
# Other models
# ==============================================================================
# Dummify boolean features and non-numeric features
kickstarter_df = pd.get_dummies(kickstarter_df, columns=boolean_features) # 13435 rows x 41 columns
kickstarter_df = pd.get_dummies(kickstarter_df, columns=non_numeric_features) # 13435 rows x 89 columns

# Display the first 5 rows of the dataset
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(kickstarter_df.head(5))
    
# Standardize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_std = scaler.fit_transform(kickstarter_df)

# ==============================================================================
# Hierarchical Agglomerative Clustering
# ==============================================================================
from sklearn.cluster import AgglomerativeClustering

# Perform hierarchical agglomerative clustering - complete linkage
cluster = AgglomerativeClustering(n_clusters=3, linkage='complete', affinity='euclidean')
cluster_labels = cluster.fit_predict(X_std)
# n_clusters defines the number of clusters you want to create
# method defines the linkage algorithm - 'single', 'complete', 'average', 'centroid', 'ward'
# metric defines the distance metric - 'euclidean', 'minkowski', 'cityblock', 'cosine', 'precomputed'

# Print cluster labels
print(cluster_labels)

pd.DataFrame(list(zip(kickstarter_df['goal_usd'],np.transpose(cluster.labels_))), columns = ['Goal','Cluster label'])

# Visualize the dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage

linked = linkage(X_std, 'complete', 'euclidean')

plt.figure(figsize=(15, 8))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Agglomerative Clustering Dendrogram - Complete Linkage')
plt.xlabel('Sample Index')
plt.ylabel('Cluster Distance')
plt.show()

# ==============================================================================
# Perform hierarchical agglomerative clustering - average linkage
cluster = AgglomerativeClustering(n_clusters=3, linkage='average', affinity='euclidean')
cluster_labels = cluster.fit_predict(X_std)

# Print cluster labels
print(cluster_labels)

# Visualize the dendrogram
linked = linkage(X_std, 'average', 'euclidean')

plt.figure(figsize=(15, 8))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Agglomerative Clustering Dendrogram - Average Linkage')
plt.xlabel('Sample Index')
plt.ylabel('Cluster Distance')
plt.show()