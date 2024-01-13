# Import libraries
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)
custom_colors = ["#ebdc78", "#63bff0", "#1984c5", "#54bebe", "#df979e", "#d7658b", "#ffd3b6", "#ee4035"]

# Read the dataset
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
# Remove identified anomalies from kickstarter_df_copy
kickstarter_df_copy = kickstarter_df_copy.drop(index=anomaly_values.index) # 13435 rows reduced to 12091 rows
# Display the anomaly values from dataset
anomaly_values.describe()
# Display the descriptive statistics of the dataset
kickstarter_df.describe()

# ==============================================================================
# K-Prototypes Clustering
# ==============================================================================
# The K-Prototypes clustering algorithm is a partitioning-based clustering algorithm that we can use to cluster 
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

print(kickstarter_array)

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
num_distance = kproto.cost_

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
# Initialize the cluster list
cluster_dict = []
# Append cluster labels to the list 'cluster_dict'
for c in clusters:
    cluster_dict.append(c)
# Print the cluster labels    
cluster_dict
# Create copy of the original DataFrame
# kickstarter_df_copy = kickstarter_df.copy()
kickstarter_df_copy['cluster'] = cluster_dict
# Display the count of projects in each cluster
kickstarter_df_copy['cluster'].value_counts()
    
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
avg_values_by_cluster = kickstarter_df_copy.groupby('cluster').agg({
    'usd_pledged': 'mean',
    'goal_usd': 'mean', 
    'backers_count': 'mean',
    'deadline_yr': 'mean',
    'created_at_yr': 'mean',
    'launched_at_yr': 'mean',
    'create_to_launch_days': 'mean',
    'launch_to_deadline_days': 'mean',
    'launch_to_state_change_days': 'mean',
}).reset_index()  
# Rename columns for clarity
avg_values_by_cluster.columns = ['Cluster', 'Average Pledged', 'Average Goal (USD)', 'Average Backers Count',
                                 'Deadline Year', 'Created At Year', 'Launched At Year', 'Create To Launch_days',
                                 'Launch To Deadline Days', 'Launch To State Change Days']
# Display the DataFrame
print(avg_values_by_cluster)    

# =================================================================================
# Plot the average goal_usd for each cluster
# =================================================================================
# Calculate the average goal_usd for each cluster
avg_goal_by_cluster = kickstarter_df_copy.groupby('cluster')['goal_usd'].mean()
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
plt.savefig('average_goal_usd_plot.png', bbox_inches='tight', dpi=300)
# Show the plot
plt.show()

# =================================================================================
# Plot the average usd_pledged for each cluster
# =================================================================================
# Calculate the average usd_pledged for each cluster
avg_pledged_by_cluster = kickstarter_df_copy.groupby('cluster')['usd_pledged'].mean()
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
plt.savefig('average_pledged_usd_plot.png', bbox_inches='tight', dpi=300)
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
plt.savefig('average_backers_count_plot.png', bbox_inches='tight', dpi=300)
# Show the plot
plt.show()

# =================================================================================
# Convert 'state' to numeric (successful=1, failure=0)
kickstarter_df['state_numeric'] = (kickstarter_df['state'] == 'successful').astype(int)
# Create a copy of the original DataFrame
kickstarter_df_copy = kickstarter_df.copy()
# Add 'Cluster_Label' column to the DataFrame
kickstarter_df_copy['Cluster_Label'] = clusters

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
plt.savefig('pair_plots_numeric.png', bbox_inches='tight', dpi=300)
# Show the plot
plt.show()

# =================================================================================
# View for each cluster
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
plt.savefig(f'pair_plots_cluster_{selected_cluster_id}.png', bbox_inches='tight', dpi=300)
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
plt.savefig('pair_plots_dates_month.png', bbox_inches='tight', dpi=300)
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
plt.savefig('pair_plots_dates_day.png', bbox_inches='tight', dpi=300)
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
plt.savefig('pair_plots_dates_year.png', bbox_inches='tight', dpi=300)
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
plt.savefig('pair_plots_booleans.png', bbox_inches='tight', dpi=300)
# Show the plot
plt.show()

# ==============================================================================
# Measuring clustering performance: Silhouette method
# ==============================================================================
# It is hard to measure ‘distance’ between categorical variables.
# I was unable to run 'kprototypes.matching_dissim' and 'kprototypes.check_distance' in Python

# ==============================================================================
# Another model: Hierarchical Agglomerative Clustering
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
# Plot the dendrogram
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
# Plot the dendrogram
plt.figure(figsize=(15, 8))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Agglomerative Clustering Dendrogram - Average Linkage')
plt.xlabel('Sample Index')
plt.ylabel('Cluster Distance')
plt.show()

################################################################################