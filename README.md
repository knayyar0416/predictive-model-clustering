# Exploring distinctive features for various projects using unsupverised clustering algorithm
In this project, 

## 🌐 About Kickstarter
[Kickstarter](https://www.kickstarter.com) is a platform where creators share their project visions with the communities that will come together to fund them.

## 💼 What did I discover?
By grouping the Kickstarter projects using unsupervised clustering, the management could uncover distinct characteristics within each cluster. Below are the unique characteristics of 8 clusters.
#### ⚖️🚀 Moderate Goals, Quick Launchers
These projects showed a preference for swift project initiation, and the pledged amount tended to increase as the launch- to-deadline period extended.
### 💰📆 High Goals, Recent Projects
Projects with high fundraising goals and recent deadlines. Interestingly, some projects in this cluster managed to attract a high number of backers, even with lofty fundraising goals.
### 🤝👥 Backer-Friendly Projects
This cluster stood out for its projects' ability to attract a significant number of backers.
### ⚖️💡 Modest Achievers
These projects struck a balance between goal and backers, with an average duration between project creation and launch.
### 👏💰 Well-Supported Initiatives
Projects in this cluster enjoyed high backers and pledged amounts. These projects maintained a relatively low goal and were picked by staff, potentially for the spotlight.
### ⚡🚀 Rapid Projects
This cluster embodied a preference for rapid project development and execution.
### 🌟🎯 Ambitious Newcomers
With a 24% success rate, Cluster 7 features projects with relatively higher fundraising goals compared to the amount pledged.
### 🕰️💼 Long-Term High-Stakes
Cluster 8, with a 29% success rate, represents projects with the highest fundraising goals across clusters. Notably, these projects were created a long time back, suggesting a lower success rate over time.

## 🔄 How did I achieve this? 
Below is the detailed process for model building.
1. 🧹 Data preprocessing:
   - Removed non- informative columns including 𝑖𝑑, 𝑛𝑎𝑚𝑒, 𝑛𝑎𝑚𝑒_𝑙𝑒𝑛, 𝑏𝑙𝑢𝑟𝑏_𝑙𝑒𝑛, and 𝑝𝑙𝑒𝑑𝑔𝑒𝑑.
   - Created a new feature, 𝑔𝑜𝑎𝑙_𝑢𝑠𝑑, by multiplying 𝑔𝑜𝑎𝑙 and 𝑠𝑡𝑎𝑡𝑖𝑐_𝑢𝑠𝑑_𝑟𝑎𝑡𝑒.
   - Replaced non-US countries with 'Non-US' and filled missing values in 𝑐𝑎𝑡𝑒𝑔𝑜𝑟𝑦 with 'No Category'.
   - Dropped irrelevant columns like original date columns and hour- specific columns.
2. 🔍 Anomaly detection:
   - Before running any clustering algorithm, I ran an Isolation Forest Model for Anomaly Detection to identify and remove anomalies. The model deteted 1,344 anomalies with unusually high 𝑔𝑜𝑎𝑙_𝑢𝑠𝑑, 𝑏𝑎𝑐𝑘𝑒𝑟𝑠_𝑐𝑜𝑢𝑛𝑡 and 𝑢𝑠𝑑_𝑝𝑙𝑒𝑑𝑔𝑒𝑑.
3. 🤖 Clustering model:
   - I decided to perform K-Prototypes clustering to accommodate both the numerical and categorical features.
   - By testing cost function for different values of K for K-Prototypes Clustering, I could observe K=8 and K=10 are the elbow points at which the cost drops drastically. I chose K=8 since it was giving me better business interpretations.

## 🎉 Conclusion
In summary, the K-Prototypes clustering algorithm provided valuable insights into diverse project profiles, offering a comprehensive understanding of Kickstarter projects. Overall, successful projects have high number of backers, pledged amounts, and are staff-picked, ultimately securing a place on Kickstarter spotlight page.

## 🔗 Supporting files
- 👩‍💻 [Python script for final selected clustering model]()
- 👩‍💻 [Python script for all clustering models]()
- 📁 [Entire dataset](kickstarter.xlsx) and [Data Dictionary](kickstarter-data-dictionary.xlsx)
- 📊 [Data exploration and other charts](Images)

