#import pandas library
import datasets as datasets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling
import scipy.stats as stats
from scipy.cluster import hierarchy
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn import preprocessing
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn import datasets

#load the data set file
from scipy.sparse import data

#read data set file
dataset = pd.read_csv('CW1_COVID_world_data.csv')

#display number of records in the data set
print("Total number of Columns, Rows in Covid World Data")
print(dataset.shape)

#display the dataset
print("Covid World Data")
print(dataset)

#DataCleaning
#identify missing values
print("Total Missing Values in Data")
print(dataset.isna().sum())

#replacing missing values using fillna() function
dataset.fillna(0, inplace=True)

#check replacement of missing values
print("Total Missing Values in Data")
print(dataset.isna().sum())

#deleting first and last column
dataset.drop([0,226], inplace = True)

#verify deleted rows
print("Covid World Data")
print(dataset)

#displays the datatype of the data set
print("Datatypes of Covid World Data")
print(dataset.dtypes)

#change the datatype of the object fields to float
import numpy as np
obj_columns =dataset.select_dtypes(include=object).columns.tolist()
dataset[obj_columns] = dataset[obj_columns].astype('string')
#replace spaces between words with "_"
dataset.columns = dataset.columns.str.replace(' ','_')
#removes all commas from data in the population column
dataset['Population'] = dataset['Population'].str.replace(',','')
#converts datatype of Population column from string to float
dataset['Population'] = dataset['Population'].astype(int)

#removes all commas from data in the total deaths column
dataset['Total_Deaths'] = dataset['Total_Deaths'].str.replace(',','')
#converts datatyoe of total deaths column from string to integer
dataset['Total_Deaths'] = dataset['Total_Deaths'].astype(int)

print("Changed datatypes for columns")
print(dataset.dtypes)

#Data Normalisation
no_country_dataset = dataset.iloc[:,1:]
print("Iloc")
print(no_country_dataset)
scaler = MinMaxScaler()
scaled_covid_dataset = scaler.fit_transform(no_country_dataset)
print("Normalised Data")
print(scaled_covid_dataset)
print(scaled_covid_dataset.shape)

#Identification of Outliers
fig = plt.figure(figsize =(10, 7))
plt.boxplot(scaled_covid_dataset)
plt.show()

print("test")
#PCA using Dimensionality Reduction
pca = PCA()
dataset_pca = pca.fit_transform(scaled_covid_dataset)

plt.figure(figsize = (10,10))
var = np.round(pca.explained_variance_ratio_ * 100, decimals = 1)
labels=[str(x) for x in range(1,len(var)+1)]
plt.bar(x=range(1,len(var)+1),height = var, tick_label = labels)
plt.show()

pca = PCA(0.9)
dataset_pca = pca.fit_transform(scaled_covid_dataset)
# plot with the explained variances
features = range(pca.n_components_)
print("Features", features)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
#plt.show(output=".png")
plt.show()

#save components to a Dataframe
PCA_components = pd.DataFrame(dataset_pca)

wcss = []
for i in range(1,15):
    model = KMeans(n_clusters = i, init = "k-means++")
    model.fit(PCA_components.iloc[:,0:15])
    wcss.append(model.inertia_)
#Elbow method graph
plt.plot(range(1,15),wcss)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('CW_EMG.png', dpi=1080, format='png')
plt.show()

wcss = []
for i in range(1,15):
    model = KMeans(n_clusters = i, init = "k-means++")
    model.fit(PCA_components.iloc[:,0:15])
    wcss.append(model.inertia_)
#Elbow method with Yellowbrick Visualiser
visualizer = KElbowVisualizer(model, k=(1,15))
visualizer.fit(PCA_components.iloc[:,0:15])
visualizer.show(outpath="CW_EMG_YB_ex2.png")
visualizer.show()

print("Inertia",model.inertia_)

#clustering using KMeans
model = KMeans(n_clusters = 4, init = "k-means++")
label = model.fit_predict(PCA_components.iloc[:,0:15])
centers = np.array(model.cluster_centers_)
uniq = np.unique(label)

#K-means clustering with PCA results
df_coviddata_pca_kmeans = pd.concat([dataset.reset_index(drop=True), pd.DataFrame(dataset_pca)], axis=1)
df_coviddata_pca_kmeans.columns.values[-4:] = ["component 1","component 2","component 3","component 4"]
df_coviddata_pca_kmeans["Covid Data K-means PCA"] = model.labels_
print(df_coviddata_pca_kmeans)
#df_coviddata_pca_kmeans.to_csv('New_CW_Components.csv')
colors = ['red', 'green','orange','blue']
#assign a color to each features (note that we are using features as target)
features_colors = [ colors[label[i]] for i in
                    range(len(PCA_components.iloc[:,0:15]))]
T=PCA_components.iloc[:,0:15]
#plot the PCA cluster components
plt.scatter(T[0], T[1], c=features_colors, marker='o', alpha=0.4 )
plt.show()
plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=100, linewidths=3, c=colors )
#store the values of PCA component in variable: for easy writing
xvector = pca.components_[0] * max(T[0])
yvector = pca.components_[1] * max(T[1])
columns = no_country_dataset.columns
#plot the 'name of individual features' along with vector length
for i in range(len(columns)-1):
#plot arrows
    plt.arrow(0, 0, xvector[i], yvector[i], color='b', width=0.005, head_width=0.08, alpha=0.5)
    #plot name of features
    plt.text(xvector[i], yvector[i], list(columns)[i], color='b', alpha=0.75)
plt.scatter(T[0], T[1], c=features_colors, marker='o', alpha=0.4)
#plot the centroids
plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=100, linewidths=3, c=colors )
plt.text(-0.2650367677890810, 0.024799663146565900, "Ghana",color='black', alpha=0.75)
plt.show()

#Agglomerative Clustering hierarchical clustering
plt.figure()
plt.title("Coivd World Dendograms")
dend = shc.dendrogram(shc.linkage(scaled_covid_dataset, method='ward'))
# Add horizontal line.
plt.axhline(y=2.2, c='black', lw=1, linestyle='dashed')
plt.show()
#plt.savefig('CovidWorld_dendograms.png', dpi=1080, format='png')

#Projection
plt.figure()
plt.scatter(T[0], T[1], c=model.labels_, cmap='rainbow')
plt.title('Clusters by PCA Componets using Heirarchical means')
plt.xlabel('x_axis')
plt.ylabel('y_axis')
#plt.savefig('Covid19_hierarchical_clustering.png', dpi=1080, format='png')
plt.show()

#print("pca.n_components_:",pca.n_components_)
#print("model:",model.labels_)

#Sillohuette Clustering
# load petal data
iris = datasets.load_iris()
X = iris.data
y = iris.target
#Instantiate the KMeans models
km = KMeans(n_clusters=3, random_state=42)
#Fit the KMeans model
km.fit_predict(X)
#Calculate Silhoutte Score
score = silhouette_score(X, km.labels_, metric='euclidean')
#Print the score
print('Silhouetter Score: %.3f' % score)
fig, ax = plt.subplots(2, 2, figsize=(15,8))
for i in [2, 3, 4, 5]:
    '''
    Create KMeans instance for different number of clusters
    '''
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
    q, mod = divmod(i, 2)
    '''
    Create SilhouetteVisualizer instance with KMeans instance
    Fit the visualizer
    '''
    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(X)
visualizer.show()