# %%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sklearn

diabetic_data = pd.read_csv('diabetic_data.csv')

from sklearn.cluster import KMeans

#placing data in x and y
x = diabetic_data['time_in_hospital']
y = diabetic_data['discharge_disposition_id']

#zipping the data together
data = list(zip(x, y))
inertias = []

#use elbow method to find best fit for number of clusters
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# %%
#use the data from the elbow method to cluster the data
#the number of clusters that is a good K value is 3 clusters
km = KMeans(n_clusters= 3)
km.fit(data)

kmean = KMeans(n_clusters = 3)
km1 = kmean.fit_predict(data)

#plot the data
plt.scatter(x,y, c=km.labels_)
plt.scatter(kmean.cluster_centers_[:, 0], kmean.cluster_centers_[:, 1], s = 200, c = 'red', marker = '*', label = 'centroids')
plt.grid()
plt.xlabel('Time in Hospital')
plt.ylabel('discharge disposition id')
plt.show()

# %%



