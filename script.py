import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tslearn.clustering import TimeSeriesKMeans
import math
import numpy as np



def read_data(folder_path="Datasets_CSV/Domain1_csv/"):
    matrix_list = []
    y_list = []
    df_Y= pd.DataFrame(columns=['number'])
    count=0
    maxlen=0
    for subject in range(1):
        for number in range(10):
            for attemp in range(10):
                
                #we read the data from the specific .csv file
                file_path = folder_path + 'Subject{}-{}-{}.csv'.format(subject+1, number, attemp+1) #To change if your files have different name
                data_X = pd.read_csv(file_path)
                
                count+=len(data_X)
                if len(data_X)>maxlen:
                    maxlen=len(data_X)
                #Add the specific label to the labels' dataframe
                df_Y = pd.concat([df_Y, pd.DataFrame({'number': [number]})], ignore_index=True) 
                

                #Before the the dataframe converted to array to the matrix list we have to preprocess it:
                
                # Extract the x and y columns
                x = data_X['<x>']
                y = data_X['<y>']
                z = data_X['<z>']
                t = data_X['<t>']

                # Standardize the data
                scaler = StandardScaler()  #TODO: possible option to change ro RobustScaler?
                x = scaler.fit_transform(x.values.reshape(-1, 1)).flatten()
                y = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
                z = scaler.fit_transform(z.values.reshape(-1, 1)).flatten()
                t = scaler.fit_transform(t.values.reshape(-1, 1)).flatten()

                # Apply PCA with just 2 components so we have just two coordinates (which actually represent X,Y)
                pca = PCA(n_components=2)
                data_X_standarized = pd.DataFrame({'x': x, 'y': y, 'z': z})
                pca.fit(data_X_standarized)
                data_X_transformed = pca.transform(data_X_standarized) #it returns an array.

                
                #add the preprocessed data to the matrix_list (actually the data is a matrix)
                matrix_list.append(data_X_transformed)
    avg=count/1000
    #convert the List of matrix with the data into a np array
    array_matrix = np.array(matrix_list,dtype=object)
    #convert the dataframe of the Labels into a np array 
    labels = np.array(np.ravel(df_Y))
    return array_matrix,labels,avg,maxlen


arr,lab,avg,max_len=read_data()

# pad the time series to the max_len
X_padded = np.zeros((arr.shape[0], max_len, arr[0].shape[1]))
for i in range(arr.shape[0]):
    X_padded[i, :len(arr[i]), :] = arr[i]

#------------------ 2D plot ------------------------------

# Create the scatter plot
"""plt.scatter(x, y)

# Add labels and a title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter plot of X and Y')

# Show the plot
plt.show()"""

#--------------------------- 3D plot ---------------------------

"""fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)

# Add labels and a title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter plot of X, Y, and Z')

# Show the plot
plt.show()"""

# -------------------- Plot 2D with the PCA --------------------------------------

"""plt.scatter(transformed[:,0], transformed[:,1])

# Add labels and a title
plt.xlabel('PCA_1')
plt.ylabel('PCA_2')
plt.title('Scatter plot of X and Y')

# Show the plot
plt.show()"""
print(X_padded.shape)
cluster_count = 10
X = X_padded.reshape(-1, 2)
print(X.shape)
km = TimeSeriesKMeans(n_clusters=cluster_count, metric="dtw")
string = km.fit_predict(X)
string = string.reshape(100, 172)
print(string)
print(string.shape)