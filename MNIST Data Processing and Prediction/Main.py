import pdb
import numpy as np
import Functions as hw3
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 


#-------------------------------------------------------------------------------
# MNIST Data
#-------------------------------------------------------------------------------

def raw_mnist_features(x):
    """
    @param x (n_samples,m,n) array with values in (0,1)
    @return (m*n,n_samples) reshaped array where each entry is preserved
    """
    (n, rows, columns) = x.shape
    return x.reshape(rows*columns, n)




def row_average_features(x):
    """

    @param x (n_samples,m,n) array with values in (0,1)
    @return (m,n_samples) array where each entry is the average of a row
    """
    (n_samples, m, n) = x.shape
    
    averages = np.mean(x, axis=2).T  
    
    return averages

def col_average_features(x):
    """
    @param x (m,n) array with values in (0,1)
    @return (n,1) array where each entry is the average of a column
    """
    (n_samples,m,n) = x.shape
    
    return np.mean(x, axis=1).T


def top_bottom_features(x):
    """
    Compute the average values for the top and bottom halves of the images.
    
    @param x: (n_samples, m, n) array with values in (0,1)
    @return: (2, n_samples) array where the first row contains the averages 
             of the top halves, and the second row contains the averages of the bottom halves.
    """
    (n_samples, rows, columns) = x.shape

    # Calculate the split point for top and bottom halves
    split = rows // 2

    # Compute the averages for the top and bottom halves for each sample
    top_half_averages = np.mean(x[:, :split, :], axis=(1, 2))  
    bottom_half_averages = np.mean(x[:, split:, :], axis=(1, 2)) 

    # Stack the results as a 2-row array
    result = np.vstack([top_half_averages, bottom_half_averages])
    
    return result


mnist_data_all = hw3.load_mnist_data(range(10))

print('mnist_data_all loaded. shape of single images is', mnist_data_all[0]["images"][0].shape)


d0 = mnist_data_all[0]["images"]
d1 = mnist_data_all[1]["images"]
d2 = mnist_data_all[2]["images"]
d3 = mnist_data_all[3]["images"]
d4 = mnist_data_all[4]["images"]
d5 = mnist_data_all[5]["images"]
d6 = mnist_data_all[6]["images"]
d7 = mnist_data_all[7]["images"]
d8 = mnist_data_all[8]["images"]
d9 = mnist_data_all[9]["images"]
y0 = np.repeat(-1, len(d0)).reshape(1,-1)
y1 = np.repeat(1, len(d1)).reshape(1,-1)
y2 = np.repeat(1, len(d2)).reshape(1,-1)
y3 = np.repeat(1, len(d3)).reshape(1,-1)
y4 = np.repeat(1, len(d4)).reshape(1,-1)
y5 = np.repeat(1, len(d5)).reshape(1,-1)
y6 = np.repeat(1, len(d6)).reshape(1,-1)
y7 = np.repeat(1, len(d7)).reshape(1,-1)
y8 = np.repeat(1, len(d8)).reshape(1,-1)
y9 = np.repeat(1, len(d9)).reshape(1,-1)
# data goes into the feature computation functions
data = np.vstack((d0, d1))
# labels can directly go into the perceptron algorithm
labels = np.vstack((y0.T, y1.T)).T


# use this function to evaluate accuracy
acc = hw3.get_classification_accuracy(top_bottom_features(data), labels)
print('accuracy: ', acc)


    

