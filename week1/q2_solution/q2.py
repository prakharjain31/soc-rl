import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def PCA(init_array: pd.DataFrame):

    sorted_eigenvalues = None
    final_data = None
    dimensions = 2

    # TODO: transform init_array to final_data using PCA
    final_data = pd.DataFrame.to_numpy(init_array)

    normalised_data = (final_data - final_data.mean(axis=0)) #1000,4
    # print(normalised_data)
    
    covariance_matrix = np.cov(normalised_data , rowvar=0) #4,4
    
    eigen_values , eigen_vectors = np.linalg.eig(covariance_matrix)
    # print(eigen_vectors)
    eigen_indices = np.argsort((np.abs(eigen_values)))[:-(dimensions+1):-1]
    
    sorted_eigenvectors = eigen_vectors[:,eigen_indices] # 4,2
    # print(sorted_eigenvectors)
    
    sorted_eigenvalues = eigen_values[eigen_indices]
    # print(sorted_eigenvalues)
    final_data = normalised_data @ sorted_eigenvectors #1000,2
    
    
    # END TODO
    

    return sorted_eigenvalues, final_data


if __name__ == '__main__':
    init_array = pd.read_csv("pca_data.csv", header = None)
    sorted_eigenvalues, final_data = PCA(init_array)
    np.savetxt("transform.csv", final_data, delimiter = ',')
    for eig in sorted_eigenvalues:
        print(eig)
    print(final_data)
    # TODO: plot and save a scatter plot of final_data to out.png
    
    plt.scatter(final_data[:,0] ,final_data[:,1])
    plt.axis((-15,15,-15,15))
    plt.savefig("out.png")
    # END TODO
