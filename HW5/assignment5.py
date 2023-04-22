#!/usr/bin/env python
# coding: utf-8

print("Acknowledgment:")
print("https://github.com/pritishuplavikar/Face-Recognition-on-Yale-Face-Dataset")

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import glob
from numpy import linalg as la
import random
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import os

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import sklearn
from typing import Tuple, List
from typeguard import typechecked


@typechecked
def qa1_load(folder_path:str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the dataset (tuple of x, y the label).

    x should be of shape [165, 243 * 320]
    label can be extracted from the subject number in filename. ('subject01' -> '01 as label)
    """
    image_files = sorted([f for f in os.listdir(folder_path) if f.startswith('subject')])
    if len(image_files) == 0:
        raise ValueError('No image files found in the directory')

    # Load the images and store them in a numpy array
    images = np.zeros((len(image_files), 243*320))
    labels = np.zeros(len(image_files), dtype=np.int32)
    for i, filename in enumerate(image_files):
        # Load the image using matplotlib.image.imread() function
        image = mpimg.imread(os.path.join(folder_path, filename))

        # Reshape the image to a 1D array and store it in the images array
        images[i] = image.reshape(-1)

        # Extract the label from the filename and store it in the labels array
        label = int(filename[7:9])
        labels[i] = label

    return images, labels

@typechecked
def qa2_preprocess(dataset:np.ndarray) -> np.ndarray:
    """
    returns data (x) after pre processing

    hint: consider using preprocessing.MinMaxScaler
    """
    # Create a MinMaxScaler object and fit it to the dataset
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(dataset)

    # Use the scaler to transform the dataset
    preprocessed_dataset = scaler.transform(dataset)

    return preprocessed_dataset

@typechecked
def qa3_calc_eig_val_vec(dataset:np.ndarray, k:int)-> Tuple[PCA, np.ndarray, np.ndarray]:
    """
    Calculate eig values and eig vectors.
    Use PCA as imported in the code to create an instance
    return them as tuple PCA, eigen_value, eigen_vector
    """
    # Create a PCA object with n_components=k and fit it to the dataset
    pca = PCA(n_components=k)
    pca.fit(dataset)

    # Get the top k eigenvalues and eigenvectors from the PCA object
    eigen_values = pca.explained_variance_[:k]
    eigen_vectors = pca.components_[:k]

    return pca, eigen_values, eigen_vectors

def qb_plot_written(eig_values:np.ndarray):
    """
    No structure is required for this question. It does not have to return anything.
    Use this function to produce plots
    """
    # Sort the eigenvalues in descending order
    sorted_eig_values = np.sort(eig_values)[::-1]

    # Compute the cumulative sum of the eigenvalues
    cumsum_eig_values = np.cumsum(sorted_eig_values)

    # Compute the fraction of total energy captured by each principal component
    energy_fraction = cumsum_eig_values / cumsum_eig_values[-1]

    # Plot the curve of eigenvalues and energy fraction
    plt.plot(sorted_eig_values)
    plt.plot(energy_fraction)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Eigenvalues / Energy Fraction')
    plt.title('Eigenvalues and Energy Fraction vs. Number of Principal Components')
    plt.legend(['Eigenvalues', 'Energy Fraction'])
    plt.show()

    # Calculate the number of components needed to capture 50% of the energy
    num_components_50_percent = np.argmax(energy_fraction >= 0.5) + 1
    print('Number of components needed to capture 50% of the energy:', num_components_50_percent)

@typechecked
def qc1_reshape_images(pca:PCA, dim_x:int = 243, dim_y:int = 320) -> np.ndarray:
    """
    Reshape the PCA components into the shape of original image so that it can be visualized
    """
    # Get the number of PCA components (eigen vectors)
    n_components = pca.components_.shape[0]
    
    # Create a new array to store the reshaped eigen faces
    eigen_faces = np.zeros((n_components, dim_x, dim_y))

    # Reshape each eigen vector into the shape of the original image and store it in the eigen_faces array
    for i in range(n_components):
        eigen_faces[i] = pca.components_[i].reshape(dim_x, dim_y)

    return eigen_faces

def qc2_plot(org_dim_eig_faces: np.ndarray):
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 5))

    for i, ax in enumerate(axs.flat):
        ax.imshow(org_dim_eig_faces[i].reshape(243, 320), cmap='gray')
        ax.set_title(f"Eigenface {i+1}")
        ax.axis('off')

    plt.show()

@typechecked
def qd1_project(dataset:np.ndarray, pca:PCA) -> np.ndarray:
    """
    Return the projection of the dataset 
    NOTE: TO TEST CORRECTNESS, please submit to autograder
    """
    return pca.transform(dataset)

@typechecked
def qd2_reconstruct(projected_input:np.ndarray, pca:PCA) -> np.ndarray:
    """
    Return the reconstructed image given the pca components
    NOTE: TO TEST CORRECTNESS, please submit to autograder
    """
    return pca.inverse_transform(projected_input)  

@typechecked
def qd3_visualize(dataset:np.ndarray, pca:PCA, dim_x = 243, dim_y = 320):
    """
    Select a couple of images from the data. Use the first k eigenfaces as a
    basis to reconstruct the images (use functions written in previous sub-questions). Visualize the
    reconstructed images using 1, 10, 20, 30, 40, 50 components.
    """
    # Select a few random images from the dataset
    num_images = 4
    selected_indices = np.random.choice(range(len(dataset)), size=num_images, replace=False)
    selected_images = dataset[selected_indices]

    # Reconstruct the selected images using different number of components
    num_components_to_test = [1, 10, 20, 30, 40, 50]
    reconstructed_images = []
    for k in num_components_to_test:
        pca_k = PCA(n_components=k)
        pca_k.fit(dataset)
        # Project the images onto the first k eigenfaces
        projected_images = qd1_project(selected_images, pca_k)

        # Reconstruct the images from the projected data
        reconstructed = qd2_reconstruct(projected_images, pca_k)

        reconstructed_images.append(reconstructed)

    # Visualize the results
    fig, axs = plt.subplots(num_images, len(num_components_to_test) + 1, figsize=(20, 6*num_images))
    for i, image in enumerate(selected_images):
        # Plot the original image
        axs[i][0].imshow(image.reshape(dim_x, dim_y), cmap='gray')
        axs[i][0].set_title('Original')

        # Plot the reconstructed images using different number of components
        for j, reconstructed in enumerate(reconstructed_images):
            axs[i][j+1].imshow(reconstructed[i].reshape(dim_x, dim_y), cmap='gray')
            axs[i][j+1].set_title(f'{num_components_to_test[j]} components')
            
    plt.show()
    
@typechecked
def qe1_svm(trainX:np.ndarray, trainY:np.ndarray, pca:PCA) -> Tuple[int, float]:
    """
    Given the data, and PCA components. Select a subset of them in range [1,100]
    Project the dataset and train svm (with 5 fold cross validation) and return
    best_k, and test accuracy (averaged across fold).

    Hint: you can pick 5 `k' values uniformly distributed
    """
    
    # Define the range of k values to test
    k_values = np.arange(10, 101, 20)

    # Project the training set using PCA
    trainX_pca = qd1_project(trainX, pca)

    # Perform 5-fold cross-validation to find the best k value and SVM hyperparameters
    param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 0.01, 0.001]}
    clf = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
    clf.fit(trainX_pca, trainY)

    best_k = 0
    best_acc = 0.0

    for k in k_values:
        # Use the top k PCA components
        trainX_pca_k = trainX_pca[:, :k]

        # Train SVM with the best hyperparameters found from cross-validation
        clf_k = SVC(kernel='rbf', C=clf.best_params_['C'], gamma=clf.best_params_['gamma'])
        clf_k.fit(trainX_pca_k, trainY)
        predY = clf_k.predict(trainX_pca_k)
        acc_k = accuracy_score(trainY, np.round(predY))

        if acc_k > best_acc:
            best_k = k
            best_acc = acc_k

    return int(best_k), best_acc

@typechecked
def qe2_lasso(trainX:np.ndarray, trainY:np.ndarray, pca:PCA) -> Tuple[int, float]:
    """
    Given the data, and PCA components. Select a subset of them in range [1,100]
    Project the dataset and train lasso regression (with 5 fold stratified cross validation) and return
    best_k, and test accuracy (averaged across fold) in that order.

    Hint: you can pick 5 `k' values uniformly distributed
    """
    from sklearn.linear_model import LassoCV
    
    # Define the range of k values to test
    k_values = np.arange(10, 101, 20)

    # Project the training set using PCA
    trainX_pca = qd1_project(trainX, pca)

    # Perform 5-fold cross-validation to find the best k value and Lasso hyperparameters
    clf = LassoCV(cv=5)
    clf.fit(trainX_pca, trainY)

    best_k = 0
    best_acc = 0.0

    for k in k_values:
        # Use the top k PCA components
        trainX_pca_k = trainX_pca[:, :k]

        # Train Lasso with the best hyperparameters found from cross-validation
        clf_k = Lasso()
        clf_k.fit(trainX_pca_k, trainY)
        acc_k = clf_k.score(trainX_pca_k, trainY)

        if acc_k > best_acc:
            best_k = k
            best_acc = acc_k

    return int(best_k), best_acc


if __name__ == "__main__":

    faces, y_target = qa1_load("./data/")
    dataset = qa2_preprocess(faces)
    pca, eig_values, eig_vectors = qa3_calc_eig_val_vec(dataset, len(dataset))

    qb_plot_written(eig_values)

    num = len(dataset)
    org_dim_eig_faces = qc1_reshape_images(pca)
    qc2_plot(org_dim_eig_faces)

    qd3_visualize(dataset, pca)
    best_k, result = qe1_svm(dataset, y_target, pca)
    print(best_k, result)
    best_k, result = qe2_lasso(dataset, y_target, pca)
    print(best_k, result)
