import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import ListedColormap
import skimage.io as io
import SimpleITK


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.show()


def visualize_prediction(input_path_ls, input_path_hs, predicted_data, output_path, input_path_extra1=None, input_path_extra2=None, label_extra1='CT', label_extra2='Quantified image'):
    # Load input data
    LS = io.imread(input_path_ls, plugin='simpleitk')
    HS = io.imread(input_path_hs, plugin='simpleitk')

    # Load extra images if provided
    if input_path_extra1:
        extra1 = io.imread(input_path_extra1, plugin='simpleitk')
    else:
        extra1 = None

    if input_path_extra2:
        extra2 = np.load(input_path_extra2)
    else:
        extra2 = None

    # Define a custom colormap
    top = cm.get_cmap('viridis', 64)
    bottom = cm.get_cmap('plasma', 960)
    newcolors = np.vstack((top(np.linspace(0, 1, 64)), bottom(np.linspace(1, 0, 960))))
    newcmp = ListedColormap(newcolors, name='MonteCarlo')

    # Create the visualization figure
    fig1 = plt.figure(constrained_layout=True, figsize=(12, 3))  # Adjusted figure size
    spec = gridspec.GridSpec(ncols=5, nrows=1, figure=fig1)  # Adjusted ncols

    # Plot Low Sampling
    ax1 = fig1.add_subplot(spec[0])
    plt.imshow(LS, interpolation=None, cmap=newcmp)
    plt.ylabel('Une coupe du bassin', fontweight='bold')
    plt.xlabel('Low Sampling', fontweight='bold')
    plt.xticks([], [])
    plt.yticks([], [])

    # Plot High Sampling
    ax1b = fig1.add_subplot(spec[1])
    plt.imshow(HS, interpolation=None, cmap=newcmp)
    plt.xlabel('High Sampling', fontweight='bold')
    plt.xticks([], [])
    plt.yticks([], [])

    # Plot Extra Image 1
    if extra1 is not None:
        ax_extra1 = fig1.add_subplot(spec[2])
        plt.imshow(extra1, interpolation=None, cmap='gray')
        plt.xlabel(label_extra1, fontweight='bold')
        plt.xticks([], [])
        plt.yticks([], [])

    # Plot Extra Image 2
    if extra2 is not None:
        ax_extra2 = fig1.add_subplot(spec[3])
        plt.imshow(extra2, interpolation=None, cmap='gray')
        plt.xlabel(label_extra2, fontweight='bold')
        plt.xticks([], [])
        plt.yticks([], [])

    # Plot Prediction
    ax2 = fig1.add_subplot(spec[-1])
    plt.imshow(predicted_data, interpolation=None, cmap=newcmp)
    plt.xlabel('Prediction', fontweight='bold')
    plt.xticks([], [])
    plt.yticks([], [])

    # Save the plot as an image file
    plt.savefig(output_path)
    
    # Show the plot
    plt.show()



def visualize_prediction_1c(input_path_ls, input_path_hs, predicted_data):
    # Load input data
    LS = io.imread(input_path_ls, plugin='simpleitk')
    HS = io.imread(input_path_hs, plugin='simpleitk')

    # Define a custom colormap
    top = cm.get_cmap('viridis', 64)
    bottom = cm.get_cmap('plasma', 960)
    newcolors = np.vstack((top(np.linspace(0, 1, 64)), bottom(np.linspace(1, 0, 960))))
    newcmp = ListedColormap(newcolors, name='MonteCarlo')

    # Create the visualization figure
    fig1 = plt.figure(constrained_layout=True, figsize=(9, 3))
    spec = gridspec.GridSpec(ncols=3, nrows=1, figure=fig1)

    # Plot Low Sampling
    ax1 = fig1.add_subplot(spec[0])
    plt.imshow(LS, interpolation=None, cmap=newcmp)
    plt.ylabel('Une coupe du bassin', fontweight='bold')
    plt.xlabel('Low Sampling', fontweight='bold')
    plt.xticks([], [])
    plt.yticks([], [])

    # Plot High Sampling
    ax1b = fig1.add_subplot(spec[1])
    plt.imshow(HS, interpolation=None, cmap=newcmp)
    plt.xlabel('High Sampling', fontweight='bold')
    plt.xticks([], [])
    plt.yticks([], [])
    
    # Plot Predicted Variable
    ax1c = fig1.add_subplot(spec[2])
    plt.imshow(predicted_data, interpolation=None, cmap=newcmp)
    plt.xlabel('Predicted Variable', fontweight='bold')
    plt.xticks([], [])
    plt.yticks([], [])



