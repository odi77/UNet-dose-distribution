import os
import torch
import SimpleITK as sitk
import skimage.io as io
import numpy as np


# 1 channel in

def load_XY_1chanel(data_folder):
    identifiers = set()

    # Collect all identifiers present in the dataset
    for folder in os.listdir(data_folder):
        if folder.startswith('sample_') and os.path.isdir(os.path.join(data_folder, folder)):
            identifier = folder.split('_')[-1]
            identifiers.add(identifier)

    # Convert to a sorted list for consistent order
    identifiers = sorted(list(identifiers))

    X_data = []
    y_data = []

    for identifier in identifiers:
        # Load X data
        X_item = []
        x_filename = f"low_edep.mhd"       
        x_path = os.path.join(data_folder, f"sample_{identifier}", x_filename)
        x_image = io.imread(x_path, plugin='simpleitk')

        X_item.append(x_image)

        # Load y data
        y_filename = f"high_edep.mhd"
        y_path = os.path.join(data_folder, f"sample_{identifier}", y_filename)
        y_image = io.imread(y_path, plugin='simpleitk')

        X_data.append(X_item)
        y_data.append(y_image)
        
    # Convert lists to NumPy arrays
    X_data = np.array(X_data)
    y_data = np.array(y_data)

    return X_data, y_data


# 2 channels in

def load_XY_2chanel(data_folder):
    identifiers = set()

    # Collect all identifiers present in the dataset
    for folder in os.listdir(data_folder):
        if folder.startswith('sample_') and os.path.isdir(os.path.join(data_folder, folder)):
            identifier = folder.split('_')[-1]
            identifiers.add(identifier)

    # Convert to a sorted list for consistent order
    identifiers = sorted(list(identifiers))

    X_data = []
    y_data = []

    for identifier in identifiers:
        # Load X data
        X_item = []
        for x_suffix in ["low_edep", "ct"]:

            x_filename = f"{x_suffix}.mhd"
            
            x_path = os.path.join(data_folder, f"sample_{identifier}", x_filename)
        
            x_image = io.imread(x_path, plugin='simpleitk')

            X_item.append(x_image)

        # Load y data
        y_filename = f"high_edep.mhd"
        y_path = os.path.join(data_folder, f"sample_{identifier}", y_filename)
        y_image = io.imread(y_path, plugin='simpleitk')

        X_data.append(X_item)
        y_data.append(y_image)

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    return X_data, y_data


# 3 channels in

def load_XY_3chanel(data_folder):
    identifiers = set()

    # Collect all identifiers present in the dataset
    for folder in os.listdir(data_folder):
        if folder.startswith('sample_') and os.path.isdir(os.path.join(data_folder, folder)):
            identifier = folder.split('_')[-1]
            identifiers.add(identifier)

    # Convert to a sorted list for consistent order
    identifiers = sorted(list(identifiers))

    X_data = []
    y_data = []

    for identifier in identifiers:
        # Load X data
        X_item = []
        for x_suffix in ["low_edep", "ct", "annotated_ct"]:
            if x_suffix == "annotated_ct":
                x_filename = f"{x_suffix}.npy"
            else:
                x_filename = f"{x_suffix}.mhd"
            
            x_path = os.path.join(data_folder, f"sample_{identifier}", x_filename)
        

            if x_suffix == "annotated_ct":
                x_image = np.load(x_path)
            else:
                x_image = io.imread(x_path, plugin='simpleitk')

            X_item.append(x_image)

        # Load y data
        y_filename = f"high_edep.mhd"
        y_path = os.path.join(data_folder, f"sample_{identifier}", y_filename)
        y_image = io.imread(y_path, plugin='simpleitk')

        X_data.append(X_item)
        y_data.append(y_image)

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    return X_data, y_data