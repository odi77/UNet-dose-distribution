import torch
import os
import skimage.io as io
import numpy as np

# Predict 3 channels

def load_image(data_folder, x_suffix):
    if x_suffix == "annotated_ct":
        x_filename = f"{x_suffix}.npy"
    else:
        x_filename = f"{x_suffix}.mhd"
    
    x_path = os.path.join(data_folder, x_filename)

    if x_suffix == "annotated_ct":
        return np.load(x_path)
    else:
        return io.imread(x_path, plugin='simpleitk')

def load_X_data(data_folder):
    X_item = []
    for x_suffix in ["low_edep", "ct", "annotated_ct"]:
        x_image = load_image(data_folder, x_suffix)
        X_item.append(x_image)
    
    X_item = np.array(X_item)[np.newaxis, ...]
    
    return X_item

def predict3(model, data_folder, device):
    model.eval()

    X_item = load_X_data(data_folder)

    with torch.no_grad():
        input_tensor = torch.from_numpy(X_item).to(dtype=torch.float32).to(device)
        prediction = model(input_tensor)
    
    return prediction.squeeze().cpu().numpy()

# predict with 2 channels
# Predict 2 images .mhd

def predict2(model, *x_paths, device):
    model.eval()
    
    x_images = [io.imread(x_path, plugin='simpleitk') for x_path in x_paths]
    x_images = [np.array(x_image) for x_image in x_images]
    x_images = np.array(x_images)

    with torch.no_grad():
        input_tensor = torch.from_numpy(x_images).to(dtype=torch.float32).to(device)
        prediction = model(input_tensor)
    return prediction.squeeze().cpu().numpy()


# 1 channel

def predict1(model, x_path, device):
    model.eval()
    x_image = io.imread(x_path, plugin='simpleitk')
    x_image = np.array(x_image)
    with torch.no_grad():
        input_tensor = torch.from_numpy(x_image).to(dtype=torch.float32).to(device)
        # If the model expects a batch dimension, add it here:
        input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        prediction = model(input_tensor)
    return prediction.squeeze().cpu().numpy()
