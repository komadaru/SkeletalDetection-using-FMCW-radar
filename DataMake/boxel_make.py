import os
import json
import numpy as np

def filter_and_update_points(data, voxel_size=(1.7, 1.4, 2.2)):
    # Extract coordinates and values
    x, y, z, v, snr = np.array(data["PointCloud"][0]), np.array(data["PointCloud"][1]), np.array(data["PointCloud"][2]), np.array(data["PointCloud"][3]), np.array(data["PointCloud"][4])

    # Define the voxel boundaries
    x_min, x_max = -1000, 1000
    y_min, y_max = 1500, 3900
    z_min, z_max = 0, 2400

    # Update voxel center (centroid) based on filtered points
    centroid = np.array([np.mean(x), np.mean(y), np.mean(z)])

    # Filter points based on voxel boundaries
    mask = ((x >= x_min) & (x <= x_max) &
            (y >= y_min) & (y <= y_max) &
            (z >= z_min) & (z <= z_max))

    filtered_x = x[mask]
    filtered_y = y[mask]
    filtered_z = z[mask]
    filtered_snr = snr[mask]

    # Normalize data to 37.5*2 cells in xy plane
    x_normalized = ((filtered_x - x_min) / (x_max - x_min) * 2 * 37.5).astype(int)
    y_normalized = ((filtered_y - y_min) / (y_max - y_min) * 2 * 37.5).astype(int)
    z_normalized = ((filtered_z - z_min) / (z_max - z_min) * 2 * 37.5).astype(int)

    # Create matrix for xy plane
    xy_matrix = np.zeros((27, 75))

    # Update matrix with normalized snr values
    xy_matrix[x_normalized, y_normalized] = filtered_snr

    # Create matrix for xz plane
    xz_matrix = np.zeros((37, 75))

    # Update matrix with normalized snr values
    xz_matrix[x_normalized, z_normalized] = filtered_snr

    return [xy_matrix.tolist(), xz_matrix.tolist()], centroid

def process_json_file(input_path, output_path):
    with open(input_path, 'r') as file:
        data = json.load(file)

    point_cloud, centroid = filter_and_update_points(data)

    # Update PointCloud in the data dictionary
    data["PointCloud"] = point_cloud

    with open(output_path, 'w') as file:
        json.dump(data, file, indent=2)

    return centroid

def process_folder(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Process each JSON file
            centroid = process_json_file(input_path, output_path)

            print(f"Processed: {filename}, Centroid: {centroid}")

# Example usage:
input_folder_path = "C:/Users/kodam/PycharmProjects/mmModels/Datasets/ScannedCombinedTwoRadars_tilt"
output_folder_path = "C:/Users/kodam/PycharmProjects/mmModels/Datasets/BoxScannedCombinedTwoRadars_tilt"
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

process_folder(input_folder_path, output_folder_path)
