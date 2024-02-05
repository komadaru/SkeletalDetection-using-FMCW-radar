import os
import json
import numpy as np
from sklearn.cluster import DBSCAN

# Function to apply DBSCAN to PointCloud and save the result
def process_and_save(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    if data["NumPoints"] != 0:
        # Extract PointCloud data
        point_cloud = np.array(data["PointCloud"])
        x = point_cloud[0, :]
        y = point_cloud[1, :]
        z = point_cloud[2, :]
        v = point_cloud[3, :]
        snr = point_cloud[4, :]

        # DBSCAN configuration
        eps = 400.0  # Epsilon (threshold distance for neighborhood)
        min_samples = 5  # Minimum number of samples to recognize as a cluster
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)

        # Apply DBSCAN to x, y, z
        labels = dbscan.fit_predict(np.column_stack((x, y, z)))

        # Get indices of non-noise points
        non_noise_indices = np.where(labels != -1)[0]

        # Get coordinates of non-noise points
        cleaned_data = point_cloud[:, non_noise_indices]

        # Update the original data with cleaned PointCloud
        data["PointCloud"] = cleaned_data.tolist()

        # Remove v values for noise points
        data["PointCloud"][3] = [v[i] for i in non_noise_indices]

        # Remove snr values for noise points
        data["PointCloud"][4] = [snr[i] for i in non_noise_indices]

        # Update numPoints
        data["NumPoints"] = len(non_noise_indices)

    # Save the updated data to a new JSON file
    output_path = os.path.join(output_folder, os.path.basename(file_path))
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(data, output_file)

# Folder paths
input_folder = "C:/Users/kodam/PycharmProjects/mmModels/Datasets/OneRadar3/"
output_folder = "C:/Users/kodam/PycharmProjects/mmModels/Datasets/ScannedOneRadar3/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# Iterate over each file in the input folder in sorted order
for filename in sorted(os.listdir(input_folder)):
    if filename.endswith(".json"):
        file_path = os.path.join(input_folder, filename)
        print(file_path)
        process_and_save(file_path)
