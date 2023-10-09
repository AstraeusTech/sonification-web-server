import cv2
import numpy as np
import open3d as o3d
import io

def corrected_colors(colors):
    # Check if the input is not empty
    if not colors:
        return []

    # Iterate through the input array and swap the values
    swapped_colors = [[color[2], color[1], color[0]] for color in colors]

    return swapped_colors


def create_point_cloud(image_path, output_path):
    # Load the image using OpenCV
    # image = cv2.imread(image_path)
    image_np_array = np.frombuffer(image_path, np.uint8)
    image = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Get the dimensions of the grayscale image
    height, width = gray_image.shape

    # Create a list to store the point cloud data
    point_cloud_data = []
    colors = []

    # Generate the point cloud data
    for y in range(height):
        for x in range(width):
            z = gray_image[y, x]  # Grayscale value as the z-coordinate
            point_cloud_data.append([x, y, z])
            colors.append(image[y, x].tolist())

    # Create a PLY point cloud
    pcd = o3d.geometry.PointCloud()

    # Set X, Y, and Z data using the point_cloud_data list
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data)

    correct_color = corrected_colors(colors)
    # Set RGB data using the colors list
    pcd.colors = o3d.utility.Vector3dVector(np.float32(correct_color) / 255)

    # Define the output PLY file path
    output_ply_file = output_path

    # Save the PLY point cloud to a file
    o3d.io.write_point_cloud(output_ply_file, pcd)

    print(f"PLY point cloud saved to {output_ply_file}")

    # Display the point cloud in Open3D viewer
    # o3d.visualization.draw_geometries([pcd])
