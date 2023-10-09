import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.io import wavfile as wav
from dataclasses import dataclass


@dataclass
class sonificationConfig:
    amplitude: str  # recommended 4096
    guassian_sigma: int  # recommended 8
    sonification_duration: int  # recommended 30
    output_path: str  # recommend "soundfiles/sound-combined.wav"


# Define a function to calculate the average grayscale value of an array
def average_grayscale(arr):
    return np.mean(arr)

# Set a threshold for outlier detection (you can adjust this threshold as needed)
def calculate_threshold(data, z_score_threshold=3):
    mean = np.mean(data)
    std_dev = np.std(data)
    threshold = mean + z_score_threshold * std_dev
    return threshold

# Define a function to identify outliers in a grayscale image
def find_outliers(arr, threshold):
    return arr[arr > threshold]


# Define a function to get a sound wave from passed data
def get_sine_wave(
    frequency, duration, sample_rate=44100, amplitude=4096, ramp_down_duration=0.01
):
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave = amplitude * np.sin(2 * np.pi * frequency * t)

    # Calculate the number of samples for the ramp-down segment
    ramp_down_samples = int(sample_rate * ramp_down_duration)

    # Create a linear ramp-down segment
    ramp_down = np.linspace(1, 0, ramp_down_samples)

    # Concatenate the ramp-down with the main sine wave
    wave[-ramp_down_samples:] *= ramp_down

    return wave


# Define a function to scale an array to correct frequency range
def scale_array(array, new_min=200, new_max=400):
    # Find the minimum and maximum values of the array while ignoring NaN values
    old_min = np.nanmin(array)
    old_max = np.nanmax(array)

    # Check if old_min and old_max are equal, which can happen if all elements are NaN
    if old_min == old_max:
        return array  # Return the input array as it is

    # Apply the scaling
    scaled_array = (array - old_min) * (new_max - new_min) / (
        old_max - old_min
    ) + new_min

    return scaled_array


# Define function to create a wave for each value in array
def generate_wave(arr, duration, amplitude):
    wave = []
    for t in range(
        len(arr)
    ):  # loop over dataset observations, create one note per observation
        if np.isnan(arr[t]):
            new_wave = get_sine_wave(
                frequency=0, duration=duration, amplitude=amplitude
            )
        else:
            new_wave = get_sine_wave(
                frequency=arr[t], duration=duration, amplitude=amplitude
            )
        wave = np.concatenate((wave, new_wave))

    return wave


def runSonification(image_filepath: str, config: sonificationConfig):
    # Load the image using OpenCV
    image = cv2.imread(image_filepath)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get the dimensions of the grayscale image
    height, width = gray_image.shape

    # Initialize lists to store recorded values
    x_values = list(range(width))
    avg_values = []
    outlier_group_avgs = []
    largest_outlier_avgs = []
    other_outlier_avgs = []

    # Iterate through each column (x) of the grayscale image
    for x in x_values:
        column = gray_image[:, x]  # Get the current column

        # Calculate the average grayscale value of the column
        avg_value = average_grayscale(column)
        avg_values.append(avg_value)

        threshold = calculate_threshold(gray_image)

        # Identify outliers with grayscale threshold
        outliers = find_outliers(column, threshold)

        if len(outliers) == 0:
            # No outliers, record the average grayscale value
            outlier_group_avgs.append(avg_value)
            largest_outlier_avgs.append(np.nan)
            other_outlier_avgs.append(np.nan)
        else:
            # Sort the outliers by their grayscale values
            outliers.sort()

            if len(outliers) == 1:
                # Only one outlier group, record values accordingly
                outlier_group_avgs.append(
                    np.mean(column[column != outliers[0]])
                )  # Calculate average without outlier
                largest_outlier_avgs.append(outliers[0])
                other_outlier_avgs.append(np.nan)
            else:
                # Multiple outlier groups, record values accordingly
                largest_outlier = outliers[-1]
                other_outliers = outliers[:-1]
                other_outlier_avg = average_grayscale(other_outliers)

                # Calculate average without all outliers
                column_no_outliers = column[~np.isin(column, outliers)]
                outlier_group_avgs.append(average_grayscale(column_no_outliers))
                largest_outlier_avgs.append(largest_outlier)
                other_outlier_avgs.append(other_outlier_avg)

    # Smooth the avg_values array using Gaussian filter
    smoothed_avg_values = gaussian_filter(avg_values, sigma=config.guassian_sigma)
    smoothed_largest_outlier_avgs = gaussian_filter(
        largest_outlier_avgs, sigma=config.guassian_sigma
    )
    smoother_outlier_group_avgs = gaussian_filter(
        outlier_group_avgs, sigma=config.guassian_sigma
    )
    smoothed_other_outlier_avgs = gaussian_filter(
        other_outlier_avgs, sigma=config.guassian_sigma
    )

    # Add the smoothed largest outlier values back to the largest outlier array, replacing non-NaN values
    for i in range(len(largest_outlier_avgs)):
        if not np.isnan(smoothed_largest_outlier_avgs[i]):
            largest_outlier_avgs[i] = smoothed_largest_outlier_avgs[i]
    # Add the smoothed Average group outlier values back to the group outlier array, replacing non Nan Values
    for i in range(len(other_outlier_avgs)):
        if not np.isnan(smoothed_other_outlier_avgs[i]):
            other_outlier_avgs[i] = smoothed_other_outlier_avgs[i]

    # 60 second sound file
    note_duration = config.sonification_duration / len(smoother_outlier_group_avgs)

    # Base - Average grayscale without outliers
    scaled_avg_base = scale_array(smoother_outlier_group_avgs)

    wave_low = generate_wave(scaled_avg_base, note_duration, config.amplitude)

    # Mid - Largest Grayscale outlier
    scaled_avg_mid = scale_array(other_outlier_avgs, 400, 800)

    wave_mid = generate_wave(scaled_avg_mid, note_duration, config.amplitude)

    # High - Largest Grayscale outlier
    scaled_avg_high = scale_array(largest_outlier_avgs, 800, 1200)

    wave_high = generate_wave(scaled_avg_high, note_duration, config.amplitude)

    # Combine all three arrays
    combined_wave = np.vstack((wave_low, wave_mid, wave_high))

    # Transpose the combined_wave to make it a 2D array
    combined_wave = combined_wave.T

    # Write the combined_wave to a WAV file
    wav.write(config.output_path, rate=44100, data=combined_wave.astype(np.int16))

    print(f"Combined sound written to {config.output_path}")
