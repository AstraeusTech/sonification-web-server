import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.io import wavfile as wav
from dataclasses import dataclass
import io


@dataclass
class sonificationConfig:
    amplitude: str  # recommended 4096
    guassian_sigma: int  # recommended 8
    sonification_duration: int  # recommended 30
    output_path: str  # recommend "soundfiles/sound-combined.wav"


# Set a threshold for outlier detection (you can adjust this threshold as needed)
def calculate_threshold(data, z_score_threshold=3):
    mean = np.mean(data)
    std_dev = np.std(data)
    threshold = mean + z_score_threshold * std_dev
    return threshold

# Define a function to scale an array to correct frequency range
def map_array_to_scale(array, scale):
    new_min = np.nanmin(scale)
    new_max = np.nanmax(scale)

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

    # Coerce the array to match the scale
    coerced_values = []

    for value in scaled_array:
        if value != np.nan:
            closest_note = min(scale, key=lambda x: abs(x - value))
        else:
            closest_note = 0
        coerced_values.append(closest_note)

    return coerced_values


# Define a function to calculate the average grayscale value of an array
def average_grayscale(arr):
    return np.mean(arr)


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
    # image = cv2.imread(image_filepath)
    image_np_array = np.frombuffer(image_filepath, np.uint8)
    image = cv2.imdecode(image_np_array, cv2.IMREAD_COLOR)

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
        # Identify outliers with grayscale higher than 200
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

    # -- Create the list of musical notes

    scale = []
    for k in range(35, 65):
        note = 440 * 2 ** ((k - 49) / 12)
        if k % 12 != 0 and k % 12 != 2 and k % 12 != 5 and k % 12 != 7 and k % 12 != 10:
            scale.append(note)  # add musical note (skip half tones)
    n_notes = len(scale)  # number of musical notes

    # Sort the scale in ascending order
    scale.sort()

    # Calculate the length of each sub-array
    total_notes = len(scale)
    subarray_length = total_notes // 3

    # Calculate the indices to split the scale
    split_index1 = subarray_length
    split_index2 = 2 * subarray_length

    # Split the scale into three arrays
    scale_low = scale[:split_index1]
    scale_mid = scale[split_index1:split_index2]
    scale_high = scale[split_index2:]

    # If needed, remove the lowest and highest values
    if len(scale_low) > len(scale_high):
        scale_low.pop(0)
    elif len(scale_high) > len(scale_low):
        scale_high.pop()

    low_map = map_array_to_scale(smoother_outlier_group_avgs, scale_low)
    wave_low = generate_wave(low_map, note_duration, config.amplitude)
    # wav.write("soundfiles/low.wav", rate=44100, data=wave_low.astype(np.int16))

    mid_map = map_array_to_scale(other_outlier_avgs, scale_mid)
    wave_mid = generate_wave(mid_map, note_duration, config.amplitude)
    # wav.write("soundfiles/mid.wav", rate=44100, data=wave_mid.astype(np.int16))

    high_map = map_array_to_scale(largest_outlier_avgs, scale_high)
    wave_high = generate_wave(high_map, note_duration, config.amplitude)
    # wav.write("soundfiles/high.wav", rate=44100, data=wave_high.astype(np.int16))

    # Combine all three arrays
    # Assuming wave_low, wave_mid, and wave_high have the same length
    combined_wave = np.vstack((wave_low, wave_mid, wave_high))

    # Transpose the combined_wave to make it a 2D array
    combined_wave = combined_wave.T

    # Define the output filename for the combined sound
    output_filename = config.output_path

    wav_data_bytes = io.BytesIO()

    # Write the combined_wave to a WAV file
    wav.write(wav_data_bytes, rate=44100, data=combined_wave.astype(np.int16))
    wav_data_bytes.seek(0)

    print(f"Combined sound written to {output_filename}")

    return wav_data_bytes
