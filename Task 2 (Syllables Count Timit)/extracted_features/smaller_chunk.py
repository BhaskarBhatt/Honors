import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn.metrics.pairwise import cosine_similarity
import math

# Read audio file and find its sample rate
samplerate, data = wavfile.read('/home/bhaskar/Desktop/Honors/Syllables Count Timit/wav/SI1612.wav')

print("Sample rate:", samplerate)

# Compute total duration of audio clip
total_duration = len(data) / samplerate
print("Total duration:", total_duration, "seconds")

# Load the features
features = np.load('SI1612.npy') 

# If features have an extra dimension, squeeze it
print("Features shape:", features.shape)
features = np.squeeze(features)
print("Features shape:", features.shape)

# Calculate frame shift (assuming it's 20ms for wav2vec 2.0)
frame_shift = 0.02  # 20ms frame shift
frames_per_second = 1 / frame_shift

# Read phoneme boundaries from the text file
phoneme_bounds = []
phoneme_labels = []
with open('/home/bhaskar/Desktop/Honors/Syllables Count Timit/TIMIT Samples/Phoneme/SI1612.PHN', 'r') as file:
    for line in file:
        start_sample, end_sample, phoneme = line.strip().split()
        start_sample = int(start_sample)
        end_sample = int(end_sample)
        
        # Calculate the start and end times in seconds
        start_time = start_sample / samplerate
        end_time = end_sample / samplerate
        
        # Calculate the corresponding frame indices
        start_frame = int(start_time * frames_per_second)
        end_frame = int(end_time * frames_per_second)
        
        # Add the end frame and phoneme to the lists
        phoneme_bounds.append((start_frame, end_frame))
        phoneme_labels.append(phoneme)


# Compute cosine similarity matrix
result = cosine_similarity(features, features)
# result = np.log(result)
print("Cosine Similarity Matrix shape:", result.shape)

# Visualize the full size x size matrix
plt.figure(figsize=(12, 12))

# Adjust the colormap for better visibility
plt.imshow(result, cmap='viridis', aspect='auto')

plt.colorbar()
plt.title('Cosine Similarity Matrix', fontsize=16)
# plt.xlabel('Column Index', fontsize=14)
# plt.ylabel('Row Index', fontsize=14)

# Add vertical and horizontal lines at specified indices with different color
for (start_frame, end_frame), phoneme in zip(phoneme_bounds, phoneme_labels):
    plt.axvline(x=start_frame, color='red', linestyle='--', linewidth=1)
    plt.axvline(x=end_frame, color='red', linestyle='--', linewidth=1)
    plt.axhline(y=start_frame, color='red', linestyle='--', linewidth=1)
    plt.axhline(y=end_frame, color='red', linestyle='--', linewidth=1)
    
    # Add phoneme labels to the plot with contrasting colors
    plt.text((start_frame + end_frame) / 2, -3, phoneme, ha='center', color='black', fontsize=10, weight='bold')  # Horizontal axis
    plt.text(-4, (start_frame + end_frame) / 2, phoneme, va='center', rotation=90, color='black', fontsize=10, weight='bold')  # Vertical axis

plt.tight_layout()
plt.show()


# /home/bhaskar/Desktop/Honors/Syllables Count Timit/TIMIT Samples/Phoneme/SI1612.PHN

# Subarray 1 for 1st frame to 53rd frame (i.e., 0th index to 52nd index)
# subarray1 = features[0:53, :]
# size_sub1 = subarray1.shape[0]

# # Compute cosine similarity matrix
# res_sub1 = cosine_similarity(subarray1, subarray1)
# print(res_sub1)
# # print(res_sub1.shape)


# # Invert the rows of the similarity matrix
# res_sub1_inv = res_sub1[::-1]

# # Original Indices where vertical lines should be drawn
# vertical_phoneme_bound = [8, 15, 18, 22, 26, 30, 34, 37, 40, 44, 49, 52]
# horizontal_phoneme_bound = [8, 15, 18, 22, 26, 30, 34, 37, 40, 44, 49, 52]

# # Indices where vertical lines should be drawn
# # vertical_phoneme_bound = [7, 14, 17, 21, 25, 29, 33, 36, 39, 43, 48, 51]
# # horizontal_phoneme_bound = [7, 14, 17, 21, 25, 29, 33, 36, 39, 43, 48, 51]

# # # Visualize the full size x size matrix
# plt.figure(figsize=(10, 10))
# plt.imshow(res_sub1, cmap='cividis', aspect='auto')

# plt.colorbar()
# plt.title('Cosine Similarity Matrix (Inverted Rows)')
# plt.xlabel('Column Index')
# plt.ylabel('Row Index')

# # Add vertical lines at specified indices
# for index in vertical_phoneme_bound:
#     plt.axvline(x=index, color='red', linestyle='--')
# # Add vertical lines at specified indices

# for index in horizontal_phoneme_bound:
#     plt.axhline(y=index, color='red', linestyle='--')

# plt.show()

