import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn.metrics.pairwise import cosine_similarity
import os

# concatenated audio sequence: ia_ih_ow_oy_uh_uw

# Input paths for audio files and feature files
audio_files_path = '/home/bhaskar/Desktop/Honors/Task 5 (Concatenated Isolated Vowels)/isolated_wav/'
feature_files_path = '/home/bhaskar/Desktop/Honors/Task 5 (Concatenated Isolated Vowels)/extracted_features/'

# List of audio file names and feature file names (assuming both have the same base name with different extensions)
audio_files = ['ia.wav', 'ih.wav', 'ow.wav', 'oy.wav', 'uh.wav', 'uw.wav']  # Replace with actual file names if different
feature_files = ['ia.npy', 'ih.npy', 'ow.npy', 'oy.npy', 'uh.npy', 'uw.npy']

all_features = []
feature_labels = []

# Loop over each file, load and concatenate features
for i, (audio_file, feature_file) in enumerate(zip(audio_files, feature_files), start=1):
    # Read audio file to get sample rate and duration (optional)
    samplerate, data = wavfile.read(os.path.join(audio_files_path, audio_file))
    print(f"Sample rate for {audio_file}:", samplerate)
    total_duration = len(data) / samplerate
    print(f"Total duration for {audio_file}:", total_duration, "seconds")
    
    # Load the features
    features = np.load(os.path.join(feature_files_path, feature_file))
    
    # If features have an extra dimension, squeeze it
    features = np.squeeze(features)
    print(f"Features shape for {feature_file}:", features.shape)
    
    # Append the features and labels
    all_features.append(features)
    feature_labels.extend([i] * features.shape[0])  # Assign label i to each feature of this file

# Concatenate all features and labels
all_features = np.vstack(all_features)
feature_labels = np.array(feature_labels)

print("Concatenated features shape:", all_features.shape)
print("Feature labels shape:", feature_labels.shape)

# Compute cosine similarity matrix
result = cosine_similarity(all_features, all_features)
print("Cosine Similarity Matrix shape:", result.shape)

# Visualize the cosine similarity matrix
plt.figure(figsize=(10, 10))
plt.imshow(result, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Cosine Similarity Matrix')
plt.xlabel('Feature Index')
plt.ylabel('Feature Index')

plt.tight_layout()
plt.show()
