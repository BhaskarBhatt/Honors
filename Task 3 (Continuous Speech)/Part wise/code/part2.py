import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn.metrics.pairwise import cosine_similarity

# the → /DH AH/
# money → /M AH N IY/

# Read audio file and find its sample rate
samplerate, data = wavfile.read('/home/bhaskar/Desktop/Honors/Task 3/Part wise/wav/Part 2.wav')

print("Sample rate:", samplerate)

# Compute total duration of audio clip
total_duration = len(data) / samplerate
print("Total duration:", total_duration, "seconds")

# Load the features
features = np.load('/home/bhaskar/Desktop/Honors/Task 3/Part wise/extracted_features/Part 2.npy') 

# If features have an extra dimension, squeeze it
features = np.squeeze(features)
print("Features shape:", features.shape)


# Compute cosine similarity matrix
result = cosine_similarity(features, features)
# result = np.log(result)
print("Cosine Similarity Matrix shape:", result.shape)

# # Visualize the full size x size matrix
plt.figure(figsize=(10, 10))
plt.imshow(result, cmap='viridis', aspect='auto')

plt.colorbar()
plt.title('Cosine Similarity Matrix')
plt.xlabel('Column Index')
plt.ylabel('Row Index')

plt.tight_layout()
plt.show()