import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from sklearn.metrics.pairwise import cosine_similarity


# to read audio file and find its sample-rate
samplerate, data = wavfile.read('/home/bhaskar/Desktop/Honors/Syllables Count/wav/100-121669-0002.wav')
# to compute total duration of audio clip
print(len(data)/samplerate)

# Load the features
features = np.load('100-121669-0002.npy')  # Change this path to your actual file path

# If features have an extra dimension, squeeze it
features = np.squeeze(features)

# Now features should be 2D
# print(features.shape)

# Transposed array
transposed_feature = features.T

# Get the size from the second dimension of the transposed_feature
size = transposed_feature.shape[1]

print(size)

result = np.empty((size, size))

result = cosine_similarity(features,features)

# Step 3: Normalize the vectors to unit vectors
# norms = np.linalg.norm(transposed_feature, axis=0)
# normalized_vectors = transposed_feature / norms

# # Step 4: Compute the cosine similarity for each pair of vectors
# for i in range(size):
#     for j in range(size):
#         result[i, j] = np.dot(normalized_vectors[:, i], normalized_vectors[:, j])


# print(result.shape)
# Now 'result' is a sizexsize ndarray with each element being the cosine similarity between the corresponding pair of vectors


# finding similarity of individual frames with other
subarray = result[8, 8:17]
# print(subarray)

# traversal between row
# Example result array for demonstration purposes

left_frame = 15    # index of left_frame
right_frame = 23  # index of right_frame
mid = int((left_frame + right_frame) / 2 + 1)

data_dict = {}
for i in range(left_frame, right_frame+1):
    data_dict[i] = {}
    for j in range(i, right_frame + 1):
        print(i, j, result[i, j])
        data_dict[i][j] = result[i, j]

# Prepare data for grouped bar plot
rows = list(data_dict.keys())
columns = sorted({col for subdict in data_dict.values() for col in subdict.keys()})
values = np.zeros((len(rows), len(columns)))

for i, row in enumerate(rows):
    for j, col in enumerate(columns):
        if col in data_dict[row]:
            values[i, j] = data_dict[row][col]

# Plotting the grouped bar plot
fig, ax = plt.subplots(figsize=(14, 8))
bar_width = 0.1
group_spacing = 0.4  # Increase this value to increase spacing between groups
indices = np.arange(len(rows)) * (bar_width * len(columns) + group_spacing)

for j, col in enumerate(columns):
    bar_positions = indices + j * bar_width
    ax.bar(bar_positions, values[:, j], bar_width, label=f'Column {col}')

ax.set_xlabel('Rows')
ax.set_ylabel('Values')
ax.set_title('Grouped Bar Plot of Dictionary Values')
ax.set_xticks(indices + bar_width * (len(columns) - 1) / 2)
ax.set_xticklabels(rows)
ax.legend()

plt.show()










result_inverted = result[::-1]
# # print(result)

# col_index = 0
# row_index = 0
# dict = {}
# while (col_index<size):
#     dict[col_index] = -1
#     while (row_index<size):
#         if (result[row_index,col_index]<0.8):
#             break
#         dict[col_index] = row_index
#         row_index+=1
#     col_index+=1

# # print(dict)
# # Prepare data for plotting

# x_values = list(dict.keys())
# y_values = list(dict.values())

# Plot the data
# plt.figure(figsize=(10, 6))
# plt.scatter(x_values, y_values, color='green')
# plt.xlabel('Column Index')
# plt.ylabel('Row Index')
# plt.title('Row Index Where Cosine Similarity Falls Below 0.8')
# plt.grid(True)
# plt.show()


# Visualize the full size x size matrix
plt.figure(figsize=(10, 10))
plt.imshow(result_inverted, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Cosine Similarity Matrix (Inverted Rows)')
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.show()



# # Visualize the subarray within the boundaries (10, 10) to (35, 35)
# subarray1 = result[17:30, 17:30]

# plt.figure(figsize=(8, 8))
# plt.imshow(subarray1, cmap='viridis', aspect='auto')
# plt.colorbar()
# plt.title('Cosine Similarity Subarray (17,17) to (29,29)')
# plt.xlabel('Column Index (17 to 29)')
# plt.ylabel('Row Index (17 to 29)')
# plt.xticks(ticks=np.arange(13), labels=np.arange(17, 30))
# plt.yticks(ticks=np.arange(13), labels=np.arange(17, 30))
# plt.show()


# # Visualize the subarray within the boundaries (29,29) to (52, 52)
# 

# plt.figure(figsize=(8, 8))
# plt.imshow(subarray2, cmap='viridis', aspect='auto')
# plt.colorbar()
# plt.title('Cosine Similarity Subarray (29,29) to (52,52)')
# plt.xlabel('Column Index (29 to 52)')
# plt.ylabel('Row Index (29 to 52)')
# plt.xticks(ticks=np.arange(24), labels=np.arange(29, 53))
# plt.yticks(ticks=np.arange(24), labels=np.arange(29, 53))
# plt.show()


# # Visualize the subarray within the boundaries (52,52) to (70, 70)
# subarray3 = result[52:71, 52:71]

# plt.figure(figsize=(8, 8))
# plt.imshow(subarray3, cmap='viridis', aspect='auto')
# plt.colorbar()
# plt.title('Cosine Similarity Subarray (52,52) to (70,70)')
# plt.xlabel('Column Index (52 to 70)')
# plt.ylabel('Row Index (52 to 70)')
# plt.xticks(ticks=np.arange(19), labels=np.arange(52, 71))
# plt.yticks(ticks=np.arange(19), labels=np.arange(52, 71))
# plt.show()