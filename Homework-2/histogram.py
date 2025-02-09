import numpy as np
import matplotlib.pyplot as plt

# Generate example data
data = np.random.randn(1000)  # 1000 random numbers from a normal distribution

# Step 1: Define number of bins and find min/max
num_bins = 4 
data_min, data_max = min(data), max(data)

# Step 2: Compute bin width
bin_width = (data_max - data_min) / num_bins

# Step 3: Compute bin edges
bin_edges = [data_min + i * bin_width for i in range(num_bins + 1)]

# Step 4: Count occurrences in each bin
counts = [0] * num_bins

for value in data:
    for i in range(num_bins):
        if bin_edges[i] <= value < bin_edges[i + 1]:  # Last bin is handled separately
            counts[i] += 1
            break
# Ensure the max value falls in the last bin
counts[-1] += sum(1 for x in data if x == data_max)

# Print results
# print("Bin edges:", bin_edges)
# print("Counts per bin:", counts)
data = np.array([3.840381,
3.564491,
3.165412,
5.357139,
4.382139,
1.483324,
6.132465,
9.730042,
5.638801,
4.865761])


counts, bin_edges = np.histogram(data, 6)
print("Bin edges:", bin_edges)
print("Counts per bin:", counts)
# Plot histogram manually
# plt.bar(bin_edges[:-1], counts, width=bin_width, edgecolor='black', align='edge')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Manually Computed Histogram')
# plt.show()
