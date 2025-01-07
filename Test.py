import matplotlib.pyplot as plt
import torch

def plot_distribution(tensors):
    plt.figure(figsize=(8, 6))
        
    # List to collect all the flattened tensor values
    all_values = []
    
    # Flatten and collect all tensors
    for tensor in tensors:
        all_values.append(tensor.flatten().cpu().numpy())  # Move to CPU and convert to numpy

    # Concatenate all values into a single array
    all_values = torch.cat([torch.tensor(val) for val in all_values], dim=0)
    
    # # Plot the histogram
    # plt.hist(all_values.numpy(), bins=50, alpha=0.75, color='blue', edgecolor='black')
    # plt.title('Distribution of Weights in Tensors')
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.show()

    unique_values, counts = torch.unique(all_values, return_counts=True)

    print("Unique Values and Their Counts:")
    for value, count in zip(unique_values, counts):
        print(f"Value: {value.item()}, Count: {count.item()}")
    
