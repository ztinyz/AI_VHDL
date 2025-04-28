import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Define the original model architecture
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        
        # Input: 14x14 → Conv → 14x14 → Pool → 7x7 → Conv → 7x7 → Pool → 3x3
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create model instance
model = DigitClassifier()

# Load the state dictionary
state_dict = torch.load("digit_classifier.pth")
model.load_state_dict(state_dict)

# Set model to evaluation mode
model.eval()

print("=== Model Architecture ===")
print(model)

print("\n=== Parameter Statistics ===")
total_params = 0
for name, param in model.named_parameters():
    print(f"{name}: {param.size()}, {param.numel()} parameters")
    total_params += param.numel()
print(f"Total parameters: {total_params:,}")

# Visualize convolutional filters
def visualize_conv_filters(model, layer_name, figsize=(12, 8)):
    # Get the layer
    layer = dict([*model.named_modules()])[layer_name]
    
    # Get weights
    weights = layer.weight.data.cpu().numpy()
    
    # Number of filters
    num_filters = weights.shape[0]
    
    # Create a figure
    plt.figure(figsize=figsize)
    plt.suptitle(f"Filters in {layer_name}")
    
    # Plot each filter
    for i in range(min(32, num_filters)):  # Display up to 32 filters
        plt.subplot(4, 8, i+1)
        plt.imshow(weights[i, 0], cmap='viridis')
        plt.axis('off')
        plt.title(f'Filter {i+1}')
    
    plt.tight_layout()
    plt.show()

# Visualize weight distributions
def visualize_weight_distributions(model, figsize=(12, 8)):
    plt.figure(figsize=figsize)
    
    i = 1
    for name, param in model.named_parameters():
        if 'weight' in name:
            plt.subplot(2, 2, i)
            sns.histplot(param.data.cpu().numpy().flatten(), kde=True)
            plt.title(f"{name} distribution")
            plt.xlabel("Weight value")
            plt.ylabel("Frequency")
            i += 1
            if i > 4:  # Show first 4 weight layers
                break
    
    plt.tight_layout()
    plt.show()

# Visualize the first convolutional layer filters
visualize_conv_filters(model, 'conv1')

# Visualize weight distributions
try:
    import seaborn as sns
    visualize_weight_distributions(model)
except ImportError:
    print("seaborn not installed, skipping weight distribution visualization")

# Try to create a visual graph of the model
try:
    from torchviz import make_dot
    
    # Create a sample input
    x = torch.zeros(1, 1, 14, 14)
    y = model(x)
    
    # Create graph
    dot = make_dot(y, params=dict(model.named_parameters()))
    
    # Save graph to file
    dot.render("model_graph", format="png")
    print("Model graph saved as model_graph.png")
except ImportError:
    print("torchviz not installed. Install with: pip install torchviz graphviz")