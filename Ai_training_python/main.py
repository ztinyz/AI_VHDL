import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")

# Define a proper function instead of lambda for pickling compatibility
def binarize(x):
    return (x > 0.5).float()

# Define the neural network
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        
        # Calculate output size after convolutions and pooling
        # Input: 14x14 → Conv → 14x14 → Pool → 7x7 → Conv → 7x7 → Pool → 3x3
        self.fc1 = nn.Linear(6 * 3 * 3, 32)
        self.fc2 = nn.Linear(32, 10)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

# Optimize transforms for speed
transform = transforms.Compose([
    transforms.Resize((14, 14), antialias=True),  # Disable antialiasing for speed
    transforms.ToTensor(),
    transforms.Lambda(binarize)
])

transform_train = transforms.Compose([
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # Randomly shift digits
    transforms.Resize((14, 14), antialias=True),
    transforms.ToTensor(),
    transforms.Lambda(binarize)
])

# Update your dataset to use this transform
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    transform=transform_train,  # Use augmented transform for training
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    transform=transform, 
    download=True
)

# Create data loaders with optimized settings
train_loader = DataLoader(
    train_dataset, 
    batch_size=100,          # Increased batch size if memory allows
    shuffle=True,
    num_workers=8,           # Use multiple CPU cores for data loading
    pin_memory=True,         # Faster transfer to GPU
    persistent_workers=True, # Keep workers alive between iterations
    drop_last=True           # Skip incomplete final batch
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=256,          # Can use larger batches for testing
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

# Initialize the model, loss function, and optimizer
model = DigitClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def center_digit(matrix):
    """Center the digit in a matrix"""
    # Find non-zero points
    indices = torch.nonzero(matrix)
    if indices.size(0) == 0:  # Empty matrix
        return matrix
    
    # Find bounding box
    min_y, min_x = indices.min(dim=0)[0]
    max_y, max_x = indices.max(dim=0)[0]
    
    # Calculate current center and desired center
    current_center_y = (min_y + max_y) // 2
    current_center_x = (min_x + max_x) // 2
    desired_center_y = matrix.size(0) // 2
    desired_center_x = matrix.size(1) // 2
    
    # Calculate shift
    shift_y = desired_center_y - current_center_y
    shift_x = desired_center_x - current_center_x
    
    # Create centered matrix
    centered = torch.zeros_like(matrix)
    
    # Calculate new bounds with clipping to matrix size
    new_min_y = max(0, min_y + shift_y)
    new_max_y = min(matrix.size(0) - 1, max_y + shift_y)
    new_min_x = max(0, min_x + shift_x)
    new_max_x = min(matrix.size(1) - 1, max_x + shift_x)
    
    orig_min_y = max(0, min_y)
    orig_max_y = min(matrix.size(0) - 1, max_y)
    orig_min_x = max(0, min_x)
    orig_max_x = min(matrix.size(1) - 1, max_x)
    
    height = min(new_max_y - new_min_y + 1, orig_max_y - orig_min_y + 1)
    width = min(new_max_x - new_min_x + 1, orig_max_x - orig_min_x + 1)
    
    centered[new_min_y:new_min_y+height, new_min_x:new_min_x+width] = matrix[orig_min_y:orig_min_y+height, orig_min_x:orig_min_x+width]
    
    return centered

# Training function
def train(model, train_loader, criterion, optimizer, num_epochs=200):
    model.train()
    for epoch in range(num_epochs):
        # Rest of the function remains the same

        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            # Asynchronous transfer to GPU
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        epoch_accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {epoch_accuracy:.2f}%')

# Evaluation function
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# Function to predict a digit from a 11x11 matrix
def predict_digit(model, matrix):
    model.eval()
    
    # Convert to tensor if not already
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix, dtype=torch.float32)
    
    # Ensure correct shape: [1, 1, 11, 11]
    if matrix.dim() == 2:  # If just a 11x11 matrix
        matrix = matrix.unsqueeze(0).unsqueeze(0)
    
    # Move to same device as model
    matrix = center_digit(matrix)
    
    with torch.no_grad():
        output = model(matrix)
        _, predicted = torch.max(output, 1)
        
    return predicted.item()

# Main execution
if __name__ == "__main__":
      
    # Set multiprocessing start method for Windows
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    
    # Train the model
    print("Starting training...")
    train(model, train_loader, criterion, optimizer, num_epochs=200)
    
    
    # Evaluate the model
    print("Evaluating model...")
    evaluate(model, test_loader)
    
    # Save the model
    torch.save(model.state_dict(), 'digit_classifier.pth')
    print("Model saved to digit_classifier.pth")
    
    # Example usage:
    # Let's create a sample 11x11 matrix (this would represent a digit)
    # In a real application, you would get this matrix from your input
    sample = torch.zeros((14, 14))
    # Draw a simple pattern (e.g., for digit 1)
    sample[2:7, 4] = 1
    
    print("Sample matrix:")
    print(sample)
    
    # Predict the digit
    digit = predict_digit(model, sample)
    print(f"Predicted digit: {digit}")