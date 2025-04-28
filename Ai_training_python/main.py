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
        # Input: 9x9 = 81 features
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(81, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 outputs for digits 0-9
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Optimize transforms for speed
transform = transforms.Compose([
    transforms.Resize((9, 9), antialias=True),  # Disable antialiasing for speed
    transforms.ToTensor(),
    transforms.Lambda(binarize)
])

# Load datasets
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    transform=transform, 
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
    num_workers=16,           # Use multiple CPU cores for data loading
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

# Training function
def train(model, train_loader, criterion, optimizer, num_epochs=2000):
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

# Function to predict a digit from a 9x9 matrix
def predict_digit(model, matrix):
    model.eval()
    
    # Convert to tensor if not already
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix, dtype=torch.float32)
    
    # Ensure correct shape: [1, 1, 9, 9]
    if matrix.dim() == 2:  # If just a 9x9 matrix
        matrix = matrix.unsqueeze(0).unsqueeze(0)
    
    # Move to same device as model
    matrix = matrix.to(device)
    
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
    train(model, train_loader, criterion, optimizer, num_epochs=2000)
    
    
    # Evaluate the model
    print("Evaluating model...")
    evaluate(model, test_loader)
    
    # Save the model
    torch.save(model.state_dict(), 'digit_classifier.pth')
    print("Model saved to digit_classifier.pth")
    
    # Example usage:
    # Let's create a sample 9x9 matrix (this would represent a digit)
    # In a real application, you would get this matrix from your input
    sample = torch.zeros((9, 9))
    # Draw a simple pattern (e.g., for digit 1)
    sample[2:7, 4] = 1
    
    print("Sample matrix:")
    print(sample)
    
    # Predict the digit
    digit = predict_digit(model, sample)
    print(f"Predicted digit: {digit}")