import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def binarize(x):
    return (x > 0.5).float()

# Better Quantized Linear Layer
class ImprovedQuantLinear(nn.Module):
    def __init__(self, in_features, out_features, bits=8):
        super(ImprovedQuantLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        
        # Initialize weights and bias as floating point for training
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Quantization range
        self.qmin = -(2**(bits-1))
        self.qmax = 2**(bits-1) - 1
        
        # Training phase flag
        self.quantization_training = False
        
    def enable_quantization_training(self):
        """Enable quantization during training"""
        self.quantization_training = True
        
    def quantize_weights(self):
        """Quantize weights to integers with better scaling"""
        # Use a learnable scale based on weight statistics
        weight_abs = torch.abs(self.weight)
        scale = torch.max(weight_abs) / self.qmax
        scale = torch.clamp(scale, min=1e-8)  # Avoid division by zero
        
        # Scale and quantize
        weight_scaled = self.weight / scale
        weight_quantized = torch.clamp(torch.round(weight_scaled * self.qmax), self.qmin, self.qmax)
        
        # Scale back for actual computation
        return weight_quantized * scale / self.qmax
    
    def forward(self, x):
        if self.training and self.quantization_training:
            # During quantization training, use straight-through estimator
            weight_q = self.quantize_weights()
            # Straight-through: forward uses quantized, backward uses original
            weight_ste = weight_q + (self.weight - self.weight.detach())
            output = F.linear(x, weight_ste, self.bias)
        elif not self.training:
            # During inference, use quantized weights
            weight_q = self.quantize_weights()
            output = F.linear(x, weight_q, self.bias)
        else:
            # Normal training without quantization
            output = F.linear(x, self.weight, self.bias)
        
        return output
    
    def get_quantized_weights(self):
        """Get the actual integer weights for export"""
        with torch.no_grad():
            weight_abs = torch.abs(self.weight)
            scale = torch.max(weight_abs) / self.qmax
            scale = torch.clamp(scale, min=1e-8)
            
            weight_scaled = self.weight / scale
            weight_quantized = torch.clamp(torch.round(weight_scaled * self.qmax), self.qmin, self.qmax)
            return weight_quantized.int()

# Define the neural network
class ImprovedDigitClassifier(nn.Module):
    def __init__(self, use_quantized=True, bits=8):
        super(ImprovedDigitClassifier, self).__init__()
        
        # Convolutional layers (keep these as regular layers)
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        
        # Calculate flattened size: 14x14 -> 7x7 -> 3x3 (after two pooling operations)
        flattened_size = 16 * 3 * 3
        
        # Use quantized linear layers
        if use_quantized:
            self.fc1 = ImprovedQuantLinear(flattened_size, 128, bits=bits)
            self.fc2 = ImprovedQuantLinear(128, 64, bits=bits)
            self.fc3 = ImprovedQuantLinear(64, 10, bits=bits)
        else:
            self.fc1 = nn.Linear(flattened_size, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 10)
        
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.use_quantized = use_quantized
        
        print(f"Model created with quantization: {use_quantized}")
        print(f"Conv layers: 1->8->16")
        print(f"FC layers: {flattened_size}->128->64->10")
        
    def enable_quantization_training(self):
        """Enable quantization training for all quantized layers"""
        if self.use_quantized:
            self.fc1.enable_quantization_training()
            self.fc2.enable_quantization_training()
            self.fc3.enable_quantization_training()
            print("Quantization training enabled!")
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
    
    def get_quantized_weights(self):
        """Get all quantized weights for export"""
        if self.use_quantized:
            return {
                'fc1': self.fc1.get_quantized_weights(),
                'fc2': self.fc2.get_quantized_weights(),
                'fc3': self.fc3.get_quantized_weights()
            }
        return {}

# Data setup
transform = transforms.Compose([
    transforms.Resize((14, 14), antialias=True),
    transforms.ToTensor(),
    transforms.Lambda(binarize)
])

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

train_loader = DataLoader(
    train_dataset, 
    batch_size=100,          # Increased batch size if memory allows
    shuffle=True,
    num_workers=16,           # Use multiple CPU cores for data loading
    pin_memory=False,         # Faster transfer to GPU
    persistent_workers=True, # Keep workers alive between iterations
    drop_last=True           # Skip incomplete final batch
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=256,          # Can use larger batches for testing
    shuffle=False,
    num_workers=8,
    pin_memory=False,
    persistent_workers=True
)

# Initialize the model
model = ImprovedDigitClassifier(use_quantized=True, bits=8).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def predict_digit(model, matrix, debug=False):
    model.eval()
    
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix, dtype=torch.float32)
    
    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0).unsqueeze(0)
    
    matrix = matrix.to(device)
    
    with torch.no_grad():
        output = model(matrix)
        probabilities = torch.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
        
        if debug:
            print(f"Raw output: {output[0]}")
            print(f"Probabilities: {probabilities[0]}")
            print(f"Predicted: {predicted.item()}")
            print(f"Confidence: {probabilities[0][predicted].item():.3f}")
        
    return predicted.item()

def train_phase(model, train_loader, criterion, optimizer, num_epochs, phase_name):
    print(f"\n=== {phase_name} ===")
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 200 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        epoch_accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {epoch_accuracy:.2f}%')

# Main execution
if __name__ == "__main__":
    print("=== TWO-PHASE TRAINING APPROACH ===")
    
    # Phase 1: Train normally without quantization
    train_phase(model, train_loader, criterion, optimizer, 25, "PHASE 1: Normal Training")
    
    # Phase 2: Enable quantization and fine-tune
    model.enable_quantization_training()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate for fine-tuning
    train_phase(model, train_loader, criterion, optimizer, 100, "PHASE 2: Quantization Fine-tuning")
    
    # Test with sample patterns
    print("\n" + "="*50)
    print("TESTING MODEL PREDICTIONS AFTER TWO-PHASE TRAINING")
    print("="*50)
    
    # Create different digit patterns
    test_samples = []
    
    # Sample 1 (vertical line for digit 1)
    sample1 = torch.zeros((14, 14))
    sample1[2:12, 6:8] = 1
    test_samples.append((sample1, "Vertical line (should be 1)"))
    
    # Sample 0 (circle/oval for digit 0)
    sample0 = torch.zeros((14, 14))
    sample0[3:5, 5:9] = 1    # top
    sample0[5:9, 4:5] = 1    # left
    sample0[5:9, 9:10] = 1   # right
    sample0[9:11, 5:9] = 1   # bottom
    test_samples.append((sample0, "Oval shape (should be 0)"))
    
    # Sample 8 (figure-8 for digit 8)
    sample8 = torch.zeros((14, 14))
    sample8[3:5, 5:9] = 1    # top
    sample8[5:7, 4:5] = 1    # left top
    sample8[5:7, 9:10] = 1   # right top
    sample8[6:8, 5:9] = 1    # middle
    sample8[8:10, 4:5] = 1   # left bottom
    sample8[8:10, 9:10] = 1  # right bottom
    sample8[10:12, 5:9] = 1  # bottom
    test_samples.append((sample8, "Figure-8 shape (should be 8)"))
    
    for i, (sample, description) in enumerate(test_samples):
        print(f"\nTest {i+1}: {description}")
        print("Sample matrix:")
        for row in sample:
            line = ""
            for val in row:
                line += "█" if val > 0 else "."
            print(line)
        
        prediction = predict_digit(model, sample, debug=True)
        print(f"Final prediction: {prediction}")
        print("-" * 30)
    
    # Save the model
    torch.save(model.state_dict(), 'improved_digit_classifier.pth')
    print("\nModel saved to improved_digit_classifier.pth")
    
    # Extract and display quantized weights
    quantized_weights = model.get_quantized_weights()
    if quantized_weights:
        print("\nQuantized weights extracted successfully!")
        for layer_name, weights in quantized_weights.items():
            print(f"{layer_name} weights shape: {weights.shape}")
            print(f"{layer_name} weights range: [{weights.min().item()}, {weights.max().item()}]")
