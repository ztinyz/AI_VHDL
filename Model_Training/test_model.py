import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.patches as patches

def draw_digit_with_mouse():
    """Let the user draw a digit with the mouse on a 14x14 grid"""
    # Create a new figure with specified size
    fig, ax = plt.subplots(figsize=(14, 14))
    plt.subplots_adjust(bottom=0.2)  # Make room for buttons
    
    # Create empty matrix
    matrix = torch.zeros((14, 14))
    drawing = True
    
    # Set up the grid display
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 14)
    ax.set_xticks(range(15))
    ax.set_yticks(range(15))
    ax.grid(True)
    ax.set_title("Draw a digit (click and drag to draw)")
    
    # Invert y-axis so 0,0 is at the top-left
    ax.invert_yaxis()
    
    # Create rectangles for each cell
    rects = {}
    for i in range(14):
        for j in range(14):
            rects[(i, j)] = patches.Rectangle((j, i), 1, 1, fill=False)
            ax.add_patch(rects[(i, j)])
    
    # Function to update display when matrix changes
    def update_display():
        for i in range(14):
            for j in range(14):
                rects[(i, j)].set_facecolor('black' if matrix[i, j] > 0 else 'white')
                rects[(i, j)].set_fill(matrix[i, j] > 0)
        fig.canvas.draw_idle()
    
    # Mouse event handlers
    def on_mouse_press(event):
        if event.inaxes != ax or not drawing:
            return
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= x < 14 and 0 <= y < 14:
            matrix[y, x] = 1
            update_display()
    
    def on_mouse_move(event):
        if event.inaxes != ax or not event.button or not drawing:
            return
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= x < 14 and 0 <= y < 14:
            matrix[y, x] = 1
            update_display()
    
    # Clear button callback
    def on_clear(event):
        nonlocal matrix
        matrix = torch.zeros((14, 14))
        update_display()
    
    # Done button callback
    def on_done(event):
        nonlocal drawing
        drawing = False
        plt.close(fig)
    
    # Add the Clear and Done buttons
    ax_clear = plt.axes([0.2, 0.05, 0.2, 0.075])
    ax_done = plt.axes([0.6, 0.05, 0.2, 0.075])
    btn_clear = Button(ax_clear, 'Clear')
    btn_done = Button(ax_done, 'Done')
    btn_clear.on_clicked(on_clear)
    btn_done.on_clicked(on_done)
    
    # Connect event handlers
    fig.canvas.mpl_connect('button_press_event', on_mouse_press)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    
    # Show the drawing interface
    plt.show()
    
    return matrix

# Improved Quantized Linear Layer (from improved_training.py)
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
    
    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bits={self.bits}'

# Improved model definition that matches improved_training.py
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
        
    def enable_quantization_training(self):
        """Enable quantization training for all quantized layers"""
        if self.use_quantized:
            self.fc1.enable_quantization_training()
            self.fc2.enable_quantization_training()
            self.fc3.enable_quantization_training()
        
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

# Legacy model definition for backward compatibility
class DigitClassifier(nn.Module):
    def __init__(self, use_quantized=True, bits=8, model_size='large'):
        super(DigitClassifier, self).__init__()
        
        # Define model sizes
        if model_size == 'small':
            conv1_channels, conv2_channels = 3, 6
            fc1_neurons, fc2_neurons, fc3_neurons = 32, 16, 10
        elif model_size == 'medium':
            conv1_channels, conv2_channels = 8, 16
            fc1_neurons, fc2_neurons, fc3_neurons = 128, 64, 10
        elif model_size == 'large':
            conv1_channels, conv2_channels = 16, 32
            fc1_neurons, fc2_neurons, fc3_neurons = 256, 128, 10
        elif model_size == 'xlarge':
            conv1_channels, conv2_channels = 32, 64
            fc1_neurons, fc2_neurons, fc3_neurons = 512, 256, 10
        else:
            raise ValueError("model_size must be 'small', 'medium', 'large', or 'xlarge'")
        
        # Convolutional layers (increased channels)
        self.conv1 = nn.Conv2d(1, conv1_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        
        # Calculate flattened size: 14x14 -> 7x7 -> 3x3 (after two pooling operations)
        flattened_size = conv2_channels * 3 * 3
        
        # Use quantized linear layers with more neurons
        if use_quantized:
            self.fc1 = ImprovedQuantLinear(flattened_size, fc1_neurons, bits=bits)
            self.fc2 = ImprovedQuantLinear(fc1_neurons, fc2_neurons, bits=bits)
            self.fc3 = ImprovedQuantLinear(fc2_neurons, fc3_neurons, bits=bits)
        else:
            self.fc1 = nn.Linear(flattened_size, fc1_neurons)
            self.fc2 = nn.Linear(fc1_neurons, fc2_neurons)
            self.fc3 = nn.Linear(fc2_neurons, fc3_neurons)
        
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.use_quantized = use_quantized
        self.model_size = model_size
        
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
        quantized_weights = {}
        if self.use_quantized:
            quantized_weights['fc1'] = self.fc1.get_quantized_weights()
            quantized_weights['fc2'] = self.fc2.get_quantized_weights()
            quantized_weights['fc3'] = self.fc3.get_quantized_weights()
        return quantized_weights

def center_digit(matrix):
    """Center the digit in a matrix"""
    # Find non-zero points
    indices = torch.nonzero(matrix)
    if indices.size(0) == 0:  # Empty matrix
        return matrix
    
    # Find bounding box - indices are [row, col] format
    min_coords = indices.min(dim=0)[0]
    max_coords = indices.max(dim=0)[0]
    
    min_y, min_x = min_coords[0].item(), min_coords[1].item()
    max_y, max_x = max_coords[0].item(), max_coords[1].item()
    
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

def predict_digit(model, matrix, device):
    """Predict a digit from a 14x14 matrix"""
    model.eval()
    
    # Convert to tensor if not already
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix, dtype=torch.float32)
    
    # Ensure we're working with a 2D matrix first for centering
    if matrix.dim() == 4:  # [batch, channel, height, width]
        matrix = matrix.squeeze(0).squeeze(0)
    elif matrix.dim() == 3:  # [channel, height, width]
        matrix = matrix.squeeze(0)
    
    # Center the digit
    matrix = center_digit(matrix)
    
    # Add batch and channel dimensions: [1, 1, height, width]
    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0).unsqueeze(0)
    
    # Move to same device as model
    matrix = matrix.to(device)
    
    with torch.no_grad():
        output = model(matrix)
        _, predicted = torch.max(output, 1)
        
    return predicted.item()

def display_matrix(matrix):
    """Display a matrix as an image"""
    plt.figure(figsize=(8, 8))
    plt.imshow(matrix, cmap='binary')
    plt.grid(True)
    plt.title("14x14 Matrix Input")
    plt.show()

def create_sample_matrices():
    """Create sample matrices for digits 0-9"""
    random_samples = []
    
    # Random digit 0 - slightly off-center
    zero = torch.zeros((14, 14))
    zero[2:5, 4:10] = 1
    zero[4:10, 3:5] = 1
    zero[4:10, 9:11] = 1
    zero[9:12, 4:10] = 1
    random_samples.append((zero, 0))
    
    # Random digit 1 - thick
    one = torch.zeros((14, 14))
    one[2:12, 6:8] = 1
    one[10:12, 5:6] = 1
    random_samples.append((one, 1))
    
    # Random digit 2 - stylized
    two = torch.zeros((14, 14))
    two[2:5, 4:10] = 1
    two[4:7, 8:11] = 1
    two[6:9, 6:9] = 1
    two[8:11, 3:6] = 1
    two[10:12, 3:11] = 1
    random_samples.append((two, 2))
    
    # Random digit 3 - narrow
    three = torch.zeros((14, 14))
    three[2:4, 4:10] = 1
    three[4:7, 8:10] = 1
    three[6:8, 5:9] = 1
    three[8:10, 8:10] = 1
    three[10:12, 4:10] = 1
    random_samples.append((three, 3))
    
    # Random digit 4 - sharp angles
    four = torch.zeros((14, 14))
    four[2:11, 8:10] = 1
    four[6:8, 4:9] = 1
    four[2:7, 4:6] = 1
    random_samples.append((four, 4))
    
    # Random digit 5 - angular
    five = torch.zeros((14, 14))
    five[2:4, 3:11] = 1
    five[4:7, 3:5] = 1
    five[6:8, 3:10] = 1
    five[8:10, 8:11] = 1
    five[10:12, 3:9] = 1
    random_samples.append((five, 5))
    
    # Random digit 6 - tilted
    six = torch.zeros((14, 14))
    six[2:11, 4:6] = 1
    six[6:8, 5:10] = 1
    six[8:11, 7:10] = 1
    six[10:12, 5:8] = 1
    random_samples.append((six, 6))
    
    # Random digit 7 - serif style
    seven = torch.zeros((14, 14))
    seven[2:4, 3:11] = 1
    seven[4:7, 9:11] = 1
    seven[7:12, 7:9] = 1
    random_samples.append((seven, 7))
    
    # Random digit 8 - thin
    eight = torch.zeros((14, 14))
    eight[2:4, 5:9] = 1
    eight[4:6, 4:5] = 1
    eight[4:6, 9:10] = 1
    eight[6:8, 5:9] = 1
    eight[8:10, 4:5] = 1
    eight[8:10, 9:10] = 1
    eight[10:12, 5:9] = 1
    random_samples.append((eight, 8))
    
    # Random digit 9 - rotated slightly
    nine = torch.zeros((14, 14))
    nine[2:4, 4:10] = 1
    nine[4:8, 3:5] = 1
    nine[4:10, 9:11] = 1
    nine[9:12, 4:10] = 1
    random_samples.append((nine, 0))
    
    return random_samples

def detect_model_size(model_path):
    """Detect the model size by examining the saved state dict"""
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Check conv2 output channels to determine model size
        conv2_channels = state_dict['conv2.weight'].shape[0]
        
        if conv2_channels == 6:
            return 'small'
        elif conv2_channels == 16:
            return 'medium'
        elif conv2_channels == 32:
            return 'large'
        elif conv2_channels == 64:
            return 'xlarge'
        else:
            print(f"Unknown model size with {conv2_channels} conv2 channels. Defaulting to 'large'")
            return 'large'
    except:
        print("Could not detect model size. Defaulting to 'large'")
        return 'large'

def detect_model_type(model_path):
    """Detect if the model is improved or legacy type"""
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Check if it's an improved model (fixed architecture)
        if 'conv2.weight' in state_dict:
            conv2_shape = state_dict['conv2.weight'].shape
            if conv2_shape == torch.Size([16, 8, 3, 3]):  # Fixed improved architecture
                return 'improved'
        
        return 'legacy'
    except:
        return 'unknown'

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Try to load the improved model first, then fallback to legacy model
    model = None
    model_path = None
    
    # Check for improved model first
    model_path = 'digit_classifier.pth'
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        # Detect model size from saved weights
        model_size = detect_model_size(model_path)
        print(f"Detected model size: {model_size}")
        
        # Create model with detected size
        model = DigitClassifier(use_quantized=True, bits=8, model_size=model_size).to(device)
        
        # Load the weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"model loaded successfully!")
    except FileNotFoundError:
        print(f"Error: No model files found. Please train a model first.")
        print(f"Looking for: {model_path}")
        return
    except Exception as e:
        print(f"Error loading legacy model: {e}")
        return
            
    except Exception as e:
        print(f"Error loading improved model: {e}")
        return
    
    if model is None:
        print("Failed to load any model. Exiting.")
        return
    
    print(f"Model loaded from: {model_path}")
    model.eval()  # Set to evaluation mode
    
    while True:
        print("\nDigit Recognition Test Program")
        print("1. Test with sample digits (0-9)")
        print("2. Create your own 14x14 matrix")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            samples = create_sample_matrices()
            for i, (matrix, true_label) in enumerate(samples):
                print(f"\nTesting sample {i+1} - True digit: {true_label}")
                display_matrix(matrix)
                prediction = predict_digit(model, matrix, device)
                print(f"Predicted digit: {prediction}")
                
                if prediction == true_label:
                    print("✓ Correct")
                else:
                    print("✗ Incorrect")
                
        elif choice == '2':
            print("Drawing interface will open. Click and drag to draw, then click 'Done' when finished.")
            matrix = draw_digit_with_mouse()
            print("Processing your drawing...")
            display_matrix(matrix)
            prediction = predict_digit(model, matrix, device)
            print(f"Predicted digit: {prediction}")
            
        elif choice == '3':
            print("Exiting program")
            break
        
        else:
            print("Invalid choice, please try again")

if __name__ == "__main__":
    main()