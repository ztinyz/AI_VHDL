import torch
import numpy as np
import torch.nn as nn
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

# Import the model definition to ensure compatibility with the trained model
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

def predict_digit(model, matrix, device):
    """Predict a digit from a 14x14 matrix"""
    model.eval()
    
    # Convert to tensor if not already
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix, dtype=torch.float32)
    
    # Ensure correct shape: [1, 1, 14, 14]
    if matrix.dim() == 2:  # If just a 14x14 matrix
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

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    model_path = 'digit_classifier.pth'
    model = DigitClassifier().to(device)
    
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found. Make sure to train the model first.")
        return
    
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