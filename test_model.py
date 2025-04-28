import torch
import numpy as np
import matplotlib.pyplot as plt

# Import the model definition to ensure compatibility
class DigitClassifier(torch.nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        # Input: 9x9 = 81 features
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(81, 128)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 10)  # 10 outputs for digits 0-9
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def predict_digit(model, matrix, device):
    """Predict a digit from a 9x9 matrix"""
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

def display_matrix(matrix):
    """Display a matrix as an image"""
    plt.figure(figsize=(5, 5))
    plt.imshow(matrix, cmap='binary')
    plt.grid(True)
    plt.title("9x9 Matrix Input")
    plt.show()

def create_sample_matrices():
    """Create sample matrices for digits 0-9"""
    samples = []
    
    # Digit 0
    zero = torch.zeros((9, 9))
    zero[1:3, 3:6] = 1
    zero[3:6, 2:3] = 1
    zero[3:6, 6:7] = 1
    zero[6:8, 3:6] = 1
    samples.append(zero)
    
    # Digit 1
    one = torch.zeros((9, 9))
    one[1:7, 4:5] = 1
    samples.append(one)
    
    # Digit 2
    two = torch.zeros((9, 9))
    two[1:3, 2:7] = 1
    two[3:5, 5:7] = 1
    two[4:6, 3:5] = 1
    two[6:8, 2:7] = 1
    samples.append(two)
    
    # Digit 3
    three = torch.zeros((9, 9))
    three[1:3, 2:7] = 1
    three[3:5, 5:7] = 1
    three[5:6, 3:6] = 1
    three[6:8, 2:7] = 1
    samples.append(three)
    
    # Digit 4
    four = torch.zeros((9, 9))
    four[1:5, 2:3] = 1
    four[3:5, 3:6] = 1
    four[1:8, 6:7] = 1
    samples.append(four)
    
    # Digit 5
    five = torch.zeros((9, 9))
    five[1:3, 2:7] = 1
    five[3:5, 2:4] = 1
    five[4:6, 3:7] = 1
    five[6:8, 2:6] = 1
    samples.append(five)
    
    # Digit 6
    six = torch.zeros((9, 9))
    six[1:8, 2:4] = 1
    six[3:5, 4:7] = 1
    six[5:8, 5:7] = 1
    six[7:8, 3:5] = 1
    samples.append(six)
    
    # Digit 7
    seven = torch.zeros((9, 9))
    seven[1:3, 2:7] = 1
    seven[3:8, 5:7] = 1
    samples.append(seven)
    
    # Digit 8
    eight = torch.zeros((9, 9))
    eight[1:3, 3:6] = 1
    eight[3:4, 2:3] = 1
    eight[3:4, 6:7] = 1
    eight[4:5, 3:6] = 1
    eight[5:6, 2:3] = 1
    eight[5:6, 6:7] = 1
    eight[6:8, 3:6] = 1
    samples.append(eight)
    
    # Digit 9
    nine = torch.zeros((9, 9))
    nine[1:4, 3:6] = 1
    nine[2:6, 6:7] = 1
    nine[4:6, 3:6] = 1
    nine[6:8, 3:6] = 1
    samples.append(nine)
    
    return samples

def create_matrix_from_input():
    """Let the user create a 9x9 matrix by entering each row"""
    print("Enter 9 rows of 9 values (0 or 1) separated by spaces:")
    matrix = []
    for i in range(9):
        while True:
            try:
                row = input(f"Row {i+1}: ")
                row_values = [int(x) for x in row.split()]
                if len(row_values) != 9 or not all(x in [0, 1] for x in row_values):
                    print("Please enter exactly 9 values (0 or 1) separated by spaces")
                    continue
                matrix.append(row_values)
                break
            except ValueError:
                print("Invalid input, please use only 0s and 1s")
    
    return torch.tensor(matrix, dtype=torch.float32)

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
        print("2. Create your own 9x9 matrix")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            samples = create_sample_matrices()
            for i, sample in enumerate(samples):
                display_matrix(sample)
                prediction = predict_digit(model, sample, device)
                print(f"Predicted digit: {prediction}")
                
        elif choice == '2':
            matrix = create_matrix_from_input()
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