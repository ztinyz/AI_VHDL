import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the original model architecture
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
    
# Assuming model is your initialized DigitClassifier
model = DigitClassifier().to(device)

state_dict = torch.load("digit_classifier.pth")
model.load_state_dict(state_dict)

summary(model, (1, 14, 14))  # Input shape: channels, height, width