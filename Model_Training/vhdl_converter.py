import numpy as np
import torch
from torch.quantization import QuantStub, DeQuantStub
import torch.nn as nn
import torch.nn.functional as F

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

class QuantizedDigitClassifier(nn.Module):
    def __init__(self, original_model):
        super(QuantizedDigitClassifier, self).__init__()
        # Quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # Copy weights from original model
        self.conv1 = original_model.conv1
        self.pool = original_model.pool
        self.conv2 = original_model.conv2
        self.flatten = original_model.flatten
        self.fc1 = original_model.fc1
        self.fc2 = original_model.fc2
        
    def forward(self, x):
        x = self.quant(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dequant(x)
        return x

# Add calibration before converting to int8
def calibrate_model(model, num_samples=100):
    # Create sample calibration data (random 14x14 images)
    dummy_input = torch.zeros((num_samples, 1, 14, 14), dtype=torch.float32)
    
    # Put model in eval mode for calibration
    model.eval()
    
    # Feed data through the model to calibrate observers
    with torch.no_grad():
        model(dummy_input)
    
    print("Calibration complete")

# Quantize model
model_fp32 = DigitClassifier()
model_fp32.load_state_dict(torch.load("digit_classifier.pth"))
model_fp32.eval()

model_int8 = QuantizedDigitClassifier(model_fp32)
model_int8.qconfig = torch.quantization.QConfig(
    activation=torch.quantization.observer.MinMaxObserver.with_args(
        qscheme=torch.per_tensor_symmetric, 
        dtype=torch.quint8
    ),
    weight=torch.quantization.observer.MinMaxObserver.with_args(
        qscheme=torch.per_tensor_symmetric, 
        dtype=torch.qint8
    )
)
torch.quantization.prepare(model_int8, inplace=True)

# Calibrate with sample data
calibrate_model(model_int8)

torch.quantization.convert(model_int8, inplace=True)
# Store original fp32 weights before quantization
model_fp32_state_dict = model_fp32.state_dict()

def extract_weights_to_hex_from_dict(state_dict, bit_width=8):
    weights_dict = {}
    
    # Extract all weights and biases
    for name, param in state_dict.items():
        # Convert to fixed-point by scaling and rounding
        scale_factor = 2**(bit_width-1) - 1
        fixed_point = (param.detach().cpu().numpy() * scale_factor).round().astype(np.int32)
        
        # Convert to hex strings
        hex_values = []
        for value in fixed_point.flatten():
            # Format as width-bit hex
            hex_str = format(value & ((1 << bit_width) - 1), f'0{bit_width//4}x')
            hex_values.append(hex_str)
        
        weights_dict[name] = {
            'shape': param.shape,
            'hex_values': hex_values
        }
    
    return weights_dict

# Extract weights from the FP32 model instead of the quantized model
weights_hex = extract_weights_to_hex_from_dict(model_fp32_state_dict)

# Write weights to a file in VHDL-friendly format
with open('model_weights.vhd', 'w') as f:
    f.write("library IEEE;\n")
    f.write("use IEEE.STD_LOGIC_1164.ALL;\n")
    f.write("use IEEE.NUMERIC_STD.ALL;\n\n")
    f.write("package weights_pkg is\n")
    
    for name, data in weights_hex.items():
        shape_str = '_'.join(str(dim) for dim in data['shape'])
        f.write(f"  type {name.replace('.', '_')}_{shape_str}_t is array (0 to {len(data['hex_values'])-1}) of std_logic_vector(7 downto 0);\n")
        f.write(f"  constant {name.replace('.', '_')} : {name.replace('.', '_')}_{shape_str}_t := (\n")
        
        # Write hex values in groups of 8
        for i in range(0, len(data['hex_values']), 8):
            line = ', '.join([f'x"{val}"' for val in data['hex_values'][i:i+8]])
            if i + 8 < len(data['hex_values']):
                line += ','
            f.write(f"    {line}\n")
        
        f.write("  );\n\n")
    
    f.write("end package weights_pkg;\n")