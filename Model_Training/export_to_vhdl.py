import torch
import numpy as np
import sys
import os

def load_quantized_weights_from_model(model_file='improved_digit_classifier.pth'):
    """
    Load quantized weights from a trained model file
    Supports both improved and legacy model formats
    """
    try:
        # Try to import model classes
        sys.path.append(os.path.dirname(__file__))
        from test_model import ImprovedDigitClassifier, DigitClassifier
        
        device = torch.device("cpu")  # Load on CPU for export
        
        # First try to load as improved model
        try:
            print(f"Attempting to load improved model from {model_file}...")
            model = ImprovedDigitClassifier(use_quantized=True, bits=8)
            model.load_state_dict(torch.load(model_file, map_location=device))
            model.eval()
            print("✓ Improved model loaded successfully")
            return model.get_quantized_weights(), 'improved'
            
        except (FileNotFoundError, RuntimeError, KeyError) as e:
            print(f"Failed to load as improved model: {e}")
            
            # Try legacy model with different sizes
            legacy_file = 'digit_classifier.pth'
            if os.path.exists(legacy_file):
                print(f"Attempting to load legacy model from {legacy_file}...")
                
                # Detect model size
                state_dict = torch.load(legacy_file, map_location=device)
                conv2_channels = state_dict['conv2.weight'].shape[0]
                
                if conv2_channels == 6:
                    model_size = 'small'
                elif conv2_channels == 16:
                    model_size = 'medium'
                elif conv2_channels == 32:
                    model_size = 'large'
                elif conv2_channels == 64:
                    model_size = 'xlarge'
                else:
                    model_size = 'large'
                
                print(f"Detected legacy model size: {model_size}")
                model = DigitClassifier(use_quantized=True, bits=8, model_size=model_size)
                model.load_state_dict(state_dict)
                model.eval()
                print("✓ Legacy model loaded successfully")
                return model.get_quantized_weights(), 'legacy'
            else:
                raise FileNotFoundError(f"Neither {model_file} nor {legacy_file} found")
                
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def export_weights_to_vhdl(model_file='improved_digit_classifier.pth', output_file='quantized_weights.vhd'):
    """
    Export quantized integer weights to VHDL format
    Supports both improved and legacy model formats
    """
    
    # First try to load from a model file
    quantized_weights, model_type = load_quantized_weights_from_model(model_file)
    
    if quantized_weights is None:
        # Fallback: try to load from a direct weights file
        try:
            print(f"Attempting to load weights directly from {model_file}...")
            quantized_weights = torch.load(model_file, map_location='cpu')
            model_type = 'direct'
        except Exception as e:
            print(f"Could not load weights from {model_file}: {e}")
            print("Please make sure you have trained a model first using improved_training.py or main.py")
            return None, None, None
    
    print(f"Loaded quantized weights from {model_type} model")
    
    try:
        with open(output_file, 'w') as f:
            f.write("-- Quantized Neural Network Weights\n")
            f.write("-- Generated automatically from PyTorch model\n")
            f.write(f"-- Model type: {model_type}\n\n")
            f.write("library IEEE;\n")
            f.write("use IEEE.STD_LOGIC_1164.ALL;\n")
            f.write("use IEEE.NUMERIC_STD.ALL;\n\n")
            
            f.write("package quantized_weights is\n\n")
            
            # Export FC1 weights
            if 'fc1' in quantized_weights:
                fc1_weights = quantized_weights['fc1']
                if hasattr(fc1_weights, 'cpu'):
                    fc1_weights = fc1_weights.cpu().numpy()
                else:
                    fc1_weights = np.array(fc1_weights)
                    
                f.write(f"    -- FC1 Layer: {fc1_weights.shape[1]} inputs -> {fc1_weights.shape[0]} outputs\n")
                f.write(f"    constant FC1_INPUTS : integer := {fc1_weights.shape[1]};\n")
                f.write(f"    constant FC1_OUTPUTS : integer := {fc1_weights.shape[0]};\n")
                f.write("    type fc1_weight_array is array (0 to FC1_OUTPUTS-1, 0 to FC1_INPUTS-1) of signed(7 downto 0);\n")
                f.write("    constant FC1_WEIGHTS : fc1_weight_array := (\n")
                
                for i, row in enumerate(fc1_weights):
                    f.write("        (")
                    for j, weight in enumerate(row):
                        if j > 0:
                            f.write(", ")
                        # Ensure weight is in valid 8-bit signed range
                        weight_val = int(np.clip(weight, -128, 127))
                        f.write(f'"{format(weight_val & 0xFF, "08b")}"')  # Convert to 8-bit binary
                    f.write(")")
                    if i < len(fc1_weights) - 1:
                        f.write(",")
                    f.write("\n")
                
                f.write("    );\n\n")
            else:
                print("Warning: FC1 weights not found in model")
                fc1_weights = None
            
            # Export FC2 weights
            if 'fc2' in quantized_weights:
                fc2_weights = quantized_weights['fc2']
                if hasattr(fc2_weights, 'cpu'):
                    fc2_weights = fc2_weights.cpu().numpy()
                else:
                    fc2_weights = np.array(fc2_weights)
                    
                f.write(f"    -- FC2 Layer: {fc2_weights.shape[1]} inputs -> {fc2_weights.shape[0]} outputs\n")
                f.write(f"    constant FC2_INPUTS : integer := {fc2_weights.shape[1]};\n")
                f.write(f"    constant FC2_OUTPUTS : integer := {fc2_weights.shape[0]};\n")
                f.write("    type fc2_weight_array is array (0 to FC2_OUTPUTS-1, 0 to FC2_INPUTS-1) of signed(7 downto 0);\n")
                f.write("    constant FC2_WEIGHTS : fc2_weight_array := (\n")
                
                for i, row in enumerate(fc2_weights):
                    f.write("        (")
                    for j, weight in enumerate(row):
                        if j > 0:
                            f.write(", ")
                        # Ensure weight is in valid 8-bit signed range
                        weight_val = int(np.clip(weight, -128, 127))
                        f.write(f'"{format(weight_val & 0xFF, "08b")}"')  # Convert to 8-bit binary
                    f.write(")")
                    if i < len(fc2_weights) - 1:
                        f.write(",")
                    f.write("\n")
                
                f.write("    );\n\n")
            else:
                print("Warning: FC2 weights not found in model")
                fc2_weights = None
            
            # Export FC3 weights (if exists)
            fc3_weights = None
            if 'fc3' in quantized_weights:
                fc3_weights = quantized_weights['fc3']
                if hasattr(fc3_weights, 'cpu'):
                    fc3_weights = fc3_weights.cpu().numpy()
                else:
                    fc3_weights = np.array(fc3_weights)
                    
                f.write(f"    -- FC3 Layer: {fc3_weights.shape[1]} inputs -> {fc3_weights.shape[0]} outputs\n")
                f.write(f"    constant FC3_INPUTS : integer := {fc3_weights.shape[1]};\n")
                f.write(f"    constant FC3_OUTPUTS : integer := {fc3_weights.shape[0]};\n")
                f.write("    type fc3_weight_array is array (0 to FC3_OUTPUTS-1, 0 to FC3_INPUTS-1) of signed(7 downto 0);\n")
                f.write("    constant FC3_WEIGHTS : fc3_weight_array := (\n")
                
                for i, row in enumerate(fc3_weights):
                    f.write("        (")
                    for j, weight in enumerate(row):
                        if j > 0:
                            f.write(", ")
                        # Ensure weight is in valid 8-bit signed range
                        weight_val = int(np.clip(weight, -128, 127))
                        f.write(f'"{format(weight_val & 0xFF, "08b")}"')  # Convert to 8-bit binary
                    f.write(")")
                    if i < len(fc3_weights) - 1:
                        f.write(",")
                    f.write("\n")
                
                f.write("    );\n\n")
            
            f.write("end package quantized_weights;\n")
        
        print(f"✓ VHDL weights exported to {output_file}")
        if fc1_weights is not None:
            print(f"✓ FC1 layer: {fc1_weights.shape[1]} inputs -> {fc1_weights.shape[0]} outputs")
        if fc2_weights is not None:
            print(f"✓ FC2 layer: {fc2_weights.shape[1]} inputs -> {fc2_weights.shape[0]} outputs")
        if fc3_weights is not None:
            print(f"✓ FC3 layer: {fc3_weights.shape[1]} inputs -> {fc3_weights.shape[0]} outputs")
        
        # Also create a more readable text format
        text_file = output_file.replace('.vhd', '_readable.txt')
        with open(text_file, 'w') as f:
            f.write("Quantized Neural Network Weights (Integer Format)\n")
            f.write("=" * 50 + "\n")
            f.write(f"Model type: {model_type}\n\n")
            
            if fc1_weights is not None:
                f.write("FC1 Layer Weights:\n")
                f.write(f"Shape: {fc1_weights.shape}\n")
                f.write(f"Range: [{fc1_weights.min()}, {fc1_weights.max()}]\n")
                f.write("Weights:\n")
                np.savetxt(f, fc1_weights, fmt='%4d', delimiter=' ')
            
            if fc2_weights is not None:
                f.write("\n\nFC2 Layer Weights:\n")
                f.write(f"Shape: {fc2_weights.shape}\n")
                f.write(f"Range: [{fc2_weights.min()}, {fc2_weights.max()}]\n")
                f.write("Weights:\n")
                np.savetxt(f, fc2_weights, fmt='%4d', delimiter=' ')
            
            if fc3_weights is not None:
                f.write("\n\nFC3 Layer Weights:\n")
                f.write(f"Shape: {fc3_weights.shape}\n")
                f.write(f"Range: [{fc3_weights.min()}, {fc3_weights.max()}]\n")
                f.write("Weights:\n")
                np.savetxt(f, fc3_weights, fmt='%4d', delimiter=' ')
        
        print(f"✓ Readable weights exported to {text_file}")
        
        return fc1_weights, fc2_weights, fc3_weights
        
    except Exception as e:
        print(f"Error during export: {e}")
        return None, None, None

def create_vhdl_inference_entity():
    """Create a VHDL entity template for neural network inference"""
    
    vhdl_entity = """-- Quantized Neural Network Inference Entity
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;
use work.quantized_weights.all;

entity quantized_nn_inference is
    Port ( 
        clk : in STD_LOGIC;
        rst : in STD_LOGIC;
        start : in STD_LOGIC;
        input_data : in signed(7 downto 0);  -- 8-bit signed input
        input_valid : in STD_LOGIC;
        output_class : out integer range 0 to 9;
        output_valid : out STD_LOGIC
    );
end quantized_nn_inference;

architecture Behavioral of quantized_nn_inference is
    
    -- Internal signals for layer computations
    type fc1_input_array is array (0 to FC1_INPUTS-1) of signed(7 downto 0);
    type fc1_output_array is array (0 to FC1_OUTPUTS-1) of signed(15 downto 0);
    type fc2_output_array is array (0 to FC2_OUTPUTS-1) of signed(15 downto 0);
    
    signal fc1_inputs : fc1_input_array;
    signal fc1_outputs : fc1_output_array;
    signal fc2_outputs : fc2_output_array;
    
    signal input_counter : integer range 0 to FC1_INPUTS-1;
    signal computation_state : integer range 0 to 3;
    
begin

    process(clk, rst)
    begin
        if rst = '1' then
            input_counter <= 0;
            computation_state <= 0;
            output_valid <= '0';
            
        elsif rising_edge(clk) then
            case computation_state is
                when 0 => -- Input loading state
                    if input_valid = '1' then
                        fc1_inputs(input_counter) <= input_data;
                        if input_counter = FC1_INPUTS-1 then
                            input_counter <= 0;
                            computation_state <= 1;
                        else
                            input_counter <= input_counter + 1;
                        end if;
                    end if;
                    
                when 1 => -- FC1 computation
                    -- Implement matrix multiplication for FC1 layer
                    -- This would require multiple clock cycles for full computation
                    computation_state <= 2;
                    
                when 2 => -- FC2 computation
                    -- Implement matrix multiplication for FC2 layer
                    computation_state <= 3;
                    
                when 3 => -- Output generation
                    -- Find maximum output and generate class prediction
                    output_valid <= '1';
                    computation_state <= 0;
                    
                when others =>
                    computation_state <= 0;
            end case;
        end if;
    end process;

end Behavioral;"""

    with open('quantized_nn_inference.vhd', 'w') as f:
        f.write(vhdl_entity)
    
    print("VHDL inference entity template created: quantized_nn_inference.vhd")

if __name__ == "__main__":
    print("Exporting quantized weights to VHDL format...")
    print("Checking for available model files...")
    
    # Check for improved model first, then legacy model
    model_files_to_try = [
        'Ai_training_python/improved_digit_classifier.pth',
        'improved_digit_classifier.pth',
        'Ai_training_python/digit_classifier.pth', 
        'digit_classifier.pth'
    ]
    
    success = False
    for model_file in model_files_to_try:
        if os.path.exists(model_file):
            print(f"Found model file: {model_file}")
            fc1_weights, fc2_weights, fc3_weights = export_weights_to_vhdl(model_file)
            
            if fc1_weights is not None:
                success = True
                print("\n✓ VHDL export completed successfully!")
                print("\nCreating VHDL inference entity template...")
                create_vhdl_inference_entity()
                
                print("\n" + "="*50)
                print("EXPORT SUMMARY")
                print("="*50)
                print(f"✓ Model file: {model_file}")
                print(f"✓ All weights are 8-bit signed integers")
                print(f"✓ FC1: {fc1_weights.shape[1]} → {fc1_weights.shape[0]} (input → output)")
                print(f"✓ FC2: {fc2_weights.shape[1]} → {fc2_weights.shape[0]} (input → output)")
                if fc3_weights is not None:
                    print(f"✓ FC3: {fc3_weights.shape[1]} → {fc3_weights.shape[0]} (input → output)")
                print(f"✓ Weight ranges: FC1[{fc1_weights.min()}, {fc1_weights.max()}], FC2[{fc2_weights.min()}, {fc2_weights.max()}]", end="")
                if fc3_weights is not None:
                    print(f", FC3[{fc3_weights.min()}, {fc3_weights.max()}]")
                else:
                    print()
                
                print("\nGenerated files:")
                print("  - quantized_weights.vhd (VHDL package)")
                print("  - quantized_weights_readable.txt (human-readable)")
                print("  - quantized_nn_inference.vhd (VHDL entity template)")
                break
            else:
                print(f"Failed to load weights from {model_file}")
    
    if not success:
        print("\nNo valid model files found!")
        print("\nPlease train a model first using one of these commands:")
        print("  python Ai_training_python/improved_training.py")
        print("  python Ai_training_python/main.py")
        print("\nThe script looked for these files:")
        for model_file in model_files_to_try:
            print(f"  - {model_file}")
        print("\nMake sure at least one of these files exists before running the export.")
