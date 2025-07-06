import torch
import os
import sys

# Test the improved export_to_vhdl.py functionality

def test_export_functionality():
    print("Testing improved export_to_vhdl.py functionality...")
    
    # Import the updated export function
    sys.path.append('.')
    from export_to_vhdl import load_quantized_weights_from_model, export_weights_to_vhdl
    
    # Check for available model files
    model_files = [
        'Ai_training_python/improved_digit_classifier.pth',
        'improved_digit_classifier.pth',
        'Ai_training_python/digit_classifier.pth',
        'digit_classifier.pth'
    ]
    
    found_model = None
    for model_file in model_files:
        if os.path.exists(model_file):
            found_model = model_file
            print(f"✓ Found model file: {model_file}")
            break
    
    if found_model is None:
        print("❌ No model files found. Creating a dummy test...")
        
        # Create a dummy improved model for testing
        from test_model import ImprovedDigitClassifier
        
        print("Creating dummy improved model...")
        model = ImprovedDigitClassifier(use_quantized=True, bits=8)
        
        # Save the dummy model
        dummy_path = 'test_improved_model.pth'
        torch.save(model.state_dict(), dummy_path)
        print(f"✓ Dummy model saved to {dummy_path}")
        
        # Test loading weights from dummy model
        weights, model_type = load_quantized_weights_from_model(dummy_path)
        if weights is not None:
            print(f"✓ Successfully loaded {model_type} model weights")
            print(f"  - Available layers: {list(weights.keys())}")
            for layer_name, layer_weights in weights.items():
                print(f"  - {layer_name}: {layer_weights.shape}")
        else:
            print("❌ Failed to load weights from dummy model")
        
        # Clean up
        if os.path.exists(dummy_path):
            os.remove(dummy_path)
            print(f"✓ Cleaned up {dummy_path}")
    else:
        # Test with real model
        print(f"Testing export with real model: {found_model}")
        fc1, fc2, fc3 = export_weights_to_vhdl(found_model, 'test_export.vhd')
        
        if fc1 is not None:
            print("✓ Export test successful!")
            print(f"  - FC1 shape: {fc1.shape}")
            print(f"  - FC2 shape: {fc2.shape}")
            if fc3 is not None:
                print(f"  - FC3 shape: {fc3.shape}")
            
            # Check if files were created
            files_to_check = ['test_export.vhd', 'test_export_readable.txt']
            for file_path in files_to_check:
                if os.path.exists(file_path):
                    print(f"✓ Generated: {file_path}")
                    # Clean up test files
                    os.remove(file_path)
                else:
                    print(f"❌ Missing: {file_path}")
        else:
            print("❌ Export test failed")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_export_functionality()
