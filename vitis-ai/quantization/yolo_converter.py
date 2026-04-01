import torch
import sys
import os

def model_conversion(model, activation_function='hardswish'):
    """
    Convert SiLU activation functions to the specified activation function
    
    Args:
        model: PyTorch model
        activation_function: str, one of ['leakyrelu', 'hardswish']
    """
    activation_map = {
        'leakyrelu': torch.nn.LeakyReLU,
        'hardswish': torch.nn.Hardswish
    }
    
    if activation_function not in activation_map:
        raise ValueError(f"Unsupported activation function: {activation_function}. "
                        f"Choose from {list(activation_map.keys())}")
    
    activation_class = activation_map[activation_function]
    
    def _iter(module, prefix=''):
        for name, sub_module in module.named_children():
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(sub_module, torch.nn.SiLU) and sub_module.__class__.__name__ == "SiLU":
                if activation_function == 'leakyrelu':
                    setattr(module, name, activation_class(0.1015625))
                else:
                    setattr(module, name, activation_class())
            _iter(sub_module, prefix=full)
    _iter(model)


if __name__ == "__main__":
    if len(sys.argv) not in [4, 5]: 
        print("Use: python YOLO_converter.py <path_to>/best.pt <output_dir> <my_model_name> [activation_function]")
        print("Available activation functions: relu, hardsigmoid, hardswish (default: hardswish)")
    else:
        # Get activation function from command line or use default
        activation_function = sys.argv[4] if len(sys.argv) == 5 else 'hardswish'
        
        # Validate activation function
        valid_activations = ['leakyrelu', 'hardswish']
        if activation_function not in valid_activations:
            print(f"Error: Invalid activation function '{activation_function}'. "
                  f"Choose from {valid_activations}")
            sys.exit(1)
            
        print(f"Converting model with activation function: {activation_function}")
        
        # Load YOLO Detection model
        model = torch.load(sys.argv[1], weights_only=False)

        model = model["model"]
        model = model.float().to("cpu")

        # Convert SiLU to specified activation function
        model_conversion(model, activation_function)
        torch.save(model, os.path.join(sys.argv[2], f"{sys.argv[3]}.pt"))
        print(f"Model successfully converted and saved with {activation_function} activation!")
