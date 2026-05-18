import torch
import numpy as np
import os
import glob


def convert_to_numpy(tensor_or_array):
    """
    Convert input to numpy array.
    If it's a torch.Tensor, move to CPU and convert to numpy.
    If it's already a numpy.ndarray, return as is.
    """
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.cpu().numpy()
    elif isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array
    else:
        raise ValueError(f"Unsupported type: {type(tensor_or_array)}. Expected torch.Tensor or np.ndarray.")


def batch_convert_npy_to_numpy(result_dir, methods):
    """
    Batch process existing .npy files in result_dir for specified methods.
    Loads each file, converts 'inte' to NumPy if needed, and saves back.
    """
    for method in methods:
        npy_file = os.path.join(result_dir, method + ".npy")
        if os.path.exists(npy_file):
            print(f"Processing {npy_file}...")
            # Load the result
            result = np.load(npy_file, allow_pickle=True).item()

            # Convert 'inte' entries
            if 'inte' in result:
                rna_inte = convert_to_numpy(result['inte'][0])
                atac_inte = convert_to_numpy(result['inte'][1])
                result['inte'] = [rna_inte, atac_inte]
                print(f"  Converted 'inte' for {method}: tensor -> numpy")

            # Save back to the same file (overwrite)
            np.save(npy_file, result)
            print(f"  Saved updated {npy_file}")
        else:
            print(f"File not found: {npy_file}")


# Example usage
if __name__ == "__main__":
    # Customize these paths and methods
    result_dir = "path/to/your/result_dir"  # e.g., "./results/P0BraCor"
    methods = ['scGSI', 'MMDMA', 'JointMDS', 'scMGCL', 'other_method']  # List your methods

    batch_convert_npy_to_numpy(result_dir, methods)
    print("Batch conversion complete! Now load_result will skip tensor conversions.")