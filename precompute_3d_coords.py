"""
Precompute 3D coordinates for all molecules and save to .pt file.
This script should be run once before training to cache 3D coordinates.
Supports parallel processing for faster computation.
"""

import os
import csv
import numpy as np
import torch
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools

# Configuration
SMILES_FILE = 'pretrain_data.txt'
COORDS_FILE = 'precomputed_3d_coords.pt'
NUM_WORKERS = cpu_count()  # Use all available CPU cores by default


def read_smiles(data_path):
    """Read SMILES strings from CSV file."""
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            smiles = row[-1]
            smiles_data.append(smiles)
    return smiles_data


def compute_3d_coordinates(smiles):
    """
    Compute 3D coordinates for a molecule.
    Returns the conformer positions as a numpy array, or None if computation fails.
    This function is designed to be called in parallel processes.
    """
    # Initialize RDKit in each worker process
    # This is necessary because RDKit may have issues with multiprocessing
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        # Generate 3D coordinates
        try:
            # Add conformer to molecule
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            # If embedding fails, use ETKDG method as fallback
            try:
                AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                AllChem.MMFFOptimizeMolecule(mol)
            except:
                # If all methods fail, return None
                return None
        # Extract 3D coordinates
        try:
            conformer = mol.GetConformer()
        except:
            return None
        pos = []
        for i in range(conformer.GetNumAtoms()):
            position = conformer.GetAtomPosition(i)
            pos.append([position.x, position.y, position.z])
        return np.array(pos, dtype=np.float32)
    except Exception as e:
        # Return None for any unexpected errors
        return None


def compute_3d_coordinates_wrapper(args):
    """
    Wrapper function for parallel processing.
    Takes (index, smiles) tuple and returns (index, coordinates).
    """
    index, smiles = args
    coords = compute_3d_coordinates(smiles)
    return (index, coords)


def precompute_coordinates(smiles_data, output_file, num_workers=NUM_WORKERS):
    """
    Precompute 3D coordinates for all molecules and save to .pt file.
    Uses parallel processing for faster computation.
    
    Args:
        smiles_data: List of SMILES strings
        output_file: Path to save the coordinates
        num_workers: Number of parallel processes to use
    """
    print(f"Precomputing 3D coordinates for {len(smiles_data)} molecules...")
    print(f"Using {num_workers} CPU cores for parallel processing")
    
    # Prepare arguments for parallel processing
    args_list = [(idx, smiles) for idx, smiles in enumerate(smiles_data)]
    
    # Dictionary to store coordinates: index -> pos_tensor
    coords_dict = {}
    
    failed_count = 0
    success_count = 0
    
    # Process molecules in parallel
    with Pool(processes=num_workers) as pool:
        # Use tqdm with imap for progress tracking
        results = list(tqdm(
            pool.imap(compute_3d_coordinates_wrapper, args_list),
            total=len(args_list),
            desc="Processing molecules"
        ))
    
    # Collect results
    for idx, coords in results:
        if coords is None:
            failed_count += 1
            coords_dict[idx] = None
        else:
            success_count += 1
            coords_dict[idx] = torch.from_numpy(coords)
    
    # Save to .pt file
    torch.save(coords_dict, output_file)
    
    print(f"\nPrecomputation completed!")
    print(f"Successfully processed: {success_count} molecules")
    print(f"Failed to process: {failed_count} molecules")
    print(f"Total molecules: {len(smiles_data)}")
    print(f"Success rate: {success_count/len(smiles_data)*100:.2f}%")
    print(f"3D coordinates saved to: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    print("=" * 60)
    print("3D Coordinate Precomputation Script (Parallel Version)")
    print("=" * 60)
    
    # Check if SMILES file exists
    if not os.path.exists(SMILES_FILE):
        print(f"Error: SMILES file '{SMILES_FILE}' not found!")
        exit(1)
    
    # Read SMILES data
    print(f"Reading SMILES from {SMILES_FILE}...")
    smiles_data = read_smiles(SMILES_FILE)
    print(f"Loaded {len(smiles_data)} SMILES strings")
    
    # Determine number of workers
    print(f"\nAvailable CPU cores: {cpu_count()}")
    print(f"Using {NUM_WORKERS} workers for parallel processing")
    
    # Precompute coordinates with parallel processing
    precompute_coordinates(smiles_data, COORDS_FILE, num_workers=NUM_WORKERS)
    
    print("\n" + "=" * 60)
    print("Precomputation complete! You can now use the cached dataset.")
    print("=" * 60)
