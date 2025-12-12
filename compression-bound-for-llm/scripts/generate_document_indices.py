#!/usr/bin/env python3
"""
Generate document indices files for bound computation.

This script creates two numpy arrays needed for document-level bound evaluation:
1. openwebtext_train_eot_indices_file.npy - indices of EOT tokens separating documents
2. empirical_document_length_distribution_file.npy - lengths of each document

Usage:
    python scripts/generate_document_indices.py --data_dir ./data/openwebtext --eot_token 50256
"""

import numpy as np
import argparse
import os
from tqdm import tqdm


def generate_document_indices(train_bin_path, eot_token, output_dir):
    """
    Generate document indices and length distribution files.
    
    Args:
        train_bin_path: Path to train.bin file
        eot_token: Token ID for end-of-text token (50256 for GPT-2)
        output_dir: Directory to save the output files
    """
    print(f"Loading training data from {train_bin_path}...")
    train_data = np.memmap(train_bin_path, dtype=np.uint16, mode='r')
    print(f"Loaded data with {len(train_data):,} tokens")
    
    print(f"Finding all EOT tokens (token ID: {eot_token})...")
    # Find all positions where EOT token appears
    eot_indices = np.where(train_data == eot_token)[0]
    print(f"Found {len(eot_indices):,} EOT tokens (documents)")
    
    # Compute document lengths
    print("Computing document lengths...")
    eot_indices_shift_left_by_1 = np.insert(eot_indices[:-1], 0, 0)
    document_lengths = eot_indices - eot_indices_shift_left_by_1
    
    # Save files
    os.makedirs(output_dir, exist_ok=True)
    
    eot_indices_file = os.path.join(output_dir, 'openwebtext_train_eot_indices_file.npy')
    doc_lengths_file = os.path.join(output_dir, 'empirical_document_length_distribution_file.npy')
    
    print(f"Saving EOT indices to {eot_indices_file}...")
    np.save(eot_indices_file, eot_indices)
    
    print(f"Saving document lengths to {doc_lengths_file}...")
    np.save(doc_lengths_file, document_lengths)
    
    # Print statistics
    print("\n=== Statistics ===")
    print(f"Total documents: {len(eot_indices):,}")
    print(f"Document length statistics:")
    print(f"  Min: {document_lengths.min():,} tokens")
    print(f"  Max: {document_lengths.max():,} tokens")
    print(f"  Mean: {document_lengths.mean():.2f} tokens")
    print(f"  Median: {np.median(document_lengths):.2f} tokens")
    print(f"  Std: {document_lengths.std():.2f} tokens")
    
    print("\n=== Files created ===")
    print(f"1. {eot_indices_file}")
    print(f"2. {doc_lengths_file}")
    
    return eot_indices_file, doc_lengths_file


def main():
    parser = argparse.ArgumentParser(description='Generate document indices for bound computation')
    parser.add_argument('--data_dir', type=str, default='./data/openwebtext',
                       help='Directory containing train.bin file')
    parser.add_argument('--eot_token', type=int, default=50256,
                       help='End-of-text token ID (default: 50256 for GPT-2)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (defaults to same as data_dir)')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.data_dir
    
    # Construct path to train.bin
    train_bin_path = os.path.join(args.data_dir, 'train.bin')
    
    if not os.path.exists(train_bin_path):
        print(f"Error: {train_bin_path} not found!")
        print(f"Please ensure train.bin exists in {args.data_dir}")
        return
    
    # Generate files
    generate_document_indices(train_bin_path, args.eot_token, args.output_dir)


if __name__ == '__main__':
    main()
