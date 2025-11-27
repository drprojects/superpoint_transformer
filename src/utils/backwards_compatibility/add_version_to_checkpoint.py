"""
Script to add version metadata to an existing checkpoint.

Usage:
    python -m src.utils.backwards_compatibility.add_version_to_checkpoint path/to/checkpoint.ckpt
    python -m src.utils.backwards_compatibility.add_version_to_checkpoint path/to/checkpoint.ckpt --version 2.1.0
    python -m src.utils.backwards_compatibility.add_version_to_checkpoint path/to/checkpoint.ckpt --output new_checkpoint.ckpt
"""

import argparse
import torch
from pathlib import Path


def add_version_to_checkpoint(
    checkpoint_path: str,
    version: str = '3.0.0',
    output_path: str = None
):
    """
    Add version metadata to a checkpoint file.
    
    :param checkpoint_path: Path to the input checkpoint
    :param version: Version string to add (default: '3.0.0')
    :param output_path: Path for the modified checkpoint. If None, overwrites
        the input checkpoint
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Initialize metadata if it doesn't exist
    if 'metadata' not in checkpoint:
        checkpoint['metadata'] = {}
    
    # Check if version already exists
    existing_version = checkpoint['metadata'].get('__version__', None)
    if existing_version:
        print(f"⚠️  Warning: Checkpoint already has version '{existing_version}'")
        response = input(f"Do you want to overwrite it with '{version}'? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Add/update version and commit hash
    checkpoint['metadata']['__version__'] = version
    
    # Add commit hash if not present
    if 'commit_hash' not in checkpoint['metadata']:
        checkpoint['metadata']['commit_hash'] = 'unknown'
        print(f"✓ Added metadata['commit_hash'] = 'unknown'")
    
    print(f"✓ Added metadata['__version__'] = '{version}'")
    
    # Determine output path
    if output_path is None:
        output_path = checkpoint_path
        print(f"Overwriting original checkpoint...")
    else:
        output_path = Path(output_path)
        if output_path.exists():
            response = input(f"Output file exists. Overwrite? (y/N): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
    
    # Save modified checkpoint
    print(f"Saving checkpoint to: {output_path}")
    torch.save(checkpoint, output_path)
    print(f"✓ Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Add version metadata to a PyTorch Lightning checkpoint")
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the checkpoint file")
    parser.add_argument(
        "--version",
        type=str,
        default='3.0.0',
        help="Version string to add (default: 3.0.0)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the modified checkpoint (default: overwrite input)")
    
    args = parser.parse_args()
    
    try:
        add_version_to_checkpoint(
            args.checkpoint_path,
            version=args.version,
            output_path=args.output
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

