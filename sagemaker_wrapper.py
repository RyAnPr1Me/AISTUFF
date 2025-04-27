#!/usr/bin/env python3
"""
SageMaker wrapper script to ensure compatibility and proper error handling
"""

import os
import sys
import logging
import traceback
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def run_training():
    """Run the training script with proper error handling"""
    try:
        # Get command line arguments and pass them to train_model.py
        args = sys.argv[1:] if len(sys.argv) > 1 else []
        
        # Add default SageMaker paths if not specified
        if '--input-data' not in ' '.join(args):
            args.extend(['--input-data', '/opt/ml/input/data/train/optimized_data.csv'])
        
        # Construct command
        cmd = [sys.executable, 'train_model.py'] + args
        logging.info(f"Running command: {' '.join(cmd)}")
        
        # Execute the training script
        result = subprocess.run(cmd, check=True)
        logging.info(f"Training completed with exit code: {result.returncode}")
    
    except subprocess.CalledProcessError as e:
        logging.error(f"Training failed with exit code: {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        logging.error(f"Error running training: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_training()
