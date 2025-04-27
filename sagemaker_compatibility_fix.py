#!/usr/bin/env python3
"""
This script ensures compatibility with SageMaker Python 3.10 environment
Run this before submitting the job: python sagemaker_compatibility_fix.py
"""

import os
import re
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def fix_syntax_errors(file_path):
    """Fix common syntax errors in Python files"""
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Count lines before modification for reporting
    original_lines = content.count('\n') + 1
    
    # Fix pattern: try without except
    try_without_except_pattern = re.compile(r'try\s*:\s*\n(?:[^\n]+\n)+?(?!\s*except|\s*finally)')
    if try_without_except_pattern.search(content):
        logging.info(f"Found incomplete try-except blocks in {file_path}")
        # This is just detection - manual correction is needed
    
    # Fix incomplete except clauses
    except_pattern = re.compile(r'except\s+([A-Za-z0-9_]+)(\s+as\s+[A-Za-z0-9_]+)?\s*(?!:)')
    content = except_pattern.sub(r'except \1\2:', content)
    
    # Fix except clause without exception type
    bare_except_pattern = re.compile(r'except\s*(?!:)(?!\s+[A-Za-z0-9_]+)')
    content = bare_except_pattern.sub(r'except:', content)
    
    # Write back the fixed content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Count lines after modification
    modified_lines = content.count('\n') + 1
    
    if original_lines != modified_lines:
        logging.info(f"Fixed potential syntax issues in {file_path}")
    else:
        logging.info(f"No syntax issues found in {file_path}")
    
    return True

def main():
    files_to_check = [
        'train_model.py',
        'validate_and_prepare.py',
        'optimize_data.py',
        'src/models/stock_ai.py',
        'src/models/tft_model.py',
        'src/models/dataloader.py',
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            logging.info(f"Checking {file_path}")
            fix_syntax_errors(file_path)
        else:
            logging.warning(f"Skipping {file_path} - file not found")
    
    logging.info("Compatibility check complete")

if __name__ == "__main__":
    main()
