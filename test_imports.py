#!/usr/bin/env python3
"""
Test script to check what's causing the hang
"""

import csv
import math
import random
from collections import defaultdict, Counter
import datetime

print("Testing basic imports...")
print("All standard library imports successful")

# Test numpy import
try:
    import numpy as np
    print("Numpy is available")
except ImportError:
    print("Numpy is NOT available - this was likely the issue")

# Quick test of data loading
try:
    with open('raw_data_v2.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        # Just read first few rows to test
        data = []
        for i, row in enumerate(reader):
            data.append(row)
            if i >= 10:  # Just test first 10 rows
                break
    
    print(f"Data loading test successful: {len(header)} columns")
except Exception as e:
    print(f"Data loading failed: {e}")

print("Test complete")