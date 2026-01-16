"""
Quick script to check and fix version compatibility issues
Run this before starting the Flask app
"""

import sys
import subprocess

def check_and_fix_versions():
    print("Checking package versions...")
    
    try:
        import numpy as np
        import sklearn
        print(f"Current NumPy version: {np.__version__}")
        print(f"Current scikit-learn version: {sklearn.__version__}")
        
        # Check if numpy is 2.x (incompatible)
        if np.__version__.startswith('2.'):
            print("\n⚠️  NumPy 2.x detected - incompatible with pickle files!")
            print("The pickle files were created with NumPy 1.x")
            print("\nRecommended fix:")
            print("  pip install 'numpy<2.0' 'scikit-learn>=1.0.0,<1.4'")
            return False
        else:
            print("✓ NumPy version looks compatible")
            return True
            
    except ImportError as e:
        print(f"Error importing packages: {e}")
        return False

if __name__ == "__main__":
    if not check_and_fix_versions():
        print("\nPlease run the recommended pip install command above")
        sys.exit(1)
    else:
        print("\n✓ Versions are compatible!")
