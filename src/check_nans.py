import numpy as np
import pandas as pd

def check_data():
    print("--- Checking for NaNs ---")
    
    # 1. Check Latents
    try:
        latents = np.load('data/processed/latents.npy')
        if np.isnan(latents).any():
            print(f"!! CRITICAL: Found {np.isnan(latents).sum()} NaNs in latents.npy")
        elif np.isinf(latents).any():
            print(f"!! CRITICAL: Found Infinity values in latents.npy")
        else:
            print("OK: latents.npy is clean.")
    except Exception as e:
        print(f"Error loading latents: {e}")

    # 2. Check Metadata
    try:
        df = pd.read_csv('data/processed/metadata_aligned.csv')
        # Check specific columns used for training
        cols = ['age', 'height', 'gender']
        if df[cols].isnull().values.any():
            print(f"!! CRITICAL: Found missing values in metadata:")
            print(df[cols].isnull().sum())
        else:
            print("OK: metadata_aligned.csv is clean.")
            
    except Exception as e:
        print(f"Error loading metadata: {e}")

if __name__ == "__main__":
    check_data()
