import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_spiro_raw(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None
    try:
        print(f"Reading data...")
        # We only need a few columns for the plot
        df = pd.read_sas(filepath, format='sas7bdat')
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None

def parse_raw_curve(raw_string):
    """
    Converts the byte string '0,0,1...' into a NumPy array of integers.
    """
    # Decode bytes to string, then split by comma and convert to int
    decode_str = raw_string.decode('utf-8')
    return np.fromstring(decode_str, sep=',')

if __name__ == "__main__":
    raw_path = "data/raw/nhanes/SPXRAW_G/spxraw_g.sas7bdat"
    df = load_spiro_raw(raw_path)
    
    if df is not None:
        # Let's pick the first valid-looking curve (index 0)
        first_row_raw = df.iloc[0]['SPXRAW']
        flow_array = parse_raw_curve(first_row_raw)
        
        # Calculate Volume by integrating Flow over time (Time step = 0.01s)
        # Flow is in mL/s, so we divide by 1000 to get Liters
        volume_array = np.cumsum(flow_array) * 0.01 / 1000 
        flow_l_s = flow_array / 1000
        
        # Plotting the Flow-Volume Loop
        plt.figure(figsize=(10, 6))
        plt.plot(volume_array, flow_l_s, color='blue', label='Exhalation')
        
        plt.title(f"Flow-Volume Curve (Subject: {df.iloc[0]['SEQN']})")
        plt.xlabel("Volume (L)")
        plt.ylabel("Flow (L/s)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        print("Displaying plot... close the window to continue.")
        plt.show()