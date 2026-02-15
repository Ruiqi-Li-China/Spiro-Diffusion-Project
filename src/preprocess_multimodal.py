import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os

class SpiroPreprocessor:
    def __init__(self, target_length=512):
        self.target_length = target_length # Required for VQ-VAE input [cite: 59, 74]

    def parse_raw_curve(self, raw_bytes):
        """Decodes NHANES byte string to NumPy array[cite: 172]."""
        if isinstance(raw_bytes, bytes):
            try:
                decode_str = raw_bytes.decode('utf-8')
                return np.fromstring(decode_str, sep=',')
            except:
                return None
        return None

    def resample_signal(self, signal):
        """Standardizes signal length to 512 for VQ-VAE[cite: 59]."""
        x_old = np.linspace(0, 1, len(signal))
        x_new = np.linspace(0, 1, self.target_length)
        # Convert mL/s to L/s for physiological consistency [cite: 102, 121]
        return interp1d(x_old, signal / 1000.0, kind='linear')(x_new)

    def run(self):
        # Paths confirmed from your project structure
        raw_path = "data/raw/nhanes/SPXRAW_G/spxraw_g.sas7bdat"
        demo_path = "data/raw/nhanes/DEMO_G.xpt"
        bmx_path = "data/raw/nhanes/BMX_G.xpt"
        
        print("--- Phase 1: Multimodal Data Alignment ---")
        
        try:
            # 1. Load Raw Signals
            print("Reading Raw Signals...")
            raw_df = pd.read_sas(raw_path, format='sas7bdat')
            raw_df['SEQN'] = raw_df['SEQN'].astype(int)

            # 2. Load Metadata (Native pandas reader for XPORT)
            print("Reading Demographics (DEMO_G)...")
            demo_df = pd.read_sas(demo_path, format='xport')
            demo_df['SEQN'] = demo_df['SEQN'].astype(int)

            print("Reading Body Measures (BMX_G)...")
            bmx_df = pd.read_sas(bmx_path, format='xport')
            bmx_df['SEQN'] = bmx_df['SEQN'].astype(int)

            # 3. Multimodal Merge: Age (RIDAGEYR), Gender (RIAGENDR), Height (BMXHT) [cite: 31, 39]
            metadata = pd.merge(demo_df[['SEQN', 'RIDAGEYR', 'RIAGENDR']], 
                                bmx_df[['SEQN', 'BMXHT']], on='SEQN')
            
            merged_df = pd.merge(raw_df, metadata, on='SEQN', how='inner')
            print(f"Successfully merged {len(merged_df)} records.")
            
        except Exception as e:
            print(f"Error during data loading: {e}")
            return

        processed_signals = []
        aligned_metadata = []

        for _, row in merged_df.iterrows():
            flow_array = self.parse_raw_curve(row['SPXRAW'])
            
            # Threshold to ensure signal quality for VQ-VAE [cite: 37, 51]
            if flow_array is not None and len(flow_array) > 100:
                resampled = self.resample_signal(flow_array)
                processed_signals.append(resampled)
                
                aligned_metadata.append({
                    'SEQN': int(row['SEQN']),
                    'age': float(row['RIDAGEYR']),
                    'height': float(row['BMXHT']),
                    'gender': int(row['RIAGENDR'])
                })
        
        if processed_signals:
            os.makedirs('data/processed', exist_ok=True)
            # Save for Phase 1: VQ-VAE Representation Learning [cite: 59]
            np.save('data/processed/signals_L512.npy', np.array(processed_signals))
            # Save for Phase 2: Diffusion Guidance [cite: 39, 89]
            pd.DataFrame(aligned_metadata).to_csv('data/processed/metadata_aligned.csv', index=False)
            print(f"Success! Processed {len(processed_signals)} signals.")
        else:
            print("No valid signals found after parsing.")

if __name__ == "__main__":
    SpiroPreprocessor().run()
