import os
import urllib.request

def download_nhanes_file(url, destination):
    if not os.path.exists(destination):
        print(f"Downloading {os.path.basename(url)}...")
        urllib.request.urlretrieve(url, destination)
        print("Download complete.")
    else:
        print(f"File already exists: {destination}")

# Ensure the directory exists
raw_dir = "data/raw/nhanes/"
os.makedirs(raw_dir, exist_ok=True)

# URLs for Cycle G (2011-2012)
files = {
    "DEMO_G.xpt": "https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/DEMO_G.XPT",
    "BMX_G.xpt": "https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/BMX_G.XPT"
}

for filename, url in files.items():
    download_nhanes_file(url, os.path.join(raw_dir, filename))

print("\nAll metadata files are ready in data/raw/nhanes/")
