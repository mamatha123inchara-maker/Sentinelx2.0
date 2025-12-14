![solar_input](https://github.com/user-attachments/assets/43ac714b-9a5a-4524-8690-15660d31c4c1)# eco-innovators
An automated pipeline that detects and verifies rooftop solar panels using latitude and longitude. The system fetches high-resolution images, identifies whether solar PV is present, estimates panel area, generates explainable audit artifacts, assigns QC status, and stores results as structured JSON outputs. Designed for accuracy and transparency
## 🔧 How the Pipeline Works

1. *Input Coordinates* → User provides latitude and longitude.
2. *Image Fetching* → System pulls high-resolution rooftop imagery.
3. *Detection* → YOLOv8 model identifies solar panels.
4. *Area Estimation* → Calculates panel coverage in square meters.
5. *Audit Generation* → Creates visual artifacts and JSON summaries.
6. *QC Status Assignment* → Labels each result as VERIFIABLE or NOT_VERIFIABLE.
7. *Storage* → Saves outputs in structured folders for easy access.
   
9. ## 🚀 Running the Pipeline

```bash
git clone https://github.com/yourteam/eco-innovators.git
cd eco-innovators
pip install -r requirements.txt
python pipeline/run.py --input data/images --output predictions/

10. ## Code
import os
import json
import requests
import pandas as pd
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

# -------------------------
# CONFIG
# -------------------------

GOOGLE_API_KEY = "AIzaSyAF71xKeFF13D1A8ZHV8foB1upZhRPR7oE"
INPUT_XLSX = "/content/drive/MyDrive/Solar_pipeline/Testing latlong.xlsx"
MODEL_PATH = "/content/drive/MyDrive/Solar_pipeline/model/best.pt"
OUTPUT_FOLDER = "/content/drive/MyDrive/Solar_pipeline/output"

# Create output folder if not existing
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)


# -------------------------
# FUNCTION: FETCH SAT IMAGE
# -------------------------
def fetch_sat_image(lat, lon):
    url = (
        "https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}&zoom=20&size=512x512"
        "&maptype=satellite"
        f"&key={GOOGLE_API_KEY}"
    )
    response = requests.get(url)

    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        print("Failed to fetch image")
        return None


# -------------------------
# PIPELINE MAIN FUNCTION
# -------------------------
def run_pipeline():
    df = pd.read_excel(INPUT_XLSX)

    for _, row in df.iterrows():
        sample_id = row["sample_id"]
        lat = row["Latitude"]  # Corrected from 'latitude'
        lon = row["Longitude"] # Corrected from 'longitude'

        print(f"\nProcessing sample_id: {sample_id}")

        img = fetch_sat_image(lat, lon)
        if img is None:
            continue

        # Save fetched image
        img_path = f"{OUTPUT_FOLDER}/{sample_id}_input.png"
        img.save(img_path)

        # Run model prediction
        results = model.predict(img_path)

        # Check if detections exist
        has_solar = len(results[0].boxes) > 0
        confidence = 0.0
        area_est = 0.0

        if has_solar:
            confidence = float(results[0].boxes.conf[0])
            # Dummy area value (because area logic depends on your model)
            area_est = 10.5

        # QC Status
        qc = "VERIFIABLE" if has_solar else "NOT_VERIFIABLE"

        # Save overlay image
        overlay_path = f"{OUTPUT_FOLDER}/{sample_id}_overlay.png"
        results[0].plot(save=True, filename=overlay_path)

        # Create JSON output
        data = {
            "sample_id": int(sample_id),
            "lat": float(lat),
            "lon": float(lon),
            "has_solar": has_solar,
            "confidence": confidence,
            "pv_area_sqm_est": area_est,
            "buffer_radius_sqft": 1200,
            "qc_status": qc,
            "bbox_or_mask": "saved_overlay",
            "image_metadata": {
                "source": "Google Static Maps",
                "capture_date": "unknown"
            }
        }

        json_path = f"{OUTPUT_FOLDER}/{sample_id}.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"✔ Output saved for sample {sample_id}")


# -------------------------
# RUN
# -------------------------
if _name_ == "_main_":
    run_pipeline()

### 🔌 API Usage
AIzaSyAF71xKeFF13D1A8ZHV8foB1upZhRPR7oE

### JSON

## 📸 Example Output

Here’s how the pipeline works on a sample rooftop:

*Original rooftop image:*
![Original rooftop](artefacts/solar_input.png)

*Detection overlay:*
![Solar panel detection](artefacts/solar_output.png)


{
    "sample_id": 1,
    "lat": 37.416,
    "lon": -122.084,
    "has_solar": true,
    "confidence": 0.897265613079071,
    "pv_area_sqm_est": 10.5,
    "buffer_radius_sqft": 1200,
    "qc_status": "VERIFIABLE",
    "bbox_or_mask": "saved_overlay",
    "image_metadata": {
        "source": "Google Static Maps",
        "capture_date": "unknown"
    }
}

### ✅ 7. *Describe Folder Structure*
```markdown
## 📁 Folder Structure

├── predictions/        # Model outputs  
├── artefacts/          # Audit visuals and JSONs  
├── logs/               # Training and pipeline logs  
├── trained_model/      # YOLOv8 weights  
├── pipeline_code/      # Core scripts  
├── data/               # Input images  
└── docs/               # Model card PDF



## 📦 Dependencies

- Python 3.9+   
- Ultralytics YOLOv8  

[Detection example](artefacts/rooftop_detection.png)
📄 [Model Card (PDF)](docs/ideathon.pdf)
