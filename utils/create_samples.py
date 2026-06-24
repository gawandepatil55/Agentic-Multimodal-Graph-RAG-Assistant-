import os
from pathlib import Path
from PIL import Image, ImageDraw

def create_sample_dataset():
    # Define and create output directory
    data_dir = Path(__file__).resolve().parent.parent / "sample_data"
    data_dir.mkdir(exist_ok=True)
    
    print(f"Creating sample dataset in: {data_dir}")
    
    # 1. Write medical report sample
    report_path = data_dir / "medical_imaging_report.txt"
    report_text = """CLINICAL REPORT: MRI SCAN OF LOWER EXTREMITIES
--------------------------------------------------
PATIENT NAME: Jane Doe
AGE: 48
DIAGNOSIS: Type 2 Diabetes mellitus (since 2018), Peripheral Neuropathy.
DATE OF SCAN: 2026-05-12
PROCEDURE: MRI Scan of the left foot and ankle with contrast.
CLINICAL INDICATIONS: Persistent diabetic neuropathy, suspected osteomyelitis.

FINDINGS:
1. Significant inflammation in the soft tissues of the lower left ankle.
2. The MRI Scan shows no active signs of osteomyelitis (bone infection) at this time.
3. Minor neuropathic joint changes consistent with Charcot arthropathy.

INTERPRETATION:
The MRI Scan findings were analyzed by Dr. Sarah Jenkins. 
The analysis was processed using the Medical Imaging Pipeline (MIP-v2).
Dr. Jenkins recommends strict glycemic control and specialized diabetic footwear.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text.strip())
    print(f"Generated text report: {report_path.name}")
    
    # 2. Write project team logistics sample
    project_path = data_dir / "project_logistics.txt"
    project_text = """ORGANIZATION REPORT: R&D ENGINEERING PROJECTS
--------------------------------------------------
PROJECT: Project Medical Imaging (Code: MIP-v2)
MISSION: Build next-generation deep learning model pipelines to assist clinical radiologists.
DEPARTMENT: Applied Medical AI, Biotech Corp.

TEAM ROLES & RESPONSIBILITIES:
- Marcus Vance: Lead ML Research Engineer. Marcus Vance designed the core convolutional segmentation models used to analyze MRI Scans in MIP-v2.
- Lisa Chen: Senior Systems Integration Engineer. Lisa Chen handles the integration of clinical DICOM images into the MIP-v2 pipeline.
- Dr. Sarah Jenkins: Medical Advisor and Consulting Radiologist. Dr. Jenkins validates the algorithmic outputs of Project Medical Imaging against real clinical cases.

LOGISTICS:
The project is funded by the Biotech Corp Innovations Grant. All development is tracked under the MIP-v2 repository.
"""
    with open(project_path, "w", encoding="utf-8") as f:
        f.write(project_text.strip())
    print(f"Generated text report: {project_path.name}")
    
    # 3. Create a dummy medical chart image using Pillow
    img_path = data_dir / "sample_mri_chart.png"
    
    # Create dark-themed medical diagram
    img = Image.new('RGB', (600, 400), color='#1e222b')
    draw = ImageDraw.Draw(img)
    
    # Draw header bar
    draw.rectangle([0, 0, 600, 50], fill='#2e3440')
    draw.text((20, 18), "MIP-v2: MRI Scan Segment Analysis", fill='#88c0d0')
    
    # Draw diagnostic bounding boxes
    draw.rectangle([100, 120, 300, 320], outline='#bf616a', width=3)
    draw.text((105, 125), "Suspected Soft Tissue Inflammation", fill='#bf616a')
    
    # Draw mock chart elements (circles/lines)
    draw.ellipse([150, 180, 250, 280], outline='#eceff4', width=2) # Represents scan subject
    draw.line([250, 230, 420, 230], fill='#e5e9f0', width=2)
    draw.line([420, 230, 420, 200], fill='#e5e9f0', width=2)
    
    # Text data blocks
    draw.rectangle([350, 80, 550, 180], fill='#282c34', outline='#4b5263')
    draw.text((360, 90), "PATIENT: Jane Doe", fill='#e5e9f0')
    draw.text((360, 110), "CONDITION: Diabetic Ankle", fill='#a3be8c')
    draw.text((360, 130), "STATUS: Completed Scan", fill='#ebcb8b')
    draw.text((360, 150), "ANALYST: Dr. Sarah Jenkins", fill='#b48ead')
    
    # Footer
    draw.text((20, 370), "CONFIDENTIAL - MEDICAL AI RESEARCH DATA ONLY", fill='#d8dee9')
    
    img.save(img_path)
    print(f"Generated chart image: {img_path.name}")

if __name__ == "__main__":
    create_sample_dataset()
