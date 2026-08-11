# Configurable OCR Extraction System for Pakistani CNIC and Passport Documents

A production-oriented Python OCR and document-extraction system built on top of **PaddleOCR** and an **Adaptive Document Image Preprocessing Pipeline**, designed specifically for **Pakistani CNIC (Front & Back, Bilingual Urdu + English)** and **Pakistani Passports (Biodata & MRZ)**.

---

## Architecture Overview

```text
Input Image
     │
     ▼
Adaptive Preprocessing Pipeline
(Perspective Warp + Deskew + Illumination + Denoise)
     │
     ▼
Document Type Routing
     │
     ├───────────────┐
     ▼               ▼
CNIC Pipeline    Passport Pipeline
     │               │
     ▼               ▼
  PaddleOCR       PaddleOCR
     │               │
     └───────┬───────┘
             ▼
      Raw OCR Tokens
   + text & confidence
   + [x1, y1, x2, y2] bbox
   + page / index
             │
             ▼
    Spatial Field Parser
 (Normalized 0-1 Regions & Directional Anchors)
             │
             ▼
       Normalization
   (CNIC: XXXXX-XXXXXXX-X, Date: DD.MM.YYYY)
             │
             ▼
        Validation
   (Checksums & Format Verification)
             │
             ▼
     Structured JSON
+ Raw OCR Token Storage & Debug Overlay Image
```

---

## Key Design Principles

1. **Decoupled Architecture**:
   - PaddleOCR produces raw, structured `OCRToken` bounding boxes.
   - Field parsers and YAML configurations (`cnic_front.yaml`, `cnic_back.yaml`, `passport.yaml`) determine field mappings independently. No document rules are hard-coded into the OCR engine.

2. **Native Bilingual Representation**:
   - Preserves Urdu text natively as Unicode without translation:
     ```json
     {
       "name": {
         "en": "MUHAMMAD AHMAD",
         "ur": "محمد احمد"
       }
     }
     ```

3. **Deterministic Spatial Extraction (Regions & Anchors)**:
   - **Fixed Normalized Regions**: Uses normalized coordinates (0.0 to 1.0) so extraction works across different image resolutions.
   - **Directional Anchors**: Locates label tokens (e.g. `"Name"`, `"Date of Birth"`) and extracts adjacent text (`right`, `below`, `left`, `above`).

4. **Traceability & Raw OCR Persistence**:
   - Every extracted field retains token-level confidence scores, combined bounding boxes, and constituent tokens.
   - Raw OCR tokens are stored separately so field parsers can be re-run without re-executing PaddleOCR.

5. **Field-Specific Normalization & Validation**:
   - Normalizes CNIC numbers (`12345 1234567 1` $\rightarrow$ `12345-1234567-1`), dates (`15/08/1990` $\rightarrow$ `15.08.1990`), and passport numbers (`AB1234567`).
   - Marks fields with `validated: true/false` badges without discarding raw values.

6. **Interactive Calibration & Debug Visualization**:
   - Generates annotated debug images showing original document, raw OCR bounding boxes, configured regions, extracted text, and validation status badges.
   - Includes CLI calibration tool (`calibrate-region`) and REST API endpoints (`POST /config/region`).

---

## Directory Structure

```text
OCR/
├── app/
│   ├── main.py                       # Unified FastAPI Web API & CLI entrypoint
│   ├── cli.py                        # Standalone CLI tool
│   │
│   ├── preprocessing/                # Adaptive document image preprocessor
│   │   ├── pipeline.py
│   │   ├── quality.py
│   │   ├── geometry.py
│   │   ├── orientation.py
│   │   ├── illumination.py
│   │   ├── contrast.py
│   │   ├── denoise.py
│   │   ├── mrz.py
│   │   └── variants.py
│   │
│   ├── ocr/                          # Reusable OCR Layer
│   │   ├── models.py                 # OCRToken & RawOCRResult dataclasses
│   │   ├── paddle.py                 # PaddleOCRAdapter returning OCRTokens
│   │   └── evaluator.py              # Candidate variant evaluator
│   │
│   ├── extraction/                   # Spatial Extraction & Normalization
│   │   ├── regions.py                # Normalized coordinate (0-1) intersection
│   │   ├── anchors.py                # Directional anchor extraction
│   │   ├── reading_order.py          # Top-to-bottom, left-to-right sorting
│   │   ├── normalization.py          # Field normalizers (CNIC, Date, Passport)
│   │   └── config_tool.py            # Region calibration helper
│   │
│   ├── documents/                    # Document Pipelines & Parsers
│   │   ├── base.py
│   │   ├── cnic/
│   │   │   ├── pipeline.py           # CNICPipeline (Front & Back)
│   │   │   ├── parser.py             # Bilingual CNICParser
│   │   │   ├── profiles.py           # Evaluator profiles
│   │   │   └── validators.py         # CNIC validators
│   │   │
│   │   └── passport/
│   │       ├── pipeline.py           # PassportPipeline
│   │       ├── parser.py             # Passport & MRZ Parser
│   │       ├── profiles.py           # Passport profiles
│   │       └── validators.py         # Passport & MRZ validators
│   │
│   ├── visualization/
│   │   └── debug.py                  # Debug annotation overlay generator
│   │
│   └── api/                          # FastAPI REST API
│       ├── routes.py                 # POST /ocr, POST /ocr/debug, POST /config/region
│       └── models.py                 # API Schemas
│
├── configs/
│   ├── cnic_front.yaml               # CNIC Front field region & anchor definitions
│   ├── cnic_back.yaml                # CNIC Back field region definitions
│   ├── passport.yaml                 # Passport biodata & MRZ field definitions
│   └── preprocessing/                # Adaptive preprocessing configs
│
├── tests/                            # Unit & Integration test suite (42 tests)
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Installation & Setup

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run Pytest Suite**:
```bash
pytest tests/ -v
```

---

## Running FastAPI Web API

Start Uvicorn dev server:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints:
- `POST /ocr`: Accepts image file + `document_type` form parameter (`cnic_front`, `cnic_back`, `passport`). Returns structured JSON.
- `POST /ocr/debug`: Returns annotated debug JPEG visualization overlay image.
- `POST /config/region`: Programmatically updates field region coordinates in YAML configs.

---

## Running CLI Tool

1. **Extract Fields from Image**:
```bash
python app/main.py --document-type cnic_front --image sample_cnic.jpg --output-dir ./output --save-ocr --save-visualization
```

2. **Calibrate Region for a Field**:
```bash
python app/cli.py calibrate-region --config cnic_front --field cnic_number --x1 0.20 --y1 0.52 --x2 0.85 --y2 0.65 --normalization cnic --validator cnic_number
```

---

## Python API Usage Example

```python
from app.documents.cnic.pipeline import CNICPipeline
from app.documents.passport.pipeline import PassportPipeline

# Process CNIC Front
cnic_pipe = CNICPipeline()
cnic_result = cnic_pipe.process("sample_cnic.jpg", doc_side="front", debug=True)

print("Bilingual Name:", cnic_result["name"])
print("CNIC Number:", cnic_result["fields"]["cnic_number"]["value"])
print("Validated:", cnic_result["fields"]["cnic_number"]["validated"])

# Process Passport
passport_pipe = PassportPipeline()
passport_result = passport_pipe.process("sample_passport.jpg")

print("Passport Number:", passport_result["fields"]["passport_number"]["value"])
print("MRZ Line 1:", passport_result["fields"]["mrz_line1"]["value"])
```
