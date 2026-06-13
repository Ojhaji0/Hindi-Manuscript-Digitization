# Hindi Manuscript Digitization

An end-to-end OCR and NLP pipeline for digitizing handwritten Hindi manuscripts (Bahi records — genealogical/ritual ledgers). Processes scanned images through image preprocessing, Google Cloud Vision OCR, entity recognition, relation extraction, and stores structured results in a searchable SQLite database.

---

## Project Structure

```
Hindi-Manuscript-Digitization/
│
├── pipeline/
│   ├── preprocessing.py     # 5-step OpenCV image preprocessing
│   ├── ocr.py               # Google Vision OCR + hash-based cache
│   ├── nlp.py               # 5-stage NLP pipeline (NER + relation extraction)
│   └── database.py          # SQLite schema, write, index layer
│
├── batch_processor.py       # Process entire image folders end-to-end
├── search.py                # CLI search interface for researchers
├── evaluate.py              # Accuracy metrics (CER, F1) + query benchmarking
├── augment.py               # Image augmentation for robustness testing
├── tune_preprocessing.py    # Hyperparameter grid search for preprocessing
├── create_ground_truth.py   # Helper: auto-generate ground truth JSON templates
│
├── ground_truth/            # Manually verified JSON annotation files (you create these)
├── manuscripts.db           # SQLite output database (auto-generated)
├── cache.db                 # Vision API response cache (auto-generated)
├── processing_log.csv       # Batch processing audit trail (auto-generated)
├── best_params.json         # Best preprocessing params from tuning (auto-generated)
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up Google Cloud credentials

```bash
# Download your service account JSON from Google Cloud Console
# Then set the environment variable:

# Windows
set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\your-credentials.json

# Linux / Mac
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your-credentials.json
```

You need two APIs enabled in Google Cloud Console:
- **Cloud Vision API** — for OCR
- **Cloud Natural Language API** — for entity recognition

---

## Pipeline Architecture

```
Image File
    │
    ▼
[Stage 1] preprocessing.py
  Grayscale → Denoise → Adaptive Binarize → Deskew → Border Crop
    │
    ▼
[Stage 2] ocr.py
  Google Cloud Vision (Document Text Detection, Hindi language hint)
  + MD5 hash cache (skips API call on repeat images)
    │
    ▼
[Stage 3] nlp.py — clean_text()
  Whitespace normalization, Devanagari digit conversion
    │
    ▼
[Stage 4] nlp.py — segment_paragraphs()
  Split by newline markers into paragraph blocks
    │
    ▼
[Stage 5] nlp.py — extract_entities()
  Google Natural Language API: identifies PERSON and LOCATION entities
    │
    ▼
[Stage 6] nlp.py — extract_relations()
  60+ regex patterns: primary person markers, family relationships,
  ritual details, caste/subcaste, dates (Gregorian + Hindu calendar)
    │
    ▼
[Stage 7] nlp.py — validate_records()
  Confidence scoring (0.0–1.0), low-confidence records flagged
    │
    ▼
[Stage 8] database.py
  SQLite: documents, persons, rituals, locations tables
  Indexed on name, caste, place, date, family_id
```

---

## Usage

### Process a single image

```python
from pipeline.ocr import detect_text
from pipeline.nlp import run_nlp_pipeline
from pipeline.database import write_records

ocr_text = detect_text("path/to/image.jpg")
records  = run_nlp_pipeline(ocr_text, file_name="image.jpg")
write_records(records, file_name="image.jpg")
```

### Process an entire folder (batch)

```bash
python batch_processor.py images/ --workers 4
```

Options:
```
--workers N       Parallel Vision API workers (default: 4)
--no-cache        Disable OCR response cache
--no-preprocess   Skip image preprocessing
--credentials     Path to Google Cloud JSON key
--log             Custom path for processing log CSV
```

### Search the database

```bash
# Search by person name
python search.py --name "राम कुमार"

# Filter by caste
python search.py --caste "राजपूत"

# Filter by village/place
python search.py --place "आगरा"

# Date range (Gregorian years)
python search.py --date-range 1900 1950

# All members of a family
python search.py --family-id F042

# By relationship type
python search.py --relation "पिता"

# Records flagged as low-confidence (for human review)
python search.py --flagged

# Database summary statistics
python search.py --stats

# Export any search to Excel
python search.py --name "राम" --export results.xlsx
```

### Augment images for robustness testing

```bash
python augment.py images/ augmented_images/ --variants 5
```

Generates variants per image: rotation (±2°, ±5°, ±10°), brightness, contrast, Gaussian noise, blur.

### Tune preprocessing hyperparameters

```bash
python tune_preprocessing.py \
  --ground-truth ground_truth/ \
  --image-folder images/ \
  --save
```

Grid searches over: blur kernel size, adaptive threshold block size, threshold constant, deskew tolerance, Otsu vs adaptive. Saves best parameters to `best_params.json`.

### Evaluate accuracy and benchmark queries

```bash
# Accuracy against ground truth
python evaluate.py accuracy --ground-truth ground_truth/ --image-folder images/

# Query speed benchmark (before vs after indexing)
python evaluate.py benchmark --runs 100

# Both at once
python evaluate.py all --ground-truth ground_truth/
```

---

## Ground Truth Annotation

Ground truth files are manually verified records used to measure accuracy. See the [Ground Truth Guide](#ground-truth-guide) section below.

### Quick start — auto-generate templates

Instead of typing from scratch, run `create_ground_truth.py` on your images. It runs the pipeline, pre-fills a JSON with the pipeline's best guess, and you just **correct the mistakes**:

```bash
# Single image
python create_ground_truth.py image00015.jpg --output-dir ground_truth/

# Entire folder (generates one JSON per image)
python create_ground_truth.py images/ --output-dir ground_truth/ --batch
```

Then open each JSON in any text editor, fix wrong fields, and save. That's it.

### Ground Truth Guide

Each file in `ground_truth/` corresponds to one manuscript image.

**File naming:** `image00015.json` for `image00015.jpg`

**Format:**

```json
{
  "file_name": "image00015.jpg",
  "ocr_text": "राजपूत वासी आगरा प्रा0 राम कुमार बेटा श्याम लाल...",
  "records": [
    {
      "Given Name": "राम कुमार",
      "Relation": "स्वयं",
      "Gender": "पुरुष",
      "Caste": "राजपूत",
      "Subcaste": "",
      "From Which Place": "आगरा",
      "Date of Ritual (Gregorian)": "1945-03-15"
    },
    {
      "Given Name": "श्याम लाल",
      "Relation": "पिता (राम कुमार का)",
      "Gender": "पुरुष",
      "Caste": "राजपूत",
      "Subcaste": "",
      "From Which Place": "आगरा",
      "Date of Ritual (Gregorian)": "1945-03-15"
    }
  ]
}
```

**Fields to fill in each record:**

| Field | What to write | Example |
|---|---|---|
| `Given Name` | Person's first/given name only | `राम कुमार` |
| `Relation` | Their relationship in the record | `स्वयं`, `पिता (राम का)` |
| `Gender` | `पुरुष` or `स्त्री` | `पुरुष` |
| `Caste` | Caste name | `राजपूत` |
| `Subcaste` | Subcaste if mentioned, else `""` | `राठौर` |
| `From Which Place` | Village/place | `आगरा` |
| `Date of Ritual (Gregorian)` | Date in YYYY-MM-DD, or `""` if absent | `1945-03-15` |

**Tips:**
- The `ocr_text` field should be the **exact raw text** you see in the image (as you would read it). This is used to measure OCR Character Error Rate.
- Leave a field as `""` if the information is genuinely absent from the image — do not guess.
- Aim for 30–50 annotated images for statistically meaningful accuracy numbers.

---

## Output Database Schema

The pipeline writes to `manuscripts.db` (SQLite):

```sql
TABLE documents    -- one row per processed image
TABLE persons      -- one row per extracted individual
TABLE rituals      -- ritual dates linked to each person
TABLE locations    -- normalized unique place names
```

**Indexes** on: `persons.given_name`, `persons.caste`, `persons.place`, `persons.family_id`, `rituals.ritual_date_gregorian` — these produce the query speed improvement measured by `evaluate.py benchmark`.

---

## Extracted Fields

Each person record contains:

| Field | Description |
|---|---|
| Individual ID | Unique identifier (P0001, P0002, ...) |
| Given Name | Person's given/first name |
| Surname | Extracted surname if present |
| Gender | पुरुष / स्त्री |
| Relation | Relationship to the primary person in the record |
| Caste | Caste name |
| Subcaste | Subcaste if mentioned |
| From Which Place | Village or town of origin |
| Date of Ritual | Original date text from manuscript |
| Date of Ritual (Gregorian) | Converted to YYYY-MM-DD |
| Whose Ritual 1 | Primary person the ritual belongs to |
| Family Id | Groups related individuals within a document block |
| confidence | Extraction confidence score (0.0–1.0) |
| flagged | True if below confidence threshold (needs review) |

---

## Performance Notes

- **Caching:** The `cache.db` stores Vision API responses keyed by image MD5 hash. Re-processing the same images skips API calls entirely, significantly reducing latency on re-runs.
- **Parallel OCR:** `batch_processor.py` uses `ThreadPoolExecutor` (default 4 workers) since Vision API calls are network I/O-bound. Increase `--workers` for faster batch processing on large datasets.
- **Resume capability:** `processing_log.csv` tracks processed files. Re-running `batch_processor.py` skips already-successful images automatically.
- **Indexes:** Database indexes are built after bulk insert (faster writes), then queries run against indexed columns for fast retrieval.
