from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import io

import numpy as np
import torch
import easyocr


from PIL import Image

from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification
)


# ============================================================
# 1. APPLICATION SETUP
# ============================================================

app = FastAPI(
    title="BOM Mismatch Detection API",
    description="Deep Learning-Based BOM Mismatch Detection System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

BASE_MODEL = "microsoft/layoutlmv3-base"

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "model"
    / "layoutlmv3_bom_corrected_best.pth"
)

LABELS = [
    "O",
    "B-PART_NO", "I-PART_NO",
    "B-DESCRIPTION", "I-DESCRIPTION",
    "B-MATERIAL", "I-MATERIAL",
    "B-UOM", "I-UOM",
    "B-QTY", "I-QTY"
]

label2id = {
    label: idx
    for idx, label in enumerate(LABELS)
}

id2label = {
    idx: label
    for idx, label in enumerate(LABELS)
}


# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 4. LOAD LAYOUTLM PROCESSOR
# ============================================================

processor = LayoutLMv3Processor.from_pretrained(
    BASE_MODEL,
    apply_ocr=False
)


# ============================================================
# 5. LOAD LAYOUTLM MODEL
# ============================================================

model = LayoutLMv3ForTokenClassification.from_pretrained(
    BASE_MODEL,
    num_labels=len(LABELS),
    id2label=id2label,
    label2id=label2id
)


# ============================================================
# 6. LOAD TRAINED WEIGHTS
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()


# ============================================================
# 7. LOAD EASYOCR
# ============================================================

reader = easyocr.Reader(
    ["en"],
    gpu=torch.cuda.is_available()
)


# ============================================================
# 8. OCR ROW RECONSTRUCTION
# ============================================================
def reconstruct_ocr_rows(ocr_results, y_tolerance=8):
    """
    Convert EasyOCR detections into rows.

    EasyOCR returns:
        (bbox, text, confidence)

    Each row is sorted from left to right.
    """

    detections = []

    for detection in ocr_results:

        bbox, text, confidence = detection

        bbox = [
            [float(point[0]), float(point[1])]
            for point in bbox
        ]

        text = str(text).strip()

        if not text:
            continue

        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]

        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)

        height = max(ys) - min(ys)

        detections.append({
            "text": text,
            "bbox": bbox,
            "confidence": float(confidence),
            "center_x": center_x,
            "center_y": center_y,
            "height": height
        })

    # ---------------------------------------------------------
    # Sort detections top-to-bottom
    # ---------------------------------------------------------

    detections.sort(
        key=lambda item: item["center_y"]
    )

    # ---------------------------------------------------------
    # Group detections into rows
    # ---------------------------------------------------------

    rows = []

    for detection in detections:

        placed = False

        for row in rows:

            row_center_y = sum(
                item["center_y"]
                for item in row
            ) / len(row)

            if abs(
                detection["center_y"] - row_center_y
            ) <= y_tolerance:

                row.append(detection)
                placed = True
                break

        if not placed:
            rows.append([detection])

    # ---------------------------------------------------------
    # Sort each row left-to-right
    # ---------------------------------------------------------

    rows.sort(
        key=lambda row: min(
            item["center_y"]
            for item in row
        )
    )

    reconstructed_rows = []

    for row_id, row in enumerate(rows):

        row.sort(
            key=lambda item: item["center_x"]
        )

        reconstructed_rows.append({
            "row_id": row_id,

            "text": " | ".join(
                item["text"]
                for item in row
            ),

            "tokens": row
        })

    return reconstructed_rows


# ============================================================
# 9. LAYOUTLMV3 BOM FIELD EXTRACTION
# ============================================================

def extract_bom_fields(
    rows,
    bom_crop
):
    """
    Uses the trained LayoutLMv3 model to assign
    OCR words to BOM fields.
    """

    extracted_rows = []

    crop_width, crop_height = bom_crop.size

    for row in rows:

        tokens = row["tokens"]

        if not tokens:
            continue

        # ----------------------------------------------------
        # Prepare OCR words and bounding boxes
        # ----------------------------------------------------

        words = [
            token["text"]
            for token in tokens
        ]

        boxes = [
            token["bbox"]
            for token in tokens
        ]

        normalized_boxes = []

        for bbox in boxes:

            x_coordinates = [
                point[0]
                for point in bbox
            ]

            y_coordinates = [
                point[1]
                for point in bbox
            ]

            x0 = min(x_coordinates)
            y0 = min(y_coordinates)
            x1 = max(x_coordinates)
            y1 = max(y_coordinates)

            # Normalize coordinates to LayoutLM range 0-1000
            x0 = int(
                (x0 / crop_width) * 1000
            )

            y0 = int(
                (y0 / crop_height) * 1000
            )

            x1 = int(
                (x1 / crop_width) * 1000
            )

            y1 = int(
                (y1 / crop_height) * 1000
            )

            normalized_boxes.append([
                max(0, min(1000, x0)),
                max(0, min(1000, y0)),
                max(0, min(1000, x1)),
                max(0, min(1000, y1))
            ])

        # ----------------------------------------------------
        # LayoutLMv3 encoding
        # ----------------------------------------------------

        encoding = processor(
            images=bom_crop,
            text=words,
            boxes=normalized_boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256
        )

        # Keep the original BatchEncoding object.
        # This is required for encoding.word_ids().
        model_inputs = {
            key: value.to(device)
            for key, value in encoding.items()
            if isinstance(value, torch.Tensor)
        }

        # ----------------------------------------------------
        # Model inference
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(
                **model_inputs
            )

        predictions = torch.argmax(
            outputs.logits,
            dim=-1
        )[0].cpu().tolist()

        # Map subword tokens back to original OCR words
        word_ids = encoding.word_ids(
            batch_index=0
        )

        # ----------------------------------------------------
        # Store predicted fields
        # ----------------------------------------------------

        field_values = {
            "PART_NO": [],
            "DESCRIPTION": [],
            "MATERIAL": [],
            "UOM": [],
            "QTY": []
        }

        previous_word_id = None

        for token_index, word_id in enumerate(word_ids):

            if word_id is None:
                continue

            # Ignore additional subword tokens
            if word_id == previous_word_id:
                continue

            previous_word_id = word_id

            if word_id >= len(words):
                continue

            predicted_label = id2label[
                predictions[token_index]
            ]

            word = words[word_id]

            if predicted_label in [
                "B-PART_NO",
                "I-PART_NO"
            ]:

                field_values["PART_NO"].append(
                    word
                )

            elif predicted_label in [
                "B-DESCRIPTION",
                "I-DESCRIPTION"
            ]:

                field_values["DESCRIPTION"].append(
                    word
                )

            elif predicted_label in [
                "B-MATERIAL",
                "I-MATERIAL"
            ]:

                field_values["MATERIAL"].append(
                    word
                )

            elif predicted_label in [
                "B-UOM",
                "I-UOM"
            ]:

                field_values["UOM"].append(
                    word
                )

            elif predicted_label in [
                "B-QTY",
                "I-QTY"
            ]:

                field_values["QTY"].append(
                    word
                )

        # ----------------------------------------------------
        # Create structured BOM row
        # ----------------------------------------------------

        structured_row = {
            "row_id": row["row_id"],

            "PART_NO": " ".join(
                field_values["PART_NO"]
            ).strip(),

            "DESCRIPTION": " ".join(
                field_values["DESCRIPTION"]
            ).strip(),

            "MATERIAL": " ".join(
                field_values["MATERIAL"]
            ).strip(),

            "UOM": " ".join(
                field_values["UOM"]
            ).strip(),

            "QTY": " ".join(
                field_values["QTY"]
            ).strip()
        }

        extracted_rows.append(
            structured_row
        )

    return extracted_rows


# ============================================================
# 10. BOM OUTPUT CLEANUP
# ============================================================

def clean_structured_bom(
    structured_bom
):
    """
    Removes obvious title/header rows and empty rows.
    """

    cleaned_bom = []

    for row in structured_bom:

        values = {
            key: str(
                row.get(key, "")
            ).strip()

            for key in [
                "PART_NO",
                "DESCRIPTION",
                "MATERIAL",
                "UOM",
                "QTY"
            ]
        }

        combined_text = " ".join(
            values.values()
        ).upper()

        # Remove obvious title/header rows
        if (
            "BILL OF MATERIALS" in combined_text
            or (
                "DESCRIPTION" in combined_text
                and "MATERIAL" in combined_text
            )
        ):
            continue

        # Remove completely empty rows
        if not any(values.values()):
            continue

        cleaned_bom.append({
            "row_id": row["row_id"],
            **values
        })

    return cleaned_bom

# ============================================================
# PART NUMBER NORMALIZATION
# ============================================================

import re
from difflib import SequenceMatcher


def normalize_part_number(
    value,
    reference_part_numbers,
    threshold=0.85
):
    """
    Normalize an OCR-extracted PART_NO against a
    training-only PART_NO vocabulary.

    Returns the original value when no sufficiently
    strong match exists.
    """

    value = str(value).strip()

    if not value:
        return value

    def canonical(text):
        return re.sub(
            r"[^A-Z0-9]",
            "",
            str(text).upper()
        )

    candidate = canonical(value)

    if not candidate:
        return value

    best_match = None
    best_score = 0.0

    for reference_part in reference_part_numbers:

        reference_canonical = canonical(
            reference_part
        )

        if not reference_canonical:
            continue

        score = SequenceMatcher(
            None,
            candidate,
            reference_canonical
        ).ratio()

        if score > best_score:
            best_score = score
            best_match = reference_part

    if (
        best_match is not None
        and best_score >= threshold
    ):
        return best_match

    return value
# ============================================================
# 11. ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "BOM Mismatch Detection API is running",
        "model": "LayoutLMv3",
        "device": str(device),
        "model_loaded": True,
        "ocr_loaded": True
    }


# ============================================================
# 12. MODEL STATUS
# ============================================================

@app.get("/model-status")
def model_status():

    return {
        "model_loaded": True,
        "ocr_loaded": True,
        "device": str(device),
        "model_path": str(MODEL_PATH),
        "labels": LABELS
    }


# ============================================================
# 13. BLUEPRINT UPLOAD
# ============================================================

@app.post("/upload-blueprint")
async def upload_blueprint(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # 1. Read uploaded image
    # --------------------------------------------------------

    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    width, height = image.size

    # --------------------------------------------------------
    # 2. Crop BOM region
    #
    # Current project-layout assumption:
    # BOM is located in the lower-right region.
    # --------------------------------------------------------

    x1 = 760
    y1 = 515
    x2 = min(1245, width)
    y2 = min(760, height)

    bom_crop = image.crop(
        (x1, y1, x2, y2)
    )

    # --------------------------------------------------------
    # 3. Convert crop to NumPy array
    # --------------------------------------------------------

    bom_array = np.array(
        bom_crop
    )

    # --------------------------------------------------------
    # 4. EasyOCR
    # --------------------------------------------------------

    ocr_results = reader.readtext(
        bom_array,
        detail=1,
        paragraph=False
    )

    # --------------------------------------------------------
    # 5. Convert OCR results
    # --------------------------------------------------------

    detections = []

    for bbox, text, confidence in ocr_results:

        detections.append({
            "text": text,

            "bbox": [
                [
                    int(point[0]),
                    int(point[1])
                ]
                for point in bbox
            ],

            "confidence": float(
                confidence
            )
        })

    # --------------------------------------------------------
    # 6. Reconstruct OCR rows
    # --------------------------------------------------------

    rows = reconstruct_ocr_rows(
        detections
    )

    # --------------------------------------------------------
    # 7. LayoutLMv3 field extraction
    # --------------------------------------------------------

    structured_bom = extract_bom_fields(
        rows,
        bom_crop
    )

    # --------------------------------------------------------
    # 8. Clean BOM output
    # --------------------------------------------------------

    structured_bom = clean_structured_bom(
        structured_bom
    )

    # --------------------------------------------------------
    # 9. Return result
    # --------------------------------------------------------

    return {
        "status": "success",

        "filename": file.filename,

        "image_width": width,
        "image_height": height,

        "bom_crop": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": x2 - x1,
            "height": y2 - y1
        },

        "ocr_detections": len(
            detections
        ),

        "ocr_rows": len(
            rows
        ),

        "bom": structured_bom
    }

# ============================================================
# 14. BOM MISMATCH DETECTION
# ============================================================

from pydantic import BaseModel
from typing import List


class BOMItem(BaseModel):
    PART_NO: str
    DESCRIPTION: str
    MATERIAL: str
    UOM: str
    QTY: str


class BOMComparisonRequest(BaseModel):
    reference_bom: List[BOMItem]
    detected_bom: List[BOMItem]


@app.post("/check-bom")
def check_bom(request: BOMComparisonRequest):

    reference = {
        item.PART_NO.strip(): item
        for item in request.reference_bom
        if item.PART_NO.strip()
    }

    detected = {
        item.PART_NO.strip(): item
        for item in request.detected_bom
        if item.PART_NO.strip()
    }

    results = []

    # --------------------------------------------------------
    # Compare detected BOM against reference BOM
    # --------------------------------------------------------

    for part_no, detected_item in detected.items():

        # Extra part
        if part_no not in reference:

            results.append({
                "PART_NO": part_no,
                "FIELD": "ROW",
                "EXPECTED": "Part should exist in reference BOM",
                "DETECTED": "Extra part",
                "STATUS": "MISMATCH"
            })

            continue

        expected_item = reference[part_no]

        fields = [
            "DESCRIPTION",
            "MATERIAL",
            "UOM",
            "QTY"
        ]

        for field in fields:

            expected_value = str(
                getattr(
                    expected_item,
                    field
                )
            ).strip()

            detected_value = str(
                getattr(
                    detected_item,
                    field
                )
            ).strip()

            status = (
                "MATCH"
                if expected_value == detected_value
                else "MISMATCH"
            )

            results.append({
                "PART_NO": part_no,
                "FIELD": field,
                "EXPECTED": expected_value,
                "DETECTED": detected_value,
                "STATUS": status
            })

    # --------------------------------------------------------
    # Detect missing parts
    # --------------------------------------------------------

    for part_no in reference:

        if part_no not in detected:

            results.append({
                "PART_NO": part_no,
                "FIELD": "ROW",
                "EXPECTED": "Part exists in reference BOM",
                "DETECTED": "Missing part",
                "STATUS": "MISMATCH"
            })

    # --------------------------------------------------------
    # Calculate summary
    # --------------------------------------------------------

    total_fields = len(results)

    matching_fields = sum(
        1
        for result in results
        if result["STATUS"] == "MATCH"
    )

    mismatched_fields = sum(
        1
        for result in results
        if result["STATUS"] == "MISMATCH"
    )

    overall_status = (
        "MATCH"
        if mismatched_fields == 0
        else "MISMATCH DETECTED"
    )

    return {
        "status": overall_status,

        "summary": {
            "total_fields_checked": total_fields,
            "matching_fields": matching_fields,
            "mismatched_fields": mismatched_fields
        },

        "results": results
    }

# ============================================================
# 15. COMPLETE BLUEPRINT ANALYSIS
# ============================================================

@app.post("/analyze-blueprint")
async def analyze_blueprint(
    blueprint: UploadFile = File(...),
    reference_bom: UploadFile | None = File(None)
):
    """
    Analyze an uploaded engineering blueprint.

    Reference BOM priority:
    1. User-uploaded reference BOM
    2. Project ground-truth BOM for known dataset blueprints
    3. No comparison for unknown blueprints
    """

    # ---------------------------------------------------------
    # 1. Read uploaded blueprint
    # ---------------------------------------------------------

    image_bytes = await blueprint.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    width, height = image.size

    # ---------------------------------------------------------
    # 2. Select BOM crop
    # ---------------------------------------------------------

    dynamic_crop_dir = (
        Path(__file__).resolve().parent.parent
        / "outputs"
        / "dynamic_bom_crops"
    )

    uploaded_stem = Path(
        blueprint.filename
    ).stem.strip().lower()

    dynamic_crop_path = (
        dynamic_crop_dir
        / f"{uploaded_stem}_dynamic_bom.png"
    )

    if dynamic_crop_path.exists():

        bom_crop = Image.open(
            dynamic_crop_path
        ).convert("RGB")

        crop_source = "saved_dynamic_crop"

        x1 = 0
        y1 = 0
        x2 = bom_crop.width
        y2 = bom_crop.height

    else:

        # Fallback for an unseen blueprint
        x1 = 760
        y1 = 515
        x2 = min(1245, width)
        y2 = min(760, height)

        bom_crop = image.crop(
            (x1, y1, x2, y2)
        )

        crop_source = "fallback_crop"

    # ---------------------------------------------------------
    # 3. OCR
    # ---------------------------------------------------------

    image_array = np.array(
        bom_crop
    )

    ocr_results = reader.readtext(
        image_array
    )

    rows = reconstruct_ocr_rows(
        ocr_results
    )

    # ---------------------------------------------------------
    # 4. LayoutLMv3 BOM extraction
    # ---------------------------------------------------------

    extracted_bom = extract_bom_fields(
        rows,
        bom_crop
    )

    extracted_bom = clean_structured_bom(
        extracted_bom
    )

    # ---------------------------------------------------------
    # 5. Determine reference BOM
    # ---------------------------------------------------------
    reference = None
    reference_source = None
    comparison = None

    # Use user-uploaded reference BOM when provided
    if reference_bom is not None and reference_bom.filename:

        reference_bytes = await reference_bom.read()
        reference_text = reference_bytes.decode("utf-8-sig")

        import csv

        csv_reader = csv.DictReader(io.StringIO(reference_text))

        reference = []

        for row in csv_reader:
            reference.append({
                "PART_NO": str(row.get("PART_NO", "")).strip(),
                "DESCRIPTION": str(row.get("DESCRIPTION", "")).strip(),
                "MATERIAL": str(row.get("MATERIAL", "")).strip(),
                "UOM": str(row.get("UOM", "")).strip(),
                "QTY": str(row.get("QTY", "")).strip()
            })

        reference_source = "user_uploaded"

    # Otherwise use project ground truth when the uploaded
    # filename belongs to the project dataset
    else:

        ground_truth_path = Path(
            r"C:\Users\MANJIRI\deep_learning\data\ground_truth\blueprint_ground_truth.csv"
            )

        if ground_truth_path.exists():

            import pandas as pd
            import json

            gt_df = pd.read_csv(ground_truth_path)
            print("GROUND TRUTH PATH:", ground_truth_path)
            print("GROUND TRUTH EXISTS:", ground_truth_path.exists())
            print("UPLOADED FILENAME:", blueprint.filename)
            print(
                "MATCHES:",
                gt_df[
                    gt_df["filename"]
                    .astype(str)
                    .apply(
                        lambda x: Path(x).stem.strip().lower()
                        == Path(blueprint.filename).stem.strip().lower()
                    )
                ]["filename"].tolist()
)
            uploaded_filename = (
                Path(blueprint.filename)
                .stem
                .strip()
                .lower()
            )

            for _, row in gt_df.iterrows():

                gt_filename = (
                    Path(str(row["filename"]))
                    .stem
                    .strip()
                    .lower()
                )

                if gt_filename == uploaded_filename:

                    data = json.loads(row["json_data"])

                    reference = []

                    for item in data["bill_of_materials"]:
                        reference.append({
                            "PART_NO": str(
                                item.get("part_no", "")
                            ).strip(),

                            "DESCRIPTION": str(
                                item.get("description", "")
                            ).strip(),

                            "MATERIAL": str(
                                item.get("material", "")
                            ).strip(),

                            "UOM": str(
                                item.get("uom", "")
                            ).strip(),

                            "QTY": str(
                                item.get("qty", "")
                            ).strip()
                        })

                    reference_source = "project_ground_truth"

                    break
    

# PART_NO NORMALIZATION DIAGNOSTIC
# ---------------------------------------------------------

        training_part_numbers = {
            str(item.get("PART_NO", "")).strip()
            for item in reference or []
            if str(item.get("PART_NO", "")).strip()
            }
        print("\n========== PART_NO NORMALIZATION ==========")
        for item in extracted_bom:
            original = item.get("PART_NO", "")
            normalized = normalize_part_number(
                original,
                training_part_numbers,
                threshold=0.85
            )
            print(
                repr(original),
                "->",
                repr(normalized)
            )
            print("===========================================\n")

    if reference is not None:

        # -----------------------------------------------------
        # Row-aware comparison
        # -----------------------------------------------------

        results = []

        # Group rows by PART_NO while preserving occurrence order.
        reference_groups = {}
        detected_groups = {}

        for item in reference:
            part_no = str(item.get("PART_NO", "")).strip()

            if part_no:
                reference_groups.setdefault(
                    part_no,
                    []
                ).append(item)

        for item in extracted_bom:
            part_no = str(item.get("PART_NO", "")).strip()

            if part_no:
                detected_groups.setdefault(
                    part_no,
                    []
                ).append(item)

        all_part_numbers = list(
            dict.fromkeys(
                list(reference_groups.keys()) +
                list(detected_groups.keys())
            )
        )

        fields = [
            "DESCRIPTION",
            "MATERIAL",
            "UOM",
            "QTY"
        ]

        for part_no in all_part_numbers:

            expected_rows = reference_groups.get(
                part_no,
                []
            )

            detected_rows = detected_groups.get(
                part_no,
                []
            )

            max_occurrences = max(
                len(expected_rows),
                len(detected_rows)
            )

            for occurrence in range(max_occurrences):

                expected_item = (
                    expected_rows[occurrence]
                    if occurrence < len(expected_rows)
                    else None
                )

                detected_item = (
                    detected_rows[occurrence]
                    if occurrence < len(detected_rows)
                    else None
                )

                if (
                    expected_item is not None
                    and detected_item is None
                ):

                    results.append({
                        "PART_NO": part_no,
                        "FIELD": "ROW",
                        "EXPECTED": "Part exists",
                        "DETECTED": "Missing part",
                        "STATUS": "MISMATCH"
                    })

                    continue

                if (
                    expected_item is None
                    and detected_item is not None
                ):

                    results.append({
                        "PART_NO": part_no,
                        "FIELD": "ROW",
                        "EXPECTED": "Part should not exist",
                        "DETECTED": "Extra part",
                        "STATUS": "MISMATCH"
                    })

                    continue

                for field in fields:

                    expected_value = str(
                        expected_item.get(field, "")
                    ).strip()

                    detected_value = str(
                        detected_item.get(field, "")
                    ).strip()

                    if field == "QTY":

                        try:

                            expected_qty = float(
                                expected_value
                                .replace(",", "")
                                .replace(" ", "")
                            )

                            detected_qty = float(
                                detected_value
                                .replace(",", "")
                                .replace(" ", "")
                            )

                            is_match = (
                                abs(
                                    expected_qty -
                                    detected_qty
                                ) < 1e-6
                            )

                        except (ValueError, TypeError):

                            is_match = (
                                expected_value ==
                                detected_value
                            )

                    else:

                        is_match = (
                            expected_value ==
                            detected_value
                        )

                    results.append({
                        "PART_NO": part_no,
                        "FIELD": field,
                        "EXPECTED": expected_value,
                        "DETECTED": detected_value,
                        "STATUS": (
                            "MATCH"
                            if is_match
                            else "MISMATCH"
                        )
                    })

        total_fields = len(results)

        matching_fields = sum(
            1
            for result in results
            if result["STATUS"] == "MATCH"
        )

        mismatched_fields = sum(
            1
            for result in results
            if result["STATUS"] == "MISMATCH"
        )

        comparison = {
            "status": (
                "MATCH"
                if mismatched_fields == 0
                else "MISMATCH DETECTED"
            ),

            "summary": {
                "total_fields_checked": total_fields,
                "matching_fields": matching_fields,
                "mismatched_fields": mismatched_fields
            },

            "results": results
        }

    # ---------------------------------------------------------
    # 7. Final response
    # ---------------------------------------------------------

    return {
        "filename": blueprint.filename,

        "image": {
            "width": width,
            "height": height
        },

        "bom_crop": {
            "source": crop_source,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": bom_crop.width,
            "height": bom_crop.height
        },

        "ocr_detections": len(
            ocr_results
        ),

        "ocr_rows": len(
            rows
        ),

        "extracted_bom_rows": len(
            extracted_bom
        ),

        "extracted_bom": extracted_bom,

        "reference_available": (
            reference is not None
        ),

        "reference_source": reference_source,

        "reference_bom_rows": (
            len(reference)
            if reference is not None
            else 0
        ),

        "comparison_available": (
            comparison is not None
        ),

        "comparison": comparison
    }


