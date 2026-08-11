"""
Production FastAPI REST API Routes.
Exposes endpoints for OCR extraction, debug overlay rendering, adaptive preprocessing,
document classification, and configuration management.
"""

from typing import Dict, Any, Optional
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Response
from fastapi.responses import JSONResponse

from app.documents.cnic.pipeline import CNICPipeline
from app.documents.passport.pipeline import PassportPipeline
from app.preprocessing.pipeline import AdaptivePreprocessor
from app.classification.classifier import DocumentClassifier
from app.ocr.paddle import PaddleOCRAdapter
from app.api.schemas import (
    ProductionOCRResponse, ClassifyResponse, PreprocessResponse
)
from app.api.models import RegionConfigRequest, CreateConfigRequest, SaveFieldRequest
from app.extraction.config_tool import (
    list_all_configs,
    get_single_config,
    create_new_config,
    save_field_config,
    delete_field_config,
    update_field_region_config
)
from app.visualization.debug import draw_debug_visualization

router = APIRouter()

cnic_pipeline = CNICPipeline()
passport_pipeline = PassportPipeline()
preprocessor = AdaptivePreprocessor()
ocr_engine = PaddleOCRAdapter()


@router.post("/ocr", response_model=ProductionOCRResponse)
async def process_ocr_endpoint(
    file: UploadFile = File(...),
    document_type: str = Form("cnic_front")
):
    """
    Upload document image and execute adaptive preprocessing + field extraction.
    Supports auto-detecting document type (cnic_front, cnic_back, passport).
    """
    try:
        contents = await file.read()
        doc_type = document_type.lower().strip()

        if doc_type == "auto":
            raw_ocr_sample = ocr_engine.extract_tokens(contents)
            cls_res = DocumentClassifier.classify(raw_ocr_sample)
            doc_type = cls_res.document_type if cls_res.document_type != "unknown" else "cnic_front"

        if "passport" in doc_type:
            result = passport_pipeline.process(contents)
        elif "cnic_back" in doc_type or doc_type == "back":
            result = cnic_pipeline.process(contents, doc_side="back")
        else:
            result = cnic_pipeline.process(contents, doc_side="front")

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ocr/debug")
async def process_ocr_debug_endpoint(
    file: UploadFile = File(...),
    document_type: str = Form("cnic_front")
):
    """
    Upload image and return multi-layer annotated debug visualization image.
    """
    try:
        contents = await file.read()
        doc_type = document_type.lower().strip()

        if doc_type == "auto":
            raw_ocr_sample = ocr_engine.extract_tokens(contents)
            cls_res = DocumentClassifier.classify(raw_ocr_sample)
            doc_type = cls_res.document_type if cls_res.document_type != "unknown" else "cnic_front"

        if "passport" in doc_type:
            result = passport_pipeline.process(contents)
            prep_res = passport_pipeline.preprocessor.process(contents, document_type="passport")
        elif "cnic_back" in doc_type or doc_type == "back":
            result = cnic_pipeline.process(contents, doc_side="back")
            prep_res = cnic_pipeline.preprocessor.process(contents, document_type="cnic_back")
        else:
            result = cnic_pipeline.process(contents, doc_side="front")
            prep_res = cnic_pipeline.preprocessor.process(contents, document_type="cnic_front")

        annotated = draw_debug_visualization(prep_res["best_image"], result)

        _, encoded_img = cv2.imencode(".jpg", annotated)
        return Response(content=encoded_img.tobytes(), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_image_endpoint(file: UploadFile = File(...)):
    """
    Execute adaptive preprocessing pipeline without running OCR extraction.
    """
    try:
        contents = await file.read()
        prep_res = preprocessor.process(contents)
        return PreprocessResponse(
            status="success",
            quality_report=prep_res["quality_report"].__dict__,
            preprocessing_plan=prep_res["preprocessing_plan"].__dict__,
            stages=[s.name for s in prep_res["stages"]],
            selected_variant=prep_res["variants"][0].name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classify", response_model=ClassifyResponse)
async def classify_document_endpoint(file: UploadFile = File(...)):
    """
    Classify document type (cnic_front, cnic_back, passport, unknown).
    """
    try:
        contents = await file.read()
        raw_ocr = ocr_engine.extract_tokens(contents)
        cls_res = DocumentClassifier.classify(raw_ocr)
        return ClassifyResponse(
            document_type=cls_res.document_type,
            confidence=cls_res.confidence,
            method=cls_res.method
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/configs")
async def get_all_configs_endpoint():
    """
    List all YAML configuration files and schemas.
    """
    return list_all_configs()


@router.get("/configs/{config_name}")
async def get_single_config_endpoint(config_name: str):
    """
    Get detailed schema of a single configuration file.
    """
    cfg = get_single_config(config_name)
    if not cfg:
        raise HTTPException(status_code=404, detail=f"Configuration '{config_name}' not found.")
    return cfg


@router.post("/config/create")
async def create_config_endpoint(req: CreateConfigRequest):
    """
    Create a new configuration file.
    """
    success = create_new_config(
        config_name=req.config_name,
        document_type=req.document_type,
        language=req.language or "en"
    )
    if success:
        return {"status": "success", "message": f"Created configuration file '{req.config_name}'."}
    raise HTTPException(status_code=500, detail="Failed to create configuration file.")


@router.post("/config/field")
async def save_field_config_endpoint(req: SaveFieldRequest):
    """
    Add or update a field configuration entry.
    """
    success = save_field_config(
        config_name=req.config_name,
        field_key=req.field_key,
        label=req.label,
        language=req.language or "en",
        strategy=req.strategy or "region",
        region={"x1": req.x1, "y1": req.y1, "x2": req.x2, "y2": req.y2},
        anchor={"keyword": req.anchor_keyword, "direction": req.anchor_direction} if req.anchor_keyword else None,
        normalization=req.normalization or "none",
        validator=req.validator or "none"
    )
    if success:
        return {"status": "success", "message": f"Saved field '{req.field_key}' in '{req.config_name}'."}
    raise HTTPException(status_code=500, detail="Failed to save field configuration.")


@router.delete("/config/field/{config_name}/{field_key}")
async def delete_field_config_endpoint(config_name: str, field_key: str):
    """
    Delete a field entry from a configuration file.
    """
    success = delete_field_config(config_name, field_key)
    if success:
        return {"status": "success", "message": f"Deleted field '{field_key}' from '{config_name}'."}
    raise HTTPException(status_code=404, detail=f"Field '{field_key}' not found in '{config_name}'.")


@router.post("/config/region")
async def update_config_region_endpoint(req: RegionConfigRequest):
    """
    Backward-compatible region configuration endpoint.
    """
    success = update_field_region_config(
        config_path=req.config_name,
        field_name=req.field_name,
        region={"x1": req.x1, "y1": req.y1, "x2": req.x2, "y2": req.y2},
        label=req.label,
        strategy=req.strategy,
        normalization=req.normalization,
        validator=req.validator
    )
    if success:
        return {"status": "success", "message": f"Updated region for field '{req.field_name}' in {req.config_name}"}
    raise HTTPException(status_code=500, detail="Failed to update configuration.")
