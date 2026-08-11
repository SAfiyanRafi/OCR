"""
Unified Application Entry Point (FastAPI Web API + Web UI + CLI Interface).
"""

import os
import sys
import json
import click
import cv2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api.routes import router as api_router
from app.documents.cnic.pipeline import CNICPipeline
from app.documents.passport.pipeline import PassportPipeline
from app.visualization.debug import draw_debug_visualization

# Initialize FastAPI app instance
app = FastAPI(
    title="Pakistani Document Image Preprocessing & OCR System",
    description="Production-grade adaptive image preprocessor and configurable OCR field extractor for Pakistani CNIC & Passports.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount UI static directory
ui_dir = os.path.join(os.path.dirname(__file__), "ui")
if os.path.exists(ui_dir):
    app.mount("/ui", StaticFiles(directory=ui_dir, html=True), name="ui")

app.include_router(api_router, prefix="/api/v1")
app.include_router(api_router)


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "Pakistani CNIC & Passport OCR System",
        "ui": "/ui",
        "endpoints": ["POST /ocr", "POST /ocr/debug", "POST /config/region"]
    }


@app.get("/")
def read_root():
    """Serve Web UI at root URL."""
    index_path = os.path.join(ui_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return health_check()


# Click CLI Command Definitions
@click.command()
@click.option("--document-type", "-d", default="cnic_front", type=click.Choice(["cnic_front", "cnic_back", "passport"]), help="Document type")
@click.option("--image", "-i", "image_path", required=True, type=click.Path(exists=True), help="Input image file path")
@click.option("--output-dir", "-o", default="./output", help="Directory to save extraction results")
@click.option("--debug", is_flag=True, help="Enable step-by-step preprocessing debug images")
@click.option("--save-ocr", is_flag=True, help="Save raw OCR tokens JSON")
@click.option("--save-visualization", is_flag=True, help="Save annotated debug visualization overlay image")
def main_cli(document_type: str, image_path: str, output_dir: str, debug: bool, save_ocr: bool, save_visualization: bool):
    """Run Document Image Preprocessing & OCR Field Extraction CLI."""
    click.echo(f"Processing image: {image_path} (Document Type: {document_type})")
    os.makedirs(output_dir, exist_ok=True)

    doc_type = document_type.lower().strip()

    if "passport" in doc_type:
        pipeline = PassportPipeline()
        result = pipeline.process(image_path, debug=debug)
        prep_result = pipeline.preprocessor.process(image_path, document_type="passport")
    elif "back" in doc_type:
        pipeline = CNICPipeline()
        result = pipeline.process(image_path, doc_side="back", debug=debug)
        prep_result = pipeline.preprocessor.process(image_path, document_type="cnic_back")
    else:
        pipeline = CNICPipeline()
        result = pipeline.process(image_path, doc_side="front", debug=debug)
        prep_result = pipeline.preprocessor.process(image_path, document_type="cnic_front")

    # Save extraction result JSON
    out_json_path = os.path.join(output_dir, "extracted_fields.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    click.echo(f"Successfully extracted fields! Saved to: {out_json_path}")

    if save_ocr and "raw_ocr" in result:
        ocr_path = os.path.join(output_dir, "raw_ocr_tokens.json")
        with open(ocr_path, "w", encoding="utf-8") as f:
            json.dump(result["raw_ocr"], f, indent=2, ensure_ascii=False)
        click.echo(f"Raw OCR tokens saved to: {ocr_path}")

    if save_visualization:
        annotated = draw_debug_visualization(prep_result.best_image, result)
        vis_path = os.path.join(output_dir, "annotated_debug.jpg")
        cv2.imwrite(vis_path, annotated)
        click.echo(f"Annotated visualization image saved to: {vis_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] not in ["uvicorn", "run"]:
        main_cli()
    else:
        import uvicorn
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
