"""
Command Line Interface (CLI) for Pakistani Document Image Preprocessing & Configurable OCR.
"""

import os
import json
import logging
import click
import cv2
from app.preprocessing import Preprocessor
from app.ocr.paddle import PaddleOCRAdapter
from app.documents.cnic.pipeline import CNICPipeline
from app.documents.passport.pipeline import PassportPipeline
from app.extraction.config_tool import update_field_region_config
from app.visualization.debug import draw_debug_visualization

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@click.group()
def cli():
    """Pakistani CNIC and Passport Preprocessing & OCR Extraction CLI Tool."""
    pass


@cli.command("extract")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True), help="Input image file path")
@click.option("--doc-type", "-d", default="cnic_front", type=click.Choice(["cnic_front", "cnic_back", "passport"]), help="Document type")
@click.option("--output-dir", "-o", default="./output", help="Directory to save preprocessed results")
@click.option("--debug", is_flag=True, help="Save step-by-step debug images")
@click.option("--save-visualization", is_flag=True, help="Save annotated visualization overlay image")
def extract_cmd(input_path: str, doc_type: str, output_dir: str, debug: bool, save_visualization: bool):
    """Run full preprocessing + OCR tokenization + field extraction pipeline."""
    click.echo(f"Extracting fields for: {input_path} (Type: {doc_type})")
    os.makedirs(output_dir, exist_ok=True)

    if "passport" in doc_type:
        pipeline = PassportPipeline()
        result = pipeline.process(input_path, debug=debug)
    elif "back" in doc_type:
        pipeline = CNICPipeline()
        result = pipeline.process(input_path, doc_side="back", debug=debug)
    else:
        pipeline = CNICPipeline()
        result = pipeline.process(input_path, doc_side="front", debug=debug)

    out_json = os.path.join(output_dir, "extraction_result.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    click.echo(f"Extraction complete! Results saved to: {out_json}")

    if save_visualization:
        prep_result = pipeline.preprocessor.process(input_path, document_type=doc_type)
        annotated = draw_debug_visualization(prep_result.best_image, result)
        vis_path = os.path.join(output_dir, "annotated_debug.jpg")
        cv2.imwrite(vis_path, annotated)
        click.echo(f"Annotated visualization saved to: {vis_path}")


@cli.command("calibrate-region")
@click.option("--config", "-c", "config_name", required=True, help="Config file name (cnic_front, cnic_back, passport)")
@click.option("--field", "-f", "field_name", required=True, help="Field name key (e.g. cnic_number, name_en)")
@click.option("--x1", required=True, type=float, help="Normalized x1 (0.0 - 1.0)")
@click.option("--y1", required=True, type=float, help="Normalized y1 (0.0 - 1.0)")
@click.option("--x2", required=True, type=float, help="Normalized x2 (0.0 - 1.0)")
@click.option("--y2", required=True, type=float, help="Normalized y2 (0.0 - 1.0)")
@click.option("--strategy", default="region", type=click.Choice(["region", "anchor"]), help="Extraction strategy")
@click.option("--normalization", default="none", help="Normalization rule")
@click.option("--validator", default="none", help="Validator rule")
def calibrate_cmd(config_name: str, field_name: str, x1: float, y1: float, x2: float, y2: float, strategy: str, normalization: str, validator: str):
    """Calibrate and update a field region in YAML configuration."""
    if not config_name.endswith(".yaml"):
        config_name = f"{config_name}.yaml"
    config_path = os.path.join(os.getcwd(), "configs", config_name)

    success = update_field_region_config(
        config_path=config_path,
        field_name=field_name,
        region={"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        strategy=strategy,
        normalization=normalization,
        validator=validator
    )

    if success:
        click.echo(f"Successfully calibrated region for '{field_name}' in {config_path}!")


if __name__ == "__main__":
    cli()
