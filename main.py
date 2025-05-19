import io
import base64
import time
import os
import uuid
from pathlib import Path
import random # For colors
import json # For loading colors.json
from typing import List, Dict
from collections import defaultdict

from fastapi import FastAPI, File, UploadFile, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
# import torchvision.transforms as T # Not directly used if DETR processor handles it

from ultralytics import YOLO
# Transformers imports are now more for type hinting or if HF hub is fallback
from transformers import DetrImageProcessor, DetrForObjectDetection 

# Import from our new model loading module
from model import load_detr_from_local_pth 
from preprocessing import preprocess_image

# --- Configuration ---
CKPT_DIR = Path("CKPT")
CKPT_DIR.mkdir(exist_ok=True)  # Ensure CKPT directory exists

# Path for the local DETR .pth file
DETR_LOCAL_PTH_NAME = "detr-r101-2c7b67e5.pth"
DETR_MODEL_PATH = CKPT_DIR / DETR_LOCAL_PTH_NAME

# Fallback Hugging Face model name if local .pth fails or isn't preferred
DETR_HF_HUB_MODEL_NAME = "facebook/detr-resnet-101"

# Dictionary to store loaded YOLO models
yolo_models = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DETR_CLASSES_FROM_MODEL = []

try:
    FONT = ImageFont.truetype("arial.ttf", 15)
    INFERENCE_FONT = ImageFont.truetype("arial.ttf", 18)
except IOError:
    FONT = ImageFont.load_default()
    INFERENCE_FONT = ImageFont.load_default()

# --- Color Palette for Detections ---
# Default colors, to be overridden by colors.json if available
DEFAULT_COLORS = [
    (255, 59, 59), (59, 130, 246), (34, 197, 94), (250, 204, 21),
    (168, 85, 247), (236, 72, 153), (249, 115, 22), (20, 184, 166),
    (100, 116, 139), (248, 113, 113), (96, 165, 250), (74, 222, 128),
]
LOADED_COLORS = []
CLASS_COLOR_CACHE = {}

def load_colors_from_json(file_path: Path = Path("colors.json")):
    global LOADED_COLORS
    try:
        if file_path.exists():
            with open(file_path, 'r') as f:
                colors_from_file = json.load(f)
            # Basic validation: check if it's a list of lists/tuples with 3 numbers
            if isinstance(colors_from_file, list) and all(
                isinstance(c, (list, tuple)) and len(c) == 3 and all(isinstance(i, int) for i in c)
                for c in colors_from_file
            ):
                LOADED_COLORS = [tuple(c) for c in colors_from_file] # Ensure tuples
                print(f"Successfully loaded {len(LOADED_COLORS)} colors from {file_path}.")
            else:
                raise ValueError("Invalid color format in JSON file.")
        else:
            print(f"Warning: {file_path} not found. Using default color palette.")
            LOADED_COLORS = DEFAULT_COLORS
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error loading or parsing {file_path}: {e}. Using default color palette.")
        LOADED_COLORS = DEFAULT_COLORS
    except Exception as e:
        print(f"An unexpected error occurred while loading colors: {e}. Using default color palette.")
        LOADED_COLORS = DEFAULT_COLORS
    
    if not LOADED_COLORS: # Final fallback if something went terribly wrong
        print("Critical: LOADED_COLORS is empty. Falling back to DEFAULT_COLORS one last time.")
        LOADED_COLORS = DEFAULT_COLORS

# Call it once at startup
load_colors_from_json()

def get_color_for_class(class_name):
    global CLASS_COLOR_CACHE, LOADED_COLORS
    if not LOADED_COLORS:
        # This case should ideally not be hit if load_colors_from_json works correctly
        print("Warning: LOADED_COLORS is empty in get_color_for_class. Using default fallback.")
        current_color_palette = DEFAULT_COLORS
    else:
        current_color_palette = LOADED_COLORS

    if class_name not in CLASS_COLOR_CACHE:
        color_index = abs(hash(class_name)) % len(current_color_palette)
        CLASS_COLOR_CACHE[class_name] = current_color_palette[color_index]
    return CLASS_COLOR_CACHE[class_name]


# --- FastAPI App ---
app = FastAPI(title="Object Detection Service")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

detr_model = None
detr_processor = None

def load_models():
    global yolo_models, detr_model, detr_processor, DETR_CLASSES_FROM_MODEL

    # Load all YOLO models from CKPT directory
    for model_file in CKPT_DIR.glob("*.pt"):
        try:
            model_name = model_file.stem
            model = YOLO(model_file)
            model.to(DEVICE)
            yolo_models[model_name] = model
            print(f"YOLO model '{model_name}' loaded from {model_file} on {DEVICE}")
        except Exception as e:
            print(f"Error loading YOLO model {model_file}: {e}")

    # If no models found in CKPT, try to download default models
    if not yolo_models:
        default_models = ["yolov8n", "yolov8m"]
        for model_name in default_models:
            try:
                model = YOLO(model_name)
                model.to(DEVICE)
                yolo_models[model_name] = model
                print(f"Downloaded and loaded {model_name} on {DEVICE}")
            except Exception as e:
                print(f"Error downloading/loading {model_name}: {e}")

    # Load DETR Model
    # Priority 1: Try loading from local .pth file using our new loader
    if DETR_MODEL_PATH.exists():
        print(f"Attempting to load DETR from local .pth: {DETR_MODEL_PATH}")
        detr_model_local, detr_processor_local, detr_classes_local = load_detr_from_local_pth(DETR_MODEL_PATH, DEVICE)
        if detr_model_local and detr_processor_local:
            detr_model = detr_model_local
            detr_processor = detr_processor_local
            DETR_CLASSES_FROM_MODEL = detr_classes_local
            print(f"DETR model '{DETR_MODEL_PATH.name}' loaded successfully from local .pth file.")
        else:
            print(f"Failed to load DETR from local .pth: {DETR_MODEL_PATH}. Attempting Hugging Face Hub fallback.")
            # Fallback to Hugging Face Hub if local load fails
            try:
                detr_processor = DetrImageProcessor.from_pretrained(DETR_HF_HUB_MODEL_NAME)
                detr_model = DetrForObjectDetection.from_pretrained(DETR_HF_HUB_MODEL_NAME).to(DEVICE)
                detr_model.eval()
                DETR_CLASSES_FROM_MODEL = list(detr_model.config.id2label.values())
                print(f"DETR model '{DETR_HF_HUB_MODEL_NAME}' loaded from Hugging Face Hub on {DEVICE}.")
            except Exception as e_hf:
                print(f"Error loading DETR model from Hugging Face Hub as fallback: {e_hf}")
                detr_model = None # Ensure it's None on failure
                detr_processor = None
                DETR_CLASSES_FROM_MODEL = []
    else:
        print(f"Local DETR .pth file {DETR_MODEL_PATH} not found. Attempting Hugging Face Hub.")
        # If local .pth doesn't exist, try Hugging Face Hub directly
        try:
            detr_processor = DetrImageProcessor.from_pretrained(DETR_HF_HUB_MODEL_NAME)
            detr_model = DetrForObjectDetection.from_pretrained(DETR_HF_HUB_MODEL_NAME).to(DEVICE)
            detr_model.eval()
            DETR_CLASSES_FROM_MODEL = list(detr_model.config.id2label.values())
            print(f"DETR model '{DETR_HF_HUB_MODEL_NAME}' loaded from Hugging Face Hub on {DEVICE}.")
        except Exception as e_hf_direct:
            print(f"Error loading DETR model from Hugging Face Hub: {e_hf_direct}")
            detr_model = None # Ensure it's None on failure
            detr_processor = None
            DETR_CLASSES_FROM_MODEL = []

load_models()

os.makedirs("uploads", exist_ok=True) # Still useful if you want to debug uploads

def draw_text_with_background(draw, position, text, font, text_color, bg_color_tuple, padding=2):
    text_bbox = draw.textbbox(position, text, font=font)
    bg_bbox = (
        text_bbox[0] - padding,
        text_bbox[1] - padding,
        text_bbox[2] + padding,
        text_bbox[3] + padding,
    )
    # Ensure background color has alpha if it's a tuple of 3 (RGB)
    if len(bg_color_tuple) == 3:
        bg_color_with_alpha = (*bg_color_tuple, 180) # Add alpha for semi-transparency
    else:
        bg_color_with_alpha = bg_color_tuple # Assume alpha is already included

    draw.rectangle(bg_bbox, fill=bg_color_with_alpha)
    draw.text(position, text, fill=text_color, font=font)


def process_with_yolo(selected_yolo_model: YOLO, image_pil: Image.Image, confidence_threshold: float):
    if not selected_yolo_model:
        raise RuntimeError("Selected YOLO Model not loaded.")

    start_time = time.time()
    results = selected_yolo_model(image_pil, conf=confidence_threshold)
    inference_time = time.time() - start_time

    draw_image = image_pil.copy()
    draw = ImageDraw.Draw(draw_image, "RGBA")
    detection_outputs = []

    if results and results[0].boxes:
        for box in results[0].boxes:
            xyxy = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            class_name = selected_yolo_model.names[cls_id]
            color = get_color_for_class(class_name)

            draw.rectangle(xyxy, outline=color, width=2)
            label = f"{class_name} ({conf:.2f})"
            
            label_y_pos = xyxy[1] - FONT.getbbox(label)[3] - 5
            if label_y_pos < 0: label_y_pos = xyxy[1] + 5

            draw_text_with_background(draw, (xyxy[0], label_y_pos), label, FONT, "white", color)
            
            detection_outputs.append({
                "class_name": class_name,
                "confidence": f"{conf:.2f}",
                "color": f"rgb({color[0]},{color[1]},{color[2]})"
            })

    time_text = f"Inference: {inference_time:.2f}s"
    img_width, img_height = draw_image.size
    time_text_bbox = INFERENCE_FONT.getbbox(time_text)
    time_text_width = time_text_bbox[2] - time_text_bbox[0]
    time_text_height = time_text_bbox[3] - time_text_bbox[1]
    
    text_x = 10 
    text_y = img_height - time_text_height - 10 
    draw_text_with_background(draw, (text_x, text_y), time_text, INFERENCE_FONT, "white", (0,0,0, 180))

    return draw_image, detection_outputs, inference_time

def process_with_detr(image_pil: Image.Image, confidence_threshold: float):
    if not detr_model or not detr_processor:
        raise RuntimeError("DETR Model or Processor not loaded.")

    start_time = time.time()
    inputs = detr_processor(images=image_pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = detr_model(**inputs)
    inference_time = time.time() - start_time

    target_sizes = torch.tensor([image_pil.size[::-1]], device=DEVICE)
    results = detr_processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=confidence_threshold
    )[0]

    draw_image = image_pil.copy()
    draw = ImageDraw.Draw(draw_image, "RGBA")
    detection_outputs = []

    for score, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
        box_coords = [round(i, 2) for i in box.tolist()]
        class_name = DETR_CLASSES_FROM_MODEL[label_idx.item()]
        conf = score.item()
        color = get_color_for_class(class_name)

        draw.rectangle(box_coords, outline=color, width=2)
        label = f"{class_name} ({conf:.2f})"
        
        label_y_pos = box_coords[1] - FONT.getbbox(label)[3] - 5
        if label_y_pos < 0: label_y_pos = box_coords[1] + 5

        draw_text_with_background(draw, (box_coords[0], label_y_pos), label, FONT, "white", color)

        detection_outputs.append({
            "class_name": class_name,
            "confidence": f"{conf:.2f}",
            "color": f"rgb({color[0]},{color[1]},{color[2]})"
        })

    time_text = f"Inference: {inference_time:.2f}s"
    img_width, img_height = draw_image.size
    time_text_bbox = INFERENCE_FONT.getbbox(time_text)
    time_text_width = time_text_bbox[2] - time_text_bbox[0]
    time_text_height = time_text_bbox[3] - time_text_bbox[1]
    text_x = 10
    text_y = img_height - time_text_height - 10
    draw_text_with_background(draw, (text_x, text_y), time_text, INFERENCE_FONT, "white", (0,0,0, 180))

    return draw_image, detection_outputs, inference_time

# Update AVAILABLE_MODELS list
AVAILABLE_MODELS = []
for model_name, model in yolo_models.items():
    AVAILABLE_MODELS.append({"value": model_name, "name": f"{model_name.upper()} (COCO)"})
if detr_model:
    source = "Hub"
    if DETR_MODEL_PATH.exists() and detr_model.config._name_or_path == DETR_HF_HUB_MODEL_NAME:
        source = "Local .pth"
    elif detr_model.config._name_or_path == DETR_HF_HUB_MODEL_NAME:
        source = "HuggingFace Hub"
    AVAILABLE_MODELS.append({"value": "detr_r101", "name": f"DETR R-101 (COCO) - {source}"})

# Add session storage for uploaded images
UPLOADED_IMAGES = defaultdict(dict)  # session_id -> {image_id -> image_data}
BATCH_SIZE = 5

def process_batch(images: List[Image.Image], model_choice: str, confidence_threshold: float,
                  apply_denoise: bool = False, apply_contrast: bool = False, apply_sharpen: bool = False) -> List[Dict]:
    results = []
    total_preprocessing_time_ms = 0
    try:
        preprocessed_images = []
        for img in images:
            img_to_process = img.copy()
            if apply_denoise or apply_contrast or apply_sharpen:
                prep_start_time = time.time()
                img_to_process = preprocess_image(
                    img_to_process,
                    apply_denoise=apply_denoise,
                    apply_contrast=apply_contrast,
                    apply_sharpen=apply_sharpen
                )
                total_preprocessing_time_ms += (time.time() - prep_start_time) * 1000
            preprocessed_images.append(img_to_process)
        avg_preprocessing_time_ms = total_preprocessing_time_ms / len(images) if images else 0
        if model_choice in yolo_models:
            model = yolo_models[model_choice]
            for proc_img in preprocessed_images:
                processed_img_model_output, detections, inference_time = process_with_yolo(model, proc_img, confidence_threshold)
                results.append({
                    "image": processed_img_model_output,
                    "detections": detections,
                    "inference_time": inference_time,
                    "preprocessing_time_ms": avg_preprocessing_time_ms
                })
        elif model_choice == "detr_r101" and detr_model:
            for proc_img in preprocessed_images:
                processed_img_model_output, detections, inference_time = process_with_detr(proc_img, confidence_threshold)
                results.append({
                    "image": processed_img_model_output,
                    "detections": detections,
                    "inference_time": inference_time,
                    "preprocessing_time_ms": avg_preprocessing_time_ms
                })
        else:
            raise ValueError(f"Invalid model choice: {model_choice}")
    except Exception as e:
        print(f"Error in process_batch: {e}")
        raise
    return results

@app.get("/")
async def main_page(request: Request, image_data: str = None, detections: list = None,
                    original_filename: str = None, inference_time: str = None,
                    model_used: str = None, error_message: str = None,
                    selected_model: str = "yolov8n", current_confidence: float = 0.3):
    """
    Serves the main page. Can also display results if they are passed as query parameters
    (though POST is preferred for submissions). For simplicity, we re-render `index.html`
    with results after POST.
    """
    return templates.TemplateResponse("index.html", {
        "request": request,
        "available_models": AVAILABLE_MODELS,
        "models_loaded": bool(yolo_models or detr_model),
        "image_data_b64": image_data, # For displaying processed image
        "detection_pills": detections, # For displaying pills
        "original_filename": original_filename,
        "inference_time_display": inference_time,
        "model_used_display": model_used,
        "error_message": error_message,
        "selected_model_on_load": selected_model, # To retain selection after submit
        "current_confidence_on_load": current_confidence # To retain confidence
    })

@app.get("/predict/")
async def predict_get(request: Request):
    """
    Handle GET requests to /predict/ by redirecting to the main page
    """
    return templates.TemplateResponse("index.html", {
        "request": request,
        "available_models": AVAILABLE_MODELS,
        "models_loaded": bool(yolo_models or detr_model),
        "error_message": "Please use the form to submit your image for detection.",
        "selected_model_on_load": "yolov8n",
        "current_confidence_on_load": 0.3
    })

@app.post("/predict/")
async def predict_image_endpoint(
    request: Request,
    file: UploadFile = File(None),
    model_choice: str = Form("yolov8n"),
    confidence_threshold: float = Form(0.3),
    action: str = Form("detect"),
    apply_denoise: bool = Form(False),
    apply_contrast: bool = Form(False),
    apply_sharpen: bool = Form(False)
):
    if action == "clear":
        return templates.TemplateResponse("index.html", {
            "request": request,
            "available_models": AVAILABLE_MODELS,
            "models_loaded": bool(yolo_models or detr_model),
            "selected_model_on_load": model_choice,
            "current_confidence_on_load": confidence_threshold,
            "current_denoise_on_load": apply_denoise,
            "current_contrast_on_load": apply_contrast,
            "current_sharpen_on_load": apply_sharpen,
        })

    error_msg = None
    processed_image_b64_out = None
    detections_out = []
    inference_time_out = 0.0
    preprocessing_time_ms = 0.0

    if not (yolo_models or detr_model):
        error_msg = "No models are loaded. Please check server logs."
    elif not file or not file.filename:
        error_msg = "Please upload an image file."
    elif not file.content_type.startswith("image/"):
        error_msg = "File uploaded is not a valid image."
    else:
        try:
            contents = await file.read()
            pil_image_original = Image.open(io.BytesIO(contents)).convert("RGB")
            pil_image_to_process = pil_image_original.copy()

            # --- Apply Preprocessing ---
            if apply_denoise or apply_contrast or apply_sharpen:
                prep_start_time = time.time()
                pil_image_to_process = preprocess_image(
                    pil_image_to_process,
                    apply_denoise=apply_denoise,
                    apply_contrast=apply_contrast,
                    apply_sharpen=apply_sharpen
                )
                preprocessing_time_ms = (time.time() - prep_start_time) * 1000
            # --- End Preprocessing ---

            processed_pil_image = None
            model_to_use_for_display = model_choice

            if model_choice in yolo_models:
                processed_pil_image, detections_out, inference_time_out = process_with_yolo(
                    yolo_models[model_choice], pil_image_to_process, confidence_threshold
                )
            elif model_choice == "detr_r101" and detr_model:
                processed_pil_image, detections_out, inference_time_out = process_with_detr(
                    pil_image_to_process, confidence_threshold
                )
            else:
                error_msg = f"Selected model '{model_choice}' is not available or loaded."
                model_to_use_for_display = None

            if processed_pil_image:
                buffered = io.BytesIO()
                processed_pil_image.save(buffered, format="PNG")
                processed_image_b64_out = base64.b64encode(buffered.getvalue()).decode()

        except RuntimeError as re: error_msg = str(re)
        except ValueError as ve: error_msg = str(ve)
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            import traceback
            traceback.print_exc()
            error_msg = f"An unexpected error occurred: {e}"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "available_models": AVAILABLE_MODELS,
        "models_loaded": bool(yolo_models or detr_model),
        "image_data_b64": processed_image_b64_out,
        "detection_pills": detections_out,
        "original_filename": file.filename if file else None,
        "inference_time_display": f"{inference_time_out:.2f}s" if inference_time_out > 0 else None,
        "preprocessing_time_display": f"{preprocessing_time_ms:.0f}ms" if preprocessing_time_ms > 0 else None,
        "model_used_display": model_to_use_for_display if processed_image_b64_out else None,
        "error_message": error_msg,
        "selected_model_on_load": model_choice,
        "current_confidence_on_load": confidence_threshold,
        "current_denoise_on_load": apply_denoise,
        "current_contrast_on_load": apply_contrast,
        "current_sharpen_on_load": apply_sharpen
    })

@app.post("/upload/")
async def upload_images(
    request: Request,
    files: List[UploadFile] = File(...),
    session_id: str = Form(...)
):
    """Handle multiple image uploads"""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    uploaded_data = []
    for file in files:
        if not file.content_type.startswith("image/"):
            continue
            
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Create thumbnail
        thumbnail = image.copy()
        thumbnail.thumbnail((200, 200))
        
        # Generate unique ID for this image
        image_id = str(uuid.uuid4())
        
        # Store original image and thumbnail
        UPLOADED_IMAGES[session_id][image_id] = {
            "original": image,
            "thumbnail": thumbnail,
            "filename": file.filename,
            "processed": None
        }
        
        # Convert thumbnail to base64 for display
        buffered = io.BytesIO()
        thumbnail.save(buffered, format="PNG")
        thumbnail_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        uploaded_data.append({
            "id": image_id,
            "filename": file.filename,
            "thumbnail": thumbnail_b64
        })
    
    return JSONResponse(content={"uploaded": uploaded_data})

@app.post("/process/")
async def process_images(
    request: Request,
    session_id: str = Form(...),
    model_choice: str = Form("yolov8n"),
    confidence_threshold: float = Form(0.3),
    image_ids: str = Form(...),
    apply_denoise: bool = Form(False),
    apply_contrast: bool = Form(False),
    apply_sharpen: bool = Form(False)
):
    try:
        image_ids_list = image_ids.split(',') if isinstance(image_ids, str) else []
        if not image_ids_list:
            raise HTTPException(status_code=400, detail="No images selected")
        images_to_process = []
        valid_image_ids = []
        for img_id in image_ids_list:
            if img_id in UPLOADED_IMAGES[session_id]:
                images_to_process.append(UPLOADED_IMAGES[session_id][img_id]["original"])
                valid_image_ids.append(img_id)
        if not images_to_process:
            raise HTTPException(status_code=400, detail="No valid images found to process")
        results = []
        for i in range(0, len(images_to_process), BATCH_SIZE):
            batch = images_to_process[i:i + BATCH_SIZE]
            try:
                batch_results = process_batch(batch, model_choice, confidence_threshold,
                                              apply_denoise, apply_contrast, apply_sharpen)
                results.extend(batch_results)
            except Exception as e:
                print(f"Error processing batch: {e}")
                raise HTTPException(status_code=500, detail=f"Error processing images: {str(e)}")
        processed_data = []
        for img_id, result in zip(valid_image_ids, results):
            if img_id in UPLOADED_IMAGES[session_id]:
                UPLOADED_IMAGES[session_id][img_id]["processed"] = result
                buffered = io.BytesIO()
                result["image"].save(buffered, format="PNG")
                image_b64 = base64.b64encode(buffered.getvalue()).decode()
                processed_data.append({
                    "id": img_id,
                    "image": image_b64,
                    "detections": result["detections"],
                    "inference_time": f"{result['inference_time']:.2f}s",
                    "preprocessing_time": f"{result.get('preprocessing_time_ms', 0):.0f}ms"
                })
        return JSONResponse(content={"processed": processed_data})
    except Exception as e:
        print(f"Error in process_images: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images/{session_id}")
async def get_session_images(session_id: str):
    if session_id not in UPLOADED_IMAGES:
        return JSONResponse(content={"images": []})
    images_data = []
    for img_id, img_data in UPLOADED_IMAGES[session_id].items():
        buffered = io.BytesIO()
        img_data["thumbnail"].save(buffered, format="PNG")
        thumbnail_b64 = base64.b64encode(buffered.getvalue()).decode()
        processed_b64 = None
        if img_data["processed"]:
            buffered = io.BytesIO()
            img_data["processed"]["image"].save(buffered, format="PNG")
            processed_b64 = base64.b64encode(buffered.getvalue()).decode()
        prep_time_ms = None
        if img_data["processed"] and "preprocessing_time_ms" in img_data["processed"]:
            prep_time_ms = f"{img_data['processed']['preprocessing_time_ms']:.0f}ms"
        images_data.append({
            "id": img_id,
            "filename": img_data["filename"],
            "thumbnail": thumbnail_b64,
            "processed": processed_b64,
            "detections": img_data["processed"]["detections"] if img_data["processed"] else None,
            "inference_time": f"{img_data['processed']['inference_time']:.2f}s" if img_data["processed"] else None,
            "preprocessing_time": prep_time_ms
        })
    return JSONResponse(content={"images": images_data})

@app.delete("/images/{session_id}/{image_id}")
async def delete_image(session_id: str, image_id: str):
    """Delete an image from the session"""
    if session_id in UPLOADED_IMAGES and image_id in UPLOADED_IMAGES[session_id]:
        del UPLOADED_IMAGES[session_id][image_id]
        return JSONResponse(content={"status": "success"})
    return JSONResponse(content={"status": "not found"}, status_code=404)

@app.delete("/clear/{session_id}")
async def clear_session(session_id: str):
    """Clear all images for a session"""
    if session_id in UPLOADED_IMAGES:
        UPLOADED_IMAGES[session_id].clear()
        return JSONResponse(content={"status": "success"})
    return JSONResponse(content={"status": "not found"}, status_code=404)

@app.post("/save/")
async def save_detected_images(
    session_id: str = Form(...),
    model_choice: str = Form(...)
):
    """Save all processed images for a session to detected_images folder."""
    save_dir = Path("detected_images")
    save_dir.mkdir(exist_ok=True)
    if session_id not in UPLOADED_IMAGES:
        return JSONResponse(content={"status": "no session"}, status_code=404)
    saved_files = []
    for img_id, img_data in UPLOADED_IMAGES[session_id].items():
        if img_data.get("processed"):
            original_name = img_data["filename"].rsplit('.', 1)[0]
            save_name = f"{original_name}_{model_choice}.jpg"
            save_path = save_dir / save_name
            img_data["processed"]["image"].save(save_path, format="JPEG")
            saved_files.append(str(save_path))
    return JSONResponse(content={"status": "success", "saved": saved_files})

if __name__ == "__main__":
    import uvicorn
    # Ensure models are loaded before starting app, especially if load_models() wasn't called or failed partially
    # load_models() # This is already called at the global scope. Let's re-evaluate if it needs to be here.
    
    # Repopulate AVAILABLE_MODELS based on what actually loaded, in case load_models was called earlier
    # and something changed (e.g., file downloaded). This ensures the UI is up-to-date.
    AVAILABLE_MODELS.clear() 
    if yolo_models:
        for model_name, model in yolo_models.items():
            AVAILABLE_MODELS.append({"value": model_name, "name": f"{model_name.upper()} (COCO)"})
    if detr_model:
        source = "Unknown"
        # Check if the currently loaded detr_model matches what we would expect from local path
        # This is a heuristic. A more robust way would be to store the source during loading.
        # For now, if the local path exists AND detr_model is loaded, assume it's local.
        if DETR_MODEL_PATH.exists() and detr_model.config._name_or_path == DETR_HF_HUB_MODEL_NAME:
             # If loaded from local, its config might still point to HF hub if we initialized from there.
             # A better check might be to see if the loading message for local .pth was successful earlier.
             # For simplicity here, we assume if .pth exists and model loaded, it was from .pth
             source = "Local .pth"
        elif detr_model.config._name_or_path == DETR_HF_HUB_MODEL_NAME:
            source = "HuggingFace Hub"
        AVAILABLE_MODELS.append({"value": "detr_r101", "name": f"DETR R-101 (COCO) - {source}"})

    if not (yolo_models or detr_model):
         print("Warning: No models were loaded successfully. The application might not function as expected.")
    uvicorn.run(app, host="0.0.0.0", port=8000)