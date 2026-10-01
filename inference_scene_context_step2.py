"""
Scene Context Analysis -- Inference & Evaluation (Step 2)

Description:
    This script analyzes the broader scene context of video clips from the JAAD
    dataset. Unlike pedestrian-focused analysis, this script evaluates the
    entire frame to classify environmental and situational attributes using a
    suite of specialized Vision Transformer (ViT) adapter models.

    The script performs inference on the following scene contextual attributes:
    -   WEATHER: `RAINY`, `SNOWY`, `CLOUDY`, `SUNNY`
    -   TIME_OF_DAY: `NIGHT`, `DAY`
    -   PEDESTRIAN_DENSITY: `HIGH_PEDESTRIAN_DENSITY`, `MEDIUM_PEDESTRIAN_DENSITY`, `LOW_PEDESTRIAN_DENSITY`
    -   ROAD_PRESENCE: `NARROW_ROAD`, `WIDE_ROAD`

    The analysis is divided into two types:
    -   Static Analysis: For attributes assumed to be constant throughout a short
        video (e.g., weather, time of day), the script samples the first few
        frames and uses a majority vote to assign a single label for the video.
    -   Dynamic Analysis: For attributes that can change frame-by-frame (e.g.,
        pedestrian density, road presence), the script runs inference on every
        processed frame.

    For comparison, this script also runs a fine-tuned YOLO model to perform
    a rule-based pedestrian density classification, which is then compared
    against the dedicated `ped_density` adapter's performance.

The pipeline operates as follows:
    1.  Loads configurations for all scene-based adapter models and the YOLO model.
    2.  For each video, performs static analysis on the initial frames.
    3.  Processes the video frame-by-frame (respecting the `FRAME_SKIP` setting).
    4.  In each frame, runs dynamic adapter models on the full image.
    5.  In parallel, runs YOLO to get a pedestrian count for a baseline density
        classification.
    6.  Compiles results, including adapter predictions, YOLO-based density,
        confidences, and performance metrics (e.g., FPS, GFLOPs), into a row.
    7.  Aggregates results from all videos and saves them to a single CSV file
        for further analysis in Step 3.

Inputs:
    - Fine-tuned YOLO Model:
        - ./Best_Model_Configs/yolo_best_model.pt
    - Video Clips:
        - ./JAAD/JAAD_clips/*.mp4
    - Fine-tuned PEFT Adapter Models:
        - ./Best_Model_Configs/scene_contextual_final_models/...

Outputs:
    - Raw Data CSV for Analysis (for Step 3):
        - raw_data_scene_contextual_analysis_step2.csv
"""

import os
import sys
import time
import json
import logging
from contextlib import contextmanager
from typing import Dict, List, Tuple

import cv2
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import ViTForImageClassification, AutoImageProcessor
from peft import PeftModel
from fvcore.nn import FlopCountAnalysis
from ultralytics import YOLO
import numpy as np
from pathlib import Path

# =============================================================================
# A1: Utility Helpers
# =============================================================================

logging.getLogger("transformers").setLevel(logging.ERROR)

@contextmanager
def suppress_stdout_stderr():
    """Suppresses stdout/stderr for verbose operations like FLOP calculation."""

    with open(os.devnull, "w") as fnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

def load_mappings(mapping_path: str) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Load label to id and id to label mappings from JSON file."""

    with open(mapping_path, "r") as fp:
        data = json.load(fp)
    return data["label2id"], data["id2label"]

# =============================================================================
# A2: Model Loading & Inference
# =============================================================================

def load_adapter_model(base_model_name: str,
                       adapter_path: str,
                       num_labels: int,
                       device: torch.device) -> Tuple[torch.nn.Module, AutoImageProcessor]:
    """Load ViT base model + PEFT adapter for scene contextual analysis."""
    
    base_model = ViTForImageClassification.from_pretrained(
        base_model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)

    def _patched_forward(self, pixel_values=None, **kwargs):
        return self.base_model(pixel_values=pixel_values, **kwargs)

    import types
    model.forward = types.MethodType(_patched_forward, model)

    model.eval()
    model.to(device)

    processor = AutoImageProcessor.from_pretrained(base_model_name, use_fast=True)
    return model, processor

def predict_single_frame(model: torch.nn.Module,
                        processor: AutoImageProcessor,
                        frame: np.ndarray,
                        device: torch.device) -> Tuple[int, float, List[float]]:
    """Predict scene contextual attribute for a single frame.
    
    Returns:
        prediction: Predicted class index
        confidence: Confidence score (max probability)
        probabilities: Full probability distribution
    """

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        logits = model(**inputs).logits  
        probabilities = torch.softmax(logits, dim=-1)
        
        confidence, prediction = torch.max(probabilities, dim=-1)
        
        return (prediction.item(), 
                confidence.item(),
                probabilities.squeeze().cpu().tolist())

def compute_model_flops(model: torch.nn.Module,
                        processor: AutoImageProcessor,
                        device: torch.device) -> float:
    """Calculate theoretical GFLOPs for a single forward pass."""

    model.eval()
    dummy_img = Image.new("RGB", (224, 224))
    dummy_input = processor(images=dummy_img, return_tensors="pt")["pixel_values"].to(device)

    with suppress_stdout_stderr():
        try:
            flops = FlopCountAnalysis(model, (dummy_input,))
            return flops.total() / 1e9  
        except Exception:
            return -1.0

# =============================================================================
# A3: YOLO-Based Pedestrian Density Analysis
# =============================================================================

def calculate_pedestrian_density(frame_data: List[Dict]) -> str:
    """Calculate pedestrian density based on number of bounding boxes in frame."""
    
    cnt = len(frame_data)
    if cnt > 7:
        return 'high_pedestrian_density'
    elif 4 <= cnt <= 7:
        return 'medium_pedestrian_density'
    return 'low_pedestrian_density'

def yolo_pedestrian_density_analysis(frame: np.ndarray, yolo_model: YOLO) -> Tuple[int, str, float]:
    """Analyze pedestrian density using YOLO detection + rule-based classification.
    
    Uses only 'pedestrian' and 'ped' classes, excluding broader 'person' class
    for accurate pedestrian-specific counting and fair comparison with adapters.
    
    Args:
        frame: Input frame (BGR format)
        yolo_model: Loaded YOLO model
        
    Returns:
        pedestrian_count: Number of pedestrians detected (pedestrian + ped classes only)
        density_classification: low/medium/high_pedestrian_density
        inference_time_ms: Time taken for detection + classification
    """
    
    start_time = time.time()
    
    allowed_labels = {"pedestrian", "ped"}  
    results = yolo_model(frame, verbose=False)
    
    pedestrian_boxes = []
    if results and len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            cls_idx = int(boxes.cls[i])
            class_name = yolo_model.names[cls_idx].lower()
            if class_name in allowed_labels:
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                pedestrian_boxes.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(boxes.conf[i]) if boxes.conf is not None else 1.0
                })
    
    pedestrian_count = len(pedestrian_boxes)
    density_classification = calculate_pedestrian_density(pedestrian_boxes)
    
    inference_time_ms = (time.time() - start_time) * 1000.0
    
    return pedestrian_count, density_classification, inference_time_ms

# =============================================================================
# A4: Scene Contextual Analysis Functions
# =============================================================================

def analyze_video_static_attributes(video_path: str,
                                   static_models: Dict[str, Dict],
                                   sample_frames: int = 10) -> Dict[str, Dict]:
    """Analyze static attributes (weather, time_of_day) from first few frames of video.
    
    Args:
        video_path: Path to video file
        static_models: Dictionary containing weather and time_of_day model configs
        sample_frames: Number of initial frames to sample for analysis
        
    Returns:
        Dictionary with static attribute predictions and confidence scores
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    results = {}
    
    for attr_name, model_config in static_models.items():
        model = model_config["model"]
        processor = model_config["processor"]
        id2label = model_config["id2label"]
        device = next(model.parameters()).device
        
        predictions = []
        confidences = []
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  
        
        for frame_idx in range(sample_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            pred_idx, confidence, probabilities = predict_single_frame(
                model, processor, frame, device
            )
            
            predictions.append(pred_idx)
            confidences.append(confidence)
        
        if predictions:
            from collections import Counter
            pred_counts = Counter(predictions)
            final_pred_idx = pred_counts.most_common(1)[0][0]
            final_pred_label = id2label[str(final_pred_idx)]
            avg_confidence = np.mean(confidences)
            
            results[attr_name] = {
                "prediction": final_pred_label,
                "confidence": avg_confidence,
                "pred_idx": final_pred_idx,
                "sample_count": len(predictions)
            }
        else:
            results[attr_name] = {
                "prediction": "unknown",
                "confidence": 0.0,
                "pred_idx": -1,
                "sample_count": 0
            }
    
    cap.release()
    return results

def infer_scene_context_on_video(video_path: str,
                                adapters_cfg: Dict[str, Dict[str, str]],
                                vit_base_model: str,
                                yolo_model_path: str,
                                frame_skip: int = 1) -> pd.DataFrame:
    """Perform scene contextual inference on a single video.
    
    Args:
        video_path: Path to video file
        adapters_cfg: Configuration for all scene contextual adapters
        vit_base_model: Base ViT model name
        yolo_model_path: Path to YOLO model for pedestrian detection
        frame_skip: Process every Nth frame (1 = every frame)
        
    Returns:
        DataFrame with scene contextual and YOLO comparison results
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_id = video_name.replace("video_", "")
    
    # ==========================================================================
    # B1: Device & Model Preparation
    # ==========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    yolo_model = YOLO(yolo_model_path).to(device)
    
    adapter_models = {}
    flops_metrics = {}
    
    for attr_name, cfg in adapters_cfg.items():
        label2id, id2label = load_mappings(cfg["mappings"])
        model, processor = load_adapter_model(vit_base_model, cfg["path"], len(label2id), device)
        
        adapter_models[attr_name] = {
            "model": model,
            "processor": processor,
            "id2label": id2label,
        }
        flops_metrics[f"gflops_{attr_name}"] = compute_model_flops(model, processor, device)
    
    # ==========================================================================
    # B2: Static Attribute Analysis (Weather, Time of Day)
    # ==========================================================================
    static_adapters = ["weather", "time_of_day"]
    static_models = {attr: adapter_models[attr] for attr in static_adapters if attr in adapter_models}
    
    static_results = analyze_video_static_attributes(video_path, static_models, sample_frames=15)
    
    # ==========================================================================
    # B3: Dynamic Attribute Analysis (Pedestrian Density, Road Presence)
    # ==========================================================================
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    results: List[Dict] = []
    frame_idx = 0
    processed_frames = 0
    
    dynamic_adapters = ["ped_density", "road_presence"]
    
    pbar = tqdm(total=total_frames, unit="frame", desc=f"Scene Analysis {video_id}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            pbar.update(1)
            continue
            
        frame_start = time.time()
        
        row = {
            "video_id": video_id,
            "frame_id": frame_idx,
            "video_fps": fps,
            "frame_timestamp": frame_idx / fps if fps > 0 else 0.0,
        }
        
        for attr_name in static_adapters:
            if attr_name in static_results:
                static_result = static_results[attr_name]
                row[f"pred_{attr_name}"] = static_result["prediction"]
                row[f"pred_{attr_name}_confidence"] = static_result["confidence"]
                row[f"pred_{attr_name}_sample_count"] = static_result["sample_count"]
            else:
                row[f"pred_{attr_name}"] = "unknown"
                row[f"pred_{attr_name}_confidence"] = 0.0
                row[f"pred_{attr_name}_sample_count"] = 0
        
        for attr_name in dynamic_adapters:
            if attr_name in adapter_models:
                model_config = adapter_models[attr_name]
                
                adapter_start = time.time()
                pred_idx, confidence, probabilities = predict_single_frame(
                    model_config["model"], 
                    model_config["processor"], 
                    frame, 
                    device
                )
                adapter_time_ms = (time.time() - adapter_start) * 1000.0
                
                prediction_label = model_config["id2label"][str(pred_idx)]
                
                row[f"pred_{attr_name}"] = prediction_label
                row[f"pred_{attr_name}_confidence"] = confidence
                row[f"pred_{attr_name}_probabilities"] = probabilities
                row[f"pred_{attr_name}_inference_time_ms"] = adapter_time_ms
            else:
                row[f"pred_{attr_name}"] = "unknown"
                row[f"pred_{attr_name}_confidence"] = 0.0
                row[f"pred_{attr_name}_probabilities"] = []
                row[f"pred_{attr_name}_inference_time_ms"] = 0.0
        
        ped_count, yolo_density, yolo_time_ms = yolo_pedestrian_density_analysis(frame, yolo_model)
        row["yolo_pedestrian_count"] = ped_count
        row["yolo_ped_density"] = yolo_density
        row["yolo_ped_density_inference_time_ms"] = yolo_time_ms
        
        frame_time_ms = (time.time() - frame_start) * 1000.0
        row["frame_processing_time_ms"] = frame_time_ms
        row["frame_fps_processed"] = 1000.0 / frame_time_ms if frame_time_ms > 0 else 0.0
        
        row.update(flops_metrics)
        
        results.append(row)
        processed_frames += 1
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    df = pd.DataFrame(results)
    
    if not df.empty:
        df["total_processed_frames"] = processed_frames
        df["processing_fps"] = processed_frames / df["frame_processing_time_ms"].sum() * 1000 if not df.empty else 0.0
    
    return df

# =============================================================================
# C1: Main Execution Flow
# =============================================================================

def main():
    # ==========================================================================
    # Configuration - Adjust paths as needed
    # ==========================================================================
    VIT_BASE = "google/vit-base-patch16-224-in21k"
    VIDEO_FOLDER = "./JAAD/JAAD_clips"
    OUTPUT_CSV_PATH = os.path.join("Generated_Data", "raw_data_scene_contextual_analysis_step2.csv")
    YOLO_MODEL_PATH = "./Best_Model_Configs/yolo_best_model.pt"
    
    adapters = {
        "ped_density": {
            "path": "./Best_Model_Configs/scene_contextual_final_models/PED_DENSITY_ADAPTER_MODEL_lora",
            "mappings": "./Best_Model_Configs/scene_contextual_final_models/PED_DENSITY_ADAPTER_MODEL_lora/mappings.json",
        },
        "weather": {
            "path": "./Best_Model_Configs/scene_contextual_final_models/WEATHER_ADAPTER_MODEL_lora",
            "mappings": "./Best_Model_Configs/scene_contextual_final_models/WEATHER_ADAPTER_MODEL_lora/mappings.json",
        },
        "time_of_day": {
            "path": "./Best_Model_Configs/scene_contextual_final_models/TIME_OF_DAY_ADAPTER_MODEL_lora",
            "mappings": "./Best_Model_Configs/scene_contextual_final_models/TIME_OF_DAY_ADAPTER_MODEL_lora/mappings.json",
        },
        "road_presence": {
            "path": "./Best_Model_Configs/scene_contextual_final_models/ROAD_PRESENCE_ADAPTER_MODEL_lora",
            "mappings": "./Best_Model_Configs/scene_contextual_final_models/ROAD_PRESENCE_ADAPTER_MODEL_lora/mappings.json",
        },
    }
    
    # ==========================================================================
    # Video Processing Configuration
    # ==========================================================================
    FRAME_SKIP = 1  # Process every frame (set to 5 or 10 to skip frames for faster processing)
    MAX_VIDEOS = None  # Set to a number for testing, None for all videos
    
    # ==========================================================================
    # Input Validation
    # ==========================================================================
    if not os.path.isdir(VIDEO_FOLDER):
        raise NotADirectoryError(f"Video folder not found: {VIDEO_FOLDER}")
    
    Path("Generated_Data").mkdir(exist_ok=True)
    
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"YOLO model not found: {YOLO_MODEL_PATH}")
    
    for attr_name, cfg in adapters.items():
        if not os.path.exists(cfg["path"]):
            print(f" Warning: Adapter path not found: {cfg['path']}")
            print(f"   Skipping {attr_name} adapter")
            continue
        if not os.path.exists(cfg["mappings"]):
            print(f" Warning: Mappings file not found: {cfg['mappings']}")
            print(f"   Skipping {attr_name} adapter")
            continue
    
    # ==========================================================================
    # Video File Discovery
    # ==========================================================================
    video_files = sorted([f for f in os.listdir(VIDEO_FOLDER) if f.endswith(".mp4")])
    if not video_files:
        raise RuntimeError(f"No *.mp4 videos found in {VIDEO_FOLDER}")
    
    if MAX_VIDEOS is not None:
        video_files = video_files[:MAX_VIDEOS]
        print(f" Testing mode: processing only the first {len(video_files)} videos")
        print(f"   Set MAX_VIDEOS=None for full dataset processing")
    
    print(f"Found {len(video_files)} video files to process")
    print(f"Frame skip: {FRAME_SKIP} (processing every {FRAME_SKIP} frame{'s' if FRAME_SKIP > 1 else ''})")
    print(f"Adapters loaded: {list(adapters.keys())}")
    print(f"YOLO model: {YOLO_MODEL_PATH}")
    print(f"Analysis: Adapter-based vs YOLO-based pedestrian density comparison")
    
    # ==========================================================================
    # Process All Videos
    # ==========================================================================
    all_results = []
    
    for i, video_file in enumerate(video_files, 1):
        video_path = os.path.join(VIDEO_FOLDER, video_file)
        print(f"\n [{i}/{len(video_files)}] Processing {video_file}...")
        
        try:
            df_video = infer_scene_context_on_video(
                video_path, 
                adapters, 
                VIT_BASE,
                YOLO_MODEL_PATH,
                frame_skip=FRAME_SKIP
            )
            
            if not df_video.empty:
                all_results.append(df_video)
                print(f"   Processed {len(df_video)} frames")
            else:
                print(f"   No frames processed for {video_file}")
                
        except Exception as e:
            print(f"   Error processing {video_file}: {str(e)}")
            continue
    
    # ==========================================================================
    # Save Results
    # ==========================================================================
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(OUTPUT_CSV_PATH, index=False)
        
        print(f"\n Scene contextual analysis complete!")
        print(f" Total frames processed: {len(final_df):,}")
        print(f" Results saved to: '{OUTPUT_CSV_PATH}'")
        
        if len(final_df) > 0:
            total_videos = final_df['video_id'].nunique()
            avg_processing_time = final_df['frame_processing_time_ms'].mean()
            
            print(f"\n Processing Summary:")
            print(f"   Videos processed: {total_videos}")
            print(f"   Average frame processing time: {avg_processing_time:.2f} ms")
            print(f"   Average FPS: {1000/avg_processing_time:.1f} fps")
            
            for attr in ["weather", "time_of_day", "ped_density", "road_presence"]:
                pred_col = f"pred_{attr}"
                if pred_col in final_df.columns:
                    unique_preds = final_df[pred_col].value_counts().head(3)
                    print(f"   {attr.upper()} distribution: {dict(unique_preds)}")
            
            if "pred_ped_density" in final_df.columns and "yolo_ped_density" in final_df.columns:
                adapter_avg_time = final_df["pred_ped_density_inference_time_ms"].mean()
                yolo_avg_time = final_df["yolo_ped_density_inference_time_ms"].mean()
                avg_ped_count = final_df["yolo_pedestrian_count"].mean()
                
                print(f"\n   PEDESTRIAN DENSITY COMPARISON:")
                print(f"   Average pedestrians per frame: {avg_ped_count:.1f} (pedestrian + ped classes only)")
                print(f"   Adapter inference time: {adapter_avg_time:.2f} ms")
                print(f"   YOLO inference time: {yolo_avg_time:.2f} ms")
                print(f"   Speed difference: {yolo_avg_time/adapter_avg_time:.2f}x {'(YOLO slower)' if yolo_avg_time > adapter_avg_time else '(YOLO faster)'}")
                
                agreement = (final_df["pred_ped_density"] == final_df["yolo_ped_density"]).mean()
                print(f"   Classification agreement: {agreement:.3f} ({agreement*100:.1f}%)")
    else:
        print("\n No results to save - all videos failed processing")

if __name__ == "__main__":
    main()
