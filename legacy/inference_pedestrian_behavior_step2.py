"""
Pedestrian Behavior Analysis -- Inference & Evaluation (Step 2)

Description:
    This script orchestrates a two-stage, end-to-end inference process for
    analyzing pedestrian behavior in video clips.

    Stage 1: Pedestrian Detection
      - A fine-tuned YOLO model processes each video frame to predict the
        location (bounding box) of every pedestrian.

    Stage 2: Behavior Classification
      - The bounding box for each detected pedestrian is cropped from the frame.
      - These cropped images are then passed to a suite of fine-tuned Vision
        Transformer (ViT) adapter models, which classify multiple behavioral
        attributes.

    Classified Attributes:
    -   ACTION: `standing`, `walking`
    -   LOOK: `looking`, `not_looking`
    -   CROSS: `crossing`, `not_crossing`
    -   OCCLUSION: `no_occlusion`, `partial_occlusion`, `full_occlusion`

The pipeline operates as follows:
    1.  Loads a ground-truth (GT) CSV file containing annotations.
    2.  For each video, it processes frame by frame.
    3.  In each frame, it detects pedestrians using either YOLO or GT boxes (depending on the BBOX_SOURCE variable set by the user).
    4.  The detected bounding boxes are extended by 5% to provide more context to the adapter models.
    5.  These extended regions are cropped and passed in batches to the behavior
        adapter models for prediction.
    6.  The location predictions (bounding boxes) from YOLO are matched with
        ground-truth annotations using an Intersection over Union (IoU) threshold
        to validate the detection before evaluating the behavior classification.
    7.  A comprehensive set of raw data, including predictions, confidences,
        GT labels, and performance metrics (e.g., FPS, GFLOPs), is compiled.
    8.  The final results from all videos are aggregated and saved to a single
        CSV file for further analysis in Step 3.

Inputs:
    - Fine-tuned YOLO Model:
        - ./Best_Model_Configs/yolo_best_model.pt
    - Video Clips:
        - ./JAAD/JAAD_clips/*.mp4
    - Ground-Truth Annotations CSV (from Step 1):
        - jaad_annotations_extracted_data_step1.csv
    - Fine-tuned PEFT Adapter Models:
        - ./Best_Model_Configs/pedestrian_behavior_final_models/...

Outputs:
    - Raw Data CSV for Analysis (for Step 3):
        - raw_data_pedestrian_behavior_step2.csv
"""

import os
import sys
import time
import json
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
# A. UTILITY HELPERS
# =============================================================================

@contextmanager
def suppress_stdout_stderr():
    """Suppresses stdout/stderr - useful around verbose FLOP calculation."""
    with open(os.devnull, "w") as fnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = fnull, fnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


def load_mappings(mapping_path: str) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Load label <-> id mappings stored by `jaad_annotations_extraction_step1.py`."""
    with open(mapping_path, "r") as fp:
        data = json.load(fp)
    return data["label2id"], data["id2label"]

# =============================================================================
# B. MODEL LOADING & INFERENCE HELPERS
# =============================================================================

def load_adapter_model(base_model_name: str,
                       adapter_path: str,
                       num_labels: int,
                       device: torch.device) -> Tuple[torch.nn.Module, AutoImageProcessor]:
    """Load ViT base model + PEFT adapter, patch forward, and send to device."""

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


def predict_batch(model: torch.nn.Module,
                  processor: AutoImageProcessor,
                  crops: List[Image.Image],
                  device: torch.device) -> Tuple[List[int], List[float], List[List[float]]]:
    """Batched prediction returning class indices, confidence scores, and full probability distributions.
    
    Returns:
        predictions: List of predicted class indices
        confidences: List of confidence scores (max probability for each prediction)
        probabilities: List of full probability distributions for each sample
    """
    if not crops:
        return [], [], []

    inputs = processor(images=crops, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)
        confidences, preds = torch.max(probabilities, dim=-1)
        
        return (preds.cpu().tolist(), 
                confidences.cpu().tolist(),
                probabilities.cpu().tolist())


def calculate_iou(box_a: List[int], box_b: List[int]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    
    xA, yA = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xB, yB = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])

    inter_w, inter_h = max(0, xB - xA), max(0, yB - yA)
    inter_area = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def compute_model_flops(model: torch.nn.Module,
                        processor: AutoImageProcessor,
                        device: torch.device) -> float:
    """Return theoretical GFLOPs for a single forward pass on a dummy image."""
    
    model.eval()
    dummy_img = Image.new("RGB", (224, 224))
    dummy_input = processor(images=dummy_img, return_tensors="pt")["pixel_values"].to(device)

    with suppress_stdout_stderr():
        try:
            flops = FlopCountAnalysis(model, (dummy_input,))
            return flops.total() / 1e9
        except Exception:
            return -1.0


def extend_single_bbox(box: List[int], frame_shape: Tuple[int, int], delta: float = 0.05) -> List[int]:
    """Extend a single bounding box by a given percentage, ensuring it stays within frame boundaries."""
    H, W = frame_shape
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    
    new_x1 = max(0, x1 - delta * w)
    new_y1 = max(0, y1 - delta * h)
    new_x2 = min(W, x2 + delta * w)
    new_y2 = min(H, y2 + delta * h)
    
    return [int(c) for c in [new_x1, new_y1, new_x2, new_y2]]


# =============================================================================
# C. GLOBAL CONFIGURATION
# =============================================================================
# Specifies the source for bounding boxes: "yolo" for detection-based tracking
# or "gt" to use ground-truth boxes with an ROI extension.
BBOX_SOURCE = "yolo"


# =============================================================================
# D. CORE EVALUATION LOOP
# =============================================================================

def infer_and_evaluate_on_video(video_path: str,
                                adapters_cfg: Dict[str, Dict[str, str]],
                                vit_base_model: str,
                                yolo_model_path: str,
                                gt_df: pd.DataFrame,
                                iou_threshold: float = 0.3) -> pd.DataFrame:
    """Run end-to-end inference on one video and return a DataFrame of results."""
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_id   = video_name.replace("video_", "")                 

    gt_df['video_id'] = gt_df['video_id'].astype(str).str.zfill(4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = YOLO(yolo_model_path).to(device)
    allowed_labels = {"pedestrian"} # "ped", "people" are other classes but we dont want their behavior

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

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    results: List[Dict] = []
    frame_idx = 0
    total_detections = 0
    total_gt_boxes   = 0
    total_matched    = 0
    skipped_frames   = 0

    pbar = tqdm(total=total_frames, unit="frame", desc=f"{video_id}")
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame_start = time.time()

        crops: List[Image.Image] = []
        bboxes: List[List[int]] = []
        track_ids: List[int] = []

        if BBOX_SOURCE == "yolo":
            yolo_out = yolo.track(frame, verbose=False, persist=True)
            detections = yolo_out[0].boxes 

            if detections.id is not None:
                H, W, _ = frame.shape
                for i in range(len(detections)):
                    cls_idx = int(detections.cls[i])
                    if yolo.names[cls_idx] not in allowed_labels:
                        continue

                    # Original box is stored for accurate IoU matching against GT
                    original_bbox = detections.xyxy[i].cpu().numpy().astype(int).tolist()
                    
                    # Extended box is used for cropping to give the model more context
                    extended_bbox = extend_single_bbox(original_bbox, (H, W), delta=0.05)
                    ex1, ey1, ex2, ey2 = extended_bbox
                    
                    crop = frame[ey1:ey2, ex1:ex2]
                    if crop.size == 0:
                        continue
                    
                    crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                    bboxes.append(original_bbox) # Use original for IoU
                    track_ids.append(int(detections.id[i].item()))
        else:
            gt_frame_boxes = gt_df[(gt_df['video_id'] == video_id) & (gt_df['frame_id'] == frame_idx)]
            for _, gt_row in gt_frame_boxes.iterrows():
                x1, y1, x2, y2 = gt_row["extended_bbox"]
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                bboxes.append([x1, y1, x2, y2])
                track_ids.append(int(gt_row.name)) 

        total_detections += len(crops)

        batch_preds: Dict[str, List[str]] = {}
        batch_confidences: Dict[str, List[float]] = {}
        batch_probabilities: Dict[str, List[List[float]]] = {}
        
        if crops:
            for attr_name, m in adapter_models.items():
                pred_indices, confidences, probabilities = predict_batch(m["model"], m["processor"], crops, device)
                batch_preds[attr_name] = [m["id2label"][str(idx)] for idx in pred_indices]
                batch_confidences[attr_name] = confidences
                batch_probabilities[attr_name] = probabilities
        else:
            frame_idx += 1
            pbar.update(1)
            continue

        gt_allowed = {"pedestrian"}
        gt_frame = gt_df[(gt_df['video_id'] == video_id) &
                 (gt_df['frame_id'] == frame_idx) &
                 (gt_df['label'].isin(gt_allowed))]
        gt_records = gt_frame.to_dict("records")

        if not gt_records:
            skipped_frames += 1
            frame_idx += 1
            pbar.update(1)
            continue

        total_gt_boxes += len(gt_records)
        used_gt = set()

        matched: List[Tuple[int, Dict[str, str], Dict]] = []  
        for pred_i, bbox_pred in enumerate(bboxes):
            best_iou, best_gt_idx = -1.0, -1
            for i, gt in enumerate(gt_records):
                if i in used_gt:
                    continue
                iou = calculate_iou(bbox_pred, gt["bounding_box"])
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, i
            if best_iou >= iou_threshold and best_gt_idx != -1:
                used_gt.add(best_gt_idx)
                matched.append((pred_i, {attr: batch_preds[attr][pred_i] for attr in adapters_cfg}, gt_records[best_gt_idx]))

        total_matched += len(matched)

        total_frame_ms = (time.time() - frame_start) * 1000.0
        fps = 1000.0 / total_frame_ms if total_frame_ms > 0 else 0.0
        num_peds = len(crops)

        for pred_idx, pred_labels, gt_row in matched:
            row = {
                "video_id": video_id,
                "frame_id": frame_idx,
                "inference_track_id": track_ids[pred_idx],
                "num_pedestrians_in_frame": num_peds,
                "total_frame_time_ms": total_frame_ms,
                "frame_fps": fps,
                "iou_match_score": calculate_iou(bboxes[pred_idx], gt_row["bounding_box"]),
                "pred_bbox": bboxes[pred_idx],
                "gt_bbox": gt_row["bounding_box"],
            }
            for attr in adapters_cfg.keys():
                row[f"pred_{attr}"] = pred_labels[attr]
                row[f"pred_{attr}_confidence"] = batch_confidences[attr][pred_idx]
                row[f"pred_{attr}_probabilities"] = batch_probabilities[attr][pred_idx]
                row[f"gt_{attr}"] = gt_row.get(attr)

            row.update(flops_metrics)

            # In the ground truth data, the occlusion parameter is not present for "pedestrians" but only for "ped" class
            if gt_row["label"] == "pedestrian":
                row["gt_occlusion"] = "no_occlusion"     
            elif gt_row["label"] == "ped":               
                for k in ("gt_action", "gt_look", "gt_cross"):
                    row[k] = None

            results.append(row)

        if not matched and crops:
            for pred_i in range(len(crops)):
                row = {
                    "video_id": video_id,
                    "frame_id": frame_idx,
                    "inference_track_id": track_ids[pred_i],
                    "num_pedestrians_in_frame": num_peds,
                    "total_frame_time_ms": total_frame_ms,
                    "frame_fps": fps,
                    "iou_match_score": np.nan,
                    "pred_bbox": bboxes[pred_i],
                    "gt_bbox": np.nan,
                }
                for attr in adapters_cfg.keys():
                    row[f"pred_{attr}"] = batch_preds[attr][pred_i]
                    row[f"pred_{attr}_confidence"] = batch_confidences[attr][pred_i]
                    row[f"pred_{attr}_probabilities"] = batch_probabilities[attr][pred_i]
                    row[f"gt_{attr}"] = np.nan
                row.update(flops_metrics)
                results.append(row)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    return pd.DataFrame(results)

# =============================================================================
# E. SCRIPT ENTRYPOINT
# =============================================================================


def main():
    YOLO_MODEL_PATH = "./Best_Model_Configs/yolo_best_model.pt"
    VIT_BASE = "google/vit-base-patch16-224-in21k"
    VIDEO_FOLDER = "./JAAD/JAAD_clips" 
    GT_CSV_PATH = os.path.join("Generated_Data", "jaad_annotations_extracted_data_step1.csv")
    OUTPUT_CSV_PATH = os.path.join("Generated_Data", "raw_data_pedestrian_behavior_step2.csv")

    adapters = {
        "action": {
            "path": "./Best_Model_Configs/pedestrian_behavior_final_models/ACTION_ADAPTER_MODEL_loha",
            "mappings": "./Best_Model_Configs/pedestrian_behavior_final_models/ACTION_ADAPTER_MODEL_loha/mappings.json",
        },
        "look": {
            "path": "./Best_Model_Configs/pedestrian_behavior_final_models/LOOK_ADAPTER_MODEL_loha",
            "mappings": "./Best_Model_Configs/pedestrian_behavior_final_models/LOOK_ADAPTER_MODEL_loha/mappings.json",
        },
        "cross": {
            "path": "./Best_Model_Configs/pedestrian_behavior_final_models/CROSS_ADAPTER_MODEL_adalora",
            "mappings": "./Best_Model_Configs/pedestrian_behavior_final_models/CROSS_ADAPTER_MODEL_adalora/mappings.json",
        },
        "occlusion": {
            "path": "./Best_Model_Configs/pedestrian_behavior_final_models/OCCLUSION_ADAPTER_MODEL_adalora",
            "mappings": "./Best_Model_Configs/pedestrian_behavior_final_models/OCCLUSION_ADAPTER_MODEL_adalora/mappings.json",
        },
    }

    if not os.path.exists(GT_CSV_PATH):
        raise FileNotFoundError(f"Ground-truth CSV not found: {GT_CSV_PATH}")

    gt_df = pd.read_csv(GT_CSV_PATH)

    bbox_cols = ["bbox_xtl", "bbox_ytl", "bbox_xbr", "bbox_ybr"]
    missing_cols = [c for c in bbox_cols if c not in gt_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required bounding box columns in GT CSV: {missing_cols}")

    if "bounding_box" not in gt_df.columns:
        gt_df["bounding_box"] = gt_df[bbox_cols].values.tolist()

    global BBOX_SOURCE  
    if BBOX_SOURCE == "gt":
        def extend_gt_bbox_df(row):
            H, W = 1080, 1920
            bbox = [row["bbox_xtl"], row["bbox_ytl"], row["bbox_xbr"], row["bbox_ybr"]]
            return extend_single_bbox(bbox, (H, W), delta=0.05)

        gt_df["extended_bbox"] = gt_df.apply(extend_gt_bbox_df, axis=1)

    if not os.path.isdir(VIDEO_FOLDER):
        raise NotADirectoryError(f"Video folder not found: {VIDEO_FOLDER}")

    Path("Generated_Data").mkdir(exist_ok=True)

    video_files = sorted([f for f in os.listdir(VIDEO_FOLDER) if f.endswith(".mp4")])
    if not video_files:
        raise RuntimeError(f"No *.mp4 videos found in {VIDEO_FOLDER}")

    MAX_VIDEOS = None

    if MAX_VIDEOS is not None:
        video_files = video_files[:MAX_VIDEOS]
        print(f"Testing mode: processing only the first {len(video_files)} videos (set MAX_VIDEOS=None for full run)")

    all_results = []
    for vf in video_files:
        vid_path = os.path.join(VIDEO_FOLDER, vf)
        print(f"\n Processing {vf} …")
        df_video = infer_and_evaluate_on_video(vid_path, adapters, VIT_BASE, YOLO_MODEL_PATH, gt_df)
        all_results.append(df_video)

    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\n All processing complete. Raw data saved to '{OUTPUT_CSV_PATH}'")

if __name__ == "__main__":
    main()
