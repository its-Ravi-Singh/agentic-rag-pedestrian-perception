# Pedestrian Behavior and Scene Contextual Analysis using PEFT

This project provides an end-to-end pipeline for analyzing pedestrian behavior and scene context from the JAAD dataset. It leverages a fine-tuned YOLOv8 model for pedestrian detection and a suite of Vision Transformer (ViT) models enhanced with Parameter-Efficient Fine-Tuning (PEFT) adapters for detailed attribute classification.

## Features

-   **Two-Stage Inference**: Combines YOLO for robust pedestrian detection with specialized ViT adapters for nuanced behavior classification.
-   **Comprehensive Behavior Analysis**: Classifies pedestrian attributes including `Action` (walking/standing), `Look` (looking/not-looking), `Cross` (crossing/not-crossing), and `Occlusion`.
-   **Detailed Scene Context Analysis**: Classifies environmental attributes like `Weather`, `Time of Day`, `Pedestrian Density`, and `Road Presence`.
-   **Automated Pipeline**: Scripts to handle the entire workflow from raw data extraction to final analysis and visualization.
-   **In-Depth Reporting**: Generates detailed markdown reports and publication-ready plots for all analyses.
-   **Ablation Studies**: Includes a script to compare the performance of an adaptive, context-aware analysis strategy against a baseline approach.

## Installation

This guide provides instructions for setting up a single, unified environment for all project tasks.

### 1. Install Deep Learning Libraries (GPU Recommended)

For users with an NVIDIA GPU, install PyTorch and its related libraries first. This project is tested with CUDA 11.8.

```bash
# Install PyTorch, torchvision, and torchaudio for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
For other CUDA versions or for a CPU-only installation, please refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 2. Install Remaining Dependencies

Install all other required packages using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Dataset Setup

This project requires the **JAAD (Joint Attention in Autonomous Driving)** dataset, including its annotations and video clips.

1.  **Clone the JAAD Repository**:
    From the root of this project directory, clone the official JAAD repository which contains all the necessary annotation files.

    ```bash
    git clone https://github.com/ykotseruba/JAAD.git
    ```
    This will create a `JAAD/` directory inside your project.

2.  **Download the Video Clips**:
    The JAAD repository includes a script to download all the video clips. Navigate into the new `JAAD` directory and run it.

    ```bash
    cd JAAD
    bash download_clips.sh
    cd ..
    ```
    This command will create a `JAAD/JAAD_clips/` folder and populate it with all the required video files. After the script finishes, you will be returned to the project's root directory.

3.  **Final Directory Structure**:
    After completing the steps above, your project directory should be organized as follows. The scripts in this project are configured to find the dataset files in this specific structure.

    ```
    pedestrian_behavior_and_scene_contextual_analysis_peft/
    ├── JAAD/
    │   ├── JAAD_clips/
    │   │   ├── video_0001.mp4
    │   │   └── ...
    │   ├── annotations/
    │   │   ├── video_0001.xml
    │   │   └── ...
    │   ├── annotations_attributes/
    │   │   ├── video_0001_attributes.xml
    │   │   └── ...
    │   └── ... (other JAAD repo files like download_clips.sh)
    ├── Best_Model_Configs/
    ├── Generated_Data/
    ├── venv/
    ├── requirements.txt
    ├── README.md
    └── ... (python scripts)
    ```

## Running the Analysis Pipeline

The pipeline is designed to be run in three sequential steps. Ensure you have activated your virtual environment (`source venv/bin/activate`) before running the scripts.

### Step 1: Extract and Prepare Annotations

This initial step parses the raw JAAD XML annotation files, normalizes the labels, and generates a clean CSV file that serves as the ground truth for all subsequent steps.

```bash
python jaad_annotations_extraction_step1.py
```
-   **Input**: `JAAD/annotations/` and `JAAD/annotations_attributes/`
-   **Output**: `Generated_Data/jaad_annotations_extracted_data_step1.csv`
    -   The generated CSV contains the ground-truth data with the following columns: `video_id`, `frame_id`, `track_id`, `label`, `bbox_xtl`, `bbox_ytl`, `bbox_xbr`, `bbox_ybr`, `occlusion`, `action`, `look`, `cross`, `road_presence`, `weather`, `time_of_day`, and `pedestrian_density`.
    -   A text file is also generated in `class_distribution_cleaned_[timestamp].txt`.

### Step 2: Run Inference Models

This step uses the pre-trained models to perform inference on the video clips, generating raw data for the final analysis. The two inference scripts can be run in any order.

-   **A) Pedestrian Behavior Inference**:
    This script runs the two-stage pipeline: YOLO detects pedestrians, and then the behavior adapters classify their actions, gaze, and crossing status.
    ```bash
    python inference_pedestrian_behavior_step2.py
    ```
    -   **Input**: `JAAD/JAAD_clips/`, `Generated_Data/jaad_annotations_extracted_data_step1.csv`, and models from `Best_Model_Configs/`.
    -   **Output**: `Generated_Data/raw_data_pedestrian_behavior_step2.csv`
        - This CSV contains the raw inference output with detailed predictions, ground-truth labels, and performance metrics. The columns include: `video_id`, `frame_id`, `inference_track_id`, `pred_bbox`, `gt_bbox`, `iou_match_score`, `pred_action`, `gt_action`, `pred_look`, `gt_look`, `pred_cross`, `gt_cross`, `pred_occlusion`, `gt_occlusion`, as well as their corresponding `confidence` and `probabilities`, and `gflops` for each attribute.

-   **B) Scene Context Inference**:
    This script runs the scene-level adapters on full video frames to classify the environment (weather, time of day, etc.).
    ```bash
    python inference_scene_context_step2.py
    ```
    -   **Input**: `JAAD/JAAD_clips/` and models from `Best_Model_Configs/`.
    -   **Output**: `Generated_Data/raw_data_scene_contextual_analysis_step2.csv`
        - This CSV contains the frame-by-frame scene analysis. It includes predictions for `weather`, `time_of_day`, `ped_density`, and `road_presence`, their associated `confidence` scores, and performance metrics like `inference_time_ms` and `gflops`. It also contains the baseline `yolo_pedestrian_count` and `yolo_ped_density` for comparison.


### Step 3: Analyze Metrics and Generate Reports

This final step processes the raw data from Step 2 to generate all reports and visualizations, providing a deep dive into model performance and behavioral patterns. These scripts can be run in any order.

-   **A) Analyze Pedestrian Behavior Metrics**:
    Analyzes the performance of the behavior adapters, including classification accuracy, latency, and confidence.
    ```bash
    python analyze_metrics_pedestrian_behavior_step3.py
    ```
    -   **Input**: `Generated_Data/raw_data_pedestrian_behavior_step2.csv`
    -   **Output**: Plots and reports in `Analysis_Plots_Pedestrian_Behavior/`

-   **B) Analyze Scene Context Metrics**:
    Analyzes the performance of the scene adapters and compares them against the YOLO baseline.
    ```bash
    python analyze_metrics_scene_context_step3.py
    ```
    -   **Input**: `Generated_Data/raw_data_scene_contextual_analysis_step2.csv`
    -   **Output**: Plots and reports in `Scene_Context_Analysis_Plots/`

-   **C) Run Ablation Study**:
    Compares the computational cost and performance of the adaptive approach against a non-adaptive baseline.
    ```bash
    python ablation_studies.py
    ```
    -   **Input**: Both CSVs from `Generated_Data/`.
    -   **Output**: Report and plot in `Ablation_Studies_Report/`

## Script Descriptions

-   `jaad_annotations_extraction_step1.py`: Extracts, cleans, and normalizes annotations from the raw JAAD XML files.
-   `inference_pedestrian_behavior_step2.py`: Runs YOLO and ViT adapters to infer pedestrian-specific behaviors and matches them with ground truth.
-   `inference_scene_context_step2.py`: Runs ViT adapters on full frames to classify scene-level attributes (e.g., weather, density).
-   `analyze_metrics_pedestrian_behavior_step3.py`: Generates the final set of reports and visualizations for the pedestrian behavior analysis.
-   `analyze_metrics_scene_context_step3.py`: Generates the final set of reports and visualizations for the scene context analysis.
-   `ablation_studies.py`: Simulates and compares the performance of an adaptive analysis strategy versus a baseline non-adaptive one.
