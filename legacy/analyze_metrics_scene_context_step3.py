"""
Scene Context Metrics Analysis & Visualization (Step 3)

Description:
    This script serves as the final analysis stage for the scene context
    evaluation pipeline. It processes the raw data CSVs from Step 2 to generate
    a wide array of analyses, reports, and visualizations focused on the interplay
    between the environment and pedestrian behavior.

    The script performs several key analyses:
    -   **Adapter vs. YOLO**: Compares the performance, speed, and agreement of the
        PEFT adapter for density classification against a rule-based YOLO approach.
    -   **Environmental Impact**: Investigates how weather conditions affect
        pedestrian safety behaviors (e.g., looking while crossing).
    -   **Road Geometry**: Analyzes how road width influences crossing dynamics and
        model performance.
    -   **Density Performance**: Evaluates system performance and resource utilization
        across different pedestrian density levels.
    -   **Multi-Factor Interactions**: Explores the combined effects of multiple
        contextual variables (weather, time, density) on system performance.
    -   **Adaptive Strategy**: Generates a detailed report and a decision tree
        visualization with recommendations for an adaptive analysis strategy.

Inputs:
    -   Raw Scene Context Data (required): Path specified by `--scene`.
        (Default: `raw_data_scene_contextual_analysis_step2.csv`)
    -   Raw Pedestrian Behavior Data (optional): Path specified by `--behavior`.
        If provided, enables cross-analysis between scene and behavior.
        (Default: `raw_data_pedestrian_behavior_step2.csv`)

Outputs:
    -   A directory named `Scene_Context_Analysis_Plots/` containing all
        generated plots (.png) and markdown reports (.md).

Command-Line Arguments:
    -   `--scene` (str, optional):
        Specifies the path to the raw scene context data CSV.
    -   `--behavior` (str, optional):
        Specifies the path to the raw pedestrian behavior data CSV.
"""
from __future__ import annotations

import os
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
import argparse

# =============================================================================
# A. CONFIGURATION & SETUP
# =============================================================================
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
COLORS = ["#004488", "#DDAA33", "#BB5566", "#FF5733", "#228833", "#CC6677", "#AA4499", "#DDCC77"]
SCENE_ATTRS = ["weather", "time_of_day", "ped_density", "road_presence"]
BEHAVIOR_ATTRS = ["action", "look", "cross", "occlusion"]
PLOTS_DIR = Path("Scene_Context_Analysis_Plots")

# =============================================================================
# B. DATA LOADING & PREPROCESSING
# =============================================================================
def load_scene_data(csv_path: str) -> pd.DataFrame:
    """Loads and preprocesses the raw scene context data from a CSV file."""
    df = pd.read_csv(csv_path)
    prob_cols = [col for col in df.columns if col.endswith('_probabilities')]
    for col in prob_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [])
    
    if 'yolo_pedestrian_count' in df.columns and 'pred_ped_density' in df.columns:
        df['density_agreement'] = (df['pred_ped_density'] == df['yolo_ped_density'])
    
    if 'frame_timestamp' in df.columns:
        df['video_duration'] = df.groupby('video_id')['frame_timestamp'].transform('max')
        df['frame_progress'] = df['frame_timestamp'] / df['video_duration']
    
    return df

def merge_behavior_data(scene_df: pd.DataFrame, behavior_path: Optional[str] = None) -> pd.DataFrame:
    """Merges scene context data with pedestrian behavior data if the latter is available."""
    if behavior_path and os.path.exists(behavior_path):
        behavior_df = pd.read_csv(behavior_path)
        
        behavior_cols_to_merge = ['video_id', 'frame_id'] + \
                                 [col for col in behavior_df.columns if any(attr in col for attr in BEHAVIOR_ATTRS)]
        
        merged = pd.merge(scene_df, behavior_df[behavior_cols_to_merge], on=['video_id', 'frame_id'], how='left')
        print("   - Successfully merged scene context with pedestrian behavior data.")
        return merged
    
    print("   - No behavior data provided; proceeding with scene-only analysis.")
    return scene_df

# =============================================================================
# C. ANALYSIS MODULES & REPORTING
# =============================================================================

# =============================================================================
# C1: Adapter vs YOLO Performance Comparison
# =============================================================================

def adapter_vs_yolo_performance_analysis(df: pd.DataFrame):
    """Generates a report and visualization comparing adapter and YOLO-based density classification."""
    
    # =============================================================================
    # C1A: Adapter vs YOLO Performance Comparison Visualizations
    # =============================================================================

    if 'yolo_ped_density' not in df.columns or 'pred_ped_density' not in df.columns:
        print("   - Skipping Adapter vs. YOLO analysis: Missing required data columns.")
        return
        
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # =========================================================================
    # Plot 1: Classification Agreement Heatmap by Density Level
    # =========================================================================
    
    if 'yolo_pedestrian_count' in df.columns:
        df['density_bin'] = pd.cut(df['yolo_pedestrian_count'], 
                                  bins=[-0.1, 3, 7, float('inf')], 
                                  labels=['Low (≤3)', 'Medium (4-7)', 'High (>7)'])
        
        agreement_data = df.groupby(['density_bin', 'pred_ped_density', 'yolo_ped_density']).size().unstack(fill_value=0)
        
        density_agreement = []
        for density_bin in df['density_bin'].cat.categories:
            subset = df[df['density_bin'] == density_bin]
            if not subset.empty:
                agreement_rate = (subset['pred_ped_density'] == subset['yolo_ped_density']).mean()
                density_agreement.append({'Density': density_bin, 'Agreement': agreement_rate})
        
        if density_agreement:
            agreement_df = pd.DataFrame(density_agreement)
            bars = ax1.bar(agreement_df['Density'], agreement_df['Agreement'], 
                          color=COLORS[:len(agreement_df)], alpha=0.8, edgecolor='black')
            
            for bar, agreement in zip(bars, agreement_df['Agreement']):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{agreement:.3f}', ha='center', va='bottom', fontweight='bold')
            
            ax1.set_ylabel('Classification Agreement')
            ax1.set_title('A) Adapter-YOLO Agreement by Pedestrian Density')
            ax1.set_ylim(0, 1.1)
            ax1.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # Plot 2: Timing Performance Matrix
    # =========================================================================

    if 'pred_ped_density_inference_time_ms' in df.columns and 'yolo_ped_density_inference_time_ms' in df.columns:
        timing_data = []
        
        for density in df['pred_ped_density'].unique():
            subset = df[df['pred_ped_density'] == density]
            if not subset.empty and len(subset) > 5:
                adapter_time = subset['pred_ped_density_inference_time_ms'].mean()
                yolo_time = subset['yolo_ped_density_inference_time_ms'].mean()
                
                timing_data.append({
                    'Density': density, 
                    'Adapter (ms)': adapter_time,
                    'YOLO (ms)': yolo_time,
                    'Speedup': yolo_time / adapter_time if adapter_time > 0 else 1
                })
        
        if timing_data:
            timing_df = pd.DataFrame(timing_data)
            x = np.arange(len(timing_df))
            width = 0.35
            
            ax2.bar(x - width/2, timing_df['Adapter (ms)'], width, label='Adapter', 
                   color=COLORS[0], alpha=0.8)
            ax2.bar(x + width/2, timing_df['YOLO (ms)'], width, label='YOLO', 
                   color=COLORS[1], alpha=0.8)
            
            ax2.set_xlabel('Predicted Density Level')
            ax2.set_ylabel('Inference Time (ms)')
            ax2.set_title('B) Inference Time Comparison by Density')
            ax2.set_xticks(x)
            ax2.set_xticklabels(timing_df['Density'], rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')    
    
    # =========================================================================
    # Plot 3: Accuracy-Speed Trade-off Scatter
    # =========================================================================

    if 'pred_ped_density_confidence' in df.columns:
        method_performance = []
        
        adapter_accuracy = df['density_agreement'].mean() if 'density_agreement' in df.columns else 0
        adapter_speed = df['pred_ped_density_inference_time_ms'].mean() if 'pred_ped_density_inference_time_ms' in df.columns else 0
        adapter_confidence = df['pred_ped_density_confidence'].mean()
        
        method_performance.append({
            'Method': 'Adapter', 'Accuracy': adapter_accuracy, 'Speed (ms)': adapter_speed, 
            'Confidence': adapter_confidence, 'Size': 150
        })
        
        yolo_accuracy = adapter_accuracy
        yolo_speed = df['yolo_ped_density_inference_time_ms'].mean() if 'yolo_ped_density_inference_time_ms' in df.columns else 0
        
        method_performance.append({
            'Method': 'YOLO+Rules', 'Accuracy': yolo_accuracy, 'Speed (ms)': yolo_speed,
            'Confidence': 1.0, 'Size': 150
        })
        
        perf_df = pd.DataFrame(method_performance)
        
        if not perf_df.empty:
            scatter = ax3.scatter(perf_df['Speed (ms)'], perf_df['Accuracy'], 
                                s=perf_df['Size'], c=perf_df['Confidence'], 
                                cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
            
            for i, row in perf_df.iterrows():
                ax3.annotate(row['Method'], (row['Speed (ms)'], row['Accuracy']),
                           xytext=(10, 5), textcoords='offset points', fontsize=10, fontweight='bold')
            
            ax3.set_xlabel('Inference Time (ms)')
            ax3.set_ylabel('Classification Accuracy')
            ax3.set_title('C) Accuracy-Speed Trade-off')
            ax3.grid(True, alpha=0.3)
            
            # Add colorbar for confidence
            plt.colorbar(scatter, ax=ax3, label='Confidence Score')
    
    
    # =========================================================================
    # Plot 4: Density Change Frequency Timeline
    # =========================================================================
    
    if 'video_id' in df.columns and 'frame_id' in df.columns:
        density_changes = []
        
        for video_id in df['video_id'].unique():
            video_data = df[df['video_id'] == video_id].sort_values('frame_id')
            if len(video_data) > 1:
                adapter_changes = (video_data['pred_ped_density'] != video_data['pred_ped_density'].shift()).sum() - 1
                yolo_changes = (video_data['yolo_ped_density'] != video_data['yolo_ped_density'].shift()).sum() - 1
                video_duration = video_data['frame_timestamp'].max() if 'frame_timestamp' in df.columns else len(video_data) / 30.0
                
                density_changes.append({
                    'Video': video_id,
                    'Duration (s)': video_duration,
                    'Adapter Changes': adapter_changes,
                    'YOLO Changes': yolo_changes,
                    'Adapter Change Rate': adapter_changes / video_duration if video_duration > 0 else 0,
                    'YOLO Change Rate': yolo_changes / video_duration if video_duration > 0 else 0
                })
        
        if density_changes:
            changes_df = pd.DataFrame(density_changes)
            
            ax4.scatter(changes_df['Adapter Change Rate'], changes_df['YOLO Change Rate'], 
                       alpha=0.6, s=50, color=COLORS[0])
            
            max_rate = max(changes_df['Adapter Change Rate'].max(), changes_df['YOLO Change Rate'].max())
            ax4.plot([0, max_rate], [0, max_rate], 'k--', alpha=0.5, label='Perfect Agreement')
            
            ax4.set_xlabel('Adapter Density Change Rate (changes/sec)')
            ax4.set_ylabel('YOLO Density Change Rate (changes/sec)')
            ax4.set_title('D) Density Change Frequency Comparison')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            avg_duration = changes_df['Duration (s)'].mean()
            avg_adapter_changes = changes_df['Adapter Changes'].mean()
            avg_yolo_changes = changes_df['YOLO Changes'].mean()
            
            stats_text = f'Avg Duration: {avg_duration:.1f}s\nAdapter: {avg_adapter_changes:.1f} changes\nYOLO: {avg_yolo_changes:.1f} changes'
            ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_adapter_yolo_performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # =============================================================================
    # C1B: Adapter vs YOLO Performance Comparison Report
    # =============================================================================

    with open(PLOTS_DIR / "01_adapter_yolo_comparison.md", "w") as f:
        f.write("# Adapter vs YOLO Performance Comparison\n\n")
        f.write("Comprehensive analysis of pedestrian density classification performance.\n\n")
        f.write("**Note**: YOLO counts are based on 'pedestrian' and 'ped' classes only, ")
        f.write("not the broader 'people' class, ensuring fair comparison with adapter classifications.\n\n")
        
        if 'density_agreement' in df.columns:
            overall_agreement = df['density_agreement'].mean()
            f.write(f"## Overall Performance\n\n")
            f.write(f"- **Classification Agreement**: {overall_agreement:.3f} ({overall_agreement*100:.1f}%)\n")
            
            if timing_data:
                avg_adapter_time = np.mean([d['Adapter (ms)'] for d in timing_data])
                avg_yolo_time = np.mean([d['YOLO (ms)'] for d in timing_data])
                f.write(f"- **Average Adapter Time**: {avg_adapter_time:.2f} ms\n")
                f.write(f"- **Average YOLO Time**: {avg_yolo_time:.2f} ms\n")
                f.write(f"- **Speed Advantage**: {avg_yolo_time/avg_adapter_time:.2f}x {'(YOLO faster)' if avg_yolo_time < avg_adapter_time else '(Adapter faster)'}\n\n")
        
        if density_changes:
            f.write(f"## Density Change Analysis\n\n")
            f.write(f"- **Videos analyzed**: {len(density_changes)}\n")
            f.write(f"- **Average video duration**: {avg_duration:.1f} seconds\n")
            f.write(f"- **Average density changes per video**:\n")
            f.write(f"  - Adapter: {avg_adapter_changes:.1f}\n")
            f.write(f"  - YOLO: {avg_yolo_changes:.1f}\n")
    
    print("   - Adapter vs YOLO performance analysis saved")

# =============================================================================
# C2: Environmental Impact on Pedestrian Behavior
# =============================================================================

def analyze_density_progression_by_weather(df: pd.DataFrame, use_predictions: bool = True) -> Dict[str, Dict]:
    """Analyze pedestrian density sustainability rates over time by weather condition.
    
    Tracks high density events and measures how long they are sustained at different
    time intervals (30-100 frames, or 1-3.3 seconds).
    
    Args:
        df: DataFrame with scene contextual data
        use_predictions: If True, uses adapter predictions. If False, uses YOLO data.
    
    Returns:
        Dictionary with weather conditions as keys, containing timeline sustainability data
    """

    if use_predictions:
        density_col = 'pred_ped_density'
        if density_col not in df.columns:
            return {}
    else:
        density_col = 'yolo_ped_density' 
        if density_col not in df.columns:
            return {}
    
    if 'pred_weather' not in df.columns or 'video_id' not in df.columns:
        return {}
    
    density_progression = {}
    
    for weather in df['pred_weather'].unique():
        if pd.isna(weather) or weather == 'unknown':
            continue
            
        weather_data = df[df['pred_weather'] == weather].copy()
        if weather_data.empty:
            continue
        
        high_density_starts = []
        
        for video_id in weather_data['video_id'].unique():
            video_data = weather_data[weather_data['video_id'] == video_id].sort_values('frame_id')
            
            if len(video_data) < 30:
                continue
            
            for i in range(len(video_data) - 30):
                current_frame = video_data.iloc[i]
                
                current_density = current_frame[density_col]
                is_high_density = 'high' in str(current_density).lower()
                
                prev_is_high = False
                if i > 0:
                    prev_density = video_data.iloc[i-1][density_col]
                    prev_is_high = 'high' in str(prev_density).lower()
                
                if is_high_density and not prev_is_high:
                    high_density_starts.append({
                        'video_id': video_id,
                        'start_frame': current_frame['frame_id'],
                        'start_index': i,
                        'frames_available': len(video_data) - i
                    })
        
        if not high_density_starts:
            continue
            
        timeline_sustainability = {}
        
        for time_point in range(10, 101, 10):
            sustained_count = 0
            total_trackable = 0
            
            for start_event in high_density_starts:
                if start_event['frames_available'] >= time_point:
                    total_trackable += 1
                    
                    video_data = weather_data[weather_data['video_id'] == start_event['video_id']].sort_values('frame_id')
                    check_index = start_event['start_index'] + time_point
                    
                    if check_index < len(video_data):
                        recent_frames = video_data.iloc[start_event['start_index']:check_index+1]
                        
                        high_density_frames = 0
                        for _, frame in recent_frames.iterrows():
                            if 'high' in str(frame[density_col]).lower():
                                high_density_frames += 1
                        
                        if high_density_frames / len(recent_frames) > 0.4:
                            sustained_count += 1
            
            if total_trackable > 0:
                sustainability_rate = (sustained_count / total_trackable) * 100
                timeline_sustainability[time_point] = sustainability_rate
        
        density_progression[weather.title()] = {
            'timeline': timeline_sustainability,
            'total_events': len(high_density_starts),
            'event_data': high_density_starts
        }
    
    return density_progression

def check_crossing_completion(video_data: pd.DataFrame, start_event: Dict, check_index: int, cross_col: str) -> bool:
    """
    ROBUST multi-criteria crossing completion detection.
    
    A pedestrian is considered to have "completed crossing" if ANY of these conditions are met:
    1. EXPLICIT COMPLETION: crossing → not_crossing (if available in dataset)
    2. SUSTAINED CROSSING: Maintained crossing behavior for reasonable duration (>3 seconds)
    3. TRACK DISAPPEARANCE: Track ID disappears after sustained crossing (left camera view)
    4. END-OF-VIDEO CROSSING: Still crossing at end of video (assume completion)
    5. MOVEMENT PATTERN: Significant horizontal displacement with crossing behavior
    
    Note: Criterion 1 may not be available if dataset only contains 'crossing' labels
    without explicit 'not_crossing' or 'completed' states.
    
    Args:
        video_data: Sorted video data for this video
        start_event: Crossing start event info
        check_index: Current frame index to check
        cross_col: Column name ('pred_cross' or 'gt_cross')
    
    Returns:
        bool: True if crossing is considered completed
    """
    start_idx = start_event['start_index']
    track_id = start_event.get('track_id', -1)
    
    journey_frames = video_data.iloc[start_idx:check_index+1]
    
    if len(journey_frames) == 0:
        return False
    
    # CRITERION 1: EXPLICIT COMPLETION (crossing → not_crossing)
    crossing_states = journey_frames[cross_col].values
    has_explicit_completion = False
    for i in range(1, len(crossing_states)):
        if crossing_states[i-1] == 'crossing' and crossing_states[i] == 'not_crossing':
            has_explicit_completion = True
            break
    
    unique_states = set(crossing_states)
    if 'not_crossing' in unique_states and 'crossing' in unique_states and has_explicit_completion:
        return True  
    
    # CRITERION 2: SUSTAINED CROSSING (>3 seconds = >90 frames at 30fps)
    crossing_frames = (journey_frames[cross_col] == 'crossing').sum()
    total_journey_frames = len(journey_frames)
    
    if (crossing_frames >= 90 and
        crossing_frames / total_journey_frames >= 0.6):
        return True  
    
    # CRITERION 3: TRACK DISAPPEARANCE after substantial crossing
    if track_id != -1 and crossing_frames >= 30:  # At least 1 second of crossing
        remaining_frames = video_data.iloc[check_index+1:]
        if len(remaining_frames) > 10:  
            track_id_col = 'track_id' if 'track_id' in video_data.columns else 'inference_track_id'
            if track_id_col in video_data.columns:
                remaining_track_ids = remaining_frames[track_id_col].values
                if track_id not in remaining_track_ids:
                    return True  
    
    # CRITERION 4: END-OF-VIDEO CROSSING
    if check_index >= len(video_data) * 0.9:
        recent_crossing = (journey_frames[cross_col] == 'crossing').sum()
        if recent_crossing / len(journey_frames) >= 0.4:
            return True  
    
    # CRITERION 5: MOVEMENT PATTERN (if bounding box data available)
    if 'pred_bbox' in journey_frames.columns or 'gt_bbox' in journey_frames.columns:
        bbox_col = 'pred_bbox' if 'pred_bbox' in journey_frames.columns else 'gt_bbox'
        try:
            start_bbox = journey_frames.iloc[0][bbox_col]
            end_bbox = journey_frames.iloc[-1][bbox_col] 
            
            if (isinstance(start_bbox, (list, tuple)) and len(start_bbox) == 4 and
                isinstance(end_bbox, (list, tuple)) and len(end_bbox) == 4):
                
                start_center_x = (start_bbox[0] + start_bbox[2]) / 2
                end_center_x = (end_bbox[0] + end_bbox[2]) / 2
                
                horizontal_movement = abs(end_center_x - start_center_x)
                if horizontal_movement > 600 and crossing_frames >= 20:
                    return True  
                    
        except Exception:
            pass  
    
    return False  

def analyze_pedestrian_safety_behavior_by_weather(df: pd.DataFrame, use_predictions: bool = True) -> Dict[str, Dict]:
    """Analyze pedestrian safety behavior patterns by weather condition.
    
    Focuses on critical safety comparison:
    - SAFE BEHAVIOR: look + cross (pedestrian checks for vehicles before crossing)
    - RISKY BEHAVIOR: not_look + cross (pedestrian crosses without checking)
    
    This is especially important during reduced visibility conditions (rainy, snowy).
    
    Args:
        df: DataFrame with behavior + scene contextual data
        use_predictions: If True, uses pred_look/pred_cross. If False, uses gt_look/gt_cross.
    
    Returns:
        Dictionary with weather conditions and safety behavior timeline data
    """
    
    if use_predictions:
        cross_col = 'pred_cross'
        look_col = 'pred_look'
        data_type = "Predictions"
        
        missing_cols = []
        if cross_col not in df.columns:
            missing_cols.append(cross_col)
        if look_col not in df.columns:
            missing_cols.append(look_col)
            
        if missing_cols:
            print(f"   - Missing columns for prediction analysis: {missing_cols}")
            return {}
    else:
        cross_col = 'gt_cross' 
        look_col = 'gt_look'
        data_type = "Ground Truth"
        
        missing_cols = []
        if cross_col not in df.columns:
            missing_cols.append(cross_col)
        if look_col not in df.columns:
            missing_cols.append(look_col)
            
        if missing_cols:
            print(f"   - Missing columns for ground truth analysis: {missing_cols}")
            return {}
            
        cross_unique = df[cross_col].value_counts()
        look_unique = df[look_col].value_counts()
    
    if 'pred_weather' not in df.columns or 'video_id' not in df.columns:
        print("   - Missing required columns for safety behavior analysis")
        return {}
    
    print(f"   - METHODOLOGY: INDIVIDUAL Pedestrian Crossing Safety Analysis")
    print(f"   - 1. Find ALL pedestrians in each weather condition")
    print(f"   - 2. For crossing pedestrians, categorize safety behavior:")
    print(f"   -     - SAFE: look + cross (checks for vehicles)")
    print(f"   -     - RISKY: not_look + cross (crosses without checking)")
    print(f"   - 3. Track INDIVIDUAL pedestrian crossing timelines (not video timeline!)")
    print(f"   - 4. Focus on FIRST 50% of crossing = CRITICAL SAFETY WINDOW")
    print(f"   - 5. Answer: 'During first 50% of crossing, how many pedestrians were safe vs risky?'")
    print(f"   - 6. Key insight: After 50%, pedestrians focus on completing crossing, not looking")
    
    safety_behavior_analysis = {}
    
    weather_conditions = [w for w in df['pred_weather'].unique() if not pd.isna(w) and w != 'unknown']
    
    for i, weather in enumerate(weather_conditions):
        
        weather_data = df[df['pred_weather'] == weather].copy()
        if weather_data.empty:
            print(f"   - No data for weather: {weather}")
            continue
        
        pedestrian_behaviors = []
        total_pedestrians = 0
        crossing_pedestrians = 0
        safe_behaviors = 0
        risky_behaviors = 0
        
        video_ids = weather_data['video_id'].unique()
        for j, video_id in enumerate(video_ids):
            video_data = weather_data[weather_data['video_id'] == video_id].sort_values('frame_id')
            
            if len(video_data) < 10:
                continue
            
            for idx, row in video_data.iterrows():
                total_pedestrians += 1
                
                is_crossing = (row[cross_col] == 'crossing')
                is_looking = (row[look_col] == 'looking')
                is_not_looking = (row[look_col] == 'not_looking')
                
                if is_crossing:
                    crossing_pedestrians += 1
                    
                    # Categorize safety behavior
                    if is_looking:
                        # SAFE: looking while crossing
                        behavior_type = 'SAFE_look_cross'
                        safe_behaviors += 1
                    elif is_not_looking:
                        # RISKY: not looking while crossing  
                        behavior_type = 'RISKY_not_look_cross'
                        risky_behaviors += 1
                    else:
                        behavior_type = 'UNKNOWN_look_cross'
                    
                    pedestrian_behaviors.append({
                        'video_id': video_id,
                        'frame_id': row['frame_id'],
                        'track_id': row.get('track_id', row.get('inference_track_id', -1)),
                        'behavior_type': behavior_type,
                        'is_crossing': is_crossing,
                        'is_looking': is_looking,
                        'weather': weather
                    })
        
        if not pedestrian_behaviors:
            print(f"   - No crossing pedestrian behaviors found for weather: {weather}")
            continue
            
       
        if crossing_pedestrians > 0:
            safe_percentage = (safe_behaviors / crossing_pedestrians) * 100
            risky_percentage = (risky_behaviors / crossing_pedestrians) * 100
            print(f"   - Safety ratio -> SAFE: {safe_percentage:.1f}% | RISKY: {risky_percentage:.1f}%")
            
        timeline_safety = {}
        
        pedestrian_crossing_journeys = {}
        for behavior in pedestrian_behaviors:
            track_id = behavior.get('track_id', -1)
            video_id = behavior['video_id']
            journey_key = f"{video_id}_{track_id}"
            
            if journey_key not in pedestrian_crossing_journeys:
                pedestrian_crossing_journeys[journey_key] = []
            pedestrian_crossing_journeys[journey_key].append(behavior)
        
        for journey_key in pedestrian_crossing_journeys:
            pedestrian_crossing_journeys[journey_key].sort(key=lambda x: x['frame_id'])
        
        crossing_timeline_percentages = [25, 50, 75, 100]
        
        for tp_idx, crossing_pct in enumerate(crossing_timeline_percentages):
                
            safe_count = 0
            risky_count = 0
            total_trackable_pedestrians = 0
            
            for journey_key, journey_frames in pedestrian_crossing_journeys.items():
                if len(journey_frames) < 3:
                    continue
                    
                crossing_duration = len(journey_frames)
                checkpoint_frame_idx = int(crossing_duration * crossing_pct / 100) - 1
                checkpoint_frame_idx = max(0, min(checkpoint_frame_idx, crossing_duration - 1))
                
                analyzed_frames = journey_frames[:checkpoint_frame_idx + 1]
                
                if not analyzed_frames:
                    continue
                    
                total_trackable_pedestrians += 1
                
                pedestrian_safe_count = sum(1 for f in analyzed_frames if f['behavior_type'] == 'SAFE_look_cross')
                pedestrian_risky_count = sum(1 for f in analyzed_frames if f['behavior_type'] == 'RISKY_not_look_cross')
                
                if pedestrian_safe_count > pedestrian_risky_count:
                    safe_count += 1
                elif pedestrian_risky_count > pedestrian_safe_count:
                    risky_count += 1
            
            if total_trackable_pedestrians > 0:
                safe_percentage = (safe_count / total_trackable_pedestrians) * 100
                risky_percentage = (risky_count / total_trackable_pedestrians) * 100
                timeline_safety[crossing_pct] = {
                    'safe_percentage': safe_percentage,
                    'risky_percentage': risky_percentage,
                    'total_instances': total_trackable_pedestrians,
                    'safe_count': safe_count,
                    'risky_count': risky_count
                }
                
                critical_marker = " CRITICAL SAFETY WINDOW" if crossing_pct == 50 else ""
            else:
                timeline_safety[crossing_pct] = {
                    'safe_percentage': 0,
                    'risky_percentage': 0,
                    'total_instances': 0,
                    'safe_count': 0,
                    'risky_count': 0
                }
        
        safety_behavior_analysis[weather.title()] = {
            'timeline': timeline_safety,
            'total_pedestrians': total_pedestrians,
            'crossing_pedestrians': crossing_pedestrians,
            'safe_behaviors': safe_behaviors,
            'risky_behaviors': risky_behaviors,
            'behavior_data': pedestrian_behaviors
        }
        
        print(f"   - Completed INDIVIDUAL pedestrian safety analysis for {weather}: {len(timeline_safety)} crossing timeline checkpoints analyzed")
        
        if 50 in timeline_safety:
            critical_data = timeline_safety[50]
            safe_50 = critical_data['safe_percentage']
            risky_50 = critical_data['risky_percentage']
            total_50 = critical_data['total_instances']
            print(f"   - CRITICAL 50% WINDOW for {weather}: {safe_50:.1f}% SAFE, {risky_50:.1f}% RISKY (n={total_50} pedestrians)")
    
    print(f"   - Safety behavior analysis complete for {data_type}")
    return safety_behavior_analysis

def environmental_impact_analysis(df: pd.DataFrame):
    """Analyze how weather and time of day impact pedestrian behavior."""

    behavior_data = None
    behavior_file = "raw_data_pedestrian_behavior_analysis_step2.csv"
    
    if os.path.exists(behavior_file):
        try:
            behavior_data = pd.read_csv(behavior_file)
            
            if 'video_id' in df.columns and 'frame_id' in df.columns:
                merged_df = pd.merge(
                    behavior_data,
                    df[['video_id', 'frame_id', 'pred_weather', 'pred_time_of_day']], 
                    on=['video_id', 'frame_id'], 
                    how='left'
                )
                
                crossing_cols = [col for col in merged_df.columns if 'cross' in col.lower()]
                weather_cols = [col for col in merged_df.columns if 'weather' in col.lower()]
            else:
                merged_df = behavior_data
                print("   - Could not merge - using behavior data only")
                
        except Exception as e:
            print(f"   - Could not load behavior data: {e}")
            merged_df = df
    else:
        print(f" - Behavior data file not found: {behavior_file}")
        merged_df = df
    
    # =========================================================================
    # Weather Impact Dashboard - Single Plot
    # =========================================================================
    
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    
    if behavior_data is not None and 'pred_cross' in merged_df.columns and 'pred_look' in merged_df.columns and 'pred_weather' in merged_df.columns:
        safety_behavior_pred = analyze_pedestrian_safety_behavior_by_weather(merged_df, use_predictions=True)
        
        if safety_behavior_pred:
            weather_conditions = list(safety_behavior_pred.keys())
            
            crossing_checkpoints = [25, 50, 75, 100]
            
            for i, weather in enumerate(weather_conditions):
                if weather in safety_behavior_pred and safety_behavior_pred[weather]['timeline']:
                    timeline_data = safety_behavior_pred[weather]['timeline']
                    
                    # Plot SAFE behaviors (look + cross)
                    safe_rates = [timeline_data.get(t, {}).get('safe_percentage', 0) for t in crossing_checkpoints]
                    ax1.plot(crossing_checkpoints, safe_rates, 
                            marker='o', linewidth=3.0, markersize=8, linestyle='-',
                            color=COLORS[i % len(COLORS)], 
                            label=f"{weather} SAFE (n={safety_behavior_pred[weather]['safe_behaviors']})")
                    
                    # Plot RISKY behaviors (not_look + cross)
                    risky_rates = [timeline_data.get(t, {}).get('risky_percentage', 0) for t in crossing_checkpoints]
                    ax1.plot(crossing_checkpoints, risky_rates, 
                            marker='s', linewidth=3.0, markersize=8, linestyle='--',
                            color=COLORS[i % len(COLORS)], alpha=0.7,
                            label=f"{weather} RISKY (n={safety_behavior_pred[weather]['risky_behaviors']})")
            
            # Highlight critical 50% window
            ax1.axvline(x=50, color='red', linestyle=':', alpha=0.7, linewidth=2, label='CRITICAL SAFETY WINDOW (50%)')
            ax1.fill_betweenx([0, 100], 0, 50, alpha=0.1, color='red', label='Critical Period')
            
            ax1.set_xlabel('Individual Crossing Timeline Progress (%)', fontsize=14)
            ax1.set_ylabel('Pedestrians Showing Behavior (%)', fontsize=14)
            ax1.set_title('Pedestrian Safety During Individual Crossing Timeline - Predictions\n(Focus: First 50% = Critical Safety Window)', fontsize=18, fontweight='bold')
            ax1.legend(loc='center right', fontsize=11, framealpha=0.9)
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(20, 105) 
            ax1.set_ylim(0, 100)
            ax1.set_xticks(crossing_checkpoints)
            ax1.set_xticklabels(['First 25%', 'First 50%\n⭐CRITICAL⭐', 'First 75%', 'Complete\nCrossing'], fontsize=12)
            ax1.tick_params(axis='both', which='major', labelsize=12)
            
            
            if len(weather_conditions) >= 2:
                
                critical_window_risky_rates = {}
                critical_window_safe_rates = {}
                
                for weather in weather_conditions:
                    if weather in safety_behavior_pred and safety_behavior_pred[weather]['timeline']:
                        timeline_data = safety_behavior_pred[weather]['timeline']
                        critical_50_data = timeline_data.get(50, {})
                        if critical_50_data.get('total_instances', 0) > 0:
                            risky_pct_50 = critical_50_data.get('risky_percentage', 0)
                            safe_pct_50 = critical_50_data.get('safe_percentage', 0)
                            critical_window_risky_rates[weather] = risky_pct_50
                            critical_window_safe_rates[weather] = safe_pct_50
                
                if critical_window_risky_rates:
                    most_risky_weather_50 = max(critical_window_risky_rates, key=critical_window_risky_rates.get)
                    safest_weather_50 = min(critical_window_risky_rates, key=critical_window_risky_rates.get)
                    
                    ax1.annotate(f'Most Risky in Critical 50%:\n{most_risky_weather_50}\n({critical_window_risky_rates[most_risky_weather_50]:.1f}% risky)', 
                               xy=(50, critical_window_risky_rates[most_risky_weather_50]), xytext=(55, critical_window_risky_rates[most_risky_weather_50] + 10),
                               arrowprops=dict(arrowstyle='->', color='darkred', alpha=0.8, lw=2),
                               fontsize=9, ha='left', color='darkred', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='red'))
                    
                    ax1.annotate(f'Safest in Critical 50%:\n{safest_weather_50}\n({critical_window_safe_rates[safest_weather_50]:.1f}% safe)', 
                               xy=(50, critical_window_safe_rates[safest_weather_50]), xytext=(55, critical_window_safe_rates[safest_weather_50] - 15),
                               arrowprops=dict(arrowstyle='->', color='darkgreen', alpha=0.8, lw=2),
                               fontsize=9, ha='left', color='darkgreen', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='green'))
        else:
            ax1.text(0.5, 0.5, 'Insufficient data for\nsafety behavior analysis\n(predictions)', 
                    ha='center', va='center', transform=ax1.transAxes,
                    fontsize=14, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    else:
        ax1.text(0.5, 0.5, 'Behavior data not available\nfor safety analysis', 
                ha='center', va='center', transform=ax1.transAxes,
                fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
   
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "02_weather_impact_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   - Weather impact analysis (single plot) saved")

# =============================================================================
# C3: Road Geometry & Crossing Dynamics  
# =============================================================================

def road_geometry_analysis(df: pd.DataFrame):
    """Analyze the impact of road width on pedestrian crossing behavior."""
    
    if 'pred_road_presence' not in df.columns:
        print("[skip] No road presence data available")
        return
        
    # =========================================================================
    # PLOT 1: Crossing Duration by Road Type
    # =========================================================================
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    if 'video_id' in df.columns and 'frame_timestamp' in df.columns:
        crossing_durations = []
        
        for video_id in df['video_id'].unique():
            video_data = df[df['video_id'] == video_id].sort_values('frame_timestamp')
            if len(video_data) > 1:
                road_presence = video_data['pred_road_presence'].mode().iloc[0] if not video_data['pred_road_presence'].mode().empty else 'unknown'
                duration = video_data['frame_timestamp'].max() - video_data['frame_timestamp'].min()
                
                crossing_activity = 0
                if 'pred_cross' in df.columns:
                    crossing_activity = (video_data['pred_cross'] == 'crossing').sum()
                elif 'yolo_pedestrian_count' in df.columns:
                    crossing_activity = video_data['yolo_pedestrian_count'].sum()
                
                if road_presence != 'unknown' and duration > 0:
                    crossing_durations.append({
                        'Road Type': road_presence.replace('_', ' ').title(),
                        'Duration': duration,
                        'Activity Level': crossing_activity,
                        'Video ID': video_id
                    })
        
        if crossing_durations:
            duration_df = pd.DataFrame(crossing_durations)
            
            road_presences = duration_df['Road Type'].unique()
            duration_data = [duration_df[duration_df['Road Type'] == rt]['Duration'].values for rt in road_presences]
            
            bp = ax1.boxplot(duration_data, labels=road_presences, patch_artist=True)
            
            for patch, color in zip(bp['boxes'], COLORS[:len(road_presences)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax1.set_ylabel('Video Duration (seconds)')
            ax1.set_title('A) Scene Duration by Road Type')
            ax1.grid(True, alpha=0.3, axis='y')
            
            for i, rt in enumerate(road_presences):
                rt_data = duration_df[duration_df['Road Type'] == rt]['Duration']
                mean_duration = rt_data.mean()
                ax1.text(i+1, mean_duration, f'μ={mean_duration:.1f}s', 
                        ha='center', va='bottom', fontweight='bold', 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # =========================================================================
    # PLOT 2: Crossing Success Rates by Road Type
    # =========================================================================
    
    if 'pred_road_presence' in df.columns:
        road_success = []
        
        for road_presence in df['pred_road_presence'].unique():
            if pd.isna(road_presence) or road_presence == 'unknown':
                continue
                
            road_subset = df[df['pred_road_presence'] == road_presence]
            
            metrics = {
                'Road Type': road_presence.replace('_', ' ').title(),
                'Sample Count': len(road_subset)
            }
            
            if 'pred_cross' in df.columns:
                metrics['Crossing Rate'] = (road_subset['pred_cross'] == 'crossing').mean()
                metrics['Standing Rate'] = (road_subset['pred_cross'] == 'not_crossing').mean()
            
            if 'density_agreement' in df.columns:
                metrics['Detection Accuracy'] = road_subset['density_agreement'].mean()
                
            if 'pred_ped_density_confidence' in df.columns:
                metrics['Avg Confidence'] = road_subset['pred_ped_density_confidence'].mean()
                
            road_success.append(metrics)
        
        if road_success:
            success_df = pd.DataFrame(road_success)
            
            success_metrics = [col for col in success_df.columns if 'Rate' in col or 'Accuracy' in col or 'Confidence' in col]
            
            if success_metrics:
                x = np.arange(len(success_df))
                bar_width = 0.2
                
                for i, metric in enumerate(success_metrics):
                    if metric in success_df.columns:
                        ax2.bar(x + i * bar_width, success_df[metric], bar_width,
                               label=metric, color=COLORS[i % len(COLORS)], alpha=0.8)
                
                ax2.set_xlabel('Road Type')
                ax2.set_ylabel('Rate/Score')
                ax2.set_title('B) Performance Metrics by Road Type')
                ax2.set_xticks(x + bar_width * (len(success_metrics) - 1) / 2)
                ax2.set_xticklabels(success_df['Road Type'])
                ax2.legend()
                ax2.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # PLOT 3: Processing Time by Road Complexity
    # =========================================================================
    
    if 'pred_road_presence' in df.columns and 'frame_processing_time_ms' in df.columns:
        processing_by_road = df.groupby('pred_road_presence').agg({
            'frame_processing_time_ms': ['mean', 'std', 'count'],
            'pred_ped_density_inference_time_ms': 'mean' if 'pred_ped_density_inference_time_ms' in df.columns else lambda x: 0,
            'yolo_ped_density_inference_time_ms': 'mean' if 'yolo_ped_density_inference_time_ms' in df.columns else lambda x: 0
        }).round(2)
        
        if not processing_by_road.empty:
            road_presences = [rt.replace('_', ' ').title() for rt in processing_by_road.index if not pd.isna(rt)]
            total_times = processing_by_road[('frame_processing_time_ms', 'mean')].values
            
            if 'pred_ped_density_inference_time_ms' in df.columns:
                adapter_times = processing_by_road[('pred_ped_density_inference_time_ms', 'mean')].values
            else:
                adapter_times = np.zeros_like(total_times)
                
            if 'yolo_ped_density_inference_time_ms' in df.columns:
                yolo_times = processing_by_road[('yolo_ped_density_inference_time_ms', 'mean')].values  
            else:
                yolo_times = np.zeros_like(total_times)
            
            x = np.arange(len(road_presences))
            width = 0.25
            
            ax3.bar(x - width, total_times, width, label='Total Processing', color=COLORS[0], alpha=0.8)
            ax3.bar(x, adapter_times, width, label='Adapter Only', color=COLORS[1], alpha=0.8)
            ax3.bar(x + width, yolo_times, width, label='YOLO Only', color=COLORS[2], alpha=0.8)
            
            ax3.set_xlabel('Road Type')
            ax3.set_ylabel('Processing Time (ms)')
            ax3.set_title('C) Processing Time by Road Complexity')
            ax3.set_xticks(x)
            ax3.set_xticklabels(road_presences)
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # PLOT 4: Pedestrian Density Patterns by Road Type
    # =========================================================================
    
    if 'pred_road_presence' in df.columns and 'yolo_pedestrian_count' in df.columns:
        density_patterns = df.groupby(['pred_road_presence', 'pred_ped_density']).size().unstack(fill_value=0)
        density_patterns_pct = density_patterns.div(density_patterns.sum(axis=1), axis=0)
        
        if not density_patterns_pct.empty:
            density_patterns_pct.plot(kind='bar', ax=ax4, stacked=True, 
                                    color=COLORS[:len(density_patterns_pct.columns)], alpha=0.8)
            
            ax4.set_xlabel('Road Type')
            ax4.set_ylabel('Proportion')
            ax4.set_title('D) Pedestrian Density Patterns by Road Type')
            ax4.legend(title='Density Level', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax4.grid(True, alpha=0.3, axis='y')
            plt.setp(ax4.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "04_road_geometry_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   - Road geometry analysis saved")

# =============================================================================
# C4: Density-Based System Performance
# =============================================================================

def density_performance_analysis(df: pd.DataFrame):
    """Analyze system performance across different pedestrian density levels."""
    
    if 'pred_ped_density' not in df.columns:
        print("   - Skipping density performance analysis: No pedestrian density data available")
        return
        
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # =========================================================================  
    # PLOT 1: FPS Performance by Density
    # =========================================================================
    
    if 'frame_processing_time_ms' in df.columns:
        fps_by_density = df.groupby('pred_ped_density').agg({
            'frame_processing_time_ms': ['mean', 'std', 'count', 'min', 'max'],
            'frame_fps_processed': 'mean' if 'frame_fps_processed' in df.columns else lambda x: 0
        }).round(2)
        
        density_levels = fps_by_density.index.tolist()
        mean_times = fps_by_density[('frame_processing_time_ms', 'mean')].values
        std_times = fps_by_density[('frame_processing_time_ms', 'std')].values
        
        bars = ax1.bar(range(len(density_levels)), mean_times, yerr=std_times, capsize=5,
                      color=COLORS[:len(density_levels)], alpha=0.8, edgecolor='black')
        
        ax1.set_xlabel('Pedestrian Density Level')
        ax1.set_ylabel('Processing Time (ms)')
        ax1.set_title('A) Processing Time by Pedestrian Density')
        ax1.set_xticks(range(len(density_levels)))
        ax1.set_xticklabels([dl.replace('_', '\n').replace('pedestrian', 'ped').title() for dl in density_levels])
        ax1.grid(True, alpha=0.3, axis='y')
        
        if 'frame_fps_processed' in df.columns:
            fps_values = fps_by_density[('frame_fps_processed', 'mean')].values
            for bar, fps, count in zip(bars, fps_values, fps_by_density[('frame_processing_time_ms', 'count')]):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height() * 0.05,
                        f'{fps:.1f} FPS\n(n={int(count)})', ha='center', va='bottom', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # =========================================================================
    # PLOT 2: Resource Utilization Patterns
    # =========================================================================
    
    if 'pred_ped_density_inference_time_ms' in df.columns and 'yolo_ped_density_inference_time_ms' in df.columns:
        resource_data = []
        
        for density in df['pred_ped_density'].unique():
            if pd.isna(density):
                continue
                
            density_subset = df[df['pred_ped_density'] == density]
            
            if not density_subset.empty:
                adapter_time = density_subset['pred_ped_density_inference_time_ms'].mean()
                yolo_time = density_subset['yolo_ped_density_inference_time_ms'].mean()
                total_time = density_subset['frame_processing_time_ms'].mean() if 'frame_processing_time_ms' in df.columns else adapter_time + yolo_time
                overhead_time = total_time - adapter_time - yolo_time
                
                resource_data.append({
                    'Density': density.replace('_', '\n').replace('pedestrian', 'ped').title(),
                    'Adapter': adapter_time,
                    'YOLO': yolo_time, 
                    'Overhead': max(0, overhead_time),
                    'Total': total_time
                })
        
        if resource_data:
            resource_df = pd.DataFrame(resource_data)
            
            bottom_adapter = resource_df['Adapter'].values
            bottom_yolo = bottom_adapter + resource_df['YOLO'].values
            
            ax2.bar(resource_df['Density'], resource_df['Adapter'], 
                   label='Adapter Time', color=COLORS[0], alpha=0.8)
            ax2.bar(resource_df['Density'], resource_df['YOLO'], bottom=bottom_adapter,
                   label='YOLO Time', color=COLORS[1], alpha=0.8)
            ax2.bar(resource_df['Density'], resource_df['Overhead'], bottom=bottom_yolo,
                   label='Overhead', color=COLORS[2], alpha=0.8)
            
            ax2.set_xlabel('Pedestrian Density Level')
            ax2.set_ylabel('Processing Time (ms)')
            ax2.set_title('B) Resource Utilization Breakdown')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # PLOT 3: Accuracy vs Density Relationship
    # =========================================================================
    
    if 'density_agreement' in df.columns and 'pred_ped_density_confidence' in df.columns:
        accuracy_confidence = df.groupby('pred_ped_density').agg({
            'density_agreement': ['mean', 'std', 'count'],
            'pred_ped_density_confidence': 'mean'
        }).round(3)
        
        density_levels = accuracy_confidence.index.tolist()
        accuracies = accuracy_confidence[('density_agreement', 'mean')].values
        confidences = accuracy_confidence[('pred_ped_density_confidence', 'mean')].values
        
        sizes = accuracy_confidence[('density_agreement', 'count')].values
        sizes_normalized = (sizes / sizes.max()) * 300
        
        scatter = ax3.scatter(accuracies, confidences, s=sizes_normalized, 
                            c=range(len(density_levels)), cmap='viridis', 
                            alpha=0.7, edgecolors='black', linewidth=2)
        
        for i, (acc, conf, density) in enumerate(zip(accuracies, confidences, density_levels)):
            ax3.annotate(density.replace('_pedestrian_density', '').title(), 
                        (acc, conf), xytext=(10, 5), textcoords='offset points', 
                        fontsize=9, fontweight='bold')
        
        ax3.set_xlabel('Classification Accuracy')
        ax3.set_ylabel('Average Confidence Score')
        ax3.set_title('C) Accuracy vs Confidence by Density')
        ax3.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=ax3, label='Density Level Index')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "05_density_performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   - Density-based performance analysis saved")

# =============================================================================
# C5: Computational Efficiency Matrix
# =============================================================================

def computational_efficiency_analysis(df: pd.DataFrame):
    """Analyze computational efficiency across all 8 adapters."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # =========================================================================
    # PLOT 1: Processing Time Heatmap (All 8 Adapters)
    # =========================================================================

    adapter_timing_cols = [col for col in df.columns if col.endswith('_inference_time_ms')]
    
    if adapter_timing_cols:
        timing_matrix = []
        adapter_names = []
        
        for col in adapter_timing_cols:
            if col in df.columns:
                adapter_name = col.replace('pred_', '').replace('_inference_time_ms', '').replace('_', ' ').title()
                adapter_names.append(adapter_name)
                
                timing_stats = df[col].agg(['mean', 'std', 'min', 'max', 'median']).values
                timing_matrix.append(timing_stats)
        
        if timing_matrix:
            timing_df = pd.DataFrame(timing_matrix, 
                                   columns=['Mean', 'Std', 'Min', 'Max', 'Median'],
                                   index=adapter_names)
            
            sns.heatmap(timing_df, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax1,
                       cbar_kws={'label': 'Time (ms)'})
            ax1.set_title('A) Processing Time Matrix (All Adapters)')
            ax1.set_xlabel('Statistics')
            ax1.set_ylabel('Adapter Type')
    
    # =========================================================================
    # PLOT 2: Memory Usage Estimation by Adapter Type
    # =========================================================================
    
    if adapter_timing_cols:
        memory_estimates = {
            'Weather': {'base': 50, 'variable': 10},
            'Time Of Day': {'base': 45, 'variable': 8}, 
            'Ped Density': {'base': 60, 'variable': 15},
            'Presence': {'base': 55, 'variable': 12}
        }
        
        memory_data = []
        for adapter_name in adapter_names:
            if 'Weather' in adapter_name:
                base_mem = memory_estimates['Weather']['base']
                var_mem = memory_estimates['Weather']['variable']
            elif 'Time' in adapter_name:
                base_mem = memory_estimates['Time Of Day']['base']
                var_mem = memory_estimates['Time Of Day']['variable']
            elif 'Density' in adapter_name:
                base_mem = memory_estimates['Ped Density']['base']
                var_mem = memory_estimates['Ped Density']['variable']
            elif 'Presence' in adapter_name:
                base_mem = memory_estimates['Presence']['base']
                var_mem = memory_estimates['Presence']['variable']
            else:
                base_mem, var_mem = 50, 10
            
            processing_factor = 1.0
            if adapter_name in timing_df.index:
                avg_time = timing_df.loc[adapter_name, 'Mean']
                processing_factor = 1 + (avg_time - timing_df['Mean'].mean()) / timing_df['Mean'].mean() * 0.2
            
            estimated_memory = base_mem * processing_factor
            memory_data.append({'Adapter': adapter_name, 'Memory (MB)': estimated_memory})
        
        if memory_data:
            memory_df = pd.DataFrame(memory_data)
            bars = ax2.bar(range(len(memory_df)), memory_df['Memory (MB)'],
                          color=COLORS[:len(memory_df)], alpha=0.8, edgecolor='black')
            
            ax2.set_xlabel('Adapter Type')
            ax2.set_ylabel('Estimated Memory Usage (MB)')  
            ax2.set_title('B) Memory Usage by Adapter Type')
            ax2.set_xticks(range(len(memory_df)))
            ax2.set_xticklabels(memory_df['Adapter'], rotation=45, ha='right')
            ax2.grid(True, alpha=0.3, axis='y')
            
            for bar, mem in zip(bars, memory_df['Memory (MB)']):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{mem:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # =========================================================================
    # PLOT 3: Batch Processing Efficiency 
    # =========================================================================
    
    if 'video_id' in df.columns and 'frame_processing_time_ms' in df.columns:
        batch_efficiency = []
        
        for video_id in df['video_id'].unique():
            video_data = df[df['video_id'] == video_id].sort_values('frame_id')
            
            if len(video_data) > 1:
                total_time = video_data['frame_processing_time_ms'].sum()
                frame_count = len(video_data)
                avg_time_per_frame = total_time / frame_count
                
                batch_efficiency.append({
                    'Video': video_id,
                    'Frames': frame_count,
                    'Total Time (s)': total_time / 1000,
                    'Avg Time/Frame (ms)': avg_time_per_frame,
                    'Theoretical FPS': 1000 / avg_time_per_frame if avg_time_per_frame > 0 else 0
                })
        
        if batch_efficiency:
            batch_df = pd.DataFrame(batch_efficiency)
            
            scatter = ax3.scatter(batch_df['Frames'], batch_df['Theoretical FPS'], 
                                c=batch_df['Avg Time/Frame (ms)'], cmap='coolwarm_r',
                                s=60, alpha=0.7, edgecolors='black')
            
            ax3.set_xlabel('Number of Frames')
            ax3.set_ylabel('Theoretical FPS') 
            ax3.set_title('C) Batch Processing Efficiency')
            ax3.grid(True, alpha=0.3)

            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('Avg Time/Frame (ms)')
            
            if len(batch_df) > 3:
                z = np.polyfit(batch_df['Frames'], batch_df['Theoretical FPS'], 1)
                p = np.poly1d(z)
                ax3.plot(batch_df['Frames'], p(batch_df['Frames']), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "06_computational_efficiency_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   - Computational efficiency analysis saved")

# =============================================================================
# C6: Multi-Factor Interaction Studies
# =============================================================================

def multi_factor_interaction_analysis(df: pd.DataFrame):
    """Advanced multi-factor interaction analysis and adaptive system design."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # =========================================================================
    # PLOT 1: Weather × Time × Density → Performance Impact
    # =========================================================================
    
    if all(col in df.columns for col in ['pred_weather', 'pred_time_of_day', 'pred_ped_density', 'frame_processing_time_ms']):
        
        interaction_data = df.groupby(['pred_weather', 'pred_time_of_day', 'pred_ped_density']).agg({
            'frame_processing_time_ms': 'mean',
            'density_agreement': 'mean' if 'density_agreement' in df.columns else lambda x: 0,
            'pred_ped_density_confidence': 'mean' if 'pred_ped_density_confidence' in df.columns else lambda x: 0
        }).reset_index()
        
        heatmap_data = interaction_data.pivot_table(
            values='frame_processing_time_ms',
            index=['pred_weather', 'pred_time_of_day'], 
            columns='pred_ped_density',
            fill_value=0
        )
        
        if not heatmap_data.empty:
            sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='coolwarm', ax=ax1,
                       cbar_kws={'label': 'Processing Time (ms)'})
            ax1.set_title('A) Multi-Factor Performance Impact')
            ax1.set_xlabel('Pedestrian Density')
            ax1.set_ylabel('Weather × Time of Day')
    
    # =========================================================================
    # PLOT 2: Risk Assessment Matrix
    # =========================================================================
    
    if all(col in df.columns for col in ['pred_weather', 'pred_road_presence', 'pred_ped_density']):
        
        risk_scores = []
        
        for _, row in df.iterrows():
            risk_score = 0
            
            weather = row['pred_weather']
            if weather in ['rainy', 'snowy', 'foggy']:
                risk_score += 3
            elif weather == 'cloudy':
                risk_score += 1
            
            if 'pred_time_of_day' in df.columns and row['pred_time_of_day'] == 'night':
                risk_score += 2
                
            density = row['pred_ped_density']
            if 'high' in density:
                risk_score += 3
            elif 'medium' in density:
                risk_score += 2
                
            if row['pred_road_presence'] == 'narrow_road':
                risk_score += 2
                
            risk_scores.append(risk_score)
        
        df_temp = df.copy()
        df_temp['risk_score'] = risk_scores
        
        risk_matrix = df_temp.groupby(['pred_weather', 'pred_road_presence'])['risk_score'].mean().unstack(fill_value=0)
        
        if not risk_matrix.empty:
            sns.heatmap(risk_matrix, annot=True, fmt='.1f', cmap='Reds', ax=ax2,
                       cbar_kws={'label': 'Risk Score'})
            ax2.set_title('B) Risk Assessment Matrix')
            ax2.set_xlabel('Road Type')
            ax2.set_ylabel('Weather Condition')
    
    # =========================================================================
    # PLOT 3: Adaptive Triggering Strategy
    # =========================================================================
    
    if all(col in df.columns for col in ['pred_time_of_day', 'pred_ped_density', 'pred_weather']):
        
        strategy_recommendations = []
        
        for _, row in df.iterrows():
            recommendations = {
                'YOLO_Priority': 0,
                'Behavior_Analysis': 0,
                'Resource_Allocation': 'Normal'
            }
            
            if row['pred_time_of_day'] == 'night':
                recommendations['YOLO_Priority'] = 3
                recommendations['Behavior_Analysis'] = 3
                
            density = row['pred_ped_density']
            if 'low' in density:
                recommendations['YOLO_Priority'] = 3
                recommendations['Behavior_Analysis'] = 3
            elif 'medium' in density:
                recommendations['YOLO_Priority'] = 2
                recommendations['Behavior_Analysis'] = 2
            else:
                recommendations['YOLO_Priority'] = 1
                recommendations['Behavior_Analysis'] = 1
                
            weather = row['pred_weather']
            if weather in ['rainy', 'snowy', 'foggy']:
                recommendations['YOLO_Priority'] += 1
                recommendations['Behavior_Analysis'] += 1
                recommendations['Resource_Allocation'] = 'High'
            elif weather == 'sunny':
                recommendations['Resource_Allocation'] = 'Optimized'
                
            strategy_recommendations.append(recommendations)
        
        strategy_df = pd.DataFrame(strategy_recommendations)
        strategy_summary = df_temp[['pred_time_of_day', 'pred_ped_density']].copy()
        strategy_summary['YOLO_Priority'] = strategy_df['YOLO_Priority']
        strategy_summary['Behavior_Analysis'] = strategy_df['Behavior_Analysis']
        
        yolo_strategy = strategy_summary.groupby(['pred_time_of_day', 'pred_ped_density'])['YOLO_Priority'].mean().unstack(fill_value=0)
        
        if not yolo_strategy.empty:
            sns.heatmap(yolo_strategy, annot=True, fmt='.1f', cmap='Blues', ax=ax3,
                       cbar_kws={'label': 'YOLO Priority Level'})
            ax3.set_title('C) Adaptive YOLO Triggering Strategy')
            ax3.set_xlabel('Pedestrian Density')
            ax3.set_ylabel('Time of Day')
    
    # =========================================================================
    # PLOT 4: System Performance Optimization
    # =========================================================================
    
    if 'frame_processing_time_ms' in df.columns:
        
        optimization_data = []
        
        for _, row in df.iterrows():
            complexity_score = 0
            
            if 'pred_ped_density' in df.columns and 'high' in row['pred_ped_density']:
                complexity_score += 3
            elif 'pred_ped_density' in df.columns and 'medium' in row['pred_ped_density']:
                complexity_score += 2
            else:
                complexity_score += 1
                
            if 'pred_time_of_day' in df.columns and row['pred_time_of_day'] == 'night':
                complexity_score += 2
                
            if 'pred_weather' in df.columns and row['pred_weather'] in ['rainy', 'snowy']:
                complexity_score += 2
            
            processing_time = row['frame_processing_time_ms']
            efficiency_score = 1 / (processing_time / 1000) if processing_time > 0 else 0  # FPS
            
            optimization_data.append({
                'Complexity': complexity_score,
                'Processing_Time': processing_time,
                'Efficiency': efficiency_score
            })
        
        opt_df = pd.DataFrame(optimization_data)
        
        scatter = ax4.scatter(opt_df['Complexity'], opt_df['Efficiency'], 
                            c=opt_df['Processing_Time'], cmap='coolwarm_r',
                            s=50, alpha=0.7, edgecolors='black')
        
        ax4.set_xlabel('Scenario Complexity Score')
        ax4.set_ylabel('Processing Efficiency (FPS)')
        ax4.set_title('D) Performance Optimization Analysis')
        ax4.grid(True, alpha=0.3)
        
        if len(opt_df) > 5:
            z = np.polyfit(opt_df['Complexity'], opt_df['Efficiency'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(opt_df['Complexity'].min(), opt_df['Complexity'].max(), 100)
            ax4.plot(x_trend, p(x_trend), "r-", alpha=0.8, linewidth=2, label='Trend')
            ax4.legend()
        
        plt.colorbar(scatter, ax=ax4, label='Processing Time (ms)')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "07_multi_factor_interaction_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   - Multi-factor interaction analysis saved")

# =============================================================================
# C7: Adaptive System Design Recommendations
# =============================================================================

def adaptive_system_design_analysis(df: pd.DataFrame):
    """Generate adaptive system design recommendations based on analysis."""
    
    with open(PLOTS_DIR / "08_adaptive_system_recommendations.md", "w") as f:
        f.write("# Adaptive System Design Recommendations\n\n")
        f.write("Evidence-based recommendations for intelligent pedestrian monitoring system.\n\n")
        
        f.write("## Performance Summary\n\n")
        
        if 'frame_processing_time_ms' in df.columns:
            avg_processing = df['frame_processing_time_ms'].mean()
            p95_processing = df['frame_processing_time_ms'].quantile(0.95)
            f.write(f"- **Average processing time**: {avg_processing:.2f} ms\n")
            f.write(f"- **95th percentile**: {p95_processing:.2f} ms\n")
        
        if 'density_agreement' in df.columns:
            avg_agreement = df['density_agreement'].mean()
            f.write(f"- **Adapter-YOLO agreement**: {avg_agreement:.3f} ({avg_agreement*100:.1f}%)\n\n")
        
        f.write("## Hypothesis Validation\n\n")
        
        if 'pred_ped_density' in df.columns:
            low_density_data = df[df['pred_ped_density'] == 'low_pedestrian_density']
            if not low_density_data.empty:
                low_density_pct = len(low_density_data) / len(df) * 100
                f.write(f"### H1: Low Density Individual Analysis\n")
                f.write(f"- **Low density frames**: {low_density_pct:.1f}% of dataset\n")
                
                if 'pred_ped_density_confidence' in df.columns:
                    low_density_conf = low_density_data['pred_ped_density_confidence'].mean()
                    f.write(f"- **Confidence in low density**: {low_density_conf:.3f}\n")
                
                f.write(f"- ** Recommendation**: Always trigger detailed analysis for sparse scenes\n\n")
        
        if 'pred_time_of_day' in df.columns:
            night_data = df[df['pred_time_of_day'] == 'night']
            day_data = df[df['pred_time_of_day'] == 'day']
            
            if not night_data.empty and not day_data.empty:
                f.write(f"### H2: Nighttime Enhanced Monitoring\n")
                
                if 'pred_ped_density_confidence' in df.columns:
                    night_conf = night_data['pred_ped_density_confidence'].mean()
                    day_conf = day_data['pred_ped_density_confidence'].mean()
                    f.write(f"- **Night confidence**: {night_conf:.3f} vs **Day confidence**: {day_conf:.3f}\n")
                    
                if 'frame_processing_time_ms' in df.columns:
                    night_time = night_data['frame_processing_time_ms'].mean()
                    day_time = day_data['frame_processing_time_ms'].mean()
                    f.write(f"- **Night processing**: {night_time:.1f}ms vs **Day processing**: {day_time:.1f}ms\n")
                
                f.write(f"- ** Recommendation**: Increase monitoring frequency during nighttime\n\n")
        
        if 'pred_weather' in df.columns:
            f.write(f"### H3: Weather Impact Analysis\n")
            weather_impact = df.groupby('pred_weather').agg({
                'frame_processing_time_ms': 'mean' if 'frame_processing_time_ms' in df.columns else lambda x: 0,
                'pred_ped_density_confidence': 'mean' if 'pred_ped_density_confidence' in df.columns else lambda x: 0
            }).round(2)
            
            for weather, metrics in weather_impact.iterrows():
                if not pd.isna(weather):
                    f.write(f"- **{weather.title()}**: ")
                    if 'frame_processing_time_ms' in df.columns:
                        f.write(f"{metrics['frame_processing_time_ms']}ms processing, ")
                    if 'pred_ped_density_confidence' in df.columns:
                        f.write(f"{metrics['pred_ped_density_confidence']:.3f} confidence")
                    f.write("\n")
            
            f.write(f"- ** Recommendation**: Adjust detection sensitivity based on weather conditions\n\n")
        
        if 'pred_road_presence' in df.columns:
            f.write(f"### H4: Road Width Impact\n")
            road_impact = df.groupby('pred_road_presence').agg({
                'frame_processing_time_ms': 'mean' if 'frame_processing_time_ms' in df.columns else lambda x: 0,
                'pred_ped_density': lambda x: x.mode().iloc[0] if not x.empty else 'unknown'
            })
            
            for road_presence, metrics in road_impact.iterrows():
                if not pd.isna(road_presence):
                    f.write(f"- **{road_presence.replace('_', ' ').title()}**: ")
                    f.write(f"Typical density: {metrics['pred_ped_density']}, ")
                    if 'frame_processing_time_ms' in df.columns:
                        f.write(f"{metrics['frame_processing_time_ms']:.1f}ms processing")
                    f.write("\n")
            
            f.write(f"- ** Recommendation**: Adapt crossing time estimates based on road geometry\n\n")
        
        # Adaptive Triggering Strategy
        f.write("## Adaptive Triggering Strategy\n\n")
        f.write("Based on analysis results, implement the following decision tree:\n\n")
        
        f.write("### Priority Level 1 (Always Trigger)\n")
        f.write("- Nighttime conditions (visibility reduced)\n")
        f.write("- Low pedestrian density (≤3 pedestrians)\n") 
        f.write("- Adverse weather (rainy, snowy, foggy)\n")
        
        f.write("### Priority Level 2 (Selective Triggering)\n")
        f.write("- Medium pedestrian density (4-7 pedestrians)\n")
        f.write("- Cloudy weather conditions\n")
        f.write("- Wide road with high activity\n\n")
        
        f.write("### Priority Level 3 (Optimized Triggering)\n")
        f.write("- High pedestrian density (>7 pedestrians)\n") 
        f.write("- Sunny daytime conditions\n")
        f.write("- Stable environmental conditions\n\n")
        
        f.write("## Resource Allocation Strategy\n\n")
        
        if 'pred_ped_density_inference_time_ms' in df.columns and 'yolo_ped_density_inference_time_ms' in df.columns:
            adapter_time = df['pred_ped_density_inference_time_ms'].mean()
            yolo_time = df['yolo_ped_density_inference_time_ms'].mean()
            
            f.write(f"### Computational Efficiency\n")
            f.write(f"- **Adapter processing**: {adapter_time:.2f}ms average\n")
            f.write(f"- **YOLO processing**: {yolo_time:.2f}ms average\n")
            f.write(f"- **Speed advantage**: {yolo_time/adapter_time:.2f}x {'(YOLO faster)' if yolo_time < adapter_time else '(Adapter faster)'}\n\n")
        
        f.write("### Recommended Architecture\n")
        f.write("1. **Primary Detection**: YOLO for rapid pedestrian counting\n")
        f.write("2. **Behavioral Analysis**: Adapters for detailed behavior prediction\n") 
        f.write("3. **Context Awareness**: Scene adapters for environmental understanding\n")
        f.write("4. **Adaptive Pipeline**: Dynamic triggering based on scenario complexity\n\n")
        
       
        # Performance Targets
        f.write("## Performance Targets\n\n")
        f.write("| Metric | Target | Current Performance |\n")
        f.write("|--------|--------|---------------------|\n")
        
        if 'frame_processing_time_ms' in df.columns:
            current_fps = 1000 / df['frame_processing_time_ms'].mean()
            f.write(f"| Processing Speed | >30 FPS | {current_fps:.1f} FPS |\n")
        
        if 'density_agreement' in df.columns:
            current_accuracy = df['density_agreement'].mean()
            f.write(f"| Classification Accuracy | >95% | {current_accuracy*100:.1f}% |\n")
        
        if 'pred_ped_density_confidence' in df.columns:
            current_confidence = df['pred_ped_density_confidence'].mean()
            f.write(f"| Confidence Score | >0.8 | {current_confidence:.3f} |\n")
        
        f.write("\n")
        
        f.write("---\n")
        f.write("*Report generated by Scene Contextual Analysis System*\n")
    
    # Generate decision tree visualization
    create_decision_tree_visualization(df)
    
    print("  - Adaptive system design recommendations saved")

def create_decision_tree_visualization(df: pd.DataFrame):
    """Create visual decision tree for adaptive system design."""
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    decision_nodes = {
        'Root': {'pos': (0.5, 0.9), 'text': 'New Frame\nDetected', 'color': COLORS[0]},
        'Time Check': {'pos': (0.3, 0.7), 'text': 'Time of Day?', 'color': COLORS[1]},
        'Weather Check': {'pos': (0.7, 0.7), 'text': 'Weather\nConditions?', 'color': COLORS[1]},
        'Night': {'pos': (0.15, 0.5), 'text': 'NIGHT\n→ High Priority', 'color': COLORS[2]},
        'Day': {'pos': (0.45, 0.5), 'text': 'DAY\n→ Check Density', 'color': COLORS[3]},
        'Adverse': {'pos': (0.6, 0.5), 'text': 'ADVERSE\n→ High Priority', 'color': COLORS[2]},
        'Clear': {'pos': (0.85, 0.5), 'text': 'CLEAR\n→ Optimize', 'color': COLORS[3]},
        'Low Density': {'pos': (0.2, 0.3), 'text': 'LOW DENSITY\n→ Full Analysis', 'color': COLORS[2]},
        'Med Density': {'pos': (0.45, 0.3), 'text': 'MED DENSITY\n→ Selective', 'color': COLORS[1]},
        'High Density': {'pos': (0.7, 0.3), 'text': 'HIGH DENSITY\n→ Crowd Mode', 'color': COLORS[3]},
        'Action 1': {'pos': (0.1, 0.1), 'text': 'Trigger:\n• YOLO: ON\n• Behavior: ON\n• Context: ON', 'color': COLORS[2]},
        'Action 2': {'pos': (0.35, 0.1), 'text': 'Trigger:\n• YOLO: ON\n• Behavior: SELECTIVE\n• Context: BASIC', 'color': COLORS[1]},
        'Action 3': {'pos': (0.65, 0.1), 'text': 'Trigger:\n• YOLO: OPTIMIZED\n• Behavior: OFF\n• Context: BASIC', 'color': COLORS[3]},
    }
    
    connections = [
        ('Root', 'Time Check'),
        ('Root', 'Weather Check'),
        ('Time Check', 'Night'),
        ('Time Check', 'Day'),
        ('Weather Check', 'Adverse'), 
        ('Weather Check', 'Clear'),
        ('Day', 'Low Density'),
        ('Day', 'Med Density'),
        ('Day', 'High Density'),
        ('Night', 'Action 1'),
        ('Adverse', 'Action 1'),
        ('Low Density', 'Action 1'),
        ('Med Density', 'Action 2'),
        ('High Density', 'Action 3'),
        ('Clear', 'Action 3')
    ]
    
    for start, end in connections:
        start_pos = decision_nodes[start]['pos']
        end_pos = decision_nodes[end]['pos']
        ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
               'k-', alpha=0.5, linewidth=2)
    
    for node_name, node_info in decision_nodes.items():
        pos = node_info['pos']
        text = node_info['text']
        color = node_info['color']
        
        if 'Action' in node_name:
            bbox_props = dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.8, edgecolor='black')
            ax.text(pos[0], pos[1], text, ha='center', va='center', 
                   bbox=bbox_props, fontsize=9, fontweight='bold')
        else:
            circle = plt.Circle(pos, 0.06, color=color, alpha=0.8, ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], text, ha='center', va='center', 
                   fontsize=8, fontweight='bold', color='white')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Adaptive System Decision Tree', fontsize=16, fontweight='bold', pad=20)
    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[2], 
                   markersize=10, label='High Priority'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[1],
                   markersize=10, label='Medium Priority'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[3],
                   markersize=10, label='Optimized Mode')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "08_adaptive_decision_tree.png", dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# D. MAIN EXECUTION FLOW
# =============================================================================
def main(scene_csv_path: str, behavior_csv_path: Optional[str] = None):
    """Main execution function for the complete scene context analysis pipeline."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print(" Starting Scene Context Analysis & Visualization (Step 3)")
    print("=" * 60)
    
    if not os.path.exists(scene_csv_path):
        raise FileNotFoundError(f"Input scene CSV file not found at: {scene_csv_path}")
    
    print(f"Loading data from '{scene_csv_path}'...")
    df = load_scene_data(scene_csv_path)
    df = merge_behavior_data(df, behavior_csv_path)
    print("Data loading and preprocessing complete.")
    
    print("\n--- Generating Analysis Modules ---")
    adapter_vs_yolo_performance_analysis(df)
    environmental_impact_analysis(df)
    road_geometry_analysis(df)
    density_performance_analysis(df)
    computational_efficiency_analysis(df)
    multi_factor_interaction_analysis(df)
    adaptive_system_design_analysis(df)
    
    print("\n" + "=" * 60)
    print(" Scene Context Analysis Successfully Completed!")
    print(f" All outputs saved to '{PLOTS_DIR.resolve()}'")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run comprehensive analysis on scene contextual data from Step 2.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--scene", 
        type=str, 
        default=os.path.join("Generated_Data", "raw_data_scene_contextual_analysis_step2.csv"),
        help="Path to the raw scene context CSV file."
    )
    parser.add_argument(
        "--behavior", 
        type=str, 
        default=os.path.join("Generated_Data", "raw_data_pedestrian_behavior_step2.csv"),
        help="Optional path to the raw pedestrian behavior CSV for merged analysis."
    )
    args = parser.parse_args()
    main(args.scene, args.behavior)
