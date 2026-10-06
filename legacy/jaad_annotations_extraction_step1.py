"""
JAAD Dataset Annotation Extraction (Step 1)

Description:
    This script is the first step in the data preparation pipeline. It reads the
    raw XML annotation files from the JAAD dataset, processes them, and extracts
    relevant information about pedestrian behavior and scene context.

    The pipeline performs several key functions:
    1.  Parses individual XML files for video attributes and pedestrian track data.
    2.  Normalizes and maps raw annotation labels (e.g., 'clear', 'rain') into
        a standardized set of categories (e.g., 'sunny', 'rainy').
    3.  Calculates frame-level pedestrian density based on the number of bounding
        boxes present.
    4.  Cleans the data by removing entries with incomplete or 'unknown' labels
        for critical attributes like weather and time of day.
    5.  Aggregates the cleaned data from all videos into a single dataset.
    6.  Saves the final, processed dataset in both JSON and CSV formats.
    7.  Generates a class distribution report (`class_distribution_cleaned...txt`)
        to provide a statistical summary of the cleaned dataset.

Inputs:
    - JAAD Dataset Annotations:
        - ./JAAD/annotations/video_*.xml
        - ./JAAD/annotations_attributes/video_*_attributes.xml

Outputs:
    - Processed Data (JSON): jaad_annotations_extracted_data_step1.json
    - Processed Data (CSV): jaad_annotations_extracted_data_step1.csv
    - Class Distribution Report: class_distribution_cleaned_[timestamp].txt

Command-Line Arguments:
    -   `--base_path` (str, optional):
        Specifies the base path to the JAAD dataset directory. If not provided,
        the script will search for the dataset in the current directory and
        then in a subdirectory named `JAAD/`.
        Default: ""

    -   `--output` (str, optional):
        Defines the base name for the output JSON and CSV files.
        Default: "jaad_annotations_extracted_data_step1.json"

    -   `--video_id` (str, optional):
        If provided, the script will process only the specified video ID
        instead of all videos in the dataset. This is useful for debugging
        or targeted data extraction.
"""
import xml.etree.ElementTree as ET
from typing import Dict, List
from collections import defaultdict
import json
import csv
import sys
from datetime import datetime
from pathlib import Path
import os

# =============================================================================
# A. DATA MAPPING & NORMALIZATION FUNCTIONS
# =============================================================================

def map_time_of_day(time: str) -> str:
    """Map time of day to day/night categories."""
    
    time = str(time).lower()
    if time in ['daytime', 'afternoon']:
        return 'day'
    elif time in ['evening', 'nighttime']:
        return 'night'
    return 'unknown'

def map_weather(weather: str) -> str:
    """Map weather conditions to simplified categories."""
    
    weather = str(weather).lower()
    if weather in ['clear', 'sunny']:
        return 'sunny'
    elif weather in ['cloud', 'cloudy', 'overcast']:
        return 'cloudy'
    elif weather in ['rain', 'rainy']:
        return 'rainy'
    elif weather in ['snow', 'snowy']:
        return 'snowy'
    return 'unknown'

def map_road_presence(num_lanes: int) -> str:
    """Map number of lanes to road type."""

    try:
        num_lanes = int(num_lanes)
        if num_lanes >= 3:
            return 'wide_road'
        elif num_lanes <= 2:
            return 'narrow_road'
    except (ValueError, TypeError):
        pass
    return 'unknown'

def map_occlusion(occ_attr: str) -> str:
    """Map occlusion attribute to standardized categories."""
    
    if occ_attr == 'full':
        return 'full_occlusion'
    elif occ_attr == 'part':
        return 'partial_occlusion'
    elif occ_attr == 'none':
        return 'no_occlusion'
    return 'unknown'

def map_crossing(cross_attr: str) -> str:
    """Map crossing attribute to standardized categories."""

    if cross_attr == 'crossing':
        return 'crossing'
    elif cross_attr == 'not-crossing':
        return 'not_crossing'
    return 'unknown'

def map_looking(look_attr: str) -> str:
    """Map looking attribute to standardized categories."""
    
    if look_attr == 'looking':
        return 'looking'
    elif look_attr == 'not-looking':
        return 'not_looking'
    return 'unknown'

def map_action(action_attr: str) -> str:
    """Map action attribute to walking/standing."""

    if action_attr in ['walking', 'walk']:
        return 'walking'
    elif action_attr in ['standing', 'stand']:
        return 'standing'
    return 'unknown'

def calculate_pedestrian_density(frame_data: List[Dict]) -> str:
    """Calculate pedestrian density based on number of bounding boxes in frame."""

    cnt = len(frame_data)
    if cnt > 7:
        return 'high_pedestrian_density'
    elif 4 <= cnt <= 7:
        return 'medium_pedestrian_density'
    return 'low_pedestrian_density'

# =============================================================================
# B. FILE I/O AND PARSING UTILITIES
# =============================================================================

def resolve_dataset_root(base_path: str) -> Path:
    """Return the directory that actually contains the annotation folders.

    1. If ``base_path`` itself already contains ``annotations/`` it is returned.
    2. Otherwise, if ``base_path/JAAD`` contains ``annotations/`` that path is
       returned. This allows users to keep the original behaviour (dataset at
       project root) *or* the new layout (all data inside a ``JAAD`` folder)
       without changing the command line flags.
    """

    base = Path(base_path or ".").resolve()

    if (base / "annotations").is_dir():
        return base

    nested = base / "JAAD"
    if (nested / "annotations").is_dir():
        return nested

    return base

def print_progress(current: int, total: int, prefix: str = '', suffix: str = '') -> None:
    """Print a simple command-line progress bar."""
    
    bar_length = 50
    filled_length = int(round(bar_length * current / float(total)))
    percents = round(100.0 * current / float(total), 1)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write(f'\r{prefix} [{bar}] {percents}% {suffix}')
    sys.stdout.flush()
    if current == total:
        print()

def load_annotations_xml(file_path: str) -> Dict:
    """Load and parse a video's primary annotations XML file."""

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        video_attrs = {}
        meta = root.find('meta')
        if meta is not None:
            task = meta.find('task')
            if task is not None:
                video_attributes = task.find('video_attributes')
                if video_attributes is not None:
                    for attr in video_attributes:
                        video_attrs[attr.tag] = attr.text
        
        tracks = []
        for track in root.findall('track'):
            track_label = track.get('label', '')
            track_data = {
                'label': track_label,
                'boxes': []
            }
            
            for box in track.findall('box'):
                box_data = {
                    'frame': int(box.get('frame', 0)),
                    'xtl': float(box.get('xtl', 0)),
                    'ytl': float(box.get('ytl', 0)),
                    'xbr': float(box.get('xbr', 0)),
                    'ybr': float(box.get('ybr', 0)),
                    'attributes': {}
                }
                
                for attr in box.findall('attribute'):
                    attr_name = attr.get('name', '')
                    attr_value = attr.text
                    box_data['attributes'][attr_name] = attr_value
                
                track_data['boxes'].append(box_data)
            
            tracks.append(track_data)
        
        return {
            'video_attributes': video_attrs,
            'tracks': tracks
        }
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_attributes_xml(file_path: str) -> List[Dict]:
    """Load and parse the corresponding pedestrian attributes XML file."""

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        pedestrians = []
        for ped in root.findall('pedestrian'):
            ped_data = {}
            for key, value in ped.attrib.items():
                ped_data[key] = value
            pedestrians.append(ped_data)
        
        return pedestrians
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

# =============================================================================
# C. CORE DATA PROCESSING
# =============================================================================

def clean_data(data: List[Dict]) -> List[Dict]:
    """Remove rows with unknown time_of_day and weather."""

    cleaned_data = [
        row for row in data 
        if row.get('time_of_day') != 'unknown' and row.get('weather') != 'unknown'
    ]
    return cleaned_data

def process_video(video_id: str, base_path: str = "") -> List[Dict]:
    """Process a single video's annotations and extract all required data."""

    dataset_root = resolve_dataset_root(base_path)

    annotations_file = dataset_root / "annotations" / f"video_{video_id}.xml"
    attributes_file = dataset_root / "annotations_attributes" / f"video_{video_id}_attributes.xml"
    
    if not annotations_file.exists():
        print(f"Annotations file not found: {annotations_file}")
        return []

    if not attributes_file.exists():
        print(f"Attributes file not found: {attributes_file}")
        return []
    
    annotations_data = load_annotations_xml(str(annotations_file))
    attributes_data = load_attributes_xml(str(attributes_file))
    
    if annotations_data is None:
        return []
    
    ped_attributes = {}
    for ped in attributes_data:
        ped_id = ped.get('id', '')
        ped_attributes[ped_id] = ped
    
    video_attrs = annotations_data['video_attributes']
    time_of_day = map_time_of_day(video_attrs.get('time_of_day', ''))
    weather = map_weather(video_attrs.get('weather', ''))
    
    all_frames_data = []
    frame_groups = defaultdict(list)
    
    for track in annotations_data['tracks']:
        track_label = track['label']
        
        if track_label not in ['pedestrian', 'ped', 'people']:
            continue
        
        for box in track['boxes']:
            frame_id = box['frame']
            track_id = box['attributes'].get('id', '')
            
            ped_attr = ped_attributes.get(track_id, {})
            num_lanes = ped_attr.get('num_lanes', '2')
            road_presence = map_road_presence(num_lanes)
            
            bounding_box = [box['xtl'], box['ytl'], box['xbr'], box['ybr']]
            
            occlusion_status = map_occlusion(box['attributes'].get('occlusion', 'none'))
            action = map_action(box['attributes'].get('action', 'standing'))
            look_status = map_looking(box['attributes'].get('look', 'not-looking'))
            crossing_status = map_crossing(box['attributes'].get('cross', 'not-crossing'))
            
            frame_data = {
                'video_id': video_id,
                'frame_id': frame_id,
                'track_id': track_id,
                'label': track_label,
                'bbox_xtl': bounding_box[0],
                'bbox_ytl': bounding_box[1],
                'bbox_xbr': bounding_box[2],
                'bbox_ybr': bounding_box[3],
                'occlusion': occlusion_status,
                'action': action,
                'look': look_status,
                'cross': crossing_status,
                'road_presence': road_presence,
                'weather': weather,
                'time_of_day': time_of_day
            }
            
            frame_groups[frame_id].append(frame_data)
    
    for frame_id, frame_data_list in frame_groups.items():
        ped_density = calculate_pedestrian_density(frame_data_list)
        for data in frame_data_list:
            data['pedestrian_density'] = ped_density
            all_frames_data.append(data)
    
    return all_frames_data

def get_video_ids(base_path: str = "") -> List[str]:
    """Return a sorted list of all available video IDs in the dataset."""

    dataset_root = resolve_dataset_root(base_path)
    annotations_dir = dataset_root / "annotations"

    video_ids: List[str] = []

    if annotations_dir.is_dir():
        for file in annotations_dir.glob("video_*.xml"):
            video_ids.append(file.stem.replace("video_", ""))

    return sorted(video_ids)

# =============================================================================
# D. DATA ANALYSIS & REPORTING
# =============================================================================

def analyze_class_distribution(data: List[Dict]) -> Dict:
    """Analyze and calculate class distribution for all categorical variables."""

    distributions = defaultdict(lambda: defaultdict(int))
    categorical_cols = ['label', 'occlusion', 'action', 'look', 'cross', 
                       'pedestrian_density', 'road_presence', 'weather', 'time_of_day']
    
    total = len(data)
    for entry in data:
        for col in categorical_cols:
            if col in entry:
                distributions[col][entry[col]] += 1
    
    result = {}
    for col, counts in distributions.items():
        result[col] = {
            'counts': dict(counts),
            'percentages': {k: round(v/total * 100, 2) for k, v in counts.items()}
        }
    
    return result

def save_class_distribution(distribution: Dict, output_dir: Path) -> Path:
    """Saves the class distribution analysis to a text file and returns the path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"class_distribution_cleaned_{timestamp}.txt"
    output_path = output_dir / output_filename
    
    with open(output_path, 'w') as f:
        f.write("JAAD Dataset Class Distribution Analysis (After Data Cleaning)\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        for category, data in distribution.items():
            f.write(f"\n{category.upper()}\n")
            f.write("-" * len(category) + "\n")
            f.write("Counts:\n")
            for label, count in data['counts'].items():
                f.write(f"  {label}: {count}\n")
            f.write("\nPercentages:\n")
            for label, percentage in data['percentages'].items():
                f.write(f"  {label}: {percentage}%\n")
            f.write("\n")
    return output_path

# =============================================================================
# E. MAIN EXECUTION FLOW
# =============================================================================

def process_all_videos(base_path: str = "", output_file: str = "jaad_annotations_extracted_data_step1") -> None:
    """Process all available videos, save the aggregated data, and print summary stats."""

    video_ids = get_video_ids(base_path)
    total_videos = len(video_ids)
    print(f"Found {total_videos} videos to process")
    
    all_data = []
    
    for i, video_id in enumerate(video_ids, 1):
        print_progress(i, total_videos, prefix='Processing videos:', 
                      suffix=f'Video {video_id} ({i}/{total_videos})')
        video_data = process_video(video_id, base_path)
        all_data.extend(video_data)
    
    print("\nCleaning data...")
    cleaned_data = clean_data(all_data)
    
    json_file = output_file + '.json'
    print(f"\nSaving {len(cleaned_data)} cleaned data points to {json_file}")
    with open(json_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    csv_file = output_file + '.csv'
    print(f"Saving cleaned data to CSV: {csv_file}")
    if cleaned_data:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=cleaned_data[0].keys())
            writer.writeheader()
            writer.writerows(cleaned_data)
    
    print("\nAnalyzing class distribution...")
    distribution = analyze_class_distribution(cleaned_data)
    output_dir = Path(output_file).parent
    dist_file_path = save_class_distribution(distribution, output_dir)
    
    video_counts = defaultdict(int)
    frame_counts = defaultdict(int)
    track_counts = set()
    
    for data in cleaned_data:
        video_counts[data['video_id']] += 1
        frame_counts[f"{data['video_id']}_{data['frame_id']}"] += 1
        track_counts.add(f"{data['video_id']}_{data['track_id']}")
    
    print("\nFinal Statistics (After Cleaning):")
    print(f"Total videos processed: {len(video_counts)}")
    print(f"Total unique frames: {len(frame_counts)}")
    print(f"Total unique pedestrian tracks: {len(track_counts)}")
    print(f"Total data points: {len(cleaned_data)}")
    print(f"\nResults saved to:")
    print(f"- JSON: {json_file}")
    print(f"- CSV: {csv_file}")
    print(f"- Class distribution: {dist_file_path}")

def main():
    """Main function to parse arguments and run the data extraction."""

    import argparse
    parser = argparse.ArgumentParser(description='Process JAAD annotations and extract pedestrian behavior data and scene context')
    parser.add_argument('--base_path', type=str, default="",
                      help='Base path to JAAD dataset directory')
    parser.add_argument('--output', type=str, default=os.path.join("Generated_Data", "jaad_annotations_extracted_data_step1"),
                      help='Output file base path for processed data (without extension)')
    parser.add_argument('--video_id', type=str,
                      help='Process specific video ID only')
    
    args = parser.parse_args()
    
    Path("Generated_Data").mkdir(exist_ok=True)
    
    if args.video_id:
        print(f"Processing video {args.video_id}")
        data = process_video(args.video_id, args.base_path)
        cleaned_data = clean_data(data)
        with open(args.output, 'w') as f:
            json.dump(cleaned_data, f, indent=2)
        print(f"Processed {len(cleaned_data)} cleaned data points")
    else:
        process_all_videos(args.base_path, args.output)

if __name__ == "__main__":
    main()
