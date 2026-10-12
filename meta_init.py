import os
import json
import argparse
from pathlib import Path

# Fixed parameters as requested

DEFAULT_CAMERA_EXTRINSICS = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0]
]

# Standard EGO intrinsics (512x512 scale) required by EgoX downscaling model
DEFAULT_EGO_INTRINSICS = [
    [150.0, 0.0, 255.5],
    [0.0, 150.0, 255.5],
    [0.0, 0.0, 1.0]
]

def create_meta_init(folder_path, output_json_path, overwrite=False):
    """
    Initialize meta.json by scanning for exo.mp4 files in the given folder.
    Automatically calculates camera_intrinsics based on video resolution.
    Structure:
      folder_path/videos/take_name/exo.mp4
    """
    
    base_dir = Path(folder_path).resolve()
    videos_dir = base_dir / "videos"
    
    if not videos_dir.exists():
        print(f"Error: Videos directory not found at {videos_dir}")
        print(f"Make sure your structure is: {folder_path}/videos/take_name/exo.mp4")
        return
        
    print(f"Scanning for videos in {videos_dir}...")
    
    datasets = []
    
    # Iterate over subdirectories in videos folder
    for video_folder in videos_dir.iterdir():
        if video_folder.is_dir():
            exo_path = video_folder / "exo.mp4"
            
            if exo_path.exists():
                try:
                    rel_exo_path = os.path.relpath(exo_path, os.getcwd())
                    if not rel_exo_path.startswith(".") and not rel_exo_path.startswith("/"):
                        rel_exo_path = "./" + rel_exo_path
                except ValueError:
                    rel_exo_path = str(exo_path)

                # Try to load video with cv2 to calculate intrinsics
                w, h = 1920, 1080  # Default fallback
                try:
                    import cv2
                    cap = cv2.VideoCapture(str(exo_path))
                    w_cap = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h_cap = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    if w_cap > 0 and h_cap > 0:
                        w, h = w_cap, h_cap
                        print(f"Detected video resolution for {video_folder.name}: {w} x {h}")
                except Exception as e:
                    print(f"Note: OpenCV not available in current env, using 1920x1080 fallback: {e}")

                # Calculate camera_intrinsics dynamically
                f = float(max(w, h))
                cx = float(w) / 2.0
                cy = float(h) / 2.0
                camera_intrinsics = [
                    [f, 0.0, cx],
                    [0.0, f, cy],
                    [0.0, 0.0, 1.0]
                ]

                dataset_entry = {
                    "exo_path": rel_exo_path,
                    "prompt": "", 
                    "camera_intrinsics": camera_intrinsics,
                    "camera_extrinsics": DEFAULT_CAMERA_EXTRINSICS,
                    "ego_intrinsics": DEFAULT_EGO_INTRINSICS,
                }
                
                # Check for ego_Prior.mp4
                ego_path = video_folder / "ego_Prior.mp4"
                if ego_path.exists():
                     try:
                        rel_ego_path = os.path.relpath(ego_path, os.getcwd())
                        if not rel_ego_path.startswith(".") and not rel_ego_path.startswith("/"):
                            rel_ego_path = "./" + rel_ego_path
                        dataset_entry["ego_prior_path"] = rel_ego_path
                     except:
                        dataset_entry["ego_prior_path"] = str(ego_path)

                datasets.append(dataset_entry)
                print(f"Found and configured: {video_folder.name}")
    
    if not datasets:
        print("No video folders with exo.mp4 found.")
        return

    output_data = {
        "test_datasets": datasets
    }
    
    output_path = Path(output_json_path)
    
    # Handle existing file
    if output_path.exists() and not overwrite:
        print(f"Error: Output file {output_json_path} already exists. Use --overwrite to replace it.")
        return
    
    # Ensure parent dir exists
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)
        
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        print(f"\nSuccessfully created {output_json_path}")
        print(f"Total tasks: {len(datasets)}")
    except Exception as e:
        print(f"Error writing to file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Initialize meta.json from video folder structure")
    parser.add_argument("--folder_path", "-f", required=True,
                       help="Path to the custom dataset folder (containing 'videos' subdir)")
    parser.add_argument("--output_json", "-o", required=True,
                       help="Path to output meta.json file (e.g., ./your_dataset/meta.json)")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing file")
    
    args = parser.parse_args()
    
    create_meta_init(args.folder_path, args.output_json, args.overwrite)

if __name__ == "__main__":
    main()
