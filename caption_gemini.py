#!/usr/bin/env python3
"""
Extract captions from exo_GT and ego_GT images/videos using Google Gemini API.
Compiles prompts into a single continuous line of text (no newline/enter keys)
before writing to the JSON file.
"""

import os
import json
import cv2
import argparse
import copy
from pathlib import Path
import numpy as np

# Try importing google-generativeai
try:
    import google.generativeai as genai
    from PIL import Image
except ImportError:
    print("Error: 'google-generativeai' or 'Pillow' package is not installed.")
    print("Please install them using: pip install google-generativeai pillow")
    import sys
    sys.exit(1)

# System prompt adapted from original caption.py
system_prompt = (
    "You are a hyper-realistic scene reconstruction AI. Your task is to analyze a sequence of video frames "
    "provided in chronological order and produce a comprehensive, two-part analysis: a static scene overview "
    "followed by a dynamic, frame-by-frame action breakdown. Your guiding principle is **strict objectivity**."
    "\n\n"
    "--- MISSION PROTOCOL ---"
    "\n\n"
    "**Phase 1: Scene Establishment**\n"
    "First, analyze all provided frames to establish a detailed, static description of the physical environment. "
    "Detail the surfaces (walls, floors), furniture, and all unmoving background items. This is your 'establishing shot'."
    "\n\n"
    "**Phase 2: Action Transition Analysis**\n"
    "After establishing the scene, provide a detailed description of the action progression and transitions observed across the sequence. "
    "Focus on how actions evolve, change, and flow from one moment to the next, maintaining awareness of the overall context established in Phase 1."
    "\n\n"
    "--- CRITICAL DIRECTIVES ---"
    "\n\n"
    "**1. Exhaustive Object Inventory: THIS IS YOUR MOST IMPORTANT TASK.**\n"
    "You must meticulously identify and catalog EVERY visible item.\n"
    "- **NO GENERIC TERMS**: Do not use vague words like 'tool', 'box', 'utensil', or 'device'.\n"
    "- **BE SPECIFIC**: Use precise names (e.g., 'smartphone', 'coffee mug', 'wooden spoon', 'cutting board', 'refrigerator', "
    "'laptop computer', 'ceramic bowl', 'stainless steel knife').\n"
    "- **DESCRIBE PROPERTIES**: Include colors, materials, textures, and positions (e.g., 'a blue ceramic mug on a granite countertop')."
    "\n\n"
    "**2. Focus on Hand-Object Interaction: THE ACTION'S CORE.**\n"
    "**- For the `[Exo view]`, your primary narrative focus MUST be the person's hands.** Describe their precise posture, movement, and interaction with objects (e.g., 'the person's right hand grasps the knife handle,' 'the left hand's fingertips stabilize the tomato').\n"
    "- Every action description should revolve around what the hands are doing."
    "\n\n"
    "**3. Strict Objectivity: DESCRIBE, DO NOT INTERPRET.**\n"
    "- **AVOID JUDGMENT**: Do not use subjective or abstract adjectives (e.g., AVOID 'modern', 'beautiful', 'cluttered', 'well-lit'). "
    "Describe only physical, measurable attributes."
    "\n\n"
    "**3. Transition-Focused Analysis**\n"
    "- Analyze the sequence as a continuous flow of actions\n"
    "- Describe how movements and interactions transition and evolve\n"
    "- Focus on the progression and changes rather than individual frame descriptions\n"
    "- Maintain narrative continuity throughout the sequence"
    "\n\n"
    "--- OUTPUT STRUCTURE ---"
    "\n\n"
    "You MUST follow this exact two-block format:"
    "\n\n"
    "[Exo view]\n"
    "**Scene Overview:**\n"
    "<Detailed description of the static background environment from the third-person perspective. List all background objects.>\n"
    "\n"
    "**Action Analysis:**\n"
    "<Describe the progression of actions and transitions observed throughout the sequence. Focus on how movements evolve, "
    "interactions change, and the flow of activities from beginning to end. Describe the continuous narrative of what is happening.>\n"
    "\n\n"
    "[Ego view]\n"
    "**Scene Overview:**\n"
    "<<Detailed description of the environment as it would appear from the subject’s first-person perspective. Anchor every observation in the exocentric evidence.>\n"
    "\n"
    "**Action Analysis:**\n"
    "<Describe the progression of actions and transitions observed throughout the sequence (or the current action for a single image) from the inferred subject’s viewpoint. Focus on how the subject's hands, any held objects, and the surrounding scene would appear, and how the scene evolves from moment to moment.>"
)

def cv2_to_pil(cv_img):
    """Convert OpenCV BGR image to PIL RGB Image"""
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def extract_frames_from_video(video_path, num_frames=4, limit_frames=None):
    """Extract evenly distributed frames from a video file."""
    if not os.path.exists(video_path):
        print(f"Warning: Video file not found: {video_path}")
        return []
        
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video file {video_path}")
            return []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"Warning: Video {video_path} has 0 frames")
            cap.release()
            return []
            
        # Limit total frames to first N frames if specified
        if limit_frames is not None and limit_frames > 0:
            total_frames = min(total_frames, limit_frames)
            
        if total_frames <= num_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                print(f"Warning: Could not read frame {idx} from {video_path}")
                
        cap.release()
        return frames
        
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        return []

def extract_caption_from_video_gemini(model, exo_path, ego_path=None, num_frames=4, limit_frames=None):
    """Generate caption using Google Gemini API."""
    print(f"Processing video: {exo_path}")
    
    # Extract frames
    exo_images = extract_frames_from_video(exo_path, num_frames, limit_frames=limit_frames)
    if not exo_images:
        print(f"Warning: No frames extracted from {exo_path}")
        return None
        
    ego_images = []
    if ego_path and os.path.exists(ego_path):
        ego_images = extract_frames_from_video(ego_path, num_frames, limit_frames=limit_frames)
    
    # Process and combine frames as PIL images for Gemini
    pil_images = []
    for i in range(len(exo_images)):
        exo_image = exo_images[i]
        
        if ego_images and i < len(ego_images):
            ego_image = ego_images[i]
            
            # Combine side-by-side
            exo_height, exo_width = exo_image.shape[:2]
            ego_height, ego_width = ego_image.shape[:2]
            
            if exo_height != ego_height:
                ego_image_resized = cv2.resize(ego_image, (int(ego_width * exo_height / ego_height), exo_height))
                combined_image = np.hstack([exo_image, ego_image_resized])
            else:
                combined_image = np.hstack([exo_image, ego_image])
            
            pil_images.append(cv2_to_pil(combined_image))
        else:
            pil_images.append(cv2_to_pil(exo_image))
            
    if not pil_images:
        return None
        
    try:
        print(f"Sending {len(pil_images)} frame sequence to Gemini API...")
        
        # User message instruction
        user_message = (
            f"Analyze this sequence of {len(pil_images)} video frames in chronological order. "
            + (
                "Each frame shows both an exocentric (third-person) and egocentric (first-person) view side by side. "
                "Provide the scene overview and frame-by-frame action analysis as specified in your instructions."
                if ego_images
                else "Only the exocentric (third-person) view is provided. "
                "Infer the first-person experience and deliver the scene overview and action analysis exactly as instructed."
            )
        )
        
        # Send contents (user message and image list) to Gemini
        contents = [user_message] + pil_images
        response = model.generate_content(contents)
        
        caption_text = response.text.strip()
        
        # Process caption text to be a single, continuous line (no newline characters)
        # Replacing all tabs, newlines, and multiple spaces with a single space
        single_line_caption = " ".join(caption_text.split())
        
        return single_line_caption
        
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None

def process_all_tasks(model, json_file_path, output_json_path, max_tasks=None, start_from=None, overwrite=False, num_frames=4, limit_frames=None):
    """Process tasks from JSON and update the JSON with new prompts."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file {json_file_path}: {e}")
        return

    dataset_keys = ["datasets", "train_datasets", "val_datasets", "test_datasets"]
    total_processed = 0
    total_successful = 0
    
    for key in dataset_keys:
        if key not in data:
            continue
            
        datasets = data[key]
        print(f"Processing {len(datasets)} tasks in section '{key}'")
        
        for i, dataset in enumerate(datasets):
            if start_from and total_processed < start_from:
                total_processed += 1
                continue
                
            if max_tasks and total_successful >= max_tasks:
                break
            
            exo_path = dataset.get("exo_path")
            ego_path = dataset.get("ego_prior_path")
            current_prompt = dataset.get("prompt", "")
            
            if not exo_path:
                print(f"Skipping task {i}: No exo_path found")
                continue
                
            if current_prompt and not overwrite and len(current_prompt) > 50:
                print(f"Skipping task {i}: Prompt already exists")
                continue
            
            # Resolve relative paths relative to json file location
            json_dir = os.path.dirname(os.path.abspath(json_file_path))
            
            if exo_path and not os.path.isabs(exo_path) and not os.path.exists(exo_path):
                alt_path = os.path.join(json_dir, exo_path)
                if os.path.exists(alt_path):
                    exo_path = alt_path
            
            if ego_path and not os.path.isabs(ego_path) and not os.path.exists(ego_path):
                alt_path = os.path.join(json_dir, ego_path)
                if os.path.exists(alt_path):
                    ego_path = alt_path
            
            # Generate caption using Gemini
            caption = extract_caption_from_video_gemini(model, exo_path, ego_path=ego_path, num_frames=num_frames, limit_frames=limit_frames)
            
            if caption:
                dataset["prompt"] = caption
                total_successful += 1
                print(f"✓ Updated prompt for {os.path.basename(exo_path)} (Single-line formatting applied)")
                
                # Intermediate save
                if total_successful % 5 == 0:
                    with open(output_json_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)
                        print(f"Progress saved to {output_json_path}")
            else:
                print(f"✗ Failed to generate caption for {os.path.basename(exo_path)}")
            
            total_processed += 1
            
        if max_tasks and total_successful >= max_tasks:
            break
            
    # Final save
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"\nProcessing complete. Updated {total_successful} tasks.")
    print(f"Saved result to {output_json_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract captions from exo_GT and ego_GT using Google Gemini API")
    parser.add_argument("--json_file", "-j", 
                        default="./datasets/egoexo4d_test1/meta_compiled.json",
                        help="Path to JSON file containing task paths")
    parser.add_argument("--output_json", "-o",
                        default="./datasets/egoexo4d_test1/meta_compiled.json",
                        help="Path to output updated JSON file")
    parser.add_argument("--max_tasks", "-m", type=int,
                        help="Maximum number of tasks to process")
    parser.add_argument("--start_from", "-s", type=int,
                        help="Task index to start from (for resuming)")
    parser.add_argument("--num_frames", "-n", type=int, default=4,
                        help="Number of frames to process per view (default: 4)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Reprocess tasks that already have captions")
    parser.add_argument("--gemini_model", default="gemini-2.5-flash",
                        help="Gemini model choice (e.g. gemini-2.5-flash, gemini-1.5-pro, etc.)")
    
    parser.add_argument("--limit_frames", type=int, default=None,
                        help="Limit video frame range to first N frames (e.g. 49)")
    
    args = parser.parse_args()
    
    # 1. Initialize Gemini API Client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        print("Please export it in your terminal: export GEMINI_API_KEY='your_api_key'")
        return
        
    genai.configure(api_key=api_key)
    
    # 2. Configure Model with system prompt instruction
    print(f"Initializing Gemini Model: {args.gemini_model}...")
    model = genai.GenerativeModel(
        model_name=args.gemini_model,
        system_instruction=system_prompt
    )
    
    if not os.path.exists(args.json_file):
        print(f"Error: JSON file not found: {args.json_file}")
        return
    
    # 3. Process all tasks
    process_all_tasks(
        model=model,
        json_file_path=args.json_file,
        output_json_path=args.output_json,
        max_tasks=args.max_tasks,
        start_from=args.start_from,
        overwrite=args.overwrite,
        num_frames=args.num_frames,
        limit_frames=args.limit_frames
    )

if __name__ == "__main__":
    main()
