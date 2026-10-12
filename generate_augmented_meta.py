#!/usr/bin/env python3
"""
Generate 5 augmented variations of a dataset's metadata prompts using OpenAI (GPT) or Google Gemini API.
Ensures semantic consistency and physical compatibility of objects and backgrounds.
"""

import os
import json
import argparse
import copy
from pathlib import Path

def call_openai(api_key, system_instruction, prompt_to_send, model_name="gpt-4o"):
    from openai import OpenAI
    print(f"Calling OpenAI API ({model_name}) to generate 5 variations...")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt_to_send}
        ],
        response_format={"type": "json_object"},
        temperature=0.7
    )
    result_text = response.choices[0].message.content.strip()
    return json.loads(result_text)

def call_gemini(api_key, system_instruction, prompt_to_send, model_name="gemini-2.5-flash"):
    import google.generativeai as genai
    print(f"Calling Gemini API ({model_name}) to generate 5 variations...")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_instruction
    )
    response = model.generate_content(
        prompt_to_send,
        generation_config={"response_mime_type": "application/json"}
    )
    result_text = response.text.strip()
    return json.loads(result_text)

def load_key_from_file_or_env(file_name, env_name):
    key = None
    possible_paths = [
        file_name,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            key = line
                            print(f"Loaded {env_name} from file: {path}")
                            break
                if key:
                    break
            except Exception:
                pass
    if not key:
        key = os.environ.get(env_name, "")
    if key:
        key = key.strip().strip('"').strip("'")
    return key

def main():
    parser = argparse.ArgumentParser(description="Augment metadata prompts using GPT or Gemini API")
    parser.add_argument("--json_file", "-j", required=True, help="Path to input meta.json file")
    parser.add_argument("--model", "-m", default=None, help="Model name (e.g. gpt-4o or gemini-2.5-flash)")
    args = parser.parse_args()

    # 1. Check API Keys
    openai_key = load_key_from_file_or_env("openai_api_key.txt", "OPENAI_API_KEY")
    gemini_key = load_key_from_file_or_env("api_key.txt", "GEMINI_API_KEY")

    if not openai_key and not gemini_key:
        print("Error: Neither OPENAI_API_KEY nor GEMINI_API_KEY is provided.")
        print("Please save your key in 'openai_api_key.txt' or 'api_key.txt', or export environment variables.")
        return

    # 2. Check and load input file
    input_path = Path(args.json_file).resolve()
    if not input_path.exists():
        print(f"Error: Input file not found at {input_path}")
        return

    print(f"Loading metadata from: {input_path}")
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    dataset_keys = ["test_datasets", "train_datasets", "val_datasets", "datasets"]
    found_key = None
    for key in dataset_keys:
        if key in data and len(data[key]) > 0:
            found_key = key
            break

    if not found_key:
        print("Error: Could not find any valid dataset section with entries (e.g. 'test_datasets') in JSON.")
        return

    base_entry = data[found_key][0]
    original_prompt = base_entry.get("prompt", "")
    if not original_prompt:
        print("Error: The first dataset entry does not contain a 'prompt' field.")
        return

    print(f"Original Prompt found (from {found_key}[0]):\n{original_prompt[:250]}...\n")

    # 3. Call LLM to generate variations
    system_instruction = (
        "You are an expert video caption augmentation assistant for a video diffusion model. "
        "Your task is to take an original description prompt containing [Exo view] and [Ego view] sections, "
        "and generate exactly 5 alternative prompt variations for video data augmentation.\n\n"
        "Rules for each variation:\n"
        "1. Identify the main interactive objects (e.g., food, plates/bowls, utensils, drinking containers, countertops) "
        "and background elements (refrigerator, cabinets, walls, floors).\n"
        "2. Replace these objects with new ones that have highly similar sizes, physical shapes, and hand-grasp poses "
        "to ensure the original hand actions and depth map structure remain physically plausible. "
        "For example, replace 'fork' with 'spoon' or 'chopsticks', 'spaghetti on a plate' with 'fried rice on a plate' or 'salad in a bowl', "
        "'glass cup' with 'ceramic mug' or 'plastic tumbler', 'granite countertop' with 'wooden countertop'.\n"
        "3. Modify the background elements naturally (e.g., change refrigerator colors/materials, cabinet wood type, wall color/pattern).\n"
        "4. Keep the sentence structures, action descriptions, and trajectories identical to the original prompt, "
        "only substituting the semantic object/background nouns and adjectives.\n"
        "5. You MUST maintain perfect semantic consistency between the [Exo view] and [Ego view] sections of the same variation. "
        "If you change a fork to a spoon in the [Exo view], it must also be a spoon in the [Ego view].\n"
        "6. Do not add any new actions or change the timing of events.\n\n"
        "Provide your output strictly in JSON format matching this schema:\n"
        "{\n"
        "  \"prompts\": [\n"
        "    \"<Variation 1 prompt string with [Exo view] and [Ego view]>\",\n"
        "    \"<Variation 2 prompt string>\",\n"
        "    \"<Variation 3 prompt string>\",\n"
        "    \"<Variation 4 prompt string>\",\n"
        "    \"<Variation 5 prompt string>\"\n"
        "  ]\n"
        "}"
    )

    prompt_to_send = f"Original Prompt:\n{original_prompt}\n\nPlease generate the 5 augmented variations now."
    
    response_data = None
    # Determine model and call
    if args.model:
        model_name = args.model
        if "gpt" in model_name.lower():
            if not openai_key:
                print(f"Error: Specified model {model_name} requires OPENAI_API_KEY to be set.")
                return
            response_data = call_openai(openai_key, system_instruction, prompt_to_send, model_name)
        else:
            if not gemini_key:
                print(f"Error: Specified model {model_name} requires GEMINI_API_KEY to be set.")
                return
            response_data = call_gemini(gemini_key, system_instruction, prompt_to_send, model_name)
    else:
        # Auto fallback: OpenAI first, then Gemini
        if openai_key:
            try:
                response_data = call_openai(openai_key, system_instruction, prompt_to_send, "gpt-4o")
            except Exception as e:
                print(f"OpenAI call failed ({e}). Trying Gemini...")
                if gemini_key:
                    response_data = call_gemini(gemini_key, system_instruction, prompt_to_send, "gemini-2.5-flash")
                else:
                    print("OpenAI call failed and GEMINI_API_KEY is not set.")
                    return
        elif gemini_key:
            response_data = call_gemini(gemini_key, system_instruction, prompt_to_send, "gemini-2.5-flash")

    if not response_data or "prompts" not in response_data:
        print("Error: Could not retrieve prompts from LLM response.")
        return

    augmented_prompts = response_data["prompts"]
    if len(augmented_prompts) != 5:
        print(f"Warning: Expected 5 prompts, but received {len(augmented_prompts)} prompts. Proceeding with received list.")

    # 4. Create new metadata entries
    new_test_datasets = []
    # Index 0: Original Entry
    new_test_datasets.append(copy.deepcopy(base_entry))

    for i, p_text in enumerate(augmented_prompts):
        entry = copy.deepcopy(base_entry)
        entry['prompt'] = p_text
        new_test_datasets.append(entry)

    data[found_key] = new_test_datasets

    # Save to meta_augmented_5.json in the same directory as original
    output_path = input_path.parent / "meta_aug_5.json"
    print(f"Saving augmented metadata with {len(new_test_datasets)} entries to: {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print("✓ Successfully generated augmented meta file!")
    except Exception as e:
        print(f"Error saving output JSON file: {e}")

if __name__ == "__main__":
    main()
