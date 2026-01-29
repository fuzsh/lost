"""
Upload the annotation_data to Hugging Face as a dataset.

Usage:
    pip install datasets huggingface_hub
    huggingface-cli login
    python dataset_lost.py
"""

import json
from datasets import Dataset, Features, Value, Sequence, Image

GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/fuzsh/lost/main/annotation_data/"
)

ANNOTATION_DATA_PATH = "annotation_data/annotation_data.json"
HF_DATASET_REPO = "fuzsh/lost"


def load_annotation_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def build_rows(data: dict) -> list[dict]:
    """Flatten each instruction entry into a row for the dataset."""
    rows = []
    for key, entry in data["instructions"].items():
        images = entry.get("images", {})
        exec_status = entry.get("execution_status", {})
        route_info = entry.get("route_info", {})

        # Use the GitHub-hosted URL for the local visualize image
        visualize_path = images.get("visualize_path", "")
        if visualize_path:
            visualize_url = GITHUB_RAW_BASE + visualize_path
        else:
            visualize_url = ""

        row = {
            "key": key,
            "instruction_id": entry["instruction_id"],
            "data_split": entry["data_split"],
            "instruction_text": entry["instruction_text"],
            "instruction_highlighted": entry["instruction_highlighted"],
            # Serialize nested structures as JSON strings
            "landmarks": json.dumps(entry.get("landmarks", [])),
            "sub_goals": json.dumps(entry.get("sub_goals", [])),
            "step_reasoning": json.dumps(entry.get("step_reasoning", [])),
            "structured_output": entry.get("structured_output", ""),
            # Execution status fields
            "execution_status": exec_status.get("status", ""),
            "execution_predicted_path": json.dumps(
                exec_status.get("predicated_path", [])
            ),
            "execution_current_instruction_index": exec_status.get(
                "current_instruction_index", -1
            ),
            "execution_num_retry": exec_status.get(
                "num_retry_current_instruction", 0
            ),
            "execution_current_instruction_status": exec_status.get(
                "current_instruction_status", ""
            ),
            "execution_current_heading": exec_status.get("current_heading", 0),
            "execution_error": exec_status.get("error", ""),
            # Images
            "online_image_url": images.get("online_url", ""),
            "visualize_image_url": visualize_url,
            # Route info
            "osm_path": json.dumps(route_info.get("osm_path", [])),
            "predicted_path": json.dumps(route_info.get("predicted_path", [])),
        }
        rows.append(row)
    return rows


def main():
    print(f"Loading annotation data from {ANNOTATION_DATA_PATH}...")
    data = load_annotation_data(ANNOTATION_DATA_PATH)
    print(
        f"Metadata: {data['metadata']['total_instructions']} instructions, "
        f"splits: {data['metadata']['splits']}"
    )

    rows = build_rows(data)
    print(f"Built {len(rows)} rows")

    dataset = Dataset.from_list(rows)
    print(dataset)
    print(f"\nPushing to Hugging Face Hub: {HF_DATASET_REPO}...")
    dataset.push_to_hub(HF_DATASET_REPO)
    print("Done!")


if __name__ == "__main__":
    main()
