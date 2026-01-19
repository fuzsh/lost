import json
import os
import sys

from src.utils import highlight_landmarks, read_steps_from_logs, verify_step_completion_and_map_subgoals, \
    generate_structured_output
from src.visualize import visualize_area
from src.data_loader import get_data_by_instruction

# Configuration
OUTPUT_DIR = 'annotation_data'
IMAGES_DIR = os.path.join(OUTPUT_DIR, 'images')

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

with open('results/failed_seen.json', 'r') as fs:
    seen = json.loads(fs.read())

with open('results/failed_unseen.json', 'r') as fu:
    unseen = json.loads(fu.read())

wrongly_predicated_instructions = {
    'seen': seen,
    'unseen': unseen,
}

# Collect all annotation data
annotation_data = {
    'metadata': {
        'generated_at': None,
        'total_instructions': 0,
        'splits': list(wrongly_predicated_instructions.keys())
    },
    'instructions': {}
}

from datetime import datetime

annotation_data['metadata']['generated_at'] = datetime.now().isoformat()

for wpi_category, wpi_items in wrongly_predicated_instructions.items():
    data_split = wpi_category

    with open(f'results/main_test_{data_split}.json', 'r') as f:
        test_results = json.loads(f.read())

    split_file = f"test_{data_split}.json"

    for instruction_id in wpi_items:
        print(f"Processing instruction {instruction_id} from {data_split}...")

        # Get the data, expanding the neighborhood by 20 degrees
        area_data = get_data_by_instruction(
            instruction_id,
            split_file,
            base_path='./data/map2seq/',
            neighbor_degrees=20
        )

        test_result = test_results.get(str(instruction_id), {})
        if not test_result:
            print(f"No test results for {instruction_id}")
            continue

        landmarks = [l['name'] for l in test_result.get('landmarks', [])]
        sub_goals = [sg['description'] for sg in test_result.get('sub_instructions', [])]

        # 1. Get all the landmarks and tag them in the text
        instruction_text = area_data['instruction_data']['instructions']
        text_highlighted = highlight_landmarks(instruction_text, landmarks)

        # 2. Generate visualization image
        visualize_image_name = f"visualize_{data_split}_{instruction_id}.png"
        visualize_image_path = os.path.join(IMAGES_DIR, visualize_image_name)
        visualize_area(
            area_data,
            landmarks=landmarks,
            output_name=visualize_image_path,
            focused_node_id=test_result['json']['predicated_path'],
        )

        # 3. Online image URL
        online_image_url = f"https://a3s.fi/swift/v1/AUTH_7cb48c21b32644c19940b85807a2f91a/GeoR2LLM_data_paper1/map2seq_processed_data/map_images_test_{data_split}/route_{instruction_id}.png"

        # 4. Get reasoning for each step alongside the sub-goals
        results = read_steps_from_logs(data_split, instruction_id)
        step_subgoal_mapping = verify_step_completion_and_map_subgoals(results, sub_goals)

        # 5. Generate structured output for display
        structured_output = generate_structured_output(step_subgoal_mapping, sub_goals)

        # Compile instruction data for annotation
        instruction_key = f"{data_split}_{instruction_id}"
        annotation_data['instructions'][instruction_key] = {
            'instruction_id': instruction_id,
            'data_split': data_split,
            'instruction_text': instruction_text,
            'instruction_highlighted': text_highlighted,
            'landmarks': test_result.get('landmarks', []),
            'sub_goals': test_result.get('sub_instructions', []),
            'step_reasoning': step_subgoal_mapping,
            'structured_output': structured_output,
            'execution_status': test_result.get('json', {}),
            'images': {
                'online_url': online_image_url,
                'visualize_path': f"images/{visualize_image_name}"
            },
            'route_info': {
                'osm_path': area_data['instruction_data']['route']['osm_path'],
                'predicted_path': test_result['json'].get('predicated_path', [])
            }
        }

        print(f"  - Generated visualization: {visualize_image_path}")
        print(f"  - Sub-goals: {len(sub_goals)}, Steps: {len(step_subgoal_mapping)}")

annotation_data['metadata']['total_instructions'] = len(annotation_data['instructions'])

# Save annotation data file
output_file = os.path.join(OUTPUT_DIR, 'annotation_data.json')
with open(output_file, 'w') as f:
    json.dump(annotation_data, f, indent=2)

print(f"\n{'=' * 50}")
print(f"Annotation data generated successfully!")
print(f"  - Output file: {output_file}")
print(f"  - Total instructions: {annotation_data['metadata']['total_instructions']}")
print(f"  - Images directory: {IMAGES_DIR}")
print(f"\nOpen annotation_tool.html in a browser to start annotating.")
