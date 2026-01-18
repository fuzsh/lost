import json
import sys

from src.utils import highlight_landmarks
from src.visualize import visualize_area
from src.data_loader import get_data_by_instruction


data_split = "unseen"
with open(f'results/main_test_{data_split}.json', 'r') as f:
    test_results = json.loads(f.read())

instruction_id = 6018
split_file = f"test_{data_split}.json"

# Get the data, expanding the neighborhood by 20 degrees
area_data = get_data_by_instruction(
    instruction_id,
    split_file,
    base_path='/Users/fuzzy/Projects/sisu/data/map2seq/',
    neighbor_degrees=20
)

test_result = test_results.get(str(instruction_id), {})
if not test_result:
    print("No test results")
    sys.exit(1)

landmarks = [l['name'] for l in test_result.get('landmarks', [])]
sub_goals = [sg['description'] for sg in test_result.get('sub_instructions', [])]

# 1. Get all the landmarks and tag them in the text --> separate them
text_highlighted = highlight_landmarks(area_data['instruction_data']['instructions'], landmarks)
print(text_highlighted)

# 2. Get all the landmarks and annotate them on the image for visualization
visualize_area(
    area_data,
    landmarks=landmarks,
    output_name=f"test_{data_split}_{instruction_id}.png",
    focused_node_id=test_result['json']['predicated_path'],
)

# 4. Find the associated link to each image from what subra generated
image_url = f"https://a3s.fi/swift/v1/AUTH_7cb48c21b32644c19940b85807a2f91a/GeoR2LLM_data_paper1/map2seq_processed_data/map_images_test_{data_split}/route_{instruction_id}.png"

# 5. Get reasoning for each step alongside the sub-goals

results = []

for i in range(1, 11):
    with open(f"./paper_results/main_results_{data_split}/step{i}_json.jsonl", 'r') as f:
        for line in f:
            line = json.loads(line)
            if line['key'] == str(instruction_id):
                results.append(line)
                continue

for idx, result in enumerate(results):
    candidate_text = ""
    for candidate in result['response']['candidates']:
        for part in candidate["content"]["parts"]:
            candidate_text += f"{part['text']}\n"

    print(f"Step {idx+1}: {candidate_text}")
