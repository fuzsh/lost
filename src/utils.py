import json
import re


def highlight_landmarks(text, landmarks):
    """
    Wraps landmark mentions in the text with HTML <mark> tags.

    Args:
        text: Navigation instruction text
        landmarks: List of landmark strings to highlight

    Returns:
        Text with landmarks wrapped in <mark class="landmark">...</mark> tags
    """
    # Sort landmarks by length (descending) to handle overlapping cases
    # This prevents partial matches (e.g., "Art" matching before "Museum of African Art")
    sorted_landmarks = sorted(landmarks, key=len, reverse=True)

    result = text

    for landmark in sorted_landmarks:
        # Use word boundaries to match complete landmark phrases
        # Case-insensitive matching with re.IGNORECASE
        pattern = re.escape(landmark)
        replacement = f'<mark class="landmark" data-landmark="{landmark}">{landmark}</mark>'

        # Replace all occurrences of the landmark
        result = re.sub(
            pattern,
            replacement,
            result,
            flags=re.IGNORECASE
        )

    return result


def read_steps_from_logs(data_split, instruction_id):
    results = []

    for i in range(1, 11):
        try:
            with open(f"./results/main_results_{data_split}/step{i}_json.jsonl", 'r') as f:
                for line in f:
                    line = json.loads(line)
                    if line['key'] == str(instruction_id):
                        results.append(line)
                        continue
        except FileNotFoundError:
            continue

    return results

def extract_step_data(result):
    """Extract reasoning, status, and next_place from step result."""
    reasoning = ""
    status = None
    next_place = None

    candidates = result.get('response', {}).get('candidates', [])
    if candidates:
        content = candidates[0].get('content', {})
        parts = content.get('parts', [])
        for part in parts:
            if part.get('thought'):
                reasoning = part.get('text', '').strip()
            else:
                text = part.get('text', '')
                if text:
                    try:
                        json_data = json.loads(text)
                        status = json_data.get('subplan_status')
                        next_place = json_data.get('next_place')
                    except json.JSONDecodeError:
                        pass
    return reasoning, status, next_place

def verify_step_completion_and_map_subgoals(results, sub_goals):
    """
    Verify completion status of each step and map to sub-goals.

    Logic:
    - For completed steps: advance to next sub-goal for subsequent step
    - For incomplete steps: both current and next step stay with same sub-goal
    """
    step_subgoal_mapping = []
    current_subgoal_idx = 0

    for step_idx, result in enumerate(results):
        reasoning, status, next_place = extract_step_data(result)

        # Ensure we don't exceed sub_goals bounds
        if current_subgoal_idx >= len(sub_goals):
            current_subgoal_idx = len(sub_goals) - 1

        # Get the current sub-goal for this step
        current_subgoal = sub_goals[current_subgoal_idx] if sub_goals else "N/A"

        step_data = {
            'step_number': step_idx + 1,
            'sub_goal': current_subgoal,
            'sub_goal_index': current_subgoal_idx,
            'status': status,
            'reasoning': reasoning,
            'next_place': next_place,
            'is_completed': status == 'COMPLETED'
        }
        step_subgoal_mapping.append(step_data)

        # Update sub-goal index for next step based on completion status
        if status == 'COMPLETED':
            # Completed: advance to next sub-goal for subsequent step
            current_subgoal_idx += 1
        # If incomplete: current_subgoal_idx stays the same for next step

    return step_subgoal_mapping

def generate_structured_output(step_mapping, sub_goals):
    """Generate well-structured textual output from step-subgoal mapping."""
    output_lines = []
    # Sub-goals overview
    output_lines.append("SUB-GOALS:")
    output_lines.append("-" * 40)
    for idx, sg in enumerate(sub_goals):
        output_lines.append(f"  [{idx+1}] {sg}")
    output_lines.append("")

    # Step-by-step analysis
    output_lines.append("STEP-BY-STEP ANALYSIS:")
    output_lines.append("-" * 40)

    for step in step_mapping:
        status_marker = "✓" if step['is_completed'] else "○"
        output_lines.append(f"\nStep {step['step_number']}: [{status_marker}] {step['status'] or 'N/A'}")
        output_lines.append(f"  Associated Sub-Goal [{step['sub_goal_index']+1}]: {step['sub_goal']}")
        output_lines.append(f"  Target Node: {step['next_place'] or 'N/A'}")
        if step['reasoning']:
            # Clean and truncate reasoning for display
            clean_reasoning = step['reasoning'].replace('\\n', ' ').strip()
            if len(clean_reasoning) > 200:
                clean_reasoning = clean_reasoning[:200] + "..."
            output_lines.append(f"  Reasoning: {clean_reasoning}")

    return "\n".join(output_lines)