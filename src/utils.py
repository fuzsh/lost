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