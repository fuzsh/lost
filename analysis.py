import json
from collections import Counter

def analyze_explorer_annotations(path="annotation_data/explorer_annotations.json"):
    with open(path, "r") as f:
        data = json.load(f)

    total_entries = len(data)

    # Counters for each category
    categories = {"agent": Counter(), "execution": Counter(), "linguistic": Counter()}

    for entry in data:
        responses = entry.get("responses", {})
        for category in categories:
            annotations = responses.get(category, [])
            for annotation in annotations:
                values = annotation.get("value", [])
                if isinstance(values, list):
                    for v in values:
                        categories[category][v] += 1

    # Print results
    print(f"Total annotated entries: {total_entries}\n")
    print("=" * 65)

    for category, counter in categories.items():
        cat_total = sum(counter.values())
        print(f"\n{'[' + category.upper() + ']':^65}")
        print(f"  Total labels: {cat_total}")
        print(f"  {'Subcategory':<35} {'Count':>6} {'Rate':>8}")
        print(f"  {'-'*35} {'-'*6} {'-'*8}")
        for subcategory, count in counter.most_common():
            rate = count / total_entries * 100
            print(f"  {subcategory:<35} {count:>6} {rate:>7.1f}%")
        print()

    # Summary
    print("=" * 65)
    grand_total = sum(sum(c.values()) for c in categories.values())
    print(f"Grand total labels: {grand_total}")
    print(f"Average labels per entry: {grand_total / total_entries:.2f}")


if __name__ == "__main__":
    analyze_explorer_annotations()
