#!/usr/bin/env python3
"""
Inter-Annotator Agreement Calculator for Navigation Error Annotation

This script calculates inter-annotator agreement at three hierarchical levels:
- Dimension (top-level): L, T, A, E error types
- Category (mid-level): e.g., L1, L2, T1, T2, A1, etc.
- Subcategory (leaf-level): e.g., L1a, L1b, T1a, T1b, etc.

Metrics computed:
- Pairwise Cohen's Kappa between annotator pairs
- Agreement percentages and disagreement analysis

Usage:
    python inter-annotator.py                  # Run with real annotations

Reference:
    Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement
    for categorical data. Biometrics, 33(1), 159-174.
"""

import json
import glob
import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from itertools import combinations
import numpy as np

# Define the error taxonomy
TAXONOMY = {
    'L': {  # Linguistic Errors
        'L1': ['L1a', 'L1b', 'L1c'],  # Semantic errors
        'L2': ['L2a', 'L2b', 'L2c'],  # Syntactic errors
    },
    'T': {  # Trajectory Errors
        'T1': ['T1a', 'T1b', 'T1c'],  # Route selection
        'T2': ['T2a', 'T2b', 'T2c'],  # Path following
    },
    'A': {  # Alignment Errors
        'A1': ['A1a', 'A1b'],  # Spatial reference
        'A2': ['A2a', 'A2b'],  # Temporal reference
        'A3': ['A3a', 'A3b'],  # Landmark reference
        'A4': ['A4a', 'A4b'],  # Direction errors
        'A5': ['A5a', 'A5b', 'A5c'],  # Hallucination
    },
    'E': {  # Execution Errors
        'E1': ['E1a', 'E1b'],  # Action errors
        'E2': ['E2a', 'E2b'],  # Completion errors
    },
}


def load_annotation_files(annotations_dir: str) -> Dict[str, Dict]:
    """
    Load all annotation files from the annotations directory.

    Returns:
        Dictionary mapping annotator names to their annotation data
    """
    annotators = {}
    pattern = os.path.join(annotations_dir, "annotations_*.json")

    for filepath in glob.glob(pattern):
        with open(filepath, 'r') as f:
            data = json.load(f)
            annotator_name = data['session']['annotatorName']
            annotators[annotator_name] = data
            print(f"Loaded annotations from: {annotator_name} ({filepath})")

    return annotators


def extract_annotations_per_instance(annotators: Dict[str, Dict]) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Extract annotations for each instance from all annotators.

    Returns:
        Dictionary: instance_id -> annotator_name -> list of annotations
    """
    instances = defaultdict(lambda: defaultdict(list))

    for annotator_name, data in annotators.items():
        for instance_id, instance_data in data['annotations'].items():
            if instance_data['status'] == 'completed' and instance_data['annotations']:
                instances[instance_id][annotator_name] = instance_data['annotations']

    return instances


def get_all_labels(instances: Dict, level: str) -> Set[str]:
    """Get all unique labels at a given hierarchical level."""
    labels = set()
    for instance_id, annotator_data in instances.items():
        for annotator, annotations in annotator_data.items():
            for ann in annotations:
                if level == 'dimension':
                    labels.add(ann['dimension'])
                elif level == 'category':
                    labels.add(ann['category'])
                elif level == 'subcategory':
                    labels.add(ann['subcategory'])
    return labels


def get_labels_for_instance(annotations: List[Dict], level: str) -> Set[str]:
    """Extract labels at a given level from a list of annotations."""
    labels = set()
    for ann in annotations:
        if level == 'dimension':
            labels.add(ann['dimension'])
        elif level == 'category':
            labels.add(ann['category'])
        elif level == 'subcategory':
            labels.add(ann['subcategory'])
    return labels


def compute_pairwise_cohens_kappa(instances: Dict[str, Dict[str, List[Dict]]],
                                  annotator1: str,
                                  annotator2: str,
                                  level: str) -> Tuple[float, Dict]:
    """
    Compute Cohen's Kappa between two annotators at a given level.

    For multi-label annotations, we use a set-based approach:
    - Each unique combination of labels is treated as a distinct category
    - Agreement occurs when both annotators assign the exact same set of labels
    """
    common_instances = []
    for instance_id, annotator_data in instances.items():
        if annotator1 in annotator_data and annotator2 in annotator_data:
            common_instances.append(instance_id)

    if not common_instances:
        return float('nan'), {'error': 'No common instances'}

    # Collect all unique label sets (treating each set as a category)
    all_label_sets = set()
    instance_label_sets = {}

    for instance_id in common_instances:
        labels1 = frozenset(get_labels_for_instance(instances[instance_id][annotator1], level))
        labels2 = frozenset(get_labels_for_instance(instances[instance_id][annotator2], level))
        all_label_sets.add(labels1)
        all_label_sets.add(labels2)
        instance_label_sets[instance_id] = (labels1, labels2)

    # Convert to sorted list for consistent indexing
    all_categories = sorted([tuple(sorted(ls)) for ls in all_label_sets])
    category_to_idx = {cat: idx for idx, cat in enumerate(all_categories)}
    n_categories = len(all_categories)

    if n_categories == 0:
        return float('nan'), {'error': 'No categories found'}

    # Build contingency matrix
    contingency = np.zeros((n_categories, n_categories))

    total_agreements = 0
    total_comparisons = 0

    for instance_id in common_instances:
        labels1, labels2 = instance_label_sets[instance_id]

        # For exact set-based comparison
        if labels1 == labels2:
            total_agreements += 1
        total_comparisons += 1

        # Update contingency matrix using the full label set as the category
        cat1 = tuple(sorted(labels1))
        cat2 = tuple(sorted(labels2))
        if cat1 in category_to_idx and cat2 in category_to_idx:
            contingency[category_to_idx[cat1], category_to_idx[cat2]] += 1

    # Compute Cohen's Kappa
    n = np.sum(contingency)
    if n == 0:
        return float('nan'), {'error': 'Empty contingency matrix'}

    # Observed agreement (diagonal sum)
    p_o = np.trace(contingency) / n

    # Expected agreement
    row_sums = np.sum(contingency, axis=1)
    col_sums = np.sum(contingency, axis=0)
    p_e = np.sum(row_sums * col_sums) / (n * n)

    # Cohen's Kappa
    if 1 - p_e > 0:
        kappa = (p_o - p_e) / (1 - p_e)
    else:
        kappa = 1.0 if p_o == 1.0 else 0.0

    # Get individual labels for reporting
    individual_labels = sorted(get_all_labels(instances, level))

    stats = {
        'n_instances': len(common_instances),
        'n_categories': n_categories,
        'categories': [list(cat) for cat in all_categories],
        'individual_labels': individual_labels,
        'observed_agreement': p_o,
        'expected_agreement': p_e,
        'exact_set_agreement': total_agreements / total_comparisons if total_comparisons > 0 else 0
    }

    return float(kappa), stats


def compute_per_annotation_kappa(instances: Dict[str, Dict[str, List[Dict]]],
                                 annotator1: str,
                                 annotator2: str,
                                 level: str) -> Tuple[float, Dict]:
    """
    Compute Cohen's Kappa between two annotators at a given level using
    per-annotation comparison (rather than set-based).

    This approach pairs up individual annotations from each annotator and
    compares them one-to-one. For multi-label instances, annotations are
    matched by their dimension (T, A, L, E) to ensure comparable pairs.

    This is more appropriate when:
    - Each annotator assigns multiple independent error types per instance
    - You want to measure agreement on individual error identification
    """
    common_instances = []
    for instance_id, annotator_data in instances.items():
        if annotator1 in annotator_data and annotator2 in annotator_data:
            common_instances.append(instance_id)

    if not common_instances:
        return float('nan'), {'error': 'No common instances'}

    # Collect all annotation pairs by matching on dimension
    all_labels = set()
    annotation_pairs = []  # List of (label1, label2) tuples

    for instance_id in common_instances:
        anns1 = instances[instance_id][annotator1]
        anns2 = instances[instance_id][annotator2]

        # Group annotations by dimension for matching
        anns1_by_dim = {}
        anns2_by_dim = {}

        for ann in anns1:
            dim = ann['dimension']
            if dim not in anns1_by_dim:
                anns1_by_dim[dim] = []
            anns1_by_dim[dim].append(ann)

        for ann in anns2:
            dim = ann['dimension']
            if dim not in anns2_by_dim:
                anns2_by_dim[dim] = []
            anns2_by_dim[dim].append(ann)

        # Match annotations within each dimension
        all_dims = set(anns1_by_dim.keys()) | set(anns2_by_dim.keys())

        for dim in all_dims:
            list1 = anns1_by_dim.get(dim, [])
            list2 = anns2_by_dim.get(dim, [])

            # Extract labels at the specified level
            if level == 'dimension':
                labels1 = [ann['dimension'] for ann in list1]
                labels2 = [ann['dimension'] for ann in list2]
            elif level == 'category':
                labels1 = [ann['category'] for ann in list1]
                labels2 = [ann['category'] for ann in list2]
            elif level == 'subcategory':
                labels1 = [ann['subcategory'] for ann in list1]
                labels2 = [ann['subcategory'] for ann in list2]

            # Pair up annotations (use None for missing pairs)
            max_len = max(len(labels1), len(labels2))
            for i in range(max_len):
                l1 = labels1[i] if i < len(labels1) else None
                l2 = labels2[i] if i < len(labels2) else None
                annotation_pairs.append((l1, l2))
                if l1:
                    all_labels.add(l1)
                if l2:
                    all_labels.add(l2)

    if not annotation_pairs:
        return float('nan'), {'error': 'No annotation pairs found'}

    # Add None as a category for missing annotations
    all_labels.add(None)
    all_labels_list = sorted([l for l in all_labels if l is not None]) + [None]
    label_to_idx = {label: idx for idx, label in enumerate(all_labels_list)}
    n_categories = len(all_labels_list)

    # Build contingency matrix
    contingency = np.zeros((n_categories, n_categories))

    for l1, l2 in annotation_pairs:
        contingency[label_to_idx[l1], label_to_idx[l2]] += 1

    # Compute Cohen's Kappa
    n = np.sum(contingency)
    if n == 0:
        return float('nan'), {'error': 'Empty contingency matrix'}

    # Observed agreement (diagonal sum)
    p_o = np.trace(contingency) / n

    # Expected agreement
    row_sums = np.sum(contingency, axis=1)
    col_sums = np.sum(contingency, axis=0)
    p_e = np.sum(row_sums * col_sums) / (n * n)

    # Cohen's Kappa
    if 1 - p_e > 0:
        kappa = (p_o - p_e) / (1 - p_e)
    else:
        kappa = 1.0 if p_o == 1.0 else 0.0

    # Count agreements
    agreements = sum(1 for l1, l2 in annotation_pairs if l1 == l2)

    stats = {
        'n_annotation_pairs': len(annotation_pairs),
        'n_categories': n_categories - 1,  # Exclude None for reporting
        'categories': [l for l in all_labels_list if l is not None],
        'observed_agreement': p_o,
        'expected_agreement': p_e,
        'raw_agreement': agreements / len(annotation_pairs) if annotation_pairs else 0,
        'agreements': agreements,
        'total_pairs': len(annotation_pairs)
    }

    return float(kappa), stats


def analyze_disagreements(instances: Dict[str, Dict[str, List[Dict]]],
                          annotator_names: List[str]) -> Dict:
    """
    Analyze disagreements and compute resolution statistics.
    """
    results = {
        'complete_agreement': [],
        'leaf_level_disagreement': [],
        'mid_level_disagreement': [],
        'top_level_disagreement': []
    }

    for instance_id, annotator_data in instances.items():
        if len(annotator_data) < 2:
            continue

        # Get annotations from all annotators for this instance
        all_dimensions = []
        all_categories = []
        all_subcategories = []

        for annotator in annotator_data:
            dims = get_labels_for_instance(annotator_data[annotator], 'dimension')
            cats = get_labels_for_instance(annotator_data[annotator], 'category')
            subs = get_labels_for_instance(annotator_data[annotator], 'subcategory')
            all_dimensions.append(dims)
            all_categories.append(cats)
            all_subcategories.append(subs)

        # Check agreement levels
        dims_agree = len(set(frozenset(d) for d in all_dimensions)) == 1
        cats_agree = len(set(frozenset(c) for c in all_categories)) == 1
        subs_agree = len(set(frozenset(s) for s in all_subcategories)) == 1

        if subs_agree and cats_agree and dims_agree:
            results['complete_agreement'].append(instance_id)
        elif cats_agree and dims_agree:
            # Disagreement only at leaf level
            results['leaf_level_disagreement'].append(instance_id)
        elif dims_agree:
            # Disagreement at mid-level (category)
            results['mid_level_disagreement'].append(instance_id)
        else:
            # Disagreement at top level (dimension)
            results['top_level_disagreement'].append(instance_id)

    return results


def get_deepest_common_ancestor(subcats: List[str]) -> str:
    """
    Find the deepest common ancestor for a list of subcategories.

    Hierarchy:
    - Subcategory: T1a, T1b, etc. -> Parent is T1
    - Category: T1, T2, etc. -> Parent is T
    - Dimension: T, A, L, E
    """
    if not subcats:
        return ""

    # Extract categories (remove last character if it's a letter after number)
    categories = set()
    for sub in subcats:
        # Subcategory format: T1a -> Category is T1
        if len(sub) >= 2:
            # Find where the category ends (number followed by letter)
            cat = ""
            for i, c in enumerate(sub):
                if c.isalpha() and i > 0 and sub[i - 1].isdigit():
                    break
                cat += c
            categories.add(cat if cat else sub[:2])

    if len(categories) == 1:
        return list(categories)[0]

    # Extract dimensions
    dimensions = set(cat[0] for cat in categories if cat)

    if len(dimensions) == 1:
        return list(dimensions)[0]

    return "MULTI"  # Multiple top-level dimensions disagree


def resolve_disagreements(disagreement_analysis: Dict,
                          instances: Dict[str, Dict[str, List[Dict]]]) -> Dict:
    """
    Apply disagreement resolution strategy and return statistics.
    """
    resolved = {
        'complete_agreement_count': len(disagreement_analysis['complete_agreement']),
        'ancestor_resolved_count': len(disagreement_analysis['leaf_level_disagreement']),
        'adjudicator_required_count': len(disagreement_analysis['mid_level_disagreement']) +
                                      len(disagreement_analysis['top_level_disagreement']),
        'resolved_labels': [],
        'adjudicator_cases': []
    }

    # For leaf-level disagreements, find deepest common ancestor
    for instance_id in disagreement_analysis['leaf_level_disagreement']:
        all_subcats = []
        for annotator, annotations in instances[instance_id].items():
            for ann in annotations:
                all_subcats.append(ann['subcategory'])

        ancestor = get_deepest_common_ancestor(all_subcats)
        resolved['resolved_labels'].append({
            'instance_id': instance_id,
            'original_subcategories': all_subcats,
            'resolved_to': ancestor
        })

    # Cases requiring adjudicator
    for instance_id in (disagreement_analysis['mid_level_disagreement'] +
                        disagreement_analysis['top_level_disagreement']):
        resolved['adjudicator_cases'].append(instance_id)

    return resolved


def get_detailed_disagreements(instances: Dict[str, Dict[str, List[Dict]]],
                               disagreement_analysis: Dict) -> Dict:
    """
    Generate detailed disagreement information showing what each annotator said.

    Returns a dictionary with detailed conflict information for each instance.
    """
    details = {
        'complete_agreement': [],
        'leaf_level': [],
        'mid_level': [],
        'top_level': []
    }

    # Complete agreement cases
    for instance_id in disagreement_analysis['complete_agreement']:
        annotator_data = instances[instance_id]
        case = {
            'instance_id': instance_id,
            'annotators': {}
        }
        for annotator, annotations in annotator_data.items():
            case['annotators'][annotator] = [
                f"{ann['subcategory']} ({ann['subcategoryName']})"
                for ann in annotations
            ]
        details['complete_agreement'].append(case)

    # Leaf-level disagreements (same category, different subcategory)
    for instance_id in disagreement_analysis['leaf_level_disagreement']:
        annotator_data = instances[instance_id]
        case = {
            'instance_id': instance_id,
            'disagreement_type': 'Subcategory disagreement (same category)',
            'annotators': {},
            'conflicts': []
        }

        # Collect annotations by dimension
        annotations_by_dim = {}
        for annotator, annotations in annotator_data.items():
            case['annotators'][annotator] = []
            for ann in annotations:
                case['annotators'][annotator].append({
                    'dimension': ann['dimension'],
                    'category': ann['category'],
                    'subcategory': ann['subcategory'],
                    'name': ann.get('subcategoryName', '')
                })
                dim = ann['dimension']
                if dim not in annotations_by_dim:
                    annotations_by_dim[dim] = {}
                if annotator not in annotations_by_dim[dim]:
                    annotations_by_dim[dim][annotator] = []
                annotations_by_dim[dim][annotator].append(ann)

        # Find specific conflicts
        for dim, ann_by_annotator in annotations_by_dim.items():
            annotators = list(ann_by_annotator.keys())
            if len(annotators) >= 2:
                for i, ann1_name in enumerate(annotators):
                    for ann2_name in annotators[i + 1:]:
                        anns1 = ann_by_annotator[ann1_name]
                        anns2 = ann_by_annotator[ann2_name]
                        for a1 in anns1:
                            for a2 in anns2:
                                if a1['subcategory'] != a2['subcategory']:
                                    case['conflicts'].append({
                                        'dimension': dim,
                                        ann1_name: f"{a1['category']}/{a1['subcategory']}",
                                        ann2_name: f"{a2['category']}/{a2['subcategory']}",
                                        'conflict_level': 'subcategory' if a1['category'] == a2[
                                            'category'] else 'category'
                                    })

        details['leaf_level'].append(case)

    # Mid-level disagreements (same dimension, different category)
    for instance_id in disagreement_analysis['mid_level_disagreement']:
        annotator_data = instances[instance_id]
        case = {
            'instance_id': instance_id,
            'disagreement_type': 'Category disagreement (same dimension)',
            'annotators': {},
            'conflicts': []
        }

        annotations_by_dim = {}
        for annotator, annotations in annotator_data.items():
            case['annotators'][annotator] = []
            for ann in annotations:
                case['annotators'][annotator].append({
                    'dimension': ann['dimension'],
                    'category': ann['category'],
                    'subcategory': ann['subcategory'],
                    'name': ann.get('subcategoryName', '')
                })
                dim = ann['dimension']
                if dim not in annotations_by_dim:
                    annotations_by_dim[dim] = {}
                if annotator not in annotations_by_dim[dim]:
                    annotations_by_dim[dim][annotator] = []
                annotations_by_dim[dim][annotator].append(ann)

        # Find specific conflicts
        for dim, ann_by_annotator in annotations_by_dim.items():
            annotators = list(ann_by_annotator.keys())
            if len(annotators) >= 2:
                for i, ann1_name in enumerate(annotators):
                    for ann2_name in annotators[i + 1:]:
                        anns1 = ann_by_annotator[ann1_name]
                        anns2 = ann_by_annotator[ann2_name]
                        for a1 in anns1:
                            for a2 in anns2:
                                if a1['category'] != a2['category']:
                                    case['conflicts'].append({
                                        'dimension': dim,
                                        ann1_name: f"{a1['category']}/{a1['subcategory']}",
                                        ann2_name: f"{a2['category']}/{a2['subcategory']}",
                                        'conflict_level': 'category'
                                    })

        details['mid_level'].append(case)

    # Top-level disagreements (different dimensions)
    for instance_id in disagreement_analysis['top_level_disagreement']:
        annotator_data = instances[instance_id]
        case = {
            'instance_id': instance_id,
            'disagreement_type': 'Dimension disagreement',
            'annotators': {},
            'conflicts': []
        }

        dims_by_annotator = {}
        for annotator, annotations in annotator_data.items():
            case['annotators'][annotator] = []
            dims_by_annotator[annotator] = set()
            for ann in annotations:
                case['annotators'][annotator].append({
                    'dimension': ann['dimension'],
                    'category': ann['category'],
                    'subcategory': ann['subcategory'],
                    'name': ann.get('subcategoryName', '')
                })
                dims_by_annotator[annotator].add(ann['dimension'])

        # Find dimension-level conflicts
        annotators = list(dims_by_annotator.keys())
        if len(annotators) >= 2:
            for i, ann1_name in enumerate(annotators):
                for ann2_name in annotators[i + 1:]:
                    dims1 = dims_by_annotator[ann1_name]
                    dims2 = dims_by_annotator[ann2_name]
                    only_in_1 = dims1 - dims2
                    only_in_2 = dims2 - dims1
                    if only_in_1 or only_in_2:
                        case['conflicts'].append({
                            ann1_name: sorted(dims1),
                            ann2_name: sorted(dims2),
                            f'only_{ann1_name}': sorted(only_in_1),
                            f'only_{ann2_name}': sorted(only_in_2),
                            'conflict_level': 'dimension'
                        })

        details['top_level'].append(case)

    return details


def interpret_kappa(kappa: float) -> str:
    """
    Interpret Kappa value according to Landis & Koch (1977).
    """
    if np.isnan(kappa):
        return "N/A"
    elif kappa < 0:
        return "Poor (< 0)"
    elif kappa < 0.20:
        return "Slight (0.00-0.20)"
    elif kappa < 0.40:
        return "Fair (0.21-0.40)"
    elif kappa < 0.60:
        return "Moderate (0.41-0.60)"
    elif kappa < 0.80:
        return "Substantial (0.61-0.80)"
    else:
        return "Almost Perfect (0.81-1.00)"


def main():
    """Main function to compute and report inter-annotator agreement."""
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    annotations_dir = os.path.join(script_dir, "annotations")

    print("=" * 70)
    print("Inter-Annotator Agreement Analysis")
    print("=" * 70)
    print()

    # Load real annotation files
    print("Loading annotation files...")
    annotators = load_annotation_files(annotations_dir)

    if not annotators:
        print("No annotation files found!")
        print(f"Looking in: {annotations_dir}")
        print("Expected file pattern: annotations_*.json")
        return

    annotator_names = list(annotators.keys())
    print(f"\nFound {len(annotators)} annotator(s): {', '.join(annotator_names)}")

    # Extract annotations per instance
    print("\nExtracting annotations...")
    instances = extract_annotations_per_instance(annotators)
    print(f"Found {len(instances)} annotated instances")

    # Count instances by number of annotators
    annotator_counts = defaultdict(int)
    for instance_id, annotator_data in instances.items():
        annotator_counts[len(annotator_data)] += 1

    print("\nInstances by number of annotators:")
    for n_ann, count in sorted(annotator_counts.items()):
        print(f"  {n_ann} annotator(s): {count} instances")

    # Store results
    results = {
        'per_annotation_kappa': {},
        'disagreement_analysis': None,
        'resolution': None,
        'detailed_disagreements': None
    }

    # Compute pairwise Cohen's Kappa (per-annotation) if multiple annotators
    if len(annotator_names) >= 2:
        # Per-annotation kappa (matches individual annotations within dimensions)
        print("\n" + "=" * 70)
        print("Pairwise Cohen's Kappa Analysis (Per-Annotation)")
        print("=" * 70)
        print("Note: Per-annotation approach compares individual annotations,")
        print("matching by dimension. More appropriate for multi-label instances.")

        results['per_annotation_kappa'] = {}

        for ann1, ann2 in combinations(annotator_names, 2):
            print(f"\n{ann1} vs {ann2}:")
            pair_key = f"{ann1}_vs_{ann2}"
            results['per_annotation_kappa'][pair_key] = {}

            for level in ['dimension', 'category', 'subcategory']:
                kappa, stats = compute_per_annotation_kappa(instances, ann1, ann2, level)
                results['per_annotation_kappa'][pair_key][level] = {'kappa': kappa, 'stats': stats}

                agreement_pct = stats.get('raw_agreement', 0) * 100
                n_pairs = stats.get('n_annotation_pairs', 0)
                print(f"  {level.capitalize()}: κ = {kappa:.4f} ({interpret_kappa(kappa)}) " +
                      f"[{agreement_pct:.1f}% agreement, {n_pairs} pairs]")

    # Analyze disagreements
    print("\n" + "=" * 70)
    print("Disagreement Analysis")
    print("=" * 70)

    disagreement_analysis = analyze_disagreements(instances, annotator_names)
    results['disagreement_analysis'] = disagreement_analysis

    total_instances = (len(disagreement_analysis['complete_agreement']) +
                       len(disagreement_analysis['leaf_level_disagreement']) +
                       len(disagreement_analysis['mid_level_disagreement']) +
                       len(disagreement_analysis['top_level_disagreement']))

    print(f"\nTotal annotated instances with 2+ annotators: {total_instances}")
    print(f"  Complete agreement: {len(disagreement_analysis['complete_agreement'])} " +
          f"({len(disagreement_analysis['complete_agreement']) / total_instances * 100:.1f}%)" if total_instances > 0 else "")
    print(f"  Leaf-level disagreement: {len(disagreement_analysis['leaf_level_disagreement'])} " +
          f"({len(disagreement_analysis['leaf_level_disagreement']) / total_instances * 100:.1f}%)" if total_instances > 0 else "")
    print(f"  Mid-level disagreement: {len(disagreement_analysis['mid_level_disagreement'])} " +
          f"({len(disagreement_analysis['mid_level_disagreement']) / total_instances * 100:.1f}%)" if total_instances > 0 else "")
    print(f"  Top-level disagreement: {len(disagreement_analysis['top_level_disagreement'])} " +
          f"({len(disagreement_analysis['top_level_disagreement']) / total_instances * 100:.1f}%)" if total_instances > 0 else "")

    # Resolve disagreements
    print("\n" + "=" * 70)
    print("Disagreement Resolution")
    print("=" * 70)

    resolution = resolve_disagreements(disagreement_analysis, instances)
    results['resolution'] = resolution

    print(f"\nResolution Strategy Results:")
    print(f"  Complete agreement (no adjudication needed): {resolution['complete_agreement_count']}")
    print(f"  Resolved by deepest common ancestor: {resolution['ancestor_resolved_count']}")
    print(f"  Required adjudicator: {resolution['adjudicator_required_count']}")

    # Get and display detailed disagreements
    detailed = get_detailed_disagreements(instances, disagreement_analysis)
    results['detailed_disagreements'] = detailed

    # Display detailed conflict information
    print("\n" + "-" * 70)
    print("Detailed Disagreement Information")
    print("-" * 70)

    # Complete agreements
    if detailed['complete_agreement']:
        print(f"\n[COMPLETE AGREEMENT] ({len(detailed['complete_agreement'])} instances)")
        for case in detailed['complete_agreement']:
            print(f"\n  Instance: {case['instance_id']}")
            for annotator, labels in case['annotators'].items():
                print(f"    {annotator}: {', '.join(labels)}")

    # Leaf-level (subcategory) disagreements
    if detailed['leaf_level']:
        print(f"\n[LEAF-LEVEL DISAGREEMENT - Subcategory] ({len(detailed['leaf_level'])} instances)")
        for case in detailed['leaf_level']:
            print(f"\n  Instance: {case['instance_id']}")
            print(f"  Type: {case['disagreement_type']}")
            print("  Annotations:")
            for annotator, anns in case['annotators'].items():
                ann_strs = [f"{a['dimension']}/{a['category']}/{a['subcategory']}" for a in anns]
                print(f"    {annotator}: {', '.join(ann_strs)}")
            if case['conflicts']:
                print("  Conflicts:")
                for conflict in case['conflicts']:
                    annotators = [k for k in conflict.keys() if k not in ['dimension', 'conflict_level']]
                    conflict_str = f"    [{conflict.get('dimension', '?')}] "
                    conflict_str += " vs ".join([f"{ann}={conflict[ann]}" for ann in annotators])
                    print(conflict_str)

    # Mid-level (category) disagreements
    if detailed['mid_level']:
        print(f"\n[MID-LEVEL DISAGREEMENT - Category] ({len(detailed['mid_level'])} instances)")
        for case in detailed['mid_level']:
            print(f"\n  Instance: {case['instance_id']}")
            print(f"  Type: {case['disagreement_type']}")
            print("  Annotations:")
            for annotator, anns in case['annotators'].items():
                ann_strs = [f"{a['dimension']}/{a['category']}/{a['subcategory']}" for a in anns]
                print(f"    {annotator}: {', '.join(ann_strs)}")
            if case['conflicts']:
                print("  Conflicts:")
                for conflict in case['conflicts']:
                    annotators = [k for k in conflict.keys() if k not in ['dimension', 'conflict_level']]
                    conflict_str = f"    [{conflict.get('dimension', '?')}] "
                    conflict_str += " vs ".join([f"{ann}={conflict[ann]}" for ann in annotators])
                    print(conflict_str)

    # Top-level (dimension) disagreements
    if detailed['top_level']:
        print(f"\n[TOP-LEVEL DISAGREEMENT - Dimension] ({len(detailed['top_level'])} instances)")
        for case in detailed['top_level']:
            print(f"\n  Instance: {case['instance_id']}")
            print(f"  Type: {case['disagreement_type']}")
            print("  Annotations:")
            for annotator, anns in case['annotators'].items():
                ann_strs = [f"{a['dimension']}/{a['category']}/{a['subcategory']}" for a in anns]
                print(f"    {annotator}: {', '.join(ann_strs)}")
            if case['conflicts']:
                print("  Conflicts:")
                for conflict in case['conflicts']:
                    for key, val in conflict.items():
                        if key != 'conflict_level':
                            print(f"    {key}: {val}")

    # Save results to JSON
    output_file = os.path.join(script_dir, "inter_annotator_results.json")

    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, set):
            return list(obj)
        return obj

    serializable_results = convert_to_serializable(results)

    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    results = main()
