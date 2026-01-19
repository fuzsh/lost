#!/usr/bin/env python3
"""
Inter-Annotator Agreement Calculator for Navigation Error Annotation

This script calculates inter-annotator agreement at three hierarchical levels:
- Dimension (top-level): L, T, A, E error types
- Category (mid-level): e.g., L1, L2, T1, T2, A1, etc.
- Subcategory (leaf-level): e.g., L1a, L1b, T1a, T1b, etc.

Metrics computed:
- Fleiss' Kappa for multi-annotator agreement
- Pairwise Cohen's Kappa between annotator pairs
- Agreement percentages and disagreement analysis

Usage:
    python inter-annotator.py                  # Run with real annotations
    python inter-annotator.py --simulate       # Run with simulated data
    python inter-annotator.py --simulate -n 1000  # Simulate with 1000 instances

Reference:
    Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement
    for categorical data. Biometrics, 33(1), 159-174.
"""

import json
import glob
import os
import argparse
import random
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
        'A1': ['A1a', 'A1b'],         # Spatial reference
        'A2': ['A2a', 'A2b'],         # Temporal reference
        'A3': ['A3a', 'A3b'],         # Landmark reference
        'A4': ['A4a', 'A4b'],         # Direction errors
        'A5': ['A5a', 'A5b', 'A5c'],  # Hallucination
    },
    'E': {  # Execution Errors
        'E1': ['E1a', 'E1b'],         # Action errors
        'E2': ['E2a', 'E2b'],         # Completion errors
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


def compute_fleiss_kappa(instances: Dict[str, Dict[str, List[Dict]]],
                         annotator_names: List[str],
                         level: str) -> Tuple[float, Dict]:
    """
    Compute Fleiss' Kappa for multi-annotator agreement at a given hierarchical level.

    For multi-label annotations, we use a set-based approach:
    - Each unique combination of labels is treated as a distinct category
    - This properly handles cases where annotators can assign multiple labels

    Args:
        instances: Dictionary of instance_id -> annotator_name -> annotations
        annotator_names: List of annotator names
        level: 'dimension', 'category', or 'subcategory'

    Returns:
        Tuple of (kappa value, detailed statistics)
    """
    n_raters = len(annotator_names)

    # Collect instances where all annotators provided annotations
    common_instances = []
    for instance_id, annotator_data in instances.items():
        if len(annotator_data) == n_raters:
            common_instances.append(instance_id)

    if not common_instances:
        # Fall back to instances with at least 2 annotators
        for instance_id, annotator_data in instances.items():
            if len(annotator_data) >= 2:
                common_instances.append(instance_id)
        n_raters = 2 if common_instances else 0

    if not common_instances:
        return float('nan'), {'error': 'No common instances found'}

    n_instances = len(common_instances)

    # For multi-label annotations, convert each annotator's label SET to a single category
    # This way, {A, T} is one category, {A} is another, etc.
    all_label_sets = set()
    instance_label_sets = {}  # instance_id -> annotator -> frozenset of labels

    for instance_id in common_instances:
        annotator_data = instances[instance_id]
        instance_label_sets[instance_id] = {}
        raters_for_instance = list(annotator_data.keys())[:n_raters]

        for annotator in raters_for_instance:
            labels = get_labels_for_instance(annotator_data[annotator], level)
            label_set = frozenset(labels)
            instance_label_sets[instance_id][annotator] = label_set
            all_label_sets.add(label_set)

    # Convert label sets to category indices
    all_categories = sorted([tuple(sorted(ls)) for ls in all_label_sets])
    category_to_idx = {cat: idx for idx, cat in enumerate(all_categories)}
    n_categories = len(all_categories)

    if n_categories == 0:
        return float('nan'), {'error': 'No categories found'}

    # Build the matrix n_ij: number of raters who assigned category j to instance i
    matrix = np.zeros((n_instances, n_categories))

    for i, instance_id in enumerate(common_instances):
        raters_for_instance = list(instance_label_sets[instance_id].keys())[:n_raters]
        for annotator in raters_for_instance:
            label_set = instance_label_sets[instance_id][annotator]
            cat_tuple = tuple(sorted(label_set))
            if cat_tuple in category_to_idx:
                matrix[i, category_to_idx[cat_tuple]] += 1

    # Compute Fleiss' Kappa
    # n = number of raters per instance (assumed constant)
    n = n_raters

    # P_i = proportion of agreeing pairs for instance i
    # P_i = (1 / (n*(n-1))) * sum_j(n_ij * (n_ij - 1))
    P_i = np.zeros(n_instances)
    for i in range(n_instances):
        row_sum = 0
        for j in range(n_categories):
            row_sum += matrix[i, j] * (matrix[i, j] - 1)
        if n * (n - 1) > 0:
            P_i[i] = row_sum / (n * (n - 1))
        else:
            P_i[i] = 1.0

    # P_bar = mean of P_i
    P_bar = np.mean(P_i)

    # p_j = proportion of all assignments that were category j
    total_assignments = n_instances * n
    p_j = np.sum(matrix, axis=0) / total_assignments if total_assignments > 0 else np.zeros(n_categories)

    # P_e = sum of p_j^2 (expected agreement by chance)
    P_e = np.sum(p_j ** 2)

    # Fleiss' Kappa
    if 1 - P_e > 0:
        kappa = (P_bar - P_e) / (1 - P_e)
    else:
        kappa = 1.0 if P_bar == 1.0 else 0.0

    # Get individual labels for reporting
    individual_labels = sorted(get_all_labels(instances, level))

    stats = {
        'n_instances': n_instances,
        'n_raters': n_raters,
        'n_categories': n_categories,
        'categories': [list(cat) for cat in all_categories],
        'individual_labels': individual_labels,
        'observed_agreement': P_bar,
        'expected_agreement': P_e,
    }

    return float(kappa), stats


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
                if c.isalpha() and i > 0 and sub[i-1].isdigit():
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


def generate_latex_table(results: Dict) -> str:
    """Generate LaTeX table for the paper."""
    latex = r"""
\begin{table}[h]
\centering
\caption{Inter-Annotator Agreement Across Hierarchical Levels}
\label{tab:agreement_levels}
\begin{tabular}{lccc}
\toprule
\textbf{Level} & \textbf{Fleiss' $\kappa$} & \textbf{Agreement \%} & \textbf{Interpretation} \\
\midrule
"""
    for level in ['dimension', 'category', 'subcategory']:
        level_data = results['fleiss_kappa'][level]
        kappa = level_data['kappa']
        agreement = level_data['stats'].get('observed_agreement', 0) * 100
        interpretation = interpret_kappa(kappa)

        level_display = level.capitalize()
        if level == 'dimension':
            level_display = "Top-level (Dimension)"
        elif level == 'category':
            level_display = "Mid-level (Category)"
        else:
            level_display = "Leaf-level (Subcategory)"

        latex += f"{level_display} & {kappa:.2f} & {agreement:.1f}\\% & {interpretation} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def generate_resolution_latex_table(resolution: Dict, total: int) -> str:
    """Generate LaTeX for disagreement resolution statistics."""
    complete = resolution['complete_agreement_count']
    ancestor = resolution['ancestor_resolved_count']
    adjudicator = resolution['adjudicator_required_count']

    complete_pct = (complete / total * 100) if total > 0 else 0
    ancestor_pct = (ancestor / total * 100) if total > 0 else 0
    adjudicator_pct = (adjudicator / total * 100) if total > 0 else 0

    latex = f"""
Disagreement Resolution Summary:
- Complete agreement: {complete} instances ({complete_pct:.0f}%)
- Resolved by deepest common ancestor: {ancestor} instances ({ancestor_pct:.0f}%)
- Required adjudicator: {adjudicator} instances ({adjudicator_pct:.0f}%)
"""
    return latex


def generate_simulated_annotations(n_instances: int = 1000,
                                    n_annotators: int = 3,
                                    agreement_rate: float = 0.85) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Generate simulated annotation data for testing the inter-annotator agreement calculator.

    Args:
        n_instances: Number of instances to generate
        n_annotators: Number of annotators
        agreement_rate: Probability of annotators agreeing (0.0 to 1.0)

    Returns:
        Dictionary of instance_id -> annotator_name -> annotations
    """
    random.seed(42)  # For reproducibility
    np.random.seed(42)

    # Get all possible subcategories
    all_subcategories = []
    for dim, categories in TAXONOMY.items():
        for cat, subcats in categories.items():
            for subcat in subcats:
                all_subcategories.append({
                    'dimension': dim,
                    'category': cat,
                    'subcategory': subcat
                })

    annotator_names = [f"annotator_{i+1}" for i in range(n_annotators)]
    instances = defaultdict(lambda: defaultdict(list))

    for i in range(n_instances):
        instance_id = f"simulated_{i}"

        # First annotator picks a random annotation
        base_ann = random.choice(all_subcategories)

        for annotator in annotator_names:
            if annotator == annotator_names[0] or random.random() < agreement_rate:
                # Agree with base annotation
                ann = base_ann.copy()
            else:
                # Disagree - choose a different annotation
                # Decide at which level to disagree
                disagree_level = random.choices(
                    ['subcategory', 'category', 'dimension'],
                    weights=[0.5, 0.3, 0.2]  # More likely to disagree at leaf level
                )[0]

                if disagree_level == 'subcategory':
                    # Same category, different subcategory
                    dim = base_ann['dimension']
                    cat = base_ann['category']
                    other_subcats = [s for s in TAXONOMY[dim][cat] if s != base_ann['subcategory']]
                    if other_subcats:
                        new_subcat = random.choice(other_subcats)
                        ann = {'dimension': dim, 'category': cat, 'subcategory': new_subcat}
                    else:
                        ann = base_ann.copy()
                elif disagree_level == 'category':
                    # Same dimension, different category
                    dim = base_ann['dimension']
                    other_cats = [c for c in TAXONOMY[dim].keys() if c != base_ann['category']]
                    if other_cats:
                        new_cat = random.choice(other_cats)
                        new_subcat = random.choice(TAXONOMY[dim][new_cat])
                        ann = {'dimension': dim, 'category': new_cat, 'subcategory': new_subcat}
                    else:
                        ann = base_ann.copy()
                else:
                    # Different dimension entirely
                    other_dims = [d for d in TAXONOMY.keys() if d != base_ann['dimension']]
                    if other_dims:
                        new_dim = random.choice(other_dims)
                        new_cat = random.choice(list(TAXONOMY[new_dim].keys()))
                        new_subcat = random.choice(TAXONOMY[new_dim][new_cat])
                        ann = {'dimension': new_dim, 'category': new_cat, 'subcategory': new_subcat}
                    else:
                        ann = base_ann.copy()

            ann['subcategoryName'] = f"{ann['subcategory']}_name"
            ann['timestamp'] = "2026-01-19T00:00:00.000Z"
            instances[instance_id][annotator].append(ann)

    print(f"Generated {n_instances} simulated instances with {n_annotators} annotators")
    print(f"  Agreement rate: {agreement_rate:.0%}")

    return instances


def main():
    """Main function to compute and report inter-annotator agreement."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate inter-annotator agreement for navigation error annotations')
    parser.add_argument('--simulate', action='store_true', help='Generate simulated data for testing')
    parser.add_argument('-n', '--n-instances', type=int, default=1000, help='Number of instances to simulate (default: 1000)')
    parser.add_argument('--n-annotators', type=int, default=2, help='Number of annotators to simulate (default: 3)')
    parser.add_argument('--agreement-rate', type=float, default=0.85, help='Agreement rate for simulation (default: 0.85)')
    args = parser.parse_args()

    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    annotations_dir = os.path.join(script_dir, "annotations")

    print("=" * 70)
    print("Inter-Annotator Agreement Analysis")
    print("=" * 70)
    print()

    if args.simulate:
        # Use simulated data
        print("Running in SIMULATION mode")
        print("-" * 70)
        instances = generate_simulated_annotations(
            n_instances=args.n_instances,
            n_annotators=args.n_annotators,
            agreement_rate=args.agreement_rate
        )
        annotator_names = [f"annotator_{i+1}" for i in range(args.n_annotators)]
        print()
    else:
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
        'fleiss_kappa': {},
        'pairwise_kappa': {},
        'disagreement_analysis': None,
        'resolution': None
    }

    # Compute Fleiss' Kappa at each level
    print("\n" + "=" * 70)
    print("Fleiss' Kappa Analysis")
    print("=" * 70)

    for level in ['dimension', 'category', 'subcategory']:
        kappa, stats = compute_fleiss_kappa(instances, annotator_names, level)
        results['fleiss_kappa'][level] = {'kappa': kappa, 'stats': stats}

        print(f"\n{level.upper()} Level:")
        print(f"  Fleiss' Kappa: {kappa:.4f}")
        print(f"  Interpretation: {interpret_kappa(kappa)}")
        print(f"  Observed Agreement: {stats.get('observed_agreement', 0):.4f}")
        print(f"  Expected Agreement: {stats.get('expected_agreement', 0):.4f}")
        print(f"  Number of instances: {stats.get('n_instances', 0)}")
        print(f"  Number of categories: {stats.get('n_categories', 0)}")

    # Compute pairwise Cohen's Kappa if multiple annotators
    if len(annotator_names) >= 2:
        print("\n" + "=" * 70)
        print("Pairwise Cohen's Kappa Analysis")
        print("=" * 70)

        for ann1, ann2 in combinations(annotator_names, 2):
            print(f"\n{ann1} vs {ann2}:")
            pair_key = f"{ann1}_vs_{ann2}"
            results['pairwise_kappa'][pair_key] = {}

            for level in ['dimension', 'category', 'subcategory']:
                kappa, stats = compute_pairwise_cohens_kappa(instances, ann1, ann2, level)
                results['pairwise_kappa'][pair_key][level] = {'kappa': kappa, 'stats': stats}

                print(f"  {level.capitalize()}: κ = {kappa:.4f} ({interpret_kappa(kappa)})")

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
          f"({len(disagreement_analysis['complete_agreement'])/total_instances*100:.1f}%)" if total_instances > 0 else "")
    print(f"  Leaf-level disagreement: {len(disagreement_analysis['leaf_level_disagreement'])} " +
          f"({len(disagreement_analysis['leaf_level_disagreement'])/total_instances*100:.1f}%)" if total_instances > 0 else "")
    print(f"  Mid-level disagreement: {len(disagreement_analysis['mid_level_disagreement'])} " +
          f"({len(disagreement_analysis['mid_level_disagreement'])/total_instances*100:.1f}%)" if total_instances > 0 else "")
    print(f"  Top-level disagreement: {len(disagreement_analysis['top_level_disagreement'])} " +
          f"({len(disagreement_analysis['top_level_disagreement'])/total_instances*100:.1f}%)" if total_instances > 0 else "")

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

    # Generate LaTeX tables
    print("\n" + "=" * 70)
    print("LaTeX Output for Paper")
    print("=" * 70)

    latex_table = generate_latex_table(results)
    print(latex_table)

    resolution_text = generate_resolution_latex_table(resolution, total_instances)
    print(resolution_text)

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

    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics for Paper")
    print("=" * 70)

    print(f"""
Key findings:
- Dimension-level Fleiss' κ: {results['fleiss_kappa']['dimension']['kappa']:.2f}
- Category-level Fleiss' κ: {results['fleiss_kappa']['category']['kappa']:.2f}
- Subcategory-level Fleiss' κ: {results['fleiss_kappa']['subcategory']['kappa']:.2f}

Disagreement resolution:
- Total instances: {total_instances}
- Complete agreement: {resolution['complete_agreement_count']} ({resolution['complete_agreement_count']/total_instances*100:.0f}%)
- Ancestor resolution: {resolution['ancestor_resolved_count']} ({resolution['ancestor_resolved_count']/total_instances*100:.0f}%)
- Adjudicator required: {resolution['adjudicator_required_count']} ({resolution['adjudicator_required_count']/total_instances*100:.0f}%)
""" if total_instances > 0 else "\nNo instances with multiple annotators found yet.")

    return results


if __name__ == "__main__":
    results = main()