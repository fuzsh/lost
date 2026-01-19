"""
Evaluation Metrics for Navigation Predictions.

This script analyzes test_seen.json predictions and computes:
1. Path overlap metrics (comparing expanded predicted paths with osm_path)
2. Final node distance metrics (haversine distance between endpoints)
3. Correctness by number of sub-goals
4. UpSet plots showing method agreement patterns

Usage:
    python evaluation_metrics.py --input test_seen.json --output evaluation_results
"""

import argparse
import json
import math
import os
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Set

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Try to import upsetplot, fall back gracefully
try:
    from upsetplot import UpSet, from_memberships

    HAS_UPSETPLOT = True
except ImportError:
    HAS_UPSETPLOT = False
    print("Warning: upsetplot not installed. UpSet plots will be skipped.")

from src.data_loader import get_data_by_instruction

# =============================================================================
# CONSTANTS
# =============================================================================

METHODS = ['json']
# METHODS = ['action_sampling', 'heuristic_agent','random_walker']

DISTANCE_THRESHOLD = 25  # meters - default threshold for considering prediction correct
DISTANCE_THRESHOLDS = [25, 50, 100, 150]  # Multiple thresholds for analysis
NAV_SUCCESS_THRESHOLD = 25  # meters - threshold for navigation success (SR, OSR)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """
    Calculate the Haversine distance between two points in meters.

    Args:
        lat1, lng1: First point coordinates
        lat2, lng2: Second point coordinates

    Returns:
        Distance in meters
    """
    R = 6371000  # Earth's radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lng2 - lng1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def build_adjacency_graph(area_links: List[Dict]) -> Dict[str, List[str]]:
    """Build adjacency list from area links."""
    graph = defaultdict(list)
    for link in area_links:
        src = link['source']
        tgt = link['target']
        graph[src].append(tgt)
    return graph


def bfs_path(graph: Dict[str, List[str]], start: str, end: str) -> Optional[List[str]]:
    """
    Find shortest path between two nodes using BFS.

    Returns:
        List of node IDs from start to end, or None if no path exists.
    """
    if start == end:
        return [start]

    queue = deque([[start]])
    visited = {start}

    while queue:
        path = queue.popleft()
        current = path[-1]

        if current == end:
            return path

        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)

    return None


def expand_predicted_path(
        predicted_waypoints: List[str],
        graph: Dict[str, List[str]],
        area_nodes: Dict[str, Dict]
) -> List[str]:
    """
    Expand predicted waypoints into full path by finding BFS paths between
    consecutive waypoints.

    Args:
        predicted_waypoints: List of waypoint node IDs
        graph: Adjacency graph
        area_nodes: Node coordinate data

    Returns:
        Expanded path with all intermediate nodes
    """
    if not predicted_waypoints:
        return []

    # Remove duplicates while preserving order
    seen = set()
    unique_waypoints = []
    for wp in predicted_waypoints:
        if wp is not None and wp not in seen:
            seen.add(wp)
            unique_waypoints.append(wp)

    if len(unique_waypoints) <= 1:
        return unique_waypoints

    # Expand path between consecutive waypoints
    expanded = [unique_waypoints[0]]

    for i in range(len(unique_waypoints) - 1):
        start = unique_waypoints[i]
        end = unique_waypoints[i + 1]

        # Skip if node doesn't exist
        if start not in area_nodes or end not in area_nodes:
            continue

        path = bfs_path(graph, start, end)

        if path:
            # Add path excluding start (already in expanded)
            expanded.extend(path[1:])
        else:
            # No path found, just add the endpoint
            if end not in expanded:
                expanded.append(end)

    return expanded


# =============================================================================
# NAVIGATION METRICS (NE, SR, OSR, SDTW)
# =============================================================================

def calculate_navigation_error(
        predicted_path: List[str],
        osm_path: List[str],
        area_nodes: Dict[str, Dict]
) -> float:
    """
    Calculate Navigation Error (NE): the distance between the agent's stopping
    point and the actual destination.

    Args:
        predicted_path: List of predicted node IDs (expanded path)
        osm_path: List of ground truth node IDs
        area_nodes: Node coordinate data {node_id: {lat, lng}}

    Returns:
        Distance in meters between final predicted node and goal, or inf if invalid
    """
    if not predicted_path or not osm_path:
        return float('inf')

    # Get last valid predicted node
    pred_end = None
    for node in reversed(predicted_path):
        if node and node in area_nodes:
            pred_end = node
            break

    # Get destination node (last in osm_path)
    osm_end = None
    for node in reversed(osm_path):
        if node and node in area_nodes:
            osm_end = node
            break

    if not pred_end or not osm_end:
        return float('inf')

    pred_coords = area_nodes[pred_end]
    osm_coords = area_nodes[osm_end]

    return haversine_distance(
        pred_coords['lat'], pred_coords['lng'],
        osm_coords['lat'], osm_coords['lng']
    )


def calculate_success_rate(
        navigation_error: float,
        threshold: float = NAV_SUCCESS_THRESHOLD
) -> bool:
    """
    Calculate Success Rate (SR): whether the navigation reaches the destination
    within the threshold (default 25 meters).

    Args:
        navigation_error: The NE value in meters
        threshold: Success threshold in meters (default 25m)

    Returns:
        True if navigation_error <= threshold, False otherwise
    """
    return navigation_error <= threshold


def calculate_oracle_success_rate(
        predicted_path: List[str],
        osm_path: List[str],
        area_nodes: Dict[str, Dict],
        threshold: float = NAV_SUCCESS_THRESHOLD
) -> Tuple[bool, float]:
    """
    Calculate Oracle Success Rate (OSR): whether ANY point on the predicted
    trajectory comes within the threshold of the destination.

    This is an idealized measure assuming an oracle could stop at the optimal point.

    Args:
        predicted_path: List of predicted node IDs (expanded path)
        osm_path: List of ground truth node IDs
        area_nodes: Node coordinate data
        threshold: Success threshold in meters (default 25m)

    Returns:
        Tuple of (success: bool, min_distance: float to destination)
    """
    if not predicted_path or not osm_path:
        return False, float('inf')

    # Get destination node
    osm_end = None
    for node in reversed(osm_path):
        if node and node in area_nodes:
            osm_end = node
            break

    if not osm_end:
        return False, float('inf')

    goal_coords = area_nodes[osm_end]
    min_distance = float('inf')

    # Check every point on the predicted path
    for node in predicted_path:
        if node and node in area_nodes:
            node_coords = area_nodes[node]
            distance = haversine_distance(
                node_coords['lat'], node_coords['lng'],
                goal_coords['lat'], goal_coords['lng']
            )
            min_distance = min(min_distance, distance)

    return min_distance <= threshold, min_distance


def calculate_path_length_meters(
        path: List[str],
        area_nodes: Dict[str, Dict]
) -> float:
    """
    Calculate the total length of a path in meters.

    Args:
        path: List of node IDs
        area_nodes: Node coordinate data

    Returns:
        Total path length in meters
    """
    if not path or len(path) < 2:
        return 0.0

    total_length = 0.0
    prev_node = None

    for node in path:
        if node and node in area_nodes:
            if prev_node and prev_node in area_nodes:
                prev_coords = area_nodes[prev_node]
                curr_coords = area_nodes[node]
                total_length += haversine_distance(
                    prev_coords['lat'], prev_coords['lng'],
                    curr_coords['lat'], curr_coords['lng']
                )
            prev_node = node

    return total_length


def calculate_dtw(
        predicted_path: List[str],
        osm_path: List[str],
        area_nodes: Dict[str, Dict]
) -> float:
    """
    Calculate Dynamic Time Warping (DTW) distance between two paths.

    DTW measures the similarity between two temporal sequences that may vary
    in speed or timing.

    Args:
        predicted_path: List of predicted node IDs
        osm_path: List of ground truth node IDs
        area_nodes: Node coordinate data

    Returns:
        DTW distance (sum of minimum distances at each alignment)
    """
    # Get valid coordinates for each path
    pred_coords = []
    for node in predicted_path:
        if node and node in area_nodes:
            coords = area_nodes[node]
            pred_coords.append((coords['lat'], coords['lng']))

    osm_coords = []
    for node in osm_path:
        if node and node in area_nodes:
            coords = area_nodes[node]
            osm_coords.append((coords['lat'], coords['lng']))

    if not pred_coords or not osm_coords:
        return float('inf')

    n, m = len(pred_coords), len(osm_coords)

    # Create DTW matrix
    dtw_matrix = np.full((n + 1, m + 1), float('inf'))
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = haversine_distance(
                pred_coords[i - 1][0], pred_coords[i - 1][1],
                osm_coords[j - 1][0], osm_coords[j - 1][1]
            )
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],  # insertion
                dtw_matrix[i, j - 1],  # deletion
                dtw_matrix[i - 1, j - 1]  # match
            )

    return dtw_matrix[n, m]


def calculate_ndtw(
        predicted_path: List[str],
        osm_path: List[str],
        area_nodes: Dict[str, Dict]
) -> float:
    """
    Calculate Normalized DTW (nDTW) using path length normalization.

    nDTW normalizes the DTW distance by the length of the reference path.

    Args:
        predicted_path: List of predicted node IDs
        osm_path: List of ground truth node IDs
        area_nodes: Node coordinate data

    Returns:
        Normalized DTW value (0-1 range, lower is better) or 1.0 if path is invalid
    """
    dtw = calculate_dtw(predicted_path, osm_path, area_nodes)

    if dtw == float('inf'):
        return 1.0

    osm_length = calculate_path_length_meters(osm_path, area_nodes)

    if osm_length == 0:
        return 1.0

    # Normalize by path length - use exp(-dtw/length) for 0-1 range
    # Following VLN-CE convention: nDTW = exp(-DTW / (path_length * scale_factor))
    # Scale factor ensures meaningful normalization
    scale_factor = max(len(osm_path), 1)
    ndtw = np.exp(-dtw / (osm_length * scale_factor / len(osm_path)))

    return ndtw


def calculate_sdtw(
        predicted_path: List[str],
        osm_path: List[str],
        area_nodes: Dict[str, Dict],
        threshold: float = NAV_SUCCESS_THRESHOLD
) -> float:
    """
    Calculate Success-weighted Dynamic Time Warping (SDTW).

    SDTW combines success rate with trajectory similarity:
    SDTW = SR * nDTW

    Where SR is 1 if the agent reaches within threshold of goal, 0 otherwise.
    This ensures that unsuccessful navigations get 0 regardless of path similarity.

    Args:
        predicted_path: List of predicted node IDs
        osm_path: List of ground truth node IDs
        area_nodes: Node coordinate data
        threshold: Success threshold in meters

    Returns:
        SDTW value (0-1 range, higher is better)
    """
    # Calculate navigation error
    ne = calculate_navigation_error(predicted_path, osm_path, area_nodes)

    # Check success
    success = calculate_success_rate(ne, threshold)

    if not success:
        return 0.0

    # Calculate normalized DTW
    ndtw = calculate_ndtw(predicted_path, osm_path, area_nodes)

    return ndtw


def calculate_all_nav_metrics(
        predicted_path: List[str],
        osm_path: List[str],
        area_nodes: Dict[str, Dict],
        threshold: float = NAV_SUCCESS_THRESHOLD
) -> Dict:
    """
    Calculate all navigation metrics: NE, SR, OSR, SDTW.

    Args:
        predicted_path: List of predicted node IDs (expanded path)
        osm_path: List of ground truth node IDs
        area_nodes: Node coordinate data
        threshold: Success threshold in meters

    Returns:
        Dict with all navigation metrics
    """
    # Navigation Error
    ne = calculate_navigation_error(predicted_path, osm_path, area_nodes)

    # Success Rate
    sr = calculate_success_rate(ne, threshold)

    # Oracle Success Rate
    osr, min_dist_to_goal = calculate_oracle_success_rate(
        predicted_path, osm_path, area_nodes, threshold
    )

    # SDTW
    sdtw = calculate_sdtw(predicted_path, osm_path, area_nodes, threshold)

    # nDTW (for reference)
    ndtw = calculate_ndtw(predicted_path, osm_path, area_nodes)

    return {
        'navigation_error': ne,
        'success': sr,
        'oracle_success': osr,
        'min_distance_to_goal': min_dist_to_goal,
        'sdtw': sdtw,
        'ndtw': ndtw
    }


def calculate_path_overlap(predicted_path: List[str], osm_path: List[str]) -> Dict:
    """
    Calculate overlap metrics between predicted and ground truth paths.

    Returns:
        Dict with overlap metrics (overlap_ratio, precision, recall, f1)
    """
    if not predicted_path or not osm_path:
        return {
            'overlap_ratio': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'predicted_length': len(predicted_path) if predicted_path else 0,
            'osm_length': len(osm_path) if osm_path else 0,
            'common_nodes': 0
        }

    predicted_set = set(predicted_path)
    osm_set = set(osm_path)

    common = predicted_set & osm_set

    precision = len(common) / len(predicted_set) if predicted_set else 0.0
    recall = len(common) / len(osm_set) if osm_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Also compute sequence overlap (ordered)
    sequence_overlap = 0
    osm_index = {node: i for i, node in enumerate(osm_path)}
    last_osm_idx = -1

    for node in predicted_path:
        if node in osm_index:
            idx = osm_index[node]
            if idx > last_osm_idx:
                sequence_overlap += 1
                last_osm_idx = idx

    sequence_ratio = sequence_overlap / len(osm_path) if osm_path else 0.0

    return {
        'overlap_ratio': len(common) / len(osm_set) if osm_set else 0.0,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sequence_overlap': sequence_ratio,
        'predicted_length': len(predicted_path),
        'osm_length': len(osm_path),
        'common_nodes': len(common)
    }


def calculate_endpoint_distance(
        predicted_path: List[str],
        osm_path: List[str],
        area_nodes: Dict[str, Dict]
) -> Dict:
    """
    Calculate distance between final predicted node and final OSM node.

    Returns:
        Dict with distance metrics
    """
    result = {
        'distance_meters': float('inf'),
        'is_correct': False,
        'predicted_end': None,
        'osm_end': None
    }

    if not predicted_path or not osm_path:
        return result

    # Get last valid nodes
    pred_end = None
    for node in reversed(predicted_path):
        if node and node in area_nodes:
            pred_end = node
            break

    osm_end = None
    for node in reversed(osm_path):
        if node and node in area_nodes:
            osm_end = node
            break

    if not pred_end or not osm_end:
        return result

    pred_coords = area_nodes[pred_end]
    osm_coords = area_nodes[osm_end]

    distance = haversine_distance(
        pred_coords['lat'], pred_coords['lng'],
        osm_coords['lat'], osm_coords['lng']
    )

    return {
        'distance_meters': distance,
        'is_correct': distance <= DISTANCE_THRESHOLD,
        'predicted_end': pred_end,
        'osm_end': osm_end
    }


# =============================================================================
# MAIN EVALUATION FUNCTIONS
# =============================================================================

def evaluate_single_instruction(
        key: str,
        instruction_data: Dict,
        split_file: str = "test_seen_200.json"
) -> Optional[Dict]:
    """
    Evaluate a single instruction across all methods.

    Returns:
        Dict with evaluation metrics for each method
    """
    osm_path = instruction_data.get('osm_path', [])
    sub_instructions = instruction_data.get('sub_instructions', [])
    num_sub_goals = len(sub_instructions)

    if not osm_path:
        return None

    # Load area data
    try:
        area_data = get_data_by_instruction(
            int(key),
            split_file,
            base_path='./data/map2seq/',
            neighbor_degrees=20
        )
    except Exception as e:
        print(f"Error loading area data for key {key}: {e}")
        return None

    if not area_data:
        return None

    area_nodes = area_data.get('area_nodes', {})
    area_links = area_data.get('area_links', [])
    graph = build_adjacency_graph(area_links)

    results = {
        'key': key,
        'num_sub_goals': num_sub_goals,
        'osm_path_length': len(osm_path),
        'methods': {}
    }

    for method in METHODS:
        method_data = instruction_data.get(method, {})

        if not method_data:
            results['methods'][method] = {
                'status': 'MISSING',
                'path_overlap': None,
                'endpoint_distance': None
            }
            continue

        status = method_data.get('status', 'UNKNOWN')
        predicted_waypoints = method_data.get('predicated_path', [])

        # Filter out None values
        predicted_waypoints = [p for p in predicted_waypoints if p is not None]

        if len(predicted_waypoints) <= 1:
            # Completely wrong - only start node or no path
            results['methods'][method] = {
                'status': status,
                'is_completely_wrong': True,
                'path_overlap': {
                    'overlap_ratio': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'sequence_overlap': 0.0,
                    'predicted_length': len(predicted_waypoints),
                    'osm_length': len(osm_path),
                    'common_nodes': 0
                },
                'endpoint_distance': {
                    'distance_meters': float('inf'),
                    'is_correct': False,
                    'predicted_end': None,
                    'osm_end': osm_path[-1] if osm_path else None
                },
                'nav_metrics': {
                    'navigation_error': float('inf'),
                    'success': False,
                    'oracle_success': False,
                    'min_distance_to_goal': float('inf'),
                    'sdtw': 0.0,
                    'ndtw': 1.0
                }
            }
            continue

        # Expand predicted path using BFS
        expanded_path = expand_predicted_path(predicted_waypoints, graph, area_nodes)

        # Calculate metrics
        path_overlap = calculate_path_overlap(expanded_path, osm_path)
        endpoint_distance = calculate_endpoint_distance(expanded_path, osm_path, area_nodes)

        # Calculate navigation metrics (NE, SR, OSR, SDTW)
        nav_metrics = calculate_all_nav_metrics(expanded_path, osm_path, area_nodes)

        results['methods'][method] = {
            'status': status,
            'is_completely_wrong': False,
            'num_waypoints': len(predicted_waypoints),
            'expanded_path_length': len(expanded_path),
            'path_overlap': path_overlap,
            'endpoint_distance': endpoint_distance,
            'nav_metrics': nav_metrics
        }

    return results


def aggregate_results(all_results: List[Dict]) -> Dict:
    """
    Aggregate evaluation results across all instructions.

    Returns:
        Dict with aggregated metrics
    """
    aggregated = {
        'total_instructions': len(all_results),
        'by_method': {},
        'by_subgoals': defaultdict(lambda: defaultdict(list)),
        'method_agreement': defaultdict(int),
        'by_threshold': {t: {} for t in DISTANCE_THRESHOLDS}
    }

    for method in METHODS:
        aggregated['by_method'][method] = {
            'total': 0,
            'completely_wrong': 0,
            'endpoint_correct': 0,
            'avg_overlap_ratio': [],
            'avg_f1': [],
            'avg_sequence_overlap': [],
            'avg_endpoint_distance': [],
            'endpoint_distances': [],
            'correct_by_threshold': {t: 0 for t in DISTANCE_THRESHOLDS},
            # Navigation metrics (NE, SR, OSR, SDTW)
            'navigation_errors': [],
            'success_count': 0,
            'oracle_success_count': 0,
            'sdtw_values': [],
            'ndtw_values': [],
            'min_distances_to_goal': []
        }

    for result in all_results:
        num_subgoals = result['num_sub_goals']

        # Track which methods are correct for this instruction (at primary threshold)
        correct_methods = set()
        # Track for each threshold
        correct_by_threshold = {t: set() for t in DISTANCE_THRESHOLDS}

        for method in METHODS:
            method_result = result['methods'].get(method, {})

            if method_result.get('status') == 'MISSING':
                continue

            agg = aggregated['by_method'][method]
            agg['total'] += 1

            if method_result.get('is_completely_wrong', False):
                agg['completely_wrong'] += 1
                # Still collect nav metrics for completely wrong cases
                nav_metrics = method_result.get('nav_metrics', {})
                if nav_metrics:
                    ne = nav_metrics.get('navigation_error', float('inf'))
                    if ne != float('inf'):
                        agg['navigation_errors'].append(ne)
                    agg['sdtw_values'].append(nav_metrics.get('sdtw', 0.0))
                    agg['ndtw_values'].append(nav_metrics.get('ndtw', 1.0))
                continue

            path_overlap = method_result.get('path_overlap', {})
            endpoint_dist = method_result.get('endpoint_distance', {})

            agg['avg_overlap_ratio'].append(path_overlap.get('overlap_ratio', 0))
            agg['avg_f1'].append(path_overlap.get('f1', 0))
            agg['avg_sequence_overlap'].append(path_overlap.get('sequence_overlap', 0))

            dist = endpoint_dist.get('distance_meters', float('inf'))
            if dist != float('inf'):
                agg['avg_endpoint_distance'].append(dist)
                agg['endpoint_distances'].append(dist)

                # Check against all thresholds
                for threshold in DISTANCE_THRESHOLDS:
                    if dist <= threshold:
                        agg['correct_by_threshold'][threshold] += 1
                        correct_by_threshold[threshold].add(method)

            if endpoint_dist.get('is_correct', False):
                agg['endpoint_correct'] += 1
                correct_methods.add(method)

                # Track correctness by sub-goals
                aggregated['by_subgoals'][num_subgoals][method].append(result['key'])
            else:
                print(method, result['key'])

            # Collect navigation metrics (NE, SR, OSR, SDTW)
            nav_metrics = method_result.get('nav_metrics', {})
            if nav_metrics:
                ne = nav_metrics.get('navigation_error', float('inf'))
                if ne != float('inf'):
                    agg['navigation_errors'].append(ne)
                    min_dist = nav_metrics.get('min_distance_to_goal', float('inf'))
                    if min_dist != float('inf'):
                        agg['min_distances_to_goal'].append(min_dist)

                if nav_metrics.get('success', False):
                    agg['success_count'] += 1
                if nav_metrics.get('oracle_success', False):
                    agg['oracle_success_count'] += 1

                agg['sdtw_values'].append(nav_metrics.get('sdtw', 0.0))
                agg['ndtw_values'].append(nav_metrics.get('ndtw', 1.0))

        # Track method agreement (for UpSet plot) - use 100m threshold for agreement
        for threshold in DISTANCE_THRESHOLDS:
            if correct_by_threshold[threshold]:
                agreement_key = tuple(sorted(correct_by_threshold[threshold]))
                if threshold not in aggregated['method_agreement']:
                    aggregated['method_agreement'][threshold] = defaultdict(int)
                aggregated['method_agreement'][threshold][agreement_key] += 1

    # Calculate averages
    for method in METHODS:
        agg = aggregated['by_method'][method]

        if agg['avg_overlap_ratio']:
            agg['avg_overlap_ratio'] = np.mean(agg['avg_overlap_ratio'])
        else:
            agg['avg_overlap_ratio'] = 0.0

        if agg['avg_f1']:
            agg['avg_f1'] = np.mean(agg['avg_f1'])
        else:
            agg['avg_f1'] = 0.0

        if agg['avg_sequence_overlap']:
            agg['avg_sequence_overlap'] = np.mean(agg['avg_sequence_overlap'])
        else:
            agg['avg_sequence_overlap'] = 0.0

        if agg['avg_endpoint_distance']:
            agg['avg_endpoint_distance'] = np.mean(agg['avg_endpoint_distance'])
            # Add percentiles
            distances = agg['endpoint_distances']
            agg['distance_percentiles'] = {
                '10th': np.percentile(distances, 10),
                '25th': np.percentile(distances, 25),
                '50th': np.percentile(distances, 50),
                '75th': np.percentile(distances, 75),
                '90th': np.percentile(distances, 90),
                'min': np.min(distances),
                'max': np.max(distances)
            }
        else:
            agg['avg_endpoint_distance'] = float('inf')
            agg['distance_percentiles'] = {}

        # Calculate accuracy at primary threshold
        if agg['total'] > 0:
            agg['accuracy'] = agg['endpoint_correct'] / agg['total']
            # Add accuracy at each threshold
            agg['accuracy_by_threshold'] = {
                t: agg['correct_by_threshold'][t] / agg['total']
                for t in DISTANCE_THRESHOLDS
            }
        else:
            agg['accuracy'] = 0.0
            agg['accuracy_by_threshold'] = {t: 0.0 for t in DISTANCE_THRESHOLDS}

        # Calculate navigation metrics averages
        # NE (Navigation Error) - average distance to goal
        if agg['navigation_errors']:
            agg['avg_navigation_error'] = np.mean(agg['navigation_errors'])
            agg['median_navigation_error'] = np.median(agg['navigation_errors'])
        else:
            agg['avg_navigation_error'] = float('inf')
            agg['median_navigation_error'] = float('inf')

        # SR (Success Rate) - proportion within 25m threshold
        if agg['total'] > 0:
            agg['success_rate'] = agg['success_count'] / agg['total']
        else:
            agg['success_rate'] = 0.0

        # OSR (Oracle Success Rate) - any point within 25m
        if agg['total'] > 0:
            agg['oracle_success_rate'] = agg['oracle_success_count'] / agg['total']
        else:
            agg['oracle_success_rate'] = 0.0

        # SDTW (Success-weighted DTW)
        if agg['sdtw_values']:
            agg['avg_sdtw'] = np.mean(agg['sdtw_values'])
        else:
            agg['avg_sdtw'] = 0.0

        # nDTW (for reference)
        if agg['ndtw_values']:
            agg['avg_ndtw'] = np.mean(agg['ndtw_values'])
        else:
            agg['avg_ndtw'] = 0.0

    return aggregated


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_correctness_by_subgoals(aggregated: Dict, output_dir: str, all_results: List[Dict] = None):
    """Create bar chart of correctness by number of sub-goals."""
    by_subgoals = aggregated['by_subgoals']

    # If no data at primary threshold, try to compute for a higher threshold
    if not by_subgoals and all_results:
        print("Computing sub-goal data for 100m threshold...")
        by_subgoals = defaultdict(lambda: defaultdict(list))

        for result in all_results:
            num_subgoals = result['num_sub_goals']
            for method in METHODS:
                method_result = result['methods'].get(method, {})
                if method_result.get('status') == 'MISSING' or method_result.get('is_completely_wrong', False):
                    continue
                endpoint_dist = method_result.get('endpoint_distance', {})
                dist = endpoint_dist.get('distance_meters', float('inf'))
                if dist <= 100:  # Use 100m threshold for visualization
                    by_subgoals[num_subgoals][method].append(result['key'])

    if not by_subgoals:
        print("No sub-goal data to plot")
        return

    # Prepare data
    subgoal_counts = sorted(by_subgoals.keys())
    method_data = {method: [] for method in METHODS}

    for sg_count in subgoal_counts:
        for method in METHODS:
            correct_count = len(by_subgoals[sg_count].get(method, []))
            method_data[method].append(correct_count)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(subgoal_counts))
    width = 0.2

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, method in enumerate(METHODS):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, method_data[method], width, label=method, color=colors[i])

        # Add value labels on bars
        for bar, val in zip(bars, method_data[method]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        str(val), ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Number of Sub-goals')
    ax.set_ylabel('Number of Correct Predictions')
    threshold_used = DISTANCE_THRESHOLD if aggregated['by_subgoals'] else 100
    ax.set_title(f'Correct Predictions by Number of Sub-goals (threshold ≤ {threshold_used}m)')
    ax.set_xticks(x)
    ax.set_xticklabels(subgoal_counts)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correctness_by_subgoals.png'), dpi=150)
    plt.close()
    print(f"Saved: correctness_by_subgoals.png")


def plot_endpoint_distance_distribution(aggregated: Dict, output_dir: str):
    """Create histogram of endpoint distances by method."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, method in enumerate(METHODS):
        ax = axes[i]
        distances = aggregated['by_method'][method].get('endpoint_distances', [])

        if distances:
            # Filter out very large distances for visualization
            distances = [d for d in distances if d < 200]

            ax.hist(distances, bins=30, color=colors[i], alpha=0.7, edgecolor='black')
            ax.axvline(x=DISTANCE_THRESHOLD, color='red', linestyle='--',
                       label=f'Threshold ({DISTANCE_THRESHOLD}m)')

            # Add statistics
            mean_dist = np.mean(distances)
            median_dist = np.median(distances)
            correct_pct = sum(1 for d in distances if d <= DISTANCE_THRESHOLD) / len(distances) * 100

            ax.text(0.95, 0.95,
                    f'Mean: {mean_dist:.1f}m\nMedian: {median_dist:.1f}m\nCorrect: {correct_pct:.1f}%',
                    transform=ax.transAxes, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Distance to Goal (meters)')
        ax.set_ylabel('Count')
        ax.set_title(f'{method.upper()} - Endpoint Distance Distribution')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'endpoint_distance_distribution.png'), dpi=150)
    plt.close()
    print(f"Saved: endpoint_distance_distribution.png")


def plot_method_comparison(aggregated: Dict, output_dir: str):
    """Create comparison bar chart for all metrics across methods."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    methods = METHODS
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Accuracy
    accuracies = [aggregated['by_method'][m].get('accuracy', 0) * 100 for m in methods]
    axes[0].bar(methods, accuracies, color=colors)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title(f'Endpoint Accuracy (≤ {DISTANCE_THRESHOLD}m)')
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 1, f'{v:.1f}%', ha='center')

    # Average F1 Score
    f1_scores = [aggregated['by_method'][m].get('avg_f1', 0) * 100 for m in methods]
    axes[1].bar(methods, f1_scores, color=colors)
    axes[1].set_ylabel('F1 Score (%)')
    axes[1].set_title('Path Overlap F1 Score')
    axes[1].set_ylim(0, 100)
    for i, v in enumerate(f1_scores):
        axes[1].text(i, v + 1, f'{v:.1f}%', ha='center')

    # Average Endpoint Distance
    distances = [aggregated['by_method'][m].get('avg_endpoint_distance', 0) for m in methods]
    distances = [d if d != float('inf') else 0 for d in distances]
    axes[2].bar(methods, distances, color=colors)
    axes[2].set_ylabel('Avg Distance (meters)')
    axes[2].set_title('Average Endpoint Distance')
    for i, v in enumerate(distances):
        axes[2].text(i, v + 0.5, f'{v:.1f}m', ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_comparison.png'), dpi=150)
    plt.close()
    print(f"Saved: method_comparison.png")


def plot_upset(aggregated: Dict, output_dir: str, threshold: int = 100):
    """Create UpSet plot showing method agreement patterns."""
    if not HAS_UPSETPLOT:
        print("Skipping UpSet plot (upsetplot not installed)")
        return

    method_agreement = aggregated['method_agreement']

    if not method_agreement or threshold not in method_agreement:
        print(f"No method agreement data for UpSet plot at {threshold}m threshold")
        return

    try:
        # Convert to format expected by upsetplot
        memberships = []
        for methods_tuple, count in method_agreement[threshold].items():
            for _ in range(count):
                memberships.append(list(methods_tuple))

        if not memberships:
            print(f"No memberships for UpSet plot at {threshold}m")
            return

        data = from_memberships(memberships)

        fig = plt.figure(figsize=(12, 8))
        upset = UpSet(data, subset_size='count', show_counts=True, sort_by='cardinality')
        upset.plot(fig=fig)

        plt.suptitle(
            f'Method Agreement Patterns (≤{threshold}m threshold)\n(Which methods correctly predict the same instructions)',
            y=1.02)

        plt.savefig(os.path.join(output_dir, f'upset_method_agreement_{threshold}m.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: upset_method_agreement_{threshold}m.png")
    except Exception as e:
        print(f"Error creating UpSet plot for {threshold}m: {e}")
        plt.close()


def plot_accuracy_by_threshold(aggregated: Dict, output_dir: str):
    """Create line plot showing accuracy at different thresholds."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, method in enumerate(METHODS):
        metrics = aggregated['by_method'][method]
        accuracy_by_thresh = metrics.get('accuracy_by_threshold', {})

        thresholds = sorted(accuracy_by_thresh.keys())
        accuracies = [accuracy_by_thresh[t] * 100 for t in thresholds]

        ax.plot(thresholds, accuracies, marker='o', label=method, color=colors[i], linewidth=2, markersize=8)

    ax.set_xlabel('Distance Threshold (meters)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Endpoint Accuracy at Different Distance Thresholds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_by_threshold.png'), dpi=150)
    plt.close()
    print(f"Saved: accuracy_by_threshold.png")


def plot_distance_boxplot(aggregated: Dict, output_dir: str):
    """Create box plot of endpoint distances by method."""
    fig, ax = plt.subplots(figsize=(10, 6))

    data = []
    labels = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for method in METHODS:
        distances = aggregated['by_method'][method].get('endpoint_distances', [])
        if distances:
            # Filter out very large distances for visualization
            filtered = [d for d in distances if d < 400]
            data.append(filtered)
            labels.append(method)

    if data:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)

        for patch, color in zip(bp['boxes'], colors[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Add horizontal lines for thresholds
        for threshold in [50, 100, 150]:
            ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.5,
                       label=f'{threshold}m' if threshold == 50 else '')

        ax.set_ylabel('Endpoint Distance (meters)')
        ax.set_title('Distribution of Endpoint Distances by Method')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distance_boxplot.png'), dpi=150)
    plt.close()
    print(f"Saved: distance_boxplot.png")


def plot_overlap_metrics(aggregated: Dict, output_dir: str):
    """Create bar chart comparing overlap metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['Overlap Ratio', 'Precision (F1)', 'Sequence Overlap']
    x = np.arange(len(metrics))
    width = 0.2

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, method in enumerate(METHODS):
        values = [
            aggregated['by_method'][method].get('avg_overlap_ratio', 0) * 100,
            aggregated['by_method'][method].get('avg_f1', 0) * 100,
            aggregated['by_method'][method].get('avg_sequence_overlap', 0) * 100
        ]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=method, color=colors[i])

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Score (%)')
    ax.set_title('Path Overlap Metrics by Method')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overlap_metrics.png'), dpi=150)
    plt.close()
    print(f"Saved: overlap_metrics.png")


def plot_navigation_metrics(aggregated: Dict, output_dir: str):
    """Create bar chart comparing navigation metrics (NE, SR, OSR, SDTW)."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Navigation Error (lower is better)
    ax = axes[0, 0]
    ne_values = []
    for method in METHODS:
        ne = aggregated['by_method'][method].get('avg_navigation_error', float('inf'))
        ne_values.append(ne if ne != float('inf') else 0)
    bars = ax.bar(METHODS, ne_values, color=colors)
    ax.set_ylabel('Navigation Error (meters)')
    ax.set_title('Average Navigation Error (NE) - Lower is Better')
    for bar, val in zip(bars, ne_values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}m', ha='center', va='bottom', fontsize=9)

    # Success Rate (higher is better)
    ax = axes[0, 1]
    sr_values = [aggregated['by_method'][m].get('success_rate', 0) * 100 for m in METHODS]
    bars = ax.bar(METHODS, sr_values, color=colors)
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(f'Success Rate (SR) at {NAV_SUCCESS_THRESHOLD}m - Higher is Better')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, sr_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # Oracle Success Rate (higher is better)
    ax = axes[1, 0]
    osr_values = [aggregated['by_method'][m].get('oracle_success_rate', 0) * 100 for m in METHODS]
    bars = ax.bar(METHODS, osr_values, color=colors)
    ax.set_ylabel('Oracle Success Rate (%)')
    ax.set_title(f'Oracle Success Rate (OSR) at {NAV_SUCCESS_THRESHOLD}m - Higher is Better')
    ax.set_ylim(0, 100)
    for bar, val in zip(bars, osr_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # SDTW (higher is better)
    ax = axes[1, 1]
    sdtw_values = [aggregated['by_method'][m].get('avg_sdtw', 0) for m in METHODS]
    bars = ax.bar(METHODS, sdtw_values, color=colors)
    ax.set_ylabel('SDTW Score')
    ax.set_title('Success-weighted DTW (SDTW) - Higher is Better')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, sdtw_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'navigation_metrics.png'), dpi=150)
    plt.close()
    print(f"Saved: navigation_metrics.png")


# =============================================================================
# DIFFICULTY ASSESSMENT HELPERS
# =============================================================================

def load_difficulty_data(difficulty_file: str) -> Dict[str, Dict]:
    """
    Load difficulty assessment data from JSON file.

    Args:
        difficulty_file: Path to JSON file with complexity data

    Returns:
        Dict mapping instruction_id to complexity data
    """
    with open(difficulty_file, 'r') as f:
        return json.load(f)


def get_difficulty_category(
        complexity: Dict[str, int],
) -> str:
    """
    Categorize difficulty based on average complexity scores.

    Args:
        complexity: Dict with 'cognitive', 'spatial', 'execution' scores

    Returns:
        'easy', 'medium', or 'hard'
    """
    if not complexity:
        return 'unknown'

    avg_complexity = (
                             complexity.get('cognitive', 0) +
                             complexity.get('spatial', 0) +
                             complexity.get('execution', 0)
                     ) / 3.0

    min_val = 2.0  # TODO: NEED TO RUN interpret.py each time
    max_val = 7.0  # TODO: NEED TO RUN interpret.py each time

    avg_complexity = (avg_complexity - min_val) / (max_val - min_val)

    if avg_complexity <= 1 / 3:
        return 'easy'
    elif avg_complexity >= 2 / 3:
        return 'hard'
    else:
        return 'medium'


def aggregate_by_difficulty(
        all_results: List[Dict],
        difficulty_data: Dict[str, Dict],
) -> Dict:
    """
    Aggregate navigation metrics by difficulty category.

    Args:
        all_results: List of evaluation results
        difficulty_data: Difficulty assessment data

    Returns:
        Dict with metrics aggregated by difficulty level
    """
    difficulty_categories = ['easy', 'medium', 'hard']

    # Initialize aggregation structure
    by_difficulty = {}
    for diff in difficulty_categories:
        by_difficulty[diff] = {
            method: {
                'total': 0,
                'navigation_errors': [],
                'success_count': 0,
                'oracle_success_count': 0,
                'sdtw_values': [],
                'ndtw_values': []
            }
            for method in METHODS
        }

    # Aggregate results by difficulty
    for result in all_results:
        key = result['key']
        diff_info = difficulty_data.get(key, {})
        complexity = diff_info.get('complexity', {})
        difficulty = get_difficulty_category(complexity)

        if difficulty == 'unknown':
            continue

        for method in METHODS:
            method_result = result['methods'].get(method, {})
            if method_result.get('status') == 'MISSING':
                continue

            agg = by_difficulty[difficulty][method]
            agg['total'] += 1

            nav_metrics = method_result.get('nav_metrics', {})
            if nav_metrics:
                ne = nav_metrics.get('navigation_error', float('inf'))
                if ne != float('inf'):
                    agg['navigation_errors'].append(ne)

                if nav_metrics.get('success', False):
                    agg['success_count'] += 1
                if nav_metrics.get('oracle_success', False):
                    agg['oracle_success_count'] += 1

                agg['sdtw_values'].append(nav_metrics.get('sdtw', 0.0))
                agg['ndtw_values'].append(nav_metrics.get('ndtw', 1.0))

    # Calculate final metrics
    for diff in difficulty_categories:
        for method in METHODS:
            agg = by_difficulty[diff][method]

            if agg['navigation_errors']:
                agg['avg_navigation_error'] = np.mean(agg['navigation_errors'])
            else:
                agg['avg_navigation_error'] = float('inf')

            if agg['total'] > 0:
                agg['success_rate'] = agg['success_count'] / agg['total']
                agg['oracle_success_rate'] = agg['oracle_success_count'] / agg['total']
            else:
                agg['success_rate'] = 0.0
                agg['oracle_success_rate'] = 0.0

            if agg['sdtw_values']:
                agg['avg_sdtw'] = np.mean(agg['sdtw_values'])
            else:
                agg['avg_sdtw'] = 0.0

            if agg['ndtw_values']:
                agg['avg_ndtw'] = np.mean(agg['ndtw_values'])
            else:
                agg['avg_ndtw'] = 0.0

    return by_difficulty


def print_difficulty_metrics(by_difficulty: Dict):
    """Print navigation metrics broken down by difficulty level."""
    print("\n" + "=" * 80)
    print(f"NAVIGATION METRICS BY DIFFICULTY")
    print("=" * 80)

    for difficulty in ['easy', 'medium', 'hard']:
        print(f"\n--- {difficulty.upper()} ---")
        print(f"{'Method':<12} {'Count':<8} {'NE (m)':<10} {'SR':<10} {'OSR':<10} {'SDTW':<10}")
        print("-" * 60)

        for method in METHODS:
            metrics = by_difficulty[difficulty][method]
            total = metrics['total']
            ne = metrics.get('avg_navigation_error', float('inf'))
            sr = metrics.get('success_rate', 0)
            osr = metrics.get('oracle_success_rate', 0)
            sdtw = metrics.get('avg_sdtw', 0)

            ne_str = f"{ne:.1f}" if ne != float('inf') else "inf"
            print(f"{method:<12} {total:<8} {ne_str:<10} {sr * 100:.1f}%{'':<5} {osr * 100:.1f}%{'':<5} {sdtw:.3f}")


def plot_difficulty_metrics(by_difficulty: Dict, output_dir: str):
    """Create bar chart comparing navigation metrics across difficulty levels."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = {'easy': '#2ecc71', 'medium': '#f39c12', 'hard': '#e74c3c'}
    difficulties = ['easy', 'medium', 'hard']

    x = np.arange(len(METHODS))
    width = 0.25

    # Navigation Error
    ax = axes[0, 0]
    for i, diff in enumerate(difficulties):
        values = []
        for method in METHODS:
            ne = by_difficulty[diff][method].get('avg_navigation_error', float('inf'))
            values.append(ne if ne != float('inf') else 0)
        ax.bar(x + i * width, values, width, label=diff, color=colors[diff])
    ax.set_ylabel('Navigation Error (meters)')
    ax.set_title('Navigation Error by Difficulty')
    ax.set_xticks(x + width)
    ax.set_xticklabels(METHODS)
    ax.legend()

    # Success Rate
    ax = axes[0, 1]
    for i, diff in enumerate(difficulties):
        values = [by_difficulty[diff][m].get('success_rate', 0) * 100 for m in METHODS]
        ax.bar(x + i * width, values, width, label=diff, color=colors[diff])
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate by Difficulty')
    ax.set_xticks(x + width)
    ax.set_xticklabels(METHODS)
    ax.set_ylim(0, 100)
    ax.legend()

    # Oracle Success Rate
    ax = axes[1, 0]
    for i, diff in enumerate(difficulties):
        values = [by_difficulty[diff][m].get('oracle_success_rate', 0) * 100 for m in METHODS]
        ax.bar(x + i * width, values, width, label=diff, color=colors[diff])
    ax.set_ylabel('Oracle Success Rate (%)')
    ax.set_title('Oracle Success Rate by Difficulty')
    ax.set_xticks(x + width)
    ax.set_xticklabels(METHODS)
    ax.set_ylim(0, 100)
    ax.legend()

    # SDTW
    ax = axes[1, 1]
    for i, diff in enumerate(difficulties):
        values = [by_difficulty[diff][m].get('avg_sdtw', 0) for m in METHODS]
        ax.bar(x + i * width, values, width, label=diff, color=colors[diff])
    ax.set_ylabel('SDTW Score')
    ax.set_title('SDTW by Difficulty')
    ax.set_xticks(x + width)
    ax.set_xticklabels(METHODS)
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'navigation_metrics_by_difficulty.png'), dpi=150)
    plt.close()
    print(f"Saved: navigation_metrics_by_difficulty.png")


# =============================================================================
# CORRELATION ANALYSIS (Human Annotations vs Metrics)
# =============================================================================

def calculate_correlations(
        all_results: List[Dict],
        difficulty_data: Dict[str, Dict],
) -> Dict:
    """
    Calculate Pearson and Spearman correlations between human annotations
    and navigation metrics (SR, OSR, NE, SDTW, nDTW).

    Args:
        all_results: List of evaluation results
        difficulty_data: Difficulty assessment data with human_annotation field

    Returns:
        Dict with correlation results per method, per difficulty level, and overall
    """
    difficulty_categories = ['easy', 'medium', 'hard']

    # Initialize structure to collect paired data
    paired_data = {
        'overall': {method: {'human': [], 'sr': [], 'osr': [], 'ne': [], 'sdtw': [], 'ndtw': []}
                    for method in METHODS},
    }
    for diff in difficulty_categories:
        paired_data[diff] = {method: {'human': [], 'sr': [], 'osr': [], 'ne': [], 'sdtw': [], 'ndtw': []}
                             for method in METHODS}

    # Collect paired data for each instruction
    for result in all_results:
        key = result['key']
        diff_info = difficulty_data.get(key, {})

        # Get human annotation
        human_annotation_str = diff_info.get('human_annotation', None)
        if human_annotation_str is None:
            continue

        # Convert to binary (1 = correct, 0 = wrong)
        try:
            human_annotation = int(human_annotation_str)
        except (ValueError, TypeError):
            continue

        # Get difficulty category
        complexity = diff_info.get('complexity', {})
        difficulty = get_difficulty_category(complexity)

        for method in METHODS:
            method_result = result['methods'].get(method, {})
            if method_result.get('status') == 'MISSING':
                continue

            nav_metrics = method_result.get('nav_metrics', {})
            if not nav_metrics:
                continue

            # Get metric values
            sr = 1 if nav_metrics.get('success', False) else 0
            osr = 1 if nav_metrics.get('oracle_success', False) else 0
            ne = nav_metrics.get('navigation_error', float('inf'))
            sdtw = nav_metrics.get('sdtw', 0.0)
            ndtw = nav_metrics.get('ndtw', 1.0)

            # Skip if NE is invalid
            if ne == float('inf'):
                ne = 1000  # Use large value as proxy for very wrong predictions

            # Add to overall
            paired_data['overall'][method]['human'].append(human_annotation)
            paired_data['overall'][method]['sr'].append(sr)
            paired_data['overall'][method]['osr'].append(osr)
            paired_data['overall'][method]['ne'].append(ne)
            paired_data['overall'][method]['sdtw'].append(sdtw)
            paired_data['overall'][method]['ndtw'].append(ndtw)

            # Add to difficulty-specific
            if difficulty in difficulty_categories:
                paired_data[difficulty][method]['human'].append(human_annotation)
                paired_data[difficulty][method]['sr'].append(sr)
                paired_data[difficulty][method]['osr'].append(osr)
                paired_data[difficulty][method]['ne'].append(ne)
                paired_data[difficulty][method]['sdtw'].append(sdtw)
                paired_data[difficulty][method]['ndtw'].append(ndtw)

    # Calculate correlations
    correlations = {}
    categories = ['overall'] + difficulty_categories
    metrics_list = ['sr', 'osr', 'ne', 'sdtw', 'ndtw']

    for category in categories:
        correlations[category] = {}
        for method in METHODS:
            correlations[category][method] = {}
            data = paired_data[category][method]
            human = np.array(data['human'])

            if len(human) < 3:
                # Not enough data points for correlation
                for metric in metrics_list:
                    correlations[category][method][metric] = {
                        'pearson_r': np.nan,
                        'pearson_p': np.nan,
                        'spearman_r': np.nan,
                        'spearman_p': np.nan,
                        'n': len(human)
                    }
                continue

            for metric in metrics_list:
                metric_values = np.array(data[metric])

                # For NE, we expect negative correlation (lower NE = better = 1)
                # So we'll report as-is, but note this in output

                # Check for constant arrays (no variance)
                if np.std(human) == 0 or np.std(metric_values) == 0:
                    correlations[category][method][metric] = {
                        'pearson_r': np.nan,
                        'pearson_p': np.nan,
                        'spearman_r': np.nan,
                        'spearman_p': np.nan,
                        'n': len(human)
                    }
                    continue

                # Calculate Pearson correlation
                try:
                    pearson_r, pearson_p = stats.pearsonr(human, metric_values)
                except Exception:
                    pearson_r, pearson_p = np.nan, np.nan

                # Calculate Spearman correlation
                try:
                    spearman_r, spearman_p = stats.spearmanr(human, metric_values)
                except Exception:
                    spearman_r, spearman_p = np.nan, np.nan

                correlations[category][method][metric] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p,
                    'n': len(human)
                }

    return correlations


def print_correlation_results(correlations: Dict):
    """Print correlation results in a formatted table."""
    metrics_list = ['sr', 'osr', 'ne', 'sdtw', 'ndtw']
    metric_names = {
        'sr': 'SR (Success Rate)',
        'osr': 'OSR (Oracle Success)',
        'ne': 'NE (Nav Error)*',
        'sdtw': 'SDTW',
        'ndtw': 'nDTW'
    }

    print("\n" + "=" * 100)
    print("CORRELATION ANALYSIS: Human Annotations vs Navigation Metrics")
    print("=" * 100)
    print("* Note: NE (Navigation Error) should have NEGATIVE correlation")
    print("  (lower NE = correct navigation = human annotation of 1)")

    for category in ['overall', 'easy', 'medium', 'hard']:
        print(f"\n{'=' * 50}")
        print(f"  {category.upper()}")
        print(f"{'=' * 50}")

        if category not in correlations:
            print("  No data available")
            continue

        for method in METHODS:
            print(f"\n  --- {method.upper()} ---")
            method_corr = correlations[category].get(method, {})

            if not method_corr:
                print("    No correlation data")
                continue

            print(f"  {'Metric':<25} {'n':<6} {'Pearson r':<12} {'p-value':<12} {'Spearman r':<12} {'p-value':<12}")
            print("  " + "-" * 79)

            for metric in metrics_list:
                corr_data = method_corr.get(metric, {})
                n = corr_data.get('n', 0)
                pr = corr_data.get('pearson_r', np.nan)
                pp = corr_data.get('pearson_p', np.nan)
                sr = corr_data.get('spearman_r', np.nan)
                sp = corr_data.get('spearman_p', np.nan)

                pr_str = f"{pr:.4f}" if not np.isnan(pr) else "N/A"
                pp_str = f"{pp:.4f}" if not np.isnan(pp) else "N/A"
                sr_str = f"{sr:.4f}" if not np.isnan(sr) else "N/A"
                sp_str = f"{sp:.4f}" if not np.isnan(sp) else "N/A"

                # Add significance markers
                if not np.isnan(pp) and pp < 0.05:
                    pr_str += "*"
                if not np.isnan(pp) and pp < 0.01:
                    pr_str += "*"
                if not np.isnan(sp) and sp < 0.05:
                    sr_str += "*"
                if not np.isnan(sp) and sp < 0.01:
                    sr_str += "*"

                print(f"  {metric_names[metric]:<25} {n:<6} {pr_str:<12} {pp_str:<12} {sr_str:<12} {sp_str:<12}")

    print("\n  * = p < 0.05, ** = p < 0.01")


def plot_correlation_heatmap(correlations: Dict, output_dir: str):
    """Create heatmap visualization of correlations."""
    metrics_list = ['sr', 'osr', 'ne', 'sdtw', 'ndtw']
    metric_labels = ['SR', 'OSR', 'NE', 'SDTW', 'nDTW']
    categories = ['overall', 'easy', 'medium', 'hard']

    # Create figure with subplots for each category
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Top row: Pearson correlations
    # Bottom row: Spearman correlations

    for col_idx, category in enumerate(categories):
        for row_idx, corr_type in enumerate(['pearson', 'spearman']):
            ax = axes[row_idx, col_idx]

            # Build correlation matrix (methods x metrics)
            corr_matrix = np.zeros((len(METHODS), len(metrics_list)))
            mask = np.zeros((len(METHODS), len(metrics_list)), dtype=bool)

            for i, method in enumerate(METHODS):
                for j, metric in enumerate(metrics_list):
                    corr_data = correlations.get(category, {}).get(method, {}).get(metric, {})
                    r = corr_data.get(f'{corr_type}_r', np.nan)
                    if np.isnan(r):
                        corr_matrix[i, j] = 0
                        mask[i, j] = True  # Mark as missing data
                    else:
                        corr_matrix[i, j] = r

            # Create masked array for proper NaN handling
            masked_matrix = np.ma.masked_array(corr_matrix, mask)

            # Create heatmap with masked array
            cmap = plt.cm.RdYlGn.copy()
            cmap.set_bad(color='lightgray')  # Color for masked (NaN) values
            im = ax.imshow(masked_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

            # Set ticks
            ax.set_xticks(np.arange(len(metrics_list)))
            ax.set_yticks(np.arange(len(METHODS)))
            ax.set_xticklabels(metric_labels)
            ax.set_yticklabels(METHODS)

            # Add correlation values as text
            for i in range(len(METHODS)):
                for j in range(len(metrics_list)):
                    if mask[i, j]:
                        ax.text(j, i, 'N/A', ha='center', va='center',
                                color='gray', fontsize=8)
                    else:
                        val = corr_matrix[i, j]
                        text_color = 'white' if abs(val) > 0.5 else 'black'
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                                color=text_color, fontsize=8)

            title_type = 'Pearson' if corr_type == 'pearson' else 'Spearman'
            ax.set_title(f'{title_type} - {category.upper()}')

    # Add colorbar
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04,
                 label='Correlation Coefficient')

    plt.suptitle('Human Annotation vs Metric Correlations\n(Note: NE should show negative correlation for good performance. Gray=N/A)',
                 fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: correlation_heatmap.png")


def plot_correlation_bars(correlations: Dict, output_dir: str):
    """Create bar chart visualization of correlations for each method."""
    metrics_list = ['sr', 'osr', 'ne', 'sdtw', 'ndtw']
    metric_labels = ['SR', 'OSR', 'NE', 'SDTW', 'nDTW']
    categories = ['overall', 'easy', 'medium', 'hard']
    category_colors = {'overall': '#3498db', 'easy': '#2ecc71', 'medium': '#f39c12', 'hard': '#e74c3c'}

    # Create figure: one subplot per method, showing Spearman correlations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    x = np.arange(len(metrics_list))
    width = 0.2

    for method_idx, method in enumerate(METHODS):
        ax = axes[method_idx]

        for cat_idx, category in enumerate(categories):
            values = []
            for metric in metrics_list:
                corr_data = correlations.get(category, {}).get(method, {}).get(metric, {})
                r = corr_data.get('spearman_r', np.nan)
                values.append(r if not np.isnan(r) else 0)

            offset = (cat_idx - 1.5) * width
            bars = ax.bar(x + offset, values, width, label=category, color=category_colors[category])

        ax.set_ylabel('Spearman Correlation')
        ax.set_title(f'{method.upper()} - Spearman Correlation with Human Annotation')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.legend(loc='upper right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_ylim(-1, 1)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_bars.png'), dpi=150)
    plt.close()
    print(f"Saved: correlation_bars.png")


def plot_scatter_correlations(
        all_results: List[Dict],
        difficulty_data: Dict[str, Dict],
        output_dir: str
):
    """Create scatter plots showing relationship between human annotation and metrics."""
    difficulty_categories = ['easy', 'medium', 'hard']
    diff_colors = {'easy': '#2ecc71', 'medium': '#f39c12', 'hard': '#e74c3c'}

    # Collect data
    scatter_data = {method: {'human': [], 'sr': [], 'osr': [], 'sdtw': [], 'difficulty': []}
                    for method in METHODS}

    for result in all_results:
        key = result['key']
        diff_info = difficulty_data.get(key, {})
        human_annotation_str = diff_info.get('human_annotation', None)

        if human_annotation_str is None:
            continue

        try:
            human_annotation = int(human_annotation_str)
        except (ValueError, TypeError):
            continue

        complexity = diff_info.get('complexity', {})
        difficulty = get_difficulty_category(complexity)

        for method in METHODS:
            method_result = result['methods'].get(method, {})
            if method_result.get('status') == 'MISSING':
                continue

            nav_metrics = method_result.get('nav_metrics', {})
            if not nav_metrics:
                continue

            sr = 1 if nav_metrics.get('success', False) else 0
            osr = 1 if nav_metrics.get('oracle_success', False) else 0
            sdtw = nav_metrics.get('sdtw', 0.0)

            scatter_data[method]['human'].append(human_annotation)
            scatter_data[method]['sr'].append(sr)
            scatter_data[method]['osr'].append(osr)
            scatter_data[method]['sdtw'].append(sdtw)
            scatter_data[method]['difficulty'].append(difficulty)

    # Create figure with confusion-matrix style plots for SR
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for method_idx, method in enumerate(METHODS):
        data = scatter_data[method]
        human = np.array(data['human'])
        sr = np.array(data['sr'])
        osr = np.array(data['osr'])
        difficulties = data['difficulty']

        # Top row: SR confusion matrix style
        ax = axes[0, method_idx]

        # Count occurrences for each (human, sr) combination
        for diff in difficulty_categories:
            diff_mask = np.array([d == diff for d in difficulties])
            if not any(diff_mask):
                continue

            h_diff = human[diff_mask]
            sr_diff = sr[diff_mask]

            # Add jitter for visualization
            jitter_x = np.random.uniform(-0.15, 0.15, len(h_diff))
            jitter_y = np.random.uniform(-0.15, 0.15, len(sr_diff))

            ax.scatter(h_diff + jitter_x, sr_diff + jitter_y,
                       alpha=0.5, s=30, c=diff_colors[diff], label=diff)

        ax.set_xlabel('Human Annotation')
        ax.set_ylabel('SR (Success Rate)')
        ax.set_title(f'{method.upper()} - SR')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Wrong (0)', 'Correct (1)'])
        ax.set_yticklabels(['Fail (0)', 'Success (1)'])
        if method_idx == 0:
            ax.legend(loc='upper left', fontsize=8)

        # Bottom row: OSR confusion matrix style
        ax = axes[1, method_idx]

        for diff in difficulty_categories:
            diff_mask = np.array([d == diff for d in difficulties])
            if not any(diff_mask):
                continue

            h_diff = human[diff_mask]
            osr_diff = osr[diff_mask]

            jitter_x = np.random.uniform(-0.15, 0.15, len(h_diff))
            jitter_y = np.random.uniform(-0.15, 0.15, len(osr_diff))

            ax.scatter(h_diff + jitter_x, osr_diff + jitter_y,
                       alpha=0.5, s=30, c=diff_colors[diff], label=diff)

        ax.set_xlabel('Human Annotation')
        ax.set_ylabel('OSR (Oracle Success Rate)')
        ax.set_title(f'{method.upper()} - OSR')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Wrong (0)', 'Correct (1)'])
        ax.set_yticklabels(['Fail (0)', 'Success (1)'])

    plt.suptitle('Human Annotation vs Metrics by Difficulty Level', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'correlation_scatter.png'), dpi=150)
    plt.close()
    print(f"Saved: correlation_scatter.png")


def compute_agreement_stats(
        all_results: List[Dict],
        difficulty_data: Dict[str, Dict],
) -> Dict:
    """
    Compute agreement statistics between human annotations and metrics.

    This computes:
    - Accuracy: how often human and metric agree
    - Precision/Recall/F1 for each metric predicting human correctness

    Returns:
        Dict with agreement statistics
    """
    difficulty_categories = ['easy', 'medium', 'hard']

    agreement_stats = {
        'overall': {method: {} for method in METHODS}
    }
    for diff in difficulty_categories:
        agreement_stats[diff] = {method: {} for method in METHODS}

    # Collect paired data
    paired_data = {
        'overall': {method: {'human': [], 'sr': [], 'osr': []} for method in METHODS}
    }
    for diff in difficulty_categories:
        paired_data[diff] = {method: {'human': [], 'sr': [], 'osr': []} for method in METHODS}

    for result in all_results:
        key = result['key']
        diff_info = difficulty_data.get(key, {})
        human_annotation_str = diff_info.get('human_annotation', None)

        if human_annotation_str is None:
            continue

        try:
            human_annotation = int(human_annotation_str)
        except (ValueError, TypeError):
            continue

        complexity = diff_info.get('complexity', {})
        difficulty = get_difficulty_category(complexity)

        for method in METHODS:
            method_result = result['methods'].get(method, {})
            if method_result.get('status') == 'MISSING':
                continue

            nav_metrics = method_result.get('nav_metrics', {})
            if not nav_metrics:
                continue

            sr = 1 if nav_metrics.get('success', False) else 0
            osr = 1 if nav_metrics.get('oracle_success', False) else 0

            paired_data['overall'][method]['human'].append(human_annotation)
            paired_data['overall'][method]['sr'].append(sr)
            paired_data['overall'][method]['osr'].append(osr)

            if difficulty in difficulty_categories:
                paired_data[difficulty][method]['human'].append(human_annotation)
                paired_data[difficulty][method]['sr'].append(sr)
                paired_data[difficulty][method]['osr'].append(osr)

    # Calculate agreement for each category/method
    for category in ['overall'] + difficulty_categories:
        for method in METHODS:
            data = paired_data[category][method]
            human = np.array(data['human'])
            sr = np.array(data['sr'])
            osr = np.array(data['osr'])

            if len(human) == 0:
                agreement_stats[category][method] = {'n': 0}
                continue

            # SR agreement
            sr_accuracy = np.mean(human == sr)
            sr_tp = np.sum((human == 1) & (sr == 1))
            sr_fp = np.sum((human == 0) & (sr == 1))
            sr_fn = np.sum((human == 1) & (sr == 0))
            sr_tn = np.sum((human == 0) & (sr == 0))
            sr_precision = sr_tp / (sr_tp + sr_fp) if (sr_tp + sr_fp) > 0 else 0
            sr_recall = sr_tp / (sr_tp + sr_fn) if (sr_tp + sr_fn) > 0 else 0
            sr_f1 = 2 * sr_precision * sr_recall / (sr_precision + sr_recall) if (sr_precision + sr_recall) > 0 else 0

            # OSR agreement
            osr_accuracy = np.mean(human == osr)
            osr_tp = np.sum((human == 1) & (osr == 1))
            osr_fp = np.sum((human == 0) & (osr == 1))
            osr_fn = np.sum((human == 1) & (osr == 0))
            osr_tn = np.sum((human == 0) & (osr == 0))
            osr_precision = osr_tp / (osr_tp + osr_fp) if (osr_tp + osr_fp) > 0 else 0
            osr_recall = osr_tp / (osr_tp + osr_fn) if (osr_tp + osr_fn) > 0 else 0
            osr_f1 = 2 * osr_precision * osr_recall / (osr_precision + osr_recall) if (osr_precision + osr_recall) > 0 else 0

            agreement_stats[category][method] = {
                'n': len(human),
                'sr': {
                    'accuracy': sr_accuracy,
                    'precision': sr_precision,
                    'recall': sr_recall,
                    'f1': sr_f1,
                    'tp': int(sr_tp),
                    'fp': int(sr_fp),
                    'fn': int(sr_fn),
                    'tn': int(sr_tn)
                },
                'osr': {
                    'accuracy': osr_accuracy,
                    'precision': osr_precision,
                    'recall': osr_recall,
                    'f1': osr_f1,
                    'tp': int(osr_tp),
                    'fp': int(osr_fp),
                    'fn': int(osr_fn),
                    'tn': int(osr_tn)
                }
            }

    return agreement_stats


def print_agreement_stats(agreement_stats: Dict):
    """Print agreement statistics in formatted tables."""
    print("\n" + "=" * 100)
    print("AGREEMENT STATISTICS: Human Annotation vs SR/OSR")
    print("=" * 100)

    for category in ['overall', 'easy', 'medium', 'hard']:
        print(f"\n{'=' * 50}")
        print(f"  {category.upper()}")
        print(f"{'=' * 50}")

        print(f"\n  {'Method':<12} {'Metric':<6} {'n':<6} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'TP':<6} {'FP':<6} {'FN':<6} {'TN':<6}")
        print("  " + "-" * 90)

        for method in METHODS:
            stats = agreement_stats.get(category, {}).get(method, {})
            n = stats.get('n', 0)

            for metric_name in ['sr', 'osr']:
                metric_stats = stats.get(metric_name, {})
                acc = metric_stats.get('accuracy', 0)
                prec = metric_stats.get('precision', 0)
                rec = metric_stats.get('recall', 0)
                f1 = metric_stats.get('f1', 0)
                tp = metric_stats.get('tp', 0)
                fp = metric_stats.get('fp', 0)
                fn = metric_stats.get('fn', 0)
                tn = metric_stats.get('tn', 0)

                method_str = method if metric_name == 'sr' else ''
                print(f"  {method_str:<12} {metric_name.upper():<6} {n:<6} {acc:.3f}{'':<3} {prec:.3f}{'':<3} {rec:.3f}{'':<3} {f1:.3f}{'':<3} {tp:<6} {fp:<6} {fn:<6} {tn:<6}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate navigation predictions")
    parser.add_argument('--input', '-i', default='test_seen_divider.json',
                        help='Input file with predictions (default: test_seen.json)')
    parser.add_argument('--output', '-o', default='evaluation_results',
                        help='Output directory for results (default: evaluation_results)')
    parser.add_argument('--split-file', default='test_seen_200.json',
                        help='Data split file (default: test_seen_200.json)')
    parser.add_argument('--threshold', type=float, default=25.0,
                        help='Distance threshold in meters (default: 25)')
    parser.add_argument('--difficulty-file', default=None,
                        help='Optional JSON file with difficulty/complexity assessment (e.g., correctness_hardness.json)')

    args = parser.parse_args()

    global DISTANCE_THRESHOLD
    DISTANCE_THRESHOLD = args.threshold

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load predictions
    print(f"Loading predictions from {args.input}...")
    with open(args.input, 'r') as f:
        predictions = json.load(f)

    print(f"Loaded {len(predictions)} instructions")

    # Evaluate each instruction
    print("\nEvaluating instructions...")
    all_results = []

    for i, (key, instruction_data) in enumerate(predictions.items()):
        if (i + 1) % 20 == 0:
            print(f"  Processing {i + 1}/{len(predictions)}...")

        result = evaluate_single_instruction(key, instruction_data, args.split_file)
        if result:
            all_results.append(result)

    print(f"\nEvaluated {len(all_results)} instructions successfully")

    # Aggregate results
    print("\nAggregating results...")
    aggregated = aggregate_results(all_results)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nTotal Instructions: {aggregated['total_instructions']}")
    print(f"Primary Distance Threshold: {DISTANCE_THRESHOLD}m")

    print("\n--- Method Performance ---")
    for method in METHODS:
        metrics = aggregated['by_method'][method]
        print(f"\n{method.upper()}:")
        print(f"  Total evaluated: {metrics['total']}")
        print(f"  Completely wrong: {metrics['completely_wrong']}")
        print(f"  Avg Path Overlap: {metrics.get('avg_overlap_ratio', 0) * 100:.1f}%")
        print(f"  Avg F1 Score: {metrics.get('avg_f1', 0) * 100:.1f}%")
        print(f"  Avg Sequence Overlap: {metrics.get('avg_sequence_overlap', 0) * 100:.1f}%")
        avg_dist = metrics.get('avg_endpoint_distance', float('inf'))
        if avg_dist != float('inf'):
            print(f"  Avg Endpoint Distance: {avg_dist:.1f}m")
            percentiles = metrics.get('distance_percentiles', {})
            if percentiles:
                print(f"    Min: {percentiles.get('min', 0):.1f}m, Max: {percentiles.get('max', 0):.1f}m")
                print(
                    f"    25th: {percentiles.get('25th', 0):.1f}m, Median: {percentiles.get('50th', 0):.1f}m, 75th: {percentiles.get('75th', 0):.1f}m")

    # Print Navigation Metrics Summary (NE, SR, OSR, SDTW)
    print("\n" + "=" * 60)
    print("NAVIGATION METRICS (25m threshold)")
    print("=" * 60)
    print(f"\n{'Method':<12} {'NE (m)':<10} {'SR':<10} {'OSR':<10} {'SDTW':<10} {'nDTW':<10}")
    print("-" * 62)

    for method in METHODS:
        metrics = aggregated['by_method'][method]
        ne = metrics.get('avg_navigation_error', float('inf'))
        sr = metrics.get('success_rate', 0)
        osr = metrics.get('oracle_success_rate', 0)
        sdtw = metrics.get('avg_sdtw', 0)
        ndtw = metrics.get('avg_ndtw', 0)

        ne_str = f"{ne:.1f}" if ne != float('inf') else "inf"
        print(f"{method:<12} {ne_str:<10} {sr * 100:.1f}%{'':<5} {osr * 100:.1f}%{'':<5} {sdtw:.3f}{'':<5} {ndtw:.3f}")

    print("\n--- Accuracy at Different Thresholds ---")
    print(f"{'Method':<12}", end="")
    for t in DISTANCE_THRESHOLDS:
        print(f"{t}m".rjust(10), end="")
    print()
    print("-" * (12 + 10 * len(DISTANCE_THRESHOLDS)))

    for method in METHODS:
        metrics = aggregated['by_method'][method]
        accuracy_by_thresh = metrics.get('accuracy_by_threshold', {})
        print(f"{method:<12}", end="")
        for t in DISTANCE_THRESHOLDS:
            acc = accuracy_by_thresh.get(t, 0) * 100
            print(f"{acc:>9.1f}%", end="")
        print()

    print("\n--- Correctness by Number of Sub-goals (at primary threshold) ---")
    by_subgoals = aggregated['by_subgoals']
    if by_subgoals:
        for sg_count in sorted(by_subgoals.keys()):
            print(f"\n{sg_count} sub-goals:")
            for method in METHODS:
                count = len(by_subgoals[sg_count].get(method, []))
                print(f"  {method}: {count} correct")
    else:
        print("No correct predictions at primary threshold.")

    # Save detailed results
    results_file = os.path.join(args.output, 'detailed_results.json')
    with open(results_file, 'w') as f:
        # Convert default dicts and tuple keys to regular dicts for JSON serialization
        serializable_method_agreement = {}
        for threshold, agreements in aggregated['method_agreement'].items():
            if isinstance(agreements, dict):
                serializable_method_agreement[str(threshold)] = {
                    ','.join(k) if isinstance(k, tuple) else str(k): v
                    for k, v in agreements.items()
                }
            else:
                serializable_method_agreement[str(threshold)] = str(agreements)

        serializable_aggregated = {
            'total_instructions': aggregated['total_instructions'],
            'distance_threshold': DISTANCE_THRESHOLD,
            'distance_thresholds': DISTANCE_THRESHOLDS,
            'nav_success_threshold': NAV_SUCCESS_THRESHOLD,
            'by_method': aggregated['by_method'],
            'by_subgoals': {str(k): dict(v) for k, v in aggregated['by_subgoals'].items()},
            'method_agreement': serializable_method_agreement
        }
        json.dump(serializable_aggregated, f, indent=2, default=str)
    print(f"\nSaved detailed results to {results_file}")

    # Save individual results
    individual_file = os.path.join(args.output, 'individual_results.json')
    with open(individual_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved individual results to {individual_file}")

    # Generate plots
    print("\nGenerating visualizations...")
    plot_method_comparison(aggregated, args.output)
    plot_correctness_by_subgoals(aggregated, args.output, all_results)
    plot_endpoint_distance_distribution(aggregated, args.output)
    plot_overlap_metrics(aggregated, args.output)
    plot_accuracy_by_threshold(aggregated, args.output)
    plot_distance_boxplot(aggregated, args.output)
    plot_navigation_metrics(aggregated, args.output)

    # Generate UpSet plots at different thresholds
    for threshold in [25, 50, 100, 150]:
        plot_upset(aggregated, args.output, threshold=threshold)

    # Difficulty-based analysis (if difficulty file provided)
    if args.difficulty_file:
        print(f"\nLoading difficulty data from {args.difficulty_file}...")
        try:
            difficulty_data = load_difficulty_data(args.difficulty_file)
            print(f"Loaded difficulty data for {len(difficulty_data)} instructions")

            # Aggregate by difficulty
            by_difficulty = aggregate_by_difficulty(
                all_results,
                difficulty_data
            )

            # Print difficulty-based metrics
            print_difficulty_metrics(by_difficulty)

            # Generate difficulty-based visualization
            plot_difficulty_metrics(by_difficulty, args.output)

            # Save difficulty-based results
            difficulty_results_file = os.path.join(args.output, 'results_by_difficulty.json')
            with open(difficulty_results_file, 'w') as f:
                json.dump({
                    'by_difficulty': by_difficulty
                }, f, indent=2, default=str)
            print(f"Saved difficulty-based results to {difficulty_results_file}")

            # =================================================================
            # CORRELATION ANALYSIS (Human Annotations vs Metrics)
            # =================================================================
            print("\n" + "=" * 60)
            print("COMPUTING CORRELATION ANALYSIS")
            print("=" * 60)

            # Calculate correlations between human annotations and metrics
            correlations = calculate_correlations(all_results, difficulty_data)

            # Print correlation results
            print_correlation_results(correlations)

            # Compute and print agreement statistics
            agreement_stats = compute_agreement_stats(all_results, difficulty_data)
            print_agreement_stats(agreement_stats)

            # Generate correlation visualizations
            print("\nGenerating correlation visualizations...")
            plot_correlation_heatmap(correlations, args.output)
            plot_correlation_bars(correlations, args.output)
            plot_scatter_correlations(all_results, difficulty_data, args.output)

            # Save correlation results
            correlation_results_file = os.path.join(args.output, 'correlation_results.json')
            with open(correlation_results_file, 'w') as f:
                # Convert numpy values to Python types for JSON serialization
                serializable_correlations = {}
                for category, methods_data in correlations.items():
                    serializable_correlations[category] = {}
                    for method, metrics_data in methods_data.items():
                        serializable_correlations[category][method] = {}
                        for metric, corr_vals in metrics_data.items():
                            serializable_correlations[category][method][metric] = {
                                k: (float(v) if isinstance(v, (np.floating, float)) and not np.isnan(v) else
                                    (None if isinstance(v, (np.floating, float)) and np.isnan(v) else v))
                                for k, v in corr_vals.items()
                            }
                json.dump({
                    'correlations': serializable_correlations,
                    'agreement_stats': agreement_stats
                }, f, indent=2, default=str)
            print(f"Saved correlation results to {correlation_results_file}")

        except FileNotFoundError:
            print(f"Warning: Difficulty file '{args.difficulty_file}' not found. Skipping difficulty analysis.")
        except Exception as e:
            print(f"Error loading difficulty data: {e}")

    print(f"\nAll results saved to {args.output}/")
    print("Done!")


if __name__ == "__main__":
    main()