"""
Simplified Evaluation: Returns instruction IDs where SR (Success Rate) fails.

An instruction fails SR when the navigation error (distance from predicted
endpoint to goal) exceeds NAV_SUCCESS_THRESHOLD (25 meters).

Usage:
    python evaluation_metrics.py --input test_seen.json --split-file test_seen.json
"""

import argparse
import json
import math
from collections import defaultdict, deque
from typing import Dict, List, Optional

from src.data_loader import get_data_by_instruction

NAV_SUCCESS_THRESHOLD = 25  # meters - threshold for navigation success (SR)
METHODS = ['json']


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate the Haversine distance between two points in meters."""
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
        graph[link['source']].append(link['target'])
    return graph


def bfs_path(graph: Dict[str, List[str]], start: str, end: str) -> Optional[List[str]]:
    """Find shortest path between two nodes using BFS."""
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
                queue.append(path + [neighbor])

    return None


def expand_predicted_path(
        predicted_waypoints: List[str],
        graph: Dict[str, List[str]],
        area_nodes: Dict[str, Dict]
) -> List[str]:
    """Expand predicted waypoints into full path using BFS between consecutive waypoints."""
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

    expanded = [unique_waypoints[0]]

    for i in range(len(unique_waypoints) - 1):
        start = unique_waypoints[i]
        end = unique_waypoints[i + 1]

        if start not in area_nodes or end not in area_nodes:
            continue

        path = bfs_path(graph, start, end)

        if path:
            expanded.extend(path[1:])
        elif end not in expanded:
            expanded.append(end)

    return expanded


def calculate_navigation_error(
        predicted_path: List[str],
        osm_path: List[str],
        area_nodes: Dict[str, Dict]
) -> float:
    """
    Calculate Navigation Error: distance between agent's stopping point and destination.

    Returns:
        Distance in meters, or inf if invalid.
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


def get_failed_sr_instructions(
        input_file: str,
        split_file: str = "test_seen_200.json",
        threshold: float = NAV_SUCCESS_THRESHOLD
) -> list[int]:
    """
    Get instruction IDs where SR (Success Rate) fails.

    Args:
        input_file: Path to JSON file with predictions
        split_file: Data split file name
        threshold: Success threshold in meters (default 25m)

    Returns:
        List of instruction IDs that failed SR
    """
    with open(input_file, 'r') as f:
        predictions = json.load(f)

    failed_ids = []

    for key, instruction_data in predictions.items():
        osm_path = instruction_data.get('osm_path', [])

        if not osm_path:
            failed_ids.append(key)
            continue

        # Load area data
        try:
            area_data = get_data_by_instruction(
                int(key),
                split_file,
                base_path='./data/map2seq/',
                neighbor_degrees=20
            )
        except Exception:
            failed_ids.append(key)
            continue

        if not area_data:
            failed_ids.append(key)
            continue

        area_nodes = area_data.get('area_nodes', {})
        area_links = area_data.get('area_links', [])
        graph = build_adjacency_graph(area_links)

        # Check each method
        for method in METHODS:
            method_data = instruction_data.get(method, {})

            if not method_data:
                failed_ids.append(key)
                break

            predicted_waypoints = method_data.get('predicated_path', [])
            predicted_waypoints = [p for p in predicted_waypoints if p is not None]

            if len(predicted_waypoints) <= 1:
                failed_ids.append(key)
                break

            expanded_path = expand_predicted_path(predicted_waypoints, graph, area_nodes)
            nav_error = calculate_navigation_error(expanded_path, osm_path, area_nodes)

            # SR fails if navigation error exceeds threshold
            if nav_error > threshold:
                failed_ids.append(key)
                break

    return list(map(int, failed_ids))


def main():
    parser = argparse.ArgumentParser(
        description="Get instruction IDs where SR fails (nav error > 25m)"
    )
    parser.add_argument('--input', '-i', default='results/test_seen.json',
                        help='Input file with predictions')
    parser.add_argument('--split-file', default='test_seen.json',
                        help='Data split file')
    parser.add_argument('--threshold', type=float, default=NAV_SUCCESS_THRESHOLD,
                        help=f'Distance threshold in meters (default: {NAV_SUCCESS_THRESHOLD})')

    args = parser.parse_args()

    print(f"Loading predictions from {args.input}...")
    failed_ids = get_failed_sr_instructions(args.input, args.split_file, args.threshold)

    print(f"\nInstruction IDs where SR fails (nav error > {args.threshold}m):")
    print(f"Total failed: {len(failed_ids)}")

    with open(f'{args.input.split(".")[0]}_failed.json', 'w') as f:
        f.write(json.dumps(failed_ids, indent=2))

    print(f'Saved Failed Ids at {args.input.split(".")[0]}_failed.json')

    return failed_ids


if __name__ == "__main__":
    main()