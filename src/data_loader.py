import json
import os


def get_data_by_instruction(instructions_id: int, split_file_name: str, base_path: str = './data/map2seq/',
                            neighbor_degrees: int = 2):
    """
    Given an instruction_id and a split file name, this function returns all the data related to that area.

    Args:
        instructions_id (int): The ID of the instruction to retrieve.
        split_file_name (str): The name of the split file (e.g., 'test_seen.json').
        base_path (str): The Map2seq dataset path.
        neighbor_degrees (int): The number of degrees of neighbors to include.

    Returns:
        dict: A dictionary containing the route, POIs, and graph information.
    """
    splits_path = os.path.join(base_path, 'splits', split_file_name)

    # Load the split file
    with open(splits_path, 'r') as f:
        split_data = json.load(f)

    # Find the instruction
    instruction_data = None
    for item in split_data:
        if item['instructions_id'] == instructions_id:
            instruction_data = item
            break

    if not instruction_data:
        return None

    route_osm_ids = set(instruction_data['route']['osm_path'])

    # Load all graph data
    nodes = {}
    with open(os.path.join(base_path, 'osm', 'graph', 'nodes.txt'), 'r') as f:
        for line in f:
            osm_id, lat, lng = line.strip().split(',')
            nodes[osm_id] = {'lat': float(lat), 'lng': float(lng)}

    links = []
    with open(os.path.join(base_path, 'osm', 'graph', 'links.txt'), 'r') as f:
        for line in f:
            id1, heading, id2 = line.strip().split(',')
            links.append({'source': id1, 'target': id2, 'heading': float(heading)})

    pois = {}
    with open(os.path.join(base_path, 'osm', 'graph', 'pois.txt'), 'r') as f:
        for line in f:
            poi_id, lat, lng, tags = line.strip().split(',', 3)
            pois[poi_id] = {'lat': float(lat), 'lng': float(lng), 'tags': tags}

    poi_links = []
    with open(os.path.join(base_path, 'osm', 'graph', 'poi_links.txt'), 'r') as f:
        for line in f:
            osm_id, heading, lat, lng, poi_id = line.strip().split(',')
            poi_links.append({
                'osm_id': osm_id,
                'poi_id': poi_id,
                'lat': float(lat),
                'lng': float(lng)
            })

    # Find all nodes and POIs related to the route by expanding N degrees
    all_area_nodes = set(route_osm_ids)
    newly_added_nodes = all_area_nodes

    for _ in range(neighbor_degrees):
        neighbors_of_newly_added = set()
        for link in links:
            if link['source'] in newly_added_nodes and link['target'] not in all_area_nodes:
                neighbors_of_newly_added.add(link['target'])
            if link['target'] in newly_added_nodes and link['source'] not in all_area_nodes:
                neighbors_of_newly_added.add(link['source'])

        if not neighbors_of_newly_added:
            break  # No new nodes to add

        all_area_nodes.update(neighbors_of_newly_added)
        newly_added_nodes = neighbors_of_newly_added

    # Filter links to only include those connecting area nodes
    area_links = [link for link in links if link['source'] in all_area_nodes and link['target'] in all_area_nodes]

    # Find POIs connected to the route nodes
    area_poi_ids = set()
    area_poi_links = []
    for poi_link in poi_links:
        if poi_link['osm_id'] in all_area_nodes:
            area_poi_ids.add(poi_link['poi_id'])
            area_poi_links.append(poi_link)

    # Get the data for the area
    final_nodes = {osm_id: nodes[osm_id] for osm_id in all_area_nodes if osm_id in nodes}
    final_pois = {poi_id: pois[poi_id] for poi_id in area_poi_ids if poi_id in pois}

    return {
        "instruction_data": instruction_data,
        "area_nodes": final_nodes,
        "area_links": area_links,
        "area_pois": final_pois,
        "area_poi_links": area_poi_links
    }
