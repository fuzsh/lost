import json
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from rapidfuzz import fuzz


def visualize_area(area_data, focused_node_id=None, landmarks=None,output_name="test.png"):
    """
    Visualizes the area data using matplotlib with enhanced POI features.
    """
    if not area_data:
        print("No data to visualize.")
        return

    if focused_node_id is None:
        focused_node_id = []

    nodes = area_data['area_nodes']
    links = area_data['area_links']
    pois = area_data['area_pois']
    route_osm_ids = area_data['instruction_data']['route']['osm_path']

    fig, ax = plt.subplots(figsize=(12, 12))

    # 1. Plot all links (Streets) - Low Z-order to stay in background
    for link in links:
        node1 = nodes.get(link['source'])
        node2 = nodes.get(link['target'])
        if node1 and node2:
            ax.plot([node1['lng'], node2['lng']], [node1['lat'], node2['lat']],
                    'k-', linewidth=0.5, alpha=0.3, zorder=1)

    # 2. Plot all nodes
    node_x, node_y = [], []
    focus_x, focus_y = [], []

    for node_id, node_data in nodes.items():
        if node_id in focused_node_id:
            focus_x.append(node_data['lng'])
            focus_y.append(node_data['lat'])
        else:
            node_x.append(node_data['lng'])
            node_y.append(node_data['lat'])

    ax.plot(node_x, node_y, 'ko', markersize=2, alpha=0.5, zorder=2, label='Nodes')
    ax.plot(focus_x, focus_y, 'co', markersize=12, markeredgecolor='black', zorder=4, label='Predicated Nodes')

    # 3. Plot Route (Path)
    route_x = []
    route_y = []
    for osm_id in route_osm_ids:
        if osm_id in nodes:
            route_x.append(nodes[osm_id]['lng'])
            route_y.append(nodes[osm_id]['lat'])

    if route_x:
        # Draw the line
        ax.plot(route_x, route_y, 'b-', linewidth=3, alpha=0.7, zorder=3, label='Route')
        # Draw Start (Green) and End (Red)
        ax.plot(route_x[0], route_y[0], 'go', markersize=12, markeredgecolor='black', zorder=5, label='Start')
        ax.plot(route_x[-1], route_y[-1], 'ro', markersize=12, markeredgecolor='black', zorder=5, label='End')

    # 4. Plot POIs with Logic
    bus_x, bus_y = [], []
    other_x, other_y = [], []

    for poi_id, poi_data in pois.items():
        # Safely parse the tags string into a dictionary
        try:
            tags = json.loads(poi_data.get('tags', '{}'))
        except:
            tags = {}

        name = (
            f"{tags.get('name', '')} "
            f"{tags.get('amenity', '').replace('_', ' ')} "
            f"{tags.get('cuisine', '').replace('_', ' ')} "
            f"{tags.get('leisure', '').replace('_', ' ')} "
            f"{tags.get('tourism', '').replace('_', ' ')} "
            f"{tags.get('shop', '').replace('_', ' ')} "
            f"{tags.get('highway', '').replace('_', ' ')} "
        ).strip()

        for landmark in landmarks:
            if 'light' in landmark:
                landmark = 'traffic signals'
            score = fuzz.partial_ratio(landmark, name)
            if score > 70:
                bus_x.append(poi_data['lng'])
                bus_y.append(poi_data['lat'])

                # Add a text annotation for the bus stop
                ax.annotate(name,
                            (poi_data['lng'], poi_data['lat']),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center',
                            fontsize=8,
                            fontweight='bold',
                            color='darkgoldenrod',
                            zorder=6)
        else:
            other_x.append(poi_data['lng'])
            other_y.append(poi_data['lat'])

    # Plot Non-Bus POIs
    ax.plot(other_x, other_y, 'o', color='gray', markersize=4, alpha=0.5, zorder=3, label='Other POIs')

    # Plot Bus Stops (The enhanced part)
    # Using marker='s' (square) and Gold color with black edges to pop
    if bus_x:
        ax.plot(bus_x, bus_y, 's', color='gold', markersize=10,
                markeredgecolor='black', markeredgewidth=1.5, zorder=6, label='Mentioned POIs')

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Area Visualization (Instructions ID: {area_data.get('instruction_data', {}).get('instructions_id', 'N/A')})")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axis('equal')

    # Create a nice legend
    plt.legend(loc='upper right', framealpha=0.9, shadow=True)

    # plt.show()
    plt.savefig(output_name, dpi=300)
    plt.close()

# Run with your data (assuming 'data' is your dictionary variable)
# visualize_area(data, focused_node_id=focused_node_id)