map2seq Dataset v1.0


Date: May 7th 2021
Author: Raphael Schumann
Contact: rschuman@cl.uni-heidelberg.de
Download & Demo: map2seq.schumann.pub


Cite:
@inproceedings{schumann-riezler-2021-map2seq,
title= "Generating Landmark Navigation Instructions from Maps as a Graph-to-Text Problem",
author= "Raphael Schumann and Stefan Riezler",
year= "2021",
publisher= "Association for Computational Linguistics"}


Files:
-> splits/
--> *.json:
    Json files that contain the dataset instances.
    test_seen_200.json is a subset of test_seen.json and test_unseen_200.json is a subset of test_unseen.json.
    See paper Appendix for boundaries of seen and unseen splits.
    Format:
        -route: Route to be described
        --pano_path: Panorama Ids (see Touchdown dataset) along the route
        --osm_path: OSM Ids along the route. See osm/graph/nodes.txt
        --initial_heading: Initial direction the agent is facing at the start of the route
        -instructions: Navigation Instructions text
        -instructions_id: Unique ID for each instance in the dataset (not consecutive)

-> osm/
--> graph/
---> nodes.txt:
     Discretized street layout within the covered area of Manhattan.
     Format: OSM_ID,LAT,LNG
     OSM_IDs at intersections can be linked to OpenStreetMap. Nodes in between two intersections were created every 10m.
---> links.txt:
     Edges between street nodes.
     Format: OSM_ID,heading,OSM_ID
---> pois.txt:
     Point of Interests within the covered area of Manhattan.
     Format: POI_ID,LAT,LNG,tags
     POI_ID can be linked to OpenStreetMap.
---> poi_links.txt:
     Edges between street segments (OSM_ID) and points of interest.
     Format: OSM_ID,heading,LAT,LNG,POI_ID
     LAT, LNG in this file are different from the coordinates of the POI_ID in pois.txt. It describes the point on the POI's polygon closest to the street segment (OSM_ID).
--> map_tiles/
---> map_tiles.zip:
     Rendered map tile images without street names for the covered area of Manhattan. Can be displayed by e.g. leaflet
--> map2seq_new_york-171204.osm:
    Covered area of Manhattan in OSM format. This file was used to extract the street segments and points of interest.
    Some modifications were made to better align the Street View graph of the Touchdown dataset.
--> command.txt:
    osmconvert command to extract the osm file above.
