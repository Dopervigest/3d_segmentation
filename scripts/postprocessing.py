import networkx as nx
import numpy as np
import trimesh


def postprocessing(mesh, path=None):
    # Usage:
    # a = postprocessing(mesh, path = './test.obj')

    original_colors = mesh.visual.vertex_colors

    colors = []
    for i in original_colors:
        if np.array_equal(i, [  0,   0, 255, 255]):
            colors.append('blue')
        elif np.array_equal(i, [  0,  255, 0, 255]):
            colors.append('green')
        elif np.array_equal(i, [  255,   0, 0, 255]):
            colors.append('red')
        elif np.array_equal(i, [  150,   150, 150, 255]):
            colors.append('gray')
        else:
            raise ValueError("Found a color that isn't red, green, blue nor gray while trying to read mesh!")

    # Create a graph from the mesh
    G = nx.Graph()
    
    # Add nodes and edges to the graph based on mesh connectivity
    for i, vertex in enumerate(mesh.vertices):
        G.add_node(i, color=colors[i])
    
    for edge in mesh.edges:
        G.add_edge(edge[0], edge[1])
    
    # Find connected components of each color
    components = {color: [] for color in set(colors)}
    for node, data in G.nodes(data=True):
        components[data['color']].append(node)

    def get_connected_components(nodes):
        subgraph = G.subgraph(nodes)
        return list(nx.connected_components(subgraph))
    
    # Find and process connected components
    for color, nodes in components.items():
        connected_components = get_connected_components(nodes)
        component_sizes = [(comp, len(comp)) for comp in connected_components]
        sorted_components = sorted(component_sizes, key=lambda x: x[1], reverse=True)
    
        # Assume the largest component is correctly classified, others may be misclassifications
        largest_component = sorted_components[0][0]
        for comp, size in sorted_components[1:]:
            # For each small component, check neighboring colors
            neighbor_colors = []
            for node in comp:
                neighbors = G.neighbors(node)
                neighbor_colors.extend([G.nodes[neighbor]['color'] for neighbor in neighbors if G.nodes[neighbor]['color'] != color])
    
            # Determine the most common neighboring color
            if neighbor_colors:
                most_common_color = max(set(neighbor_colors), key=neighbor_colors.count)
            else:
                most_common_color = color  # Fallback to the same color if no neighbors of different color
    
            # Recolor the small component
            for node in comp:
                G.nodes[node]['color'] = most_common_color
    
    
    # Update the mesh colors based on the corrected graph
    new_colors = [G.nodes[node]['color'] for node in G.nodes]






        
    colors = new_colors.copy()
    for i in range(len(new_colors)):
        if new_colors[i] == 'red':
            colors[i] = [255, 0, 0, 255]
        elif new_colors[i] == 'green':
            colors[i] = [0, 255, 0, 255]
        elif new_colors[i] == 'blue':
            colors[i] = [0, 0, 255, 255]
        elif new_colors[i] == 'gray': 
            colors[i] = [150, 150, 150, 255]
        else:
            raise ValueError("Found a color that isn't red, green, blue nor gray while trying to save mesh!")

    
        
    # Update mesh colors appropriately
    mesh.visual.vertex_colors = colors



    # Reality check 
    # Create a graph from the mesh
    G = nx.Graph()
    # Add nodes and edges to the graph based on mesh connectivity
    for i, vertex in enumerate(mesh.vertices):
        G.add_node(i, color=new_colors[i])
    for edge in mesh.edges:
        G.add_edge(edge[0], edge[1])

    # Find connected components of each color
    components = {color: [] for color in set(new_colors)}
    for node, data in G.nodes(data=True):
        components[data['color']].append(node)

    for color, nodes in components.items():
        connected_components = get_connected_components(nodes)
        component_sizes = [(comp, len(comp)) for comp in connected_components]
        if len(component_sizes) > 1:
            mesh = postprocessing(mesh, path=path)
            return mesh

    if path:
        file = trimesh.exchange.export.export_mesh(mesh, path, None)            
    return mesh
