import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import os
from collections import Counter

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Assuming this script is running in src/topic_modelling/hdbscan/ (same as create_dtm.py)
base_path = "../../../bertopic/dtm"
model_path = os.path.join(base_path, "models")
probs_file = os.path.join(model_path, "probabilities.npy")
topic_info_file = os.path.join(model_path, "topic_info_llm.csv")

# Output directory as requested
vis_path = os.path.abspath(os.path.join(BASE_DIR, "../../../visualisations/topic_model/modell_visualisierungen"))

# Create visualization directory if it doesn't exist
os.makedirs(vis_path, exist_ok=True)

def main():
    print("Starting network visualization generation...")
    
    # 1. Load Data
    # Check if files exist
    if not os.path.exists(probs_file):
        raise FileNotFoundError(f"Probabilities file not found at {probs_file}")
    if not os.path.exists(topic_info_file):
        raise FileNotFoundError(f"Topic info file not found at {topic_info_file}")
    
    print(f"Loading probabilities from {probs_file}...")
    probs = np.load(probs_file)
    print(f"Probabilities shape: {probs.shape}")
    
    print(f"Loading topic info from {topic_info_file}...")
    df_topics = pd.read_csv(topic_info_file)
    
    # Filter out outlier topic -1 and sort by Topic ID
    # We assume that the columns in probs correspond to Topic IDs 0, 1, 2... 
    # (since BERTopic usually aligns probs columns with topics excluding -1)
    df_topics_clean = df_topics[df_topics['Topic'] != -1].copy()
    df_topics_clean = df_topics_clean.sort_values('Topic').reset_index(drop=True)
    
    n_topics_probs = probs.shape[1]
    n_topics_df = len(df_topics_clean)
    
    print(f"Number of probability columns: {n_topics_probs}")
    print(f"Number of valid topics (excluding -1): {n_topics_df}")
    
    if n_topics_probs != n_topics_df:
        print("WARNING: Mismatch between probability columns and number of topics!")
        print("Visualization might be inaccurate if mapping is incorrect.")
        # We process anyway, assuming the first N match.
    
    # 2. Graph Construction
    # Node logic: Size based on 'Count', Label from 'LLM_Label'
    print("Building graph nodes...")
    G = nx.Graph()
    
    node_metadata = {}
    for _, row in df_topics_clean.iterrows():
        t_id = int(row['Topic'])
        # If topics don't match 0..N-1 indices, we might have an issue with probs mapping.
        # But BERTopic usually produces contiguous topic IDs from 0 to N-1 for the main clusters.
        
        count = row['Count']
        
        # Determine label: Check Custom_Label first, then LLM_Label, then fallback
        base_label = None
        
        # Check Custom_Label
        if 'Custom_Label' in row and pd.notna(row['Custom_Label']) and str(row['Custom_Label']).strip() != '':
            base_label = str(row['Custom_Label']).strip()
        # Check LLM_Label
        elif 'LLM_Label' in row and pd.notna(row['LLM_Label']) and str(row['LLM_Label']).strip() != '':
            base_label = str(row['LLM_Label']).strip()
        else:
            # Fallback
            base_label = row['Name'] if 'Name' in row and pd.notna(row['Name']) else f"Topic {t_id}"
            
        label = f"T{t_id}:{base_label}"
            
        node_metadata[t_id] = {'count': count, 'label': str(label)}
        G.add_node(t_id, size=count, label=str(label))
        
    # Edge logic: Top 2 probabilities per document
    print("Calculating edges from top-2 probabilities per document...")
    
    # Get indices of top 2 probabilities for each document
    # argsort sorts ascending, so we take the last 2 columns
    top_indices = np.argsort(probs, axis=1)[:, -2:]
    
    edge_counter = Counter()
    
    # Iterate through documents to count co-occurrences
    # row contains indices of top 2 topics
    for row in top_indices:
        # Create unique pairs from the top 2
        # If a document has < 2 topics (unlikely for soft clustering), argsort still returns 2 unless columns < 2
        topics_in_doc = sorted(row)
        
        # Add edges for all combinations (clique of size 2)
        # Pairs: (0,1)
        for i in range(len(topics_in_doc)):
            for j in range(i + 1, len(topics_in_doc)):
                u, v = topics_in_doc[i], topics_in_doc[j]
                
                # Verify nodes exist (handle potential mismatch or outlier index if any)
                if u in node_metadata and v in node_metadata:
                    edge_counter[(u, v)] += 1
    
    print(f"Found {len(edge_counter)} unique edges between topics.")
    
    # Add weighted edges to graph
    for (u, v), weight in edge_counter.items():
        G.add_edge(u, v, weight=weight)
        
    # --- Gephi Export ---
    print("Exporting data for Gephi...")
    
    # 1. Nodes
    # Gephi needs: Id, Label, [Attributes...]
    gephi_nodes = []
    for t_id, data in node_metadata.items():
        gephi_nodes.append({
            'Id': t_id,
            'Label': data['label'],
            'Count': data['count']
        })
    df_gephi_nodes = pd.DataFrame(gephi_nodes)
    nodes_path = os.path.join(vis_path, "gephi_nodes.csv")
    df_gephi_nodes.to_csv(nodes_path, index=False)
    print(f"Gephi nodes saved to {nodes_path}")

    # 2. Edges
    # Gephi needs: Source, Target, Type, Weight
    gephi_edges = []
    for (u, v), weight in edge_counter.items():
        gephi_edges.append({
            'Source': u,
            'Target': v,
            'Type': 'Undirected',
            'Weight': weight
        })
    df_gephi_edges = pd.DataFrame(gephi_edges)
    edges_path = os.path.join(vis_path, "gephi_edges.csv")
    df_gephi_edges.to_csv(edges_path, index=False)
    print(f"Gephi edges saved to {edges_path}")
    # --------------------

    # 3. Visualization with Plotly
    print("Generating layout and plot...")
    
    # Compute layout (spring layout is good for force-directed graphs)
    # k controls the distance between nodes. 
    # Weighted edges pull nodes closer.
    # Increasing k to spread nodes out more, and iterations for "Fruchterman-Reingold" stability
    pos = nx.spring_layout(G, weight='weight', seed=42, k=0.5, iterations=10000)
    
    # Prepare Node Data
    node_x = []
    node_y = []
    node_text = []
    node_text_display = [] # Text to show on the node
    node_sizes = []
    node_colors = []
    node_labels_vis = []
    
    # Determine size scaling
    counts = [d['count'] for t_id, d in node_metadata.items()]
    max_count = max(counts) if counts else 1
    
    for t_id in G.nodes():
        x, y = pos[t_id]
        node_x.append(x)
        node_y.append(y)
        
        data = node_metadata[t_id]
        count = data['count']
        label = data['label']
        
        # Hover text
        wrapper_label = "<br>".join([label[i:i+50] for i in range(0, len(label), 50)]) # simple wrap
        node_text.append(f"<b>{wrapper_label}</b><br>Docs: {count}")
        
        # Display text (on node)
        # Shorten label for display
        clean_label = label.split("\n")[0] # Take first line if multiline
        if len(clean_label) > 20:
            short_label = clean_label[:17] + "..."
        else:
            short_label = clean_label
            
        display_text = f"<b>{short_label}</b>"
        node_text_display.append(display_text)
        
        # Node Size: Scale sqrt(count) to range [20, 100] (Increased size for labels)
        size_norm = np.sqrt(count) / np.sqrt(max_count)
        size_vis = 25 + 75 * size_norm
        node_sizes.append(size_vis)
        
        node_colors.append(count)
        node_labels_vis.append(label)

    # Create Edges Traces (Binning by width)
    edge_traces = []
    
    if len(edge_counter) > 0:
        weights = list(edge_counter.values())
        min_w, max_w = min(weights), max(weights)
        
        # Create bins for edge thickness
        # We'll use 5 distinct thickness levels
        n_bins = 5
        # Use log spacing or linear? Linear is probably safer for "thickness ~ occurrences"
        # But if distribution is power-law, log might be better. 
        # Let's stick to linear bins for now.
        bins = np.linspace(min_w, max_w, n_bins + 1)
        
        for i in range(n_bins):
            w_lower = bins[i]
            w_upper = bins[i+1]
            
            # Find edges in this bin
            # Include upper bound for the last bin
            if i == n_bins - 1:
                edges_in_bin = [(u,v) for (u,v), w in edge_counter.items() if w_lower <= w <= w_upper]
            else:
                edges_in_bin = [(u,v) for (u,v), w in edge_counter.items() if w_lower <= w < w_upper]
                
            if not edges_in_bin:
                continue
                
            x_edges = []
            y_edges = []
            
            for u, v in edges_in_bin:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                x_edges += [x0, x1, None]
                y_edges += [y0, y1, None]
                
            # Width scaling: map bin index to width
            # Indices 0..4 -> Widths e.g. 1..10
            # Width calculation based on bin center or index
            # width = 1 + (i / (n_bins-1)) * 9  -> range 1 to 10
            line_width = 1 + (i / max((n_bins - 1), 1)) * 6 # Max width 7
            
            # Opacity increases with weight too
            opacity = 0.3 + (i / max((n_bins - 1), 1)) * 0.5 # 0.3 to 0.8
            
            edge_traces.append(go.Scatter(
                x=x_edges,
                y=y_edges,
                line=dict(width=line_width, color='#888'),
                hoverinfo='none',
                mode='lines',
                opacity=opacity,
                showlegend=False
            ))
            
    # Create Node Trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text', # Add 'text' to show labels on plot
        text=node_text_display, # Display ID + short label
        textposition="middle center",
        textfont=dict(
            family="sans serif",
            size=10,
            color='black' # Simple contrast, better than auto for most Viridis colors in middle range
        ),
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            reversescale=True,
            color=node_colors,
            size=node_sizes,
            colorbar=dict(
                thickness=15,
                title='Document Count',
                xanchor='left'
            ),
            line=dict(
                width=2,
                color='white'
            )
        )
    )
    
    # Create Figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    fig.update_layout(
        title=dict(
            text='Topic Network (Edges: Co-occurrence in Top-2 Probabilities)',
            font=dict(size=20)
        ),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    
    output_path = os.path.join(vis_path, "topic_network.html")
    fig.write_html(output_path)
    print(f"Network visualization saved to {output_path}")

if __name__ == "__main__":
    main()
