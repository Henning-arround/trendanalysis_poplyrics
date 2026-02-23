import os
import json
import pandas as pd
import numpy as np
from bertopic import BERTopic
import plotly.graph_objects as go

# --- Configuration & Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DTM_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../../bertopic/dtm"))
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../../data"))

MODEL_PATH = os.path.join(DTM_PATH, "models/bertopic_model_dtm")
PROBS_PATH = os.path.join(DTM_PATH, "models/probabilities.npy")
CSV_PATH = os.path.join(DTM_PATH, "visualizations/over_time/topics_over_time.csv")
JSON_PATH = os.path.join(DATA_PATH, "dataset_popmusic_v8.json")
OUTPUT_DIR = os.path.join(DTM_PATH, "visualizations/over_time")

# --- 1. Load Data ---
print("Loading Data...")

# Load CSV (Topics over Time)
df_tot = pd.read_csv(CSV_PATH)

# Calculate Relative Frequency
total_per_bin = df_tot.groupby('Timestamp')['Frequency'].transform('sum')
df_tot['Relative_Frequency'] = df_tot['Frequency'] / total_per_bin

# Load Documents (JSON)
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    all_docs = json.load(f)
df_docs = pd.DataFrame(all_docs)

# --- Filter Documents EXACTLY like create_dtm.py ---
# Step 1 & 2: Handle missing/empty values
df_docs['text'] = df_docs['text'].fillna('').str.strip()
df_docs['timestamp'] = df_docs['timestamp'].fillna('')

# Step 3: Filter out empty text or empty timestamp
df_docs = df_docs[(df_docs['text'] != '') & (df_docs['timestamp'] != '')]

# Step 4: Parse Year
def parse_year(date_str):
    try:
        return int(str(date_str)[:4])
    except:
        return None

df_docs['Year'] = df_docs['timestamp'].apply(parse_year)

# Step 5: Filter Year Range (1955-2024)
df_docs = df_docs[(df_docs['Year'] >= 1955) & (df_docs['Year'] <= 2024)].reset_index(drop=True)

print(f"Filtered Documents: {len(df_docs)}")

# Load Model and Probabilities
print("Loading Model and Probabilities...")
topic_model = BERTopic.load(MODEL_PATH)
probs = np.load(PROBS_PATH)

# --- 2. Align and Preprocess Documents ---
print("Aligning Documents...")

# 1. Assign Topics from Model (The "Truth")
if len(df_docs) != len(topic_model.topics_):
    print(f"CRITICAL WARNING: Doc count ({len(df_docs)}) != Model Topic count ({len(topic_model.topics_)}).")
    min_len = min(len(df_docs), len(topic_model.topics_))
    df_docs = df_docs.iloc[:min_len]
    df_docs['Topic'] = topic_model.topics_[:min_len]
else:
    df_docs['Topic'] = topic_model.topics_

# 2. Prepare for Lookup
df_docs['Original_Index'] = df_docs.index
df_docs_filtered = df_docs.copy()

# 3. Assign TimeBin
def assign_timebin(year):
    start_year = 1955 + ((year - 1955) // 5) * 5
    end_year = start_year + 4
    return f"{start_year}-{end_year}"

df_docs_filtered['Timestamp'] = df_docs_filtered['Year'].apply(assign_timebin)

# --- 3. Pre-calculate Click Data (Lookup Dictionary) ---
print("Generating Lookup Data for Interactivity...")
lookup_data = {}

# Create Topic Label Map for better display
# We use the first Global_Representation found for each topic in the CSV
topic_label_map = df_tot.drop_duplicates('Topic').set_index('Topic')['Global_Representation'].to_dict()

for _, row in df_tot.iterrows():
    topic = int(row['Topic'])
    timestamp = row['Timestamp']
    key = f"{topic}_{timestamp}"
    
    # Filter docs for this bin and topic
    subset = df_docs_filtered[
        (df_docs_filtered['Topic'] == topic) & 
        (df_docs_filtered['Timestamp'] == timestamp)
    ]
    
    if subset.empty:
        lookup_data[key] = {"top": [], "random": []}
        continue
        
    # Get indices to look up probs
    subset_indices = subset['Original_Index'].values
    
    # Prepare temp_df for sorting
    temp_df = subset.copy()

    if topic == -1:
        # Topic -1 (Outlier) does not have a probability column in 'probs'.
        # We cannot sort by "representativeness". We shuffle to get random examples.
        temp_df['Ref_Prob'] = 0
        temp_df = temp_df.sample(frac=1, random_state=42)
    else:
        # For regular topics, probs column index matches Topic ID directly (0 -> 0, 1 -> 1)
        col_idx = topic 
        
        # Safety check
        if col_idx < 0 or col_idx >= probs.shape[1]:
            col_idx = 0 
            
        topic_probs = probs[subset_indices, col_idx]
        temp_df['Ref_Prob'] = topic_probs
        # Sort by probability (Highest = Most Representative)
        temp_df = temp_df.sort_values('Ref_Prob', ascending=False)
    
    # Top 3
    top_3_df = temp_df.head(3)
    
    # Random 3 (from the rest)
    rest_df = temp_df.iloc[3:]
    if len(rest_df) >= 3:
        random_3_df = rest_df.sample(3)
    else:
        random_3_df = rest_df
        
    def format_docs(doc_df):
        doc_list = []
        for _, doc_row in doc_df.iterrows():
            # Get Top 3 Topics for this document based on probabilities
            orig_idx = doc_row['Original_Index']
            d_probs = probs[orig_idx]
            # Get top 3 indices
            top_indices = d_probs.argsort()[-3:][::-1]
            
            prob_str_parts = []
            for ti in top_indices:
                t_id = ti # Direct mapping
                p = d_probs[ti]
                t_label = topic_label_map.get(t_id, "Unknown")
                # Format: "T17: Label (0.04)" with HTML styling
                prob_str_parts.append(
                    f"<div style='margin-bottom:2px;'>"
                    f"<span style='font-weight:bold; color:#444;'>T{t_id}:</span> "
                    f"{t_label} "
                    f"<span style='color:#888; font-size:0.9em;'>({p:.2f})</span>"
                    f"</div>"
                )
            
            prob_str = "".join(prob_str_parts)
            
            # Escape quotes for JSON safety in HTML
            text_content = str(doc_row.get('text', '')).replace('"', '&quot;').replace("'", "&#39;")
            
            doc_list.append({
                "artist": str(doc_row.get('Künstler', 'Unknown')),
                "title": str(doc_row.get('Titel', 'Unknown')),
                "year": str(doc_row.get('Year', 'Unknown')),
                "probs": prob_str,
                "text": text_content
            })
        return doc_list

    lookup_data[key] = {
        "top": format_docs(top_3_df),
        "random": format_docs(random_3_df)
    }

# Serialize lookup data to JSON for embedding
lookup_json = json.dumps(lookup_data)

# --- 4. Create Plotly Figure ---
print("Creating Visualization...")
top_topics = df_tot.groupby('Topic')['Frequency'].sum().sort_values(ascending=False).head(7).index.tolist()

fig = go.Figure()
unique_topics = sorted(df_tot['Topic'].unique())

for topic in unique_topics:
    subset = df_tot[df_tot['Topic'] == topic]
    label = subset['Global_Representation'].iloc[0]
    visible = True if topic in top_topics else 'legendonly'
    
    fig.add_trace(go.Scatter(
        x=subset['Timestamp'],
        y=subset['Relative_Frequency'],
        mode='lines+markers',
        name=f"Topic {topic}: {label[:30]}...",
        visible=visible,
        # Store Topic and Timestamp in customdata for the click event
        customdata=np.stack((subset['Frequency'], subset['Words'], subset['Topic'], subset['Timestamp']), axis=-1),
        hovertemplate=(
            "<b>Topic %{customdata[2]}</b><br>" +
            "Time: %{customdata[3]}<br>" +
            "Words: %{customdata[1]}<br>" +
            "Abs. Freq: %{customdata[0]}<extra></extra>"
        )
    ))

fig.update_layout(
    title="Topics over Time (Relative Frequency) - Click points for details",
    xaxis_title="Time Period",
    yaxis_title="Relative Frequency",
    hovermode="closest",
    legend_title="Topics",
    template="plotly_white",
    height=600
)

# --- 5. Generate Custom HTML ---
# We embed the plot and the data + JS script in one file
plot_div = fig.to_html(full_html=False, include_plotlyjs='cdn')

html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Topics Over Time</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        #details-container {{ 
            margin-top: 20px; 
            padding: 20px; 
            border: 1px solid #ddd; 
            border-radius: 5px; 
            background-color: #f9f9f9;
            display: none;
        }}
        .doc-card {{
            background: white;
            border: 1px solid #eee;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .doc-header {{ font-size: 1.1em; margin-bottom: 5px; color: #2c3e50; }}
        .doc-probs {{ 
            font-size: 0.85em; 
            color: #555; 
            margin-top: 8px; 
            background: #f0f4f8; 
            padding: 8px; 
            border-radius: 4px; 
        }}
        h3 {{ margin-top: 0; color: #34495e; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
        .columns {{ display: flex; gap: 20px; }}
        .column {{ flex: 1; }}
        details {{ margin-top: 10px; }}
        summary {{ cursor: pointer; color: #007bff; font-weight: 500; }}
        summary:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>Interactive Topic Modelling Analysis</h1>
    <p>Click on a data point to view representative documents below.</p>
    
    <!-- Plotly Chart -->
    {plot_div}
    
    <!-- Details Section -->
    <div id="details-container">
        <h2 id="details-title">Details</h2>
        <div class="columns">
            <div class="column">
                <h3>Top 3 Representative Documents</h3>
                <div id="top-docs"></div>
            </div>
            <div class="column">
                <h3>3 Random Documents</h3>
                <div id="random-docs"></div>
            </div>
        </div>
    </div>

    <script>
        // Embedded Data
        const lookupData = {lookup_json};

        // Find the Plotly div (it's the first div with class 'plotly-graph-div')
        const plotDiv = document.getElementsByClassName('plotly-graph-div')[0];

        plotDiv.on('plotly_click', function(data){{
            const point = data.points[0];
            // customdata: [Freq, Words, Topic, Timestamp]
            const topic = point.customdata[2];
            const timestamp = point.customdata[3];
            const key = topic + "_" + timestamp;
            
            const info = lookupData[key];
            
            if (info) {{
                document.getElementById('details-container').style.display = 'block';
                document.getElementById('details-title').innerText = 'Details for Topic ' + topic + ' (' + timestamp + ')';
                
                function renderDocs(docs) {{
                    if (!docs || docs.length === 0) return '<p>No documents found.</p>';
                    return docs.map(d => `
                        <div class="doc-card">
                            <div class="doc-header"><strong>${{d.artist}} - ${{d.title}}</strong> (${{d.year}})</div>
                            <div class="doc-probs">
                                <div style="font-weight:bold; margin-bottom:4px;">Top Topics:</div>
                                ${{d.probs}}
                            </div>
                            <details>
                                <summary>Show Text</summary>
                                <div style="margin-top: 5px; white-space: pre-wrap; font-size: 0.9em; background: #fff; padding: 10px; border: 1px solid #eee; border-radius: 4px; max-height: 300px; overflow-y: auto; line-height: 1.4;">${{d.text}}</div>
                            </details>
                        </div>
                    `).join('');
                }}
                
                document.getElementById('top-docs').innerHTML = renderDocs(info.top);
                document.getElementById('random-docs').innerHTML = renderDocs(info.random);
            }}
        }});
    </script>
</body>
</html>
"""

os.makedirs(OUTPUT_DIR, exist_ok=True)
html_output_path = os.path.join(OUTPUT_DIR, "topics_over_time.html")
with open(html_output_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Interactive HTML visualization saved to: {html_output_path}")
