import json
import pandas as pd
from bertopic import BERTopic
import os
import random
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import math

# Set plot style (TeX-like)
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'text.usetex': False
})

# Paths to model and data
MODEL_PATH = "../../../bertopic/dtm/models/bertopic_model_dtm"
DATA_PATH = "../../../data/dataset_popmusic_v8.json"

def get_output_dir(script_dir):
    out_dir = os.path.join(script_dir, "../../../visualisations/chart_analysis")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def load_topic_labels(script_dir):
    labels_path = os.path.join(script_dir, "../../../bertopic/dtm/models/topic_info_llm.csv")
    if not os.path.exists(labels_path):
        print(f"Warning: Labels file not found at {labels_path}")
        return {}
    
    try:
        df_labels = pd.read_csv(labels_path)
        label_map = {}
        for _, row in df_labels.iterrows():
            topic_id = row['Topic']
            custom = row.get('Custom_Label')
            llm = row.get('LLM_Label')
            
            label = f"Topic {topic_id}"
            if pd.notna(custom) and str(custom).strip() != "":
                label = custom
            elif pd.notna(llm) and str(llm).strip() != "":
                label = llm
            
            label_map[topic_id] = label
        return label_map
    except Exception as e:
        print(f"Error loading labels: {e}")
        return {}

def create_trajectory_plot(df, topic_ids, topic_labels, title, filename_prefix, output_dir):
    """
    Creates a grid of bar charts showing the rate difference over time for specific topics.
    """
    if not topic_ids:
        print(f"No topics found for: {title}")
        return

    n_topics = len(topic_ids)
    n_cols = 3
    n_rows = math.ceil(n_topics / n_cols)
    
    # Calculate figure height based on rows
    fig_height = max(6, 4 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, fig_height))
    
    # Flatten axes for easy iteration if multiple rows, ensure it's iterable if single plot
    if n_topics > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    periods = sorted(df['period'].unique())
    
    print(f"Plotting {n_topics} topics for '{title}'...")

    for idx, topic in enumerate(topic_ids):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        topic_data = df[df['topic'] == topic].sort_values('period')
        
        # Determine label
        t_label = topic_labels.get(topic, f"Topic {topic}")
        if str(topic) not in t_label and f"Topic {topic}" not in t_label:
            t_label = f"T{topic}: {t_label}"
            
        # Prepare data for plotting
        # Ensure we have data for all periods (fill missing with 0/NaN if needed, 
        # though input csv usually has all combinations)
        
        # Bars
        bars = []
        colors = []
        
        # Collecting values for y-limits
        bar_vals = []
        sig_bar_vals = [] 
        sig_ci_vals = []
        significant_count = 0
        
        for i, period in enumerate(periods):
            row = topic_data[topic_data['period'] == period]
            if not row.empty:
                val = row['odds_ratio'].values[0]
                is_sig = row['significant'].values[0]
                p_adj = row['p_adjusted'].values[0]
                
                ci_lo = row['ci_lower'].values[0]
                ci_hi = row['ci_upper'].values[0]
                
                # Check for NaNs or Infs
                if pd.isna(val) or np.isinf(val):
                    val = 0
                if pd.isna(ci_lo) or np.isinf(ci_lo):
                    ci_lo = val
                if pd.isna(ci_hi) or np.isinf(ci_hi):
                    ci_hi = val

                bar_vals.append(val)
                if is_sig:
                    sig_bar_vals.append(val)
                    sig_ci_vals.append(ci_hi)
                
                # Error bars require relative values
                lower_err = val - ci_lo
                upper_err = ci_hi - val
                lower_err = max(0, lower_err)
                upper_err = max(0, upper_err)
                
                if is_sig:
                    significant_count += 1
                    color = 'forestgreen' if val > 1 else 'crimson'
                    # Highlights border for significant
                    ax.bar(i, val, color=color, alpha=0.8, edgecolor='black', width=0.8)
                    ax.errorbar(i, val, yerr=[[lower_err], [upper_err]], fmt='none', ecolor='black', capsize=3)
                    
                    # Add significance stars
                    stars = ""
                    if p_adj < 0.001:
                        stars = "***"
                    elif p_adj < 0.01:
                        stars = "**"
                    elif p_adj < 0.05:
                        stars = "*"
                    
                    if stars:
                        va = 'bottom'
                        # Add a small offset to avoid overlap with bar edge
                        y_pos = ci_hi + (ci_hi * 0.02 if ci_hi > 0 else 0.5)
                        
                        ax.text(i, y_pos, stars, ha='center', va=va, fontsize=12, fontweight='bold', color='black') # Increased fontsize for visibility
                else:
                    ax.bar(i, val, color='gray', alpha=0.3, width=0.8)
                    # No errorbar for non-significant
            else:
                # No data for this period
                bar_vals.append(0)
        
        ax.axhline(y=1, color='black', linestyle='-', linewidth=1)
        
        # X-Axis
        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels([str(p) for p in periods], rotation=45, ha='right', fontsize=9)
        
        # Title & Labels
        ax.set_ylabel('Odds Ratio (mit 95% KI)')
        ax.set_title(f'{t_label}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Dynamic Y-Limits
        if bar_vals:
            max_bar = max(bar_vals) if bar_vals else 0
            max_ci = max(sig_ci_vals) if sig_ci_vals else 0
            
            # Heuristic: Cap the Y-axis if CI is extremely large compared to the bar height
            # We want to show at least 3x the max bar height if CIs go that high, 
            # or the full CI if it's reasonable. 
            
            cap_threshold = max(max_bar * 3.0, 5.0)
            y_limit_max = min(max_ci * 1.1, cap_threshold)
            
            # Ensure we don't cut off the main bar if max_ci is weirdly smaller (unlikely)
            y_limit_max = max(y_limit_max, max_bar * 1.1)
            
            # Ensure at least 1.2 is shown so the baseline at 1.0 is always visible with some buffer
            y_limit_max = max(y_limit_max, 1.2)
            
            ax.set_ylim(0, y_limit_max)
        else:
            ax.set_ylim(0, 5)

    # Hide unused subplots
    for idx in range(len(topic_ids), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(title, fontsize=16, y=1.005 if n_rows > 1 else 1.05)
    
    # Add significance legend
    legend_text = "Signifikanzlevel (BH-korrigiert): * p < 0.05, ** p < 0.01, *** p < 0.001"
    plt.figtext(0.5, 0.01, legend_text, ha='center', fontsize=12, style='italic')
    
    # Adjust layout to make room for legend
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    output_path_png = os.path.join(output_dir, f"{filename_prefix}.png")
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    
    output_path_svg = os.path.join(output_dir, f"{filename_prefix}.svg")
    plt.savefig(output_path_svg, format='svg', bbox_inches='tight')
    
    plt.close()
    print(f"Saved visualization to {output_path_png} and {output_path_svg}")

def main():
    # Resolve paths to absolute to avoid CWD issues
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_model_path = os.path.join(script_dir, MODEL_PATH)
    abs_data_path = os.path.join(script_dir, DATA_PATH)

    print(f"Loading BERTopic model from: {abs_model_path}")
    if not os.path.exists(abs_model_path):
        print(f"Error: Model path does not exist: {abs_model_path}")
        return

    try:
        topic_model = BERTopic.load(abs_model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Loading data from: {abs_data_path}")
    if not os.path.exists(abs_data_path):
        print(f"Error: Data path does not exist: {abs_data_path}")
        return

    try:
        with open(abs_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print("Data loaded successfully.")
    
    # Create DataFrame for matching
    df = pd.DataFrame(data)

    # Filter data by timestamp (1955-2024) BEFORE assigning topics
    if 'timestamp' in df.columns:
        # Convert timestamp to datetime objects
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Filter rows where year is between 1955 and 2024
        initial_count = len(df)
        df = df[
            (df['timestamp_dt'].dt.year >= 1955) & 
            (df['timestamp_dt'].dt.year <= 2024)
        ]
        print(f"Filtered data to years 1955-2024. Rows kept: {len(df)}/{initial_count}")
    else:
        print("Warning: 'timestamp' column not found. Skipping filtering.")
    
    # Check alignment after filtering
    if len(df) != len(topic_model.topics_):
        print(f"Warning: Number of filtered documents ({len(df)}) does not match number of topics in model ({len(topic_model.topics_)}).")
        print("Matching might be incorrect if the data file is not the exact one used for training.")
        # If lengths mismatch significantly, assignment will fail or be wrong.
        if len(df) != len(topic_model.topics_):
             print("Error: Length mismatch prevents topic assignment.")
             # return # Depending on flow, we might want to return, but let's see if user wants to force it or inspect.
    else:
        print(f"Confirmed: Number of filtered documents ({len(df)}) matches number of topics.")

    # Assign topics from the model to the dataframe
    # We assume the order of documents in the JSON is the same as used for training
    if len(df) == len(topic_model.topics_):
        df['Topic'] = topic_model.topics_
    else:
        print("Skipping topic assignment due to length mismatch.")
    
    # --- Statistical Analysis ---
    print("\nStarting Fisher's Exact Test Analysis...")
    
    # 1. Prepare Columns
    # is_chart: True if 'Erste Notierung' is present
    df['is_chart'] = df['Erste Notierung'].apply(lambda x: pd.notna(x) and str(x).strip() != "")
    
    # period: Create 5-year aggregate periods (e.g., 1955-1959)
    if 'timestamp_dt' in df.columns:
        start_year = 1955
        df['year_temp'] = df['timestamp_dt'].dt.year
        # Calculate start of the 5-year bin
        df['period_start'] = start_year + ((df['year_temp'] - start_year) // 5) * 5
        # Format as string "YYYY-YYYY"
        df['period'] = df['period_start'].apply(lambda x: f"{int(x)}-{int(x+4)}")
    else:
        print("Error: timestamp_dt column missing. Filtering failed?")
        return

    # Load Labels
    topic_labels = load_topic_labels(script_dir)
    output_dir = get_output_dir(script_dir)

    # 2. Iterate Periods and Topics
    results = []
    
    # Sort periods based on the period_start year so they are chronological
    # We can get unique period strings and sort them directly as strings since "1955..." < "2020..." works
    periods = sorted(df['period'].unique())
    topics = sorted(df['Topic'].unique())
    
    print(f"Analyzing {len(periods)} periods (5-year aggregates) and {len(topics)} topics...")
    
    for period in periods:
        df_period = df[df['period'] == period]
        
        total_docs = len(df_period)
        if total_docs == 0: continue
            
        chart_total = df_period['is_chart'].sum()
        no_chart_total = total_docs - chart_total
        
        if chart_total == 0 or no_chart_total == 0:
            # Cannot do comparison if one group is empty in this year
            continue
            
        # Topic-wise test
        period_results = []
        for topic in topics:
            # Contingency Table
            #              Chart    NoChart
            # In Topic       a        b
            # Not In Topic   c        d
            
            in_topic = df_period['Topic'] == topic
            
            a = ((in_topic) & (df_period['is_chart'])).sum()
            b = ((in_topic) & (~df_period['is_chart'])).sum()
            
            c = chart_total - a
            d = no_chart_total - b
            
            # Fisher Exact Test
            # alternative='two-sided' to check for over or under representation
            odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]], alternative='two-sided')
            
            # Calculate Confidence Interval (95%) for Odds Ratio
            # Using Haldane-Anscombe correction for potential zeros in SE calculation
            a_adj, b_adj, c_adj, d_adj = a + 0.5, b + 0.5, c + 0.5, d + 0.5
            se_log_or = np.sqrt(1/a_adj + 1/b_adj + 1/c_adj + 1/d_adj)
            z_score = 1.96
            
            # Use adjusted counts for log(OR) center to avoid log(0)
            or_adj = (a_adj * d_adj) / (b_adj * c_adj)
            log_or_adj = np.log(or_adj)
            
            ci_lower = np.exp(log_or_adj - z_score * se_log_or)
            ci_upper = np.exp(log_or_adj + z_score * se_log_or)
            
            # Calculate Rate Difference
            # Rate of topic in charts vs rate of topic in non-charts
            rate_chart = a / chart_total if chart_total > 0 else 0
            rate_nochart = b / no_chart_total if no_chart_total > 0 else 0
            rate_diff = rate_chart - rate_nochart

            period_results.append({
                'topic': topic,
                'period': period,
                'p_value': p_value,
                'odds_ratio': odds_ratio,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'rate_difference': rate_diff,
                'count_chart_topic': int(a),
                'count_nochart_topic': int(b),
                'total_chart': int(chart_total),
                'total_nochart': int(no_chart_total)
            })
        
        # BH Correction for this period
        if period_results:
            p_values = [res['p_value'] for res in period_results]
            reject, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
            
            for i, res in enumerate(period_results):
                res['p_adjusted'] = pvals_corrected[i]
                res['significant'] = reject[i]
                
                # Direction (based on rate difference which aligns with Odds Ratio > 1)
                if res['significant']:
                    if res['rate_difference'] > 0:
                        res['direction'] = 'overrepresented'
                    elif res['rate_difference'] < 0:
                        res['direction'] = 'underrepresented'
                    else:
                        res['direction'] = 'neutral'
                else:
                    res['direction'] = 'neutral'
                    
                results.append(res)
            
    if not results:
        print("No results computed. Check if chart data exists for the years.")
        return

    # 3. Aggregation & Visualization
    results_df = pd.DataFrame(results)
    
    # Save Results CSV
    csv_path = os.path.join(output_dir, "chart_overrepresentation_fisher_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved results to {csv_path}")
    
    # Identify Consistently Overrepresented
    print("\nIdentifying consistently overrepresented topics...")
    over_sig = results_df[
        (results_df['significant'] == True) & 
        (results_df['direction'] == 'overrepresented')
    ]
    over_counts = over_sig.groupby('topic')['period'].nunique()
    over_topics = over_counts[over_counts >= 2].index.tolist()
    
    if over_topics:
        over_topics.sort(key=lambda t: over_counts[t], reverse=True)
        # Check if too many topics, maybe limit relevant ones? 
        # For now plot all >= 2
        print(f"Found {len(over_topics)} topics consistently overrepresented.")
        create_trajectory_plot(
            results_df, 
            over_topics[:10],  # Limit to top 10 to avoid clutter
            topic_labels, 
            "Chart-dominante Themen (Odds Ratio)\n(>= 2 signifikante Perioden, BH korrigiert, p < 0.05)", 
            "trajectories_consistent_overrepresented_or",
            output_dir
        )
    else:
        print("No topics found with >= 2 significant overrepresented periods.")

    # Identify Consistently Underrepresented
    print("\nIdentifying consistently underrepresented topics...")
    under_sig = results_df[
        (results_df['significant'] == True) & 
        (results_df['direction'] == 'underrepresented')
    ]
    under_counts = under_sig.groupby('topic')['period'].nunique()
    under_topics = under_counts[under_counts >= 2].index.tolist()
    
    if under_topics:
        under_topics.sort(key=lambda t: under_counts[t], reverse=True)
        print(f"Found {len(under_topics)} topics consistently underrepresented.")
        create_trajectory_plot(
            results_df, 
            under_topics[:10], 
            topic_labels, 
            "In Charts unterrepräsentierte Themen (Odds Ratio)\n(>= 2 signifikante Perioden, BH korrigiert, p < 0.05)", 
            "trajectories_consistent_underrepresented_or",
            output_dir
        )
    else:
        print("No topics found with >= 2 significant underrepresented periods.")

if __name__ == "__main__":
    main()
