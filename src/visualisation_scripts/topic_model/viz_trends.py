import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import textwrap


try:
    import pymannkendall as mk
    ADVANCED_STATS_AVAILABLE = True
except ImportError:
    ADVANCED_STATS_AVAILABLE = False
    print("Warnung: 'pymannkendall' ist nicht installiert. Erweiterte Analysen (Mann-Kendall) werden übersprungen.")
    print("Bitte installieren mit: pip install pymannkendall")

# Configure plot style for LaTeX
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


def benjamini_hochberg_correction(p_values, alpha=0.05):
    """
    Führt die Benjamini-Hochberg-Korrektur für p-Werte durch.
    Gibt ein Tupel zurück: (rejected_boolean_array, adjusted_p_values)
    """
    p_values = np.array(p_values)
    n = len(p_values)
    if n == 0:
        return np.array([]), np.array([])
    
    # Indizes sortieren
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    # Adjusted p-values berechnen
    # q_value[i] = min(1, min_{j >= i} (p[j] * n / (j + 1)))
    # Wir iterieren rückwärts
    adjusted_p_sorted = np.zeros(n)
    current_min = 1.0
    
    for i in range(n - 1, -1, -1):
        # Rang ist i+1 (1-basiert)
        p = sorted_p[i]
        adj = p * n / (i + 1)
        current_min = min(current_min, adj)
        adjusted_p_sorted[i] = min(1.0, current_min)
        
    # Zurück in ursprüngliche Reihenfolge
    adjusted_p = np.zeros(n)
    adjusted_p[sorted_indices] = adjusted_p_sorted
    
    rejected = adjusted_p < alpha
    return rejected, adjusted_p


def main():
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Path to input CSV: ../../../bertopic/dtm/models/topics_over_time.csv
    csv_path = os.path.join(current_dir, '../../../bertopic/dtm/models/topics_over_time.csv')
    
    # Path to output directory: ../../../visualisations/topic_model/trends
    output_dir = os.path.join(current_dir, '../../../visualisations/topic_model/trends')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading data from: {os.path.abspath(csv_path)}")
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # 1. Parse Timestamp to Start Year
    # Timestamp is in format "YYYY-YYYY" (e.g. 1954-1958)
    df['Start_Year'] = df['Timestamp'].str.split('-').str[0].astype(int)

    # 2. Normalize Frequency (Calculate Share of Voice per Time Bin)
    # Total songs per bin
    bin_totals = df.groupby('Timestamp')['Frequency'].sum().reset_index()
    bin_totals.rename(columns={'Frequency': 'Total_Freq_In_Bin'}, inplace=True)

    # Merge back
    df = df.merge(bin_totals, on='Timestamp')
    df['Relative_Frequency'] = df['Frequency'] / df['Total_Freq_In_Bin']

    # Define Full Range of Years (1955 to 2020)
    full_years = list(range(1955, 2025, 5)) 
    full_labels = [f"{y}-{y+4}" for y in full_years]

    # 3. Identify Trends
    # Filter out Topic -1 (Noise) for trend analysis
    clean_topics = df[df['Topic'] != -1].copy()

    # Pre-process: Reindex every topic to full year range filling missing with 0
    all_topics_data = {}

    for topic_id in clean_topics['Topic'].unique():
        subset = clean_topics[clean_topics['Topic'] == topic_id]
        
        # Create a Series indexed by Start_Year
        subset_indexed = subset.set_index('Start_Year')['Relative_Frequency']
        
        # Reindex to full range, fill with 0
        subset_full = subset_indexed.reindex(full_years, fill_value=0.0)
        
        # Save Label too
        if 'Global_Representation' in subset.columns and not pd.isna(subset['Global_Representation'].iloc[0]):
             label = subset['Global_Representation'].iloc[0]
        else:
             label = str(topic_id)
             
        all_topics_data[topic_id] = {
            'data': subset_full,
            'label': label
        }

    # Helper to setup plot
    def setup_plot(title):
        plt.figure(figsize=(12, 6))
        plt.title(title)
        plt.xlabel('Zeitraum')
        plt.xticks(full_years, full_labels, rotation=45)
        plt.ylabel('Rel. Freq (Anteil)')
        plt.grid(True, alpha=0.3)

    def plot_combined_summary(inc_df, dec_df, all_topics_data, full_years, full_labels, output_dir, top_n=10):
        """
        Creates a combined summary plot with Top N Increasing and Top N Decreasing trends.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
        
        # Get distinct colors from tab20 (20 distinct colors)
        # We assign the first 10 to increasing, and the next 10 to decreasing to ensure they look different.
        # If top_n > 10, we cycle, but tab20 is best for top 10.
        cmap = plt.get_cmap('tab20')
        colors_inc = [cmap(i) for i in range(0, 10)]
        colors_dec = [cmap(i) for i in range(10, 20)]

        # Helper to plot on axis
        def _plot_subset(ax, subset_df, title, marker, palette):
            ax.set_title(title)
            ax.set_ylabel('Rel. Freq (Anteil)')
            ax.grid(True, alpha=0.3)
            
            for idx, (_, row) in enumerate(subset_df.iterrows()):
                t_id = row['Topic']
                data = all_topics_data[t_id]['data']
                lbl = all_topics_data[t_id]['label']
                # Wrap label text (max 2 lines)
                lines = textwrap.wrap(lbl, width=30)
                if len(lines) > 2:
                    wrapped_lbl = lines[0] + "\n" + lines[1] + "..."
                else:
                    wrapped_lbl = "\n".join(lines)
                
                if 'BH_p_value' in row and not pd.isna(row['BH_p_value']):
                    p_str = f"p_adj={row['BH_p_value']:.3f}"
                else:
                    p_str = f"p={row['MK_p_value']:.3f}"
                
                slope_val = row.get('Theil_Sen_Slope', 0.0)
                legend_lbl = f"T{t_id}: {wrapped_lbl}\n({p_str}, Slope={slope_val:.4f})"
                
                # Pick color from palette
                color = palette[idx % len(palette)]
                
                ax.plot(data.index, data.values, marker=marker, label=legend_lbl, color=color, linewidth=2)
                
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=9)

        # Top Increasing
        top_inc = inc_df.head(top_n)
        if not top_inc.empty:
            _plot_subset(ax1, top_inc, f"Top {top_n} Steigende Themen (nach Anteil im Korpus)", 'o', colors_inc)
        
        # Top Decreasing
        top_dec = dec_df.head(top_n)
        if not top_dec.empty:
            _plot_subset(ax2, top_dec, f"Top {top_n} Fallende Themen (nach Anteil im Korpus)", 'x', colors_dec)
        
        # X Axis
        plt.xticks(full_years, full_labels, rotation=45, ha='right')
        plt.xlabel('Zeitraum')
        
        plt.tight_layout()
        out_file = os.path.join(output_dir, f'trends_summary_combined_top{top_n}.png')
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        print(f"Combined Visualization saved: {os.path.abspath(out_file)}")
        plt.close()

    # --- Advanced Trend Analysis (Mann-Kendall) ---
    if ADVANCED_STATS_AVAILABLE:
        print("\n--- Starte Trendanalyse (Mann-Kendall) ---")
        advanced_results = []
        
        for topic_id in clean_topics['Topic'].unique():
            # Use the filled data
            ts_data = all_topics_data[topic_id]['data']
            label = all_topics_data[topic_id]['label']
            
            signal = ts_data.values
            
            # 1. Mann-Kendall
            mk_trend = "not_calculated"
            mk_p = 1.0
            mk_slope = 0.0
            try:
                mk_res = mk.trend_free_pre_whitening_modification_test(signal)
                mk_trend = mk_res.trend
                mk_p = mk_res.p
                mk_slope = mk_res.slope
            except Exception:
                pass
            
            # Calculate Mean Relative Frequency for sorting importance
            mean_freq = ts_data.mean()

            advanced_results.append({
                'Topic': topic_id,
                'Label': label,
                'MK_Trend': mk_trend,
                'MK_p_value': mk_p,
                'Theil_Sen_Slope': mk_slope,
                'Mean_Rel_Freq': mean_freq
            })
            
        adv_df = pd.DataFrame(advanced_results)

        # --- Benjamini-Hochberg Correction ---
        print("Führe Benjamini-Hochberg-Korrektur (FDR) durch...")
        valid_p_indices = ~adv_df['MK_p_value'].isna()
        p_values = adv_df.loc[valid_p_indices, 'MK_p_value'].values
        
        _, adj_p = benjamini_hochberg_correction(p_values, alpha=0.05)
        
        adv_df.loc[valid_p_indices, 'BH_p_value'] = adj_p
        
        # Update classic trend label based on corrected p-value
        def determine_trend(row):
            if pd.isna(row['BH_p_value']):
                return 'not_calculated'
            
            # Use adjusted p-value for significance
            if row['BH_p_value'] < 0.05:
                if row['Theil_Sen_Slope'] > 0:
                    return 'increasing'
                elif row['Theil_Sen_Slope'] < 0:
                    return 'decreasing'
                else:
                    # Fallback: If slope is 0 but result is significant, use original MK direction
                    # (This happens with sparse data where median slope is 0 but rank trend exists)
                    if row['MK_Trend'] == 'increasing':
                        return 'increasing'
                    elif row['MK_Trend'] == 'decreasing':
                        return 'decreasing'
                        
            return 'no trend'

        adv_df['MK_Trend_Original'] = adv_df['MK_Trend'] # Backup original
        adv_df['MK_Trend'] = adv_df.apply(determine_trend, axis=1) # Overwrite for plotting
        
        # Save detailed results
        adv_output_path = os.path.join(output_dir, 'trends_advanced_stats_detailed.csv')
        adv_df.to_csv(adv_output_path, index=False)
        print(f"Erweiterte Analyse (mit BH-Korrektur) gespeichert unter: {os.path.abspath(adv_output_path)}")
        
        # Helper to plot in chunks
        def plot_in_chunks(df_subset, title_base, output_base, style='o'):
            if df_subset.empty:
                return

            chunk_size = 13
            num_chunks = (len(df_subset) + chunk_size - 1) // chunk_size

            for i in range(num_chunks):
                chunk = df_subset.iloc[i * chunk_size : (i + 1) * chunk_size]
                
                # Determine title suffix
                if num_chunks > 1:
                    title = f"{title_base} (Teil {i+1}/{num_chunks})"
                    filename = f"{output_base}_part{i+1}.png"
                else:
                    title = title_base
                    filename = f"{output_base}.png"
                
                setup_plot(title)
                
                # Create a distinct color map for the chunk (tab20 has 20 colors, enough for chunk_size=15)
                cmap_chunk = plt.get_cmap('tab20')
                
                for idx, (_, row) in enumerate(chunk.iterrows()):
                    ts_data = all_topics_data[row['Topic']]['data']
                    
                    if 'BH_p_value' in row and not pd.isna(row['BH_p_value']):
                        p_str = f"p_adj={row['BH_p_value']:.4f}"
                    else:
                        p_str = f"p={row['MK_p_value']:.4f}"
                    
                    slope_val = row.get('Theil_Sen_Slope', 0.0)
                    
                    # Wrap label text (max 2 lines)
                    lines = textwrap.wrap(row['Label'], width=30)
                    if len(lines) > 2:
                        wrapped_lbl = lines[0] + "\n" + lines[1] + "..."
                    else:
                        wrapped_lbl = "\n".join(lines)
                        
                    label_text = f"{row['Topic']}: {wrapped_lbl}\n({p_str}, Slope={slope_val:.4f})"
                    
                    linestyle = '-'
                    if style == 'x': # Dashed for decreasing usually
                        linestyle = '--'
                    elif style == '.': # Dotted for no trend
                        linestyle = ':'
                        
                    # Pick color from tab20
                    color = cmap_chunk(idx % 20)
                    
                    plt.plot(ts_data.index, ts_data.values, marker=style, linestyle=linestyle, label=label_text, color=color)

                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
                plt.tight_layout()
                out_path = os.path.join(output_dir, filename)
                plt.savefig(out_path, dpi=300)
                print(f"Visualization saved to {os.path.abspath(out_path)}")
                plt.close()

        # --- Visualize MK Increasing (ALL) - Sorted by Mean Relative Frequency ---
        mk_increasing = adv_df[adv_df['MK_Trend'] == 'increasing'].sort_values('Mean_Rel_Freq', ascending=False)
        plot_in_chunks(mk_increasing, 'Signifikante Positive Trends (Mann-Kendall)', 'trends_mk_increasing_all', style='o')

        # --- Visualize MK Decreasing (ALL) - Sorted by Mean Relative Frequency ---
        mk_decreasing = adv_df[adv_df['MK_Trend'] == 'decreasing'].sort_values('Mean_Rel_Freq', ascending=False)
        plot_in_chunks(mk_decreasing, 'Signifikante Negative Trends (Mann-Kendall)', 'trends_mk_decreasing_all', style='x')
        
        # --- Combined Summary Plot (Top 10) ---
        print("\nErstelle zusammenfassende Visualisierung (Top 10)...")
        plot_combined_summary(mk_increasing, mk_decreasing, all_topics_data, full_years, full_labels, output_dir, top_n=10)
        
        # --- Visualize MK No Trend (ALL) ---
        mk_notrend = adv_df[adv_df['MK_Trend'] == 'no trend'].sort_values('MK_p_value', ascending=False)
        plot_in_chunks(mk_notrend, 'Themen ohne signifikanten Trend (Mann-Kendall)', 'trends_mk_no_trend_all', style='.')


if __name__ == "__main__":
    main()
