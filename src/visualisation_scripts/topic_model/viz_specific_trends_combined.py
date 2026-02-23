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


def create_trend_plots(adv_df, all_topics_data, target_topics, output_dir_base, folder_name, plot_title_suffix, full_years, full_labels):
    print(f"\n--- Erstelle Plots für '{folder_name}' ---")
    print(f"Targeting topics: {sorted(target_topics)}")
    
    output_dir = os.path.join(output_dir_base, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter specific_df
    specific_df = adv_df[adv_df['Topic'].isin(target_topics)].copy()
    
    # Check coverage
    found_topics = set(specific_df['Topic'].unique())
    missing = set(target_topics) - found_topics
    if missing:
        print(f"Warning: Could not find data for topics: {missing}")

    # Split into categories
    inc_df = specific_df[specific_df['MK_Trend_Final'] == 'increasing'].sort_values('Mean_Rel_Freq', ascending=False)
    dec_df = specific_df[specific_df['MK_Trend_Final'] == 'decreasing'].sort_values('Mean_Rel_Freq', ascending=False)
    neu_df = specific_df[specific_df['MK_Trend_Final'] == 'no trend'].sort_values('Mean_Rel_Freq', ascending=False)
    
    # Plotting Split
    # Determine which categories have data
    valid_categories = [] # List of tuples: (dataframe, title, marker)
    if not inc_df.empty:
        valid_categories.append((inc_df, f"Steigende Trends ({plot_title_suffix})", 'o'))
    if not dec_df.empty:
        valid_categories.append((dec_df, f"Fallende Trends ({plot_title_suffix})", 'x'))
    if not neu_df.empty:
        valid_categories.append((neu_df, f"Neutrale Trends ({plot_title_suffix})", '.'))
    
    num_plots = len(valid_categories)
    
    if num_plots > 0:
        print(f"Erstelle Mehrfach-Plot für {len(specific_df)} Themen in {num_plots} Kategorien...")
        
        # Adjust figure size based on number of plots
        fig_height = 6 * num_plots
        fig, axes = plt.subplots(num_plots, 1, figsize=(14, fig_height), sharex=True)
        
        # Ensure axes is iterable even if it's just one object
        if num_plots == 1:
            axes = [axes]
        
        # Helper to plot on axis
        def _plot_subset(ax, subset_df, title, marker):
            ax.set_title(title)
            ax.set_ylabel('Rel. Freq (Anteil)')
            ax.grid(True, alpha=0.3)
            
            cmap = plt.get_cmap('tab20')
            
            if subset_df.empty:
                 # Should not happen with new logic, but safe keep
                ax.text(0.5, 0.5, "Keine Themen in dieser Kategorie", ha='center', va='center', transform=ax.transAxes)
                return

            for idx, (_, row) in enumerate(subset_df.iterrows()):
                t_id = row['Topic']
                data = all_topics_data[t_id]['data']
                lbl = all_topics_data[t_id]['label']
                
                # Wrap label text
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
                
                color = cmap(idx % 20)
                ax.plot(data.index, data.values, marker=marker, label=legend_lbl, color=color, linewidth=2)
                
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=9)

        # Plot dynamic categories
        for ax, (df_cat, title, marker) in zip(axes, valid_categories):
            _plot_subset(ax, df_cat, title, marker)

        plt.xticks(full_years, full_labels, rotation=45, ha='right')
        plt.xlabel('Zeitraum')
        
        plt.tight_layout()
        out_file = os.path.join(output_dir, f'specific_trends_{folder_name}_combined.png')
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        print(f"Specific Visualization saved: {os.path.abspath(out_file)}")
        plt.close()
    else:
        print(f"Keine Daten für spezifische Trends gefunden für {folder_name}.")

    # --- Single Combined Plot for ALL ---
    print(f"Erstelle kombinierten Plot (All-in-One) für {plot_title_suffix}...")
    plt.figure(figsize=(14, 16)) 
    ax_combined = plt.gca()
    ax_combined.set_title(f"Ausgewählte Topics ({plot_title_suffix}): Steigend, Fallend, Neutral")
    ax_combined.set_ylabel('Rel. Freq (Anteil am Korpus)')
    ax_combined.grid(True, alpha=0.3)
    
    handles_inc = []
    handles_dec = []
    handles_neu = []
    
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i) for i in range(20)]
    
    def _add_lines(subset_df, marker, linestyle, handle_list):
        if subset_df.empty: return
        
        for idx, (_, row) in enumerate(subset_df.iterrows()):
            t_id = row['Topic']
            data = all_topics_data[t_id]['data']
            lbl = all_topics_data[t_id]['label']
            
            lines = textwrap.wrap(lbl, width=30)
            if len(lines) > 2:
                wrapped_lbl = lines[0] + "\n" + lines[1] + "..."
            else:
                wrapped_lbl = "\n".join(lines)
            
            if 'BH_p_value' in row and not pd.isna(row['BH_p_value']):
                    p_str = f"p_adj={row['BH_p_value']:.3f}"
            else:
                    p_str = f"p={row.get('MK_p_value', 1.0):.3f}"
            
            slope_val = row.get('Theil_Sen_Slope', 0.0)
            legend_lbl = f"T{t_id}: {wrapped_lbl}\n({p_str}, Slope={slope_val:.4f})"
            
            color = colors[t_id % 20]
            
            line, = ax_combined.plot(data.index, data.values, marker=marker, linestyle=linestyle, 
                                        label=legend_lbl, color=color, linewidth=2)
            handle_list.append(line)
    
    _add_lines(inc_df, 'o', '-', handles_inc)
    _add_lines(dec_df, 'x', '--', handles_dec)
    _add_lines(neu_df, '.', ':', handles_neu)
    
    final_handles = []
    from matplotlib.lines import Line2D

    if handles_inc:
        h = Line2D([], [], color='black', marker='o', linestyle='-', linewidth=2, label=r'$\bf{STEIGEND}$')
        final_handles.append(h)
        final_handles.extend(handles_inc)
        final_handles.append(Line2D([], [], linestyle="None", label=" ")) 
        
    if handles_dec:
        h = Line2D([], [], color='black', marker='x', linestyle='--', linewidth=2, label=r'$\bf{FALLEND}$')
        final_handles.append(h)
        final_handles.extend(handles_dec)
        final_handles.append(Line2D([], [], linestyle="None", label=" ")) 

    if handles_neu:
        h = Line2D([], [], color='black', marker='.', linestyle=':', linewidth=2, label=r'$\bf{NEUTRAL}$')
        final_handles.append(h)
        final_handles.extend(handles_neu)

    if final_handles and final_handles[-1].get_label() == " ":
            final_handles.pop()

    plt.xticks(full_years, full_labels, rotation=45, ha='right')
    plt.legend(handles=final_handles, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=9, ncol=1)
    plt.xlabel('Zeitraum')
    plt.tight_layout()
    
    out_file_all = os.path.join(output_dir, f'specific_trends_{folder_name}_all_in_one.png')
    plt.savefig(out_file_all, dpi=300, bbox_inches='tight')
    print(f"Combined All-in-One Visualization saved: {os.path.abspath(out_file_all)}")
    plt.close()


def main():
    # Topics definitions
    topics_liebe = list(set([18, 29, 24, 30, 60, 14, 11, 40, 46, 31, 7, 22, 14, 26, 58, 33]))
    # Extracted topics for HipHop
    # T20, T53, T37, T44, T47, T54, T65, T5, T63, T61
    topics_hiphop = list(set([20, 53, 37, 44, 47, 54, 65, 5, 63, 61]))
    # Extracted topics for Ich/Self
    # T19, T9, T0, T24
    topics_ich = list(set([19, 9, 0, 24]))

    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Path to input CSV: ../../../bertopic/dtm/models/topics_over_time.csv
    csv_path = os.path.join(current_dir, '../../../bertopic/dtm/models/topics_over_time.csv')
    
    # Base output directory
    output_base_dir = os.path.join(current_dir, '../../../visualisations/topic_model/trends')

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
    
    # We load ALL topics first for correct BH correction context, then filter later
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

    # --- Advanced Trend Analysis (Mann-Kendall) ---
    if ADVANCED_STATS_AVAILABLE:
        print("\n--- Starte globale Trendanalyse (Mann-Kendall) ---")
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
                    if row['MK_Trend'] == 'increasing':
                        return 'increasing'
                    elif row['MK_Trend'] == 'decreasing':
                        return 'decreasing'
                        
            return 'no trend'

        adv_df['MK_Trend_Final'] = adv_df.apply(determine_trend, axis=1)

        # 4. Generate All requested plots
        create_trend_plots(adv_df, all_topics_data, topics_liebe, output_base_dir, 'liebe', 'Liebe', full_years, full_labels)
        create_trend_plots(adv_df, all_topics_data, topics_hiphop, output_base_dir, 'hiphop', 'Hip-Hop/Rap', full_years, full_labels)
        create_trend_plots(adv_df, all_topics_data, topics_ich, output_base_dir, 'ich', 'Ich/Selbst', full_years, full_labels)

    else:
        print("Keine erweiterten Statistiken verfügbar, kann nicht nach Steigend/Fallend/Neutral filtern wie gewünscht.")

if __name__ == "__main__":
    main()
