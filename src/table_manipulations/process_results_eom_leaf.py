import pandas as pd
import os

# Define the input and output paths
# Assuming script is run from src/table_manipulations/
input_path = '../../bertopic/dtm/evaluation_parameters/results_eom_leaf_1.csv'
output_path = '../../bertopic/dtm/evaluation_parameters/results_eom_leaf_1_cleaned.csv'

def process_table():
    # Construct absolute path to ensure it works
    script_dir = os.path.dirname(os.path.abspath(__file__))
    res_input_path = os.path.join(script_dir, input_path)
    res_output_path = os.path.join(script_dir, output_path)

    print(f"Reading file from: {res_input_path}")
    
    try:
        df = pd.read_csv(res_input_path)
    except FileNotFoundError:
        print(f"Error: File not found at {res_input_path}")
        return

    # Columns to drop
    columns_to_drop = [
        "topic_coverage_before", 
        "n_topics_after", 
        "topic_coverage", 
        "time_seconds",
        "min_df",
        "max_df",
        "n_components",
        "cluster_selection_epsilon"
    ]
    
    # Check if columns exist before dropping
    existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if len(existing_columns_to_drop) != len(columns_to_drop):
        missing = set(columns_to_drop) - set(existing_columns_to_drop)
        print(f"Warning: The following columns to drop were not found: {missing}")

    df.drop(columns=existing_columns_to_drop, inplace=True)

    # Columns to rename
    rename_mapping = {
        "n_topics_before": "n_topics",
        "outlier_ratio_before": "outlier_ratio_before_reduction",
        "coherence_before": "coherence_before_after_reduction",
        "diversity_before": "diversity_before_reduction",
        "outlier_ratio_after": "outlier_ratio_after_reduction",
        "ratio_topic_0": "ratio_topic_0_after_reduction",
        "ratio_topic_1": "ratio_topic_1_after_reduction",
        "coherence": "coherence_after_reduction",
        "diversity": "diversity_after_reduction"
    }

    df.rename(columns=rename_mapping, inplace=True)

    # Round all values to 3 decimal places
    df = df.round(3)

    # Save the cleaned dataframe
    print(f"Saving processed file to: {res_output_path}")
    df.to_csv(res_output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    process_table()
