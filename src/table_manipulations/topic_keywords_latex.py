import pandas as pd
import ast
import os

def process_representation(rep_str):
    try:
        # Evaluate the string representation of the list
        # The representation in the CSV seems to be: "['word1', 'word2', ...]"
        if not isinstance(rep_str, str):
            return ""
        words = ast.literal_eval(rep_str)
        # Take the first 20 words
        top_20 = words[:20]
        # Join with comma and space
        return ", ".join(top_20)
    except Exception as e:
        print(f"Error parsing representation: {e}")
        return str(rep_str)

def get_custom_label(row):
    # Check if Custom_Label is not null and not empty string
    # Pandas reads missing values as NaN (float) usually
    custom_val = row.get('Custom_Label')
    if pd.notna(custom_val) and str(custom_val).strip() != "":
        return custom_val
    
    # Fallback to LLM_Label
    llm_val = row.get('LLM_Label')
    if pd.notna(llm_val):
        return llm_val
    
    return ""

def main():
    # Define file paths
    # Using absolute paths based on the workspace info provided
    base_path = "/home/tim/Desktop/Masterarbeit/masterarbeit-code"
    input_csv = os.path.join(base_path, "bertopic/dtm/models/topic_info_llm.csv")
    output_tex = os.path.join(base_path, "visualisations/keywords/topic_keywords_table_generated.tex")

    print(f"Reading from: {input_csv}")
    
    if not os.path.exists(input_csv):
        print(f"Error: File not found at {input_csv}")
        return

    # Load data
    df = pd.read_csv(input_csv)

    # Process "TF-IDF Keywords"
    # Ensure Representation is treated as string first
    df['TF-IDF Keywords'] = df['Representation'].apply(process_representation)

    # Process "Custom Label"
    df['Custom Label'] = df.apply(get_custom_label, axis=1)

    # Select and Rename columns
    # Target columns: "Topic", "Anzahl", "BERTopic Name", "Custom Label", "TF-IDF Keywords"
    # Source columns: "Topic", "Count", "Name", (calculated), (calculated)
    
    final_df = df[['Topic', 'Count', 'Name', 'Custom Label', 'TF-IDF Keywords']].copy()
    final_df.rename(columns={'Name': 'BERTopic Name', 'Count': 'Anzahl'}, inplace=True)

    # Generate LaTeX
    # formatters can be used to escape characters if needed, but to_latex defaults allow some escaping.
    # However, large tables often benefit from column specifications (e.g. p{5cm}) for long texts like keywords.
    # We will use a standard tabular generation.
    
    # Adjusting column widths might be necessary for the keywords column in LaTeX manually later,
    # or we can specify column_format. 
    # Let's use a simple longtable export which is good for multi-page tables.
    
    latex_code = final_df.to_latex(
        index=False, 
        longtable=True, 
        caption="Topic Overview", 
        label="tab:topics",
        escape=True  # Escapes special characters like _
    )

    # Save to file
    with open(output_tex, "w", encoding="utf-8") as f:
        f.write(latex_code)
    
    print(f"LaTeX table generated at: {output_tex}")

if __name__ == "__main__":
    main()
