import pandas as pd
import json
import os

def csv_to_json(csv_path, json_path):
    """
    Convert CSV file to JSON format
    
    Args:
        csv_path (str): Path to the input CSV file
        json_path (str): Path to the output JSON file
    """
    # Read CSV with specified parameters
    df = pd.read_csv(csv_path,
                     delimiter=',',
                     quotechar='"',
                     quoting=1,
                     skipinitialspace=True,
                     encoding='utf-8',
                     low_memory=False)
    
    # Convert DataFrame to JSON
    # Convert specific columns to string and handle NaN values
    columns_to_convert = ['Anzahl Wochen']
    year_columns = [col for col in df.columns if 'Jahr Album' in col]
    columns_to_convert.extend(year_columns)
    
    for col in columns_to_convert:
        if col in df.columns:
            # Convert to numeric first, then to int (removing .0), then to string
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int).astype(str).replace('0', '')
    
    # Replace all NaN values with empty strings
    df = df.fillna('')
    
    json_data = json.dumps(df.to_dict(orient='records'), indent=2, ensure_ascii=False)
    
    # Write JSON to file
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json_file.write(json_data)
    
    print(f"Successfully converted {csv_path} to {json_path}")
    print(f"Number of records: {len(df)}")

if __name__ == "__main__":
    # Define file paths
    csv_path = "../../data/charts_with_language_updated_v4.csv"
    json_path = "../../data/charts_with_language_updated_v4.json"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    
    # Convert CSV to JSON
    csv_to_json(csv_path, json_path)
