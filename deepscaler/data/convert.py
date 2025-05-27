import pandas as pd
import sys
import os

def convert_parquet(parquet_file, output_dir=None):
    # Get directory and filename
    if output_dir is None:
        output_dir = os.path.dirname(parquet_file)
    base_name = os.path.basename(parquet_file).split('.')[0]
    
    # Read the parquet file
    print(f"Reading {parquet_file}...")
    df = pd.read_parquet(parquet_file)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    print(f"Saving to CSV: {csv_path}")
    df.to_csv(csv_path, index=False)
    
    # Save as JSON (more readable for nested structures)
    json_path = os.path.join(output_dir, f"{base_name}.json")
    print(f"Saving to JSON: {json_path}")
    df.to_json(json_path, orient='records', indent=2)
    
    # Print basic info
    print(f"\nConverted file with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {', '.join(df.columns)}")
    
    return csv_path, json_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_parquet.py <parquet_file_path> [output_directory]")
        sys.exit(1)
    
    parquet_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    csv_path, json_path = convert_parquet(parquet_file, output_dir)
    print(f"\nFiles saved to:\n- {csv_path}\n- {json_path}")
