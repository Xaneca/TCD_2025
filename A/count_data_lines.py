import os
import pandas as pd

def count_total_lines():
    base_path = r"c:\Users\Xaneca\OneDrive\Uni\Mestrado\TCD\GitRepo\FORTH_TRACE_DATASET-master\FORTH_TRACE_DATASET-master"
    total_lines = 0
    
    # Walk through all directories
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                try:
                    # Read CSV and count rows
                    df = pd.read_csv(file_path)
                    lines = len(df)
                    total_lines += lines
                    print(f"File {file}: {lines} lines")
                except Exception as e:
                    print(f"Error reading {file}: {str(e)}")
    
    print(f"\nTotal number of lines across all CSV files: {total_lines:,}")

if __name__ == "__main__":
    count_total_lines()