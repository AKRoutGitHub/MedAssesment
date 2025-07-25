import pandas as pd
import numpy as np
from pathlib import Path

def load_healthcare_data():
    """
    Load the healthcare dataset from the Kaggle source
    Returns:
        pd.DataFrame: The loaded healthcare dataset
    """
    # TODO: Implement data loading from Kaggle
    # For now, this is a placeholder
    data_path = Path(r"C:\Users\iamas\Desktop\learnathon_04\healthcare\healthcare_dataset.csv")
    return pd.read_csv(data_path)

def initial_data_analysis(df):
    """
    Perform initial analysis of the dataset
    Args:
        df (pd.DataFrame): Input healthcare dataset
    Returns:
        dict: Dictionary containing basic statistics
    """
    analysis = {
        'total_records': len(df),
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'basic_stats': df.describe()
    }
    return analysis

if __name__ == "__main__":
    # Test data loading
    try:
        df = load_healthcare_data()
        analysis = initial_data_analysis(df)
        print("Data loaded successfully!")
        print(f"Total records: {analysis['total_records']}")
    except Exception as e:
        print(f"Error loading data: {e}")