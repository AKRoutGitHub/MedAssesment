import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

def create_length_of_stay(df):
    """
    Calculate length of stay by finding the difference between discharge and admission dates
    
    Args:
        df (pd.DataFrame): Input healthcare dataset
    
    Returns:
        pd.DataFrame: Dataset with Length_of_Stay feature added
    """
    # Convert date strings to datetime objects
    df['Date_of_Admission'] = pd.to_datetime(df['Date of Admission'])
    df['Discharge_Date'] = pd.to_datetime(df['Discharge Date'])
    
    # Calculate length of stay in days
    df['Length_of_Stay'] = (df['Discharge_Date'] - df['Date_of_Admission']).dt.days
    
    return df

def create_readmission_label(df, days_threshold=30):
    """
    Create readmission label based on whether a patient was readmitted within a specified
    number of days after discharge
    
    Args:
        df (pd.DataFrame): Input healthcare dataset
        days_threshold (int): Number of days to consider for readmission
    
    Returns:
        pd.DataFrame: Dataset with Readmitted_within_30 feature added
    """
    # Sort by patient name and admission date
    df = df.sort_values(['Name', 'Date_of_Admission'])
    
    # Initialize readmission column
    df['Readmitted_within_30'] = 0
    
    # Group by patient name
    for name, group in df.groupby('Name'):
        if len(group) > 1:  # Patient has multiple visits
            # For each visit except the last one
            for i in range(len(group) - 1):
                current_discharge = group.iloc[i]['Discharge_Date']
                next_admission = group.iloc[i + 1]['Date_of_Admission']
                
                # Calculate days between discharge and next admission
                days_diff = (next_admission - current_discharge).days
                
                # Mark as readmission if within threshold
                if days_diff <= days_threshold:
                    # Find the index in the original dataframe
                    idx = group.iloc[i].name
                    df.at[idx, 'Readmitted_within_30'] = 1
    
    return df

def create_previous_visits(df):
    """
    Calculate the number of previous visits for each patient
    
    Args:
        df (pd.DataFrame): Input healthcare dataset
    
    Returns:
        pd.DataFrame: Dataset with Previous_Visits feature added
    """
    # Sort by patient name and admission date
    df = df.sort_values(['Name', 'Date_of_Admission'])
    
    # Initialize previous visits column
    df['Previous_Visits'] = 0
    
    # Group by patient name
    for name, group in df.groupby('Name'):
        if len(group) > 1:  # Patient has multiple visits
            # For each visit except the first one
            for i in range(1, len(group)):
                # Find the index in the original dataframe
                idx = group.iloc[i].name
                df.at[idx, 'Previous_Visits'] = i
    
    return df

def create_age_group(df):
    """
    Create age groups: Child (0-17), Adult (18-64), Senior (65+)
    
    Args:
        df (pd.DataFrame): Input healthcare dataset
    
    Returns:
        pd.DataFrame: Dataset with Age_Group feature added
    """
    # Define age group bins and labels
    bins = [0, 17, 64, 150]
    labels = ['Child', 'Adult', 'Senior']
    
    # Create age group column
    df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)
    
    return df

def create_billing_category(df):
    """
    Create billing amount categories
    
    Args:
        df (pd.DataFrame): Input healthcare dataset
    
    Returns:
        pd.DataFrame: Dataset with Billing_Category feature added
    """
    # Define billing amount quantiles
    quantiles = df['Billing Amount'].quantile([0, 0.25, 0.5, 0.75, 1.0])
    
    # Define billing category bins and labels
    bins = [quantiles[0], quantiles[0.25], quantiles[0.5], quantiles[0.75], quantiles[1.0]]
    labels = ['Low', 'Medium-Low', 'Medium-High', 'High']
    
    # Create billing category column
    df['Billing_Category'] = pd.cut(df['Billing Amount'], bins=bins, labels=labels, right=True)
    
    return df

def create_high_risk_doctor(df):
    """
    Identify high-risk doctors based on readmission rates
    
    Args:
        df (pd.DataFrame): Input healthcare dataset with Readmitted_within_30 feature
    
    Returns:
        pd.DataFrame: Dataset with High_Risk_Doctor feature added
    """
    # Calculate readmission rate by doctor
    doctor_readmission = df.groupby('Doctor')['Readmitted_within_30'].mean().reset_index()
    doctor_readmission.columns = ['Doctor', 'Readmission_Rate']
    
    # Calculate average readmission rate
    avg_readmission_rate = doctor_readmission['Readmission_Rate'].mean()
    
    # Define high-risk doctors as those with readmission rate above average
    high_risk_doctors = doctor_readmission[doctor_readmission['Readmission_Rate'] > avg_readmission_rate]['Doctor'].tolist()
    
    # Create high-risk doctor feature
    df['High_Risk_Doctor'] = df['Doctor'].apply(lambda x: 1 if x in high_risk_doctors else 0)
    
    return df

def create_test_result_score(df):
    """
    Convert test results to numeric scores
    
    Args:
        df (pd.DataFrame): Input healthcare dataset
    
    Returns:
        pd.DataFrame: Dataset with Test_Result_Score feature added
    """
    # Define mapping for test results
    test_result_mapping = {
        'Normal': 0,
        'Inconclusive': 1,
        'Abnormal': 2
    }
    
    # Create test result score column
    df['Test_Result_Score'] = df['Test Results'].map(test_result_mapping)
    
    return df

def engineer_features(df):
    """
    Apply all feature engineering functions to the dataset
    
    Args:
        df (pd.DataFrame): Input healthcare dataset
    
    Returns:
        pd.DataFrame: Dataset with all engineered features
    """
    # Create date-based features
    df = create_length_of_stay(df)
    
    # Create patient history features
    df = create_previous_visits(df)
    
    # Create categorical features
    df = create_age_group(df)
    if len(df) > 1:
        df = create_billing_category(df)
    # For single-row prediction, skip billing category binning
    df = create_test_result_score(df)
    
    # Create readmission label (must be after length_of_stay and previous_visits)
    df = create_readmission_label(df)
    
    # Create high-risk doctor feature (must be after readmission label)
    df = create_high_risk_doctor(df)
    
    return df

if __name__ == "__main__":
    # Test feature engineering
    from load_data import load_healthcare_data
    
    try:
        # Load data
        df = load_healthcare_data()
        print("Data loaded successfully!")
        print(f"Original shape: {df.shape}")
        
        # Apply feature engineering
        df_engineered = engineer_features(df)
        print("Feature engineering completed!")
        print(f"New shape: {df_engineered.shape}")
        
        # Display new features
        new_features = ['Length_of_Stay', 'Readmitted_within_30', 'Previous_Visits', 
                       'Age_Group', 'Billing_Category', 'High_Risk_Doctor', 'Test_Result_Score']
        print("\nSample of new features:")
        print(df_engineered[new_features].head())
        
    except Exception as e:
        print(f"Error in feature engineering: {e}")