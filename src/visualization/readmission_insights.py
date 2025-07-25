import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent directory to path to import from other modules
sys.path.append(str(Path(__file__).parent.parent))
from data_processing.load_data import load_healthcare_data
from data_processing.feature_engineering import engineer_features


class ReadmissionVisualizer:
    """
    Class for visualizing hospital readmission data and generating insights
    """
    
    def __init__(self, df=None):
        """
        Initialize the visualizer with an optional dataframe
        
        Args:
            df (pd.DataFrame, optional): Preprocessed dataframe with engineered features
        """
        self.df = df
        self.output_dir = Path('output')
        self.output_dir.mkdir(exist_ok=True)
    
    def load_data(self):
        """
        Load and prepare data if not provided during initialization
        """
        if self.df is None:
            # Load raw data
            raw_df = load_healthcare_data()
            
            # Apply feature engineering
            self.df = engineer_features(raw_df)
            
        return self.df
    
    def plot_readmission_by_category(self, category, top_n=10, save_fig=True):
        """
        Plot readmission rates by a categorical variable
        
        Args:
            category (str): Column name to group by
            top_n (int): Number of top categories to display
            save_fig (bool): Whether to save the figure
        """
        # Ensure data is loaded
        if self.df is None:
            self.load_data()
        
        # Calculate readmission rate by category
        readmission_by_category = self.df.groupby(category)['Readmitted_within_30'].agg(['mean', 'count'])
        readmission_by_category.columns = ['Readmission_Rate', 'Count']
        
        # Filter for categories with sufficient data points
        min_count = 10  # Minimum number of patients in a category
        filtered_data = readmission_by_category[readmission_by_category['Count'] >= min_count]
        
        # Sort by readmission rate and get top N
        top_categories = filtered_data.sort_values('Readmission_Rate', ascending=False).head(top_n)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=top_categories.index, y='Readmission_Rate', data=top_categories.reset_index())
        plt.title(f'Top {top_n} {category} by Readmission Rate')
        plt.xlabel(category)
        plt.ylabel('Readmission Rate')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for i, v in enumerate(top_categories['Readmission_Rate']):
            ax.text(i, v + 0.01, f'{v:.2f}', ha='center')
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_fig:
            plt.savefig(self.output_dir / f'readmission_by_{category.lower().replace(" ", "_")}.png')
        
        plt.show()
        
        return top_categories
    
    def plot_length_of_stay_vs_readmission(self, bins=10, save_fig=True):
        """
        Plot relationship between length of stay and readmission rate
        
        Args:
            bins (int): Number of bins for length of stay
            save_fig (bool): Whether to save the figure
        """
        # Ensure data is loaded
        if self.df is None:
            self.load_data()
        
        # Create length of stay bins
        self.df['LOS_Bin'] = pd.cut(self.df['Length_of_Stay'], bins=bins)
        
        # Calculate readmission rate by length of stay bin
        los_readmission = self.df.groupby('LOS_Bin')['Readmitted_within_30'].agg(['mean', 'count'])
        los_readmission.columns = ['Readmission_Rate', 'Count']
        
        # Create plot
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=los_readmission.index.astype(str), y='Readmission_Rate', data=los_readmission.reset_index())
        plt.title('Readmission Rate by Length of Stay')
        plt.xlabel('Length of Stay (days)')
        plt.ylabel('Readmission Rate')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for i, v in enumerate(los_readmission['Readmission_Rate']):
            ax.text(i, v + 0.01, f'{v:.2f}', ha='center')
        
        # Add count as text below x-axis
        for i, v in enumerate(los_readmission['Count']):
            ax.text(i, -0.02, f'n={v}', ha='center', transform=ax.get_xaxis_transform())
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_fig:
            plt.savefig(self.output_dir / 'readmission_by_length_of_stay.png')
        
        plt.show()
        
        return los_readmission
    
    def plot_age_vs_readmission(self, save_fig=True):
        """
        Plot relationship between age and readmission rate
        
        Args:
            save_fig (bool): Whether to save the figure
        """
        # Ensure data is loaded
        if self.df is None:
            self.load_data()
        
        # Calculate readmission rate by age group
        age_readmission = self.df.groupby('Age_Group')['Readmitted_within_30'].mean().reset_index()
        
        # Create plot
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Age_Group', y='Readmitted_within_30', data=age_readmission)
        plt.title('Readmission Rate by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Readmission Rate')
        
        # Add value labels on bars
        for i, v in enumerate(age_readmission['Readmitted_within_30']):
            ax.text(i, v + 0.01, f'{v:.2f}', ha='center')
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_fig:
            plt.savefig(self.output_dir / 'readmission_by_age_group.png')
        
        plt.show()
        
        return age_readmission
    
    def plot_billing_vs_readmission(self, save_fig=True):
        """
        Plot relationship between billing amount and readmission rate
        
        Args:
            save_fig (bool): Whether to save the figure
        """
        # Ensure data is loaded
        if self.df is None:
            self.load_data()
        
        # Calculate readmission rate by billing category
        billing_readmission = self.df.groupby('Billing_Category')['Readmitted_within_30'].mean().reset_index()
        
        # Create plot
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Billing_Category', y='Readmitted_within_30', data=billing_readmission)
        plt.title('Readmission Rate by Billing Category')
        plt.xlabel('Billing Category')
        plt.ylabel('Readmission Rate')
        
        # Add value labels on bars
        for i, v in enumerate(billing_readmission['Readmitted_within_30']):
            ax.text(i, v + 0.01, f'{v:.2f}', ha='center')
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_fig:
            plt.savefig(self.output_dir / 'readmission_by_billing_category.png')
        
        plt.show()
        
        return billing_readmission
    
    def plot_correlation_heatmap(self, save_fig=True):
        """
        Plot correlation heatmap for numeric features
        
        Args:
            save_fig (bool): Whether to save the figure
        """
        # Ensure data is loaded
        if self.df is None:
            self.load_data()
        
        # Select numeric columns
        numeric_cols = ['Age', 'Length_of_Stay', 'Previous_Visits', 'Billing Amount', 
                       'Test_Result_Score', 'High_Risk_Doctor', 'Readmitted_within_30']
        numeric_df = self.df[numeric_cols]
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Heatmap of Numeric Features')
        plt.tight_layout()
        
        # Save figure if requested
        if save_fig:
            plt.savefig(self.output_dir / 'correlation_heatmap.png')
        
        plt.show()
        
        return corr_matrix
    
    def generate_business_insights(self):
        """
        Generate and print business insights based on the data
        
        Returns:
            dict: Dictionary containing business insights
        """
        # Ensure data is loaded
        if self.df is None:
            self.load_data()
        
        # Calculate overall readmission rate
        overall_rate = self.df['Readmitted_within_30'].mean()
        
        # Get top hospitals by readmission rate
        hospital_insights = self.plot_readmission_by_category('Hospital', save_fig=True)
        
        # Get top doctors by readmission rate
        doctor_insights = self.plot_readmission_by_category('Doctor', save_fig=True)
        
        # Get top medical conditions by readmission rate
        condition_insights = self.plot_readmission_by_category('Medical Condition', save_fig=True)
        
        # Get length of stay insights
        los_insights = self.plot_length_of_stay_vs_readmission(save_fig=True)
        
        # Get age group insights
        age_insights = self.plot_age_vs_readmission(save_fig=True)
        
        # Get billing category insights
        billing_insights = self.plot_billing_vs_readmission(save_fig=True)
        
        # Get correlation insights
        correlation_insights = self.plot_correlation_heatmap(save_fig=True)
        
        # Compile insights
        insights = {
            'overall_readmission_rate': overall_rate,
            'hospital_insights': hospital_insights,
            'doctor_insights': doctor_insights,
            'condition_insights': condition_insights,
            'los_insights': los_insights,
            'age_insights': age_insights,
            'billing_insights': billing_insights,
            'correlation_insights': correlation_insights
        }
        
        # Print key insights
        print("\n===== HOSPITAL READMISSION INSIGHTS =====\n")
        print(f"Overall 30-day readmission rate: {overall_rate:.2%}")
        
        print("\n----- Top 5 Hospitals with Highest Readmission Rates -----")
        for hospital, data in hospital_insights.head(5).iterrows():
            print(f"{hospital}: {data['Readmission_Rate']:.2%} (n={data['Count']})")
        
        print("\n----- Top 5 Medical Conditions with Highest Readmission Rates -----")
        for condition, data in condition_insights.head(5).iterrows():
            print(f"{condition}: {data['Readmission_Rate']:.2%} (n={data['Count']})")
        
        print("\n----- Key Risk Factors for Readmission -----")
        # Extract correlations with readmission
        readmission_corr = correlation_insights['Readmitted_within_30'].sort_values(ascending=False)
        for feature, corr in readmission_corr.items():
            if feature != 'Readmitted_within_30' and abs(corr) > 0.05:
                print(f"{feature}: Correlation = {corr:.3f}")
        
        return insights


def main():
    """
    Main function to demonstrate the ReadmissionVisualizer class
    """
    # Create visualizer
    visualizer = ReadmissionVisualizer()
    
    # Load data
    print("Loading and preparing data...")
    visualizer.load_data()
    
    # Generate business insights
    print("\nGenerating business insights...")
    visualizer.generate_business_insights()
    
    print("\nAnalysis complete! Check the 'output' directory for plots.")


if __name__ == "__main__":
    main()