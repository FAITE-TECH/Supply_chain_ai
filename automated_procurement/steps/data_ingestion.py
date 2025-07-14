import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict
import os
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class ProcurementDataIngestion:
    """Class for handling procurement data ingestion and initial validation"""
    
    def __init__(self, raw_data_path: str = "../data/raw", processed_data_path: str = "../data/processed"):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized data ingestion with raw path: {self.raw_data_path}")

    def load_raw_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load raw datasets from CSV files
        
        Returns:
            Tuple of (orders_df, inventory_df, fulfillment_df)
        """
        try:
            logger.info("Loading raw datasets...")
            
            # Load orders and shipments data
            orders_path = self.raw_data_path / "orders_and_shipments.csv"
            orders_df = pd.read_csv(orders_path)
            logger.info(f"Loaded orders dataset: {orders_df.shape}")
            
            # Load inventory data
            inventory_path = self.raw_data_path / "inventory.csv"
            inventory_df = pd.read_csv(inventory_path)
            logger.info(f"Loaded inventory dataset: {inventory_df.shape}")
            
            # Load fulfillment data
            fulfillment_path = self.raw_data_path / "fulfillment.csv"
            fulfillment_df = pd.read_csv(fulfillment_path)
            logger.info(f"Loaded fulfillment dataset: {fulfillment_df.shape}")
            
            return orders_df, inventory_df, fulfillment_df
            
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            raise

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean column names by removing leading/trailing whitespace
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cleaned column names
        """
        df_clean = df.copy()
        df_clean.columns = df_clean.columns.str.strip()
        return df_clean
    
    def validate_data_quality(self, df: pd.DataFrame, dataset_name: str) -> Dict:
        """
        Validate data quality and return quality metrics
        
        Args:
            df: DataFrame to validate
            dataset_name: Name of the dataset for logging
            
        Returns:
            Dictionary with quality metrics
        """
        logger.info(f"Validating data quality for {dataset_name}")
        
        quality_metrics = {
            'dataset_name': dataset_name,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Check for negative values in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        negative_values = {}
        for col in numeric_cols:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                negative_values[col] = negative_count
        
        quality_metrics['negative_values'] = negative_values
        
        # Log quality issues
        total_missing = sum(quality_metrics['missing_values'].values())
        if total_missing > 0:
            logger.warning(f"{dataset_name}: {total_missing} missing values found")
        
        if quality_metrics['duplicate_rows'] > 0:
            logger.warning(f"{dataset_name}: {quality_metrics['duplicate_rows']} duplicate rows found")
        
        if negative_values:
            logger.warning(f"{dataset_name}: Negative values found in columns: {list(negative_values.keys())}")
        
        return quality_metrics
    
    def standardize_data_types(self, orders_df: pd.DataFrame, inventory_df: pd.DataFrame, 
                              fulfillment_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Standardize data types across datasets
        
        Args:
            orders_df: Orders DataFrame
            inventory_df: Inventory DataFrame  
            fulfillment_df: Fulfillment DataFrame
            
        Returns:
            Tuple of standardized DataFrames
        """
        logger.info("Standardizing data types...")
        
        # Clean orders dataset
        orders_clean = orders_df.copy()
        
        # Convert date columns
        try:
            orders_clean['Order_Date'] = pd.to_datetime(
                orders_clean[['Order Year', 'Order Month', 'Order Day']].rename(
                    columns={'Order Year': 'year', 'Order Month': 'month', 'Order Day': 'day'}
                )
            )
            orders_clean['Shipment_Date'] = pd.to_datetime(
                orders_clean[['Shipment Year', 'Shipment Month', 'Shipment Day']].rename(
                    columns={'Shipment Year': 'year', 'Shipment Month': 'month', 'Shipment Day': 'day'}
                )
            )
        except Exception as e:
            logger.error(f"Error converting dates in orders: {e}")
        
        # Clean discount percentage
        orders_clean['Discount %'] = pd.to_numeric(orders_clean['Discount %'], errors='coerce')
        
        # Clean inventory dataset
        inventory_clean = inventory_df.copy()
        
        # Convert YearMonth to datetime
        try:
            inventory_clean['Year_Month_Date'] = pd.to_datetime(
                inventory_clean['Year Month'].astype(str), format='%Y%m'
            )
        except Exception as e:
            logger.error(f"Error converting year-month in inventory: {e}")
        
        # Fulfillment dataset is already clean
        fulfillment_clean = fulfillment_df.copy()
        
        logger.info("Data type standardization completed")
        return orders_clean, inventory_clean, fulfillment_clean
    
    def create_data_lineage(self, quality_metrics: list) -> dict:
        """
        Create data lineage information for tracking
        
        Args:
            quality_metrics: List of quality metrics from validation
            
        Returns:
            Dictionary with lineage information
        """
        lineage = {
            'ingestion_timestamp': datetime.now().isoformat(),
            'datasets_processed': len(quality_metrics),
            'total_records': sum(m['total_rows'] for m in quality_metrics),
            'quality_summary': quality_metrics,
            'ingestion_status': 'completed'
        }
        
        return lineage
    
    def save_processed_data(self, orders_df: pd.DataFrame, inventory_df: pd.DataFrame, 
                           fulfillment_df: pd.DataFrame, lineage: dict) -> None:
        """
        Save processed datasets and lineage information
        
        Args:
            orders_df: Processed orders DataFrame
            inventory_df: Processed inventory DataFrame
            fulfillment_df: Processed fulfillment DataFrame
            lineage: Data lineage information
        """
        logger.info("Saving processed datasets...")
        
        try:
            # Save processed datasets
            orders_df.to_csv(self.processed_data_path / "orders_ingested.csv", index=False)
            inventory_df.to_csv(self.processed_data_path / "inventory_ingested.csv", index=False)
            fulfillment_df.to_csv(self.processed_data_path / "fulfillment_ingested.csv", index=False)
            
            # Save lineage information with numpy-safe conversion
            import json
            import numpy as np
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            with open(self.processed_data_path / "data_lineage.json", 'w') as f:
                json.dump(lineage, f, indent=2, default=convert_numpy)
            
            logger.info(f"Processed datasets saved to {self.processed_data_path}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")

    def run_ingestion_pipeline(self) -> dict:
        """
        Run the complete data ingestion pipeline
        
        Returns:
            Dictionary with pipeline results
        """
        logger.info("Starting data ingestion pipeline...")
        
        try:
            # Load raw datasets
            orders_raw, inventory_raw, fulfillment_raw = self.load_raw_datasets()
            
            # Clean column names
            orders_clean = self.clean_column_names(orders_raw)
            inventory_clean = self.clean_column_names(inventory_raw)
            fulfillment_clean = self.clean_column_names(fulfillment_raw)
            
            # Validate data quality
            quality_metrics = []
            quality_metrics.append(self.validate_data_quality(orders_clean, "orders"))
            quality_metrics.append(self.validate_data_quality(inventory_clean, "inventory"))
            quality_metrics.append(self.validate_data_quality(fulfillment_clean, "fulfillment"))
            
            # Standardize data types
            orders_final, inventory_final, fulfillment_final = self.standardize_data_types(
                orders_clean, inventory_clean, fulfillment_clean
            )
            
            # Create lineage
            lineage = self.create_data_lineage(quality_metrics)
            
            # Save processed data
            self.save_processed_data(orders_final, inventory_final, fulfillment_final, lineage)
            
            logger.info("Data ingestion pipeline completed successfully")
            
            return {
                'status': 'success',
                'datasets_processed': 3,
                'total_records': sum(m['total_rows'] for m in quality_metrics),
                'quality_metrics': quality_metrics,
                'output_path': str(self.processed_data_path)
            }
            
        except Exception as e:
            logger.error(f"Data ingestion pipeline failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
        
def main():
    """Main function to run the data ingestion pipeline"""
    try:
        # Initialize ingestion class
        ingestion = ProcurementDataIngestion()
        
        # Run pipeline
        results = ingestion.run_ingestion_pipeline()
        
        if results['status'] == 'success':
            print(f"✅ Data ingestion completed successfully!")
            print(f"📊 Processed {results['datasets_processed']} datasets")
            print(f"📝 Total records: {results['total_records']:,}")
            print(f"💾 Output saved to: {results['output_path']}")
        else:
            print(f"❌ Data ingestion failed: {results['error']}")
            
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        print(f"❌ Fatal error: {e}")


if __name__ == "__main__":
    main()