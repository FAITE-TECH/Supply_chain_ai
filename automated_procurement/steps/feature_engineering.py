import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ProcurementFeatureEngineering:
    """Class for creating features for automated procurement model"""
    
    def __init__(self, processed_data_path: str = "../data/processed"):
        self.processed_data_path = Path(processed_data_path)
        self.features = {}
        logger.info(f"Initialized feature engineering with data path: {self.processed_data_path}")

    def load_ingested_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load ingested datasets
        
        Returns:
            Tuple of (orders_df, inventory_df, fulfillment_df)
        """
        try:
            logger.info("Loading ingested datasets...")
            
            orders_df = pd.read_csv(self.processed_data_path / "orders_ingested.csv")
            inventory_df = pd.read_csv(self.processed_data_path / "inventory_ingested.csv")
            fulfillment_df = pd.read_csv(self.processed_data_path / "fulfillment_ingested.csv")
            
            # Check for required columns
            for col in ["Order_Date", "Shipment_Date"]:
                if col not in orders_df.columns:
                    raise KeyError(f"Column '{col}' not found in orders_ingested.csv. Please ensure data ingestion step creates and saves this column.")
            
            # Convert date columns back to datetime
            orders_df['Order_Date'] = pd.to_datetime(orders_df['Order_Date'])
            orders_df['Shipment_Date'] = pd.to_datetime(orders_df['Shipment_Date'])
            inventory_df['Year_Month_Date'] = pd.to_datetime(inventory_df['Year_Month_Date'])
            
            logger.info(f"Loaded datasets - Orders: {orders_df.shape}, Inventory: {inventory_df.shape}, Fulfillment: {fulfillment_df.shape}")
            return orders_df, inventory_df, fulfillment_df
            
        except Exception as e:
            logger.error(f"Error loading ingested data: {e}")
            raise

    def create_demand_features(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create demand-related features for procurement forecasting
        
        Args:
            orders_df: Orders DataFrame
            
        Returns:
            DataFrame with demand features
        """
        logger.info("Creating demand features...")
        
        # Add time-based features
        orders_df['Year'] = orders_df['Order_Date'].dt.year
        orders_df['Month'] = orders_df['Order_Date'].dt.month
        orders_df['Quarter'] = orders_df['Order_Date'].dt.quarter
        orders_df['Day_of_Week'] = orders_df['Order_Date'].dt.dayofweek
        orders_df['Day_of_Year'] = orders_df['Order_Date'].dt.dayofyear
        orders_df['Week_of_Year'] = orders_df['Order_Date'].dt.isocalendar().week
        orders_df['Is_Weekend'] = (orders_df['Day_of_Week'] >= 5).astype(int)
        orders_df['Is_Holiday_Season'] = orders_df['Month'].isin([11, 12]).astype(int)
        orders_df['Is_Back_to_School'] = orders_df['Month'].isin([8, 9]).astype(int)
        orders_df['Is_Summer'] = orders_df['Month'].isin([6, 7, 8]).astype(int)
        
        # Calculate demand patterns by product
        demand_features = orders_df.groupby('Product Name').agg({
            'Order Quantity': ['sum', 'mean', 'std', 'min', 'max', 'count'],
            'Gross Sales': ['sum', 'mean', 'std'],
            'Profit': ['sum', 'mean'],
            'Order ID': 'nunique',  # Number of unique orders
            'Is_Holiday_Season': 'mean',
            'Is_Weekend': 'mean',
            'Quarter': lambda x: x.mode().iloc[0] if not x.empty else 1  # Most common quarter
        }).round(4)
        
        # Flatten column names
        demand_features.columns = ['_'.join(col).strip() for col in demand_features.columns]
        demand_features = demand_features.reset_index()
        
        # Calculate advanced demand metrics
        demand_features['Demand_Variability'] = (demand_features['Order Quantity_std'] / 
                                                demand_features['Order Quantity_mean']).fillna(0)
        demand_features['Revenue_per_Unit'] = (demand_features['Gross Sales_sum'] / 
                                              demand_features['Order Quantity_sum']).fillna(0)
        demand_features['Profit_Margin'] = (demand_features['Profit_sum'] / 
                                           demand_features['Gross Sales_sum']).fillna(0)
        demand_features['Order_Frequency'] = demand_features['Order ID_nunique'] / 365  # Orders per day
        
        # Create demand trend features (month-over-month growth)
        monthly_demand = orders_df.groupby(['Product Name', 'Year', 'Month']).agg({
            'Order Quantity': 'sum'
        }).reset_index()
        
        monthly_demand['Year_Month'] = monthly_demand['Year'] * 100 + monthly_demand['Month']
        monthly_demand = monthly_demand.sort_values(['Product Name', 'Year_Month'])
        monthly_demand['Demand_Growth'] = monthly_demand.groupby('Product Name')['Order Quantity'].pct_change()
        
        # Aggregate trend features
        trend_features = monthly_demand.groupby('Product Name').agg({
            'Demand_Growth': ['mean', 'std'],
            'Order Quantity': lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0  # Linear trend slope
        }).round(4)
        
        trend_features.columns = ['Demand_Growth_Mean', 'Demand_Growth_Std', 'Demand_Trend_Slope']
        trend_features = trend_features.reset_index()
        
        # Merge trend features
        demand_features = demand_features.merge(trend_features, on='Product Name', how='left')
        
        logger.info(f"Created demand features for {len(demand_features)} products")
        return demand_features 
    
    def create_inventory_features(self, inventory_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create inventory-related features
        
        Args:
            inventory_df: Inventory DataFrame
            
        Returns:
            DataFrame with inventory features
        """
        logger.info("Creating inventory features...")
        
        # Calculate inventory patterns
        inventory_features = inventory_df.groupby('Product Name').agg({
            'Warehouse Inventory': ['mean', 'std', 'min', 'max', 'count'],
            'Inventory Cost Per Unit': ['mean', 'std'],
            'Year Month': ['min', 'max']  # Data range
        }).round(4)
        
        # Flatten column names
        inventory_features.columns = ['_'.join(col).strip() for col in inventory_features.columns]
        inventory_features = inventory_features.reset_index()
        
        # Calculate advanced inventory metrics
        inventory_features['Inventory_Variability'] = (inventory_features['Warehouse Inventory_std'] / 
                                                      inventory_features['Warehouse Inventory_mean']).fillna(0)
        inventory_features['Cost_Variability'] = (inventory_features['Inventory Cost Per Unit_std'] / 
                                                 inventory_features['Inventory Cost Per Unit_mean']).fillna(0)
        
        # Calculate stockout frequency
        stockout_analysis = inventory_df.groupby('Product Name').agg({
            'Warehouse Inventory': lambda x: (x == 0).sum() / len(x)  # Stockout frequency
        }).reset_index()
        stockout_analysis.columns = ['Product Name', 'Stockout_Frequency']
        
        # Calculate low stock frequency (less than 5 units)
        low_stock_analysis = inventory_df.groupby('Product Name').agg({
            'Warehouse Inventory': lambda x: ((x > 0) & (x <= 5)).sum() / len(x)
        }).reset_index()
        low_stock_analysis.columns = ['Product Name', 'Low_Stock_Frequency']
        
        # Merge stockout features
        inventory_features = inventory_features.merge(stockout_analysis, on='Product Name', how='left')
        inventory_features = inventory_features.merge(low_stock_analysis, on='Product Name', how='left')
        
        # Get current inventory (latest month)
        latest_inventory = inventory_df.loc[inventory_df.groupby('Product Name')['Year Month'].idxmax()]
        current_inventory = latest_inventory[['Product Name', 'Warehouse Inventory', 'Inventory Cost Per Unit']]
        current_inventory.columns = ['Product Name', 'Current_Inventory', 'Current_Unit_Cost']
        
        inventory_features = inventory_features.merge(current_inventory, on='Product Name', how='left')
        
        logger.info(f"Created inventory features for {len(inventory_features)} products")
        return inventory_features
    
    def create_supplier_features(self, orders_df: pd.DataFrame, fulfillment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create supplier and fulfillment-related features
        
        Args:
            orders_df: Orders DataFrame
            fulfillment_df: Fulfillment DataFrame
            
        Returns:
            DataFrame with supplier features
        """
        logger.info("Creating supplier and fulfillment features...")
        
        # Calculate actual vs scheduled shipment performance
        orders_df['Actual_Shipment_Days'] = (orders_df['Shipment_Date'] - orders_df['Order_Date']).dt.days
        orders_df['Shipment_Delay'] = orders_df['Actual_Shipment_Days'] - orders_df['Shipment Days - Scheduled']
        orders_df['On_Time_Delivery'] = (orders_df['Shipment_Delay'] <= 0).astype(int)
        orders_df['Early_Delivery'] = (orders_df['Shipment_Delay'] < -1).astype(int)
        orders_df['Late_Delivery'] = (orders_df['Shipment_Delay'] > 1).astype(int)
        
        # Aggregate shipment performance by product
        shipment_features = orders_df.groupby('Product Name').agg({
            'Actual_Shipment_Days': ['mean', 'std'],
            'Shipment_Delay': ['mean', 'std'],
            'On_Time_Delivery': 'mean',
            'Early_Delivery': 'mean',
            'Late_Delivery': 'mean'
        }).round(4)
        
        # Flatten column names
        shipment_features.columns = ['_'.join(col).strip() for col in shipment_features.columns]
        shipment_features = shipment_features.reset_index()
        
        # Add fulfillment data
        fulfillment_clean = fulfillment_df.copy()
        fulfillment_clean.columns = ['Product Name', 'Warehouse_Fulfillment_Days']
        
        # Merge fulfillment features
        supplier_features = shipment_features.merge(fulfillment_clean, on='Product Name', how='left')
        
        # Calculate reliability metrics
        supplier_features['Delivery_Reliability'] = 1 - supplier_features['Late_Delivery_mean']
        supplier_features['Fulfillment_Consistency'] = 1 / (1 + supplier_features['Actual_Shipment_Days_std'])
        
        logger.info(f"Created supplier features for {len(supplier_features)} products")
        return supplier_features
    
    def calculate_procurement_metrics(self, demand_features: pd.DataFrame, 
                                    inventory_features: pd.DataFrame,
                                    supplier_features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate key procurement metrics for the AI model
        
        Args:
            demand_features: Demand features DataFrame
            inventory_features: Inventory features DataFrame
            supplier_features: Supplier features DataFrame
            
        Returns:
            DataFrame with procurement metrics
        """
        logger.info("Calculating procurement metrics...")
        
        # Merge all feature sets
        procurement_features = demand_features.merge(inventory_features, on='Product Name', how='outer')
        procurement_features = procurement_features.merge(supplier_features, on='Product Name', how='outer')
        
        # Fill missing values
        procurement_features = procurement_features.fillna(0)
        
        # Calculate advanced procurement metrics
        
        # 1. Safety Stock Calculation (95% service level)
        procurement_features['Safety_Stock'] = (1.65 * procurement_features['Order Quantity_std'] * 
                                               np.sqrt(procurement_features['Warehouse_Fulfillment_Days'] / 30)).fillna(0)
        
        # 2. Reorder Point Calculation
        procurement_features['Daily_Demand'] = procurement_features['Order Quantity_mean'] / 30
        procurement_features['Reorder_Point'] = (procurement_features['Daily_Demand'] * 
                                                procurement_features['Warehouse_Fulfillment_Days'] + 
                                                procurement_features['Safety_Stock']).fillna(0)
        
        # 3. Economic Order Quantity (EOQ)
        setup_cost = 50  # Assumed fixed ordering cost
        holding_cost_rate = 0.2  # 20% of unit cost
        
        procurement_features['Annual_Demand'] = procurement_features['Order Quantity_sum'] * (365 / 365)  # Annualized
        procurement_features['EOQ'] = np.sqrt((2 * procurement_features['Annual_Demand'] * setup_cost) / 
                                             (procurement_features['Current_Unit_Cost'] * holding_cost_rate)).fillna(0)
        
        # 4. Inventory Days Supply
        procurement_features['Inventory_Days_Supply'] = (procurement_features['Current_Inventory'] / 
                                                        procurement_features['Daily_Demand']).fillna(0)
        
        # 5. ABC Classification based on revenue
        procurement_features['Revenue_Rank'] = procurement_features['Gross Sales_sum'].rank(method='dense', ascending=False)
        total_products = len(procurement_features)
        
        procurement_features['ABC_Category'] = 'C'
        procurement_features.loc[procurement_features['Revenue_Rank'] <= total_products * 0.2, 'ABC_Category'] = 'A'
        procurement_features.loc[(procurement_features['Revenue_Rank'] > total_products * 0.2) & 
                                (procurement_features['Revenue_Rank'] <= total_products * 0.5), 'ABC_Category'] = 'B'
        
        # 6. Stock Status Classification
        procurement_features['Stock_Status'] = 'Normal'
        procurement_features.loc[procurement_features['Current_Inventory'] <= procurement_features['Reorder_Point'], 'Stock_Status'] = 'Reorder_Required'
        procurement_features.loc[procurement_features['Current_Inventory'] == 0, 'Stock_Status'] = 'Stockout'
        procurement_features.loc[procurement_features['Current_Inventory'] > procurement_features['Reorder_Point'] * 2, 'Stock_Status'] = 'Overstock'
        
        # 7. Procurement Priority Score (0-100)
        # Higher score = higher priority for procurement attention
        
        # Normalize components (0-1 scale)
        stockout_weight = procurement_features['Stockout_Frequency'] * 40
        demand_growth_weight = np.clip(procurement_features['Demand_Growth_Mean'] * 20, 0, 20)
        revenue_weight = (1 - (procurement_features['Revenue_Rank'] / total_products)) * 25
        reliability_weight = (1 - procurement_features['Delivery_Reliability']) * 15
        
        procurement_features['Procurement_Priority_Score'] = (stockout_weight + demand_growth_weight + 
                                                             revenue_weight + reliability_weight)
        
        # 8. Procurement Action Recommendations
        procurement_features['Recommended_Action'] = 'Monitor'
        
        # High priority actions
        procurement_features.loc[procurement_features['Stock_Status'] == 'Stockout', 'Recommended_Action'] = 'Urgent_Reorder'
        procurement_features.loc[procurement_features['Stock_Status'] == 'Reorder_Required', 'Recommended_Action'] = 'Reorder_Soon'
        procurement_features.loc[procurement_features['Stock_Status'] == 'Overstock', 'Recommended_Action'] = 'Reduce_Orders'
        
        # Adjust based on demand trends
        procurement_features.loc[(procurement_features['Demand_Growth_Mean'] > 0.1) & 
                                (procurement_features['Stock_Status'] == 'Normal'), 'Recommended_Action'] = 'Increase_Stock'
        procurement_features.loc[(procurement_features['Demand_Growth_Mean'] < -0.1) & 
                                (procurement_features['Stock_Status'] == 'Normal'), 'Recommended_Action'] = 'Decrease_Stock'
        
        logger.info(f"Calculated procurement metrics for {len(procurement_features)} products")
        return procurement_features
    
    def create_ml_features(self, procurement_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create features specifically for machine learning models
        
        Args:
            procurement_df: Procurement features DataFrame
            
        Returns:
            Tuple of (ml_features_df, feature_names)
        """
        logger.info("Creating ML-ready features...")
        
        ml_df = procurement_df.copy()
        
        # Encode categorical variables
        label_encoders = {}
        categorical_columns = ['ABC_Category', 'Stock_Status', 'Recommended_Action']
        
        for col in categorical_columns:
            if col in ml_df.columns:
                le = LabelEncoder()
                ml_df[f'{col}_encoded'] = le.fit_transform(ml_df[col].astype(str))
                label_encoders[col] = le
        
        # Select numerical features for ML
        numerical_features = [
            'Order Quantity_mean', 'Order Quantity_std', 'Demand_Variability',
            'Revenue_per_Unit', 'Profit_Margin', 'Order_Frequency',
            'Demand_Growth_Mean', 'Demand_Trend_Slope',
            'Current_Inventory', 'Stockout_Frequency', 'Low_Stock_Frequency',
            'Warehouse_Fulfillment_Days', 'On_Time_Delivery_mean', 'Delivery_Reliability',
            'Safety_Stock', 'Reorder_Point', 'EOQ', 'Inventory_Days_Supply',
            'Procurement_Priority_Score'
        ]
        
        # Add encoded categorical features
        encoded_features = [f'{col}_encoded' for col in categorical_columns if f'{col}_encoded' in ml_df.columns]
        
        # Final feature set
        feature_columns = [col for col in numerical_features if col in ml_df.columns] + encoded_features
        
        # Create final ML dataset
        ml_features = ml_df[['Product Name'] + feature_columns].copy()
        
        # Handle any remaining missing values and infinite values
        ml_features[feature_columns] = ml_features[feature_columns].replace([np.inf, -np.inf], np.nan)
        ml_features[feature_columns] = ml_features[feature_columns].fillna(0)
        
        # Feature scaling (standardization)
        scaler = StandardScaler()
        ml_features[feature_columns] = scaler.fit_transform(ml_features[feature_columns])
        
        logger.info(f"Created {len(feature_columns)} ML features for {len(ml_features)} products")
        return ml_features, feature_columns
    
    def save_engineered_features(self, procurement_df: pd.DataFrame, ml_features: pd.DataFrame, 
                                feature_names: List[str]) -> None:
        """
        Save engineered features to files
        
        Args:
            procurement_df: Complete procurement features DataFrame
            ml_features: ML-ready features DataFrame
            feature_names: List of feature names
        """
        logger.info("Saving engineered features...")
        
        try:
            # Save complete procurement features
            procurement_df.to_csv(self.processed_data_path / "procurement_features.csv", index=False)
            
            # Save ML-ready features
            ml_features.to_csv(self.processed_data_path / "ml_features.csv", index=False)
            
            # Save feature metadata
            feature_metadata = {
                'feature_names': feature_names,
                'total_features': len(feature_names),
                'total_products': len(ml_features),
                'engineering_timestamp': datetime.now().isoformat()
            }
            
            import json
            with open(self.processed_data_path / "feature_metadata.json", 'w') as f:
                json.dump(feature_metadata, f, indent=2)
            
            logger.info(f"Features saved to {self.processed_data_path}")
            
        except Exception as e:
            logger.error(f"Error saving features: {e}")
            raise

    def run_feature_engineering_pipeline(self) -> Dict:
        """
        Run the complete feature engineering pipeline
        
        Returns:
            Dictionary with pipeline results
        """
        logger.info("Starting feature engineering pipeline...")
        
        try:
            # Load ingested data
            orders_df, inventory_df, fulfillment_df = self.load_ingested_data()
            
            # Create individual feature sets
            demand_features = self.create_demand_features(orders_df)
            inventory_features = self.create_inventory_features(inventory_df)
            supplier_features = self.create_supplier_features(orders_df, fulfillment_df)
            
            # Calculate procurement metrics
            procurement_features = self.calculate_procurement_metrics(
                demand_features, inventory_features, supplier_features
            )
            
            # Create ML-ready features
            ml_features, feature_names = self.create_ml_features(procurement_features)
            
            # Save all features
            self.save_engineered_features(procurement_features, ml_features, feature_names)
            
            logger.info("Feature engineering pipeline completed successfully")
            
            return {
                'status': 'success',
                'total_products': len(procurement_features),
                'total_features': len(feature_names),
                'feature_categories': {
                    'demand_features': len([f for f in feature_names if 'Order' in f or 'Demand' in f]),
                    'inventory_features': len([f for f in feature_names if 'Inventory' in f or 'Stock' in f]),
                    'supplier_features': len([f for f in feature_names if 'Delivery' in f or 'Fulfillment' in f]),
                    'procurement_metrics': len([f for f in feature_names if any(x in f for x in ['EOQ', 'Reorder', 'Safety', 'Priority'])])
                },
                'output_path': str(self.processed_data_path)
            }
            
        except Exception as e:
            logger.error(f"Feature engineering pipeline failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
        
def main():
    """Main function to run the feature engineering pipeline"""
    try:
        # Initialize feature engineering class
        feature_eng = ProcurementFeatureEngineering()
        
        # Run pipeline
        results = feature_eng.run_feature_engineering_pipeline()
        
        if results['status'] == 'success':
            print(f"✅ Feature engineering completed successfully!")
            print(f"🎯 Processed {results['total_products']} products")
            print(f"🔧 Created {results['total_features']} features")
            print(f"📊 Feature breakdown:")
            for category, count in results['feature_categories'].items():
                print(f"   - {category}: {count} features")
            print(f"💾 Output saved to: {results['output_path']}")
        else:
            print(f"❌ Feature engineering failed: {results['error']}")
            
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        print(f"❌ Fatal error: {e}")


if __name__ == "__main__":
    main()