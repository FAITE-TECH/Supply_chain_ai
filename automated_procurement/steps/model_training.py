import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
import mlflow
import mlflow.sklearn

# Configure logging
logger = logging.getLogger(__name__)

class ProcurementModelTraining:
    """Class for training and selecting procurement ML models"""
    
    def __init__(self, processed_data_path: str = "data/processed", models_path: str = "models"):
        self.processed_data_path = Path(processed_data_path)
        self.models_path = Path(models_path)
        self.models_path.mkdir(parents=True, exist_ok=True)
        # Set MLflow experiment and tracking URI
        mlflow.set_tracking_uri("./mlruns")  # Store runs in the mlruns folder
        mlflow.set_experiment("ProcurementModelTraining")
        
        # Model configurations
        self.model_configs = {
            'demand_forecast': {
                'task_type': 'regression',
                'description': 'Predict future product demand based on historical patterns',
                'target_col': 'Order Quantity_mean',
                'metric': 'rmse',
                'business_impact': 'Inventory optimization and demand planning'
            },
            'stockout_risk': {
                'task_type': 'regression',
                'description': 'Predict probability of stockout events',
                'target_col': 'Stockout_Frequency',
                'metric': 'mae',
                'business_impact': 'Proactive stockout prevention'
            },
            'delivery_performance': {
                'task_type': 'regression',
                'description': 'Predict supplier delivery reliability',
                'target_col': 'Delivery_Reliability',
                'metric': 'r2',
                'business_impact': 'Supplier selection and management'
            },
            'procurement_priority': {
                'task_type': 'regression',
                'description': 'Predict procurement priority scores',
                'target_col': 'Procurement_Priority_Score',
                'metric': 'rmse',
                'business_impact': 'Resource allocation optimization'
            }
        }
        
        logger.info(f"Initialized model training with data path: {self.processed_data_path}")
        logger.info(f"Models will be saved to: {self.models_path}")
    
    def load_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load processed data for model training
        
        Returns:
            Tuple of (procurement_df, ml_features_df)
        """
        try:
            logger.info("Loading processed data for model training...")
            
            procurement_df = pd.read_csv(self.processed_data_path / "procurement_features.csv")
            
            # Check if ml_features.csv exists, if not create from procurement_df
            ml_features_path = self.processed_data_path / "ml_features.csv"
            if ml_features_path.exists():
                ml_features_df = pd.read_csv(ml_features_path)
            else:
                logger.warning("ml_features.csv not found, creating from procurement features")
                ml_features_df = self._create_ml_features(procurement_df)
            
            logger.info(f"Loaded procurement data: {procurement_df.shape}")
            logger.info(f"Loaded ML features: {ml_features_df.shape}")
            
            return procurement_df, ml_features_df
            
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise


    def _create_ml_features(self, procurement_df: pd.DataFrame) -> pd.DataFrame:
        """Create ML features from procurement dataframe"""
        
        # Select relevant features for ML models
        feature_columns = [
            'Order Quantity_mean', 'Order Quantity_std', 'Demand_Variability',
            'Revenue_per_Unit', 'Profit_Margin', 'Order_Frequency',
            'Demand_Growth_Mean', 'Demand_Trend_Slope',
            'Current_Inventory', 'Stockout_Frequency', 'Low_Stock_Frequency',
            'Warehouse_Fulfillment_Days', 'On_Time_Delivery_mean', 'Delivery_Reliability',
            'Safety_Stock', 'Reorder_Point', 'EOQ', 'Inventory_Days_Supply',
            'Procurement_Priority_Score'
        ]
        
        # Add encoded categorical variables
        if 'ABC_Category' in procurement_df.columns:
            le_abc = LabelEncoder()
            procurement_df['ABC_Category_encoded'] = le_abc.fit_transform(procurement_df['ABC_Category'].astype(str))
            feature_columns.append('ABC_Category_encoded')
        
        if 'Stock_Status' in procurement_df.columns:
            le_stock = LabelEncoder()
            procurement_df['Stock_Status_encoded'] = le_stock.fit_transform(procurement_df['Stock_Status'].astype(str))
            feature_columns.append('Stock_Status_encoded')
        
        # Select available features
        available_features = ['Product Name'] + [col for col in feature_columns if col in procurement_df.columns]
        ml_features_df = procurement_df[available_features].copy()
        
        # Fill missing values
        numeric_columns = ml_features_df.select_dtypes(include=[np.number]).columns
        ml_features_df[numeric_columns] = ml_features_df[numeric_columns].fillna(0)
        
        return ml_features_df
    
    def get_base_models(self) -> Dict[str, Any]:
        """Define base models for comparison"""
        models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'decision_tree': DecisionTreeRegressor(random_state=42, max_depth=10),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'extra_trees': ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
            'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbosity=-1)
        }
        return models
    
    def perform_feature_selection(self, X: pd.DataFrame, y: np.ndarray, method: str = 'selectkbest', k: int = 15) -> Tuple[np.ndarray, List[str]]:
        """
        Perform feature selection using different methods
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Selection method ('selectkbest', 'rfe', or 'all')
            k: Number of features to select
            
        Returns:
            Tuple of (selected_features_array, selected_feature_names)
        """
        if method == 'selectkbest':
            selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        elif method == 'rfe':
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=min(k, X.shape[1]))
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
        else:
            # Use all features
            X_selected = X.values
            selected_features = X.columns.tolist()
        
        return X_selected, selected_features
    

    def evaluate_model(self, model: Any, X_train: np.ndarray, X_test: np.ndarray, 
                      y_train: np.ndarray, y_test: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Train and evaluate a single model
        
        Args:
            model: ML model to train
            X_train, X_test: Training and testing features
            y_train, y_test: Training and testing targets
            
        Returns:
            Tuple of (metrics_dict, predictions)
        """
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return metrics, y_pred
    
    def train_models_for_task(self, X: pd.DataFrame, y: np.ndarray, task_name: str, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train and evaluate multiple models for a specific task
        
        Args:
            X: Feature matrix
            y: Target variable
            task_name: Name of the prediction task
            test_size: Fraction of data to use for testing
            
        Returns:
            Dictionary with model results
        """
        logger.info(f"Training models for task: {task_name}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Feature selection
        X_train_selected, selected_features = self.perform_feature_selection(
            X_train, y_train, method='selectkbest', k=15
        )
        
        # Apply same selection to test set
        feature_indices = [X.columns.tolist().index(f) for f in selected_features]
        X_test_selected = X_test.iloc[:, feature_indices].values
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # Train models
        models = self.get_base_models()
        results = {}
        
        for name, model in models.items():
            try:
                with mlflow.start_run(run_name=f"{task_name}_{name}"):
                    # Log comprehensive parameters
                    mlflow.log_param("model_type", name)
                    mlflow.log_param("task", task_name)
                    mlflow.log_param("target_column", self.model_configs[task_name]['target_col'])
                    mlflow.log_param("training_samples", len(X_train_scaled))
                    mlflow.log_param("test_samples", len(X_test_scaled))
                    mlflow.log_param("num_features", len(selected_features))
                    mlflow.log_param("feature_selection_method", "selectkbest")
                    mlflow.log_param("test_size", test_size)
                    
                    # Log model hyperparameters
                    model_params = model.get_params()
                    for param_name, param_value in model_params.items():
                        mlflow.log_param(f"model_{param_name}", param_value)
                    
                    # Evaluate model
                    metrics, y_pred = self.evaluate_model(
                        model, X_train_scaled, X_test_scaled, y_train, y_test
                    )
                    
                    # Store results
                    results[name] = {
                        'model': model,
                        'metrics': metrics,
                        'predictions': y_pred,
                        'scaler': scaler,
                        'selected_features': selected_features
                    }
                    
                    # Log comprehensive metrics
                    mlflow.log_metric("rmse", metrics['rmse'])
                    mlflow.log_metric("r2", metrics['r2'])
                    mlflow.log_metric("mae", metrics['mae'])
                    
                    # Log additional metrics for better tracking
                    mlflow.log_metric("mean_target", np.mean(y_test))
                    mlflow.log_metric("std_target", np.std(y_test))
                    mlflow.log_metric("mean_prediction", np.mean(y_pred))
                    mlflow.log_metric("std_prediction", np.std(y_pred))
                    
                    # Log feature names for reproducibility
                    mlflow.log_dict({"selected_features": selected_features}, "selected_features.json")
                    
                    logger.info(f"{name} - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
                    
                    # Log model artifact with proper signature
                    try:
                        # Log model with compatible API
                        import warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=FutureWarning)
                            warnings.filterwarnings("ignore", message=".*artifact_path.*")
                            mlflow.sklearn.log_model(
                                model,
                                "model",
                                signature=mlflow.models.infer_signature(X_train_scaled, y_pred),
                                input_example=X_train_scaled[:5]
                            )
                        
                        # Register model separately
                        try:
                            mlflow.register_model(
                                model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
                                name=f"{task_name}_{name}"
                            )
                        except Exception as reg_e:
                            logger.warning(f"Could not register model: {reg_e}")
                            
                    except Exception as e:
                        logger.warning(f"Could not log model to MLflow: {e}")
                        # Continue without MLflow logging
                
            except Exception as e:
                logger.error(f"Failed to train {name} for {task_name}: {e}")
                continue
        
        return results
    

    def optimize_hyperparameters(self, X: pd.DataFrame, y: np.ndarray, task_name: str, 
                                model_names: List[str] = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters for selected models
        
        Args:
            X: Feature matrix
            y: Target variable  
            task_name: Name of the prediction task
            model_names: List of model names to optimize
            
        Returns:
            Dictionary with optimized model results
        """
        if model_names is None:
            model_names = ['random_forest', 'xgboost', 'gradient_boosting']
        
        logger.info(f"Optimizing hyperparameters for {task_name}: {model_names}")
        
        # Parameter grids for optimization
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'xgboost': {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        }
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Feature selection and scaling
        X_train_selected, selected_features = self.perform_feature_selection(
            X_train, y_train, method='selectkbest', k=15
        )
        
        feature_indices = [X.columns.tolist().index(f) for f in selected_features]
        X_test_selected = X_test.iloc[:, feature_indices].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        optimized_models = {}
        base_models = self.get_base_models()
        
        for model_name in model_names:
            if model_name in param_grids and model_name in base_models:
                logger.info(f"Optimizing {model_name}")
                
                try:
                    # Grid search with cross-validation
                    grid_search = GridSearchCV(
                        base_models[model_name], 
                        param_grids[model_name],
                        cv=5,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1,
                        verbose=0
                    )
                    
                    grid_search.fit(X_train_scaled, y_train)
                    
                    # Evaluate best model
                    best_model = grid_search.best_estimator_
                    y_pred = best_model.predict(X_test_scaled)
                    
                    metrics = {
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'mae': mean_absolute_error(y_test, y_pred),
                        'r2': r2_score(y_test, y_pred)
                    }
                    
                    optimized_models[model_name] = {
                        'model': best_model,
                        'metrics': metrics,
                        'best_params': grid_search.best_params_,
                        'scaler': scaler,
                        'selected_features': selected_features
                    }
                    
                    logger.info(f"{model_name} optimized - RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Failed to optimize {model_name}: {e}")
        
        return optimized_models
    

    def select_best_model(self, base_results: Dict, optimized_results: Dict, config: Dict) -> Tuple[str, Dict]:
        """
        Select the best model based on the primary metric
        
        Args:
            base_results: Results from base model training
            optimized_results: Results from hyperparameter optimization
            config: Task configuration
            
        Returns:
            Tuple of (best_model_name, best_model_result)
        """
        primary_metric = config['metric']
        
        # Combine all candidates
        all_candidates = {}
        
        # Add base models
        for name, result in base_results.items():
            all_candidates[f"base_{name}"] = result
        
        # Add optimized models
        for name, result in optimized_results.items():
            all_candidates[f"optimized_{name}"] = result
        
        # Select best based on primary metric
        if primary_metric == 'r2':
            best_name = max(all_candidates.keys(), key=lambda x: all_candidates[x]['metrics']['r2'])
        else:
            best_name = min(all_candidates.keys(), key=lambda x: all_candidates[x]['metrics'][primary_metric])
        
        return best_name, all_candidates[best_name]
    

    def save_model(self, model_result: Dict, task_name: str, model_name: str, config: Dict) -> str:
        """
        Save trained model and associated artifacts
        
        Args:
            model_result: Model training result
            task_name: Name of the prediction task
            model_name: Name of the selected model
            config: Task configuration
            
        Returns:
            Path to saved model directory
        """
        # Create task-specific directory
        task_path = self.models_path / task_name
        task_path.mkdir(exist_ok=True)
        
        # Save model
        model_file = task_path / 'model.pkl'
        joblib.dump(model_result['model'], model_file)
        
        # Save scaler
        scaler_file = task_path / 'scaler.pkl'
        joblib.dump(model_result['scaler'], scaler_file)
        
        # Save metadata
        metadata = {
            'task_name': task_name,
            'model_name': model_name,
            'selected_features': model_result['selected_features'],
            'metrics': model_result['metrics'],
            'config': config,
            'training_timestamp': datetime.now().isoformat(),
            'model_file': str(model_file),
            'scaler_file': str(scaler_file)
        }
        
        metadata_file = task_path / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {task_path}")
        return str(task_path)
    

    def run_model_training_pipeline(self) -> Dict:
        """
        Run the complete model training pipeline for all tasks
        
        Returns:
            Dictionary with training results
        """
        logger.info("Starting model training pipeline...")
        
        try:
            # Load data
            procurement_df, ml_features_df = self.load_processed_data()
            
            # Prepare feature matrix
            feature_columns = [col for col in ml_features_df.columns if col != 'Product Name']
            X = ml_features_df[feature_columns].fillna(0)
            
            pipeline_results = {
                'trained_models': {},
                'performance_summary': [],
                'training_timestamp': datetime.now().isoformat()
            }
            
            # Train models for each task
            for task_name, config in self.model_configs.items():
                target_col = config['target_col']
                
                if target_col in procurement_df.columns:
                    logger.info(f"Processing task: {task_name}")
                    
                    # Prepare target variable
                    y = procurement_df[target_col].values
                    
                    # Remove rows with missing target values
                    valid_mask = ~np.isnan(y)
                    X_clean = X[valid_mask]
                    y_clean = y[valid_mask]
                    
                    if len(y_clean) == 0:
                        logger.warning(f"No valid data for task {task_name}")
                        continue
                    
                    logger.info(f"Training with {len(y_clean)} samples for {task_name}")
                    
                    # Train base models
                    base_results = self.train_models_for_task(X_clean, y_clean, task_name)
                    
                    # Optimize top models
                    optimized_results = self.optimize_hyperparameters(X_clean, y_clean, task_name)
                    
                    # Select best model
                    best_model_name, best_model_result = self.select_best_model(
                        base_results, optimized_results, config
                    )
                    
                    # Save best model
                    model_path = self.save_model(best_model_result, task_name, best_model_name, config)
                    
                    # Store results
                    pipeline_results['trained_models'][task_name] = {
                        'best_model': best_model_name,
                        'metrics': best_model_result['metrics'],
                        'model_path': model_path,
                        'config': config
                    }
                    
                    # Add to performance summary
                    pipeline_results['performance_summary'].append({
                        'task': task_name,
                        'model': best_model_name,
                        'rmse': best_model_result['metrics']['rmse'],
                        'r2': best_model_result['metrics']['r2'],
                        'mae': best_model_result['metrics']['mae']
                    })
                    
                    logger.info(f"Completed {task_name} - Best model: {best_model_name}")
                
                else:
                    logger.warning(f"Target column {target_col} not found for task {task_name}")
            
            # Save overall results
            results_file = self.models_path / 'training_results.json'
            with open(results_file, 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                json_results = {}
                for key, value in pipeline_results.items():
                    if key == 'trained_models':
                        json_results[key] = {}
                        for task, task_data in value.items():
                            json_results[key][task] = {}
                            for sub_key, sub_value in task_data.items():
                                if sub_key == 'metrics':
                                    json_results[key][task][sub_key] = {k: float(v) for k, v in sub_value.items()}
                                else:
                                    json_results[key][task][sub_key] = sub_value
                    elif key == 'performance_summary':
                        json_results[key] = []
                        for item in value:
                            json_item = {}
                            for k, v in item.items():
                                json_item[k] = float(v) if isinstance(v, np.floating) else v
                            json_results[key].append(json_item)
                    else:
                        json_results[key] = value
                
                json.dump(json_results, f, indent=2)
            
            logger.info("Model training pipeline completed successfully")
            
            return {
                'status': 'success',
                'trained_models': len(pipeline_results['trained_models']),
                'results': pipeline_results,
                'models_path': str(self.models_path)
            }
            
        except Exception as e:
            logger.error(f"Model training pipeline failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

def main():
    """Main function to run the model training pipeline"""
    try:
        # Initialize model training
        trainer = ProcurementModelTraining()
        
        # Run training pipeline
        results = trainer.run_model_training_pipeline()
        
        if results['status'] == 'success':
            print(f" Model training completed successfully!")
            print(f" Trained models: {results['trained_models']}")
            print(f" Models saved to: {results['models_path']}")
            
            # Print performance summary
            if 'results' in results and 'performance_summary' in results['results']:
                print(f"\n Model Performance Summary:")
                for model_perf in results['results']['performance_summary']:
                    print(f"   {model_perf['task']}: {model_perf['model']} - R²: {model_perf['r2']:.4f}")
        else:
            print(f" Model training failed: {results['error']}")
            
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        print(f" Fatal error: {e}")


if __name__ == "__main__":
    main()