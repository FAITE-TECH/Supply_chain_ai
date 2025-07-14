import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any
import pandas as pd

os.makedirs('logs', exist_ok=True)

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from steps.data_ingestion import ProcurementDataIngestion
from steps.feature_engineering  import ProcurementFeatureEngineering

# Configure logging
logs_dir = Path('logs')
logs_dir.mkdir(exist_ok=True)
log_file = logs_dir / 'data_pipeline.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProcurementDataPipeline:
    """Main orchestrator for the procurement data pipeline"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the data pipeline
        
        Args:
            config: Configuration dictionary for the pipeline
        """
        self.config = config or self._load_default_config()
        self.pipeline_id = f"procurement_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results = {}
        
        # Ensure required directories exist
        self._setup_directories()
        
        logger.info(f"Initialized procurement data pipeline: {self.pipeline_id}")

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default pipeline configuration"""
        return {
            'raw_data_path': 'data/raw',
            'processed_data_path': 'data/processed',
            'enable_data_validation': True,
            'enable_feature_selection': True,
            'save_intermediate_results': True,
            'pipeline_settings': {
                'safety_stock_service_level': 0.95,
                'holding_cost_rate': 0.20,
                'setup_cost': 50,
                'abc_thresholds': {'A': 0.2, 'B': 0.5},
                'reorder_buffer_days': 3
            }
        }
    
    def _setup_directories(self):
        """Setup required directories for the pipeline"""
        directories = [
            'data/raw',
            'data/processed',
            'logs',
            'artifacts',
            'models'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def _log_pipeline_step(self, step_name: str, status: str, details: Dict = None):
        """Log pipeline step execution"""
        log_entry = {
            'pipeline_id': self.pipeline_id,
            'step': step_name,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        
        self.results[step_name] = log_entry
        
        if status == 'success':
            logger.info(f"✅ {step_name} completed successfully")
        elif status == 'failed':
            logger.error(f"❌ {step_name} failed")
        elif status == 'started':
            logger.info(f"🚀 {step_name} started")

    def run_data_ingestion(self) -> bool:
        """
        Run data ingestion step
        
        Returns:
            bool: True if successful, False otherwise
        """
        step_name = "data_ingestion"
        self._log_pipeline_step(step_name, "started")
        
        try:
            ingestion = ProcurementDataIngestion(
                raw_data_path=self.config['raw_data_path'],
                processed_data_path=self.config['processed_data_path']
            )
            
            results = ingestion.run_ingestion_pipeline()
            
            if results['status'] == 'success':
                self._log_pipeline_step(step_name, "success", results)
                return True
            else:
                self._log_pipeline_step(step_name, "failed", results)
                return False
                
        except Exception as e:
            error_details = {'error': str(e), 'type': type(e).__name__}
            self._log_pipeline_step(step_name, "failed", error_details)
            logger.exception(f"Error in {step_name}")
            return False
        
    def run_feature_engineering(self) -> bool:
        """
        Run feature engineering step
        
        Returns:
            bool: True if successful, False otherwise
        """
        step_name = "feature_engineering"
        self._log_pipeline_step(step_name, "started")
        
        try:
            feature_eng = ProcurementFeatureEngineering(
                processed_data_path=self.config['processed_data_path']
            )
            
            results = feature_eng.run_feature_engineering_pipeline()
            
            if results['status'] == 'success':
                self._log_pipeline_step(step_name, "success", results)
                return True
            else:
                self._log_pipeline_step(step_name, "failed", results)
                return False
                
        except Exception as e:
            error_details = {'error': str(e), 'type': type(e).__name__}
            self._log_pipeline_step(step_name, "failed", error_details)
            logger.exception(f"Error in {step_name}")
            return False
        
    def validate_pipeline_outputs(self) -> bool:
        """
        Validate that all expected pipeline outputs exist and are valid
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        step_name = "pipeline_validation"
        self._log_pipeline_step(step_name, "started")
        
        try:
            processed_path = Path(self.config['processed_data_path'])
            
            # Check required files exist
            required_files = [
                'orders_ingested.csv',
                'inventory_ingested.csv',
                'fulfillment_ingested.csv',
                'procurement_features.csv',
                'ml_features.csv',
                'data_lineage.json',
                'feature_metadata.json'
            ]
            
            missing_files = []
            for file in required_files:
                if not (processed_path / file).exists():
                    missing_files.append(file)
            
            if missing_files:
                error_details = {'missing_files': missing_files}
                self._log_pipeline_step(step_name, "failed", error_details)
                return False
            
  
          
            
            # Check procurement features
            procurement_df = pd.read_csv(processed_path / 'procurement_features.csv')
            ml_features_df = pd.read_csv(processed_path / 'ml_features.csv')
            
            validation_results = {
                'procurement_features_shape': procurement_df.shape,
                'ml_features_shape': ml_features_df.shape,
                'procurement_missing_values': procurement_df.isnull().sum().sum(),
                'ml_missing_values': ml_features_df.isnull().sum().sum(),
                'unique_products': procurement_df['Product Name'].nunique()
            }
            
            # Check for critical issues
            if procurement_df.empty or ml_features_df.empty:
                error_details = {'error': 'Empty datasets detected', 'validation_results': validation_results}
                self._log_pipeline_step(step_name, "failed", error_details)
                return False
            
            if procurement_df['Product Name'].nunique() == 0:
                error_details = {'error': 'No products found', 'validation_results': validation_results}
                self._log_pipeline_step(step_name, "failed", error_details)
                return False
            
            self._log_pipeline_step(step_name, "success", validation_results)
            return True
            
        except Exception as e:
            error_details = {'error': str(e), 'type': type(e).__name__}
            self._log_pipeline_step(step_name, "failed", error_details)
            logger.exception(f"Error in {step_name}")
            return False
        
    def generate_pipeline_report(self) -> Dict:
        """
        Generate a comprehensive pipeline execution report
        
        Returns:
            Dict: Pipeline execution report
        """
        logger.info("Generating pipeline execution report...")
        
        # Calculate overall pipeline status
        failed_steps = [step for step, result in self.results.items() if result['status'] == 'failed']
        overall_status = 'failed' if failed_steps else 'success'
        
        # Load feature metadata if available
        feature_metadata = {}
        try:
            metadata_path = Path(self.config['processed_data_path']) / 'feature_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    feature_metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load feature metadata: {e}")
        
        # Create comprehensive report
        report = {
            'pipeline_id': self.pipeline_id,
            'execution_timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'config': self.config,
            'step_results': self.results,
            'failed_steps': failed_steps,
            'feature_metadata': feature_metadata,
            'summary': {
                'total_steps': len(self.results),
                'successful_steps': len([r for r in self.results.values() if r['status'] == 'success']),
                'failed_steps': len(failed_steps)
            }
        }
        
        # Add data summary if available
        try:
            processed_path = Path(self.config['processed_data_path'])
            if (processed_path / 'procurement_features.csv').exists():
                import pandas as pd
                procurement_df = pd.read_csv(processed_path / 'procurement_features.csv')
                
                report['data_summary'] = {
                    'total_products': len(procurement_df),
                    'products_requiring_reorder': len(procurement_df[procurement_df['Stock_Status'] == 'Reorder_Required']),
                    'products_in_stockout': len(procurement_df[procurement_df['Stock_Status'] == 'Stockout']),
                    'abc_distribution': procurement_df['ABC_Category'].value_counts().to_dict(),
                    'average_procurement_priority': procurement_df['Procurement_Priority_Score'].mean()
                }
        except Exception as e:
            logger.warning(f"Could not generate data summary: {e}")
        
        return report
    
    def save_pipeline_report(self, report: Dict) -> str:
        """
        Save pipeline report to file
        
        Args:
            report: Pipeline execution report
            
        Returns:
            str: Path to saved report
        """
        report_path = Path('artifacts') / f'pipeline_report_{self.pipeline_id}.json'
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Pipeline report saved to: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error saving pipeline report: {e}")
            raise

    def run_complete_pipeline(self) -> Dict:
        """
        Run the complete data pipeline from start to finish
        
        Returns:
            Dict: Complete pipeline execution report
        """
        logger.info(f"🚀 Starting complete procurement data pipeline: {self.pipeline_id}")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Data Ingestion
            if not self.run_data_ingestion():
                logger.error("Data ingestion failed. Stopping pipeline.")
                report = self.generate_pipeline_report()
                self.save_pipeline_report(report)
                return report
            
            # Step 2: Feature Engineering
            if not self.run_feature_engineering():
                logger.error("Feature engineering failed. Stopping pipeline.")
                report = self.generate_pipeline_report()
                self.save_pipeline_report(report)
                return report
            
            # Step 3: Pipeline Validation
            if not self.validate_pipeline_outputs():
                logger.error("Pipeline validation failed.")
                report = self.generate_pipeline_report()
                self.save_pipeline_report(report)
                return report
            
            # Pipeline completed successfully
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            logger.info(f"✅ Complete pipeline executed successfully in {execution_time:.2f} seconds")
            
            # Generate and save final report
            report = self.generate_pipeline_report()
            report['execution_time_seconds'] = execution_time
            report_path = self.save_pipeline_report(report)
            
            # Print summary
            self._print_pipeline_summary(report)
            
            return report
        
        except Exception as e:
            logger.exception("Fatal error in pipeline execution")
            report = self.generate_pipeline_report()
            report['fatal_error'] = str(e)
            self.save_pipeline_report(report)
            return report
        
    def _print_pipeline_summary(self, report: Dict):
        """Print a user-friendly pipeline summary"""
        print("\n" + "="*60)
        print("🎯 AUTOMATED PROCUREMENT PIPELINE SUMMARY")
        print("="*60)
        
        print(f"📊 Pipeline ID: {report['pipeline_id']}")
        print(f"✅ Status: {report['overall_status'].upper()}")
        print(f"⏱️  Execution Time: {report.get('execution_time_seconds', 0):.2f} seconds")
        
        if 'data_summary' in report:
            data_summary = report['data_summary']
            print(f"\n📈 DATA SUMMARY:")
            print(f"   • Total Products: {data_summary['total_products']:,}")
            print(f"   • Products Requiring Reorder: {data_summary['products_requiring_reorder']:,}")
            print(f"   • Products in Stockout: {data_summary['products_in_stockout']:,}")
            print(f"   • ABC Distribution: {data_summary['abc_distribution']}")
        
        if 'feature_metadata' in report and report['feature_metadata']:
            feat_meta = report['feature_metadata']
            print(f"\n🔧 FEATURE ENGINEERING:")
            print(f"   • Total Features Created: {feat_meta.get('total_features', 'N/A')}")
            print(f"   • Products Processed: {feat_meta.get('total_products', 'N/A')}")
        
        print(f"\n📁 Output Location: {self.config['processed_data_path']}")
        print("="*60)

def main():
    """Main function to run the procurement data pipeline"""
    
    # Parse command line arguments (simple implementation)
    import argparse
    parser = argparse.ArgumentParser(description='Run Automated Procurement Data Pipeline')
    parser.add_argument('--raw-data-path', default='data/raw', help='Path to raw data files')
    parser.add_argument('--processed-data-path', default='data/processed', help='Path for processed data output')
    parser.add_argument('--config-file', help='Path to configuration JSON file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config_file and Path(args.config_file).exists():
        try:
            with open(args.config_file, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from: {args.config_file}")
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")
    
    # Override with command line arguments
    if not config:
        config = {}
    
    config['raw_data_path'] = args.raw_data_path
    config['processed_data_path'] = args.processed_data_path
    
    try:
        # Initialize and run pipeline
        pipeline = ProcurementDataPipeline(config)
        report = pipeline.run_complete_pipeline()
        
        # Exit with appropriate code
        if report['overall_status'] == 'success':
            print("\n🎉 Pipeline completed successfully!")
            sys.exit(0)
        else:
            print("\n💥 Pipeline failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        print("\n⏹️  Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception("Fatal error in main")
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
                