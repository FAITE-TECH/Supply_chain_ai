#!/usr/bin/env python3
"""
Complete Integrated Pipeline for Automated Procurement Model
This script orchestrates the complete end-to-end pipeline from raw data to trained models
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import all pipeline components
from pipelines.data_pipeline import ProcurementDataPipeline
from pipelines.training_pipeline import ProcurementMLPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/complete_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CompleteProcurementPipeline:
    """Complete end-to-end pipeline orchestrator for procurement automation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the complete pipeline
        
        Args:
            config: Configuration dictionary for the pipeline
        """
        self.config = config or self._load_default_config()
        self.pipeline_id = f"complete_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results = {}
        
        # Ensure required directories exist
        self._setup_directories()
        
        logger.info(f"Initialized complete procurement pipeline: {self.pipeline_id}")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default pipeline configuration"""
        return {
            # Data pipeline settings
            'raw_data_path': 'data/raw',
            'processed_data_path': 'data/processed',
            'artifacts_path': 'artifacts',
            'models_path': 'models',
            'logs_path': 'logs',
            
            # Pipeline control flags
            'run_data_pipeline': True,
            'run_ml_pipeline': True,
            'enable_validation': True,
            'save_intermediate_results': True,
            
            # Data processing settings
            'data_pipeline_config': {
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
            },
            
            # ML pipeline settings
            'ml_pipeline_config': {
                'enable_hyperparameter_optimization': True,
                'enable_cross_validation': True,
                'save_intermediate_results': True,
                'ml_settings': {
                    'test_size': 0.2,
                    'random_state': 42,
                    'n_features_select': 15,
                    'cv_folds': 5,
                    'optimization_models': ['random_forest', 'xgboost', 'gradient_boosting']
                },
                'evaluation_settings': {
                    'generate_feature_importance': True,
                    'assess_business_impact': True,
                    'create_visualizations': True
                }
            },
            
            # Business settings
            'business_settings': {
                'target_service_level': 0.95,
                'acceptable_stockout_rate': 0.05,
                'minimum_model_r2': 0.4,
                'deployment_r2_threshold': 0.6
            }
        }
    
    def _setup_directories(self):
        """Setup required directories for the complete pipeline"""
        directories = [
            'data/raw',
            'data/processed',
            'models',
            'artifacts',
            'logs'
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
    
    def validate_input_data(self) -> bool:
        """
        Validate that required input data files exist
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        step_name = "input_data_validation"
        self._log_pipeline_step(step_name, "started")
        
        try:
            raw_data_path = Path(self.config['raw_data_path'])
            
            # Required input files
            required_files = [
                'orders_and_shipments.csv',
                'inventory.csv',
                'fulfillment.csv'
            ]
            
            missing_files = []
            file_info = {}
            
            for file in required_files:
                file_path = raw_data_path / file
                if not file_path.exists():
                    missing_files.append(str(file_path))
                else:
                    # Get file info
                    try:
                        import pandas as pd
                        df = pd.read_csv(file_path)
                        file_info[file] = {
                            'rows': len(df),
                            'columns': len(df.columns),
                            'size_mb': file_path.stat().st_size / (1024 * 1024)
                        }
                    except Exception as e:
                        file_info[file] = {'error': str(e)}
            
            if missing_files:
                error_details = {'missing_files': missing_files}
                self._log_pipeline_step(step_name, "failed", error_details)
                return False
            
            validation_results = {
                'files_validated': len(required_files),
                'file_info': file_info,
                'total_size_mb': sum(info.get('size_mb', 0) for info in file_info.values()),
                'total_rows': sum(info.get('rows', 0) for info in file_info.values())
            }
            
            self._log_pipeline_step(step_name, "success", validation_results)
            return True
            
        except Exception as e:
            error_details = {'error': str(e), 'type': type(e).__name__}
            self._log_pipeline_step(step_name, "failed", error_details)
            logger.exception(f"Error in {step_name}")
            return False
    
    def run_data_pipeline(self) -> bool:
        """
        Run the data processing pipeline
        
        Returns:
            bool: True if successful, False otherwise
        """
        step_name = "data_pipeline"
        self._log_pipeline_step(step_name, "started")
        
        try:
            # Configure data pipeline
            data_config = self.config['data_pipeline_config'].copy()
            data_config.update({
                'raw_data_path': self.config['raw_data_path'],
                'processed_data_path': self.config['processed_data_path']
            })
            
            # Initialize and run data pipeline
            data_pipeline = ProcurementDataPipeline(data_config)
            results = data_pipeline.run_complete_pipeline()
            
            if results['overall_status'] == 'success':
                self._log_pipeline_step(step_name, "success", {
                    'datasets_processed': results.get('datasets_processed', 0),
                    'total_records': results.get('total_records', 0),
                    'data_quality': results.get('data_summary', {})
                })
                return True
            else:
                self._log_pipeline_step(step_name, "failed", results)
                return False
                
        except Exception as e:
            error_details = {'error': str(e), 'type': type(e).__name__}
            self._log_pipeline_step(step_name, "failed", error_details)
            logger.exception(f"Error in {step_name}")
            return False
    
    def run_ml_pipeline(self) -> bool:
        """
        Run the machine learning pipeline
        
        Returns:
            bool: True if successful, False otherwise
        """
        step_name = "ml_pipeline"
        self._log_pipeline_step(step_name, "started")
        
        try:
            # Configure ML pipeline
            ml_config = self.config['ml_pipeline_config'].copy()
            ml_config.update({
                'processed_data_path': self.config['processed_data_path'],
                'models_path': self.config['models_path'],
                'artifacts_path': self.config['artifacts_path']
            })
            
            # Initialize and run ML pipeline
            ml_pipeline = ProcurementMLPipeline(ml_config)
            results = ml_pipeline.run_complete_ml_pipeline()
            
            if results['overall_status'] == 'success':
                self._log_pipeline_step(step_name, "success", {
                    'trained_models': results.get('model_summary', {}).get('total_models_trained', 0),
                    'average_r2': results.get('model_summary', {}).get('average_r2_score', 0),
                    'deployment_ready': results.get('model_summary', {}).get('models_ready_for_deployment', 0)
                })
                return True
            else:
                self._log_pipeline_step(step_name, "failed", results)
                return False
                
        except Exception as e:
            error_details = {'error': str(e), 'type': type(e).__name__}
            self._log_pipeline_step(step_name, "failed", error_details)
            logger.exception(f"Error in {step_name}")
            return False
    
    def validate_final_outputs(self) -> bool:
        """
        Validate that the complete pipeline produced expected outputs
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        step_name = "final_output_validation"
        self._log_pipeline_step(step_name, "started")
        
        try:
            processed_path = Path(self.config['processed_data_path'])
            models_path = Path(self.config['models_path'])
            artifacts_path = Path(self.config['artifacts_path'])
            
            # Check processed data files
            required_processed_files = [
                'procurement_features.csv',
                'orders_ingested.csv',
                'inventory_ingested.csv',
                'fulfillment_ingested.csv'
            ]
            
            # Check model files
            required_model_files = [
                'training_results.json'
            ]
            
            missing_files = []
            
            # Validate processed data
            for file in required_processed_files:
                if not (processed_path / file).exists():
                    missing_files.append(f"processed/{file}")
            
            # Validate models
            for file in required_model_files:
                if not (models_path / file).exists():
                    missing_files.append(f"models/{file}")
            
            # Check for trained model directories
            model_dirs = [d for d in models_path.iterdir() if d.is_dir()]
            
            # Check for evaluation reports
            evaluation_reports = list(artifacts_path.glob("model_evaluation_report_*.json"))
            
            if missing_files or len(model_dirs) == 0 or len(evaluation_reports) == 0:
                error_details = {
                    'missing_files': missing_files,
                    'model_directories': len(model_dirs),
                    'evaluation_reports': len(evaluation_reports)
                }
                self._log_pipeline_step(step_name, "failed", error_details)
                return False
            
            # Validate model quality
            try:
                with open(models_path / 'training_results.json', 'r') as f:
                    training_results = json.load(f)
                
                trained_models = training_results.get('trained_models', {})
                
                if len(trained_models) == 0:
                    error_details = {'error': 'No models were successfully trained'}
                    self._log_pipeline_step(step_name, "failed", error_details)
                    return False
                
                # Check if any models meet minimum quality threshold
                min_r2_threshold = self.config['business_settings']['minimum_model_r2']
                quality_models = 0
                
                for task_name, task_info in trained_models.items():
                    r2_score = task_info.get('metrics', {}).get('r2', 0)
                    if r2_score >= min_r2_threshold:
                        quality_models += 1
                
                if quality_models == 0:
                    error_details = {
                        'error': f'No models meet minimum R² threshold of {min_r2_threshold}',
                        'trained_models': len(trained_models),
                        'quality_models': quality_models
                    }
                    self._log_pipeline_step(step_name, "failed", error_details)
                    return False
                
                validation_results = {
                    'processed_files_count': len(required_processed_files),
                    'trained_models_count': len(trained_models),
                    'quality_models_count': quality_models,
                    'model_directories_count': len(model_dirs),
                    'evaluation_reports_count': len(evaluation_reports),
                    'min_r2_threshold': min_r2_threshold
                }
                
                self._log_pipeline_step(step_name, "success", validation_results)
                return True
                
            except Exception as e:
                error_details = {'error': f'Error validating model quality: {str(e)}'}
                self._log_pipeline_step(step_name, "failed", error_details)
                return False
            
        except Exception as e:
            error_details = {'error': str(e), 'type': type(e).__name__}
            self._log_pipeline_step(step_name, "failed", error_details)
            logger.exception(f"Error in {step_name}")
            return False
    
    def generate_business_summary(self) -> Dict[str, Any]:
        """
        Generate business-focused summary of pipeline results
        
        Returns:
            Dictionary with business summary
        """
        step_name = "business_summary_generation"
        self._log_pipeline_step(step_name, "started")
        
        try:
            # Load final results
            models_path = Path(self.config['models_path'])
            artifacts_path = Path(self.config['artifacts_path'])
            
            business_summary = {
                'pipeline_id': self.pipeline_id,
                'completion_timestamp': datetime.now().isoformat(),
                'business_readiness': {},
                'deployment_recommendations': {},
                'expected_roi': {},
                'next_steps': []
            }
            
            # Load training results
            try:
                with open(models_path / 'training_results.json', 'r') as f:
                    training_results = json.load(f)
                
                trained_models = training_results.get('trained_models', {})
                
                # Calculate business readiness metrics
                deployment_threshold = self.config['business_settings']['deployment_r2_threshold']
                
                deployment_ready = 0
                pilot_ready = 0
                needs_improvement = 0
                
                for task_name, task_info in trained_models.items():
                    r2_score = task_info.get('metrics', {}).get('r2', 0)
                    
                    if r2_score >= deployment_threshold:
                        deployment_ready += 1
                    elif r2_score >= 0.4:
                        pilot_ready += 1
                    else:
                        needs_improvement += 1
                
                business_summary['business_readiness'] = {
                    'total_models': len(trained_models),
                    'deployment_ready': deployment_ready,
                    'pilot_ready': pilot_ready,
                    'needs_improvement': needs_improvement,
                    'overall_readiness_percentage': (deployment_ready / len(trained_models) * 100) if trained_models else 0
                }
                
                # Generate deployment recommendations
                if deployment_ready > 0:
                    recommendation = "Proceed with immediate deployment"
                    timeline = "1-2 weeks"
                    confidence = "High"
                elif pilot_ready > 0:
                    recommendation = "Start with pilot implementation"
                    timeline = "1-2 months"
                    confidence = "Medium"
                else:
                    recommendation = "Continue model development"
                    timeline = "3-6 months"
                    confidence = "Low"
                
                business_summary['deployment_recommendations'] = {
                    'primary_recommendation': recommendation,
                    'timeline': timeline,
                    'confidence_level': confidence,
                    'risk_assessment': 'Low' if deployment_ready > 0 else 'Medium' if pilot_ready > 0 else 'High'
                }
                
                # Calculate expected ROI based on model quality
                avg_r2 = sum(task_info.get('metrics', {}).get('r2', 0) for task_info in trained_models.values()) / len(trained_models) if trained_models else 0
                
                if avg_r2 > 0.7:
                    roi_estimate = {
                        'inventory_cost_reduction': '15-25%',
                        'stockout_reduction': '80-90%',
                        'process_efficiency': '70-80%',
                        'payback_period': '6-12 months'
                    }
                elif avg_r2 > 0.5:
                    roi_estimate = {
                        'inventory_cost_reduction': '8-15%',
                        'stockout_reduction': '60-80%',
                        'process_efficiency': '50-70%',
                        'payback_period': '12-18 months'
                    }
                else:
                    roi_estimate = {
                        'inventory_cost_reduction': '3-8%',
                        'stockout_reduction': '40-60%',
                        'process_efficiency': '30-50%',
                        'payback_period': '18-24 months'
                    }
                
                business_summary['expected_roi'] = roi_estimate
                
            except Exception as e:
                logger.warning(f"Could not load training results for business summary: {e}")
            
            # Define next steps based on results
            if business_summary['business_readiness'].get('deployment_ready', 0) > 0:
                business_summary['next_steps'] = [
                    "Present results to executive team for deployment approval",
                    "Prepare production infrastructure and integration plan",
                    "Develop user training and change management plan",
                    "Set up model monitoring and performance tracking",
                    "Plan phased rollout starting with highest-impact models"
                ]
            elif business_summary['business_readiness'].get('pilot_ready', 0) > 0:
                business_summary['next_steps'] = [
                    "Design pilot implementation scope and success criteria",
                    "Select pilot users and procurement categories",
                    "Set up enhanced monitoring and feedback collection",
                    "Plan model improvement roadmap based on pilot results",
                    "Prepare business case for full deployment"
                ]
            else:
                business_summary['next_steps'] = [
                    "Analyze model performance gaps and improvement opportunities",
                    "Investigate additional data sources and features",
                    "Consider alternative algorithms and approaches",
                    "Plan additional data collection and feature engineering",
                    "Set timeline for model improvement and re-evaluation"
                ]
            
            self._log_pipeline_step(step_name, "success", business_summary)
            return business_summary
            
        except Exception as e:
            error_details = {'error': str(e), 'type': type(e).__name__}
            self._log_pipeline_step(step_name, "failed", error_details)
            logger.exception(f"Error in {step_name}")
            return {}
    
    def generate_final_report(self) -> Dict:
        """
        Generate comprehensive final pipeline report
        
        Returns:
            Dict: Complete pipeline execution report
        """
        logger.info("Generating final pipeline report...")
        
        # Calculate overall pipeline status
        failed_steps = [step for step, result in self.results.items() if result['status'] == 'failed']
        overall_status = 'failed' if failed_steps else 'success'
        
        # Generate business summary
        business_summary = self.generate_business_summary()
        
        # Create comprehensive report
        report = {
            'pipeline_id': self.pipeline_id,
            'execution_timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'config': self.config,
            'step_results': self.results,
            'failed_steps': failed_steps,
            'business_summary': business_summary,
            'summary': {
                'total_steps': len(self.results),
                'successful_steps': len([r for r in self.results.values() if r['status'] == 'success']),
                'failed_steps': len(failed_steps)
            }
        }
        
        return report
    
    def save_final_report(self, report: Dict) -> str:
        """
        Save final pipeline report to file
        
        Args:
            report: Complete pipeline execution report
            
        Returns:
            str: Path to saved report
        """
        report_path = Path('artifacts') / f'complete_pipeline_report_{self.pipeline_id}.json'
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Final pipeline report saved to: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error saving final pipeline report: {e}")
            raise
    
    def run_complete_pipeline(self) -> Dict:
        """
        Run the complete end-to-end pipeline
        
        Returns:
            Dict: Complete pipeline execution report
        """
        logger.info(f"🚀 Starting complete procurement automation pipeline: {self.pipeline_id}")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Validate Input Data
            if not self.validate_input_data():
                logger.error("Input data validation failed. Stopping pipeline.")
                report = self.generate_final_report()
                self.save_final_report(report)
                return report
            
            # Step 2: Run Data Pipeline (if enabled)
            if self.config.get('run_data_pipeline', True):
                if not self.run_data_pipeline():
                    logger.error("Data pipeline failed. Stopping pipeline.")
                    report = self.generate_final_report()
                    self.save_final_report(report)
                    return report
            else:
                logger.info("Data pipeline skipped (disabled in config)")
            
            # Step 3: Run ML Pipeline (if enabled)
            if self.config.get('run_ml_pipeline', True):
                if not self.run_ml_pipeline():
                    logger.error("ML pipeline failed. Stopping pipeline.")
                    report = self.generate_final_report()
                    self.save_final_report(report)
                    return report
            else:
                logger.info("ML pipeline skipped (disabled in config)")
            
            # Step 4: Final Validation (if enabled)
            if self.config.get('enable_validation', True):
                if not self.validate_final_outputs():
                    logger.error("Final output validation failed.")
                    report = self.generate_final_report()
                    self.save_final_report(report)
                    return report
            else:
                logger.info("Final validation skipped (disabled in config)")
            
            # Pipeline completed successfully
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            logger.info(f"✅ Complete pipeline executed successfully in {execution_time:.2f} seconds")
            
            # Generate and save final report
            report = self.generate_final_report()
            report['execution_time_seconds'] = execution_time
            report_path = self.save_final_report(report)
            
            # Print summary
            self._print_pipeline_summary(report)
            
            return report
            
        except Exception as e:
            logger.exception("Fatal error in complete pipeline execution")
            report = self.generate_final_report()
            report['fatal_error'] = str(e)
            self.save_final_report(report)
            return report
    
    def _print_pipeline_summary(self, report: Dict):
        """Print a user-friendly complete pipeline summary"""
        print("\n" + "="*80)
        print("🏭 COMPLETE AUTOMATED PROCUREMENT PIPELINE SUMMARY")
        print("="*80)
        
        print(f"📊 Pipeline ID: {report['pipeline_id']}")
        print(f"✅ Status: {report['overall_status'].upper()}")
        print(f"⏱️  Total Execution Time: {report.get('execution_time_seconds', 0):.2f} seconds")
        
        # Business summary
        if 'business_summary' in report and report['business_summary']:
            business = report['business_summary']
            readiness = business.get('business_readiness', {})
            
            print(f"\n🎯 BUSINESS READINESS ASSESSMENT:")
            print(f"   • Total Models Developed: {readiness.get('total_models', 0)}")
            print(f"   • Ready for Deployment: {readiness.get('deployment_ready', 0)}")
            print(f"   • Ready for Pilot: {readiness.get('pilot_ready', 0)}")
            print(f"   • Need Improvement: {readiness.get('needs_improvement', 0)}")
            print(f"   • Overall Readiness: {readiness.get('overall_readiness_percentage', 0):.1f}%")
            
            # Deployment recommendation
            deployment = business.get('deployment_recommendations', {})
            print(f"\n📋 DEPLOYMENT RECOMMENDATION:")
            print(f"   • Action: {deployment.get('primary_recommendation', 'N/A')}")
            print(f"   • Timeline: {deployment.get('timeline', 'N/A')}")
            print(f"   • Confidence: {deployment.get('confidence_level', 'N/A')}")
            print(f"   • Risk Level: {deployment.get('risk_assessment', 'N/A')}")
            
            # Expected ROI
            roi = business.get('expected_roi', {})
            if roi:
                print(f"\n💰 EXPECTED ROI:")
                print(f"   • Inventory Cost Reduction: {roi.get('inventory_cost_reduction', 'N/A')}")
                print(f"   • Stockout Reduction: {roi.get('stockout_reduction', 'N/A')}")
                print(f"   • Process Efficiency: {roi.get('process_efficiency', 'N/A')}")
                print(f"   • Payback Period: {roi.get('payback_period', 'N/A')}")
            
            # Next steps
            next_steps = business.get('next_steps', [])
            if next_steps:
                print(f"\n🚀 RECOMMENDED NEXT STEPS:")
                for i, step in enumerate(next_steps[:5], 1):
                    print(f"   {i}. {step}")
        
        print(f"\n📁 OUTPUT LOCATIONS:")
        print(f"   • Processed Data: {self.config['processed_data_path']}")
        print(f"   • Trained Models: {self.config['models_path']}")
        print(f"   • Reports & Artifacts: {self.config['artifacts_path']}")
        
        print("\n🎉 PIPELINE COMPLETION STATUS:")
        if report['overall_status'] == 'success':
            print("   ✅ All pipeline steps completed successfully")
            print("   🤖 Models trained and evaluated")
            print("   📊 Business impact assessed")
            print("   🚀 Ready for deployment planning")
        else:
            print("   ❌ Pipeline completed with errors")
            print("   🔍 Check logs for detailed error information")
            print("   🛠️  Review failed steps and retry")
        
        print("="*80)


def main():
    """Main function to run the complete pipeline"""
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run Complete Automated Procurement Pipeline')
    parser.add_argument('--raw-data-path', default='data/raw', help='Path to raw data files')
    parser.add_argument('--processed-data-path', default='data/processed', help='Path for processed data')
    parser.add_argument('--models-path', default='models', help='Path for trained models')
    parser.add_argument('--artifacts-path', default='artifacts', help='Path for artifacts and reports')
    parser.add_argument('--config-file', help='Path to configuration JSON file')
    parser.add_argument('--skip-data-pipeline', action='store_true', help='Skip data processing pipeline')
    parser.add_argument('--skip-ml-pipeline', action='store_true', help='Skip ML pipeline')
    parser.add_argument('--disable-validation', action='store_true', help='Disable pipeline validation')
    
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
    
    config.update({
        'raw_data_path': args.raw_data_path,
        'processed_data_path': args.processed_data_path,
        'models_path': args.models_path,
        'artifacts_path': args.artifacts_path,
        'run_data_pipeline': not args.skip_data_pipeline,
        'run_ml_pipeline': not args.skip_ml_pipeline,
        'enable_validation': not args.disable_validation
    })
    
    try:
        # Initialize and run complete pipeline
        pipeline = CompleteProcurementPipeline(config)
        report = pipeline.run_complete_pipeline()
        
        # Exit with appropriate code
        if report['overall_status'] == 'success':
            print("\n🎉 Complete Pipeline executed successfully!")
            print("🚀 Automated procurement models are ready for deployment!")
            sys.exit(0)
        else:
            print("\n💥 Complete Pipeline failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Complete Pipeline interrupted by user")
        print("\n⏹️  Complete Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception("Fatal error in main")
        print(f"\n💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()