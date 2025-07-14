🎯 Overview
This project implements three core AI-powered supply chain optimization features using machine learning models trained on real-world supply chain data. The system provides intelligent solutions for production scheduling, automated procurement, and demand-driven supply chain management.
Business Impact Goals

Production Efficiency: 15-25% improvement in production scheduling
Cost Reduction: 8-15% reduction in procurement costs
Forecast Accuracy: 20-35% improvement in demand prediction
Inventory Optimization: 15-25% reduction in carrying costs

✨ Features
🏭 1. Production Scheduling Optimization

AI-driven scheduling that analyzes machine availability, workforce capacity, and raw material supply
Bottleneck identification and resolution recommendations
Resource utilization optimization with constraint satisfaction
Real-time schedule adjustments based on changing conditions

🛒 2. Automated Procurement

Intelligent supplier selection based on performance metrics and cost analysis
Automated order placement when inventory falls below optimal thresholds
Price negotiation insights using market trend analysis
Risk assessment for supplier reliability and delivery performance

📈 3. AI-Powered Demand-Driven Supply Chain

Multi-horizon demand forecasting with seasonal pattern recognition
Real-time demand sensing from customer behavior and market signals
Dynamic inventory management aligned with predicted demand
Supply chain responsiveness to market changes and customer preferences



## Project Structure
- `data/`: Dataset storage
- `src/`: Source code modules
- `notebooks/`: Jupyter notebooks for exploration

## MLOps Experiment Tracking with MLflow

This project uses [MLflow](https://mlflow.org/) for experiment tracking and MLOps monitoring. All model training runs, parameters, metrics, and artifacts are logged to the `mlruns` directory.

### How to Use MLflow UI

1. Install MLflow (already in requirements.txt):
   ```bash
   pip install mlflow
   ```
2. Run your pipeline as usual. Training runs will be logged automatically.
3. To launch the MLflow UI and monitor experiments:
   ```bash
   mlflow ui
   ```
   This will start a web server at http://localhost:5000 where you can browse all experiment runs, compare metrics, and download artifacts.

### Notes
- All experiment data is stored in the `automated_procurement/mlruns` directory by default.
- You can change the tracking URI or experiment name in the code if needed.