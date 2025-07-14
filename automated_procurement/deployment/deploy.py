#!/usr/bin/env python3
"""
Deployment Script for Automated Procurement System
This script handles deployment, configuration, and startup of the production system
"""

import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime
import shutil
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProcurementDeployment:
    """Handles deployment of the automated procurement system"""
    
    def __init__(self, environment='production'):
        self.environment = environment
        self.project_root = Path(__file__).parent.parent
        self.deployment_dir = self.project_root / 'deployment'
        self.models_dir = self.project_root / 'models'
        self.config_file = self.deployment_dir / 'config.json'
        
        logger.info(f"Initializing deployment for environment: {environment}")
    
    def validate_environment(self):
        """Validate deployment environment"""
        logger.info("Validating deployment environment...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            raise RuntimeError("Python 3.8 or higher is required")
        
        # Check required directories
        required_dirs = [self.models_dir, self.deployment_dir]
        for dir_path in required_dirs:
            if not dir_path.exists():
                raise RuntimeError(f"Required directory not found: {dir_path}")
        
        # Check required model files
        required_model_files = [
            'training_results.json',
            'demand_forecast/model.pkl',
            'stockout_risk/model.pkl',
            'delivery_performance/model.pkl',
            'procurement_priority/model.pkl'
        ]
        
        for file_path in required_model_files:
            full_path = self.models_dir / file_path
            if not full_path.exists():
                raise RuntimeError(f"Required model file not found: {full_path}")
        
        logger.info("Environment validation passed")
    
    def install_dependencies(self):
        """Install required Python packages"""
        logger.info("Installing Python dependencies...")
        
        requirements_file = self.project_root / 'requirements.txt'
        if requirements_file.exists():
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
            ], check=True)
        else:
            # Install essential packages
            essential_packages = [
                'flask', 'flask-cors', 'pandas', 'numpy', 'scikit-learn',
                'joblib', 'schedule', 'requests', 'sqlite3'
            ]
            
            for package in essential_packages:
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package
                ], check=True)
        
        logger.info("Dependencies installed successfully")
    
    def setup_database(self):
        """Setup production database"""
        logger.info("Setting up production database...")
        
        # Database will be created automatically by the procurement engine
        # This is just to ensure the directory exists
        db_dir = self.deployment_dir
        db_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Database setup completed")
    
    def configure_logging(self):
        """Configure production logging"""
        logger.info("Configuring production logging...")
        
        logs_dir = self.deployment_dir / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logging configuration
        logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                },
            },
            'handlers': {
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': str(logs_dir / 'procurement_system.log'),
                    'maxBytes': 10485760,  # 10MB
                    'backupCount': 5,
                    'formatter': 'standard',
                },
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'standard',
                },
            },
            'root': {
                'level': 'INFO',
                'handlers': ['file', 'console'],
            },
        }
        
        # Save logging configuration
        with open(logs_dir / 'logging_config.json', 'w') as f:
            json.dump(logging_config, f, indent=2)
        
        logger.info("Logging configuration completed")
    
    def validate_models(self):
        """Validate trained models"""
        logger.info("Validating trained models...")
        
        # Load and validate training results
        with open(self.models_dir / 'training_results.json', 'r') as f:
            training_results = json.load(f)
        
        required_models = ['demand_forecast', 'stockout_risk', 'delivery_performance', 'procurement_priority']
        
        for model_name in required_models:
            if model_name not in training_results['trained_models']:
                raise RuntimeError(f"Model {model_name} not found in training results")
            
            model_info = training_results['trained_models'][model_name]
            r2_score = model_info['metrics']['r2']
            
            # Validate model performance
            min_performance = {
                'demand_forecast': 0.8,
                'stockout_risk': 0.7,
                'delivery_performance': 0.6,
                'procurement_priority': 0.8
            }
            
            if r2_score < min_performance[model_name]:
                logger.warning(f"Model {model_name} has low R² score: {r2_score}")
            else:
                logger.info(f"Model {model_name} validated - R² score: {r2_score:.3f}")
        
        logger.info("Model validation completed")
    
    def create_startup_script(self):
        """Create startup script for the service"""
        logger.info("Creating startup script...")
        
        startup_script = f"""#!/bin/bash
# Automated Procurement System Startup Script
# Generated on: {datetime.now().isoformat()}

export PYTHONPATH="${self.project_root}:$PYTHONPATH"
export FLASK_APP="{self.deployment_dir}/api_service.py"
export FLASK_ENV="{self.environment}"

# Change to project directory
cd "{self.project_root}"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start the service
python -m deployment.api_service
"""
        
        startup_file = self.deployment_dir / 'start_service.sh'
        with open(startup_file, 'w') as f:
            f.write(startup_script)
        
        # Make script executable
        os.chmod(startup_file, 0o755)
        
        logger.info(f"Startup script created: {startup_file}")
    
    def create_systemd_service(self):
        """Create systemd service file for Linux deployment"""
        logger.info("Creating systemd service file...")
        
        service_content = f"""[Unit]
Description=Automated Procurement System
After=network.target

[Service]
Type=simple
User=procurement
Group=procurement
WorkingDirectory={self.project_root}
Environment=PYTHONPATH={self.project_root}
ExecStart={sys.executable} -m deployment.api_service
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        service_file = self.deployment_dir / 'procurement-system.service'
        with open(service_file, 'w') as f:
            f.write(service_content)
        
        logger.info(f"Systemd service file created: {service_file}")
        logger.info("To install: sudo cp procurement-system.service /etc/systemd/system/")
        logger.info("To enable: sudo systemctl enable procurement-system")
        logger.info("To start: sudo systemctl start procurement-system")
    
    def create_docker_files(self):
        """Create Docker configuration files"""
        logger.info("Creating Docker configuration files...")
        
        # Dockerfile
        dockerfile_content = f"""FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 procurement
RUN chown -R procurement:procurement /app
USER procurement

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:5000/health || exit 1

# Start application
CMD ["python", "-m", "deployment.api_service"]
"""
        
        with open(self.deployment_dir / 'Dockerfile', 'w') as f:
            f.write(dockerfile_content)
        
        # Docker Compose
        docker_compose_content = """version: '3.8'

services:
  procurement-system:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - PYTHONPATH=/app
    volumes:
      - ./deployment/logs:/app/deployment/logs
      - ./deployment/procurement.db:/app/deployment/procurement.db
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./deployment/nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - procurement-system
    restart: unless-stopped
"""
        
        with open(self.deployment_dir / 'docker-compose.yml', 'w') as f:
            f.write(docker_compose_content)
        
        logger.info("Docker files created successfully")
    
    def create_nginx_config(self):
        """Create Nginx configuration for reverse proxy"""
        logger.info("Creating Nginx configuration...")
        
        nginx_config = """events {
    worker_connections 1024;
}

http {
    upstream procurement_backend {
        server procurement-system:5000;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        location / {
            proxy_pass http://procurement_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        location /health {
            proxy_pass http://procurement_backend/health;
            access_log off;
        }
    }
}
"""
        
        with open(self.deployment_dir / 'nginx.conf', 'w') as f:
            f.write(nginx_config)
        
        logger.info("Nginx configuration created")
    
    def create_deployment_documentation(self):
        """Create deployment documentation"""
        logger.info("Creating deployment documentation...")
        
        documentation = f"""# Automated Procurement System - Deployment Guide

## System Overview
This automated procurement system uses AI models to:
- Predict demand and stockout risks
- Optimize supplier selection
- Automate price negotiations
- Generate procurement orders
- Monitor inventory levels

## Deployment Information
- Environment: {self.environment}
- Deployment Date: {datetime.now().isoformat()}
- Python Version: {sys.version}

## Quick Start

### Option 1: Direct Python Execution
```bash
# Install dependencies
pip install -r requirements.txt

# Start the service
./deployment/start_service.sh
```

### Option 2: Docker Deployment
```bash
# Build and start with Docker Compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f procurement-system
```

### Option 3: Systemd Service (Linux)
```bash
# Install service
sudo cp deployment/procurement-system.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable procurement-system
sudo systemctl start procurement-system

# Check status
sudo systemctl status procurement-system
```

## API Endpoints

### Health Check
- `GET /health` - System health status

### Model Predictions
- `POST /api/predict/demand` - Demand forecasting
- `POST /api/predict/stockout` - Stockout risk prediction
- `POST /api/predict/supplier-performance` - Supplier performance
- `POST /api/predict/procurement-priority` - Procurement priorities
- `POST /api/analyze/comprehensive` - Comprehensive analysis

### Procurement Management
- `POST /api/procurement/cycle/run` - Trigger procurement cycle
- `GET /api/procurement/orders` - List procurement orders
- `GET /api/procurement/orders/<id>` - Get order details
- `POST /api/procurement/orders/<id>/approve` - Approve order
- `POST /api/procurement/orders/<id>/cancel` - Cancel order

### Supplier Management
- `GET /api/suppliers` - List suppliers
- `GET /api/suppliers/<id>/performance` - Supplier performance
- `POST /api/suppliers/recommend` - Recommend suppliers

### Inventory Management
- `GET /api/inventory/status` - Inventory status
- `GET /api/inventory/alerts` - Inventory alerts
- `POST /api/inventory/alerts/<id>/resolve` - Resolve alert

### Analytics
- `GET /api/analytics/dashboard` - Dashboard analytics
- `GET /api/analytics/savings` - Cost savings report

## Configuration

Configuration is stored in `deployment/config.json`. Key settings include:

- **procurement_rules**: Approval thresholds and reorder rules
- **supplier_selection**: Supplier scoring weights
- **negotiation_settings**: Automated negotiation parameters
- **alert_settings**: Inventory alert thresholds
- **automation_settings**: Automation feature flags

## Monitoring

### Logs
- Application logs: `deployment/logs/procurement_system.log`
- API access logs: Handled by Flask/Nginx
- System logs: Check systemd journal if using systemd

### Health Checks
- HTTP endpoint: `GET /health`
- Returns system status and service health

### Metrics
- Procurement cycle performance
- Model prediction accuracy
- Cost savings achieved
- Supplier performance trends

## Maintenance

### Model Updates
Models are automatically loaded from the `models/` directory. To update:
1. Train new models using the training pipeline
2. Restart the service to load new models

### Database Maintenance
- Database: SQLite (default) or configure PostgreSQL/MySQL
- Backup: Regular backups of `deployment/procurement.db`
- Cleanup: Archive old records periodically

### Updates
1. Stop the service
2. Update code
3. Install new dependencies if needed
4. Restart the service

## Troubleshooting

### Common Issues
1. **Service won't start**: Check logs for dependency issues
2. **Model loading errors**: Verify model files exist and are valid
3. **Database errors**: Check file permissions and disk space
4. **API timeouts**: Verify model prediction performance

### Debug Mode
Set `FLASK_ENV=development` for detailed error messages.

## Security Considerations

- Change default configuration values
- Use HTTPS in production
- Implement authentication for API endpoints
- Regular security updates
- Monitor for suspicious activity

## Support

For issues and questions:
1. Check application logs
2. Review this documentation
3. Contact system administrator
"""
        
        with open(self.deployment_dir / 'README.md', 'w') as f:
            f.write(documentation)
        
        logger.info("Deployment documentation created")
    
    def run_health_check(self):
        """Run post-deployment health check"""
        logger.info("Running post-deployment health check...")
        
        try:
            # Import and test model service
            sys.path.insert(0, str(self.project_root))
            from deployment.model_service import ProcurementModelService
            
            model_service = ProcurementModelService()
            logger.info("Model service initialized successfully")
            
            # Test model loading
            if len(model_service.models) == 4:
                logger.info("All 4 models loaded successfully")
            else:
                logger.warning(f"Only {len(model_service.models)} models loaded")
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise
        
        logger.info("Health check completed successfully")
    
    def deploy(self):
        """Run complete deployment process"""
        logger.info("Starting deployment process...")
        
        try:
            # Validation
            self.validate_environment()
            self.validate_models()
            
            # Setup
            self.install_dependencies()
            self.setup_database()
            self.configure_logging()
            
            # Create deployment files
            self.create_startup_script()
            self.create_systemd_service()
            self.create_docker_files()
            self.create_nginx_config()
            self.create_deployment_documentation()
            
            # Final health check
            self.run_health_check()
            
            logger.info("Deployment completed successfully!")
            logger.info(f"Service ready to start from: {self.deployment_dir}")
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise

def main():
    """Main deployment function"""
    parser = argparse.ArgumentParser(description='Deploy Automated Procurement System')
    parser.add_argument('--environment', default='production', choices=['development', 'staging', 'production'])
    parser.add_argument('--skip-dependencies', action='store_true', help='Skip dependency installation')
    parser.add_argument('--health-check-only', action='store_true', help='Run health check only')
    
    args = parser.parse_args()
    
    deployment = ProcurementDeployment(args.environment)
    
    if args.health_check_only:
        deployment.run_health_check()
    else:
        deployment.deploy()

if __name__ == '__main__':
    main()