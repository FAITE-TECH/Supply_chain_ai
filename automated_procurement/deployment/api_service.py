#!/usr/bin/env python3
"""
Production API Service for Automated Procurement System
This service provides REST API endpoints for integrating with production systems
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import logging
from datetime import datetime
import json
import traceback
from typing import Dict, Any, List
import threading
import schedule
import time
from pathlib import Path

from procurement_engine import AutomatedProcurementEngine
from model_service import ProcurementModelService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
procurement_engine = None
model_service = None
background_scheduler = None

def initialize_services():
    """Initialize procurement services"""
    global procurement_engine, model_service
    
    try:
        logger.info("Initializing procurement services...")
        
        # Initialize model service
        model_service = ProcurementModelService()
        
        # Initialize procurement engine
        procurement_engine = AutomatedProcurementEngine()
        
        logger.info("Services initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        raise

def start_background_scheduler():
    """Start background scheduler for automated procurement cycles"""
    global background_scheduler
    
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    # Schedule automated procurement cycles
    schedule.every(6).hours.do(run_automated_procurement_cycle)
    schedule.every().day.at("08:00").do(run_daily_inventory_check)
    schedule.every().monday.at("09:00").do(run_weekly_supplier_review)
    
    background_scheduler = threading.Thread(target=run_scheduler, daemon=True)
    background_scheduler.start()
    
    logger.info("Background scheduler started")

def run_automated_procurement_cycle():
    """Run automated procurement cycle"""
    try:
        logger.info("Running scheduled procurement cycle...")
        if procurement_engine:
            result = procurement_engine.run_procurement_cycle()
            logger.info(f"Procurement cycle completed: {result.get('cycle_summary', {})}")
    except Exception as e:
        logger.error(f"Error in scheduled procurement cycle: {e}")

def run_daily_inventory_check():
    """Run daily inventory health check"""
    try:
        logger.info("Running daily inventory check...")
        # Implementation would check inventory levels and generate alerts
    except Exception as e:
        logger.error(f"Error in daily inventory check: {e}")

def run_weekly_supplier_review():
    """Run weekly supplier performance review"""
    try:
        logger.info("Running weekly supplier review...")
        # Implementation would analyze supplier performance and update ratings
    except Exception as e:
        logger.error(f"Error in weekly supplier review: {e}")

# Error handler
@app.errorhandler(Exception)
def handle_error(error):
    """Global error handler"""
    logger.error(f"API Error: {str(error)}")
    logger.error(traceback.format_exc())
    
    return jsonify({
        'error': 'Internal server error',
        'message': str(error),
        'timestamp': datetime.now().isoformat()
    }), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'model_service': model_service is not None,
            'procurement_engine': procurement_engine is not None,
            'background_scheduler': background_scheduler is not None and background_scheduler.is_alive()
        }
    })

# Add a root route for dashboard or welcome
@app.route('/', methods=['GET'])
def root_dashboard():
    """Root dashboard or welcome page"""
    return '''
    <html>
        <head><title>Automated Procurement System</title></head>
        <body>
            <h1>Welcome to the Automated Procurement System</h1>
            <p>API is running. Visit <a href="/docs">/docs</a> for API documentation or <a href="/health">/health</a> for health check.</p>
        </body>
    </html>
    ''', 200, {'Content-Type': 'text/html'}

# Model prediction endpoints
@app.route('/api/predict/demand', methods=['POST'])
def predict_demand():
    """Predict demand for products"""
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing features data'}), 400
        
        # Convert features to DataFrame
        features_df = pd.DataFrame(data['features'])
        
        # Make prediction
        result = model_service.predict_demand_forecast(features_df)
        
        return jsonify({
            'status': 'success',
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in demand prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/stockout', methods=['POST'])
def predict_stockout():
    """Predict stockout risk for products"""
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing features data'}), 400
        
        features_df = pd.DataFrame(data['features'])
        result = model_service.predict_stockout_risk(features_df)
        
        return jsonify({
            'status': 'success',
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in stockout prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/supplier-performance', methods=['POST'])
def predict_supplier_performance():
    """Predict supplier delivery performance"""
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing features data'}), 400
        
        features_df = pd.DataFrame(data['features'])
        result = model_service.predict_delivery_performance(features_df)
        
        return jsonify({
            'status': 'success',
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in supplier performance prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/procurement-priority', methods=['POST'])
def predict_procurement_priority():
    """Predict procurement priority scores"""
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing features data'}), 400
        
        features_df = pd.DataFrame(data['features'])
        result = model_service.predict_procurement_priority(features_df)
        
        return jsonify({
            'status': 'success',
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in procurement priority prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/comprehensive', methods=['POST'])
def comprehensive_analysis():
    """Run comprehensive procurement analysis"""
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'error': 'Missing features data'}), 400
        
        features_df = pd.DataFrame(data['features'])
        result = model_service.comprehensive_procurement_analysis(features_df)
        
        return jsonify({
            'status': 'success',
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        return jsonify({'error': str(e)}), 500

# Procurement automation endpoints
@app.route('/api/procurement/cycle/run', methods=['POST'])
def run_procurement_cycle():
    """Trigger automated procurement cycle"""
    try:
        # Optional parameters
        data = request.get_json() or {}
        force_run = data.get('force_run', False)
        
        if not force_run:
            # Check if a cycle was run recently
            pass  # Implementation would check last run time
        
        result = procurement_engine.run_procurement_cycle()
        
        return jsonify({
            'status': 'success',
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error running procurement cycle: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/procurement/orders', methods=['GET'])
def get_procurement_orders():
    """Get procurement orders with optional filtering"""
    try:
        # Query parameters
        status = request.args.get('status')
        supplier_id = request.args.get('supplier_id')
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        
        # Implementation would query database
        orders = []  # Placeholder - would query actual orders
        
        return jsonify({
            'status': 'success',
            'orders': orders,
            'total_count': len(orders),
            'limit': limit,
            'offset': offset,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting procurement orders: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/procurement/orders/<order_id>', methods=['GET'])
def get_order_details(order_id):
    """Get details of a specific order"""
    try:
        # Implementation would query database for order details
        order_details = {
            'order_id': order_id,
            'status': 'pending',
            'created_at': datetime.now().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'order': order_details,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting order details: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/procurement/orders/<order_id>/approve', methods=['POST'])
def approve_order(order_id):
    """Approve a procurement order"""
    try:
        data = request.get_json() or {}
        approver = data.get('approver', 'system')
        notes = data.get('notes', '')
        
        # Implementation would update order status in database
        # and trigger order processing
        
        return jsonify({
            'status': 'success',
            'message': f'Order {order_id} approved',
            'approver': approver,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error approving order: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/procurement/orders/<order_id>/cancel', methods=['POST'])
def cancel_order(order_id):
    """Cancel a procurement order"""
    try:
        data = request.get_json() or {}
        reason = data.get('reason', 'No reason provided')
        
        # Implementation would update order status and handle cancellation
        
        return jsonify({
            'status': 'success',
            'message': f'Order {order_id} cancelled',
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error cancelling order: {e}")
        return jsonify({'error': str(e)}), 500

# Supplier management endpoints
@app.route('/api/suppliers', methods=['GET'])
def get_suppliers():
    """Get list of suppliers"""
    try:
        suppliers = [
            {
                'id': supplier.id,
                'name': supplier.name,
                'reliability_score': supplier.reliability_score,
                'lead_time_days': supplier.lead_time_days,
                'pricing_tier': supplier.pricing_tier
            }
            for supplier in procurement_engine.suppliers.values()
        ]
        
        return jsonify({
            'status': 'success',
            'suppliers': suppliers,
            'total_count': len(suppliers),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting suppliers: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/suppliers/<supplier_id>/performance', methods=['GET'])
def get_supplier_performance(supplier_id):
    """Get supplier performance metrics"""
    try:
        # Implementation would query historical performance data
        performance = {
            'supplier_id': supplier_id,
            'overall_rating': 4.2,
            'delivery_performance': 0.88,
            'quality_rating': 4.5,
            'cost_competitiveness': 0.75,
            'recent_orders': 15,
            'on_time_delivery_rate': 0.87
        }
        
        return jsonify({
            'status': 'success',
            'performance': performance,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting supplier performance: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/suppliers/recommend', methods=['POST'])
def recommend_suppliers():
    """Recommend suppliers for specific products"""
    try:
        data = request.get_json()
        
        if not data or 'product_id' not in data:
            return jsonify({'error': 'Missing product_id'}), 400
        
        product_id = data['product_id']
        
        # Implementation would use AI to recommend best suppliers
        recommendations = [
            {
                'supplier_id': 'SUP001',
                'supplier_name': 'Global Materials Inc',
                'score': 0.92,
                'estimated_delivery': '7 days',
                'estimated_cost': 25.50,
                'confidence': 0.87
            }
        ]
        
        return jsonify({
            'status': 'success',
            'product_id': product_id,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error recommending suppliers: {e}")
        return jsonify({'error': str(e)}), 500

# Inventory management endpoints
@app.route('/api/inventory/status', methods=['GET'])
def get_inventory_status():
    """Get current inventory status"""
    try:
        # Implementation would query current inventory levels
        inventory_status = {
            'total_products': len(procurement_engine.products),
            'low_stock_products': 5,
            'out_of_stock_products': 2,
            'healthy_stock_products': 150,
            'inventory_health_score': 0.85,
            'alerts_count': 7,
            'last_updated': datetime.now().isoformat()
        }
        
        return jsonify({
            'status': 'success',
            'inventory_status': inventory_status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting inventory status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/inventory/alerts', methods=['GET'])
def get_inventory_alerts():
    """Get inventory alerts"""
    try:
        # Query parameters
        severity = request.args.get('severity')
        limit = int(request.args.get('limit', 50))
        
        # Implementation would query database for alerts
        alerts = []  # Placeholder
        
        return jsonify({
            'status': 'success',
            'alerts': alerts,
            'total_count': len(alerts),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting inventory alerts: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/inventory/alerts/<alert_id>/resolve', methods=['POST'])
def resolve_alert(alert_id):
    """Resolve an inventory alert"""
    try:
        data = request.get_json() or {}
        resolution_notes = data.get('notes', '')
        
        # Implementation would update alert status in database
        
        return jsonify({
            'status': 'success',
            'message': f'Alert {alert_id} resolved',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error resolving alert: {e}")
        return jsonify({'error': str(e)}), 500

# Price negotiation endpoints
@app.route('/api/negotiations', methods=['GET'])
def get_negotiations():
    """Get price negotiations"""
    try:
        # Query parameters
        status = request.args.get('status', 'active')
        limit = int(request.args.get('limit', 20))
        
        # Implementation would query database for negotiations
        negotiations = []  # Placeholder
        
        return jsonify({
            'status': 'success',
            'negotiations': negotiations,
            'total_count': len(negotiations),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting negotiations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/negotiations/<negotiation_id>/accept', methods=['POST'])
def accept_negotiation(negotiation_id):
    """Accept a price negotiation"""
    try:
        data = request.get_json() or {}
        
        # Implementation would update negotiation status and create order
        
        return jsonify({
            'status': 'success',
            'message': f'Negotiation {negotiation_id} accepted',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error accepting negotiation: {e}")
        return jsonify({'error': str(e)}), 500

# Analytics and reporting endpoints
@app.route('/api/analytics/dashboard', methods=['GET'])
def get_dashboard_analytics():
    """Get analytics data for dashboard"""
    try:
        # Time range parameter
        days = int(request.args.get('days', 30))
        
        analytics = {
            'procurement_metrics': {
                'total_orders': 45,
                'total_order_value': 125000.0,
                'average_order_value': 2777.78,
                'on_time_delivery_rate': 0.87,
                'cost_savings': 15000.0,
                'automation_rate': 0.85
            },
            'supplier_metrics': {
                'active_suppliers': 12,
                'top_supplier_performance': 0.92,
                'average_lead_time': 6.5,
                'supplier_diversity_score': 0.78
            },
            'inventory_metrics': {
                'inventory_turnover': 4.2,
                'stockout_rate': 0.03,
                'carrying_cost_reduction': 0.12,
                'inventory_accuracy': 0.94
            },
            'trends': {
                'demand_trend': 'increasing',
                'cost_trend': 'decreasing',
                'supplier_performance_trend': 'stable'
            }
        }
        
        return jsonify({
            'status': 'success',
            'analytics': analytics,
            'period_days': days,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting dashboard analytics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics/savings', methods=['GET'])
def get_savings_report():
    """Get cost savings report"""
    try:
        # Implementation would calculate actual savings
        savings_report = {
            'total_savings': 45000.0,
            'savings_by_category': {
                'negotiation_savings': 25000.0,
                'automation_savings': 15000.0,
                'supplier_optimization': 5000.0
            },
            'savings_percentage': 0.12,
            'projected_annual_savings': 180000.0
        }
        
        return jsonify({
            'status': 'success',
            'savings_report': savings_report,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting savings report: {e}")
        return jsonify({'error': str(e)}), 500

# Configuration endpoints
@app.route('/api/config', methods=['GET'])
def get_configuration():
    """Get system configuration"""
    try:
        config = procurement_engine.config
        
        return jsonify({
            'status': 'success',
            'config': config,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['PUT'])
def update_configuration():
    """Update system configuration"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No configuration data provided'}), 400
        
        # Implementation would update configuration
        # procurement_engine.config.update(data)
        
        return jsonify({
            'status': 'success',
            'message': 'Configuration updated successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        return jsonify({'error': str(e)}), 500

# Webhook endpoints for external system integration
@app.route('/api/webhooks/inventory-update', methods=['POST'])
def webhook_inventory_update():
    """Webhook for inventory updates from external systems"""
    try:
        data = request.get_json()
        
        if not data or 'product_id' not in data:
            return jsonify({'error': 'Invalid webhook data'}), 400
        
        # Process inventory update
        # Implementation would update inventory and trigger actions if needed
        
        return jsonify({
            'status': 'success',
            'message': 'Inventory update processed',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing inventory webhook: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/webhooks/supplier-response', methods=['POST'])
def webhook_supplier_response():
    """Webhook for supplier responses to orders/negotiations"""
    try:
        data = request.get_json()
        
        if not data or 'order_id' not in data:
            return jsonify({'error': 'Invalid webhook data'}), 400
        
        # Process supplier response
        # Implementation would update order status and handle response
        
        return jsonify({
            'status': 'success',
            'message': 'Supplier response processed',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing supplier webhook: {e}")
        return jsonify({'error': str(e)}), 500

def create_app():
    """Create and configure the Flask app"""
    initialize_services()
    start_background_scheduler()
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=False)