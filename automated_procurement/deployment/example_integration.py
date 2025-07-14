#!/usr/bin/env python3
"""
Example Integration Script for Automated Procurement System
This script demonstrates how to integrate the procurement system with existing business systems
"""

import requests
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcurementSystemIntegration:
    """Integration client for the automated procurement system"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'ProcurementIntegration/1.0'
        })
    
    def health_check(self) -> bool:
        """Check if the procurement system is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json().get('status') == 'healthy'
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_demand_forecast(self, product_features: List[Dict]) -> Dict:
        """Get demand forecast for products"""
        try:
            payload = {'features': product_features}
            response = self.session.post(
                f"{self.base_url}/api/predict/demand",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Demand forecast failed: {e}")
            raise
    
    def get_stockout_risk(self, product_features: List[Dict]) -> Dict:
        """Get stockout risk predictions"""
        try:
            payload = {'features': product_features}
            response = self.session.post(
                f"{self.base_url}/api/predict/stockout",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Stockout risk prediction failed: {e}")
            raise
    
    def get_supplier_recommendations(self, product_id: str) -> Dict:
        """Get supplier recommendations for a product"""
        try:
            payload = {'product_id': product_id}
            response = self.session.post(
                f"{self.base_url}/api/suppliers/recommend",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Supplier recommendation failed: {e}")
            raise
    
    def trigger_procurement_cycle(self, force_run: bool = False) -> Dict:
        """Trigger an automated procurement cycle"""
        try:
            payload = {'force_run': force_run}
            response = self.session.post(
                f"{self.base_url}/api/procurement/cycle/run",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Procurement cycle trigger failed: {e}")
            raise
    
    def get_procurement_orders(self, status: str = None, limit: int = 100) -> Dict:
        """Get procurement orders"""
        try:
            params = {'limit': limit}
            if status:
                params['status'] = status
            
            response = self.session.get(
                f"{self.base_url}/api/procurement/orders",
                params=params
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Get procurement orders failed: {e}")
            raise
    
    def approve_order(self, order_id: str, approver: str, notes: str = "") -> Dict:
        """Approve a procurement order"""
        try:
            payload = {
                'approver': approver,
                'notes': notes
            }
            response = self.session.post(
                f"{self.base_url}/api/procurement/orders/{order_id}/approve",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Order approval failed: {e}")
            raise
    
    def get_inventory_status(self) -> Dict:
        """Get current inventory status"""
        try:
            response = self.session.get(f"{self.base_url}/api/inventory/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Get inventory status failed: {e}")
            raise
    
    def get_inventory_alerts(self, severity: str = None) -> Dict:
        """Get inventory alerts"""
        try:
            params = {}
            if severity:
                params['severity'] = severity
            
            response = self.session.get(
                f"{self.base_url}/api/inventory/alerts",
                params=params
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Get inventory alerts failed: {e}")
            raise
    
    def get_dashboard_analytics(self, days: int = 30) -> Dict:
        """Get dashboard analytics"""
        try:
            params = {'days': days}
            response = self.session.get(
                f"{self.base_url}/api/analytics/dashboard",
                params=params
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Get dashboard analytics failed: {e}")
            raise

class ERPIntegration:
    """Example ERP system integration"""
    
    def __init__(self, procurement_client: ProcurementSystemIntegration):
        self.procurement = procurement_client
    
    def sync_inventory_data(self):
        """Sync inventory data from ERP to procurement system"""
        logger.info("Syncing inventory data from ERP...")
        
        # Example: Get inventory data from ERP (mock data)
        erp_inventory = [
            {
                'Product Name': 'Electronic Component A',
                'Current_Inventory': 45,
                'Order Quantity_mean': 100,
                'Demand_Variability': 0.15,
                'Revenue_per_Unit': 33.15,
                'Profit_Margin': 0.3,
                'Order_Frequency': 4,
                'Stockout_Frequency': 0.05,
                'Delivery_Reliability': 0.9,
                'Procurement_Priority_Score': 75,
                'Order Quantity_std': 12,
                'Demand_Growth_Mean': 0.08,
                'Demand_Trend_Slope': 0.03,
                'Low_Stock_Frequency': 0.1,
                'Warehouse_Fulfillment_Days': 2,
                'On_Time_Delivery_mean': 0.92,
                'Safety_Stock': 20,
                'Reorder_Point': 50,
                'EOQ': 120,
                'Inventory_Days_Supply': 25,
                'ABC_Category_encoded': 1,
                'Stock_Status_encoded': 1
            },
            {
                'Product Name': 'Office Paper',
                'Current_Inventory': 150,
                'Order Quantity_mean': 200,
                'Demand_Variability': 0.1,
                'Revenue_per_Unit': 15.60,
                'Profit_Margin': 0.25,
                'Order_Frequency': 6,
                'Stockout_Frequency': 0.02,
                'Delivery_Reliability': 0.95,
                'Procurement_Priority_Score': 60,
                'Order Quantity_std': 8,
                'Demand_Growth_Mean': 0.03,
                'Demand_Trend_Slope': 0.01,
                'Low_Stock_Frequency': 0.05,
                'Warehouse_Fulfillment_Days': 1,
                'On_Time_Delivery_mean': 0.96,
                'Safety_Stock': 30,
                'Reorder_Point': 75,
                'EOQ': 150,
                'Inventory_Days_Supply': 45,
                'ABC_Category_encoded': 2,
                'Stock_Status_encoded': 0
            }
        ]
        
        # Get comprehensive analysis
        analysis = self.procurement.session.post(
            f"{self.procurement.base_url}/api/analyze/comprehensive",
            json={'features': erp_inventory}
        )
        
        if analysis.status_code == 200:
            result = analysis.json()
            logger.info("Inventory analysis completed successfully")
            return result['result']
        else:
            logger.error(f"Inventory analysis failed: {analysis.text}")
            return None
    
    def process_procurement_recommendations(self, analysis_result: Dict):
        """Process procurement recommendations from AI analysis"""
        logger.info("Processing procurement recommendations...")
        
        if not analysis_result:
            return
        
        # Extract urgent items
        urgent_items = analysis_result.get('procurement_priority', {}).get('urgent_items', [])
        
        for item in urgent_items:
            logger.info(f"Urgent procurement needed for: {item['product']} (Priority: {item['priority_level']})")
        
        # Extract high-risk stockout items
        high_risk_items = analysis_result.get('stockout_risk', {}).get('high_risk_products', [])
        
        for item in high_risk_items:
            logger.warning(f"High stockout risk for: {item['product']} (Risk: {item['risk_score']:.3f})")
        
        # Process integrated insights
        insights = analysis_result.get('integrated_insights', {})
        for action in insights.get('critical_actions', []):
            logger.critical(f"Critical action required: {action['action']} - {action['reason']}")
    
    def monitor_procurement_orders(self):
        """Monitor and process procurement orders"""
        logger.info("Monitoring procurement orders...")
        
        # Get pending orders
        orders = self.procurement.get_procurement_orders(status='pending')
        
        for order in orders.get('orders', []):
            order_id = order['order_id']
            total_amount = order.get('total_amount', 0)
            
            # Auto-approve small orders
            if total_amount < 1000:
                try:
                    self.procurement.approve_order(
                        order_id=order_id,
                        approver='system_auto',
                        notes='Auto-approved: Under approval threshold'
                    )
                    logger.info(f"Auto-approved order {order_id} for ${total_amount}")
                except Exception as e:
                    logger.error(f"Failed to auto-approve order {order_id}: {e}")
            else:
                logger.info(f"Order {order_id} requires manual approval: ${total_amount}")

class SupplierPortalIntegration:
    """Example supplier portal integration"""
    
    def __init__(self, procurement_client: ProcurementSystemIntegration):
        self.procurement = procurement_client
    
    def update_supplier_performance(self, supplier_id: str, performance_data: Dict):
        """Update supplier performance data"""
        logger.info(f"Updating performance for supplier {supplier_id}")
        
        # In a real implementation, this would update supplier data
        # and trigger model retraining if needed
        pass
    
    def process_supplier_quotes(self, quotes: List[Dict]):
        """Process supplier quotes and update pricing"""
        logger.info("Processing supplier quotes...")
        
        for quote in quotes:
            supplier_id = quote['supplier_id']
            product_id = quote['product_id']
            quoted_price = quote['price']
            
            logger.info(f"Quote from {supplier_id} for {product_id}: ${quoted_price}")
            
            # In a real implementation, this would:
            # 1. Update pricing data
            # 2. Trigger price comparison analysis
            # 3. Update supplier rankings

class ReportingIntegration:
    """Example reporting system integration"""
    
    def __init__(self, procurement_client: ProcurementSystemIntegration):
        self.procurement = procurement_client
    
    def generate_daily_report(self):
        """Generate daily procurement report"""
        logger.info("Generating daily procurement report...")
        
        # Get analytics data
        analytics = self.procurement.get_dashboard_analytics(days=1)
        
        if analytics.get('status') == 'success':
            data = analytics['analytics']
            
            report = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'summary': {
                    'total_orders': data['procurement_metrics']['total_orders'],
                    'total_value': data['procurement_metrics']['total_order_value'],
                    'cost_savings': data['procurement_metrics']['cost_savings'],
                    'automation_rate': data['procurement_metrics']['automation_rate']
                },
                'alerts': self.procurement.get_inventory_alerts()['alerts'],
                'recommendations': []
            }
            
            # Save report
            report_filename = f"procurement_report_{datetime.now().strftime('%Y%m%d')}.json"
            with open(report_filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Daily report saved: {report_filename}")
            return report
        else:
            logger.error("Failed to generate daily report")
            return None

def main():
    """Main integration example"""
    logger.info("Starting procurement system integration example...")
    
    # Initialize integration client
    procurement = ProcurementSystemIntegration()
    
    # Health check
    if not procurement.health_check():
        logger.error("Procurement system is not healthy. Aborting integration.")
        return
    
    logger.info("Procurement system is healthy. Starting integration...")
    
    # Initialize integration modules
    erp = ERPIntegration(procurement)
    supplier_portal = SupplierPortalIntegration(procurement)
    reporting = ReportingIntegration(procurement)
    
    try:
        # Example workflow
        logger.info("Starting integration workflow...")
        
        # Step 1: Sync inventory data from ERP
        analysis_result = erp.sync_inventory_data()
        
        # Step 2: Process procurement recommendations
        erp.process_procurement_recommendations(analysis_result)
        
        # Step 3: Trigger procurement cycle
        cycle_result = procurement.trigger_procurement_cycle(force_run=True)
        logger.info(f"Procurement cycle completed: {cycle_result['result']['cycle_summary']}")
        
        # Step 4: Monitor and approve orders
        erp.monitor_procurement_orders()
        
        # Step 5: Get inventory status
        inventory_status = procurement.get_inventory_status()
        logger.info(f"Inventory health score: {inventory_status['inventory_status']['inventory_health_score']}")
        
        # Step 6: Check alerts
        alerts = procurement.get_inventory_alerts(severity='high')
        if alerts['alerts']:
            logger.warning(f"High severity alerts: {len(alerts['alerts'])}")
            for alert in alerts['alerts']:
                logger.warning(f"Alert: {alert['message']}")
        
        # Step 7: Generate daily report
        report = reporting.generate_daily_report()
        if report:
            logger.info("Daily report generated successfully")
        
        logger.info("Integration workflow completed successfully!")
        
    except Exception as e:
        logger.error(f"Integration workflow failed: {e}")
        raise

if __name__ == '__main__':
    main()