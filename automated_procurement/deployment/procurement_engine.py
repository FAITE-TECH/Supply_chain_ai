#!/usr/bin/env python3
"""
Automated Procurement Engine
This engine orchestrates the entire procurement process including supplier selection,
price negotiation, and automated ordering based on AI predictions
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime, timedelta
import sqlite3
import requests
from dataclasses import dataclass
from enum import Enum
import time

from model_service import ProcurementModelService

logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    ORDERED = "ordered"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

class PriorityLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class Supplier:
    """Supplier information"""
    id: str
    name: str
    contact_email: str
    phone: str
    reliability_score: float
    lead_time_days: int
    min_order_value: float
    payment_terms: str
    product_categories: List[str]
    pricing_tier: str

@dataclass
class Product:
    """Product information"""
    id: str
    name: str
    category: str
    current_stock: int
    min_stock_level: int
    max_stock_level: int
    unit_cost: float
    lead_time: int
    supplier_ids: List[str]

@dataclass
class ProcurementOrder:
    """Procurement order"""
    order_id: str
    product_id: str
    supplier_id: str
    quantity: int
    unit_price: float
    total_amount: float
    priority: PriorityLevel
    status: OrderStatus
    created_at: datetime
    expected_delivery: datetime
    negotiated_price: Optional[float] = None

class AutomatedProcurementEngine:
    """Main procurement automation engine"""
    
    def __init__(self, 
                 models_path: str = "../models",
                 database_path: str = "deployment/procurement.db",
                 config_path: str = "deployment/config.json"):
        self.models_path = Path(models_path)
        self.database_path = Path(database_path)
        self.config_path = Path(config_path)
        
        # Initialize model service
        self.model_service = ProcurementModelService(models_path)
        
        # Initialize database
        self._init_database()
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize supplier and product data
        self.suppliers = self._load_suppliers()
        self.products = self._load_products()
        
        logger.info("Automated Procurement Engine initialized successfully")
    
    def _init_database(self):
        """Initialize SQLite database for procurement data"""
        try:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS procurement_orders (
                    order_id TEXT PRIMARY KEY,
                    product_id TEXT NOT NULL,
                    supplier_id TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    unit_price REAL NOT NULL,
                    total_amount REAL NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expected_delivery TEXT NOT NULL,
                    negotiated_price REAL,
                    actual_delivery TEXT,
                    delivery_rating INTEGER
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS inventory_alerts (
                    alert_id TEXT PRIMARY KEY,
                    product_id TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT,
                    action_taken TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS supplier_performance (
                    record_id TEXT PRIMARY KEY,
                    supplier_id TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    delivery_time_days INTEGER,
                    quality_rating INTEGER,
                    price_competitiveness REAL,
                    communication_rating INTEGER,
                    overall_rating REAL,
                    recorded_at TEXT NOT NULL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_negotiations (
                    negotiation_id TEXT PRIMARY KEY,
                    product_id TEXT NOT NULL,
                    supplier_id TEXT NOT NULL,
                    original_price REAL NOT NULL,
                    negotiated_price REAL NOT NULL,
                    savings_amount REAL NOT NULL,
                    savings_percentage REAL NOT NULL,
                    negotiation_method TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    valid_until TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def _load_config(self) -> Dict[str, Any]:
        """Load engine configuration"""
        default_config = {
            "procurement_rules": {
                "auto_approve_threshold": 1000.0,
                "critical_stock_percentage": 0.1,
                "reorder_point_multiplier": 1.5,
                "max_order_value_without_approval": 5000.0
            },
            "supplier_selection": {
                "reliability_weight": 0.4,
                "price_weight": 0.3,
                "delivery_time_weight": 0.2,
                "quality_weight": 0.1
            },
            "negotiation_settings": {
                "enable_auto_negotiation": True,
                "max_negotiation_rounds": 3,
                "target_savings_percentage": 0.05,
                "negotiation_timeout_hours": 24
            },
            "alert_settings": {
                "low_stock_threshold": 0.2,
                "stockout_risk_threshold": 0.7,
                "delivery_delay_threshold_days": 2
            }
        }
        
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            else:
                # Create default config file
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                return default_config
                
        except Exception as e:
            logger.warning(f"Error loading config, using defaults: {e}")
            return default_config
    
    def _load_suppliers(self) -> Dict[str, Supplier]:
        """Load supplier data (in production, this would come from a database/API)"""
        # Sample supplier data - replace with actual data source
        suppliers_data = [
            {
                "id": "SUP001",
                "name": "Global Materials Inc",
                "contact_email": "procurement@globalmaterials.com",
                "phone": "+1-555-0001",
                "reliability_score": 0.92,
                "lead_time_days": 7,
                "min_order_value": 500.0,
                "payment_terms": "Net 30",
                "product_categories": ["Electronics", "Components"],
                "pricing_tier": "Premium"
            },
            {
                "id": "SUP002", 
                "name": "Reliable Supplies Co",
                "contact_email": "orders@reliablesupplies.com",
                "phone": "+1-555-0002",
                "reliability_score": 0.88,
                "lead_time_days": 5,
                "min_order_value": 300.0,
                "payment_terms": "Net 15",
                "product_categories": ["Office Supplies", "General"],
                "pricing_tier": "Standard"
            },
            {
                "id": "SUP003",
                "name": "FastTrack Logistics",
                "contact_email": "sales@fasttrack.com", 
                "phone": "+1-555-0003",
                "reliability_score": 0.85,
                "lead_time_days": 3,
                "min_order_value": 200.0,
                "payment_terms": "Net 10",
                "product_categories": ["Logistics", "Packaging"],
                "pricing_tier": "Economy"
            }
        ]
        
        suppliers = {}
        for data in suppliers_data:
            supplier = Supplier(**data)
            suppliers[supplier.id] = supplier
        
        return suppliers
    
    def _load_products(self) -> Dict[str, Product]:
        """Load product data (in production, this would come from inventory system)"""
        # Sample product data - replace with actual inventory data
        products_data = [
            {
                "id": "PROD001",
                "name": "Electronic Component A",
                "category": "Electronics",
                "current_stock": 50,
                "min_stock_level": 100,
                "max_stock_level": 500,
                "unit_cost": 25.50,
                "lead_time": 7,
                "supplier_ids": ["SUP001", "SUP002"]
            },
            {
                "id": "PROD002",
                "name": "Office Paper",
                "category": "Office Supplies",
                "current_stock": 200,
                "min_stock_level": 150,
                "max_stock_level": 1000,
                "unit_cost": 12.00,
                "lead_time": 3,
                "supplier_ids": ["SUP002", "SUP003"]
            }
        ]
        
        products = {}
        for data in products_data:
            product = Product(**data)
            products[product.id] = product
        
        return products
    
    def run_procurement_cycle(self) -> Dict[str, Any]:
        """
        Run complete automated procurement cycle
        
        Returns:
            Summary of procurement cycle results
        """
        try:
            logger.info("Starting automated procurement cycle...")
            
            cycle_results = {
                'cycle_timestamp': datetime.now().isoformat(),
                'inventory_analysis': {},
                'supplier_analysis': {},
                'procurement_decisions': {},
                'orders_created': [],
                'alerts_generated': [],
                'negotiations_initiated': [],
                'cycle_summary': {}
            }
            
            # Step 1: Analyze current inventory and generate predictions
            inventory_analysis = self._analyze_inventory()
            cycle_results['inventory_analysis'] = inventory_analysis
            
            # Step 2: Analyze supplier performance
            supplier_analysis = self._analyze_suppliers()
            cycle_results['supplier_analysis'] = supplier_analysis
            
            # Step 3: Make procurement decisions
            procurement_decisions = self._make_procurement_decisions(
                inventory_analysis, supplier_analysis
            )
            cycle_results['procurement_decisions'] = procurement_decisions
            
            # Step 4: Generate automated orders
            orders_created = self._generate_automated_orders(procurement_decisions)
            cycle_results['orders_created'] = orders_created
            
            # Step 5: Initiate price negotiations
            negotiations = self._initiate_price_negotiations(orders_created)
            cycle_results['negotiations_initiated'] = negotiations
            
            # Step 6: Generate alerts and notifications
            alerts = self._generate_alerts(inventory_analysis, procurement_decisions)
            cycle_results['alerts_generated'] = alerts
            
            # Step 7: Create cycle summary
            cycle_results['cycle_summary'] = self._create_cycle_summary(cycle_results)
            
            logger.info("Procurement cycle completed successfully")
            return cycle_results
            
        except Exception as e:
            logger.error(f"Error in procurement cycle: {e}")
            raise
    
    def _analyze_inventory(self) -> Dict[str, Any]:
        """Analyze current inventory using AI models"""
        try:
            # Prepare feature data for all products
            features_data = []
            for product in self.products.values():
                # Create feature row (simplified - in production use actual feature engineering)
                feature_row = {
                    'Product Name': product.name,
                    'Current_Inventory': product.current_stock,
                    'Order Quantity_mean': product.current_stock,
                    'Demand_Variability': 0.2,
                    'Revenue_per_Unit': product.unit_cost * 1.3,
                    'Profit_Margin': 0.3,
                    'Order_Frequency': 4,
                    'Stockout_Frequency': 0.1 if product.current_stock > product.min_stock_level else 0.8,
                    'Delivery_Reliability': 0.85,
                    'Procurement_Priority_Score': 50,
                    # Add other required features with default values
                    'Order Quantity_std': 10,
                    'Demand_Growth_Mean': 0.05,
                    'Demand_Trend_Slope': 0.02,
                    'Low_Stock_Frequency': 0.15,
                    'Warehouse_Fulfillment_Days': 2,
                    'On_Time_Delivery_mean': 0.9,
                    'Safety_Stock': product.min_stock_level * 0.5,
                    'Reorder_Point': product.min_stock_level,
                    'EOQ': 100,
                    'Inventory_Days_Supply': 30,
                    'ABC_Category_encoded': 1,
                    'Stock_Status_encoded': 0 if product.current_stock > product.min_stock_level else 1
                }
                features_data.append(feature_row)
            
            features_df = pd.DataFrame(features_data)
            
            # Run comprehensive analysis using AI models
            comprehensive_analysis = self.model_service.comprehensive_procurement_analysis(features_df)
            
            # Add inventory-specific analysis
            inventory_analysis = {
                'ai_predictions': comprehensive_analysis,
                'low_stock_products': self._identify_low_stock_products(),
                'reorder_recommendations': self._generate_reorder_recommendations(comprehensive_analysis),
                'inventory_health_score': self._calculate_inventory_health_score()
            }
            
            return inventory_analysis
            
        except Exception as e:
            logger.error(f"Error in inventory analysis: {e}")
            return {}
    
    def _analyze_suppliers(self) -> Dict[str, Any]:
        """Analyze supplier performance and capabilities"""
        try:
            supplier_analysis = {
                'supplier_rankings': [],
                'performance_insights': {},
                'cost_analysis': {},
                'reliability_metrics': {}
            }
            
            # Analyze each supplier
            for supplier in self.suppliers.values():
                supplier_metrics = {
                    'supplier_id': supplier.id,
                    'name': supplier.name,
                    'reliability_score': supplier.reliability_score,
                    'lead_time_days': supplier.lead_time_days,
                    'pricing_tier': supplier.pricing_tier,
                    'cost_competitiveness': self._calculate_cost_competitiveness(supplier),
                    'delivery_performance': self._get_delivery_performance(supplier.id),
                    'overall_score': self._calculate_supplier_score(supplier)
                }
                supplier_analysis['supplier_rankings'].append(supplier_metrics)
            
            # Sort by overall score
            supplier_analysis['supplier_rankings'].sort(
                key=lambda x: x['overall_score'], reverse=True
            )
            
            # Generate insights
            supplier_analysis['performance_insights'] = self._generate_supplier_insights(
                supplier_analysis['supplier_rankings']
            )
            
            return supplier_analysis
            
        except Exception as e:
            logger.error(f"Error in supplier analysis: {e}")
            return {}
    
    def _make_procurement_decisions(self, inventory_analysis: Dict, supplier_analysis: Dict) -> Dict[str, Any]:
        """Make automated procurement decisions based on analysis"""
        try:
            decisions = {
                'immediate_orders': [],
                'scheduled_orders': [],
                'optimization_opportunities': [],
                'cost_savings_potential': 0.0
            }
            
            ai_predictions = inventory_analysis.get('ai_predictions', {})
            
            # Process urgent procurement items
            if 'procurement_priority' in ai_predictions:
                urgent_items = ai_predictions['procurement_priority'].get('urgent_items', [])
                for item in urgent_items:
                    product_id = f"PROD{item['index']:03d}"  # Map index to product ID
                    if product_id in self.products:
                        decision = self._create_procurement_decision(
                            product_id, 'immediate', item, supplier_analysis
                        )
                        decisions['immediate_orders'].append(decision)
            
            # Process high stockout risk items
            if 'stockout_risk' in ai_predictions:
                high_risk_products = ai_predictions['stockout_risk'].get('high_risk_products', [])
                for product in high_risk_products:
                    product_id = f"PROD{product['index']:03d}"
                    if product_id in self.products and product_id not in [d['product_id'] for d in decisions['immediate_orders']]:
                        decision = self._create_procurement_decision(
                            product_id, 'immediate', product, supplier_analysis
                        )
                        decisions['immediate_orders'].append(decision)
            
            # Calculate potential savings
            decisions['cost_savings_potential'] = self._calculate_savings_potential(decisions)
            
            return decisions
            
        except Exception as e:
            logger.error(f"Error making procurement decisions: {e}")
            return {}
    
    def _generate_automated_orders(self, procurement_decisions: Dict) -> List[ProcurementOrder]:
        """Generate automated procurement orders"""
        try:
            orders = []
            
            # Process immediate orders
            for decision in procurement_decisions.get('immediate_orders', []):
                order = self._create_procurement_order(decision, PriorityLevel.CRITICAL)
                if order:
                    orders.append(order)
                    self._save_order_to_database(order)
            
            # Process scheduled orders
            for decision in procurement_decisions.get('scheduled_orders', []):
                order = self._create_procurement_order(decision, PriorityLevel.HIGH)
                if order:
                    orders.append(order)
                    self._save_order_to_database(order)
            
            logger.info(f"Generated {len(orders)} automated orders")
            return orders
            
        except Exception as e:
            logger.error(f"Error generating automated orders: {e}")
            return []
    
    def _initiate_price_negotiations(self, orders: List[ProcurementOrder]) -> List[Dict[str, Any]]:
        """Initiate automated price negotiations"""
        try:
            negotiations = []
            
            if not self.config['negotiation_settings']['enable_auto_negotiation']:
                return negotiations
            
            for order in orders:
                if order.total_amount > self.config['procurement_rules']['auto_approve_threshold']:
                    negotiation = self._start_price_negotiation(order)
                    if negotiation:
                        negotiations.append(negotiation)
            
            return negotiations
            
        except Exception as e:
            logger.error(f"Error initiating price negotiations: {e}")
            return []
    
    def _generate_alerts(self, inventory_analysis: Dict, procurement_decisions: Dict) -> List[Dict[str, Any]]:
        """Generate alerts and notifications"""
        try:
            alerts = []
            
            # Low stock alerts
            for product in inventory_analysis.get('low_stock_products', []):
                alert = {
                    'alert_id': f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{product['id']}",
                    'type': 'low_stock',
                    'severity': 'high' if product['stock_ratio'] < 0.1 else 'medium',
                    'product_id': product['id'],
                    'message': f"Low stock alert for {product['name']}: {product['current_stock']} units remaining",
                    'created_at': datetime.now().isoformat(),
                    'action_required': True
                }
                alerts.append(alert)
                self._save_alert_to_database(alert)
            
            # High value order alerts
            high_value_orders = [
                order for order in procurement_decisions.get('immediate_orders', [])
                if order.get('total_amount', 0) > self.config['procurement_rules']['max_order_value_without_approval']
            ]
            
            for order in high_value_orders:
                alert = {
                    'alert_id': f"ALERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{order['product_id']}",
                    'type': 'high_value_order',
                    'severity': 'high',
                    'product_id': order['product_id'],
                    'message': f"High value order requires approval: ${order['total_amount']:.2f}",
                    'created_at': datetime.now().isoformat(),
                    'action_required': True
                }
                alerts.append(alert)
                self._save_alert_to_database(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
            return []
    
    def _identify_low_stock_products(self) -> List[Dict[str, Any]]:
        """Identify products with low stock levels"""
        low_stock_products = []
        
        for product in self.products.values():
            stock_ratio = product.current_stock / product.min_stock_level
            if stock_ratio <= self.config['alert_settings']['low_stock_threshold']:
                low_stock_products.append({
                    'id': product.id,
                    'name': product.name,
                    'current_stock': product.current_stock,
                    'min_stock_level': product.min_stock_level,
                    'stock_ratio': stock_ratio
                })
        
        return low_stock_products
    
    def _calculate_supplier_score(self, supplier: Supplier) -> float:
        """Calculate overall supplier score"""
        weights = self.config['supplier_selection']
        
        # Normalize scores (0-1 scale)
        reliability_score = supplier.reliability_score
        price_score = 1.0  # Placeholder - would be calculated from historical data
        delivery_score = max(0, 1 - (supplier.lead_time_days / 30))  # Normalize lead time
        quality_score = 0.85  # Placeholder - would come from quality ratings
        
        overall_score = (
            reliability_score * weights['reliability_weight'] +
            price_score * weights['price_weight'] +
            delivery_score * weights['delivery_time_weight'] +
            quality_score * weights['quality_weight']
        )
        
        return overall_score
    
    def _create_procurement_decision(self, product_id: str, urgency: str, 
                                   ai_item: Dict, supplier_analysis: Dict) -> Dict[str, Any]:
        """Create a procurement decision"""
        product = self.products.get(product_id)
        if not product:
            return {}
        
        # Select best supplier
        best_supplier = self._select_best_supplier(product, supplier_analysis)
        
        # Calculate order quantity
        order_quantity = self._calculate_order_quantity(product, urgency)
        
        # Estimate cost
        estimated_cost = order_quantity * product.unit_cost
        
        decision = {
            'product_id': product_id,
            'product_name': product.name,
            'supplier_id': best_supplier['supplier_id'] if best_supplier else 'SUP001',
            'supplier_name': best_supplier['name'] if best_supplier else 'Default Supplier',
            'order_quantity': order_quantity,
            'unit_price': product.unit_cost,
            'total_amount': estimated_cost,
            'urgency': urgency,
            'ai_score': ai_item.get('priority_score', ai_item.get('risk_score', 0)),
            'justification': self._generate_order_justification(product, ai_item, urgency)
        }
        
        return decision
    
    def _select_best_supplier(self, product: Product, supplier_analysis: Dict) -> Optional[Dict[str, Any]]:
        """Select the best supplier for a product"""
        available_suppliers = [
            s for s in supplier_analysis.get('supplier_rankings', [])
            if s['supplier_id'] in product.supplier_ids
        ]
        
        if available_suppliers:
            return available_suppliers[0]  # Return top-ranked supplier
        return None
    
    def _calculate_order_quantity(self, product: Product, urgency: str) -> int:
        """Calculate optimal order quantity"""
        if urgency == 'immediate':
            # Order enough to reach max stock level
            return max(0, product.max_stock_level - product.current_stock)
        else:
            # Order enough to reach reorder point
            return max(0, product.min_stock_level * 2 - product.current_stock)
    
    def _create_procurement_order(self, decision: Dict, priority: PriorityLevel) -> Optional[ProcurementOrder]:
        """Create a procurement order from a decision"""
        try:
            order_id = f"PO_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{decision['product_id']}"
            
            order = ProcurementOrder(
                order_id=order_id,
                product_id=decision['product_id'],
                supplier_id=decision['supplier_id'],
                quantity=decision['order_quantity'],
                unit_price=decision['unit_price'],
                total_amount=decision['total_amount'],
                priority=priority,
                status=OrderStatus.PENDING,
                created_at=datetime.now(),
                expected_delivery=datetime.now() + timedelta(days=7)  # Default 7 days
            )
            
            return order
            
        except Exception as e:
            logger.error(f"Error creating procurement order: {e}")
            return None
    
    def _save_order_to_database(self, order: ProcurementOrder):
        """Save order to database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO procurement_orders 
                (order_id, product_id, supplier_id, quantity, unit_price, total_amount,
                 priority, status, created_at, expected_delivery, negotiated_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                order.order_id, order.product_id, order.supplier_id, order.quantity,
                order.unit_price, order.total_amount, order.priority.value, order.status.value,
                order.created_at.isoformat(), order.expected_delivery.isoformat(),
                order.negotiated_price
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving order to database: {e}")
    
    def _save_alert_to_database(self, alert: Dict[str, Any]):
        """Save alert to database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO inventory_alerts 
                (alert_id, product_id, alert_type, severity, message, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                alert['alert_id'], alert['product_id'], alert['type'],
                alert['severity'], alert['message'], alert['created_at']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving alert to database: {e}")
    
    def _start_price_negotiation(self, order: ProcurementOrder) -> Optional[Dict[str, Any]]:
        """Start automated price negotiation"""
        try:
            target_savings = self.config['negotiation_settings']['target_savings_percentage']
            target_price = order.unit_price * (1 - target_savings)
            
            negotiation = {
                'negotiation_id': f"NEG_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{order.order_id}",
                'order_id': order.order_id,
                'product_id': order.product_id,
                'supplier_id': order.supplier_id,
                'original_price': order.unit_price,
                'target_price': target_price,
                'current_offer': order.unit_price,
                'negotiation_round': 1,
                'status': 'active',
                'created_at': datetime.now().isoformat(),
                'strategy': 'volume_discount',  # Could be ML-driven
                'expected_savings': order.total_amount * target_savings
            }
            
            # Simulate negotiation process (in production, integrate with supplier APIs)
            negotiated_result = self._simulate_negotiation(negotiation)
            
            if negotiated_result['success']:
                order.negotiated_price = negotiated_result['final_price']
                order.total_amount = order.quantity * negotiated_result['final_price']
                
                # Save negotiation result
                self._save_negotiation_result(negotiation, negotiated_result)
            
            return negotiation
            
        except Exception as e:
            logger.error(f"Error starting price negotiation: {e}")
            return None
    
    def _simulate_negotiation(self, negotiation: Dict) -> Dict[str, Any]:
        """Simulate price negotiation (replace with actual negotiation logic)"""
        try:
            # Simple simulation - in production, integrate with supplier systems
            original_price = negotiation['original_price']
            target_price = negotiation['target_price']
            
            # Simulate 60% success rate with 3-5% savings
            import random
            success = random.random() > 0.4
            
            if success:
                savings_percentage = random.uniform(0.03, 0.07)
                final_price = original_price * (1 - savings_percentage)
                final_price = max(final_price, target_price)  # Don't go below target
            else:
                final_price = original_price
            
            return {
                'success': success,
                'final_price': final_price,
                'savings_amount': original_price - final_price,
                'savings_percentage': (original_price - final_price) / original_price,
                'negotiation_method': 'automated_api'
            }
            
        except Exception as e:
            logger.error(f"Error in negotiation simulation: {e}")
            return {'success': False, 'final_price': negotiation['original_price']}
    
    def _save_negotiation_result(self, negotiation: Dict, result: Dict):
        """Save negotiation result to database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO price_negotiations 
                (negotiation_id, product_id, supplier_id, original_price, negotiated_price,
                 savings_amount, savings_percentage, negotiation_method, created_at, valid_until)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                negotiation['negotiation_id'], negotiation['product_id'], negotiation['supplier_id'],
                negotiation['original_price'], result['final_price'], result['savings_amount'],
                result['savings_percentage'], result['negotiation_method'],
                negotiation['created_at'], 
                (datetime.now() + timedelta(days=30)).isoformat()  # Valid for 30 days
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving negotiation result: {e}")
    
    def _calculate_savings_potential(self, decisions: Dict) -> float:
        """Calculate potential cost savings"""
        total_savings = 0.0
        
        for order in decisions.get('immediate_orders', []) + decisions.get('scheduled_orders', []):
            # Estimate 5% savings potential through negotiation
            potential_savings = order.get('total_amount', 0) * 0.05
            total_savings += potential_savings
        
        return total_savings
    
    def _create_cycle_summary(self, cycle_results: Dict) -> Dict[str, Any]:
        """Create summary of procurement cycle"""
        summary = {
            'total_orders_created': len(cycle_results.get('orders_created', [])),
            'total_order_value': sum(order.total_amount for order in cycle_results.get('orders_created', [])),
            'alerts_generated': len(cycle_results.get('alerts_generated', [])),
            'negotiations_initiated': len(cycle_results.get('negotiations_initiated', [])),
            'estimated_savings': cycle_results.get('procurement_decisions', {}).get('cost_savings_potential', 0),
            'processing_time_seconds': 0,  # Would calculate actual processing time
            'automation_rate': 0.85,  # Percentage of decisions made automatically
            'next_cycle_recommendation': self._get_next_cycle_recommendation(cycle_results)
        }
        
        return summary
    
    def _get_next_cycle_recommendation(self, cycle_results: Dict) -> str:
        """Get recommendation for next procurement cycle"""
        alerts_count = len(cycle_results.get('alerts_generated', []))
        orders_count = len(cycle_results.get('orders_created', []))
        
        if alerts_count > 5:
            return "Increase monitoring frequency due to high alert volume"
        elif orders_count > 10:
            return "Review supplier capacity and lead times"
        else:
            return "Continue normal procurement cycle"
    
    # Additional helper methods for calculations and insights
    def _generate_reorder_recommendations(self, ai_analysis: Dict) -> List[Dict[str, Any]]:
        """Generate reorder recommendations based on AI analysis"""
        recommendations = []
        
        # Extract recommendations from AI analysis
        if 'integrated_insights' in ai_analysis:
            for action in ai_analysis['integrated_insights'].get('critical_actions', []):
                recommendations.append({
                    'type': 'critical',
                    'action': action['action'],
                    'products_affected': action.get('items_count', 0),
                    'timeline': 'immediate'
                })
        
        return recommendations
    
    def _calculate_inventory_health_score(self) -> float:
        """Calculate overall inventory health score"""
        total_products = len(self.products)
        if total_products == 0:
            return 0.0
        
        healthy_products = sum(
            1 for product in self.products.values()
            if product.current_stock >= product.min_stock_level
        )
        
        return healthy_products / total_products
    
    def _calculate_cost_competitiveness(self, supplier: Supplier) -> float:
        """Calculate supplier cost competitiveness"""
        # Simplified calculation - in production, use historical pricing data
        tier_scores = {
            'Economy': 0.9,
            'Standard': 0.7,
            'Premium': 0.5
        }
        return tier_scores.get(supplier.pricing_tier, 0.6)
    
    def _get_delivery_performance(self, supplier_id: str) -> float:
        """Get historical delivery performance for supplier"""
        # In production, query actual delivery data from database
        return 0.85  # Default performance score
    
    def _generate_supplier_insights(self, supplier_rankings: List[Dict]) -> Dict[str, Any]:
        """Generate insights about supplier performance"""
        if not supplier_rankings:
            return {}
        
        avg_score = np.mean([s['overall_score'] for s in supplier_rankings])
        top_supplier = supplier_rankings[0]
        
        insights = {
            'average_supplier_score': float(avg_score),
            'top_performing_supplier': top_supplier['name'],
            'top_supplier_score': top_supplier['overall_score'],
            'suppliers_needing_improvement': [
                s['name'] for s in supplier_rankings 
                if s['overall_score'] < 0.7
            ],
            'cost_optimization_potential': self._calculate_cost_optimization_potential(supplier_rankings)
        }
        
        return insights
    
    def _calculate_cost_optimization_potential(self, supplier_rankings: List[Dict]) -> float:
        """Calculate potential cost optimization across suppliers"""
        # Simplified calculation
        return 0.15  # 15% potential savings through optimization
    
    def _generate_order_justification(self, product: Product, ai_item: Dict, urgency: str) -> str:
        """Generate justification for procurement order"""
        justifications = []
        
        if urgency == 'immediate':
            justifications.append("Immediate action required")
        
        if product.current_stock < product.min_stock_level:
            justifications.append(f"Below minimum stock level ({product.min_stock_level})")
        
        if 'risk_score' in ai_item and ai_item['risk_score'] > 0.7:
            justifications.append("High stockout risk predicted")
        
        if 'priority_score' in ai_item and ai_item['priority_score'] > 70:
            justifications.append("High procurement priority")
        
        return "; ".join(justifications) if justifications else "Standard reorder"