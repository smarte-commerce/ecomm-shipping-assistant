from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Dict, Any, List

Base = declarative_base()

class Order(Base):
    """Order model for tracking customer orders"""
    
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, index=True)
    order_number = Column(String(50), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    
    # Order Details
    total_amount = Column(Float, nullable=False)
    discount_amount = Column(Float, default=0.0)
    tax_amount = Column(Float, default=0.0)
    shipping_amount = Column(Float, default=0.0)
    final_amount = Column(Float, nullable=False)
    
    # Order Items (JSON array of product details)
    items = Column(JSON, nullable=False)  # [{"product_id": 1, "quantity": 2, "price": 99.99, "name": "Product Name"}]
    
    # Applied Discounts
    applied_discounts = Column(JSON)  # [{"discount_id": 1, "code": "SAVE20", "amount": 10.0}]
    
    # Status Management
    status = Column(String(50), default="pending")  # pending, confirmed, processing, shipped, delivered, cancelled, returned
    payment_status = Column(String(50), default="pending")  # pending, paid, failed, refunded
    
    # Shipping Information
    shipping_address = Column(JSON)  # {"address": "123 Main St", "city": "City", "state": "State", "zip": "12345", "country": "US"}
    billing_address = Column(JSON)
    
    shipping_method = Column(String(100))  # standard, express, overnight
    tracking_number = Column(String(100))
    estimated_delivery = Column(DateTime)
    actual_delivery = Column(DateTime)
    
    # Customer Information
    customer_notes = Column(Text)
    internal_notes = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    confirmed_at = Column(DateTime)
    shipped_at = Column(DateTime)
    delivered_at = Column(DateTime)
    
    # Metadata
    source = Column(String(50), default="chat")  # chat, web, mobile, api
    session_id = Column(String(100))  # Link to chat session
    conversation_id = Column(Integer, ForeignKey('conversations.id'))
    
    def __repr__(self):
        return f"<Order(order_number='{self.order_number}', user_id={self.user_id}, status='{self.status}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary"""
        return {
            'id': self.id,
            'order_number': self.order_number,
            'user_id': self.user_id,
            'total_amount': self.total_amount,
            'discount_amount': self.discount_amount,
            'tax_amount': self.tax_amount,
            'shipping_amount': self.shipping_amount,
            'final_amount': self.final_amount,
            'items': self.items,
            'applied_discounts': self.applied_discounts,
            'status': self.status,
            'payment_status': self.payment_status,
            'shipping_address': self.shipping_address,
            'billing_address': self.billing_address,
            'shipping_method': self.shipping_method,
            'tracking_number': self.tracking_number,
            'estimated_delivery': self.estimated_delivery.isoformat() if self.estimated_delivery else None,
            'actual_delivery': self.actual_delivery.isoformat() if self.actual_delivery else None,
            'customer_notes': self.customer_notes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'confirmed_at': self.confirmed_at.isoformat() if self.confirmed_at else None,
            'shipped_at': self.shipped_at.isoformat() if self.shipped_at else None,
            'delivered_at': self.delivered_at.isoformat() if self.delivered_at else None,
            'source': self.source,
            'session_id': self.session_id
        }
    
    def get_status_display(self) -> str:
        """Get human-readable status"""
        status_map = {
            'pending': 'Order Pending',
            'confirmed': 'Order Confirmed',
            'processing': 'Processing',
            'shipped': 'Shipped',
            'delivered': 'Delivered',
            'cancelled': 'Cancelled',
            'returned': 'Returned'
        }
        return status_map.get(self.status, self.status.title())
    
    def get_total_items(self) -> int:
        """Get total number of items in order"""
        if not self.items:
            return 0
        return sum(item.get('quantity', 1) for item in self.items)
    
    def can_be_cancelled(self) -> bool:
        """Check if order can be cancelled"""
        return self.status in ['pending', 'confirmed', 'processing']
    
    def can_be_returned(self) -> bool:
        """Check if order can be returned"""
        return self.status == 'delivered' and self.delivered_at
    
    def update_status(self, new_status: str, notes: str = None):
        """Update order status with timestamp"""
        old_status = self.status
        self.status = new_status
        self.updated_at = datetime.utcnow()
        
        # Set specific timestamps
        if new_status == 'confirmed' and not self.confirmed_at:
            self.confirmed_at = datetime.utcnow()
        elif new_status == 'shipped' and not self.shipped_at:
            self.shipped_at = datetime.utcnow()
        elif new_status == 'delivered' and not self.delivered_at:
            self.delivered_at = datetime.utcnow()
        
        # Add internal notes
        if notes:
            current_notes = self.internal_notes or ""
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            self.internal_notes = f"{current_notes}\n[{timestamp}] Status changed from {old_status} to {new_status}: {notes}".strip()

class OrderItem(Base):
    """Individual items within an order (alternative to JSON storage)"""
    
    __tablename__ = "order_items"
    
    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(Integer, ForeignKey('orders.id'), nullable=False)
    product_id = Column(Integer, ForeignKey('products.id'), nullable=False)
    
    # Item Details
    product_name = Column(String(255), nullable=False)  # Snapshot of product name at time of order
    product_sku = Column(String(100))
    quantity = Column(Integer, nullable=False, default=1)
    unit_price = Column(Float, nullable=False)
    total_price = Column(Float, nullable=False)
    
    # Product Variants
    size = Column(String(20))
    color = Column(String(50))
    variant_details = Column(JSON)  # Additional variant information
    
    # Discounts applied to this item
    discount_amount = Column(Float, default=0.0)
    discount_details = Column(JSON)  # Details of discounts applied
    
    # Status for individual items (for partial fulfillment)
    status = Column(String(50), default="pending")  # pending, confirmed, shipped, delivered, cancelled
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<OrderItem(order_id={self.order_id}, product_id={self.product_id}, quantity={self.quantity})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order item to dictionary"""
        return {
            'id': self.id,
            'order_id': self.order_id,
            'product_id': self.product_id,
            'product_name': self.product_name,
            'product_sku': self.product_sku,
            'quantity': self.quantity,
            'unit_price': self.unit_price,
            'total_price': self.total_price,
            'size': self.size,
            'color': self.color,
            'variant_details': self.variant_details,
            'discount_amount': self.discount_amount,
            'discount_details': self.discount_details,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        } 