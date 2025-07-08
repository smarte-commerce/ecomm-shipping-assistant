from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from typing import Dict, Any

Base = declarative_base()

class Discount(Base):
    __tablename__ = "discounts"
    
    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(50), unique=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    discount_type = Column(String(20), nullable=False)  # percentage, fixed_amount, bogo, free_shipping
    discount_value = Column(Float, nullable=False)
    min_purchase_amount = Column(Float, default=0)
    max_discount_amount = Column(Float)
    category = Column(String(100))  # specific category or 'all'
    brand = Column(String(100))  # specific brand or 'all'
    product_ids = Column(JSON)  # specific product IDs or empty for all
    user_segment = Column(String(50))  # 'new', 'vip', 'regular', 'all'
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    usage_limit = Column(Integer)  # global usage limit
    usage_limit_per_user = Column(Integer)  # per user usage limit
    usage_count = Column(Integer, default=0)
    priority = Column(Integer, default=1)  # for stacking discounts
    is_stackable = Column(Boolean, default=False)  # can be combined with other discounts
    is_automatic = Column(Boolean, default=False)  # automatically applied
    is_active = Column(Boolean, default=True)
    promotion_program = Column(String(100))  # e.g., "BLACK_FRIDAY", "CHRISTMAS", "FLASH_SALE"
    conditions = Column(JSON)  # additional conditions as JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'code': self.code,
            'name': self.name,
            'description': self.description,
            'discount_type': self.discount_type,
            'discount_value': self.discount_value,
            'min_purchase_amount': self.min_purchase_amount,
            'max_discount_amount': self.max_discount_amount,
            'category': self.category,
            'brand': self.brand,
            'product_ids': self.product_ids,
            'user_segment': self.user_segment,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'usage_limit': self.usage_limit,
            'usage_limit_per_user': self.usage_limit_per_user,
            'usage_count': self.usage_count,
            'priority': self.priority,
            'is_stackable': self.is_stackable,
            'is_automatic': self.is_automatic,
            'is_active': self.is_active,
            'promotion_program': self.promotion_program,
            'conditions': self.conditions,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def is_valid(self, current_time: datetime = None) -> bool:
        """Check if discount is currently valid"""
        if current_time is None:
            current_time = datetime.utcnow()
        
        return (
            self.is_active and
            self.start_date <= current_time <= self.end_date and
            (self.usage_limit is None or self.usage_count < self.usage_limit)
        )
    
    def can_apply_to_user(self, user_segment: str) -> bool:
        """Check if discount can be applied to user segment"""
        return self.user_segment == 'all' or self.user_segment == user_segment
    
    def calculate_discount_amount(self, purchase_amount: float, quantity: int = 1) -> float:
        """Calculate discount amount for given purchase"""
        if purchase_amount < self.min_purchase_amount:
            return 0
        
        if self.discount_type == 'percentage':
            discount_amount = purchase_amount * (self.discount_value / 100)
        elif self.discount_type == 'fixed_amount':
            discount_amount = self.discount_value
        elif self.discount_type == 'bogo':
            # Buy One Get One - discount applies to cheapest item
            if quantity >= 2:
                discount_amount = purchase_amount / quantity  # Assuming equal priced items
            else:
                discount_amount = 0
        elif self.discount_type == 'free_shipping':
            discount_amount = self.discount_value  # shipping cost
        else:
            discount_amount = 0
        
        # Apply maximum discount limit
        if self.max_discount_amount:
            discount_amount = min(discount_amount, self.max_discount_amount)
        
        return discount_amount 