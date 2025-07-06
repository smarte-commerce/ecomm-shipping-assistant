from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Boolean, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True, index=True)
    user_id = Column(Integer, index=True)  # Can be null for anonymous users
    status = Column(String(20), default='active')  # active, completed, abandoned
    context = Column(JSON)  # Store conversation context
    metadata = Column(JSON)  # Store additional metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    ended_at = Column(DateTime)
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'status': self.status,
            'context': self.context,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'ended_at': self.ended_at.isoformat() if self.ended_at else None
        }


class Message(Base):
    __tablename__ = "messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey('conversations.id'), index=True)
    sender_type = Column(String(20), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    intent = Column(String(50))  # detected intent
    entities = Column(JSON)  # extracted entities
    response_metadata = Column(JSON)  # additional response data
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Define relationship
    conversation = relationship("Conversation", backref="messages")
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'conversation_id': self.conversation_id,
            'sender_type': self.sender_type,
            'content': self.content,
            'intent': self.intent,
            'entities': self.entities,
            'response_metadata': self.response_metadata,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


class Order(Base):
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True, index=True)
    order_number = Column(String(50), unique=True, index=True)
    user_id = Column(Integer, index=True)
    status = Column(String(20), default='pending')  # pending, confirmed, shipped, delivered, cancelled
    items = Column(JSON)  # Store order items as JSON
    subtotal = Column(Float, nullable=False)
    discount_amount = Column(Float, default=0)
    tax_amount = Column(Float, default=0)
    shipping_amount = Column(Float, default=0)
    total_amount = Column(Float, nullable=False)
    applied_discounts = Column(JSON)  # Store applied discount codes
    shipping_address = Column(JSON)  # Store shipping address
    billing_address = Column(JSON)  # Store billing address
    payment_method = Column(String(50))
    payment_status = Column(String(20), default='pending')  # pending, paid, failed, refunded
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    shipped_at = Column(DateTime)
    delivered_at = Column(DateTime)
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'order_number': self.order_number,
            'user_id': self.user_id,
            'status': self.status,
            'items': self.items,
            'subtotal': self.subtotal,
            'discount_amount': self.discount_amount,
            'tax_amount': self.tax_amount,
            'shipping_amount': self.shipping_amount,
            'total_amount': self.total_amount,
            'applied_discounts': self.applied_discounts,
            'shipping_address': self.shipping_address,
            'billing_address': self.billing_address,
            'payment_method': self.payment_method,
            'payment_status': self.payment_status,
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'shipped_at': self.shipped_at.isoformat() if self.shipped_at else None,
            'delivered_at': self.delivered_at.isoformat() if self.delivered_at else None
        } 