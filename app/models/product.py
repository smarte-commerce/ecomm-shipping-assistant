from sqlalchemy import Column, Integer, String, Float, Text, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Product(Base):
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    category = Column(String(100), index=True)
    brand = Column(String(100), index=True)
    price = Column(Float, nullable=False)
    original_price = Column(Float)
    stock_quantity = Column(Integer, default=0)
    image_url = Column(String(500))
    features = Column(JSON)  # Store product features as JSON
    tags = Column(JSON)  # Store product tags for better search
    rating = Column(Float, default=0.0)
    review_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    is_featured = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'brand': self.brand,
            'price': self.price,
            'original_price': self.original_price,
            'stock_quantity': self.stock_quantity,
            'image_url': self.image_url,
            'features': self.features,
            'tags': self.tags,
            'rating': self.rating,
            'review_count': self.review_count,
            'is_active': self.is_active,
            'is_featured': self.is_featured,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        } 