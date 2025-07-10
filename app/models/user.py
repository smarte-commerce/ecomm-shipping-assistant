from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True)
    phone = Column(String(20))
    first_name = Column(String(100))
    last_name = Column(String(100))
    preferences = Column(JSON)  # Store user preferences as JSON
    purchase_history = Column(JSON)  # Store purchase history
    browsing_history = Column(JSON)  # Store browsing patterns
    favorite_categories = Column(JSON)  # Store favorite product categories
    favorite_brands = Column(JSON)  # Store favorite brands
    price_range_preference = Column(JSON)  # Store preferred price ranges
    notification_settings = Column(JSON)  # Store notification preferences
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'email': self.email,
            'phone': self.phone,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'preferences': self.preferences,
            'purchase_history': self.purchase_history,
            'browsing_history': self.browsing_history,
            'favorite_categories': self.favorite_categories,
            'favorite_brands': self.favorite_brands,
            'price_range_preference': self.price_range_preference,
            'notification_settings': self.notification_settings,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_active': self.last_active.isoformat() if self.last_active else None
        }
    
    def get_full_name(self):
        """Get user's full name"""
        return f"{self.first_name} {self.last_name}".strip()
    
    def update_last_active(self):
        """Update last active timestamp"""
        self.last_active = datetime.utcnow() 