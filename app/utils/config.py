import os
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # Database
    database_url: str = "sqlite:///./shopping_assistant.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # AI Model Settings
    default_ai_model: str = "gpt-3.5-turbo"
    max_tokens: int = 500
    temperature: float = 0.7
    
    # Application Settings
    app_name: str = "AI Shopping Assistant"
    debug: bool = False
    secret_key: str = "your-secret-key-here"
    
    # Discount Settings
    max_discount_percentage: float = 90.0
    default_discount_expiry_days: int = 30
    
    # Recommendation Settings
    max_recommendations: int = 10
    similarity_threshold: float = 0.5
    
    # Chat Settings
    max_conversation_history: int = 50
    session_timeout_minutes: int = 60
    
    # Email Settings (for notifications)
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    
    class Config:
        env_file = ".env"

# Create global settings instance
settings = Settings()

class DiscountConfig:
    """Discount-specific configuration"""
    
    DISCOUNT_TYPES = {
        'percentage': 'Percentage Discount',
        'fixed_amount': 'Fixed Amount Discount', 
        'bogo': 'Buy One Get One',
        'free_shipping': 'Free Shipping'
    }
    
    USER_SEGMENTS = {
        'new': 'New Customer',
        'regular': 'Regular Customer',
        'vip': 'VIP Customer',
        'all': 'All Customers'
    }
    
    PROMOTION_PROGRAMS = {
        'BLACK_FRIDAY': 'Black Friday Sale',
        'CHRISTMAS': 'Christmas Special',
        'NEW_YEAR': 'New Year Promotion',
        'FLASH_SALE': 'Flash Sale',
        'CLEARANCE': 'Clearance Sale',
        'SEASONAL': 'Seasonal Sale'
    }

class AIConfig:
    """AI-specific configuration"""
    
    INTENTS = [
        "find_products",
        "find_discounts", 
        "product_recommendation",
        "price_inquiry",
        "order_status",
        "order_management",
        "complaint",
        "greeting",
        "goodbye",
        "discount_management",
        "promotion_inquiry"
    ]
    
    ENTITIES = [
        'product_category',
        'brand',
        'price_range',
        'color',
        'size',
        'discount_type',
        'order_number',
        'date_range'
    ]
    
    RESPONSE_TEMPLATES = {
        'greeting': [
            "Hello! I'm your AI shopping assistant. How can I help you today?",
            "Hi there! Looking for great deals or need product recommendations?",
            "Welcome! I'm here to help you find the best products and discounts."
        ],
        'product_found': [
            "I found {count} products matching your criteria:",
            "Here are {count} great options for you:",
            "Perfect! I discovered {count} products that might interest you:"
        ],
        'discount_found': [
            "Great news! I found {count} active discounts for you:",
            "You're in luck! Here are {count} amazing deals:",
            "I discovered {count} discounts that could save you money:"
        ],
        'no_results': [
            "I couldn't find anything matching your criteria. Would you like to try different terms?",
            "No results found. Let me help you with alternative options.",
            "Sorry, no matches found. How about we explore similar products?"
        ]
    }

def get_ai_config():
    """Get AI configuration"""
    return AIConfig()

def get_discount_config():
    """Get discount configuration"""
    return DiscountConfig() 