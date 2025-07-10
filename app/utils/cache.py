import redis
import json
import pickle
from typing import Any, Optional, Union
from app.utils.config import settings
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """Redis cache manager"""
    
    def __init__(self):
        try:
            self.redis_client = redis.from_url(settings.redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            self.available = True
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis_client = None
            self.available = False
    
    def set(self, key: str, value: Any, expire: int = 3600) -> bool:
        """Set cache value with expiration"""
        if not self.available:
            return False
        
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            elif not isinstance(value, str):
                value = str(value)
            
            return self.redis_client.setex(key, expire, value)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get cache value"""
        if not self.available:
            return None
        
        try:
            value = self.redis_client.get(key)
            if value is None:
                return None
            
            # Try to parse as JSON first
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete cache key"""
        if not self.available:
            return False
        
        try:
            return self.redis_client.delete(key) > 0
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if cache key exists"""
        if not self.available:
            return False
        
        try:
            return self.redis_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern"""
        if not self.available:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache clear pattern error: {e}")
            return 0

# Global cache instance
cache = CacheManager()

class ConversationCache:
    """Specialized cache for conversation data"""
    
    @staticmethod
    def set_conversation_context(session_id: str, context: dict, expire: int = 3600):
        """Store conversation context"""
        key = f"conversation:{session_id}"
        return cache.set(key, context, expire)
    
    @staticmethod
    def get_conversation_context(session_id: str) -> Optional[dict]:
        """Retrieve conversation context"""
        key = f"conversation:{session_id}"
        return cache.get(key)
    
    @staticmethod
    def clear_conversation(session_id: str):
        """Clear conversation cache"""
        key = f"conversation:{session_id}"
        return cache.delete(key)

class ProductCache:
    """Specialized cache for product data"""
    
    @staticmethod
    def set_product_recommendations(user_id: int, recommendations: list, expire: int = 1800):
        """Cache product recommendations"""
        key = f"recommendations:user:{user_id}"
        return cache.set(key, recommendations, expire)
    
    @staticmethod
    def get_product_recommendations(user_id: int) -> Optional[list]:
        """Get cached product recommendations"""
        key = f"recommendations:user:{user_id}"
        return cache.get(key)
    
    @staticmethod
    def set_product_search(query: str, results: list, expire: int = 900):
        """Cache product search results"""
        key = f"search:products:{hash(query)}"
        return cache.set(key, results, expire)
    
    @staticmethod
    def get_product_search(query: str) -> Optional[list]:
        """Get cached product search results"""
        key = f"search:products:{hash(query)}"
        return cache.get(key)

class DiscountCache:
    """Specialized cache for discount data"""
    
    @staticmethod
    def set_user_discounts(user_id: int, discounts: list, expire: int = 1800):
        """Cache user-specific discounts"""
        key = f"discounts:user:{user_id}"
        return cache.set(key, discounts, expire)
    
    @staticmethod
    def get_user_discounts(user_id: int) -> Optional[list]:
        """Get cached user discounts"""
        key = f"discounts:user:{user_id}"
        return cache.get(key)
    
    @staticmethod
    def set_category_discounts(category: str, discounts: list, expire: int = 3600):
        """Cache category-specific discounts"""
        key = f"discounts:category:{category}"
        return cache.set(key, discounts, expire)
    
    @staticmethod
    def get_category_discounts(category: str) -> Optional[list]:
        """Get cached category discounts"""
        key = f"discounts:category:{category}"
        return cache.get(key)
    
    @staticmethod
    def clear_discount_cache():
        """Clear all discount-related cache"""
        return cache.clear_pattern("discounts:*") 