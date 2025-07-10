from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid
import logging

from app.models.discount import Discount
from app.models.product import Product
from app.models.user import User
from app.utils.cache import DiscountCache
from app.utils.config import get_discount_config, settings

logger = logging.getLogger(__name__)

class DiscountService:
    """Comprehensive discount management service"""
    
    def __init__(self, db: Session):
        self.db = db
        self.discount_config = get_discount_config()
    
    # Core Discount Operations
    
    def get_active_discounts(self, category: str = None, user_segment: str = None) -> List[Discount]:
        """Get all active discounts with optional filtering"""
        current_time = datetime.utcnow()
        
        query = self.db.query(Discount).filter(
            Discount.is_active == True,
            Discount.start_date <= current_time,
            Discount.end_date >= current_time
        )
        
        # Apply filters
        if category and category != 'all':
            query = query.filter(
                or_(Discount.category == 'all', Discount.category == category)
            )
        
        if user_segment and user_segment != 'all':
            query = query.filter(
                or_(Discount.user_segment == 'all', Discount.user_segment == user_segment)
            )
        
        return query.order_by(Discount.priority.desc(), Discount.discount_value.desc()).all()
    
    def get_relevant_discounts(self, entities: Dict, user_id: int = None) -> List[Dict]:
        """Get discounts relevant to user query"""
        # Check cache first
        if user_id:
            cached_discounts = DiscountCache.get_user_discounts(user_id)
            if cached_discounts:
                return self._filter_discounts_by_entities(cached_discounts, entities)
        
        try:
            # Determine user segment
            user_segment = 'all'
            if user_id:
                user = self.db.query(User).filter(User.id == user_id).first()
                if user:
                    user_segment = self._determine_user_segment(user)
            
            # Get active discounts
            discounts = self.get_active_discounts(
                category=entities.get('category'),
                user_segment=user_segment
            )
            
            # Convert to dict format and score
            discount_dicts = []
            for discount in discounts:
                discount_dict = discount.to_dict()
                discount_dict['relevance_score'] = self._calculate_relevance_score(
                    discount, entities, user_id
                )
                discount_dicts.append(discount_dict)
            
            # Sort by relevance score
            discount_dicts.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Cache results for user
            if user_id:
                DiscountCache.set_user_discounts(user_id, discount_dicts)
            
            return discount_dicts[:5]  # Return top 5
            
        except Exception as e:
            logger.error(f"Error getting relevant discounts: {e}")
            return []
    
    def _filter_discounts_by_entities(self, discounts: List[Dict], entities: Dict) -> List[Dict]:
        """Filter cached discounts by entities"""
        filtered = []
        
        for discount in discounts:
            relevance_score = 1.0
            
            # Check category match
            if 'category' in entities:
                if discount['category'] == 'all' or discount['category'] == entities['category']:
                    relevance_score *= 2.0
                else:
                    relevance_score *= 0.5
            
            # Check brand match
            if 'brand' in entities:
                if discount['brand'] == 'all' or discount['brand'] == entities['brand']:
                    relevance_score *= 1.5
                else:
                    relevance_score *= 0.7
            
            # Check discount type match
            if 'discount_type' in entities:
                if discount['discount_type'] == entities['discount_type']:
                    relevance_score *= 2.0
            
            discount['relevance_score'] = relevance_score
            filtered.append(discount)
        
        return sorted(filtered, key=lambda x: x['relevance_score'], reverse=True)[:5]
    
    def _calculate_relevance_score(self, discount: Discount, entities: Dict, user_id: int = None) -> float:
        """Calculate relevance score for a discount"""
        score = 1.0
        
        # Category matching
        if 'category' in entities:
            if discount.category == 'all':
                score += 0.5
            elif discount.category == entities['category']:
                score += 2.0
            else:
                score *= 0.3
        
        # Brand matching
        if 'brand' in entities:
            if discount.brand == 'all':
                score += 0.3
            elif discount.brand == entities['brand']:
                score += 1.5
            else:
                score *= 0.5
        
        # Discount type matching
        if 'discount_type' in entities:
            if discount.discount_type == entities['discount_type']:
                score += 2.0
        
        # Price range consideration
        if 'price_range' in entities:
            price_range = entities['price_range']
            if discount.min_purchase_amount <= price_range['max']:
                score += 1.0
        
        # Priority bonus
        score += discount.priority * 0.2
        
        # Usage availability bonus
        if discount.usage_limit:
            availability_ratio = (discount.usage_limit - discount.usage_count) / discount.usage_limit
            score += availability_ratio * 0.5
        
        return score
    
    def get_discounts_for_promotion_program(self, program: str) -> List[Dict]:
        """Get discounts for a specific promotion program"""
        try:
            current_time = datetime.utcnow()
            discounts = self.db.query(Discount).filter(
                Discount.is_active == True,
                Discount.promotion_program.ilike(f'%{program}%'),
                Discount.start_date <= current_time,
                Discount.end_date >= current_time
            ).order_by(Discount.priority.desc()).all()
            
            return [discount.to_dict() for discount in discounts]
            
        except Exception as e:
            logger.error(f"Error getting promotion discounts: {e}")
            return []
    
    # Discount Application and Validation
    
    def calculate_discount_amount(self, discount: Discount, purchase_amount: float, 
                                 quantity: int = 1, user_id: int = None) -> Dict[str, Any]:
        """Calculate discount amount for a purchase"""
        try:
            # Validate discount
            if not discount.is_valid():
                return {"valid": False, "reason": "Discount is not valid or expired"}
            
            # Check minimum purchase amount
            if purchase_amount < discount.min_purchase_amount:
                return {
                    "valid": False, 
                    "reason": f"Minimum purchase amount is ${discount.min_purchase_amount:.2f}"
                }
            
            # Check user segment eligibility
            if user_id:
                user = self.db.query(User).filter(User.id == user_id).first()
                if user:
                    user_segment = self._determine_user_segment(user)
                    if not discount.can_apply_to_user(user_segment):
                        return {"valid": False, "reason": "Not eligible for this discount"}
            
            # Calculate discount amount
            discount_amount = discount.calculate_discount_amount(purchase_amount, quantity)
            final_amount = purchase_amount - discount_amount
            
            return {
                "valid": True,
                "discount_amount": discount_amount,
                "final_amount": final_amount,
                "savings_percentage": (discount_amount / purchase_amount) * 100,
                "description": discount.description
            }
            
        except Exception as e:
            logger.error(f"Error calculating discount: {e}")
            return {"valid": False, "reason": "Error calculating discount"}
    
    def apply_discount(self, discount_code: str, purchase_amount: float, 
                      user_id: int = None, order_id: str = None) -> Dict[str, Any]:
        """Apply discount code to a purchase"""
        try:
            discount = self.db.query(Discount).filter(
                Discount.code == discount_code.upper(),
                Discount.is_active == True
            ).first()
            
            if not discount:
                return {"success": False, "error": "Invalid discount code"}
            
            # Calculate discount
            calculation = self.calculate_discount_amount(discount, purchase_amount, user_id=user_id)
            
            if not calculation["valid"]:
                return {"success": False, "error": calculation["reason"]}
            
            # Check usage limit
            if discount.usage_limit and discount.usage_count >= discount.usage_limit:
                return {"success": False, "error": "Discount usage limit exceeded"}
            
            # Update usage count
            discount.usage_count += 1
            self.db.commit()
            
            # Clear discount cache
            DiscountCache.clear_discount_cache()
            
            return {
                "success": True,
                "discount_id": discount.id,
                "discount_amount": calculation["discount_amount"],
                "final_amount": calculation["final_amount"],
                "savings_percentage": calculation["savings_percentage"],
                "discount_description": discount.description
            }
            
        except Exception as e:
            logger.error(f"Error applying discount: {e}")
            self.db.rollback()
            return {"success": False, "error": "Error applying discount"}
    
    # Discount Management and Creation
    
    def create_discount(self, discount_data: Dict) -> Dict[str, Any]:
        """Create a new discount"""
        try:
            # Generate unique code if not provided
            if 'code' not in discount_data or not discount_data['code']:
                discount_data['code'] = self._generate_discount_code()
            
            # Validate discount data
            validation_result = self._validate_discount_data(discount_data)
            if not validation_result['valid']:
                return {"success": False, "error": validation_result['error']}
            
            # Create discount
            discount = Discount(**discount_data)
            self.db.add(discount)
            self.db.commit()
            
            # Clear cache
            DiscountCache.clear_discount_cache()
            
            logger.info(f"Created discount: {discount.code}")
            
            return {
                "success": True,
                "discount_id": discount.id,
                "discount_code": discount.code,
                "message": "Discount created successfully"
            }
            
        except Exception as e:
            logger.error(f"Error creating discount: {e}")
            self.db.rollback()
            return {"success": False, "error": "Error creating discount"}
    
    def update_discount(self, discount_id: int, update_data: Dict) -> Dict[str, Any]:
        """Update an existing discount"""
        try:
            discount = self.db.query(Discount).filter(Discount.id == discount_id).first()
            if not discount:
                return {"success": False, "error": "Discount not found"}
            
            # Update fields
            for key, value in update_data.items():
                if hasattr(discount, key):
                    setattr(discount, key, value)
            
            discount.updated_at = datetime.utcnow()
            self.db.commit()
            
            # Clear cache
            DiscountCache.clear_discount_cache()
            
            return {
                "success": True,
                "message": "Discount updated successfully"
            }
            
        except Exception as e:
            logger.error(f"Error updating discount: {e}")
            self.db.rollback()
            return {"success": False, "error": "Error updating discount"}
    
    def deactivate_discount(self, discount_id: int) -> Dict[str, Any]:
        """Deactivate a discount"""
        try:
            discount = self.db.query(Discount).filter(Discount.id == discount_id).first()
            if not discount:
                return {"success": False, "error": "Discount not found"}
            
            discount.is_active = False
            discount.updated_at = datetime.utcnow()
            self.db.commit()
            
            # Clear cache
            DiscountCache.clear_discount_cache()
            
            return {
                "success": True,
                "message": "Discount deactivated successfully"
            }
            
        except Exception as e:
            logger.error(f"Error deactivating discount: {e}")
            self.db.rollback()
            return {"success": False, "error": "Error deactivating discount"}
    
    # Automatic Discount Management
    
    def create_automatic_discount(self, product_id: int, discount_percentage: float, 
                                 reason: str = "inventory_clearance") -> Dict[str, Any]:
        """Create automatic discount for products"""
        try:
            product = self.db.query(Product).filter(Product.id == product_id).first()
            if not product:
                return {"success": False, "error": "Product not found"}
            
            # Generate automatic discount
            discount_data = {
                "code": self._generate_discount_code("AUTO"),
                "name": f"Auto Discount - {product.name}",
                "description": f"Automatic {discount_percentage}% discount on {product.name}",
                "discount_type": "percentage",
                "discount_value": discount_percentage,
                "category": product.category,
                "product_ids": [product_id],
                "start_date": datetime.utcnow(),
                "end_date": datetime.utcnow() + timedelta(days=30),
                "is_automatic": True,
                "user_segment": "all",
                "priority": 1
            }
            
            return self.create_discount(discount_data)
            
        except Exception as e:
            logger.error(f"Error creating automatic discount: {e}")
            return {"success": False, "error": "Error creating automatic discount"}
    
    def create_promotion_campaign(self, campaign_data: Dict) -> Dict[str, Any]:
        """Create a comprehensive promotion campaign with multiple discounts"""
        try:
            campaign_discounts = []
            
            for discount_config in campaign_data.get('discounts', []):
                discount_config.update({
                    'promotion_program': campaign_data['program'],
                    'start_date': campaign_data['start_date'],
                    'end_date': campaign_data['end_date']
                })
                
                result = self.create_discount(discount_config)
                if result['success']:
                    campaign_discounts.append(result['discount_id'])
            
            return {
                "success": True,
                "campaign": campaign_data['program'],
                "created_discounts": len(campaign_discounts),
                "discount_ids": campaign_discounts
            }
            
        except Exception as e:
            logger.error(f"Error creating promotion campaign: {e}")
            return {"success": False, "error": "Error creating promotion campaign"}
    
    # Helper Methods
    
    def _generate_discount_code(self, prefix: str = "SAVE") -> str:
        """Generate unique discount code"""
        suffix = str(uuid.uuid4())[:8].upper()
        return f"{prefix}{suffix}"
    
    def _validate_discount_data(self, data: Dict) -> Dict[str, Any]:
        """Validate discount data"""
        required_fields = ['name', 'discount_type', 'discount_value', 'start_date', 'end_date']
        
        for field in required_fields:
            if field not in data:
                return {"valid": False, "error": f"Missing required field: {field}"}
        
        # Validate discount type
        if data['discount_type'] not in self.discount_config.DISCOUNT_TYPES:
            return {"valid": False, "error": "Invalid discount type"}
        
        # Validate discount value
        if data['discount_type'] == 'percentage' and data['discount_value'] > 100:
            return {"valid": False, "error": "Percentage discount cannot exceed 100%"}
        
        # Validate dates
        if data['end_date'] <= data['start_date']:
            return {"valid": False, "error": "End date must be after start date"}
        
        return {"valid": True}
    
    def _determine_user_segment(self, user: User) -> str:
        """Determine user segment for discount eligibility"""
        if not user.purchase_history:
            return 'new'
        
        purchase_count = len(user.purchase_history.get('products', []))
        total_spent = user.purchase_history.get('total_spent', 0)
        
        if purchase_count == 0:
            return 'new'
        elif purchase_count >= 10 or total_spent >= 1000:
            return 'vip'
        else:
            return 'regular'
    
    # Analytics and Reporting
    
    def get_discount_analytics(self, discount_id: int) -> Dict[str, Any]:
        """Get analytics for a specific discount"""
        try:
            discount = self.db.query(Discount).filter(Discount.id == discount_id).first()
            if not discount:
                return {"error": "Discount not found"}
            
            return {
                "discount_code": discount.code,
                "usage_count": discount.usage_count,
                "usage_limit": discount.usage_limit,
                "usage_rate": (discount.usage_count / discount.usage_limit * 100) if discount.usage_limit else 0,
                "total_savings": discount.usage_count * discount.discount_value,
                "status": "active" if discount.is_active else "inactive"
            }
            
        except Exception as e:
            logger.error(f"Error getting discount analytics: {e}")
            return {"error": "Error retrieving analytics"} 