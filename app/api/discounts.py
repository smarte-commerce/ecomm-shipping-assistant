from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging

from app.utils.database import get_db
from app.services.discount_service import DiscountService
from app.models.discount import Discount

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models
class DiscountCreate(BaseModel):
    name: str
    description: Optional[str] = None
    discount_type: str  # percentage, fixed_amount, bogo, free_shipping
    discount_value: float
    min_purchase_amount: float = 0
    max_discount_amount: Optional[float] = None
    category: Optional[str] = "all"
    brand: Optional[str] = "all"
    user_segment: Optional[str] = "all"
    start_date: datetime
    end_date: datetime
    usage_limit: Optional[int] = None
    usage_limit_per_user: Optional[int] = None
    is_stackable: bool = False
    is_automatic: bool = False
    promotion_program: Optional[str] = None
    priority: int = 1

class DiscountUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    discount_value: Optional[float] = None
    min_purchase_amount: Optional[float] = None
    max_discount_amount: Optional[float] = None
    end_date: Optional[datetime] = None
    usage_limit: Optional[int] = None
    priority: Optional[int] = None
    is_active: Optional[bool] = None

class DiscountApplication(BaseModel):
    discount_code: str
    purchase_amount: float
    user_id: Optional[int] = None
    order_id: Optional[str] = None

class PromotionCampaign(BaseModel):
    program: str
    start_date: datetime
    end_date: datetime
    discounts: List[DiscountCreate]

# Discount CRUD Operations

@router.get("/")
async def get_active_discounts(
    category: Optional[str] = Query(None, description="Filter by category"),
    user_segment: Optional[str] = Query(None, description="Filter by user segment"),
    promotion_program: Optional[str] = Query(None, description="Filter by promotion program"),
    limit: int = Query(20, ge=1, le=100, description="Number of discounts to return"),
    db: Session = Depends(get_db)
):
    """
    Get active discounts with optional filtering
    """
    try:
        discount_service = DiscountService(db)
        
        if promotion_program:
            discounts = discount_service.get_discounts_for_promotion_program(promotion_program)
        else:
            discount_models = discount_service.get_active_discounts(category, user_segment)
            discounts = [discount.to_dict() for discount in discount_models[:limit]]
        
        return {
            "discounts": discounts,
            "total": len(discounts),
            "filters": {
                "category": category,
                "user_segment": user_segment,
                "promotion_program": promotion_program
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting active discounts: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving discounts")

@router.post("/")
async def create_discount(
    discount_data: DiscountCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new discount
    """
    try:
        discount_service = DiscountService(db)
        result = discount_service.create_discount(discount_data.dict())
        
        if result["success"]:
            return {
                "message": "Discount created successfully",
                "discount_id": result["discount_id"],
                "discount_code": result["discount_code"]
            }
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating discount: {e}")
        raise HTTPException(status_code=500, detail="Error creating discount")

@router.get("/{discount_id}")
async def get_discount(discount_id: int, db: Session = Depends(get_db)):
    """
    Get a specific discount by ID
    """
    try:
        discount = db.query(Discount).filter(Discount.id == discount_id).first()
        if not discount:
            raise HTTPException(status_code=404, detail="Discount not found")
        
        return discount.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting discount: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving discount")

@router.put("/{discount_id}")
async def update_discount(
    discount_id: int,
    update_data: DiscountUpdate,
    db: Session = Depends(get_db)
):
    """
    Update an existing discount
    """
    try:
        discount_service = DiscountService(db)
        result = discount_service.update_discount(
            discount_id, 
            update_data.dict(exclude_unset=True)
        )
        
        if result["success"]:
            return {"message": result["message"]}
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating discount: {e}")
        raise HTTPException(status_code=500, detail="Error updating discount")

@router.delete("/{discount_id}")
async def deactivate_discount(discount_id: int, db: Session = Depends(get_db)):
    """
    Deactivate a discount
    """
    try:
        discount_service = DiscountService(db)
        result = discount_service.deactivate_discount(discount_id)
        
        if result["success"]:
            return {"message": result["message"]}
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deactivating discount: {e}")
        raise HTTPException(status_code=500, detail="Error deactivating discount")

# Discount Application

@router.post("/apply")
async def apply_discount(
    application: DiscountApplication,
    db: Session = Depends(get_db)
):
    """
    Apply a discount code to a purchase
    """
    try:
        discount_service = DiscountService(db)
        result = discount_service.apply_discount(
            discount_code=application.discount_code,
            purchase_amount=application.purchase_amount,
            user_id=application.user_id,
            order_id=application.order_id
        )
        
        if result["success"]:
            return {
                "success": True,
                "discount_amount": result["discount_amount"],
                "final_amount": result["final_amount"],
                "savings_percentage": result["savings_percentage"],
                "description": result["discount_description"]
            }
        else:
            return {
                "success": False,
                "error": result["error"]
            }
            
    except Exception as e:
        logger.error(f"Error applying discount: {e}")
        raise HTTPException(status_code=500, detail="Error applying discount")

@router.post("/validate/{discount_code}")
async def validate_discount_code(
    discount_code: str,
    purchase_amount: float = Query(..., description="Purchase amount to validate against"),
    user_id: Optional[int] = Query(None, description="User ID for eligibility check"),
    db: Session = Depends(get_db)
):
    """
    Validate a discount code without applying it
    """
    try:
        discount = db.query(Discount).filter(
            Discount.code == discount_code.upper(),
            Discount.is_active == True
        ).first()
        
        if not discount:
            return {"valid": False, "reason": "Invalid discount code"}
        
        discount_service = DiscountService(db)
        calculation = discount_service.calculate_discount_amount(
            discount, purchase_amount, user_id=user_id
        )
        
        return calculation
        
    except Exception as e:
        logger.error(f"Error validating discount code: {e}")
        raise HTTPException(status_code=500, detail="Error validating discount code")

# Recommendations

@router.post("/recommend")
async def recommend_discounts(
    request: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """
    Get discount recommendations based on user preferences and entities
    """
    try:
        discount_service = DiscountService(db)
        entities = request.get("entities", {})
        user_id = request.get("user_id")
        
        discounts = discount_service.get_relevant_discounts(entities, user_id)
        
        return {
            "discounts": discounts,
            "recommendations_count": len(discounts),
            "criteria": entities
        }
        
    except Exception as e:
        logger.error(f"Error getting discount recommendations: {e}")
        raise HTTPException(status_code=500, detail="Error getting discount recommendations")

@router.get("/user/{user_id}/personalized")
async def get_personalized_discounts(
    user_id: int,
    limit: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db)
):
    """
    Get personalized discount recommendations for a user
    """
    try:
        from app.ai.recommendation_engine import RecommendationEngine
        
        recommendation_engine = RecommendationEngine(db)
        discounts = recommendation_engine.get_personalized_discounts(user_id, limit)
        
        return {
            "user_id": user_id,
            "personalized_discounts": discounts,
            "count": len(discounts)
        }
        
    except Exception as e:
        logger.error(f"Error getting personalized discounts: {e}")
        raise HTTPException(status_code=500, detail="Error getting personalized discounts")

# Promotion Programs

@router.post("/campaigns")
async def create_promotion_campaign(
    campaign: PromotionCampaign,
    db: Session = Depends(get_db)
):
    """
    Create a comprehensive promotion campaign with multiple discounts
    """
    try:
        discount_service = DiscountService(db)
        campaign_data = campaign.dict()
        
        result = discount_service.create_promotion_campaign(campaign_data)
        
        if result["success"]:
            return {
                "message": f"Promotion campaign '{campaign.program}' created successfully",
                "campaign": result["campaign"],
                "created_discounts": result["created_discounts"],
                "discount_ids": result["discount_ids"]
            }
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating promotion campaign: {e}")
        raise HTTPException(status_code=500, detail="Error creating promotion campaign")

@router.get("/programs/{program_name}")
async def get_promotion_program(
    program_name: str,
    db: Session = Depends(get_db)
):
    """
    Get all discounts for a specific promotion program
    """
    try:
        discount_service = DiscountService(db)
        discounts = discount_service.get_discounts_for_promotion_program(program_name)
        
        return {
            "program": program_name,
            "discounts": discounts,
            "total_discounts": len(discounts)
        }
        
    except Exception as e:
        logger.error(f"Error getting promotion program: {e}")
        raise HTTPException(status_code=500, detail="Error getting promotion program")

@router.get("/programs")
async def list_promotion_programs(db: Session = Depends(get_db)):
    """
    Get list of all active promotion programs
    """
    try:
        from app.utils.config import get_discount_config
        
        discount_config = get_discount_config()
        programs = discount_config.PROMOTION_PROGRAMS
        
        # Get active programs from database
        active_programs = db.query(Discount.promotion_program).filter(
            Discount.is_active == True,
            Discount.promotion_program.isnot(None)
        ).distinct().all()
        
        active_program_names = [p[0] for p in active_programs if p[0]]
        
        return {
            "available_programs": programs,
            "active_programs": active_program_names
        }
        
    except Exception as e:
        logger.error(f"Error listing promotion programs: {e}")
        raise HTTPException(status_code=500, detail="Error listing promotion programs")

# Analytics

@router.get("/{discount_id}/analytics")
async def get_discount_analytics(discount_id: int, db: Session = Depends(get_db)):
    """
    Get analytics for a specific discount
    """
    try:
        discount_service = DiscountService(db)
        analytics = discount_service.get_discount_analytics(discount_id)
        
        if "error" in analytics:
            raise HTTPException(status_code=404, detail=analytics["error"])
        
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting discount analytics: {e}")
        raise HTTPException(status_code=500, detail="Error getting discount analytics")

# Automatic Discounts

@router.post("/auto-create")
async def create_automatic_discount(
    request: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """
    Create automatic discount for specific products (e.g., inventory clearance)
    """
    try:
        discount_service = DiscountService(db)
        
        product_id = request.get("product_id")
        discount_percentage = request.get("discount_percentage", 10.0)
        reason = request.get("reason", "inventory_clearance")
        
        if not product_id:
            raise HTTPException(status_code=400, detail="Product ID is required")
        
        result = discount_service.create_automatic_discount(
            product_id, discount_percentage, reason
        )
        
        if result["success"]:
            return {
                "message": "Automatic discount created successfully",
                "discount_id": result["discount_id"],
                "discount_code": result["discount_code"]
            }
        else:
            raise HTTPException(status_code=400, detail=result["error"])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating automatic discount: {e}")
        raise HTTPException(status_code=500, detail="Error creating automatic discount") 