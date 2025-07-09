from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional, List
import logging

from app.utils.database import get_db
from app.models.product import Product

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/")
async def get_products(
    category: Optional[str] = Query(None),
    brand: Optional[str] = Query(None),
    min_price: Optional[float] = Query(None),
    max_price: Optional[float] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get products with optional filtering"""
    try:
        query = db.query(Product).filter(Product.is_active == True)
        
        if category:
            query = query.filter(Product.category.ilike(f'%{category}%'))
        if brand:
            query = query.filter(Product.brand.ilike(f'%{brand}%'))
        if min_price is not None:
            query = query.filter(Product.price >= min_price)
        if max_price is not None:
            query = query.filter(Product.price <= max_price)
        
        products = query.limit(limit).all()
        return {"products": [product.to_dict() for product in products]}
        
    except Exception as e:
        logger.error(f"Error getting products: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving products")

@router.get("/{product_id}")
async def get_product(product_id: int, db: Session = Depends(get_db)):
    """Get a specific product by ID"""
    try:
        product = db.query(Product).filter(Product.id == product_id).first()
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        
        return product.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting product: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving product") 