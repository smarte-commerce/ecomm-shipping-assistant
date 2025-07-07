from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import logging

from app.utils.database import get_db, create_tables
from app.api import chat, products, discounts, notifications
from app.utils.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="AI-powered shopping assistant with discount management",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Include API routers
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(products.router, prefix="/api/products", tags=["products"])
app.include_router(discounts.router, prefix="/api/discounts", tags=["discounts"])
app.include_router(notifications.router, prefix="/api/notifications", tags=["notifications"])

@app.on_event("startup")
async def startup_event():
    """Initialize database and create tables on startup"""
    try:
        create_tables()
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")

@app.get("/")
async def root():
    """Root endpoint - serves the frontend"""
    return {
        "message": "AI Shopping Assistant API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    try:
        # Simple database check
        db.execute("SELECT 1")
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": "2024-01-01T00:00:00Z"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.get("/info")
async def app_info():
    """Get application information"""
    return {
        "app_name": settings.app_name,
        "version": "1.0.0",
        "environment": "development" if settings.debug else "production",
        "features": [
            "AI Chat Assistant",
            "Product Recommendations",
            "Discount Management",
            "Order Management",
            "Promotion Programs"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    ) 