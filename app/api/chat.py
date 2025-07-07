from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

from app.utils.database import get_db
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[int] = None
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    intent: str
    confidence: float
    entities: Dict[str, Any]
    suggestions: List[str]
    session_id: str
    response_type: str
    data: Dict[str, Any] = {}

class ConversationHistoryResponse(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]

@router.post("/message", response_model=ChatResponse)
async def chat_message(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Process a chat message and return AI response
    
    This endpoint handles all types of user queries including:
    - Product searches and recommendations
    - Discount and promotion inquiries
    - Order management and status checks
    - General shopping assistance
    """
    try:
        chat_service = ChatService(db)
        response = await chat_service.process_message(
            message=request.message,
            user_id=request.user_id,
            session_id=request.session_id
        )
        
        return ChatResponse(**response)
        
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Sorry, I encountered an error processing your message. Please try again."
        )

@router.get("/history/{session_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(session_id: str, db: Session = Depends(get_db)):
    """
    Get conversation history for a session
    """
    try:
        chat_service = ChatService(db)
        messages = chat_service.get_conversation_history(session_id)
        
        return ConversationHistoryResponse(
            session_id=session_id,
            messages=messages
        )
        
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving conversation history")

@router.post("/end-session/{session_id}")
async def end_chat_session(session_id: str, db: Session = Depends(get_db)):
    """
    End a chat session and cleanup resources
    """
    try:
        chat_service = ChatService(db)
        chat_service.end_conversation(session_id)
        
        return {"message": "Session ended successfully", "session_id": session_id}
        
    except Exception as e:
        logger.error(f"Error ending chat session: {e}")
        raise HTTPException(status_code=500, detail="Error ending session")

# Additional specialized endpoints

@router.post("/recommend-discounts")
async def recommend_discounts(
    request: Dict[str, Any], 
    db: Session = Depends(get_db)
):
    """
    Get discount recommendations based on user preferences
    """
    try:
        chat_service = ChatService(db)
        user_id = request.get('user_id')
        entities = request.get('entities', {})
        
        discounts = await chat_service._handle_find_discounts(entities, user_id)
        
        return {
            "discounts": discounts.get('discounts', []),
            "message": "Here are the best discounts for you"
        }
        
    except Exception as e:
        logger.error(f"Error getting discount recommendations: {e}")
        raise HTTPException(status_code=500, detail="Error getting discount recommendations")

@router.post("/recommend-products")
async def recommend_products(
    request: Dict[str, Any], 
    db: Session = Depends(get_db)
):
    """
    Get product recommendations based on user preferences
    """
    try:
        chat_service = ChatService(db)
        user_id = request.get('user_id')
        entities = request.get('entities', {})
        
        products = await chat_service._handle_product_recommendation(entities, user_id)
        
        return {
            "recommendations": products.get('recommendations', []),
            "message": "Here are my top product recommendations for you"
        }
        
    except Exception as e:
        logger.error(f"Error getting product recommendations: {e}")
        raise HTTPException(status_code=500, detail="Error getting product recommendations")

@router.get("/intents")
async def get_supported_intents():
    """
    Get list of supported intents and their descriptions
    """
    intents = {
        "find_products": "Search for products by category, brand, or price",
        "find_discounts": "Find available discounts and promotion codes",
        "product_recommendation": "Get personalized product recommendations",
        "price_inquiry": "Ask about product prices and price comparisons",
        "order_status": "Check the status of your orders",
        "order_management": "Cancel, modify, or return orders",
        "discount_management": "Create and manage discount codes",
        "promotion_inquiry": "Ask about current promotion programs",
        "complaint": "Report issues or complaints",
        "greeting": "Start a conversation",
        "goodbye": "End a conversation"
    }
    
    return {"supported_intents": intents}

@router.get("/suggestions/{intent}")
async def get_intent_suggestions(intent: str):
    """
    Get suggested follow-up questions for a specific intent
    """
    suggestions = {
        "find_products": [
            "Show me electronics under $500",
            "Find Nike shoes in size 10",
            "I need a laptop for gaming"
        ],
        "find_discounts": [
            "What discounts are available for electronics?",
            "Show me Black Friday deals",
            "Any coupons for first-time buyers?"
        ],
        "product_recommendation": [
            "Recommend a phone for photography",
            "What's the best laptop for students?",
            "Suggest gifts for my mom"
        ],
        "order_status": [
            "Where is my order #12345?",
            "When will my package arrive?",
            "Track my recent order"
        ]
    }
    
    return {
        "intent": intent,
        "suggestions": suggestions.get(intent, ["Ask me anything about shopping!"])
    } 