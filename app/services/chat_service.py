from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
import uuid
import logging
from datetime import datetime

from app.ai.intent_classifier import IntentClassifier
from app.ai.entity_extractor import EntityExtractor
from app.ai.recommendation_engine import RecommendationEngine
from app.ai.response_generator import ResponseGenerator
from app.services.discount_service import DiscountService
from app.models.conversation import Conversation, Message
from app.models.user import User
from app.models.product import Product
from app.models.order import Order
from app.utils.cache import ConversationCache

logger = logging.getLogger(__name__)

class ChatService:
    """Main chat service orchestrating AI and business logic"""
    
    def __init__(self, db: Session):
        self.db = db
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.recommendation_engine = RecommendationEngine(db)
        self.response_generator = ResponseGenerator()
        self.discount_service = DiscountService(db)
    
    async def process_message(self, message: str, user_id: int = None, 
                            session_id: str = None) -> Dict[str, Any]:
        """Process incoming message and generate response"""
        try:
            # Create or get session
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Get conversation context
            conversation = self._get_or_create_conversation(session_id, user_id)
            context = self._get_conversation_context(session_id)
            
            # Step 1: Classify intent
            intent, confidence = self.intent_classifier.classify(message)
            
            # Step 2: Extract entities
            entities = self.entity_extractor.extract(message)
            entities = self.entity_extractor.validate_entities(entities)
            
            # Step 3: Process based on intent
            response_data = await self._process_intent(intent, entities, user_id, context)
            
            # Step 4: Generate natural language response
            response = self.response_generator.generate_response(
                intent, entities, response_data
            )
            
            # Step 5: Store conversation
            self._save_message(conversation.id, 'user', message, intent, entities)
            self._save_message(conversation.id, 'assistant', response['response'], 
                             intent, response.get('data', {}))
            
            # Step 6: Update conversation context
            self._update_conversation_context(session_id, {
                'last_intent': intent,
                'last_entities': entities,
                'conversation_count': context.get('conversation_count', 0) + 1
            })
            
            return {
                "response": response['response'],
                "intent": intent,
                "confidence": confidence,
                "entities": entities,
                "suggestions": response.get('suggestions', []),
                "session_id": session_id,
                "response_type": response.get('response_type', 'general'),
                "data": response.get('data', {})
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "response": "I apologize, but I encountered an error. Please try again.",
                "intent": "error",
                "confidence": 0.0,
                "entities": {},
                "suggestions": ["Try rephrasing your question", "Contact support"],
                "session_id": session_id or str(uuid.uuid4())
            }
    
    async def _process_intent(self, intent: str, entities: Dict[str, Any], 
                            user_id: int = None, context: Dict = None) -> Dict[str, Any]:
        """Process different intents and return appropriate data"""
        
        if intent == "find_products":
            return await self._handle_find_products(entities, user_id)
        
        elif intent == "find_discounts":
            return await self._handle_find_discounts(entities, user_id)
        
        elif intent == "product_recommendation":
            return await self._handle_product_recommendation(entities, user_id)
        
        elif intent == "price_inquiry":
            return await self._handle_price_inquiry(entities, user_id)
        
        elif intent == "order_status":
            return await self._handle_order_status(entities, user_id)
        
        elif intent == "order_management":
            return await self._handle_order_management(entities, user_id)
        
        elif intent == "discount_management":
            return await self._handle_discount_management(entities, user_id)
        
        elif intent == "promotion_inquiry":
            return await self._handle_promotion_inquiry(entities, user_id)
        
        elif intent == "complaint":
            return await self._handle_complaint(entities, user_id)
        
        else:
            return {}
    
    async def _handle_find_products(self, entities: Dict, user_id: int = None) -> Dict[str, Any]:
        """Handle product search requests"""
        try:
            products = []
            
            # Search by category
            if 'category' in entities:
                products = self.recommendation_engine.get_recommendations_by_category(
                    entities['category'], n_recommendations=10
                )
            
            # Search by price range
            elif 'price_range' in entities:
                price_range = entities['price_range']
                products = self.recommendation_engine.get_recommendations_by_price_range(
                    price_range['min'], price_range['max'], 
                    category=entities.get('category'), n_recommendations=10
                )
            
            # General search - get popular products
            else:
                products = self.recommendation_engine.get_popular_products(n_recommendations=10)
            
            return {"products": products}
            
        except Exception as e:
            logger.error(f"Error handling find products: {e}")
            return {"products": []}
    
    async def _handle_find_discounts(self, entities: Dict, user_id: int = None) -> Dict[str, Any]:
        """Handle discount search requests"""
        try:
            discounts = self.discount_service.get_relevant_discounts(entities, user_id)
            return {"discounts": discounts}
            
        except Exception as e:
            logger.error(f"Error handling find discounts: {e}")
            return {"discounts": []}
    
    async def _handle_product_recommendation(self, entities: Dict, user_id: int = None) -> Dict[str, Any]:
        """Handle product recommendation requests"""
        try:
            if user_id:
                recommendations = self.recommendation_engine.get_personalized_recommendations(
                    user_id, n_recommendations=5
                )
            elif 'category' in entities:
                recommendations = self.recommendation_engine.get_recommendations_by_category(
                    entities['category'], n_recommendations=5
                )
            else:
                recommendations = self.recommendation_engine.get_popular_products(n_recommendations=5)
            
            return {"recommendations": recommendations}
            
        except Exception as e:
            logger.error(f"Error handling product recommendation: {e}")
            return {"recommendations": []}
    
    async def _handle_price_inquiry(self, entities: Dict, user_id: int = None) -> Dict[str, Any]:
        """Handle price inquiry requests"""
        try:
            products = []
            
            if 'product' in entities:
                # Search for specific product
                product_query = self.db.query(Product).filter(
                    Product.name.ilike(f'%{entities["product"]}%'),
                    Product.is_active == True
                ).first()
                
                if product_query:
                    products = [product_query.to_dict()]
            
            elif 'category' in entities:
                # Get products in category for price comparison
                products = self.recommendation_engine.get_recommendations_by_category(
                    entities['category'], n_recommendations=5
                )
            
            return {"products": products}
            
        except Exception as e:
            logger.error(f"Error handling price inquiry: {e}")
            return {"products": []}
    
    async def _handle_order_status(self, entities: Dict, user_id: int = None) -> Dict[str, Any]:
        """Handle order status inquiries"""
        try:
            orders = []
            
            if 'order_number' in entities and user_id:
                order = self.db.query(Order).filter(
                    Order.order_number == entities['order_number'],
                    Order.user_id == user_id
                ).first()
                
                if order:
                    orders = [order.to_dict()]
            
            elif user_id:
                # Get recent orders for user
                recent_orders = self.db.query(Order).filter(
                    Order.user_id == user_id
                ).order_by(Order.created_at.desc()).limit(5).all()
                
                orders = [order.to_dict() for order in recent_orders]
            
            return {"orders": orders}
            
        except Exception as e:
            logger.error(f"Error handling order status: {e}")
            return {"orders": []}
    
    async def _handle_order_management(self, entities: Dict, user_id: int = None) -> Dict[str, Any]:
        """Handle order management requests"""
        try:
            action_taken = None
            
            if 'order_number' in entities and user_id:
                order = self.db.query(Order).filter(
                    Order.order_number == entities['order_number'],
                    Order.user_id == user_id
                ).first()
                
                if order:
                    # Simulate order management actions
                    if 'cancel' in entities:
                        order.status = 'cancelled'
                        action_taken = 'cancelled'
                    elif 'return' in entities:
                        action_taken = 'return_initiated'
                    elif 'modify' in entities:
                        action_taken = 'modified'
                    
                    if action_taken:
                        self.db.commit()
            
            return {"action_taken": action_taken}
            
        except Exception as e:
            logger.error(f"Error handling order management: {e}")
            self.db.rollback()
            return {"action_taken": None}
    
    async def _handle_discount_management(self, entities: Dict, user_id: int = None) -> Dict[str, Any]:
        """Handle discount management requests"""
        try:
            action = None
            
            # This would typically require admin privileges
            if 'create' in entities or 'new' in entities:
                # For demo purposes, create a sample discount
                discount_data = {
                    'name': 'User Generated Discount',
                    'description': 'Discount created through chat',
                    'discount_type': 'percentage',
                    'discount_value': 10.0,
                    'start_date': datetime.utcnow(),
                    'end_date': datetime.utcnow().replace(day=datetime.utcnow().day + 30),
                    'user_segment': 'all'
                }
                
                result = self.discount_service.create_discount(discount_data)
                if result['success']:
                    action = 'created'
            
            return {"action": action}
            
        except Exception as e:
            logger.error(f"Error handling discount management: {e}")
            return {"action": None}
    
    async def _handle_promotion_inquiry(self, entities: Dict, user_id: int = None) -> Dict[str, Any]:
        """Handle promotion program inquiries"""
        try:
            promotions = []
            
            if 'promotion_program' in entities:
                program = entities['promotion_program']
                discounts = self.discount_service.get_discounts_for_promotion_program(program)
                promotions = [{'name': program, 'discounts': discounts}]
            else:
                # Get current promotion programs
                all_discounts = self.discount_service.get_active_discounts()
                programs = set()
                
                for discount in all_discounts:
                    if discount.promotion_program:
                        programs.add(discount.promotion_program)
                
                for program in programs:
                    program_discounts = self.discount_service.get_discounts_for_promotion_program(program)
                    promotions.append({
                        'name': program,
                        'description': f'{program} promotional offers',
                        'discounts': program_discounts[:3]  # Top 3 discounts
                    })
            
            return {"promotions": promotions}
            
        except Exception as e:
            logger.error(f"Error handling promotion inquiry: {e}")
            return {"promotions": []}
    
    async def _handle_complaint(self, entities: Dict, user_id: int = None) -> Dict[str, Any]:
        """Handle customer complaints"""
        try:
            # In a real system, this would create a support ticket
            complaint_data = {
                'user_id': user_id,
                'entities': entities,
                'timestamp': datetime.utcnow(),
                'status': 'pending'
            }
            
            return {"complaint_logged": True, "ticket_id": str(uuid.uuid4())[:8]}
            
        except Exception as e:
            logger.error(f"Error handling complaint: {e}")
            return {"complaint_logged": False}
    
    # Conversation Management
    
    def _get_or_create_conversation(self, session_id: str, user_id: int = None) -> Conversation:
        """Get existing conversation or create new one"""
        conversation = self.db.query(Conversation).filter(
            Conversation.session_id == session_id
        ).first()
        
        if not conversation:
            conversation = Conversation(
                session_id=session_id,
                user_id=user_id,
                status='active',
                context={},
                metadata={}
            )
            self.db.add(conversation)
            self.db.commit()
        
        return conversation
    
    def _get_conversation_context(self, session_id: str) -> Dict[str, Any]:
        """Get conversation context from cache"""
        context = ConversationCache.get_conversation_context(session_id)
        return context or {}
    
    def _update_conversation_context(self, session_id: str, updates: Dict[str, Any]):
        """Update conversation context"""
        context = self._get_conversation_context(session_id)
        context.update(updates)
        ConversationCache.set_conversation_context(session_id, context)
    
    def _save_message(self, conversation_id: int, sender_type: str, content: str, 
                     intent: str = None, entities: Dict = None):
        """Save message to database"""
        try:
            message = Message(
                conversation_id=conversation_id,
                sender_type=sender_type,
                content=content,
                intent=intent,
                entities=entities or {},
                response_metadata={}
            )
            self.db.add(message)
            self.db.commit()
        except Exception as e:
            logger.error(f"Error saving message: {e}")
            self.db.rollback()
    
    # Training and Analytics
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history"""
        try:
            conversation = self.db.query(Conversation).filter(
                Conversation.session_id == session_id
            ).first()
            
            if conversation:
                messages = self.db.query(Message).filter(
                    Message.conversation_id == conversation.id
                ).order_by(Message.timestamp).all()
                
                return [message.to_dict() for message in messages]
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    def end_conversation(self, session_id: str):
        """End conversation and cleanup"""
        try:
            conversation = self.db.query(Conversation).filter(
                Conversation.session_id == session_id
            ).first()
            
            if conversation:
                conversation.status = 'completed'
                conversation.ended_at = datetime.utcnow()
                self.db.commit()
            
            # Clear cache
            ConversationCache.clear_conversation(session_id)
            
        except Exception as e:
            logger.error(f"Error ending conversation: {e}")
            self.db.rollback() 