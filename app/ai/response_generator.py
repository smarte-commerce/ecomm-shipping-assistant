import random
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from app.utils.config import get_ai_config

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Generate natural language responses for the chatbot"""
    
    def __init__(self):
        self.ai_config = get_ai_config()
        self.templates = self.ai_config.RESPONSE_TEMPLATES
    
    def generate_response(self, intent: str, entities: Dict[str, Any], 
                         context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate response based on intent, entities, and context"""
        
        if intent == "greeting":
            return self._generate_greeting_response()
        elif intent == "goodbye":
            return self._generate_goodbye_response()
        elif intent == "find_products":
            return self._generate_product_search_response(entities, context)
        elif intent == "find_discounts":
            return self._generate_discount_search_response(entities, context)
        elif intent == "product_recommendation":
            return self._generate_recommendation_response(entities, context)
        elif intent == "price_inquiry":
            return self._generate_price_inquiry_response(entities, context)
        elif intent == "order_status":
            return self._generate_order_status_response(entities, context)
        elif intent == "order_management":
            return self._generate_order_management_response(entities, context)
        elif intent == "discount_management":
            return self._generate_discount_management_response(entities, context)
        elif intent == "promotion_inquiry":
            return self._generate_promotion_inquiry_response(entities, context)
        elif intent == "complaint":
            return self._generate_complaint_response(entities, context)
        else:
            return self._generate_fallback_response()
    
    def _generate_greeting_response(self) -> Dict[str, Any]:
        """Generate greeting response"""
        greetings = self.templates.get('greeting', [
            "Hello! I'm your AI shopping assistant. How can I help you today?",
            "Hi there! Looking for great deals or need product recommendations?",
            "Welcome! I'm here to help you find the best products and discounts."
        ])
        
        response = random.choice(greetings)
        
        return {
            "response": response,
            "suggestions": [
                "Show me products",
                "Find discounts",
                "Recommend something for me",
                "Check my orders"
            ],
            "response_type": "greeting"
        }
    
    def _generate_goodbye_response(self) -> Dict[str, Any]:
        """Generate goodbye response"""
        goodbyes = [
            "Thank you for shopping with us! Have a great day!",
            "Goodbye! Feel free to come back anytime for more deals.",
            "Thanks for using our shopping assistant. Happy shopping!",
            "Take care! Don't forget to check back for new discounts."
        ]
        
        return {
            "response": random.choice(goodbyes),
            "suggestions": [],
            "response_type": "goodbye"
        }
    
    def _generate_product_search_response(self, entities: Dict, context: Dict = None) -> Dict[str, Any]:
        """Generate response for product search"""
        products = context.get('products', []) if context else []
        
        if not products:
            response = random.choice(self.templates.get('no_results', [
                "I couldn't find any products matching your criteria. Would you like to try different terms?"
            ]))
            suggestions = [
                "Try different keywords",
                "Browse categories",
                "Show me popular products"
            ]
        else:
            count = len(products)
            response_template = random.choice(self.templates.get('product_found', [
                "I found {count} products matching your criteria:"
            ]))
            response = response_template.format(count=count)
            
            # Add product details to response
            product_list = []
            for product in products[:5]:  # Show top 5
                product_info = f"• {product['name']} - ${product['price']:.2f}"
                if 'rating' in product and product['rating'] > 0:
                    product_info += f" (Rating: {product['rating']}/5)"
                product_list.append(product_info)
            
            if product_list:
                response += "\n\n" + "\n".join(product_list)
            
            suggestions = [
                "Show more details",
                "Find similar products",
                "Check for discounts",
                "Add to cart"
            ]
        
        return {
            "response": response,
            "suggestions": suggestions,
            "response_type": "product_search",
            "data": {"products": products}
        }
    
    def _generate_discount_search_response(self, entities: Dict, context: Dict = None) -> Dict[str, Any]:
        """Generate response for discount search"""
        discounts = context.get('discounts', []) if context else []
        
        if not discounts:
            response = "I couldn't find any active discounts matching your criteria. Let me check for general offers."
            suggestions = [
                "Show all discounts",
                "Check promotion programs",
                "Notify me about future deals"
            ]
        else:
            count = len(discounts)
            response_template = random.choice(self.templates.get('discount_found', [
                "Great news! I found {count} active discounts for you:"
            ]))
            response = response_template.format(count=count)
            
            # Add discount details
            discount_list = []
            for discount in discounts[:5]:  # Show top 5
                if discount['discount_type'] == 'percentage':
                    discount_info = f"• {discount['name']}: {discount['discount_value']}% off"
                elif discount['discount_type'] == 'fixed_amount':
                    discount_info = f"• {discount['name']}: ${discount['discount_value']} off"
                else:
                    discount_info = f"• {discount['name']}: {discount['description']}"
                
                if discount.get('code'):
                    discount_info += f" (Code: {discount['code']})"
                
                discount_list.append(discount_info)
            
            if discount_list:
                response += "\n\n" + "\n".join(discount_list)
            
            suggestions = [
                "Apply discount",
                "Show more discounts",
                "Find products for this discount",
                "Save for later"
            ]
        
        return {
            "response": response,
            "suggestions": suggestions,
            "response_type": "discount_search",
            "data": {"discounts": discounts}
        }
    
    def _generate_recommendation_response(self, entities: Dict, context: Dict = None) -> Dict[str, Any]:
        """Generate response for product recommendations"""
        recommendations = context.get('recommendations', []) if context else []
        
        if not recommendations:
            response = "I need a bit more information to make personalized recommendations. What are you interested in?"
            suggestions = [
                "Electronics",
                "Clothing",
                "Home & Garden",
                "Tell me your preferences"
            ]
        else:
            response = f"Based on your preferences, here are my top recommendations:"
            
            # Add recommendation details
            rec_list = []
            for rec in recommendations[:3]:  # Show top 3 recommendations
                rec_info = f"• {rec['name']} - ${rec['price']:.2f}"
                if 'rating' in rec and rec['rating'] > 0:
                    rec_info += f" (Rating: {rec['rating']}/5)"
                if 'similarity_score' in rec:
                    rec_info += f" - {rec['similarity_score']:.1%} match"
                rec_list.append(rec_info)
            
            if rec_list:
                response += "\n\n" + "\n".join(rec_list)
            
            suggestions = [
                "Show more recommendations",
                "Tell me why you recommended this",
                "Find similar products",
                "Check for discounts"
            ]
        
        return {
            "response": response,
            "suggestions": suggestions,
            "response_type": "recommendation",
            "data": {"recommendations": recommendations}
        }
    
    def _generate_price_inquiry_response(self, entities: Dict, context: Dict = None) -> Dict[str, Any]:
        """Generate response for price inquiries"""
        products = context.get('products', []) if context else []
        
        if 'price_range' in entities:
            price_range = entities['price_range']
            response = f"I can help you find products between ${price_range['min']:.2f} and ${price_range['max']:.2f}."
        elif products:
            product = products[0]
            response = f"The price for {product['name']} is ${product['price']:.2f}."
            if 'original_price' in product and product['original_price'] > product['price']:
                savings = product['original_price'] - product['price']
                response += f" You're saving ${savings:.2f}!"
        else:
            response = "Please specify which product you'd like to know the price for."
        
        suggestions = [
            "Compare prices",
            "Find cheaper alternatives",
            "Check for discounts",
            "Show similar products"
        ]
        
        return {
            "response": response,
            "suggestions": suggestions,
            "response_type": "price_inquiry",
            "data": {"products": products}
        }
    
    def _generate_order_status_response(self, entities: Dict, context: Dict = None) -> Dict[str, Any]:
        """Generate response for order status inquiries"""
        orders = context.get('orders', []) if context else []
        
        if 'order_number' in entities:
            order_number = entities['order_number']
            if orders:
                order = orders[0]
                response = f"Order {order_number} status: {order['status'].replace('_', ' ').title()}"
                if order['status'] == 'shipped':
                    response += f"\nExpected delivery: {order.get('expected_delivery', 'Soon')}"
            else:
                response = f"I couldn't find order {order_number}. Please check the order number."
        else:
            response = "Please provide your order number to check the status."
        
        suggestions = [
            "Track shipment",
            "Update delivery address", 
            "Cancel order",
            "Contact support"
        ]
        
        return {
            "response": response,
            "suggestions": suggestions,
            "response_type": "order_status",
            "data": {"orders": orders}
        }
    
    def _generate_order_management_response(self, entities: Dict, context: Dict = None) -> Dict[str, Any]:
        """Generate response for order management"""
        action_taken = context.get('action_taken') if context else None
        
        if action_taken == 'cancelled':
            response = "Your order has been successfully cancelled. Refund will be processed within 3-5 business days."
        elif action_taken == 'modified':
            response = "Your order has been updated successfully."
        elif action_taken == 'return_initiated':
            response = "Return request has been initiated. You'll receive a return label via email."
        else:
            response = "I can help you cancel, modify, or return your order. What would you like to do?"
        
        suggestions = [
            "Cancel order",
            "Modify order",
            "Return item",
            "Get refund status"
        ]
        
        return {
            "response": response,
            "suggestions": suggestions,
            "response_type": "order_management"
        }
    
    def _generate_discount_management_response(self, entities: Dict, context: Dict = None) -> Dict[str, Any]:
        """Generate response for discount management"""
        action = context.get('action') if context else None
        
        if action == 'created':
            response = "New discount has been created successfully!"
        elif action == 'updated':
            response = "Discount has been updated successfully!"
        elif action == 'deactivated':
            response = "Discount has been deactivated."
        else:
            response = "I can help you create, update, or manage discount codes. What would you like to do?"
        
        suggestions = [
            "Create new discount",
            "View active discounts",
            "Update discount settings",
            "Deactivate discount"
        ]
        
        return {
            "response": response,
            "suggestions": suggestions,
            "response_type": "discount_management"
        }
    
    def _generate_promotion_inquiry_response(self, entities: Dict, context: Dict = None) -> Dict[str, Any]:
        """Generate response for promotion program inquiries"""
        promotions = context.get('promotions', []) if context else []
        
        if promotions:
            response = "Here are the current promotion programs:"
            promo_list = []
            for promo in promotions[:3]:
                promo_info = f"• {promo['name']}: {promo['description']}"
                promo_list.append(promo_info)
            
            if promo_list:
                response += "\n\n" + "\n".join(promo_list)
        else:
            response = "Currently, we have several ongoing promotion programs. Let me check what's available for you."
        
        suggestions = [
            "Show Black Friday deals",
            "Christmas promotions",
            "Flash sales",
            "Seasonal discounts"
        ]
        
        return {
            "response": response,
            "suggestions": suggestions,
            "response_type": "promotion_inquiry",
            "data": {"promotions": promotions}
        }
    
    def _generate_complaint_response(self, entities: Dict, context: Dict = None) -> Dict[str, Any]:
        """Generate response for complaints"""
        response = "I'm sorry to hear about the issue you're experiencing. I'm here to help resolve this for you."
        
        if 'order_number' in entities:
            response += f" I'll look into order {entities['order_number']} right away."
        
        suggestions = [
            "Speak to a human agent",
            "File a formal complaint",
            "Request refund",
            "Track resolution status"
        ]
        
        return {
            "response": response,
            "suggestions": suggestions,
            "response_type": "complaint"
        }
    
    def _generate_fallback_response(self) -> Dict[str, Any]:
        """Generate fallback response for unknown intents"""
        fallback_responses = [
            "I'm not sure I understand. Could you please rephrase that?",
            "I didn't quite get that. Can you tell me what you're looking for?",
            "Let me help you better. What are you trying to do today?",
            "I'm here to help! Could you be more specific about what you need?"
        ]
        
        suggestions = [
            "Find products",
            "Search for discounts",
            "Check my orders",
            "Get recommendations"
        ]
        
        return {
            "response": random.choice(fallback_responses),
            "suggestions": suggestions,
            "response_type": "fallback"
        }
    
    def format_product_list(self, products: List[Dict], max_items: int = 5) -> str:
        """Format product list for display"""
        if not products:
            return "No products found."
        
        formatted_list = []
        for i, product in enumerate(products[:max_items]):
            item = f"{i+1}. {product['name']} - ${product['price']:.2f}"
            if 'category' in product:
                item += f" ({product['category']})"
            if 'rating' in product and product['rating'] > 0:
                item += f" ⭐ {product['rating']}/5"
            formatted_list.append(item)
        
        result = "\n".join(formatted_list)
        
        if len(products) > max_items:
            result += f"\n... and {len(products) - max_items} more items"
        
        return result
    
    def format_discount_list(self, discounts: List[Dict], max_items: int = 5) -> str:
        """Format discount list for display"""
        if not discounts:
            return "No active discounts found."
        
        formatted_list = []
        for i, discount in enumerate(discounts[:max_items]):
            if discount['discount_type'] == 'percentage':
                item = f"{i+1}. {discount['name']}: {discount['discount_value']}% OFF"
            elif discount['discount_type'] == 'fixed_amount':
                item = f"{i+1}. {discount['name']}: ${discount['discount_value']} OFF"
            else:
                item = f"{i+1}. {discount['name']}: {discount['description']}"
            
            if discount.get('code'):
                item += f" | Code: {discount['code']}"
            
            formatted_list.append(item)
        
        result = "\n".join(formatted_list)
        
        if len(discounts) > max_items:
            result += f"\n... and {len(discounts) - max_items} more offers"
        
        return result 