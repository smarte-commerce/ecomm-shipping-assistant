import re
import pickle
import numpy as np
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import os
import logging
from app.utils.config import get_ai_config

logger = logging.getLogger(__name__)

class IntentClassifier:
    """Intent classification for chat messages"""
    
    def __init__(self):
        self.model = None
        self.ai_config = get_ai_config()
        self.intents = self.ai_config.INTENTS
        self.model_path = "models/intent_classifier.pkl"
        self.load_model()
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def load_model(self):
        """Load pre-trained model or create rule-based fallback"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info("Intent classifier model loaded successfully")
            else:
                self.model = self._create_rule_based_classifier()
                logger.info("Using rule-based intent classifier")
        except Exception as e:
            logger.error(f"Error loading intent model: {e}")
            self.model = self._create_rule_based_classifier()
    
    def _create_rule_based_classifier(self) -> Dict:
        """Create a rule-based classifier as fallback"""
        rules = {
            "find_products": [
                "show", "find", "search", "looking for", "need", "want", "buy",
                "purchase", "get", "shopping for", "browse"
            ],
            "find_discounts": [
                "discount", "sale", "offer", "deal", "coupon", "promo", "promotion",
                "code", "voucher", "savings", "cheap", "cheaper", "save money"
            ],
            "product_recommendation": [
                "recommend", "suggest", "best", "good", "advice", "opinion",
                "what should", "which one", "help me choose", "what do you think"
            ],
            "price_inquiry": [
                "price", "cost", "how much", "expensive", "cheap", "afford",
                "budget", "pricing", "value", "worth"
            ],
            "order_status": [
                "order", "delivery", "shipping", "track", "status", "when will",
                "where is", "delivered", "shipped", "arrival"
            ],
            "order_management": [
                "cancel order", "change order", "modify", "update order",
                "return", "refund", "exchange"
            ],
            "discount_management": [
                "manage discount", "create discount", "discount settings",
                "promotion management", "coupon management"
            ],
            "promotion_inquiry": [
                "promotion", "campaign", "program", "special offer",
                "seasonal sale", "black friday", "christmas sale"
            ],
            "complaint": [
                "problem", "issue", "wrong", "bad", "terrible", "complaint",
                "not working", "broken", "damaged", "unsatisfied"
            ],
            "greeting": [
                "hi", "hello", "hey", "good morning", "good afternoon",
                "good evening", "start", "begin"
            ],
            "goodbye": [
                "bye", "goodbye", "see you", "thanks", "thank you",
                "that's all", "done", "finish"
            ]
        }
        return rules
    
    def classify(self, text: str) -> Tuple[str, float]:
        """Classify intent of the input text"""
        text = self.preprocess_text(text)
        
        if isinstance(self.model, dict):  # Rule-based classifier
            scores = {}
            for intent, keywords in self.model.items():
                score = 0
                for keyword in keywords:
                    if keyword in text:
                        score += 1
                scores[intent] = score
            
            if not scores or max(scores.values()) == 0:
                return "unknown", 0.0
            
            best_intent = max(scores, key=scores.get)
            confidence = scores[best_intent] / len(self.model[best_intent])
            return best_intent, min(confidence, 1.0)
        
        else:  # ML-based classifier
            try:
                prediction = self.model.predict([text])[0]
                probabilities = self.model.predict_proba([text])[0]
                confidence = max(probabilities)
                return prediction, confidence
            except Exception as e:
                logger.error(f"Error in ML classification: {e}")
                return "unknown", 0.0
    
    def train(self, training_data: List[Dict]) -> bool:
        """Train the intent classifier"""
        try:
            texts = [item['text'] for item in training_data]
            labels = [item['intent'] for item in training_data]
            
            # Preprocess texts
            texts = [self.preprocess_text(text) for text in texts]
            
            # Create pipeline
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=1000, 
                    stop_words='english',
                    ngram_range=(1, 2)
                )),
                ('classifier', SVC(
                    kernel='linear', 
                    probability=True,
                    random_state=42
                ))
            ])
            
            # Train model
            pipeline.fit(texts, labels)
            
            # Save model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(pipeline, f)
            
            self.model = pipeline
            logger.info("Intent classifier trained and saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training intent classifier: {e}")
            return False
    
    def create_training_data(self) -> List[Dict]:
        """Create sample training data"""
        training_data = [
            # Find Products
            {"text": "I'm looking for a laptop", "intent": "find_products"},
            {"text": "Show me smartphones", "intent": "find_products"},
            {"text": "I need a new dress", "intent": "find_products"},
            {"text": "Search for headphones", "intent": "find_products"},
            {"text": "I want to buy shoes", "intent": "find_products"},
            {"text": "Looking for gaming chair", "intent": "find_products"},
            
            # Find Discounts
            {"text": "Are there any sales on electronics?", "intent": "find_discounts"},
            {"text": "What discounts are available?", "intent": "find_discounts"},
            {"text": "Show me current deals", "intent": "find_discounts"},
            {"text": "Any coupons for shoes?", "intent": "find_discounts"},
            {"text": "I need a promo code", "intent": "find_discounts"},
            {"text": "What's on sale today?", "intent": "find_discounts"},
            
            # Product Recommendation
            {"text": "Recommend a good phone", "intent": "product_recommendation"},
            {"text": "What's the best laptop for gaming?", "intent": "product_recommendation"},
            {"text": "Suggest something for my mom", "intent": "product_recommendation"},
            {"text": "Help me choose a camera", "intent": "product_recommendation"},
            {"text": "What do you recommend?", "intent": "product_recommendation"},
            
            # Price Inquiry
            {"text": "How much does this cost?", "intent": "price_inquiry"},
            {"text": "What's the price of iPhone?", "intent": "price_inquiry"},
            {"text": "Is this expensive?", "intent": "price_inquiry"},
            {"text": "Can I afford this?", "intent": "price_inquiry"},
            
            # Order Status
            {"text": "Where is my order?", "intent": "order_status"},
            {"text": "Track my delivery", "intent": "order_status"},
            {"text": "When will my package arrive?", "intent": "order_status"},
            {"text": "Order status check", "intent": "order_status"},
            
            # Order Management
            {"text": "Cancel my order", "intent": "order_management"},
            {"text": "I want to return this", "intent": "order_management"},
            {"text": "Change my order", "intent": "order_management"},
            {"text": "Refund request", "intent": "order_management"},
            
            # Discount Management
            {"text": "Create a new discount", "intent": "discount_management"},
            {"text": "Manage promotions", "intent": "discount_management"},
            {"text": "Set up coupon codes", "intent": "discount_management"},
            
            # Promotion Inquiry
            {"text": "What promotions are running?", "intent": "promotion_inquiry"},
            {"text": "Tell me about Black Friday deals", "intent": "promotion_inquiry"},
            {"text": "Any seasonal sales?", "intent": "promotion_inquiry"},
            
            # Complaints
            {"text": "I have a problem with my order", "intent": "complaint"},
            {"text": "This product is broken", "intent": "complaint"},
            {"text": "I'm not satisfied", "intent": "complaint"},
            
            # Greetings
            {"text": "Hello", "intent": "greeting"},
            {"text": "Hi there", "intent": "greeting"},
            {"text": "Good morning", "intent": "greeting"},
            {"text": "Hey", "intent": "greeting"},
            
            # Goodbye
            {"text": "Thank you", "intent": "goodbye"},
            {"text": "Bye", "intent": "goodbye"},
            {"text": "See you later", "intent": "goodbye"},
            {"text": "That's all", "intent": "goodbye"}
        ]
        
        return training_data 