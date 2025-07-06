# AI Shopping Assistant - Complete Tutorial & Workflow

## Table of Contents
1. [Prerequisites & Setup](#prerequisites--setup)
2. [Project Structure](#project-structure)
3. [Phase 1: Basic Foundation](#phase-1-basic-foundation)
4. [Phase 2: AI Integration](#phase-2-ai-integration)
5. [Phase 3: Advanced Features](#phase-3-advanced-features)
6. [Deployment & Monitoring](#deployment--monitoring)
7. [Code Examples](#code-examples)

---

## Prerequisites & Setup

### Required Skills
- **Basic**: Python programming, REST APIs, databases
- **Intermediate**: Machine learning concepts, natural language processing
- **Advanced**: Deep learning, recommendation systems

### Development Environment Setup

#### 1. Install Python & Dependencies
```bash
# Create virtual environment
python -m venv shopping_assistant_env
source shopping_assistant_env/bin/activate  # On Windows: shopping_assistant_env\Scripts\activate

# Install core packages
pip install fastapi uvicorn sqlalchemy pandas numpy scikit-learn
pip install transformers torch datasets
pip install redis celery python-multipart
pip install openai anthropic  # For external AI APIs
```

#### 2. Database Setup
```bash
# Install PostgreSQL
# On Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# On macOS
brew install postgresql

# Create database
createdb shopping_assistant_db
```

#### 3. Redis Setup
```bash
# Install Redis
# On Ubuntu/Debian
sudo apt-get install redis-server

# On macOS
brew install redis

# Start Redis
redis-server
```

---

## Project Structure

```
shopping_assistant/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── models/                 # Database models
│   │   ├── __init__.py
│   │   ├── product.py
│   │   ├── user.py
│   │   ├── discount.py
│   │   └── conversation.py
│   ├── api/                    # API endpoints
│   │   ├── __init__.py
│   │   ├── chat.py
│   │   ├── products.py
│   │   ├── discounts.py
│   │   └── notifications.py
│   ├── services/               # Business logic
│   │   ├── __init__.py
│   │   ├── chat_service.py
│   │   ├── recommendation_service.py
│   │   ├── discount_service.py
│   │   └── notification_service.py
│   ├── ai/                     # AI models and processors
│   │   ├── __init__.py
│   │   ├── intent_classifier.py
│   │   ├── entity_extractor.py
│   │   ├── recommendation_engine.py
│   │   └── response_generator.py
│   ├── utils/                  # Utility functions
│   │   ├── __init__.py
│   │   ├── database.py
│   │   ├── cache.py
│   │   └── config.py
│   └── data/                   # Data processing
│       ├── __init__.py
│       ├── collectors.py
│       ├── preprocessors.py
│       └── trainers.py
├── frontend/                   # Web interface
│   ├── index.html
│   ├── chat.js
│   └── styles.css
├── data/                       # Raw data files
│   ├── products.csv
│   ├── conversations.json
│   └── discounts.csv
├── models/                     # Trained models
│   ├── intent_classifier.pkl
│   ├── recommendation_model.pkl
│   └── entity_extractor.pkl
├── tests/                      # Test files
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Phase 1: Basic Foundation

### Step 1: Database Models

#### 1.1 Create Product Model
```python
# app/models/product.py
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Product(Base):
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    category = Column(String(100))
    brand = Column(String(100))
    price = Column(Float, nullable=False)
    original_price = Column(Float)
    stock_quantity = Column(Integer, default=0)
    image_url = Column(String(500))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

#### 1.2 Create User Model
```python
# app/models/user.py
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True)
    phone = Column(String(20))
    preferences = Column(JSON)  # Store user preferences as JSON
    purchase_history = Column(JSON)  # Store purchase history
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
```

#### 1.3 Create Discount Model
```python
# app/models/discount.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Discount(Base):
    __tablename__ = "discounts"
    
    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(50), unique=True)
    description = Column(Text)
    discount_type = Column(String(20))  # percentage, fixed_amount, bogo
    discount_value = Column(Float)
    min_purchase_amount = Column(Float, default=0)
    max_discount_amount = Column(Float)
    category = Column(String(100))  # specific category or 'all'
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    usage_limit = Column(Integer)
    usage_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
```

### Step 2: Basic API Setup

#### 2.1 FastAPI Application
```python
# app/main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from app.utils.database import get_db
from app.api import chat, products, discounts, notifications

app = FastAPI(title="AI Shopping Assistant", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(products.router, prefix="/api/products", tags=["products"])
app.include_router(discounts.router, prefix="/api/discounts", tags=["discounts"])
app.include_router(notifications.router, prefix="/api/notifications", tags=["notifications"])

@app.get("/")
async def root():
    return {"message": "AI Shopping Assistant API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

#### 2.2 Database Configuration
```python
# app/utils/database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/shopping_assistant_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### Step 3: Basic Chat Interface

#### 3.1 Chat API Endpoint
```python
# app/api/chat.py
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.utils.database import get_db
from app.services.chat_service import ChatService
from pydantic import BaseModel

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    user_id: int = None
    session_id: str = None

class ChatResponse(BaseModel):
    response: str
    intent: str = None
    entities: dict = None
    suggestions: list = None

@router.post("/message", response_model=ChatResponse)
async def chat_message(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        chat_service = ChatService(db)
        response = await chat_service.process_message(
            message=request.message,
            user_id=request.user_id,
            session_id=request.session_id
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

#### 3.2 Chat Service Implementation
```python
# app/services/chat_service.py
from sqlalchemy.orm import Session
from app.ai.intent_classifier import IntentClassifier
from app.ai.entity_extractor import EntityExtractor
from app.ai.response_generator import ResponseGenerator
from app.services.product_service import ProductService
from app.services.discount_service import DiscountService
import json

class ChatService:
    def __init__(self, db: Session):
        self.db = db
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.response_generator = ResponseGenerator()
        self.product_service = ProductService(db)
        self.discount_service = DiscountService(db)
    
    async def process_message(self, message: str, user_id: int = None, session_id: str = None):
        # Step 1: Classify intent
        intent = self.intent_classifier.classify(message)
        
        # Step 2: Extract entities
        entities = self.entity_extractor.extract(message)
        
        # Step 3: Process based on intent
        if intent == "find_products":
            products = await self.product_service.search_products(entities)
            response = self.response_generator.generate_product_response(products)
        
        elif intent == "find_discounts":
            discounts = await self.discount_service.get_relevant_discounts(entities, user_id)
            response = self.response_generator.generate_discount_response(discounts)
        
        elif intent == "product_recommendation":
            recommendations = await self.product_service.get_recommendations(user_id, entities)
            response = self.response_generator.generate_recommendation_response(recommendations)
        
        else:
            response = self.response_generator.generate_fallback_response()
        
        return {
            "response": response,
            "intent": intent,
            "entities": entities,
            "suggestions": self._get_suggestions(intent, entities)
        }
    
    def _get_suggestions(self, intent: str, entities: dict):
        # Generate contextual suggestions based on intent and entities
        suggestions = []
        
        if intent == "find_products":
            suggestions = [
                "Show me similar products",
                "What's on sale in this category?",
                "Compare prices"
            ]
        elif intent == "find_discounts":
            suggestions = [
                "Show me more deals",
                "What's the best discount available?",
                "Notify me about future sales"
            ]
        
        return suggestions
```

---

## Phase 2: AI Integration

### Step 1: Intent Classification

#### 1.1 Intent Classifier Implementation
```python
# app/ai/intent_classifier.py
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import re

class IntentClassifier:
    def __init__(self):
        self.model = None
        self.intents = [
            "find_products",
            "find_discounts",
            "product_recommendation",
            "price_inquiry",
            "order_status",
            "complaint",
            "greeting",
            "goodbye"
        ]
        self.load_model()
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def load_model(self):
        """Load pre-trained model or create a simple rule-based classifier"""
        try:
            with open('models/intent_classifier.pkl', 'rb') as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            # Create simple rule-based classifier for initial version
            self.model = self._create_rule_based_classifier()
    
    def _create_rule_based_classifier(self):
        """Create a simple rule-based classifier"""
        rules = {
            "find_products": ["show", "find", "search", "looking for", "need", "want"],
            "find_discounts": ["discount", "sale", "offer", "deal", "coupon", "promo"],
            "product_recommendation": ["recommend", "suggest", "best", "good", "advice"],
            "price_inquiry": ["price", "cost", "how much", "expensive", "cheap"],
            "order_status": ["order", "delivery", "shipping", "track", "status"],
            "complaint": ["problem", "issue", "wrong", "bad", "terrible", "complaint"],
            "greeting": ["hi", "hello", "hey", "good morning", "good afternoon"],
            "goodbye": ["bye", "goodbye", "see you", "thanks", "thank you"]
        }
        return rules
    
    def classify(self, text: str) -> str:
        """Classify intent of the input text"""
        text = self.preprocess_text(text)
        
        if isinstance(self.model, dict):  # Rule-based classifier
            scores = {}
            for intent, keywords in self.model.items():
                score = sum(1 for keyword in keywords if keyword in text)
                scores[intent] = score
            
            best_intent = max(scores, key=scores.get)
            return best_intent if scores[best_intent] > 0 else "unknown"
        
        else:  # ML-based classifier
            prediction = self.model.predict([text])[0]
            return prediction
    
    def train(self, training_data):
        """Train the intent classifier"""
        texts = [item['text'] for item in training_data]
        labels = [item['intent'] for item in training_data]
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
            ('classifier', SVC(kernel='linear', probability=True))
        ])
        
        # Train model
        pipeline.fit(texts, labels)
        
        # Save model
        with open('models/intent_classifier.pkl', 'wb') as f:
            pickle.dump(pipeline, f)
        
        self.model = pipeline
```

#### 1.2 Training Data Preparation
```python
# app/data/trainers.py
import json
import pandas as pd
from app.ai.intent_classifier import IntentClassifier

class IntentTrainer:
    def __init__(self):
        self.training_data = []
    
    def load_training_data(self, file_path: str):
        """Load training data from JSON file"""
        with open(file_path, 'r') as f:
            self.training_data = json.load(f)
    
    def create_sample_data(self):
        """Create sample training data"""
        sample_data = [
            # Find Products
            {"text": "I'm looking for a laptop", "intent": "find_products"},
            {"text": "Show me smartphones", "intent": "find_products"},
            {"text": "I need a new dress", "intent": "find_products"},
            {"text": "Search for headphones", "intent": "find_products"},
            
            # Find Discounts
            {"text": "Are there any sales on electronics?", "intent": "find_discounts"},
            {"text": "What discounts are available?", "intent": "find_discounts"},
            {"text": "Show me current deals", "intent": "find_discounts"},
            {"text": "Any coupons for shoes?", "intent": "find_discounts"},
            
            # Product Recommendation
            {"text": "Recommend a good phone", "intent": "product_recommendation"},
            {"text": "What's the best laptop for gaming?", "intent": "product_recommendation"},
            {"text": "Suggest something for my mom", "intent": "product_recommendation"},
            {"text": "I need advice on buying a car", "intent": "product_recommendation"},
            
            # Price Inquiry
            {"text": "How much does this cost?", "intent": "price_inquiry"},
            {"text": "What's the price of iPhone?", "intent": "price_inquiry"},
            {"text": "Is this expensive?", "intent": "price_inquiry"},
            
            # Greetings
            {"text": "Hello", "intent": "greeting"},
            {"text": "Hi there", "intent": "greeting"},
            {"text": "Good morning", "intent": "greeting"},
            
            # Goodbye
            {"text": "Thank you", "intent": "goodbye"},
            {"text": "Bye", "intent": "goodbye"},
            {"text": "See you later", "intent": "goodbye"}
        ]
        
        return sample_data
    
    def train_intent_classifier(self):
        """Train the intent classifier"""
        if not self.training_data:
            self.training_data = self.create_sample_data()
        
        classifier = IntentClassifier()
        classifier.train(self.training_data)
        
        print("Intent classifier trained successfully!")
```

### Step 2: Entity Extraction

#### 2.1 Entity Extractor Implementation
```python
# app/ai/entity_extractor.py
import re
import spacy
from typing import Dict, List

class EntityExtractor:
    def __init__(self):
        # Load spaCy model (install with: python -m spacy download en_core_web_sm)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback to simple regex-based extraction
            self.nlp = None
        
        # Define entity patterns
        self.patterns = {
            'price_range': r'\$?\d+(?:\.\d{2})?(?:\s*(?:to|-)\s*\$?\d+(?:\.\d{2})?)?',
            'brand': ['apple', 'samsung', 'nike', 'adidas', 'sony', 'lg', 'dell', 'hp'],
            'category': ['laptop', 'phone', 'smartphone', 'shoes', 'dress', 'headphones', 
                        'electronics', 'clothing', 'accessories', 'home', 'kitchen'],
            'color': ['red', 'blue', 'black', 'white', 'green', 'yellow', 'pink', 'purple'],
            'size': ['small', 'medium', 'large', 'xl', 'xxl', 's', 'm', 'l']
        }
    
    def extract(self, text: str) -> Dict:
        """Extract entities from text"""
        entities = {}
        text_lower = text.lower()
        
        # Extract price range
        price_matches = re.findall(self.patterns['price_range'], text)
        if price_matches:
            entities['price_range'] = price_matches[0]
        
        # Extract brand
        brands = [brand for brand in self.patterns['brand'] if brand in text_lower]
        if brands:
            entities['brand'] = brands[0]
        
        # Extract category
        categories = [cat for cat in self.patterns['category'] if cat in text_lower]
        if categories:
            entities['category'] = categories[0]
        
        # Extract color
        colors = [color for color in self.patterns['color'] if color in text_lower]
        if colors:
            entities['color'] = colors[0]
        
        # Extract size
        sizes = [size for size in self.patterns['size'] if size in text_lower]
        if sizes:
            entities['size'] = sizes[0]
        
        # Use spaCy for additional entity extraction
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "MONEY":
                    entities['price'] = ent.text
                elif ent.label_ == "ORG":
                    entities['brand'] = ent.text.lower()
                elif ent.label_ == "PRODUCT":
                    entities['product'] = ent.text.lower()
        
        return entities
```

### Step 3: Basic Recommendation Engine

#### 3.1 Simple Recommendation Engine
```python
# app/ai/recommendation_engine.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy.orm import Session
from app.models.product import Product
from app.models.user import User

class RecommendationEngine:
    def __init__(self, db: Session):
        self.db = db
        self.products_df = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.load_product_data()
    
    def load_product_data(self):
        """Load product data from database"""
        products = self.db.query(Product).filter(Product.is_active == True).all()
        
        self.products_df = pd.DataFrame([{
            'id': p.id,
            'name': p.name,
            'description': p.description,
            'category': p.category,
            'brand': p.brand,
            'price': p.price,
            'features': f"{p.name} {p.description} {p.category} {p.brand}"
        } for p in products])
        
        if len(self.products_df) > 0:
            self.build_content_matrix()
    
    def build_content_matrix(self):
        """Build TF-IDF matrix for content-based recommendations"""
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.products_df['features'])
    
    def get_similar_products(self, product_id: int, n_recommendations: int = 5):
        """Get similar products based on content similarity"""
        if self.tfidf_matrix is None:
            return []
        
        # Find product index
        product_idx = self.products_df[self.products_df['id'] == product_id].index
        if len(product_idx) == 0:
            return []
        
        product_idx = product_idx[0]
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(self.tfidf_matrix[product_idx], self.tfidf_matrix)
        sim_scores = list(enumerate(cosine_sim[0]))
        
        # Sort by similarity
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar products (excluding the product itself)
        similar_indices = [i[0] for i in sim_scores[1:n_recommendations+1]]
        
        return self.products_df.iloc[similar_indices]['id'].tolist()
    
    def get_recommendations_by_category(self, category: str, n_recommendations: int = 5):
        """Get top products in a specific category"""
        if self.products_df is None:
            return []
        
        category_products = self.products_df[
            self.products_df['category'].str.contains(category, case=False, na=False)
        ]
        
        if len(category_products) == 0:
            return []
        
        # Sort by price (you can change this to rating when available)
        top_products = category_products.nlargest(n_recommendations, 'price')
        
        return top_products['id'].tolist()
    
    def get_recommendations_by_price_range(self, min_price: float, max_price: float, 
                                          n_recommendations: int = 5):
        """Get products within a price range"""
        if self.products_df is None:
            return []
        
        price_filtered = self.products_df[
            (self.products_df['price'] >= min_price) & 
            (self.products_df['price'] <= max_price)
        ]
        
        if len(price_filtered) == 0:
            return []
        
        # Sort by price
        recommended = price_filtered.nsmallest(n_recommendations, 'price')
        
        return recommended['id'].tolist()
    
    def get_personalized_recommendations(self, user_id: int, n_recommendations: int = 5):
        """Get personalized recommendations for a user"""
        # Get user's purchase history
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user or not user.purchase_history:
            return self.get_popular_products(n_recommendations)
        
        # Simple approach: recommend similar products to previously purchased items
        purchased_products = user.purchase_history.get('products', [])
        all_recommendations = []
        
        for product_id in purchased_products:
            similar = self.get_similar_products(product_id, 3)
            all_recommendations.extend(similar)
        
        # Remove duplicates and limit results
        unique_recommendations = list(set(all_recommendations))
        return unique_recommendations[:n_recommendations]
    
    def get_popular_products(self, n_recommendations: int = 5):
        """Get popular products (fallback recommendation)"""
        if self.products_df is None:
            return []
        
        # For now, return products sorted by price (descending)
        # In a real system, you'd sort by sales, ratings, etc.
        popular = self.products_df.nlargest(n_recommendations, 'price')
        return popular['id'].tolist()
```

### Step 4: Discount Management

#### 4.1 Discount Service Implementation
```python
# app/services/discount_service.py
from sqlalchemy.orm import Session
from app.models.discount import Discount
from app.models.product import Product
from datetime import datetime
from typing import List, Dict

class DiscountService:
    def __init__(self, db: Session):
        self.db = db
    
    def get_active_discounts(self) -> List[Discount]:
        """Get all active discounts"""
        current_time = datetime.utcnow()
        return self.db.query(Discount).filter(
            Discount.is_active == True,
            Discount.start_date <= current_time,
            Discount.end_date >= current_time
        ).all()
    
    def get_relevant_discounts(self, entities: Dict, user_id: int = None) -> List[Discount]:
        """Get discounts relevant to user query"""
        discounts = self.get_active_discounts()
        relevant_discounts = []
        
        for discount in discounts:
            # Check if discount applies to requested category
            if 'category' in entities:
                if discount.category == 'all' or discount.category == entities['category']:
                    relevant_discounts.append(discount)
            else:
                relevant_discounts.append(discount)
        
        return relevant_discounts[:5]  # Return top 5 relevant discounts
    
    def calculate_discount_amount(self, discount: Discount, purchase_amount: float) -> float:
        """Calculate discount amount for a purchase"""
        if purchase_amount < discount.min_purchase_amount:
            return 0
        
        if discount.discount_type == 'percentage':
            discount_amount = purchase_amount * (discount.discount_value / 100)
        elif discount.discount_type == 'fixed_amount':
            discount_amount = discount.discount_value
        else:
            return 0
        
        # Apply maximum discount limit
        if discount.max_discount_amount:
            discount_amount = min(discount_amount, discount.max_discount_amount)
        
        return discount_amount
    
    def apply_discount(self, discount_code: str, purchase_amount: float) -> Dict:
        """Apply discount code to a purchase"""
        discount = self.db.query(Discount).filter(
            Discount.code == discount_code,
            Discount.is_active == True
        ).first()
        
        if not discount:
            return {"success": False, "error": "Invalid discount code"}
        
        # Check if discount is still valid
        current_time = datetime.utcnow()
        if current_time < discount.start_date or current_time > discount.end_date:
            return {"success": False, "error": "Discount code expired"}
        
        # Check usage limit
        if discount.usage_limit and discount.usage_count >= discount.usage_limit:
            return {"success": False, "error": "Discount code usage limit exceeded"}
        
        # Calculate discount amount
        discount_amount = self.calculate_discount_amount(discount, purchase_amount)
        
        if discount_amount == 0:
            return {"success": False, "error": "Purchase amount doesn't meet minimum requirement"}
        
        # Update usage count
        discount.usage_count += 1
        self.db.commit()
        
        return {
            "success": True,
            "discount_amount": discount_amount,
            "final_amount": purchase_amount - discount_amount,
            "discount_description": discount.description
        }
    
    def create_automatic_discount(self, product_id: int, discount_percentage: float, 
                                 reason: str = "inventory_clearance"):
        """Create automatic discount for slow-moving inventory"""
        product = self.db.query(Product).filter(Product.id == product_id).first()
        if not product:
            return None
        
        # Create discount
        discount = Discount(
            code=f"AUTO_{product.id}_{datetime.now().strftime('%Y%m%d')}",
            description=f"Automatic {discount_percentage}% discount on {product.name}",
            discount_type="percentage",
            discount_value=discount_percentage,
            category=product.category,
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow().replace(day=datetime.utcnow().day + 30),  # 30 days
            is_active=True
        )
        
        