# AI Shopping Assistant

A comprehensive AI-powered shopping assistant with intelligent discount recommendations, promotion management, and natural language chat interface.

## 🌟 Features

### Core Capabilities
- **AI-Powered Chat Interface**: Natural language processing for customer interactions
- **Intelligent Product Recommendations**: Content-based and collaborative filtering
- **Advanced Discount Management**: Multi-type discounts with user segmentation
- **Promotion Programs**: Automated campaign management and analytics
- **Order Management**: Track and manage customer orders through chat
- **Real-time Notifications**: User notification system
- **Conversation Context**: Session-based chat with context preservation

### AI Components
- **Intent Classification**: ML-based intent recognition with 11+ supported intents
- **Entity Extraction**: NLP entity extraction for products, prices, brands, and more
- **Response Generation**: Natural language response generation with templates
- **Recommendation Engine**: TF-IDF based similarity matching and personalized suggestions

### Discount System
- **Multiple Discount Types**: Percentage, fixed amount, BOGO, free shipping
- **User Segmentation**: Target discounts to specific customer segments
- **Promotion Campaigns**: Black Friday, Christmas, Flash Sales, and more
- **Smart Validation**: Automatic discount application and validation
- **Analytics**: Track discount performance and usage

## 🏗️ Architecture

```
shopping-assistant/
├── app/                        # Main application
│   ├── main.py                 # FastAPI application entry point
│   ├── models/                 # Database models (SQLAlchemy)
│   │   ├── product.py          # Product model with ratings, features
│   │   ├── user.py             # User model with preferences, history
│   │   ├── discount.py         # Discount model with segmentation
│   │   └── conversation.py     # Chat conversation and order models
│   ├── api/                    # REST API endpoints
│   │   ├── chat.py             # Chat and conversation management
│   │   ├── products.py         # Product search and filtering
│   │   ├── discounts.py        # Discount CRUD and recommendations
│   │   └── notifications.py    # User notification system
│   ├── services/               # Business logic layer
│   │   ├── chat_service.py     # Main chat orchestration
│   │   └── discount_service.py # Discount management and analytics
│   ├── ai/                     # AI/ML components
│   │   ├── intent_classifier.py   # Intent classification engine
│   │   ├── entity_extractor.py    # NLP entity extraction
│   │   ├── recommendation_engine.py # Product recommendation system
│   │   └── response_generator.py   # Natural language generation
│   ├── utils/                  # Utility modules
│   │   ├── database.py         # SQLAlchemy configuration
│   │   ├── cache.py            # Redis caching system
│   │   └── config.py           # Application configuration
│   └── data/                   # Data processing modules
├── frontend/                   # Web interface
│   ├── index.html              # Modern chat interface
│   └── styles.css              # Responsive CSS styling
├── data/                       # Data storage
├── models/                     # Trained ML models
└── tests/                      # Test suite
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Redis Server
- SQLite (default) or PostgreSQL

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd shopping-assistant
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install fastapi uvicorn sqlalchemy pandas numpy scikit-learn
pip install redis python-multipart pydantic
pip install spacy transformers torch
python -m spacy download en_core_web_sm
```

4. **Start Redis server**
```bash
redis-server
```

5. **Configure environment variables** (optional)
```bash
# Create .env file
echo "DATABASE_URL=sqlite:///./shopping_assistant.db" > .env
echo "REDIS_URL=redis://localhost:6379" >> .env
echo "DEBUG=true" >> .env
```

6. **Run the application**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

7. **Access the application**
- Frontend: http://localhost:8000/static/index.html
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## 🔧 Configuration

The application uses Pydantic settings with support for environment variables:

```python
# Key configuration options
DATABASE_URL=sqlite:///./shopping_assistant.db
REDIS_URL=redis://localhost:6379
OPENAI_API_KEY=your_openai_key  # Optional for enhanced AI
DEBUG=true
MAX_RECOMMENDATIONS=10
SESSION_TIMEOUT_MINUTES=60
```

### Discount Configuration
- `MAX_DISCOUNT_PERCENTAGE=90.0`
- `DEFAULT_DISCOUNT_EXPIRY_DAYS=30`
- Support for multiple discount types and user segments

### AI Configuration
- `MAX_TOKENS=500`
- `TEMPERATURE=0.7`
- `SIMILARITY_THRESHOLD=0.5`
- 11 supported intents with extensible architecture

## 📚 API Documentation

### Chat Endpoints
- `POST /api/chat/message` - Send chat message
- `GET /api/chat/history/{session_id}` - Get conversation history
- `POST /api/chat/recommend` - Get product recommendations
- `DELETE /api/chat/session/{session_id}` - Clear chat session

### Product Endpoints
- `GET /api/products/` - List products with filtering
- `GET /api/products/{product_id}` - Get product details

### Discount Endpoints
- `GET /api/discounts/` - List available discounts
- `POST /api/discounts/` - Create new discount
- `PUT /api/discounts/{discount_id}` - Update discount
- `POST /api/discounts/apply` - Apply discount to order
- `GET /api/discounts/recommendations` - Get personalized discounts

### Notification Endpoints
- `GET /api/notifications/{user_id}` - Get user notifications
- `POST /api/notifications/` - Create notification

## 💬 Usage Examples

### Basic Chat Interaction
```bash
curl -X POST "http://localhost:8000/api/chat/message" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I am looking for discounted laptops under $1000",
    "session_id": "user123",
    "user_id": 1
  }'
```

### Get Product Recommendations
```bash
curl -X POST "http://localhost:8000/api/chat/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "category": "electronics",
    "max_price": 1000,
    "limit": 5
  }'
```

### Apply Discount
```bash
curl -X POST "http://localhost:8000/api/discounts/apply" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "product_ids": [1, 2, 3],
    "discount_code": "SAVE20"
  }'
```

## 🎯 Supported Chat Intents

1. **find_products** - Search for specific products
2. **find_discounts** - Look for available discounts
3. **product_recommendation** - Get personalized recommendations
4. **price_inquiry** - Ask about pricing
5. **order_status** - Check order status
6. **order_management** - Manage existing orders
7. **complaint** - Handle customer complaints
8. **greeting** - Welcome interactions
9. **goodbye** - Farewell interactions
10. **discount_management** - Manage discount codes
11. **promotion_inquiry** - Ask about promotions

## 🛠️ Development

### Database Models

**Product Model**: Complete product information with ratings, features, tags, and inventory tracking

**User Model**: User preferences, purchase history, browsing history, and personalization data

**Discount Model**: Multi-type discounts with user segmentation, usage limits, and promotion programs

**Conversation Model**: Chat history, message tracking, and order management

### AI Components

**Intent Classifier**: Machine learning-based classification with rule-based fallback

**Entity Extractor**: spaCy-based NLP with regex patterns for structured data

**Recommendation Engine**: TF-IDF vectorization with content-based filtering

**Response Generator**: Template-based natural language generation

### Caching Strategy
- Redis-based caching for conversations, products, and discounts
- Configurable TTL for different data types
- Automatic cache invalidation for updates

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=app
```

## 🚀 Deployment

### Docker Deployment
```bash
# Build image
docker build -t shopping-assistant .

# Run container
docker run -p 8000:8000 shopping-assistant
```

### Production Considerations
- Use PostgreSQL for production database
- Configure proper Redis clustering
- Set up environment variables for API keys
- Enable logging and monitoring
- Configure CORS for specific domains

## 📊 Analytics & Monitoring

- Discount usage analytics
- Conversation tracking
- Performance metrics
- Error logging and monitoring
- User engagement metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🛟 Support

For support and questions:
- Check the API documentation at `/docs`
- Review the conversation flow guide
- Open an issue for bugs or feature requests

---

**Note**: This is a comprehensive AI shopping assistant implementation with production-ready features. The system supports real-time chat, intelligent recommendations, advanced discount management, and scalable architecture for e-commerce applications.
