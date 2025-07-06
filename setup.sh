#!/bin/bash

# AI Shopping Assistant Setup Script
# This script sets up the development environment for the shopping assistant

echo "🛍️  AI Shopping Assistant Setup"
echo "================================"

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo "⬆️  Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Download spaCy model
echo "🔤 Downloading spaCy English model..."
python -m spacy download en_core_web_sm

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data models/trained logs

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env configuration file..."
    cat > .env << EOL
# Database Configuration
DATABASE_URL=sqlite:///./shopping_assistant.db

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Application Settings
DEBUG=true
APP_NAME=AI Shopping Assistant
SECRET_KEY=your-secret-key-change-in-production

# AI Configuration
MAX_TOKENS=500
TEMPERATURE=0.7
MAX_RECOMMENDATIONS=10
SIMILARITY_THRESHOLD=0.5

# Discount Settings
MAX_DISCOUNT_PERCENTAGE=90.0
DEFAULT_DISCOUNT_EXPIRY_DAYS=30

# Chat Settings
MAX_CONVERSATION_HISTORY=50
SESSION_TIMEOUT_MINUTES=60

# Optional: AI API Keys (uncomment and add your keys)
# OPENAI_API_KEY=your_openai_api_key_here
# ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Email Settings (for notifications)
# SMTP_SERVER=smtp.gmail.com
# SMTP_PORT=587
# SMTP_USERNAME=your_email@gmail.com
# SMTP_PASSWORD=your_app_password
EOL
    echo "✅ Created .env file with default configuration"
else
    echo "✅ .env file already exists"
fi

# Check if Redis is running
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo "✅ Redis is running"
    else
        echo "⚠️  Redis is not running. Please start Redis server:"
        echo "   - On Linux/Mac: redis-server"
        echo "   - On Windows: redis-server.exe"
    fi
else
    echo "⚠️  Redis is not installed. Please install Redis:"
    echo "   - On Ubuntu/Debian: sudo apt-get install redis-server"
    echo "   - On Mac: brew install redis"
    echo "   - On Windows: Download from https://redis.io/download"
fi

# Initialize database
echo "🗃️  Initializing database..."
python -c "
try:
    from app.utils.database import create_tables
    create_tables()
    print('✅ Database tables created successfully')
except Exception as e:
    print(f'❌ Error creating database tables: {e}')
"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To start the application:"
echo "1. Activate virtual environment:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "   source venv/Scripts/activate"
else
    echo "   source venv/bin/activate"
fi
echo "2. Start Redis server (if not running)"
echo "3. Start the application:"
echo "   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "Access points:"
echo "📱 Frontend: http://localhost:8000/static/index.html"
echo "📖 API Docs: http://localhost:8000/docs"
echo "❤️  Health Check: http://localhost:8000/health"
echo ""
echo "Happy shopping! 🛒" 