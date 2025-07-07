import re
import spacy
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
from app.utils.config import get_ai_config

logger = logging.getLogger(__name__)

class EntityExtractor:
    """Extract entities from user messages"""
    
    def __init__(self):
        self.ai_config = get_ai_config()
        self.nlp = None
        self.load_nlp_model()
        
        # Define entity patterns
        self.patterns = {
            'price_range': r'\$?(\d+(?:\.\d{2})?)\s*(?:to|-|through)\s*\$?(\d+(?:\.\d{2})?)',
            'single_price': r'\$?(\d+(?:\.\d{2})?)',
            'percentage': r'(\d+(?:\.\d+)?)%',
            'order_number': r'(?:order|#)\s*([A-Z0-9]{6,})',
            'discount_code': r'(?:code|coupon|promo):\s*([A-Z0-9]{3,})',
            'date_range': r'(?:from|since|between)\s+([^,]+?)\s+(?:to|until|and)\s+([^,]+)',
            'quantity': r'(\d+)\s*(?:pieces?|items?|units?|pcs?)',
        }
        
        # Predefined categories and brands
        self.categories = [
            'electronics', 'clothing', 'shoes', 'accessories', 'home', 'kitchen',
            'laptop', 'phone', 'smartphone', 'tablet', 'headphones', 'camera',
            'dress', 'shirt', 'pants', 'jacket', 'sneakers', 'boots', 'sandals',
            'watch', 'jewelry', 'bag', 'furniture', 'appliances', 'books',
            'toys', 'sports', 'beauty', 'health', 'automotive', 'tools'
        ]
        
        self.brands = [
            'apple', 'samsung', 'google', 'microsoft', 'sony', 'lg', 'dell',
            'hp', 'lenovo', 'nike', 'adidas', 'puma', 'reebok', 'gucci',
            'prada', 'louis vuitton', 'chanel', 'versace', 'zara', 'h&m',
            'uniqlo', 'forever21', 'gap', 'levis', 'calvin klein'
        ]
        
        self.colors = [
            'red', 'blue', 'green', 'yellow', 'black', 'white', 'gray', 'grey',
            'brown', 'pink', 'purple', 'orange', 'silver', 'gold', 'navy',
            'maroon', 'beige', 'tan', 'cream', 'magenta', 'cyan', 'lime'
        ]
        
        self.sizes = [
            'xs', 'small', 's', 'medium', 'm', 'large', 'l', 'xl', 'xxl', 'xxxl',
            '6', '7', '8', '9', '10', '11', '12', '32', '34', '36', '38', '40'
        ]
        
        self.discount_types = [
            'percentage', 'fixed', 'bogo', 'buy one get one', 'free shipping',
            'clearance', 'flash sale', 'seasonal'
        ]
    
    def load_nlp_model(self):
        """Load spaCy NLP model"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def extract(self, text: str) -> Dict[str, Any]:
        """Extract all entities from text"""
        entities = {}
        text_lower = text.lower()
        
        # Extract price information
        price_info = self._extract_price(text)
        if price_info:
            entities.update(price_info)
        
        # Extract product category
        category = self._extract_category(text_lower)
        if category:
            entities['category'] = category
        
        # Extract brand
        brand = self._extract_brand(text_lower)
        if brand:
            entities['brand'] = brand
        
        # Extract color
        color = self._extract_color(text_lower)
        if color:
            entities['color'] = color
        
        # Extract size
        size = self._extract_size(text_lower)
        if size:
            entities['size'] = size
        
        # Extract discount type
        discount_type = self._extract_discount_type(text_lower)
        if discount_type:
            entities['discount_type'] = discount_type
        
        # Extract order number
        order_number = self._extract_order_number(text)
        if order_number:
            entities['order_number'] = order_number
        
        # Extract discount code
        discount_code = self._extract_discount_code(text)
        if discount_code:
            entities['discount_code'] = discount_code
        
        # Extract quantity
        quantity = self._extract_quantity(text)
        if quantity:
            entities['quantity'] = quantity
        
        # Extract date range
        date_range = self._extract_date_range(text)
        if date_range:
            entities['date_range'] = date_range
        
        # Use spaCy for additional entity extraction
        if self.nlp:
            spacy_entities = self._extract_with_spacy(text)
            entities.update(spacy_entities)
        
        return entities
    
    def _extract_price(self, text: str) -> Dict[str, Any]:
        """Extract price information"""
        entities = {}
        
        # Try to find price range first
        price_range_match = re.search(self.patterns['price_range'], text)
        if price_range_match:
            min_price = float(price_range_match.group(1))
            max_price = float(price_range_match.group(2))
            entities['price_range'] = {
                'min': min_price,
                'max': max_price
            }
        else:
            # Look for single price
            price_matches = re.findall(self.patterns['single_price'], text)
            if price_matches:
                prices = [float(p) for p in price_matches]
                entities['price'] = prices[0]  # Take the first price found
        
        # Extract percentage (for discounts)
        percentage_match = re.search(self.patterns['percentage'], text)
        if percentage_match:
            entities['percentage'] = float(percentage_match.group(1))
        
        return entities
    
    def _extract_category(self, text: str) -> Optional[str]:
        """Extract product category"""
        for category in sorted(self.categories, key=len, reverse=True):
            if category in text:
                return category
        return None
    
    def _extract_brand(self, text: str) -> Optional[str]:
        """Extract brand name"""
        for brand in sorted(self.brands, key=len, reverse=True):
            if brand in text:
                return brand
        return None
    
    def _extract_color(self, text: str) -> Optional[str]:
        """Extract color"""
        for color in self.colors:
            if color in text:
                return color
        return None
    
    def _extract_size(self, text: str) -> Optional[str]:
        """Extract size"""
        for size in self.sizes:
            if f" {size} " in f" {text} " or f" {size}," in f" {text}," or text.endswith(f" {size}"):
                return size
        return None
    
    def _extract_discount_type(self, text: str) -> Optional[str]:
        """Extract discount type"""
        for discount_type in sorted(self.discount_types, key=len, reverse=True):
            if discount_type in text:
                if discount_type == 'buy one get one':
                    return 'bogo'
                elif discount_type == 'fixed':
                    return 'fixed_amount'
                return discount_type.replace(' ', '_')
        return None
    
    def _extract_order_number(self, text: str) -> Optional[str]:
        """Extract order number"""
        match = re.search(self.patterns['order_number'], text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return None
    
    def _extract_discount_code(self, text: str) -> Optional[str]:
        """Extract discount code"""
        match = re.search(self.patterns['discount_code'], text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return None
    
    def _extract_quantity(self, text: str) -> Optional[int]:
        """Extract quantity"""
        match = re.search(self.patterns['quantity'], text)
        if match:
            return int(match.group(1))
        return None
    
    def _extract_date_range(self, text: str) -> Optional[Dict[str, str]]:
        """Extract date range"""
        match = re.search(self.patterns['date_range'], text, re.IGNORECASE)
        if match:
            return {
                'start': match.group(1).strip(),
                'end': match.group(2).strip()
            }
        return None
    
    def _extract_with_spacy(self, text: str) -> Dict[str, Any]:
        """Extract entities using spaCy"""
        entities = {}
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                if ent.label_ == "MONEY" and 'price' not in entities:
                    # Extract money amounts
                    price_text = ent.text.replace('$', '').replace(',', '')
                    try:
                        entities['price'] = float(price_text)
                    except ValueError:
                        pass
                
                elif ent.label_ == "ORG" and 'brand' not in entities:
                    # Extract organizations as potential brands
                    entities['brand'] = ent.text.lower()
                
                elif ent.label_ == "PRODUCT" and 'product' not in entities:
                    # Extract product names
                    entities['product'] = ent.text.lower()
                
                elif ent.label_ == "DATE" and 'date' not in entities:
                    # Extract dates
                    entities['date'] = ent.text
                
                elif ent.label_ == "CARDINAL" and 'quantity' not in entities:
                    # Extract numbers as potential quantities
                    try:
                        quantity = int(ent.text)
                        if 1 <= quantity <= 100:  # Reasonable quantity range
                            entities['quantity'] = quantity
                    except ValueError:
                        pass
        
        except Exception as e:
            logger.error(f"Error in spaCy entity extraction: {e}")
        
        return entities
    
    def validate_entities(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean extracted entities"""
        validated = {}
        
        for key, value in entities.items():
            if key == 'price' and isinstance(value, (int, float)) and value > 0:
                validated[key] = value
            elif key == 'price_range' and isinstance(value, dict):
                if 'min' in value and 'max' in value and value['min'] <= value['max']:
                    validated[key] = value
            elif key == 'percentage' and isinstance(value, (int, float)) and 0 <= value <= 100:
                validated[key] = value
            elif key == 'quantity' and isinstance(value, int) and value > 0:
                validated[key] = value
            elif key in ['category', 'brand', 'color', 'size', 'discount_type'] and value:
                validated[key] = str(value).lower()
            elif key in ['order_number', 'discount_code'] and value:
                validated[key] = str(value).upper()
            elif value:  # Keep other non-empty values
                validated[key] = value
        
        return validated 