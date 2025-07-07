import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta

from app.models.product import Product
from app.models.user import User
from app.models.discount import Discount
from app.utils.cache import ProductCache, DiscountCache
from app.utils.config import settings

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """Comprehensive recommendation engine for products and discounts"""
    
    def __init__(self, db: Session):
        self.db = db
        self.products_df = None
        self.discounts_df = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.load_data()
    
    def load_data(self):
        """Load product and discount data from database"""
        try:
            self.load_product_data()
            self.load_discount_data()
        except Exception as e:
            logger.error(f"Error loading recommendation data: {e}")
    
    def load_product_data(self):
        """Load product data from database"""
        products = self.db.query(Product).filter(Product.is_active == True).all()
        
        if products:
            self.products_df = pd.DataFrame([{
                'id': p.id,
                'name': p.name,
                'description': p.description or '',
                'category': p.category or '',
                'brand': p.brand or '',
                'price': p.price,
                'original_price': p.original_price or p.price,
                'rating': p.rating or 0,
                'review_count': p.review_count or 0,
                'tags': ' '.join(p.tags) if p.tags else '',
                'features': ' '.join(p.features.values()) if p.features else '',
                'content': f"{p.name} {p.description} {p.category} {p.brand} {' '.join(p.tags) if p.tags else ''}"
            } for p in products])
            
            self.build_content_matrix()
    
    def load_discount_data(self):
        """Load discount data from database"""
        current_time = datetime.utcnow()
        discounts = self.db.query(Discount).filter(
            Discount.is_active == True,
            Discount.start_date <= current_time,
            Discount.end_date >= current_time
        ).all()
        
        if discounts:
            self.discounts_df = pd.DataFrame([{
                'id': d.id,
                'code': d.code,
                'name': d.name,
                'description': d.description or '',
                'discount_type': d.discount_type,
                'discount_value': d.discount_value,
                'category': d.category or 'all',
                'brand': d.brand or 'all',
                'user_segment': d.user_segment or 'all',
                'min_purchase_amount': d.min_purchase_amount,
                'max_discount_amount': d.max_discount_amount,
                'priority': d.priority,
                'promotion_program': d.promotion_program or '',
                'usage_count': d.usage_count,
                'usage_limit': d.usage_limit
            } for d in discounts])
    
    def build_content_matrix(self):
        """Build TF-IDF matrix for content-based recommendations"""
        if self.products_df is not None and len(self.products_df) > 0:
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english', 
                max_features=1000,
                ngram_range=(1, 2)
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.products_df['content'])
    
    # Product Recommendation Methods
    
    def get_similar_products(self, product_id: int, n_recommendations: int = 5) -> List[Dict]:
        """Get similar products based on content similarity"""
        if self.tfidf_matrix is None or self.products_df is None:
            return []
        
        try:
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
            
            recommendations = []
            for idx in similar_indices:
                product_row = self.products_df.iloc[idx]
                recommendations.append({
                    'id': int(product_row['id']),
                    'name': product_row['name'],
                    'price': product_row['price'],
                    'category': product_row['category'],
                    'brand': product_row['brand'],
                    'rating': product_row['rating'],
                    'similarity_score': sim_scores[idx+1][1]
                })
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Error getting similar products: {e}")
            return []
    
    def get_recommendations_by_category(self, category: str, n_recommendations: int = 5) -> List[Dict]:
        """Get top products in a specific category"""
        if self.products_df is None:
            return []
        
        try:
            category_products = self.products_df[
                self.products_df['category'].str.contains(category, case=False, na=False)
            ]
            
            if len(category_products) == 0:
                return []
            
            # Sort by rating and review count
            category_products = category_products.copy()
            category_products['score'] = (
                category_products['rating'] * 0.7 + 
                np.log1p(category_products['review_count']) * 0.3
            )
            
            top_products = category_products.nlargest(n_recommendations, 'score')
            
            return top_products[['id', 'name', 'price', 'category', 'brand', 'rating']].to_dict('records')
        
        except Exception as e:
            logger.error(f"Error getting category recommendations: {e}")
            return []
    
    def get_recommendations_by_price_range(self, min_price: float, max_price: float, 
                                          category: str = None, n_recommendations: int = 5) -> List[Dict]:
        """Get products within a price range"""
        if self.products_df is None:
            return []
        
        try:
            filtered_products = self.products_df[
                (self.products_df['price'] >= min_price) & 
                (self.products_df['price'] <= max_price)
            ]
            
            if category:
                filtered_products = filtered_products[
                    filtered_products['category'].str.contains(category, case=False, na=False)
                ]
            
            if len(filtered_products) == 0:
                return []
            
            # Sort by value (rating/price ratio)
            filtered_products = filtered_products.copy()
            filtered_products['value_score'] = filtered_products['rating'] / (filtered_products['price'] / 100)
            
            recommended = filtered_products.nlargest(n_recommendations, 'value_score')
            
            return recommended[['id', 'name', 'price', 'category', 'brand', 'rating']].to_dict('records')
        
        except Exception as e:
            logger.error(f"Error getting price range recommendations: {e}")
            return []
    
    def get_personalized_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
        """Get personalized recommendations for a user"""
        # Check cache first
        cached_recommendations = ProductCache.get_product_recommendations(user_id)
        if cached_recommendations:
            return cached_recommendations[:n_recommendations]
        
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            if not user:
                recommendations = self.get_popular_products(n_recommendations)
            else:
                recommendations = self._generate_personalized_recommendations(user, n_recommendations)
            
            # Cache recommendations
            ProductCache.set_product_recommendations(user_id, recommendations)
            
            return recommendations
        
        except Exception as e:
            logger.error(f"Error getting personalized recommendations: {e}")
            return self.get_popular_products(n_recommendations)
    
    def _generate_personalized_recommendations(self, user: User, n_recommendations: int) -> List[Dict]:
        """Generate personalized recommendations based on user data"""
        recommendations = []
        
        # Strategy 1: Based on purchase history
        if user.purchase_history and 'products' in user.purchase_history:
            purchased_products = user.purchase_history['products']
            for product_id in purchased_products[-5:]:  # Last 5 purchases
                similar = self.get_similar_products(product_id, 2)
                recommendations.extend(similar)
        
        # Strategy 2: Based on favorite categories
        if user.favorite_categories:
            for category in user.favorite_categories[:3]:  # Top 3 categories
                category_recs = self.get_recommendations_by_category(category, 2)
                recommendations.extend(category_recs)
        
        # Strategy 3: Based on price preferences
        if user.price_range_preference:
            price_range = user.price_range_preference
            price_recs = self.get_recommendations_by_price_range(
                price_range.get('min', 0),
                price_range.get('max', 10000),
                n_recommendations=3
            )
            recommendations.extend(price_recs)
        
        # Remove duplicates and sort by relevance
        seen_ids = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec['id'] not in seen_ids:
                seen_ids.add(rec['id'])
                unique_recommendations.append(rec)
        
        return unique_recommendations[:n_recommendations]
    
    def get_popular_products(self, n_recommendations: int = 5) -> List[Dict]:
        """Get popular products (fallback recommendation)"""
        if self.products_df is None:
            return []
        
        try:
            # Calculate popularity score
            products_df = self.products_df.copy()
            products_df['popularity_score'] = (
                products_df['rating'] * 0.4 +
                np.log1p(products_df['review_count']) * 0.4 +
                (products_df['original_price'] - products_df['price']) * 0.2
            )
            
            popular = products_df.nlargest(n_recommendations, 'popularity_score')
            return popular[['id', 'name', 'price', 'category', 'brand', 'rating']].to_dict('records')
        
        except Exception as e:
            logger.error(f"Error getting popular products: {e}")
            return []
    
    # Discount Recommendation Methods
    
    def get_relevant_discounts(self, entities: Dict, user_id: int = None, 
                              n_recommendations: int = 5) -> List[Dict]:
        """Get discounts relevant to user query and profile"""
        if self.discounts_df is None:
            return []
        
        try:
            relevant_discounts = self.discounts_df.copy()
            scores = np.ones(len(relevant_discounts))
            
            # Filter by category
            if 'category' in entities:
                category_mask = (
                    (relevant_discounts['category'] == 'all') |
                    (relevant_discounts['category'].str.contains(entities['category'], case=False, na=False))
                )
                scores = scores * (category_mask.astype(int) * 2 + 1)
            
            # Filter by brand
            if 'brand' in entities:
                brand_mask = (
                    (relevant_discounts['brand'] == 'all') |
                    (relevant_discounts['brand'].str.contains(entities['brand'], case=False, na=False))
                )
                scores = scores * (brand_mask.astype(int) * 1.5 + 1)
            
            # Filter by discount type
            if 'discount_type' in entities:
                type_mask = relevant_discounts['discount_type'] == entities['discount_type']
                scores = scores * (type_mask.astype(int) * 2 + 1)
            
            # User segment preference
            if user_id:
                user = self.db.query(User).filter(User.id == user_id).first()
                if user:
                    user_segment = self._determine_user_segment(user)
                    segment_mask = (
                        (relevant_discounts['user_segment'] == 'all') |
                        (relevant_discounts['user_segment'] == user_segment)
                    )
                    scores = scores * (segment_mask.astype(int) * 1.5 + 1)
            
            # Add priority and usage-based scoring
            priority_scores = relevant_discounts['priority'] / relevant_discounts['priority'].max()
            usage_scores = 1 - (relevant_discounts['usage_count'] / 
                               relevant_discounts['usage_limit'].fillna(float('inf')))
            
            final_scores = scores * (1 + priority_scores) * (1 + usage_scores)
            
            # Sort by final score
            relevant_discounts['score'] = final_scores
            top_discounts = relevant_discounts.nlargest(n_recommendations, 'score')
            
            return top_discounts.to_dict('records')
        
        except Exception as e:
            logger.error(f"Error getting relevant discounts: {e}")
            return []
    
    def get_discounts_for_promotion(self, promotion_program: str, 
                                   n_recommendations: int = 10) -> List[Dict]:
        """Get discounts for a specific promotion program"""
        if self.discounts_df is None:
            return []
        
        try:
            program_discounts = self.discounts_df[
                self.discounts_df['promotion_program'].str.contains(
                    promotion_program, case=False, na=False
                )
            ]
            
            # Sort by priority and discount value
            program_discounts = program_discounts.copy()
            program_discounts['promo_score'] = (
                program_discounts['priority'] * 0.4 +
                program_discounts['discount_value'] * 0.6
            )
            
            top_discounts = program_discounts.nlargest(n_recommendations, 'promo_score')
            return top_discounts.to_dict('records')
        
        except Exception as e:
            logger.error(f"Error getting promotion discounts: {e}")
            return []
    
    def get_personalized_discounts(self, user_id: int, 
                                  n_recommendations: int = 5) -> List[Dict]:
        """Get personalized discount recommendations"""
        # Check cache first
        cached_discounts = DiscountCache.get_user_discounts(user_id)
        if cached_discounts:
            return cached_discounts[:n_recommendations]
        
        try:
            user = self.db.query(User).filter(User.id == user_id).first()
            if not user:
                return self.get_general_discounts(n_recommendations)
            
            user_segment = self._determine_user_segment(user)
            
            # Get discounts based on user preferences
            personalized_discounts = self.discounts_df[
                (self.discounts_df['user_segment'].isin(['all', user_segment]))
            ].copy()
            
            # Score based on user preferences
            scores = np.ones(len(personalized_discounts))
            
            if user.favorite_categories:
                for category in user.favorite_categories:
                    category_mask = (
                        (personalized_discounts['category'] == 'all') |
                        (personalized_discounts['category'] == category)
                    )
                    scores += category_mask.astype(int) * 0.5
            
            if user.favorite_brands:
                for brand in user.favorite_brands:
                    brand_mask = (
                        (personalized_discounts['brand'] == 'all') |
                        (personalized_discounts['brand'] == brand)
                    )
                    scores += brand_mask.astype(int) * 0.3
            
            personalized_discounts['user_score'] = scores
            top_discounts = personalized_discounts.nlargest(n_recommendations, 'user_score')
            
            result = top_discounts.to_dict('records')
            
            # Cache results
            DiscountCache.set_user_discounts(user_id, result)
            
            return result
        
        except Exception as e:
            logger.error(f"Error getting personalized discounts: {e}")
            return self.get_general_discounts(n_recommendations)
    
    def get_general_discounts(self, n_recommendations: int = 5) -> List[Dict]:
        """Get general discounts for all users"""
        if self.discounts_df is None:
            return []
        
        try:
            general_discounts = self.discounts_df[
                self.discounts_df['user_segment'].isin(['all', 'regular'])
            ].copy()
            
            # Score by discount value and priority
            general_discounts['general_score'] = (
                general_discounts['discount_value'] * 0.6 +
                general_discounts['priority'] * 0.4
            )
            
            top_discounts = general_discounts.nlargest(n_recommendations, 'general_score')
            return top_discounts.to_dict('records')
        
        except Exception as e:
            logger.error(f"Error getting general discounts: {e}")
            return []
    
    def _determine_user_segment(self, user: User) -> str:
        """Determine user segment based on purchase history and profile"""
        if not user.purchase_history:
            return 'new'
        
        purchase_count = len(user.purchase_history.get('products', []))
        total_spent = user.purchase_history.get('total_spent', 0)
        
        if purchase_count == 0:
            return 'new'
        elif purchase_count >= 10 or total_spent >= 1000:
            return 'vip'
        else:
            return 'regular' 