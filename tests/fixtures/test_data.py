"""Test data fixtures and utilities."""

import pandas as pd


class TestDataGenerator:
    """Generate test data for various scenarios."""
    
    @staticmethod
    def create_balanced_dataset(size=100):
        """Create a balanced dataset with equal positive/negative/neutral samples."""
        texts = []
        labels = []
        
        # Positive samples
        positive_templates = [
            "Excellent {}!", "Amazing {} quality", "Love this {}", 
            "Outstanding {} performance", "Fantastic {} experience",
            "Great {} value", "Highly recommend this {}", "Perfect {}",
            "Wonderful {} features", "Superb {} design"
        ]
        
        # Negative samples  
        negative_templates = [
            "Terrible {} quality", "Worst {} ever", "Disappointing {}",
            "Poor {} performance", "Awful {} experience", "Hate this {}",
            "Broken {} functionality", "Useless {}", "Defective {}",
            "Complete {} failure"
        ]
        
        # Neutral samples
        neutral_templates = [
            "Average {} quality", "Okay {} performance", "Standard {}",
            "Regular {} experience", "Normal {} quality", "Typical {}",
            "Ordinary {} features", "Common {} design", "Basic {}",
            "Standard {} functionality"
        ]
        
        products = ["product", "service", "item", "device", "tool", "software", 
                   "application", "system", "platform", "solution"]
        
        samples_per_class = size // 3
        
        # Generate positive samples
        for i in range(samples_per_class):
            template = positive_templates[i % len(positive_templates)]
            product = products[i % len(products)]
            texts.append(template.format(product))
            labels.append("positive")
        
        # Generate negative samples
        for i in range(samples_per_class):
            template = negative_templates[i % len(negative_templates)]
            product = products[i % len(products)]
            texts.append(template.format(product))
            labels.append("negative")
        
        # Generate neutral samples
        remaining = size - 2 * samples_per_class
        for i in range(remaining):
            template = neutral_templates[i % len(neutral_templates)]
            product = products[i % len(products)]
            texts.append(template.format(product))
            labels.append("neutral")
        
        return pd.DataFrame({"text": texts, "label": labels})
    
    @staticmethod
    def create_imbalanced_dataset(size=100, positive_ratio=0.7):
        """Create an imbalanced dataset with specified positive ratio."""
        texts = []
        labels = []
        
        positive_count = int(size * positive_ratio)
        negative_count = size - positive_count
        
        # Generate positive samples
        for i in range(positive_count):
            texts.append(f"This is a positive review number {i}")
            labels.append("positive")
        
        # Generate negative samples
        for i in range(negative_count):
            texts.append(f"This is a negative review number {i}")
            labels.append("negative")
        
        return pd.DataFrame({"text": texts, "label": labels})
    
    @staticmethod
    def create_edge_case_dataset():
        """Create dataset with edge cases and challenging examples."""
        data = {
            "text": [
                # Empty and minimal text
                "",
                "a",
                "ok",
                
                # Very long text
                "This is a very long review that goes on and on and on. " * 50,
                
                # Mixed sentiment
                "Great product but terrible customer service",
                "Love the design but hate the price",
                
                # Sarcasm and irony
                "Oh wonderful, another broken product",
                "Just what I needed, more problems",
                
                # Special characters and formatting
                "AMAZING!!! 5/5 stars ⭐⭐⭐⭐⭐",
                "terrible... just terrible :(",
                
                # Numbers and technical terms
                "CPU performance increased by 200%",
                "RAM usage: 16GB/32GB - acceptable",
                
                # Multiple languages (if applicable)
                "Très bon produit",
                "Excelente calidad",
                
                # Spam-like content
                "BUY NOW!!! AMAZING DEAL!!! CLICK HERE!!!",
                "Free money! Win big! Call now!",
            ],
            "label": [
                "neutral", "neutral", "neutral",  # Minimal text
                "positive",  # Long text
                "neutral", "neutral",  # Mixed sentiment
                "negative", "negative",  # Sarcasm
                "positive", "negative",  # Special chars
                "positive", "neutral",  # Technical
                "positive", "positive",  # Other languages
                "negative", "negative",  # Spam
            ]
        }
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_domain_specific_dataset(domain="technology"):
        """Create domain-specific test data."""
        if domain == "technology":
            data = {
                "text": [
                    "The API response time is excellent",
                    "Memory leaks causing system crashes",
                    "Clean code architecture and good documentation",
                    "Buggy software with poor error handling",
                    "Scalable microservices design",
                    "Legacy codebase needs refactoring",
                    "Fast deployment pipeline",
                    "Security vulnerabilities found",
                    "Efficient database queries",
                    "Poor user interface design"
                ],
                "label": [
                    "positive", "negative", "positive", "negative", "positive",
                    "negative", "positive", "negative", "positive", "negative"
                ]
            }
        elif domain == "restaurant":
            data = {
                "text": [
                    "Delicious food and great atmosphere",
                    "Cold food and rude service",
                    "Fresh ingredients and creative menu",
                    "Overpriced with small portions",
                    "Excellent wine selection",
                    "Long wait times and noisy environment",
                    "Perfect date night restaurant",
                    "Food poisoning after eating here",
                    "Best pizza in town",
                    "Dirty tables and slow service"
                ],
                "label": [
                    "positive", "negative", "positive", "negative", "positive",
                    "negative", "positive", "negative", "positive", "negative"
                ]
            }
        else:
            # Generic domain
            data = {
                "text": [
                    "Excellent quality and service",
                    "Poor quality and experience",
                    "Good value for money",
                    "Overpriced and disappointing",
                    "Highly recommended",
                    "Would not recommend",
                    "Great customer support",
                    "Terrible customer service",
                    "Fast and reliable",
                    "Slow and unreliable"
                ],
                "label": [
                    "positive", "negative", "positive", "negative", "positive",
                    "negative", "positive", "negative", "positive", "negative"
                ]
            }
        
        return pd.DataFrame(data)
    
    @staticmethod
    def create_multilabel_dataset():
        """Create dataset with multiple labels (for future multi-label support)."""
        data = {
            "text": [
                "Great product with fast shipping",
                "Poor quality but good customer service", 
                "Excellent design and functionality",
                "Expensive but worth the price",
                "Cheap and cheerful option",
                "Premium quality with premium price",
                "Good features but poor documentation",
                "Easy to use but limited functionality",
                "Powerful tool with steep learning curve",
                "Simple design with great usability"
            ],
            "label": ["positive", "negative", "positive", "neutral", "positive",
                     "positive", "neutral", "neutral", "positive", "positive"],
            "aspect_quality": ["positive", "negative", "positive", "positive", "negative",
                              "positive", "neutral", "neutral", "positive", "positive"],
            "aspect_price": ["neutral", "neutral", "neutral", "negative", "positive",
                            "negative", "neutral", "neutral", "negative", "neutral"],
            "aspect_usability": ["neutral", "neutral", "positive", "neutral", "positive",
                                "neutral", "negative", "positive", "negative", "positive"]
        }
        
        return pd.DataFrame(data)


# Pre-defined test datasets
SAMPLE_REVIEWS = TestDataGenerator.create_balanced_dataset(30)
EDGE_CASES = TestDataGenerator.create_edge_case_dataset()
TECH_REVIEWS = TestDataGenerator.create_domain_specific_dataset("technology")
RESTAURANT_REVIEWS = TestDataGenerator.create_domain_specific_dataset("restaurant")