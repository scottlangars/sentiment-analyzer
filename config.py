# config.py - Configuration settings for sentiment analysis

import os

# Model Configuration
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_LENGTH = 512
TRUNCATION = True

# Processing Configuration
DEFAULT_BATCH_SIZE = 16
ENABLE_TRANSLATION = False
ENABLE_VALIDATION = True

# Confidence Thresholds
CONFIDENCE_THRESHOLD_LOW = 0.55   # Below this → NEUTRAL
CONFIDENCE_THRESHOLD_HIGH = 0.70  # Above this → High confidence

# Sentiment Labels
SENTIMENT_LABELS = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
LABEL_COLORS = {
    "POSITIVE": "#2ecc71",  # Green
    "NEUTRAL": "#f39c12",   # Orange
    "NEGATIVE": "#e74c3c"   # Red
}

# Text Processing
TEXT_COLUMN_KEYWORDS = [
    'text', 'review', 'comment', 'feedback', 
    'message', 'content', 'description', 'body', 'tweet'
]

GROUND_TRUTH_COLUMNS = ['Score', 'Sentiment', 'Label', 'True_Sentiment']

# Stopwords Configuration
ENABLE_STOPWORDS_REMOVAL = True
CUSTOM_STOPWORDS = []  # Add custom stopwords here

# File Paths
DATA_DIR = "data"
OUTPUT_FILE = "cleaned_sentiment_output.csv"
SUMMARY_FILE = "translated_sentiment.csv"
CONFUSION_MATRIX_FILE = "confusion_matrix.png"

# Streamlit Configuration
STREAMLIT_THEME = {
    "primaryColor": "#1f77b4",
    "backgroundColor": "#ffffff",
    "secondaryBackgroundColor": "#f0f2f6",
    "textColor": "#262730",
}

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Device Configuration (Auto-detected)
import torch
DEVICE = 0 if torch.cuda.is_available() else -1
DEVICE_NAME = "GPU (CUDA)" if DEVICE == 0 else "CPU"

# Visualization Configuration
FIGURE_DPI = 150
WORDCLOUD_WIDTH = 1200
WORDCLOUD_HEIGHT = 600
WORDCLOUD_MAX_WORDS = 100

# Translation Configuration
TRANSLATION_TIMEOUT = 30  # seconds
MAX_TRANSLATION_LENGTH = 512

# Validation Configuration
VALIDATION_METRICS = ['accuracy', 'precision', 'recall', 'f1-score']

# Logging
ENABLE_DETAILED_LOGGING = True
LOG_FILE = "sentiment_analysis.log"

def print_config():
    """Print current configuration"""
    print("\n" + "="*60)
    print(" "*15 + "⚙️  CONFIGURATION SETTINGS")
    print("="*60 + "\n")
    
    print(f"🤖 Model: {MODEL_NAME}")
    print(f"🖥️  Device: {DEVICE_NAME}")
    print(f"📦 Batch Size: {DEFAULT_BATCH_SIZE}")
    print(f"🌍 Translation: {'Enabled' if ENABLE_TRANSLATION else 'Disabled'}")
    print(f"✅ Validation: {'Enabled' if ENABLE_VALIDATION else 'Disabled'}")
    print(f"🎯 Confidence Thresholds: {CONFIDENCE_THRESHOLD_LOW} / {CONFIDENCE_THRESHOLD_HIGH}")
    print(f"📁 Output Directory: {DATA_DIR}")
    print()

if __name__ == "__main__":
    print_config() 
 
