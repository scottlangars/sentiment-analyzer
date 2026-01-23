 # sentiment_model.py - Enhanced Version

from transformers import pipeline
import torch
import pandas as pd
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
import warnings
warnings.filterwarnings('ignore')

# Auto-detect device (GPU/CPU)
device = 0 if torch.cuda.is_available() else -1
print(f"🖥️  Device: {'GPU' if device == 0 else 'CPU'}")

# Load sentiment analysis model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=device,
    max_length=512,
    truncation=True
)

# Label mapping for different model outputs
label_map = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL", 
    "LABEL_2": "POSITIVE",
    "negative": "NEGATIVE",
    "neutral": "NEUTRAL",
    "positive": "POSITIVE"
}

def translate_to_english(text):
    """Translate non-English text to English with error handling"""
    try:
        # Skip translation if text is too short
        if len(text.strip()) < 3:
            return text
            
        lang = detect(text)
        
        # Only translate if not English
        if lang != "en":
            translator = GoogleTranslator(source=lang, target="en")
            translated = translator.translate(text[:512])
            print(f"  ℹ️  Translated from {lang}: {text[:50]}... → {translated[:50]}...")
            return translated
        return text
        
    except LangDetectException:
        # If language detection fails, assume English
        return text
    except Exception as e:
        print(f"  ⚠️  Translation error: {e}")
        return text

def analyze_sentiment(text, translate=False):
    """
    Analyze sentiment of text with optional translation
    
    Args:
        text: Input text to analyze
        translate: Whether to translate non-English text (default: False for speed)
    
    Returns:
        pd.Series: [sentiment_label, confidence_score]
    """
    # Handle empty or invalid text
    if not isinstance(text, str) or text.strip() == "":
        return pd.Series(["NEUTRAL", 0.0])
    
    try:
        # Optional translation (disabled by default for performance)
        if translate:
            text = translate_to_english(text)
        
        # Truncate to model's max length
        text = text[:512]
        
        # Run sentiment analysis
        result = sentiment_analyzer(text)[0]
        raw_label = result["label"]
        confidence = round(result["score"], 4)
        
        # Map label to standard format
        sentiment = label_map.get(raw_label, raw_label.upper())
        
        # Dynamic confidence thresholding
        # If confidence is low, classify as NEUTRAL
        if confidence < 0.55:
            sentiment = "NEUTRAL"
        # For borderline cases, be conservative
        elif confidence < 0.70 and sentiment != "NEUTRAL":
            # Keep original label but note low confidence
            pass
        
        return pd.Series([sentiment, confidence])
        
    except Exception as e:
        print(f"  ⚠️  Error analyzing text '{text[:50]}...': {e}")
        return pd.Series(["NEUTRAL", 0.0])

def batch_analyze_sentiment(texts, translate=False, batch_size=8):
    """
    Analyze sentiment for multiple texts in batches (more efficient)
    
    Args:
        texts: List of texts to analyze
        translate: Whether to translate non-English text
        batch_size: Number of texts to process at once
    
    Returns:
        List of tuples: [(sentiment, confidence), ...]
    """
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        try:
            # Optional translation
            if translate:
                batch = [translate_to_english(t) if isinstance(t, str) else t for t in batch]
            
            # Truncate all texts
            batch = [str(t)[:512] if isinstance(t, str) else "" for t in batch]
            
            # Batch inference
            batch_results = sentiment_analyzer(batch)
            
            for result in batch_results:
                raw_label = result["label"]
                confidence = round(result["score"], 4)
                sentiment = label_map.get(raw_label, raw_label.upper())
                
                # Apply confidence threshold
                if confidence < 0.55:
                    sentiment = "NEUTRAL"
                
                results.append((sentiment, confidence))
                
        except Exception as e:
            print(f"  ⚠️  Batch error: {e}")
            # Add neutral results for failed batch
            results.extend([("NEUTRAL", 0.0)] * len(batch))
    
    return results
