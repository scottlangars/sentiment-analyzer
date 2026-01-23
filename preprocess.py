# preprocess.py - Data preprocessing and sentiment analysis pipeline

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sentiment_model import analyze_sentiment, batch_analyze_sentiment
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# Download stopwords if not available
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    print("📥 Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

def find_text_column(df):
    """
    Automatically detect the text column in the dataframe
    
    Args:
        df: Input dataframe
    
    Returns:
        str: Name of the text column
    """
    # Common column names for text data
    text_keywords = [
        'text', 'review', 'comment', 'feedback', 
        'message', 'content', 'description', 'body', 
        'tweet', 'post', 'opinion', 'response'
    ]
    
    # Check exact matches (case-insensitive)
    for col in df.columns:
        if col.lower() in text_keywords:
            print(f"✅ Found text column: '{col}'")
            return col
    
    # Check partial matches
    for col in df.columns:
        for keyword in text_keywords:
            if keyword in col.lower():
                print(f"✅ Found text column: '{col}' (partial match)")
                return col
    
    # If no match, use first string column with substantial text
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if column contains text (avg length > 10)
            sample = df[col].dropna().head(100)
            if len(sample) > 0:
                avg_length = sample.astype(str).str.len().mean()
                if avg_length > 10:
                    print(f"⚠️  Using '{col}' as text column (best guess)")
                    return col
    
    raise ValueError("❌ Could not find a suitable text column! Please ensure your CSV has a text column.")

def find_ground_truth_column(df):
    """
    Find ground truth sentiment/label column if it exists
    
    Args:
        df: Input dataframe
    
    Returns:
        str or None: Name of ground truth column
    """
    truth_keywords = [
        'sentiment', 'label', 'score', 'rating',
        'true_sentiment', 'ground_truth', 'actual'
    ]
    
    for col in df.columns:
        if col.lower() in truth_keywords:
            return col
    
    return None

def clean_text(text):
    """
    Clean and preprocess text
    
    Args:
        text: Input text string
    
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove mentions and hashtags (but keep the text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    
    # Remove special characters and digits (keep letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

def map_ground_truth_to_sentiment(value):
    """
    Map various ground truth formats to standard sentiment labels
    
    Args:
        value: Ground truth value (can be numeric, string, etc.)
    
    Returns:
        str: Mapped sentiment label
    """
    if pd.isna(value):
        return None
    
    # If already a sentiment string
    value_str = str(value).upper().strip()
    if value_str in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
        return value_str
    
    # Map common variations
    positive_words = ['POSITIVE', 'POS', 'GOOD', 'HAPPY', 'SATISFIED', '1', 'TRUE']
    negative_words = ['NEGATIVE', 'NEG', 'BAD', 'UNHAPPY', 'UNSATISFIED', '0', 'FALSE']
    neutral_words = ['NEUTRAL', 'NEU', 'OKAY', 'OK', 'MIXED', 'AVERAGE', '2']
    
    if value_str in positive_words:
        return 'POSITIVE'
    elif value_str in negative_words:
        return 'NEGATIVE'
    elif value_str in neutral_words:
        return 'NEUTRAL'
    
    # Try numeric conversion (e.g., star ratings)
    try:
        numeric_value = float(value)
        if numeric_value <= 2:
            return 'NEGATIVE'
        elif numeric_value == 3:
            return 'NEUTRAL'
        else:
            return 'POSITIVE'
    except (ValueError, TypeError):
        pass
    
    return None

def preprocess_csv(csv_path, translate=False, validate=True):
    """
    Main preprocessing function for sentiment analysis
    
    Args:
        csv_path: Path to input CSV file
        translate: Whether to translate non-English text (slower)
        validate: Whether to perform validation if ground truth exists
    
    Returns:
        pd.DataFrame: Processed dataframe with sentiment predictions
    """
    print("\n" + "="*70)
    print(" "*20 + "🔄 PREPROCESSING PIPELINE")
    print("="*70 + "\n")
    
    # Step 1: Load CSV
    print("📂 Step 1: Loading CSV file...")
    try:
        df = pd.read_csv(csv_path)
        print(f"   ✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        raise ValueError(f"❌ Failed to load CSV: {e}")
    
    # Step 2: Find text column
    print("\n🔍 Step 2: Detecting text column...")
    text_col = find_text_column(df)
    
    # Step 3: Find ground truth (optional)
    print("\n🎯 Step 3: Checking for ground truth labels...")
    truth_col = find_ground_truth_column(df)
    
    if truth_col:
        print(f"   ✅ Found ground truth in column: '{truth_col}'")
        df['True_Sentiment'] = df[truth_col].apply(map_ground_truth_to_sentiment)
        has_ground_truth = True
    else:
        print("   ℹ️  No ground truth found (validation will be skipped)")
        has_ground_truth = False
    
    # Step 4: Clean text
    print("\n🧹 Step 4: Cleaning text data...")
    df['Text'] = df[text_col].fillna('')
    df['clean_comment'] = df['Text'].apply(clean_text)
    
    # Remove empty entries
    initial_count = len(df)
    df = df[df['clean_comment'].str.len() > 0].reset_index(drop=True)
    removed = initial_count - len(df)
    
    if removed > 0:
        print(f"   ⚠️  Removed {removed} empty/invalid entries")
    print(f"   ✅ Cleaned {len(df)} valid text entries")
    
    # Step 5: Sentiment Analysis
    print("\n🤖 Step 5: Running sentiment analysis...")
    print(f"   Processing {len(df)} samples...")
    
    if translate:
        print("   🌍 Translation enabled (this will be slower)")
    
    # Use batch processing for efficiency
    try:
        results = batch_analyze_sentiment(
            df['Text'].tolist(), 
            translate=translate,
            batch_size=16
        )
        
        df['Predicted_Sentiment'] = [r[0] for r in results]
        df['Confidence_Score'] = [r[1] for r in results]
        
        print("   ✅ Sentiment analysis completed successfully!")
        
    except Exception as e:
        print(f"   ❌ Error during sentiment analysis: {e}")
        raise
    
    # Step 6: Statistics
    print("\n📊 Step 6: Computing statistics...")
    sentiment_counts = df['Predicted_Sentiment'].value_counts()
    
    print(f"   Sentiment Distribution:")
    for sentiment in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
        count = sentiment_counts.get(sentiment, 0)
        percentage = (count / len(df)) * 100 if len(df) > 0 else 0
        print(f"     {sentiment}: {count} ({percentage:.1f}%)")
    
    print(f"   Average Confidence: {df['Confidence_Score'].mean():.4f}")
    
    # Step 7: Validation (if ground truth exists)
    if has_ground_truth and validate:
        print("\n✅ Step 7: Validating predictions...")
        
        # Filter valid ground truth entries
        valid_mask = df['True_Sentiment'].notna()
        valid_df = df[valid_mask]
        
        if len(valid_df) > 0:
            y_true = valid_df['True_Sentiment']
            y_pred = valid_df['Predicted_Sentiment']
            
            # Calculate accuracy
            accuracy = (y_true == y_pred).mean()
            print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Confusion matrix
            try:
                cm = confusion_matrix(
                    y_true, 
                    y_pred,
                    labels=['POSITIVE', 'NEUTRAL', 'NEGATIVE']
                )
                
                # Save confusion matrix plot
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    cm, 
                    annot=True, 
                    fmt='d', 
                    cmap='Blues',
                    xticklabels=['POSITIVE', 'NEUTRAL', 'NEGATIVE'],
                    yticklabels=['POSITIVE', 'NEUTRAL', 'NEGATIVE']
                )
                plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
                plt.ylabel('True Label', fontsize=12, fontweight='bold')
                plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
                plt.tight_layout()
                
                os.makedirs('data', exist_ok=True)
                plt.savefig('data/confusion_matrix.png', dpi=150, bbox_inches='tight')
                print("   📊 Saved confusion matrix: data/confusion_matrix.png")
                plt.close()
                
            except Exception as e:
                print(f"   ⚠️  Could not generate confusion matrix: {e}")
            
            # Classification report
            try:
                print("\n   📋 Classification Report:")
                report = classification_report(
                    y_true, 
                    y_pred,
                    target_names=['POSITIVE', 'NEUTRAL', 'NEGATIVE'],
                    digits=4
                )
                print(report)
            except Exception as e:
                print(f"   ⚠️  Could not generate classification report: {e}")
        else:
            print("   ⚠️  No valid ground truth entries for validation")
    
    # Step 8: Save results
    print("\n💾 Step 8: Saving results...")
    
    # Save full results
    output_path = 'cleaned_sentiment_output.csv'
    df.to_csv(output_path, index=False)
    print(f"   ✅ Saved full results: {output_path}")
    
    # Save summary
    summary_cols = ['Text', 'Predicted_Sentiment', 'Confidence_Score']
    if has_ground_truth:
        summary_cols.append('True_Sentiment')
    
    summary_path = 'data/translated_sentiment.csv'
    os.makedirs('data', exist_ok=True)
    df[summary_cols].to_csv(summary_path, index=False)
    print(f"   ✅ Saved summary: {summary_path}")
    
    print("\n" + "="*70)
    print(" "*25 + "✅ PREPROCESSING COMPLETE")
    print("="*70 + "\n")
    
    return df


# Test function
if __name__ == "__main__":
    print("⚠️  This module should be imported, not run directly.")
    print("\nTo analyze a CSV file, use:")
    print("  python run_analysis.py your_file.csv")
    print("\nOr run the web interface:")
    print("  streamlit run app.py") 
