# validate.py - Model Validation and Accuracy Testing

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sentiment_model import batch_analyze_sentiment
from preprocess import find_text_column, find_ground_truth_column, map_ground_truth_to_sentiment
import warnings
warnings.filterwarnings('ignore')

def plot_confusion_matrix(cm, labels):
    """
    Create a confusion matrix visualization
    
    Args:
        cm: Confusion matrix array
        labels: List of class labels
    
    Returns:
        str: Base64 encoded image
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - Sentiment Analysis', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return image_base64

def plot_metrics_comparison(metrics_dict):
    """
    Create a bar chart comparing different metrics
    
    Args:
        metrics_dict: Dictionary of metric names and values
    
    Returns:
        str: Base64 encoded image
    """
    plt.figure(figsize=(12, 6))
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
    bars = plt.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}\n({height*100:.2f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.ylim(0, 1.1)
    plt.title('Model Performance Metrics', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Score', fontsize=12, fontweight='bold')
    plt.xlabel('Metrics', fontsize=12, fontweight='bold')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return image_base64

def validate_model(csv_path, translate=False):
    """
    Validate sentiment analysis model against ground truth labels
    
    Args:
        csv_path: Path to CSV file with ground truth labels
        translate: Whether to translate non-English text
    
    Returns:
        dict: Validation results with metrics and visualizations
    """
    print("\n" + "="*70)
    print(" "*20 + "🔍 MODEL VALIDATION")
    print("="*70 + "\n")
    
    # Step 1: Load CSV
    print("📂 Step 1: Loading CSV file...")
    try:
        df = pd.read_csv(csv_path)
        print(f"   ✅ Loaded {len(df)} rows")
    except Exception as e:
        raise ValueError(f"❌ Failed to load CSV: {e}")
    
    # Step 2: Find columns
    print("\n🔍 Step 2: Detecting columns...")
    text_col = find_text_column(df)
    truth_col = find_ground_truth_column(df)
    
    if not truth_col:
        raise ValueError("❌ No ground truth column found! Validation requires a sentiment/label column.")
    
    print(f"   ✅ Text column: '{text_col}'")
    print(f"   ✅ Ground truth column: '{truth_col}'")
    
    # Step 3: Prepare data
    print("\n🧹 Step 3: Preparing data...")
    df['Text'] = df[text_col].fillna('')
    df['True_Sentiment'] = df[truth_col].apply(map_ground_truth_to_sentiment)
    
    # Remove entries without valid ground truth
    initial_count = len(df)
    df = df[df['True_Sentiment'].notna()].reset_index(drop=True)
    removed = initial_count - len(df)
    
    if removed > 0:
        print(f"   ⚠️  Removed {removed} entries without valid ground truth")
    
    if len(df) == 0:
        raise ValueError("❌ No valid ground truth entries found for validation!")
    
    print(f"   ✅ {len(df)} valid entries for validation")
    
    # Step 4: Run predictions
    print("\n🤖 Step 4: Running sentiment predictions...")
    print(f"   Processing {len(df)} samples...")
    
    try:
        results = batch_analyze_sentiment(
            df['Text'].tolist(), 
            translate=translate,
            batch_size=16
        )
        
        df['Predicted_Sentiment'] = [r[0] for r in results]
        df['Confidence_Score'] = [r[1] for r in results]
        
        print("   ✅ Predictions completed successfully!")
        
    except Exception as e:
        raise Exception(f"❌ Prediction error: {e}")
    
    # Step 5: Calculate metrics
    print("\n📊 Step 5: Calculating validation metrics...")
    
    y_true = df['True_Sentiment']
    y_pred = df['Predicted_Sentiment']
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"   F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Per-class metrics
    print("\n   📋 Per-class Metrics:")
    labels = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']
    
    class_metrics = {}
    for label in labels:
        mask = (y_true == label)
        if mask.sum() > 0:
            class_acc = accuracy_score(y_true[mask], y_pred[mask])
            class_metrics[label] = {
                'accuracy': class_acc,
                'count': int(mask.sum()),
                'correct': int((y_pred[mask] == label).sum())
            }
            print(f"      {label}: {class_acc:.4f} ({class_acc*100:.2f}%) - {class_metrics[label]['correct']}/{class_metrics[label]['count']} correct")
    
    # Confusion Matrix
    print("\n   🔲 Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_image = plot_confusion_matrix(cm, labels)
    
    # Metrics comparison chart
    print("   📊 Generating metrics chart...")
    metrics_dict = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    metrics_image = plot_metrics_comparison(metrics_dict)
    
    # Classification Report
    print("\n   📋 Full Classification Report:")
    report = classification_report(
        y_true, 
        y_pred,
        target_names=labels,
        digits=4
    )
    print(report)
    
    # Step 6: Error Analysis
    print("\n🔍 Step 6: Error Analysis...")
    errors = df[y_true != y_pred]
    error_rate = len(errors) / len(df)
    
    print(f"   Total Errors: {len(errors)} ({error_rate*100:.2f}%)")
    
    if len(errors) > 0:
        print("\n   Top 5 Misclassified Examples:")
        for idx, row in errors.head(5).iterrows():
            print(f"\n      Text: {row['Text'][:100]}...")
            print(f"      True: {row['True_Sentiment']} | Predicted: {row['Predicted_Sentiment']} | Confidence: {row['Confidence_Score']:.4f}")
    
    # Step 7: Confidence Analysis
    print("\n📈 Step 7: Confidence Analysis...")
    avg_confidence = df['Confidence_Score'].mean()
    correct_confidence = df[y_true == y_pred]['Confidence_Score'].mean()
    error_confidence = df[y_true != y_pred]['Confidence_Score'].mean() if len(errors) > 0 else 0
    
    print(f"   Average Confidence: {avg_confidence:.4f}")
    print(f"   Correct Predictions: {correct_confidence:.4f}")
    print(f"   Wrong Predictions: {error_confidence:.4f}")
    
    # Prepare results
    results_dict = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'total_samples': int(len(df)),
        'correct_predictions': int((y_true == y_pred).sum()),
        'wrong_predictions': int(len(errors)),
        'error_rate': float(error_rate),
        'avg_confidence': float(avg_confidence),
        'correct_confidence': float(correct_confidence),
        'error_confidence': float(error_confidence),
        'class_metrics': class_metrics,
        'confusion_matrix': cm_image,
        'metrics_chart': metrics_image,
        'classification_report': report,
        'sample_errors': errors.head(10)[['Text', 'True_Sentiment', 'Predicted_Sentiment', 'Confidence_Score']].to_dict('records') if len(errors) > 0 else []
    }
    
    print("\n" + "="*70)
    print(" "*23 + "✅ VALIDATION COMPLETE")
    print("="*70 + "\n")
    
    return results_dict

def compare_models(csv_path, models_config):
    """
    Compare performance of different models or configurations
    
    Args:
        csv_path: Path to CSV file with ground truth
        models_config: List of model configurations to compare
    
    Returns:
        dict: Comparison results
    """
    print("\n" + "="*70)
    print(" "*18 + "🔄 COMPARING MODEL CONFIGURATIONS")
    print("="*70 + "\n")
    
    results = {}
    
    for config in models_config:
        print(f"\n📊 Testing: {config['name']}")
        print("-" * 50)
        
        try:
            validation_results = validate_model(
                csv_path, 
                translate=config.get('translate', False)
            )
            
            results[config['name']] = {
                'accuracy': validation_results['accuracy'],
                'f1_score': validation_results['f1_score'],
                'precision': validation_results['precision'],
                'recall': validation_results['recall']
            }
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[config['name']] = None
    
    # Create comparison visualization
    print("\n📊 Creating comparison chart...")
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) > 0:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        x = np.arange(len(valid_results))
        width = 0.2
        
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
        
        for i, metric in enumerate(metrics):
            values = [valid_results[model][metric] for model in valid_results.keys()]
            ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title(), 
                   color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Model Configuration', fontweight='bold', fontsize=12)
        ax.set_ylabel('Score', fontweight='bold', fontsize=12)
        ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=16)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(valid_results.keys(), rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        comparison_image = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        results['comparison_chart'] = comparison_image
    
    print("\n✅ Comparison complete!")
    
    return results


# Test function
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python validate.py <csv_file_path>")
        print("\nExample: python validate.py data/labeled_reviews.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    try:
        results = validate_model(csv_file, translate=False)
        
        print("\n" + "="*70)
        print(" "*20 + "📊 VALIDATION SUMMARY")
        print("="*70)
        print(f"\n✅ Accuracy: {results['accuracy']*100:.2f}%")
        print(f"✅ F1 Score: {results['f1_score']*100:.2f}%")
        print(f"✅ Precision: {results['precision']*100:.2f}%")
        print(f"✅ Recall: {results['recall']*100:.2f}%")
        print(f"\n📝 Correct: {results['correct_predictions']}/{results['total_samples']}")
        print(f"❌ Errors: {results['wrong_predictions']} ({results['error_rate']*100:.2f}%)")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        sys.exit(1) 
