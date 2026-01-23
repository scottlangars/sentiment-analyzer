# backend/app.py - Deployment Version with Frontend Serving
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import os
from werkzeug.utils import secure_filename
from preprocess import preprocess_csv
from validate import validate_model
import traceback

app = Flask(__name__, static_folder='frontend/dist', static_url_path='')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('data', exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Serve React App
@app.route('/')
def serve_react():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Backend is running'})

@app.route('/api/analyze', methods=['POST'])
def analyze_sentiment():
    """Main endpoint for sentiment analysis"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only CSV files are allowed'}), 400
        
        # Get translation setting
        translate = request.form.get('translate', 'false').lower() == 'true'
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"Processing file: {filepath}")
        print(f"Translation enabled: {translate}")
        
        # Run sentiment analysis
        df = preprocess_csv(filepath, translate=translate, validate=False)
        
        # Calculate statistics
        sentiment_counts = df['Predicted_Sentiment'].value_counts()
        total = len(df)
        
        sentiments = {}
        for sentiment in ['POSITIVE', 'NEUTRAL', 'NEGATIVE']:
            count = sentiment_counts.get(sentiment, 0)
            percentage = round((count / total * 100), 1) if total > 0 else 0
            avg_confidence = round(df[df['Predicted_Sentiment'] == sentiment]['Confidence_Score'].mean(), 4)
            
            sentiments[sentiment] = {
                'count': int(count),
                'percentage': percentage,
                'avgConfidence': float(avg_confidence) if not pd.isna(avg_confidence) else 0.0
            }
        
        # Prepare sample data
        samples = []
        for _, row in df.head(100).iterrows():
            samples.append({
                'text': str(row['Text'])[:200],
                'sentiment': str(row['Predicted_Sentiment']),
                'confidence': float(row['Confidence_Score'])
            })
        
        # Clean up uploaded file
        os.remove(filepath)
        
        # Return results
        response = {
            'total': int(total),
            'sentiments': sentiments,
            'samples': samples,
            'success': True
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': f'Analysis failed: {str(e)}',
            'success': False
        }), 500

@app.route('/api/validate', methods=['POST'])
def validate_sentiment_model():
    """Endpoint for model validation with ground truth"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only CSV files are allowed'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"Validating model with file: {filepath}")
        
        # Run validation
        validation_results = validate_model(filepath, translate=False)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        # Return validation results
        response = {
            'success': True,
            'accuracy': validation_results['accuracy'],
            'precision': validation_results['precision'],
            'recall': validation_results['recall'],
            'f1_score': validation_results['f1_score'],
            'total_samples': validation_results['total_samples'],
            'correct_predictions': validation_results['correct_predictions'],
            'wrong_predictions': validation_results['wrong_predictions'],
            'error_rate': validation_results['error_rate'],
            'avg_confidence': validation_results['avg_confidence'],
            'correct_confidence': validation_results['correct_confidence'],
            'error_confidence': validation_results['error_confidence'],
            'class_metrics': validation_results['class_metrics'],
            'confusion_matrix': validation_results['confusion_matrix'],
            'metrics_chart': validation_results['metrics_chart'],
            'classification_report': validation_results['classification_report'],
            'sample_errors': validation_results['sample_errors']
        }
        
        return jsonify(response)
    
    except ValueError as e:
        print(f"Validation error: {str(e)}")
        return jsonify({
            'error': str(e),
            'success': False
        }), 400
    
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': f'Validation failed: {str(e)}',
            'success': False
        }), 500

@app.route('/api/user/history', methods=['GET'])
def get_user_history():
    """Get analysis history for a user"""
    return jsonify({
        'message': 'History is stored client-side',
        'success': True
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    host = os.environ.get('HOST', '0.0.0.0')
    
    print("=" * 70)
    print(" " * 20 + "🚀 STARTING FLASK SERVER")
    print("=" * 70)
    print(f"\n✅ Server running on http://{host}:{port}")
    print("✅ API endpoints:")
    print("   - POST /api/analyze     - Run sentiment analysis")
    print("   - POST /api/validate    - Validate model accuracy")
    print("   - GET  /api/health      - Health check")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(debug=False, host=host, port=port)