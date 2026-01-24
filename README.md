# 💬 AI Sentiment Analyzer

Advanced Customer Feedback Analysis with NLP and Deep Learning

## 🎯 Features

- **Real-time Sentiment Analysis**: Analyze customer feedback using state-of-the-art NLP models
- **Multi-language Support**: Built-in translation for non-English text
- **Interactive Dashboard**: Beautiful visualizations with charts and statistics
- **User Authentication**: Secure login/signup system with analysis history
- **Export Results**: Download results in CSV format
- **Batch Processing**: Process hundreds of reviews in seconds

## 🚀 How to Use

1. **Create an Account**: Sign up with your email and password
2. **Upload CSV File**: Click the upload area and select your CSV file
3. **Configure Settings**: Enable translation if needed
4. **Run Analysis**: Click "Run Sentiment Analysis" button
5. **View Results**: Explore interactive charts and data tables
6. **Download**: Export results in CSV format

## 📊 CSV Format

Your CSV file should contain a text column with one of these names:
- `text`
- `review`
- `comment`
- `feedback`
- `message`
- `content`

Example:
```csv
text
"This product is amazing!"
"Not satisfied with the service"
"Average experience, nothing special"
```

## 🔧 Technology Stack

- **Backend**: Flask + Python
- **Frontend**: React + Vite + Tailwind CSS
- **ML Model**: RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest)
- **Charts**: Recharts
- **Deployment**: Docker + Hugging Face Spaces

## 📈 Model Information

This application uses the **RoBERTa** model fine-tuned for sentiment analysis:
- Model: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Labels: POSITIVE, NEUTRAL, NEGATIVE

## 🛡️ Privacy

- All data is processed in real-time
- No data is permanently stored on servers
- User authentication data is stored locally in browser
- Uploaded files are deleted after processing

## 📝 License

MIT License - feel free to use this project for personal or commercial purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

---

**Note**: First run may take 1-2 minutes to download the ML model.