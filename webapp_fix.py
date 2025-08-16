#!/usr/bin/env python3
"""Quick webapp functionality test with minimal dependencies."""

from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def health_check():
    """Basic health check endpoint."""
    return jsonify({"status": "ok", "service": "sentiment-analyzer-pro"})

@app.route('/predict', methods=['POST'])
def predict():
    """Simple prediction endpoint."""
    try:
        data = request.get_json(force=True)
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field"}), 400
        
        text = data['text']
        # Simple sentiment logic for testing
        if any(word in text.lower() for word in ['good', 'great', 'excellent', 'love', 'amazing']):
            sentiment = 'positive'
        elif any(word in text.lower() for word in ['bad', 'terrible', 'hate', 'awful', 'horrible']):
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return jsonify({
            "text": text,
            "prediction": sentiment,
            "confidence": 0.95
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/version')
def version():
    """Get service version."""
    return jsonify({"version": "0.1.0", "name": "sentiment-analyzer-pro"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)