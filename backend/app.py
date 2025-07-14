import os
import re
import numpy as np
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import json

# For loading H5 models (LSTM, CNN)
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow successfully imported.")
except ImportError:
    print("TensorFlow not available. LSTM and CNN models will not work.")
    TENSORFLOW_AVAILABLE = False

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# No need for environment variables since OpenRouter API is now handled in the frontend

app = Flask(__name__)
# Enable CORS with more specific settings
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type", "Authorization"]}})

# Model directory and paths
MODELS_DIR = "models"  # Relative path to models directory
# Make sure paths are absolute
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR_ABS = os.path.join(BASE_DIR, MODELS_DIR)
VECTORIZER_PATH = os.path.join(MODELS_DIR_ABS, "tfidf_vectorizer_full.pkl")
METADATA_PATH = os.path.join(MODELS_DIR_ABS, "best_model_metadata.json")

print(f"Base directory: {BASE_DIR}")
print(f"Models directory: {MODELS_DIR_ABS}")
print(f"Vectorizer path: {VECTORIZER_PATH}")
print(f"Metadata path: {METADATA_PATH}")

# Define available models with their metadata
AVAILABLE_MODELS = {
    "gradient_boosting": {
        "name": "Gradient Boosting",
        "path": os.path.join(MODELS_DIR_ABS, "best_model_gradient_boosting.pkl"),
        "type": "sklearn",
        "metadata_path": METADATA_PATH,  # Uses the main metadata file
        "description": "Traditional ML model with high accuracy on structured features"
    },
    "lstm": {
        "name": "LSTM",
        "path": os.path.join(MODELS_DIR_ABS, "lstm_model.h5"),
        "type": "deep_learning",
        "metadata_path": METADATA_PATH,  # Uses the same metadata file as gradient boosting
        "description": "Deep learning model good at capturing sequential patterns"
    },
    "cnn": {
        "name": "CNN",
        "path": os.path.join(MODELS_DIR_ABS, "cnn_model.h5"),
        "type": "deep_learning",
        "metadata_path": METADATA_PATH,  # Uses the same metadata file as gradient boosting
        "description": "Deep learning model effective at capturing local patterns"
    }
}

# Print model paths for debugging
for model_id, model_info in AVAILABLE_MODELS.items():
    print(f"Model {model_id} path: {model_info['path']}")

# Dictionary to store loaded models and their metadata
loaded_models = {}
model_metadata = {}
current_model_name = "gradient_boosting"  # Default model

# Load vectorizer and metadata at startup
try:
    # Verify vectorizer exists
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"Vectorizer file not found: {VECTORIZER_PATH}")
    
    # Load vectorizer
    tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
    print("Vectorizer loaded successfully!")
    
    # Load shared metadata file
    if not os.path.exists(METADATA_PATH):
        print(f"Warning: Metadata file not found: {METADATA_PATH}")
        print("Using hardcoded metadata instead")
        # Use hardcoded metadata from the provided JSON
        shared_metadata = {
            "model_name": "Gradient Boosting",
            "model_type": "GradientBoostingClassifier",
            "model_path": "best_model_gradient_boosting.pkl",
            "vectorizer_path": "tfidf_vectorizer_full.pkl",
            "performance_metrics": {
                "test_accuracy": 0.9966666666666667,
                "test_precision": 1.0,
                "test_recall": 0.9933333333333333,
                "test_f1_score": 0.9966555183946488,
                "test_roc_auc": 0.9959611111111111
            },
            "dataset_info": {
                "total_samples": 3998,
                "training_samples": 2798,
                "validation_samples": 600,
                "test_samples": 600,
                "feature_count": 10012
            },
            "feature_info": {
                "total_samples": 3998,
                "real_samples": 1999,
                "fake_samples": 1999,
                "engineered_features": 12,
                "tfidf_features": 10000,
                "total_features": 10012,
                "feature_names": [
                    "title",
                    "subject",
                    "date",
                    "word_count",
                    "sentence_count",
                    "avg_word_length",
                    "sentiment_compound",
                    "exclamation_count",
                    "question_count",
                    "caps_ratio",
                    "stopword_ratio",
                    "unique_word_ratio"
                ],
                "tfidf_params": {
                    "max_features": 10000,
                    "ngram_range": [1, 2],
                    "vocabulary_size": 10000
                }
            },
            "training_timestamp": "2025-06-22 17:33:32"
        }
    else:
        # Load from file if it exists
        with open(METADATA_PATH, "r") as f:
            shared_metadata = json.load(f)
    
    # Store the same metadata for all models
    for model_id in AVAILABLE_MODELS:
        model_metadata[model_id] = shared_metadata
    
    print("Model metadata loaded successfully!")
    
    # Load the default model (gradient boosting)
    model_info = AVAILABLE_MODELS["gradient_boosting"]
    if not os.path.exists(model_info["path"]):
        raise FileNotFoundError(f"Default model file not found: {model_info['path']}")
    
    loaded_models["gradient_boosting"] = joblib.load(model_info["path"])
    print(f"Default model '{model_info['name']}' loaded successfully!")
    
    # Print the metadata for debugging
    print(f"Metadata feature names: {shared_metadata['feature_info']['feature_names']}")
    print(f"Metadata feature count: {shared_metadata['dataset_info']['feature_count']}")
    
except Exception as e:
    print(f"Error loading model files: {str(e)}")
    traceback.print_exc()
    # Exit if files can't be loaded
    exit(1)

# Extract feature names from metadata
feature_names = model_metadata["gradient_boosting"]["feature_info"]["feature_names"]

# Function to get model-specific metadata
def get_model_metadata(model_name):
    """Get metadata for a specific model"""
    if model_name not in model_metadata:
        raise ValueError(f"No metadata available for model: {model_name}")
    return model_metadata[model_name]

# AI verification is now handled in the frontend

# Function to preprocess text for deep learning models
def preprocess_text_for_deep_learning(text, model_name):
    """Preprocess text specifically for deep learning models"""
    # Clean and preprocess text
    cleaned_text = comprehensive_text_cleaning(text)
    preprocessed_text = advanced_text_preprocessing(cleaned_text, remove_stops=False)  # Keep stopwords for sequence models
    
    # Different preprocessing for different models
    if model_name == "lstm":
        # For LSTM, we might want to tokenize and pad sequences
        words = preprocessed_text.split()
        return preprocessed_text, words
    elif model_name == "cnn":
        # For CNN, we might want to create a 2D representation
        return preprocessed_text
    else:
        return preprocessed_text

# Function to load a model if not already loaded
def get_model(model_name):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    
    # If model is already loaded, return it
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    # Otherwise, load the model
    model_info = AVAILABLE_MODELS[model_name]
    model_path = model_info["path"]
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if model_info["type"] == "sklearn":
        # Load scikit-learn model
        model = joblib.load(model_path)
    elif model_info["type"] == "deep_learning":
        # Load TensorFlow model
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not available. Cannot load deep learning models.")
        
        # Custom load options for TensorFlow models
        try:
            print(f"Loading {model_info['name']} model from {model_path}...")
            
            # Suppress TensorFlow warnings during model loading
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
            
            # Load the model with custom options
            model = load_model(model_path, compile=False)  # Load without compiling
            
            # Compile the model with default optimizer and loss
            model.compile(
                optimizer='adam', 
                loss='binary_crossentropy', 
                metrics=['accuracy']
            )
            
            # Print model information
            print(f"Model loaded successfully: {model_info['name']}")
            print(f"Input shape: {model.input_shape}")
            print(f"Output shape: {model.output_shape}")
            
            # Don't print full summary as it can be verbose
            # Just print the number of layers
            print(f"Number of layers: {len(model.layers)}")
        except Exception as e:
            print(f"Error loading TensorFlow model: {str(e)}")
            traceback.print_exc()
            raise
    else:
        raise ValueError(f"Unknown model type: {model_info['type']}")
    
    # Store the loaded model
    loaded_models[model_name] = model
    print(f"Model '{model_info['name']}' loaded successfully!")
    
    return model

@app.route('/', methods=['GET'])
def index():
    """Root endpoint to check if the server is running"""
    return jsonify({
        "status": "ok",
        "message": "FactRadar API is running",
        "available_endpoints": ["/", "/models", "/predict", "/test"],
        "current_model": current_model_name
    })

@app.route('/test', methods=['GET'])
def test():
    """Test endpoint to verify server functionality"""
    try:
        # Check if vectorizer is loaded
        vectorizer_loaded = tfidf_vectorizer is not None
        
        # Check if models directory exists
        models_dir_exists = os.path.exists(MODELS_DIR_ABS)
        
        # Check if model files exist
        model_files = {}
        for model_id, model_info in AVAILABLE_MODELS.items():
            model_files[model_id] = os.path.exists(model_info["path"])
        
        return jsonify({
            "status": "ok",
            "vectorizer_loaded": vectorizer_loaded,
            "models_directory_exists": models_dir_exists,
            "model_files_exist": model_files,
            "current_model": current_model_name
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/models', methods=['GET'])
def get_models():
    """Endpoint to get available models"""
    models_info = {}
    for model_id, model_data in AVAILABLE_MODELS.items():
        models_info[model_id] = {
            "name": model_data["name"],
            "description": model_data["description"]
        }
    return jsonify({
        "models": models_info,
        "current": current_model_name
    })

@app.route('/predict', methods=['POST'])
def predict():
    global current_model_name
    
    print("Received prediction request")
    
    try:
        data = request.get_json()
        if not data:
            print("Error: No JSON data received")
            return jsonify({'error': 'No JSON data received'}), 400
            
        text = data.get('text', '')
        model_name = data.get('model', current_model_name)
        
        print(f"Request data - model: {model_name}, text length: {len(text)}")
        
        # Update current model if a valid one is provided
        if model_name in AVAILABLE_MODELS:
            current_model_name = model_name
            print(f"Using model: {current_model_name}")
        else:
            print(f"Warning: Unknown model '{model_name}', using default: {current_model_name}")
        
        if not text:
            print("Error: No text provided")
            return jsonify({'error': 'No text provided'}), 400
    except Exception as e:
        print(f"Error parsing request: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Error parsing request: {str(e)}'}), 400
    
    try:
        try:
            # Get the requested model
            model = get_model(current_model_name)
            model_info = AVAILABLE_MODELS[current_model_name]
            model_type = model_info["type"]
            print(f"Successfully loaded model: {model_info['name']}")
        except Exception as e:
            error_msg = f"Error loading model {current_model_name}: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return jsonify({'error': error_msg}), 500
        
        # Apply preprocessing based on model type
        if model_type == "deep_learning":
            # Special preprocessing for deep learning models
            if current_model_name == "lstm":
                preprocessed_text, _ = preprocess_text_for_deep_learning(text, current_model_name)
            elif current_model_name == "cnn":
                preprocessed_text = preprocess_text_for_deep_learning(text, current_model_name)
            else:
                cleaned_text = comprehensive_text_cleaning(text)
                preprocessed_text = advanced_text_preprocessing(cleaned_text)
        else:
            # Standard preprocessing for traditional ML models
            cleaned_text = comprehensive_text_cleaning(text)
            preprocessed_text = advanced_text_preprocessing(cleaned_text)
        
        # Get model-specific metadata
        model_meta = get_model_metadata(current_model_name)
        
        # No fallback function needed
        
        # Different prediction logic based on model type
        if model_type == "sklearn":
            # Traditional ML model (Gradient Boosting)
            try:
                # Vectorize text
                tfidf_vector = tfidf_vectorizer.transform([preprocessed_text]).toarray()
                
                # Compute numerical features
                num_features_dict = extract_comprehensive_features(text)
                
                # Extract numerical features in the same order as training
                gb_feature_names = model_meta['feature_info']['feature_names']
                num_features = [num_features_dict.get(name, 0) for name in gb_feature_names]
                
                # Combine features
                features = np.concatenate([tfidf_vector[0], num_features])
                
                print("Feature vector shape:", features.shape)
                print("TF-IDF shape:", tfidf_vector.shape)
                print("Num features shape:", len(num_features))
                print("Model expects:", model.n_features_in_)
                
                # Predict
                prediction = model.predict_proba([features])[0][1]  # Probability of being fake
            except Exception as e:
                print(f"Error using gradient boosting model: {str(e)}")
                traceback.print_exc()
                raise e
            
        elif model_type == "deep_learning":
            # Deep learning model (LSTM or CNN)
            # For deep learning models, we need special preprocessing
            try:
                if current_model_name == "lstm":
                    # LSTM model expects shape (None, 500) based on the error message
                    # First, get the TF-IDF vector
                    tfidf_vector = tfidf_vectorizer.transform([preprocessed_text]).toarray()
                    
                    # Get the expected input shape from the model
                    expected_input_shape = model.input_shape
                    print(f"LSTM model expected input shape: {expected_input_shape}")
                    
                    # Create a vector of exactly 500 features
                    feature_dim = 500  # From the error message
                    
                    # Prepare the input vector
                    if tfidf_vector.shape[1] >= feature_dim:
                        # If we have more features than needed, truncate
                        input_vector = tfidf_vector[0, :feature_dim].reshape(1, feature_dim)
                    else:
                        # If we have fewer features than needed, pad with zeros
                        input_vector = np.zeros((1, feature_dim))
                        input_vector[0, :tfidf_vector.shape[1]] = tfidf_vector[0]
                    
                    print(f"LSTM input vector shape: {input_vector.shape}")
                    
                    # Predict with verbose=0 to reduce output noise
                    raw_prediction = model.predict(input_vector, verbose=0)
                    print(f"LSTM prediction shape: {raw_prediction.shape}")
                    print(f"LSTM prediction value: {raw_prediction}")
                    
                elif current_model_name == "cnn":
                    # CNN model expects shape (None, 500) based on the error message
                    # First, get the TF-IDF vector
                    tfidf_vector = tfidf_vectorizer.transform([preprocessed_text]).toarray()
                    
                    # Get the expected input shape from the model
                    expected_input_shape = model.input_shape
                    print(f"CNN model expected input shape: {expected_input_shape}")
                    
                    # Create a vector of exactly 500 features
                    feature_dim = 500  # From the error message
                    
                    # Prepare the input vector
                    if tfidf_vector.shape[1] >= feature_dim:
                        # If we have more features than needed, truncate
                        input_vector = tfidf_vector[0, :feature_dim].reshape(1, feature_dim)
                    else:
                        # If we have fewer features than needed, pad with zeros
                        input_vector = np.zeros((1, feature_dim))
                        input_vector[0, :tfidf_vector.shape[1]] = tfidf_vector[0]
                    
                    print(f"CNN input vector shape: {input_vector.shape}")
                    
                    # Predict with verbose=0 to reduce output noise
                    raw_prediction = model.predict(input_vector, verbose=0)
                    print(f"CNN prediction shape: {raw_prediction.shape}")
                    print(f"CNN prediction value: {raw_prediction}")
                
                # Extract the prediction value
                if isinstance(raw_prediction, list):
                    prediction = float(raw_prediction[0])
                elif raw_prediction.ndim > 1:
                    prediction = float(raw_prediction[0][0])
                else:
                    prediction = float(raw_prediction[0])
                
                # Use the correct variable name for the next steps
                reshaped_data = input_vector  # Ensure the variable name is consistent
                    
            except Exception as e:
                print(f"Error using {current_model_name} model: {str(e)}")
                traceback.print_exc()
                # Re-raise the exception instead of using fallback
                raise
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        print("Prediction probability:", prediction)
        
        # Use the model name directly
        model_name = model_info["name"]
        
        # Determine the label and confidence
        is_fake = prediction > 0.5
        label = 'FAKE' if is_fake else 'REAL'
        confidence = float(prediction) if is_fake else float(1 - prediction)
        
        # Return prediction and confidence for frontend
        response_data = {
            'prediction': float(prediction),
            'label': label,
            'confidence': confidence,
            'version': '1.0',
            'model': model_name
        }
        
        return jsonify(response_data)
    except Exception as e:
        # Detailed error logging
        error_msg = f"Prediction error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        # More detailed error information
        print(f"Current model: {current_model_name}")
        print(f"Model type: {AVAILABLE_MODELS[current_model_name]['type']}")
        print(f"Text sample: {text[:100]}...")
        
        return jsonify({
            'error': error_msg,
            'model': current_model_name,
            'details': str(e)
        }), 500

def comprehensive_text_cleaning(text):
    """Comprehensive text cleaning pipeline"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Handle excessive punctuation
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{3,}', '...', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def advanced_text_preprocessing(text, remove_stops=True):
    """Advanced text preprocessing with NLTK (consistent with training pipeline)"""
    if pd.isna(text):
        return ""
    
    stop_words = set(stopwords.words('english'))
    text = str(text).lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove non-alphabetic tokens and short words
    tokens = [token for token in tokens if token.isalpha() and len(token) > 2]
    
    # Remove stopwords if specified
    if remove_stops:
        tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

def extract_comprehensive_features(text):
    """Extract comprehensive NLP features using NLTK"""
    if pd.isna(text):
        return {
            # Basic features
            'word_count': 0, 'char_count': 0, 'sentence_count': 0,
            'avg_word_length': 0, 'avg_sentence_length': 0,
            # Punctuation features
            'exclamation_count': 0, 'question_count': 0, 'caps_ratio': 0,
            'punctuation_density': 0,
            # Sentiment features
            'sentiment_compound': 0, 'sentiment_positive': 0, 'sentiment_negative': 0,
            # Linguistic features
            'pos_noun_ratio': 0, 'pos_verb_ratio': 0, 'pos_adj_ratio': 0,
            'unique_word_ratio': 0, 'stopword_ratio': 0,
            # Readability
            'readability_score': 0
        }
    
    text = str(text)
    
    # Basic text statistics
    words = word_tokenize(text.lower())
    sentences = sent_tokenize(text)
    
    word_count = len(words)
    char_count = len(text)
    sentence_count = len(sentences)
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    # Punctuation features
    exclamation_count = text.count('!')
    question_count = text.count('?')
    caps_count = sum(1 for c in text if c.isupper())
    caps_ratio = caps_count / len(text) if len(text) > 0 else 0
    punctuation_density = sum(1 for char in text if char in '.,;!?') / len(text) if len(text) > 0 else 0
    
    # Sentiment features
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    sentiment_compound = sentiment['compound']
    sentiment_positive = sentiment['pos']
    sentiment_negative = sentiment['neg']
    
    # Linguistic features (POS tagging)
    tagged_words = nltk.pos_tag(words)
    noun_count = sum(1 for word, tag in tagged_words if tag.startswith('N'))
    verb_count = sum(1 for word, tag in tagged_words if tag.startswith('V'))
    adj_count = sum(1 for word, tag in tagged_words if tag.startswith('J'))
    
    pos_noun_ratio = noun_count / word_count if word_count > 0 else 0
    pos_verb_ratio = verb_count / word_count if word_count > 0 else 0
    pos_adj_ratio = adj_count / word_count if word_count > 0 else 0
    
    unique_word_ratio = len(set(words)) / word_count if word_count > 0 else 0
    stop_words = set(stopwords.words('english'))
    stopword_count = sum(1 for word in words if word in stop_words)
    stopword_ratio = stopword_count / word_count if word_count > 0 else 0
    
    # Readability (simplified Flesch-Kincaid for demonstration)
    # This is a very basic approximation and might not perfectly match a full implementation
    # Syllable count is complex; using a simple heuristic for demonstration
    avg_syllables_per_word = 1.5 # Placeholder, actual calculation is complex
    readability_score = 206.835 - 1.015 * avg_word_length - 84.6 * avg_syllables_per_word
    
    return {
        'word_count': word_count,
        'char_count': char_count,
        'sentence_count': sentence_count,
        'avg_word_length': avg_word_length,
        'avg_sentence_length': avg_sentence_length,
        'exclamation_count': exclamation_count,
        'question_count': question_count,
        'caps_ratio': caps_ratio,
        'punctuation_density': punctuation_density,
        'sentiment_compound': sentiment_compound,
        'sentiment_positive': sentiment_positive,
        'sentiment_negative': sentiment_negative,
        'pos_noun_ratio': pos_noun_ratio,
        'pos_verb_ratio': pos_verb_ratio,
        'pos_adj_ratio': pos_adj_ratio,
        'unique_word_ratio': unique_word_ratio,
        'stopword_ratio': stopword_ratio,
        'readability_score': readability_score
    }

# Remove the old clean_text, tokenize_text, stem_token, and STOP_WORDS
STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
    'don', 'should', 'now'
}

if __name__ == '__main__':
    print("Starting Flask server...")
    # Allow connections from any host for frontend integration
    app.run(host='0.0.0.0', port=5000, debug=True)
