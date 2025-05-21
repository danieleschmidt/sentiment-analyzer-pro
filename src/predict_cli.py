import argparse
import pickle
import os
from src.preprocessing import enhanced_preprocess, download_nltk_resources
# For loading sklearn and transformer models if not using MLflow's generic loader
from sklearn.pipeline import Pipeline as SklearnPipelineBoilerplate # Avoid name clash
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline as hf_pipeline
import torch
import mlflow # For loading MLflow models

def load_sklearn_model(model_path):
    """Loads a scikit-learn model/pipeline from a .pkl file or an MLflow model directory."""
    if os.path.isdir(model_path): # MLflow model directory
        try:
            model = mlflow.sklearn.load_model(model_path)
            print(f"Loaded scikit-learn model from MLflow directory: {model_path}")
            return model
        except Exception as e_mlflow:
            print(f"Failed to load model from MLflow directory {model_path} using mlflow.sklearn.load_model: {e_mlflow}")
            # Try to find a .pkl file in the directory as a fallback (e.g., if it's just a dir with a pkl)
            pkl_files = [f for f in os.listdir(model_path) if f.endswith(".pkl")]
            if pkl_files:
                # Attempt to load the first .pkl file found. This is a guess.
                pkl_file_path = os.path.join(model_path, pkl_files[0])
                print(f"Attempting to load from .pkl file in directory: {pkl_file_path}")
                try:
                    with open(pkl_file_path, 'rb') as f:
                        model = pickle.load(f)
                    print(f"Loaded scikit-learn model from .pkl: {pkl_file_path}")
                    return model
                except Exception as e_pkl_in_dir:
                    print(f"Failed to load .pkl from directory {pkl_file_path}: {e_pkl_in_dir}")
                    return None
            else:
                print(f"No .pkl file found in directory {model_path}.")
                return None
    elif os.path.isfile(model_path) and model_path.endswith(".pkl"): # Direct .pkl file
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Loaded scikit-learn model from .pkl: {model_path}")
            return model
        except Exception as e_pkl:
            print(f"Failed to load .pkl file {model_path}: {e_pkl}")
            return None
    else:
        print(f"Invalid model_path for scikit-learn: {model_path}. Expected .pkl file or MLflow model directory.")
        return None


def load_transformer_model_local(model_path):
    """Loads a Hugging Face transformer model and tokenizer from a local path."""
    try:
        # Check if it's an MLflow transformers logged model directory
        if os.path.exists(os.path.join(model_path, "transformers_model_config.json")) or \
           os.path.exists(os.path.join(model_path, "config.json")): # Standard HF save or MLflow save
            print(f"Attempting to load Transformer model from path: {model_path} (could be MLflow or direct save)")
            # MLflow transformers.load_model can also be used if path convention is known
            # For simplicity, using Auto* classes which work for both direct saves and some MLflow structures
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            # Create a pipeline for easier prediction
            # Ensure model is on the correct device
            device_to_use = 0 if torch.cuda.is_available() else -1
            print(f"Using device: {'cuda:0' if device_to_use == 0 else 'cpu'} for Transformer pipeline.")
            sentiment_pipeline = hf_pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device_to_use)
            print(f"Loaded Transformer model and tokenizer from {model_path} into a pipeline.")
            return sentiment_pipeline
        else:
            print(f"Path {model_path} does not seem to be a valid saved Transformer model directory.")
            return None
    except Exception as e:
        print(f"Error loading Transformer model from {model_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="CLI for sentiment analysis predictions.")
    parser.add_argument("--text", type=str, required=True, help="Input text for sentiment analysis.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved trained model (MLflow dir, .pkl file for sklearn, or HF model dir for transformer).")
    parser.add_argument("--model_type", type=str, default="sklearn", choices=["sklearn", "transformer"], help="Type of the model to load ('sklearn' or 'transformer'). Default: sklearn.")
    
    # Preprocessing args for sklearn (if not handled by a saved full pipeline)
    parser.add_argument("--use_stemming_sklearn", action="store_true", help="Apply stemming (for sklearn model).")
    parser.add_argument("--use_lemmatization_sklearn", action="store_true", help="Apply lemmatization (for sklearn model, overrides stemming).")
    parser.add_argument("--stopwords_sklearn", type=str, default=None, help="Path to custom stopwords file (one word per line) or 'none' for no stopwords, or 'nltk' for default (for sklearn model).")

    args = parser.parse_args()

    print("Checking NLTK resources for preprocessing...")
    download_nltk_resources(quiet=False) 

    model = None
    if args.model_type == "sklearn":
        model = load_sklearn_model(args.model_path)
        if model:
            custom_stopwords_list = None
            if args.stopwords_sklearn:
                if args.stopwords_sklearn.lower() == 'none':
                    custom_stopwords_list = []
                elif args.stopwords_sklearn.lower() != 'nltk': 
                    try:
                        with open(args.stopwords_sklearn, 'r') as f:
                            custom_stopwords_list = [line.strip() for line in f if line.strip()]
                        print(f"Loaded custom stopwords for sklearn from: {args.stopwords_sklearn}")
                    except Exception as e_stopwords:
                        print(f"Warning: Could not load custom stopwords file {args.stopwords_sklearn}: {e_stopwords}. Using NLTK default.")
                        custom_stopwords_list = None 

            # The loaded sklearn model is expected to be a Pipeline including the vectorizer
            # Thus, preprocessing here should only be the text cleaning part.
            # enhanced_preprocess returns a list of tokens. We need to join them back for the pipeline.
            processed_tokens = enhanced_preprocess(
                args.text,
                use_stemming=args.use_stemming_sklearn,
                use_lemmatization=args.use_lemmatization_sklearn,
                custom_stopwords_list=custom_stopwords_list
            )
            processed_text_for_pipeline = " ".join(processed_tokens)
            
            prediction = model.predict([processed_text_for_pipeline])[0]
            # Try to get probabilities if the model supports it
            probabilities = None
            if hasattr(model, "predict_proba"):
                try:
                    probabilities = model.predict_proba([processed_text_for_pipeline])[0]
                    # Assuming binary classification and getting probability of the positive class
                    # This might need adjustment based on how classes are ordered
                    positive_class_idx = list(model.classes_).index('positive') if 'positive' in list(model.classes_) else -1
                    if positive_class_idx != -1:
                         positive_prob = probabilities[positive_class_idx]
                         print(f"Predicted Sentiment (scikit-learn): {prediction} (Positive Prob: {positive_prob:.4f})")
                    else: # Fallback if 'positive' class not found or for general case
                        print(f"Predicted Sentiment (scikit-learn): {prediction} (Probabilities: {probabilities})")
                except Exception as e_proba:
                    print(f"Could not get probabilities: {e_proba}")
                    print(f"Predicted Sentiment (scikit-learn): {prediction}")
            else:
                 print(f"Predicted Sentiment (scikit-learn): {prediction}")
            print(f"\nInput Text: \"{args.text}\"")


    elif args.model_type == "transformer":
        transformer_pipeline = load_transformer_model_local(args.model_path)
        if transformer_pipeline:
            result = transformer_pipeline(args.text)[0] 
            predicted_label = result['label'].lower()
            print(f"\nInput Text: \"{args.text}\"")
            print(f"Predicted Sentiment (Transformer): {predicted_label} (Confidence: {result['score']:.4f})")
        model = transformer_pipeline 

    if model is None:
        print("Model could not be loaded. Exiting.")
    else:
        print("\nPrediction complete.")

if __name__ == "__main__":
    main()
