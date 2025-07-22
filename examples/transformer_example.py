#!/usr/bin/env python3
"""
Example script demonstrating advanced transformer model capabilities.

This script shows how to:
1. Train a BERT-based sentiment classifier
2. Compare different model architectures
3. Evaluate model performance with comprehensive metrics
4. Save and load trained models

Requirements:
- pip install torch transformers (for full transformer training)
- Basic dependencies: pandas, scikit-learn, numpy

Usage:
    python examples/transformer_example.py
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transformer_trainer import TransformerTrainer, TransformerConfig, train_transformer_model
from src.model_comparison import ComprehensiveModelComparison, benchmark_models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_basic_comparison():
    """Basic model comparison without transformer training."""
    logger.info("=== Basic Model Comparison (Fast) ===")
    
    # Run quick comparison with traditional models + transformer baseline
    results = benchmark_models(
        csv_path="data/sample_reviews.csv",
        include_transformer_training=False  # Fast mode
    )
    
    logger.info(f"Compared {len(results)} models successfully")
    return results


def example_comprehensive_comparison():
    """Comprehensive model comparison with full transformer training."""
    logger.info("=== Comprehensive Model Comparison (Full Training) ===")
    
    try:
        # This requires transformer dependencies and significant compute
        results = benchmark_models(
            csv_path="data/sample_reviews.csv",
            include_transformer_training=True  # Full training mode
        )
        
        logger.info(f"Comprehensive comparison completed with {len(results)} models")
        return results
        
    except ImportError as e:
        logger.warning(f"Transformer dependencies not available: {e}")
        logger.info("Install torch and transformers for full functionality")
        return []
    except Exception as e:
        logger.error(f"Comprehensive comparison failed: {e}")
        return []


def example_custom_transformer_training():
    """Example of custom transformer training with specific configuration."""
    logger.info("=== Custom Transformer Training ===")
    
    try:
        # Custom configuration for transformer training
        config = TransformerConfig(
            model_name="distilbert-base-uncased",
            num_epochs=2,  # Reduced for demo
            batch_size=8,  # Smaller for demo
            learning_rate=2e-5,
            max_length=128,
            output_dir="models/custom_sentiment_model",
            early_stopping_patience=1
        )
        
        logger.info(f"Training transformer with config: {config}")
        
        # Train the model
        trainer = TransformerTrainer(config)
        results = trainer.train("data/sample_reviews.csv")
        
        logger.info("Training Results:")
        for key, value in results.items():
            if key != 'label_map':
                logger.info(f"  {key}: {value}")
        
        # Make some predictions
        sample_texts = [
            "This product is amazing!",
            "Terrible experience, would not recommend.",
            "It's okay, nothing special."
        ]
        
        predictions = trainer.predict(sample_texts)
        
        logger.info("Sample Predictions:")
        for text, pred in zip(sample_texts, predictions):
            logger.info(f"  '{text}' -> {pred}")
        
        return trainer, results
        
    except ImportError as e:
        logger.warning(f"Transformer dependencies not available: {e}")
        return None, {}
    except Exception as e:
        logger.error(f"Custom training failed: {e}")
        return None, {}


def example_detailed_analysis():
    """Example of detailed model analysis using ComprehensiveModelComparison."""
    logger.info("=== Detailed Model Analysis ===")
    
    comparison = ComprehensiveModelComparison("data/sample_reviews.csv")
    comparison.load_data()
    
    logger.info(f"Data loaded: {len(comparison.X_train)} train, {len(comparison.X_test)} test samples")
    
    # Evaluate individual model types
    logger.info("Evaluating baseline models...")
    baseline_results = comparison.evaluate_baseline_models()
    
    logger.info("Evaluating LSTM model...")
    lstm_result = comparison.evaluate_lstm_model()
    
    logger.info("Evaluating transformer model...")
    transformer_result = comparison.evaluate_transformer_model(use_full_training=False)
    
    # Combine results
    all_results = baseline_results.copy()
    if lstm_result:
        all_results.append(lstm_result)
    if transformer_result:
        all_results.append(transformer_result)
    
    # Display detailed analysis
    logger.info("\nDetailed Performance Analysis:")
    logger.info("-" * 80)
    
    for result in sorted(all_results, key=lambda x: x.accuracy, reverse=True):
        logger.info(f"Model: {result.model_name}")
        logger.info(f"  Accuracy:  {result.accuracy:.4f}")
        logger.info(f"  F1-Score:  {result.f1_score:.4f}")
        logger.info(f"  Precision: {result.precision:.4f}")
        logger.info(f"  Recall:    {result.recall:.4f}")
        logger.info(f"  Train Time: {result.training_time:.2f}s")
        logger.info(f"  Pred Time:  {result.prediction_time:.4f}s")
        
        if result.additional_metrics:
            logger.info("  Additional Metrics:")
            for key, value in result.additional_metrics.items():
                logger.info(f"    {key}: {value}")
        logger.info("")
    
    return all_results


def main():
    """Run all examples."""
    logger.info("Starting Transformer Model Examples")
    logger.info("=" * 60)
    
    # Example 1: Basic comparison (fast)
    try:
        basic_results = example_basic_comparison()
        logger.info(f"✓ Basic comparison completed with {len(basic_results)} models\n")
    except Exception as e:
        logger.error(f"✗ Basic comparison failed: {e}\n")
    
    # Example 2: Detailed analysis
    try:
        detailed_results = example_detailed_analysis()
        logger.info(f"✓ Detailed analysis completed with {len(detailed_results)} models\n")
    except Exception as e:
        logger.error(f"✗ Detailed analysis failed: {e}\n")
    
    # Example 3: Custom transformer training (requires dependencies)
    try:
        trainer, training_results = example_custom_transformer_training()
        if trainer:
            logger.info("✓ Custom transformer training completed\n")
        else:
            logger.info("⚠ Custom transformer training skipped (dependencies not available)\n")
    except Exception as e:
        logger.error(f"✗ Custom transformer training failed: {e}\n")
    
    # Example 4: Comprehensive comparison (requires dependencies and compute)
    try:
        comprehensive_results = example_comprehensive_comparison()
        if comprehensive_results:
            logger.info(f"✓ Comprehensive comparison completed with {len(comprehensive_results)} models\n")
        else:
            logger.info("⚠ Comprehensive comparison skipped (dependencies not available)\n")
    except Exception as e:
        logger.error(f"✗ Comprehensive comparison failed: {e}\n")
    
    logger.info("=" * 60)
    logger.info("All examples completed!")
    logger.info("\nNext Steps:")
    logger.info("1. Install torch and transformers for full functionality:")
    logger.info("   pip install torch transformers")
    logger.info("2. Try with larger datasets for better model comparison")
    logger.info("3. Experiment with different transformer models (BERT, RoBERTa, etc.)")
    logger.info("4. Adjust hyperparameters in TransformerConfig for your use case")


if __name__ == "__main__":
    main()