#!/usr/bin/env python3
"""
Example usage of the reorganized creator scoring models.
"""
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def example_training():
    """Example of training individual models."""
    print("🚀 Example: Training Individual Models")
    print("=" * 50)
    
    print("1. Train Credibility Model:")
    print("   python metric_score_models/train_credibility.py --sample-labels")
    print("   - Uses 7 features focused on trust and reliability")
    print("   - Generates credibility scores 1-10")
    print()
    
    print("2. Train Storytelling Model:")
    print("   python metric_score_models/train_storytelling.py --sample-labels")
    print("   - Uses 6 features focused on narrative quality")
    print("   - Generates storytelling scores 1-10")
    print()
    
    print("3. Train Communication Model:")
    print("   python metric_score_models/train_communication.py --sample-labels")
    print("   - Uses 5 features focused on audience connection")
    print("   - Generates communication scores 1-10")
    print()

def example_prediction():
    """Example of unified prediction and evaluation."""
    print("🎯 Example: Unified Prediction & Evaluation")
    print("=" * 50)
    
    print("1. Predict with All Models:")
    print("   python metric_score_models/predict_evaluate.py")
    print("   - Uses all trained models")
    print("   - Generates overall weighted score")
    print()
    
    print("2. Predict with Specific Models:")
    print("   python metric_score_models/predict_evaluate.py --models credibility storytelling")
    print("   - Uses only specified models")
    print("   - Combines scores appropriately")
    print()
    
    print("3. Single Model Prediction:")
    print("   python metric_score_models/predict_evaluate.py --models communication")
    print("   - Uses only communication model")
    print("   - Provides detailed communication analysis")
    print()
    
    print("4. Full Analysis with Insights:")
    print("   python metric_score_models/predict_evaluate.py --evaluate --generate-insights --show-features")
    print("   - Evaluates model performance (if ground truth available)")
    print("   - Generates detailed insights report")
    print("   - Shows feature importance analysis")
    print()

def example_programmatic_usage():
    """Example of using models programmatically."""
    print("💻 Example: Programmatic Usage")
    print("=" * 50)
    
    try:
        from metric_score_models.combined_scorer import CombinedScorer
        
        # Initialize scorer
        scorer = CombinedScorer()
        
        # Load models (if available)
        try:
            scorer.load_models()
            print("✅ Models loaded successfully")
            
            # Check which models are available
            available_models = []
            if scorer.credibility_scorer.is_fitted:
                available_models.append("credibility")
            if scorer.storytelling_scorer.is_fitted:
                available_models.append("storytelling")
            if scorer.communication_scorer.is_fitted:
                available_models.append("communication")
            
            print(f"📊 Available models: {available_models}")
            
            # Example prediction (would need actual data)
            print("\n🔮 To make predictions:")
            print("   df = pd.read_csv('your_data.csv')")
            print("   results = scorer.save_combined_predictions(df)")
            print("   scorer.print_combined_summary(results)")
            
        except FileNotFoundError:
            print("⚠️  No trained models found")
            print("   Train models first using the individual training scripts")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you're running from the correct directory")

def show_model_features():
    """Show features used by each model."""
    print("📋 Model Features")
    print("=" * 50)
    
    try:
        from metric_score_models.config import (
            CREDIBILITY_FEATURES, STORYTELLING_FEATURES, COMMUNICATION_FEATURES
        )
        
        print("🎯 Credibility Features (7):")
        for i, feature in enumerate(CREDIBILITY_FEATURES, 1):
            print(f"   {i}. {feature}")
        
        print("\n📚 Storytelling Features (6):")
        for i, feature in enumerate(STORYTELLING_FEATURES, 1):
            print(f"   {i}. {feature}")
        
        print("\n💬 Communication Features (5):")
        for i, feature in enumerate(COMMUNICATION_FEATURES, 1):
            print(f"   {i}. {feature}")
        
        print(f"\n📊 Total unique features: {len(set(CREDIBILITY_FEATURES + STORYTELLING_FEATURES + COMMUNICATION_FEATURES))}")
        
    except ImportError as e:
        print(f"❌ Could not load features: {e}")

def main():
    """Run all examples."""
    print("🎬 Creator Scoring Models - Usage Examples")
    print("=" * 60)
    print()
    
    show_model_features()
    print()
    
    example_training()
    print()
    
    example_prediction()
    print()
    
    example_programmatic_usage()
    print()
    
    print("📖 For more details, see metric_score_models/README.md")

if __name__ == "__main__":
    main()