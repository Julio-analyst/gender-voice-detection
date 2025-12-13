"""
Integration Test Script
Test complete MLOps workflow
"""

import sys
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_models_exist():
    """Test if trained models exist"""
    print("\n" + "="*80)
    print("TEST 1: Check Models Exist")
    print("="*80)
    
    models_found = 0
    for model_type in ['lstm', 'rnn', 'gru']:
        model_path = Path(f'models/{model_type}_production.h5')
        
        if model_path.exists():
            print(f"   ✅ {model_type.upper()} model found: {model_path}")
            models_found += 1
        else:
            print(f"   ❌ {model_type.upper()} model NOT found: {model_path}")
    
    if models_found == 3:
        print(f"\n✅ TEST PASSED: All 3 models found")
        return True
    else:
        print(f"\n⚠️  TEST WARNING: Only {models_found}/3 models found")
        return False


def test_preprocessing():
    """Test preprocessing pipeline"""
    print("\n" + "="*80)
    print("TEST 2: Preprocessing Pipeline")
    print("="*80)
    
    try:
        from src.preprocessing.audio_cleaner import AudioCleaner
        from src.preprocessing.feature_extractor import MFCCExtractor
        
        print("   ✅ Imports successful")
        
        # Create instances
        cleaner = AudioCleaner()
        extractor = MFCCExtractor()
        
        print("   ✅ Objects created")
        print(f"\n✅ TEST PASSED: Preprocessing modules working")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        print(f"\n❌ TEST FAILED: Preprocessing error")
        return False


def test_model_loading():
    """Test model loading"""
    print("\n" + "="*80)
    print("TEST 3: Model Loading")
    print("="*80)
    
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        loaded = 0
        for model_type in ['lstm', 'rnn', 'gru']:
            model_path = Path(f'models/{model_type}_production.h5')
            
            if model_path.exists():
                model = keras.models.load_model(str(model_path))
                print(f"   ✅ {model_type.upper()} model loaded successfully")
                loaded += 1
            else:
                print(f"   ⚠️  {model_type.upper()} model not found")
        
        if loaded > 0:
            print(f"\n✅ TEST PASSED: {loaded}/3 models loaded")
            return True
        else:
            print(f"\n❌ TEST FAILED: No models loaded")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        print(f"\n❌ TEST FAILED: Model loading error")
        import traceback
        traceback.print_exc()
        return False


def test_prediction():
    """Test prediction with dummy data"""
    print("\n" + "="*80)
    print("TEST 4: Prediction")
    print("="*80)
    
    try:
        import numpy as np
        import tensorflow as tf
        from tensorflow import keras
        
        # Load LSTM model
        model_path = Path('models/lstm_production.h5')
        if not model_path.exists():
            print("   ⚠️  LSTM model not found, skipping test")
            return False
        
        model = keras.models.load_model(str(model_path))
        print("   ✅ Model loaded")
        
        # Generate dummy MFCC
        dummy_mfcc = np.random.randn(1, 93, 13).astype(np.float32)
        print(f"   ✅ Dummy data generated: {dummy_mfcc.shape}")
        
        # Predict
        prediction = model.predict(dummy_mfcc, verbose=0)
        print(f"   ✅ Prediction: {prediction[0][0]:.4f}")
        
        gender = "Perempuan" if prediction[0][0] > 0.5 else "Laki-laki"
        print(f"   ✅ Predicted gender: {gender}")
        
        print(f"\n✅ TEST PASSED: Prediction working")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        print(f"\n❌ TEST FAILED: Prediction error")
        import traceback
        traceback.print_exc()
        return False


def test_feedback_system():
    """Test feedback collection"""
    print("\n" + "="*80)
    print("TEST 5: Feedback System")
    print("="*80)
    
    try:
        import pandas as pd
        from datetime import datetime
        
        # Create test feedback
        feedback_dir = Path('data/feedback')
        feedback_dir.mkdir(parents=True, exist_ok=True)
        
        feedback_file = feedback_dir / 'feedback.csv'
        
        # Create sample feedback
        test_feedback = {
            'feedback_id': f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'audio_filename': 'test_audio.wav',
            'predicted_label': 'Laki-laki',
            'actual_label': 'Laki-laki',
            'model_type': 'lstm',
            'confidence': 0.85,
            'is_correct': True,
            'user_comment': 'Test feedback'
        }
        
        # Load or create feedback file
        if feedback_file.exists():
            df = pd.read_csv(feedback_file)
            original_count = len(df)
        else:
            df = pd.DataFrame(columns=list(test_feedback.keys()))
            original_count = 0
        
        # Add test feedback
        df = pd.concat([df, pd.DataFrame([test_feedback])], ignore_index=True)
        df.to_csv(feedback_file, index=False)
        
        # Verify
        df_check = pd.read_csv(feedback_file)
        new_count = len(df_check)
        
        print(f"   ✅ Feedback saved")
        print(f"   ✅ Original count: {original_count}")
        print(f"   ✅ New count: {new_count}")
        
        print(f"\n✅ TEST PASSED: Feedback system working")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        print(f"\n❌ TEST FAILED: Feedback system error")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation():
    """Test evaluation module"""
    print("\n" + "="*80)
    print("TEST 6: Evaluation Module")
    print("="*80)
    
    try:
        from src.training.evaluate import ModelEvaluator
        import numpy as np
        
        evaluator = ModelEvaluator(model_type='lstm')
        print("   ✅ Evaluator created")
        
        # Generate dummy data
        y_true = np.random.randint(0, 2, 50)
        y_pred = np.random.rand(50)
        
        # Calculate metrics
        metrics = evaluator.evaluate(y_true, y_pred)
        print(f"   ✅ Metrics calculated: Accuracy={metrics['accuracy']:.2%}")
        
        print(f"\n✅ TEST PASSED: Evaluation module working")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        print(f"\n❌ TEST FAILED: Evaluation error")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    print("="*80)
    print("INTEGRATION TEST SUITE")
    print("Gender Voice Detection MLOps Project")
    print("="*80)
    
    tests = [
        ("Models Exist", test_models_exist),
        ("Preprocessing", test_preprocessing),
        ("Model Loading", test_model_loading),
        ("Prediction", test_prediction),
        ("Feedback System", test_feedback_system),
        ("Evaluation", test_evaluation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            time.sleep(1)  # Pause between tests
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR in {test_name}: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name:20s}: {status}")
        if result:
            passed += 1
    
    total = len(results)
    print("\n" + "="*80)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("="*80)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready for deployment!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
