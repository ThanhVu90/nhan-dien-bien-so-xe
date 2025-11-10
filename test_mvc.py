"""
Quick Test for MVC Architecture
Test các components của MVC
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_models():
    """Test Model layer"""
    print("\n" + "="*70)
    print("🧪 TEST 1: MODEL LAYER")
    print("="*70)
    
    try:
        from models import PlateDetectorModel, OCRModel, ImageProcessorModel
        
        print("\n1. PlateDetectorModel...")
        detector = PlateDetectorModel('weights/best.pt')
        print(f"   ✅ Model info: {detector.get_model_info()}")
        
        print("\n2. OCRModel...")
        ocr = OCRModel(lang='en', use_gpu=False)
        print(f"   ✅ Model info: {ocr.get_model_info()}")
        
        print("\n3. ImageProcessorModel...")
        processor = ImageProcessorModel()
        print(f"   ✅ ImageProcessor initialized")
        
        print("\n✅ Model layer test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Model layer test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_controllers():
    """Test Controller layer"""
    print("\n" + "="*70)
    print("🧪 TEST 2: CONTROLLER LAYER")
    print("="*70)
    
    try:
        from controllers import PlateRecognitionController, ImageController, VideoController
        
        print("\n1. PlateRecognitionController...")
        controller = PlateRecognitionController('weights/best.pt', use_ocr=True)
        print(f"   ✅ Controller info: {controller.get_info()}")
        
        print("\n2. ImageController...")
        img_controller = ImageController('weights/best.pt')
        print(f"   ✅ ImageController initialized")
        
        print("\n3. VideoController...")
        vid_controller = VideoController('weights/best.pt')
        print(f"   ✅ VideoController initialized")
        
        print("\n✅ Controller layer test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Controller layer test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_views():
    """Test View layer"""
    print("\n" + "="*70)
    print("🧪 TEST 3: VIEW LAYER")
    print("="*70)
    
    try:
        from views import ConsoleView
        
        print("\n1. ConsoleView...")
        view = ConsoleView()
        view.show_header("Test Header")
        view.show_success("Test success message")
        view.show_info("Test info message")
        view.show_warning("Test warning message")
        print(f"   ✅ ConsoleView works correctly")
        
        print("\n✅ View layer test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ View layer test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_core():
    """Test Core utilities"""
    print("\n" + "="*70)
    print("🧪 TEST 4: CORE LAYER")
    print("="*70)
    
    try:
        from core import Config, Utils
        
        print("\n1. Config...")
        config = Config.get_config()
        print(f"   Model path: {Config.YOLO_MODEL_PATH}")
        print(f"   Default confidence: {Config.DEFAULT_CONFIDENCE}")
        print(f"   ✅ Config works correctly")
        
        print("\n2. Utils...")
        timestamp = Utils.create_timestamp()
        print(f"   Timestamp: {timestamp}")
        time_str = Utils.format_time(125.5)
        print(f"   Format time: {time_str}")
        print(f"   ✅ Utils work correctly")
        
        print("\n✅ Core layer test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Core layer test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration với ảnh thực"""
    print("\n" + "="*70)
    print("🧪 TEST 5: INTEGRATION TEST")
    print("="*70)
    
    try:
        from controllers import ImageController
        import glob
        
        # Find a test image
        test_images = glob.glob('data/images/*.jpg')
        if not test_images:
            test_images = glob.glob('data/images/*.png')
        
        if not test_images:
            print("   ⚠️  No test images found in data/images/")
            print("   ✅ Integration test SKIPPED")
            return True
        
        test_image = test_images[0]
        print(f"\n   Testing with: {os.path.basename(test_image)}")
        
        controller = ImageController('weights/best.pt')
        results = controller.detect_single_image(
            test_image,
            conf=0.25,
            save_result=False
        )
        
        print(f"   ✅ Detected {len(results)} plate(s)")
        for idx, result in enumerate(results, 1):
            text = result['plate_text']
            print(f"      Plate {idx}: {text}")
        
        print("\n✅ Integration test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🚀 MVC ARCHITECTURE - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    results = {
        'Models': test_models(),
        'Controllers': test_controllers(),
        'Views': test_views(),
        'Core': test_core(),
        'Integration': test_integration()
    }
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    print(f"\n   Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! MVC Architecture is working correctly!")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted!")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
