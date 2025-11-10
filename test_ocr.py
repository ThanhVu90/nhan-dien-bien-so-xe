"""
Quick Test - OCR Functionality
Test xem OCR có hoạt động không
"""

import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from controllers import ImageController
import os


def main():
    print("="*70)
    print("🧪 TESTING OCR FUNCTIONALITY")
    print("="*70)
    
    # Initialize controller
    print("\n1. Initializing controller...")
    controller = ImageController('weights/best.pt')
    print("✅ Controller initialized")
    
    # Find test image
    print("\n2. Finding test image...")
    test_dirs = [
        'data/images',
        'data/sample_plates',
        'results/images'
    ]
    
    test_image = None
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for file in os.listdir(test_dir):
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    test_image = os.path.join(test_dir, file)
                    break
            if test_image:
                break
    
    if not test_image:
        print("❌ No test image found!")
        print("   Please add an image to data/images/")
        return 1
    
    print(f"✅ Found test image: {test_image}")
    
    # Process image
    print("\n3. Processing image with OCR...")
    print("-"*70)
    
    try:
        results = controller.process_image(test_image)
        
        print(f"\n✅ Processing complete!")
        print(f"📊 Results: {len(results)} plate(s) detected\n")
        
        if results:
            for i, result in enumerate(results, 1):
                text = result.get('text', 'N/A')
                conf = result.get('confidence', 0.0)
                det_conf = result.get('detection_confidence', 0.0)
                
                print(f"Plate #{i}:")
                print(f"  📝 Text: {text}")
                print(f"  🎯 OCR Confidence: {conf:.2f}")
                print(f"  🔍 Detection Confidence: {det_conf:.2f}")
                print()
        else:
            print("⚠️  No plates detected")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("="*70)
    print("✅ TEST COMPLETE - OCR IS WORKING!" if results else "⚠️  TEST COMPLETE - NO PLATES FOUND")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
