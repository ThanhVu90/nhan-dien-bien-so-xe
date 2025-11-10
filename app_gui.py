"""
GUI Application - License Plate Recognition
MVC Architecture with Tkinter GUI
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from controllers import ImageController
from views.gui_view import GUIView
from core.config import Config


def main():
    """Main function"""
    print("🚀 Starting License Plate Recognition GUI...")
    print("=" * 70)
    
    try:
        # Initialize config
        config = Config()
        
        # Initialize controller with model path
        print("📦 Loading models...")
        controller = ImageController(
            model_path=config.YOLO_MODEL_PATH
        )
        print("✅ Models loaded successfully!")
        
        # Initialize and run GUI
        print("🎨 Launching GUI...")
        gui = GUIView(controller=controller)
        gui.run()
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
