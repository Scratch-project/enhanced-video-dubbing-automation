#!/usr/bin/env python3
"""
Enhanced Video Dubbing Automation - Quick Launch Script
Run this script to quickly test or use the video dubbing system
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Video Dubbing Automation - Quick Launch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch.py --validate              # Run system validation
  python launch.py --demo                  # Run demo test
  python launch.py --process video.mp4     # Process a video
  python launch.py --help-setup            # Show setup instructions
        """
    )
    
    parser.add_argument('--validate', action='store_true',
                       help='Run complete system validation')
    parser.add_argument('--demo', action='store_true', 
                       help='Run demo pipeline test')
    parser.add_argument('--process', metavar='VIDEO_PATH',
                       help='Process a specific video file')
    parser.add_argument('--target-languages', nargs='+', default=['en', 'de'],
                       help='Target languages (default: en de)')
    parser.add_argument('--help-setup', action='store_true',
                       help='Show setup instructions')
    parser.add_argument('--local', action='store_true',
                       help='Run in local mode (not Kaggle)')
    
    args = parser.parse_args()
    
    if args.help_setup:
        show_setup_instructions()
        return
    
    if args.validate:
        print("🔍 Running system validation...")
        try:
            from validate_system import main as validate_main
            validate_main()
        except ImportError:
            print("❌ Validation script not found. Please ensure all files are present.")
            sys.exit(1)
        return
    
    if args.demo:
        print("🧪 Running demo test...")
        try:
            from demo_processor import run_demo_test
            results = run_demo_test()
            print(f"\n🎯 Demo completed: {results.get('overall_status', 'Unknown')}")
        except ImportError:
            print("❌ Demo processor not found. Please ensure all files are present.")
            sys.exit(1)
        return
    
    if args.process:
        video_path = Path(args.process)
        if not video_path.exists():
            print(f"❌ Video file not found: {video_path}")
            sys.exit(1)
        
        print(f"🎬 Processing video: {video_path}")
        print(f"🌐 Target languages: {', '.join(args.target_languages)}")
        
        try:
            from main import VideoDubbingOrchestrator
            
            orchestrator = VideoDubbingOrchestrator(local_mode=args.local)
            
            # Setup environment first
            if not orchestrator.setup_environment():
                print("❌ Environment setup failed")
                sys.exit(1)
            
            # Process the video
            result = orchestrator.process_single_video(video_path, args.target_languages)
            
            if result["status"] == "completed":
                print("✅ Video processing completed successfully!")
                print(f"📁 Output directory: {Path('output') / video_path.stem}")
            else:
                print("❌ Video processing failed")
                if "error" in result:
                    print(f"Error: {result['error']}")
                sys.exit(1)
                
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("Please ensure all required modules are installed.")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Processing error: {e}")
            sys.exit(1)
        return
    
    # If no specific action, show help
    parser.print_help()

def show_setup_instructions():
    """Display setup instructions"""
    instructions = """
🎬 Enhanced Video Dubbing Automation - Setup Instructions
========================================================

🔧 Local Setup:
1. Install Python 3.8+ and pip
2. Install dependencies: pip install -r requirements.txt
3. Install FFmpeg: https://ffmpeg.org/download.html
4. Run validation: python launch.py --validate

📊 Kaggle Setup:
1. Create a new Kaggle notebook
2. Enable GPU acceleration (Settings → Accelerator → GPU)
3. Upload the Enhanced_Video_Dubbing_Kaggle.ipynb notebook
4. Upload your video dataset to Kaggle
5. Run the notebook cells in order

🎥 Video Requirements:
- Format: MP4, AVI, MOV, MKV
- Language: Arabic (Egyptian dialect preferred)
- Duration: 60-120 minutes optimal
- Quality: 720p+ recommended
- Audio: Clear speech, minimal background noise

🌐 Output Languages:
- English (en)
- German (de)
- Extensible to other languages

📁 Directory Structure:
After setup, you'll have:
├── models/           # Cached AI models
├── temp/            # Temporary processing files
├── output/          # Final dubbed videos
├── logs/            # Processing logs
└── checkpoints/     # Recovery points

⚡ Quick Start:
1. python launch.py --validate     # Check system
2. python launch.py --demo         # Test pipeline
3. python launch.py --process video.mp4  # Process video

📚 Documentation:
- README.md          # Complete project overview
- TROUBLESHOOTING.md # Common issues and solutions
- requirements.txt   # Package dependencies

🆘 Support:
If you encounter issues:
1. Check TROUBLESHOOTING.md
2. Run validation script
3. Review error logs in logs/ directory
4. Ensure all requirements are installed

🚀 Ready to transform your Arabic videos into multilingual content!
    """
    print(instructions)

if __name__ == "__main__":
    main()
