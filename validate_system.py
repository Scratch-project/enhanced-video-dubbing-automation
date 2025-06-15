#!/usr/bin/env python3
"""
Final Validation Script
Enhanced Video Dubbing Automation Project

This script performs a comprehensive validation of the entire system
to ensure it's ready for production video processing.
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

def main():
    """Run complete system validation"""
    print("🎬 Enhanced Video Dubbing Automation - Final Validation")
    print("=" * 60)
    print(f"Validation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    validation_results = {
        "validation_time": datetime.now().isoformat(),
        "system_ready": False,
        "checks": {},
        "recommendations": []
    }
    
    # Test 1: Environment Setup
    print("🔧 Testing Environment Setup...")
    try:
        from test_environment import EnvironmentValidator
        validator = EnvironmentValidator(local_mode=True)
        env_results = validator.run_full_validation()
        validation_results["checks"]["environment"] = env_results
        
        if env_results.get("success_rate", 0) >= 80:
            print("✅ Environment validation PASSED")
        else:
            print("❌ Environment validation FAILED")
            validation_results["recommendations"].append(
                "Fix environment issues before processing videos"
            )
    except Exception as e:
        print(f"❌ Environment validation error: {e}")
        validation_results["checks"]["environment"] = {"error": str(e)}
    
    print()
    
    # Test 2: Demo Pipeline
    print("🚀 Testing Demo Pipeline...")
    try:
        from demo_processor import DemoProcessor
        demo = DemoProcessor(local_mode=True)
        demo_results = demo.run_pipeline_test()
        validation_results["checks"]["pipeline"] = demo_results
        
        if "✅" in demo_results.get("overall_status", ""):
            print("✅ Pipeline test PASSED")
        else:
            print("❌ Pipeline test FAILED")
            validation_results["recommendations"].append(
                "Fix pipeline issues before processing real videos"
            )
    except Exception as e:
        print(f"❌ Pipeline test error: {e}")
        validation_results["checks"]["pipeline"] = {"error": str(e)}
    
    print()
    
    # Test 3: Configuration Validation
    print("⚙️ Testing Configuration...")
    try:
        from config import config
        config_check = {
            "directories_created": all([
                config.MODELS_DIR.exists(),
                config.TEMP_DIR.exists(),
                config.OUTPUT_DIR.exists(),
                config.LOGS_DIR.exists(),
                config.CHECKPOINTS_DIR.exists()
            ]),
            "audio_sample_rate": config.AUDIO_SAMPLE_RATE,
            "target_languages": config.TARGET_LANGUAGES,
            "base_dir": str(config.BASE_DIR)
        }
        validation_results["checks"]["configuration"] = config_check
        
        if config_check["directories_created"]:
            print("✅ Configuration validation PASSED")
        else:
            print("❌ Configuration validation FAILED")
            validation_results["recommendations"].append(
                "Check directory creation permissions"
            )
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        validation_results["checks"]["configuration"] = {"error": str(e)}
    
    print()
    
    # Test 4: Module Import Test
    print("📦 Testing Module Imports...")
    modules_to_test = [
        "step1_audio_processing",
        "step2_transcription", 
        "step3_translation",
        "step4_voice_cloning",
        "step5_synchronization",
        "step6_subtitles",
        "step7_quality_assurance",
        "main"
    ]
    
    import_results = {}
    import_success = 0
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"  ✅ {module_name}")
            import_results[module_name] = "✅ SUCCESS"
            import_success += 1
        except Exception as e:
            print(f"  ❌ {module_name}: {e}")
            import_results[module_name] = f"❌ FAILED: {e}"
    
    validation_results["checks"]["imports"] = import_results
    
    if import_success == len(modules_to_test):
        print("✅ Module import test PASSED")
    else:
        print(f"❌ Module import test FAILED ({import_success}/{len(modules_to_test)})")
        validation_results["recommendations"].append(
            "Fix module import errors before use"
        )
    
    print()
    
    # Test 5: File Structure Validation
    print("📁 Testing File Structure...")
    required_files = [
        "config.py",
        "utils.py", 
        "setup.py",
        "main.py",
        "Enhanced_Video_Dubbing_Kaggle.ipynb",
        "README.md",
        "requirements.txt"
    ]
    
    file_check = {}
    files_present = 0
    
    for filename in required_files:
        file_path = Path(filename)
        if file_path.exists():
            file_check[filename] = "✅ EXISTS"
            files_present += 1
            print(f"  ✅ {filename}")
        else:
            file_check[filename] = "❌ MISSING"
            print(f"  ❌ {filename}")
    
    validation_results["checks"]["files"] = file_check
    
    if files_present == len(required_files):
        print("✅ File structure test PASSED")
    else:
        print(f"❌ File structure test FAILED ({files_present}/{len(required_files)})")
        validation_results["recommendations"].append(
            "Ensure all required files are present"
        )
    
    print()
    
    # Overall Assessment
    print("📊 Overall System Assessment")
    print("-" * 40)
    
    # Count successful checks
    successful_checks = 0
    total_checks = 0
    
    for check_category, check_results in validation_results["checks"].items():
        total_checks += 1
        if check_category == "environment":
            if check_results.get("success_rate", 0) >= 80:
                successful_checks += 1
        elif check_category == "pipeline":
            if "✅" in str(check_results.get("overall_status", "")):
                successful_checks += 1
        elif check_category == "configuration":
            if check_results.get("directories_created", False):
                successful_checks += 1
        elif check_category == "imports":
            if all("✅" in status for status in check_results.values()):
                successful_checks += 1
        elif check_category == "files":
            if all("✅" in status for status in check_results.values()):
                successful_checks += 1
    
    success_rate = (successful_checks / total_checks * 100) if total_checks > 0 else 0
    validation_results["success_rate"] = success_rate
    
    print(f"Success Rate: {success_rate:.1f}% ({successful_checks}/{total_checks} checks passed)")
    
    if success_rate >= 90:
        print("🎉 SYSTEM READY FOR PRODUCTION!")
        print("   All critical components are working correctly.")
        validation_results["system_ready"] = True
        validation_results["status"] = "READY"
    elif success_rate >= 70:
        print("⚠️  SYSTEM PARTIALLY READY")
        print("   Some issues detected but system may work with limitations.")
        validation_results["system_ready"] = False
        validation_results["status"] = "PARTIAL"
    else:
        print("❌ SYSTEM NOT READY")
        print("   Critical issues detected. Please fix before use.")
        validation_results["system_ready"] = False
        validation_results["status"] = "NOT_READY"
    
    # Recommendations
    if validation_results["recommendations"]:
        print("\n💡 Recommendations:")
        for i, rec in enumerate(validation_results["recommendations"], 1):
            print(f"   {i}. {rec}")
    
    # Usage Instructions
    print("\n🚀 Next Steps:")
    if validation_results["system_ready"]:
        print("   1. Upload your Arabic videos to Kaggle dataset")
        print("   2. Open the Enhanced_Video_Dubbing_Kaggle.ipynb notebook")
        print("   3. Run the validation cells")
        print("   4. Configure processing parameters")
        print("   5. Start video dubbing!")
    else:
        print("   1. Review and fix the issues listed above")
        print("   2. Re-run this validation script")
        print("   3. Check the troubleshooting guide (TROUBLESHOOTING.md)")
        print("   4. Ensure all requirements are installed")
    
    # Save validation report
    report_path = Path("validation_report.json")
    with open(report_path, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"\n📄 Detailed validation report saved to: {report_path}")
    print("\n" + "=" * 60)
    print(f"Validation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return validation_results

if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0 if results["system_ready"] else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Validation failed with error: {e}")
        sys.exit(1)
