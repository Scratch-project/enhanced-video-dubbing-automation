#!/usr/bin/env python3
"""
🧪 Enhanced Installer Testing & Validation Script (2025)
Tests the critical 2025 updates in enhanced_installer_v2.1.py
"""

import sys
import subprocess
import importlib
import traceback
from datetime import datetime

def test_section(title):
    """Print formatted test section header"""
    print(f"\n{'='*60}")
    print(f"🧪 {title}")
    print(f"{'='*60}")

def check_version_constraint(package_name, version_spec, expected_range=None):
    """Check if a package version meets the specified constraints"""
    try:
        module = importlib.import_module(package_name)
        if hasattr(module, '__version__'):
            version = module.__version__
            print(f"✅ {package_name}: {version}")
            
            if expected_range:
                # Simple version checking (for basic validation)
                major_minor = '.'.join(version.split('.')[:2])
                if expected_range[0] <= major_minor < expected_range[1]:
                    print(f"   ✅ Version {version} is within expected range {expected_range}")
                else:
                    print(f"   ⚠️  Version {version} outside expected range {expected_range}")
            return True
        else:
            print(f"⚠️  {package_name}: No version info")
            return True
    except ImportError:
        print(f"❌ {package_name}: Not installed")
        return False
    except Exception as e:
        print(f"❌ {package_name}: Error - {e}")
        return False

def test_critical_imports():
    """Test critical package imports and versions"""
    test_section("CRITICAL PACKAGE VALIDATION")
    
    critical_packages = {
        'torch': ("2.1", "2.5"),
        'transformers': ("4.42", "4.47"),
        'accelerate': ("0.28", "0.35"),
        'numpy': ("1.24", "2.0"),
        'scipy': ("1.10", "1.14"),
    }
    
    results = {}
    for package, version_range in critical_packages.items():
        results[package] = check_version_constraint(package, version_range)
    
    return results

def test_functionality():
    """Test basic functionality of critical packages"""
    test_section("FUNCTIONALITY TESTS")
    
    tests = []
    
    # PyTorch CUDA test
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
        print(f"✅ PyTorch CUDA: Available={cuda_available}, Devices={device_count}")
        tests.append(True)
    except Exception as e:
        print(f"❌ PyTorch CUDA test failed: {e}")
        tests.append(False)
    
    # Transformers pipeline test
    try:
        from transformers import pipeline
        # Test with a tiny model to avoid memory issues
        classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        result = classifier("This is a test")
        print(f"✅ Transformers pipeline: {result[0]['label']}")
        tests.append(True)
    except Exception as e:
        print(f"❌ Transformers test failed: {e}")
        tests.append(False)
    
    # Whisper test
    try:
        import whisper
        model = whisper.load_model("tiny")
        print(f"✅ Whisper: Loaded tiny model successfully")
        tests.append(True)
    except Exception as e:
        print(f"❌ Whisper test failed: {e}")
        tests.append(False)
    
    # Audio processing test
    try:
        import librosa
        import soundfile
        # Generate a simple test signal
        import numpy as np
        test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))
        stft = librosa.stft(test_audio)
        print(f"✅ Audio processing: STFT shape {stft.shape}")
        tests.append(True)
    except Exception as e:
        print(f"❌ Audio processing test failed: {e}")
        tests.append(False)
    
    # DTW test
    try:
        import dtw
        x = [1, 2, 3, 4, 5]
        y = [1, 2, 3, 4, 5]
        if hasattr(dtw, 'distance'):
            distance = dtw.distance(x, y)
        elif hasattr(dtw, 'dtw'):
            distance = dtw.dtw(x, y)[0] if isinstance(dtw.dtw(x, y), tuple) else dtw.dtw(x, y)
        else:
            distance = "Unknown DTW interface"
        print(f"✅ DTW: Distance = {distance}")
        tests.append(True)
    except ImportError:
        try:
            import fastdtw
            distance = fastdtw.fastdtw([1, 2, 3], [1, 2, 3])
            print(f"✅ FastDTW fallback: Distance = {distance[0]}")
            tests.append(True)
        except Exception as e:
            print(f"❌ DTW/FastDTW test failed: {e}")
            tests.append(False)
    except Exception as e:
        print(f"❌ DTW test failed: {e}")
        tests.append(False)
    
    return tests

def test_security_features():
    """Test security-related package versions"""
    test_section("SECURITY VALIDATION")
    
    security_packages = {
        'requests': '2.31.0',    # CVE fixes
        'gitpython': '3.1.40',   # RCE fixes
        'pillow': '10.0.0',      # Security patches
    }
    
    for package, min_version in security_packages.items():
        try:
            module = importlib.import_module(package)
            if hasattr(module, '__version__'):
                version = module.__version__
                # Simple version comparison (major.minor.patch)
                current_parts = [int(x) for x in version.split('.')]
                min_parts = [int(x) for x in min_version.split('.')]
                
                if current_parts >= min_parts:
                    print(f"✅ {package}: {version} >= {min_version} (secure)")
                else:
                    print(f"⚠️  {package}: {version} < {min_version} (may be vulnerable)")
        except Exception as e:
            print(f"❌ {package}: Could not verify - {e}")

def test_memory_usage():
    """Test memory usage and cleanup"""
    test_section("MEMORY USAGE TEST")
    
    try:
        import psutil
        process = psutil.Process()
        
        # Memory before
        mem_before = process.memory_info()
        print(f"Memory before tests: {mem_before.rss / 1024**2:.1f} MB")
        
        # Load some heavy packages
        import torch
        import transformers
        
        # Memory after
        mem_after = process.memory_info()
        print(f"Memory after loading: {mem_after.rss / 1024**2:.1f} MB")
        print(f"Memory increase: {(mem_after.rss - mem_before.rss) / 1024**2:.1f} MB")
        
        # Test cleanup
        import gc
        gc.collect()
        
        mem_final = process.memory_info()
        print(f"Memory after cleanup: {mem_final.rss / 1024**2:.1f} MB")
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")

def generate_report(import_results, functionality_results):
    """Generate final test report"""
    test_section("FINAL VALIDATION REPORT")
    
    total_imports = len(import_results)
    successful_imports = sum(import_results.values())
    
    total_functions = len(functionality_results)
    successful_functions = sum(functionality_results)
    
    print(f"📊 Import Tests: {successful_imports}/{total_imports} passed")
    print(f"📊 Function Tests: {successful_functions}/{total_functions} passed")
    
    overall_score = (successful_imports + successful_functions) / (total_imports + total_functions)
    
    if overall_score >= 0.9:
        status = "🎉 EXCELLENT"
        recommendation = "Environment is ready for production use!"
    elif overall_score >= 0.7:
        status = "✅ GOOD"
        recommendation = "Environment is functional with minor issues."
    elif overall_score >= 0.5:
        status = "⚠️  FAIR"
        recommendation = "Environment needs attention before production use."
    else:
        status = "❌ POOR"
        recommendation = "Environment has critical issues. Re-run installer."
    
    print(f"\n🎯 Overall Score: {overall_score:.1%}")
    print(f"🏆 Status: {status}")
    print(f"💡 Recommendation: {recommendation}")
    
    return overall_score

def main():
    """Main testing function"""
    print(f"🧪 Enhanced Installer Validation (2025)")
    print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python: {sys.version}")
    
    try:
        # Run all tests
        import_results = test_critical_imports()
        functionality_results = test_functionality()
        test_security_features()
        test_memory_usage()
        
        # Generate final report
        score = generate_report(import_results, functionality_results)
        
        print(f"\n⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Exit with appropriate code
        if score >= 0.7:
            print(f"✅ Validation PASSED")
            sys.exit(0)
        else:
            print(f"❌ Validation FAILED")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Validation crashed: {e}")
        traceback.print_exc()
        sys.exit(2)

if __name__ == "__main__":
    main()
