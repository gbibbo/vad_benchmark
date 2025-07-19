#!/usr/bin/env python3
"""Quick test script for VAD-Benchmark installation"""

import sys
import importlib.util

def test_import(module_name, is_file=False):
    try:
        if is_file:
            spec = importlib.util.spec_from_file_location(module_name, module_name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            __import__(module_name)
        print(f"✅ {module_name}")
        return True
    except Exception as e:
        print(f"❌ {module_name}: {e}")
        return False

def main():
    print("🧪 VAD-Benchmark Installation Test")
    print("==================================")
    
    # Test dependencies
    print("\n📦 Dependencies:")
    deps = ['torch', 'torchaudio', 'torchvision', 'numpy', 'librosa', 'webrtcvad', 'transformers', 'hear21passt']
    dep_success = sum(test_import(dep) for dep in deps)
    
    # Test wrappers
    print("\n🔧 VAD Wrappers:")
    sys.path.append('.')
    wrappers = [
        'src/wrappers/vad_silero.py',
        'src/wrappers/vad_webrtc.py', 
        'src/wrappers/vad_whisper_tiny.py',
        'src/wrappers/vad_ast.py',
        'src/wrappers/vad_panns.py',
        'src/wrappers/vad_epanns.py',
        'src/wrappers/vad_passt.py'
    ]
    wrapper_success = sum(test_import(w, True) for w in wrappers)
    
    print(f"\n📊 Results: {dep_success}/{len(deps)} deps, {wrapper_success}/{len(wrappers)} wrappers")
    
    if dep_success == len(deps) and wrapper_success == len(wrappers):
        print("🎉 All tests passed! VAD-Benchmark ready!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
