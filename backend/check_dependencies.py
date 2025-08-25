# backend/check_dependencies.py
import importlib
import subprocess
import sys

def check_and_install_dependencies():
    """Check if required packages are installed, install only if missing"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'jinja2', 
        'python_multipart',
        'aiofiles',
        'pandas',
        'yfinance',
        'requests',
        'pyyaml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace('_', '-').split('-')[0])
            print(f"✅ {package}: Already installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}: Missing")
    
    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("✅ All missing packages installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install packages: {e}")
            return False
    else:
        print("✅ All required packages are already installed!")
    
    return True

if __name__ == "__main__":
    check_and_install_dependencies()