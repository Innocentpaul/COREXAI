"""
Smart dependency installer for COREXAI Dashboard
Handles compatibility issues and installs packages in the correct order
"""

import subprocess
import sys
import platform

def run_pip_install(package):
    """Install a package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    print("COREXAI Dashboard - Dependency Installer")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print("-" * 50)
    
    # First, upgrade pip, setuptools, and wheel
    print("Upgrading pip, setuptools, and wheel...")
    try:
        run_pip_install("--upgrade pip")
        run_pip_install("--upgrade setuptools")
        run_pip_install("--upgrade wheel")
    except Exception as e:
        print(f"Warning: Could not upgrade pip/setuptools/wheel: {e}")
    
    # Core dependencies - install in order of importance
    packages = [
        # Basic packages first
        ("numpy", "numpy"),  # Let pip choose the best version
        ("pandas", "pandas"),
        ("scikit-learn", "scikit-learn"),
        ("joblib", "joblib"),
        
        # Flask and dependencies
        ("Flask", "flask"),
        ("Werkzeug", "werkzeug"),
        ("Jinja2", "jinja2"),
        ("MarkupSafe", "markupsafe"),
        ("click", "click"),
        ("itsdangerous", "itsdangerous"),
        
        # Visualization
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("plotly", "plotly"),
        ("Pillow", "pillow"),
        
        # Explainability - these might need specific versions
        ("SHAP", "shap"),
        ("LIME", "lime"),
    ]
    
    failed_packages = []
    
    for display_name, package_name in packages:
        print(f"\nInstalling {display_name}...")
        try:
            run_pip_install(package_name)
            print(f"✓ {display_name} installed successfully")
        except Exception as e:
            print(f"✗ Failed to install {display_name}: {e}")
            failed_packages.append(display_name)
    
    print("\n" + "=" * 50)
    print("Installation Summary:")
    print(f"Successfully installed: {len(packages) - len(failed_packages)}/{len(packages)} packages")
    
    if failed_packages:
        print(f"\nFailed packages: {', '.join(failed_packages)}")
        print("\nTroubleshooting tips:")
        print("1. Try installing failed packages individually")
        print("2. Check if you need Visual C++ build tools (for Windows)")
        print("3. Consider using Anaconda/Miniconda for better compatibility")
        print("4. Try: pip install --no-binary :all: <package_name>")
    else:
        print("\nAll packages installed successfully!")
        print("You can now run: python app3.py")
    
    # Test imports
    print("\nTesting imports...")
    test_imports()

def test_imports():
    """Test if key packages can be imported"""
    test_packages = [
        "flask",
        "numpy",
        "pandas",
        "sklearn",
        "shap",
        "lime",
        "matplotlib",
        "plotly"
    ]
    
    for package in test_packages:
        try:
            __import__(package)
            print(f"✓ {package} imports correctly")
        except ImportError as e:
            print(f"✗ Cannot import {package}: {e}")

if __name__ == "__main__":
    main()