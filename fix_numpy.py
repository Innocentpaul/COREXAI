"""
Fix NumPy version conflict by downgrading to a compatible version
"""

import subprocess
import sys

def run_command(command):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(e.stderr)
        return False

def main():
    print("Fixing NumPy version conflict...")
    print("-" * 50)
    
    # Step 1: Uninstall current numpy
    print("Step 1: Uninstalling current NumPy...")
    run_command(f"{sys.executable} -m pip uninstall numpy -y")
    
    # Step 2: Install compatible numpy version
    print("\nStep 2: Installing NumPy 1.26.4 (compatible version)...")
    if not run_command(f"{sys.executable} -m pip install numpy==1.26.4"):
        print("Failed to install numpy 1.26.4, trying numpy<2...")
        run_command(f"{sys.executable} -m pip install 'numpy<2'")
    
    # Step 3: Reinstall packages that might need to be rebuilt
    print("\nStep 3: Reinstalling packages that depend on NumPy...")
    packages_to_reinstall = [
        "pandas",
        "matplotlib", 
        "scikit-learn",
        "scipy",
        "shap",
        "seaborn"
    ]
    
    for package in packages_to_reinstall:
        print(f"\nReinstalling {package}...")
        run_command(f"{sys.executable} -m pip install --force-reinstall --no-deps {package}")
    
    # Step 4: Test imports
    print("\n" + "=" * 50)
    print("Testing imports...")
    
    test_packages = ["numpy", "pandas", "matplotlib", "sklearn", "shap", "flask"]
    success_count = 0
    
    for package in test_packages:
        try:
            __import__(package)
            print(f"✓ {package} imports successfully")
            success_count += 1
        except ImportError as e:
            print(f"✗ {package} failed to import: {e}")
    
    print(f"\nSuccessfully imported: {success_count}/{len(test_packages)} packages")
    
    if success_count == len(test_packages):
        print("\nAll packages fixed! You can now run: python app3.py")
    else:
        print("\nSome packages still have issues. Try running this script again or use conda.")

if __name__ == "__main__":
    main()