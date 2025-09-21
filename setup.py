#!/usr/bin/env python3
"""
Setup script for BiasLens.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up BiasLens...")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create index directory
    index_dir = Path(__file__).parent / 'app' / 'index'
    index_dir.mkdir(exist_ok=True)
    print(f"✅ Created index directory: {index_dir}")
    
    # Build FAISS index
    if not run_command("python3 scripts/index_docs.py", "Building FAISS index"):
        print("❌ Failed to build FAISS index")
        sys.exit(1)
    
    # Run quick test
    print("\n🧪 Running quick test...")
    if run_command("python3 scripts/eval_sample.py", "Testing components"):
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Start the API server: python app/server.py")
        print("2. Visit http://localhost:8000/docs for API documentation")
        print("3. Test with: curl -X POST 'http://localhost:8000/analyze' -H 'Content-Type: application/json' -d '{\"q\": \"Your text here\"}'")
    else:
        print("\n⚠️  Setup completed with warnings. Some components may not work properly.")
        print("Check the error messages above for details.")


if __name__ == "__main__":
    main()
