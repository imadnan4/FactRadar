#!/usr/bin/env python3
"""
FactRadar Project Setup Script
Automated setup for the complete FactRadar ML pipeline.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def print_header():
    """Print project header"""
    print("🔍 FactRadar Project Setup")
    print("=" * 50)
    print("AI-Powered Fake News Detection System")
    print("56K+ samples | NLTK + Scikit-learn + TensorFlow")
    print("=" * 50)

def check_python_version():
    """Check Python version"""
    print("\n🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required!")
        print(f"   Current version: {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_datasets():
    """Check if datasets are available"""
    print("\n📊 Checking datasets...")
    
    isot_path = Path("../../fake-and-real-news-dataset")
    liar_path = Path("../../liar_dataset")
    
    datasets_found = []
    
    if isot_path.exists():
        fake_csv = isot_path / "Fake.csv"
        true_csv = isot_path / "True.csv"
        if fake_csv.exists() and true_csv.exists():
            datasets_found.append("ISOT")
            print("✅ ISOT Dataset found (44K articles)")
        else:
            print("⚠️  ISOT Dataset directory found but CSV files missing")
    else:
        print("❌ ISOT Dataset not found")
    
    if liar_path.exists():
        train_tsv = liar_path / "train.tsv"
        test_tsv = liar_path / "test.tsv"
        valid_tsv = liar_path / "valid.tsv"
        if all(f.exists() for f in [train_tsv, test_tsv, valid_tsv]):
            datasets_found.append("LIAR")
            print("✅ LIAR Dataset found (12K statements)")
        else:
            print("⚠️  LIAR Dataset directory found but TSV files missing")
    else:
        print("❌ LIAR Dataset not found")
    
    if not datasets_found:
        print("\n📋 Dataset Setup Instructions:")
        print("1. Download ISOT Dataset:")
        print("   https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        print("   Extract to: fake-and-real-news-dataset/")
        print("\n2. Download LIAR Dataset:")
        print("   https://www.cs.ucsb.edu/~william/data/liar_dataset.zip")
        print("   Extract to: liar_dataset/")
        return False
    
    print(f"\n✅ Found {len(datasets_found)} dataset(s): {', '.join(datasets_found)}")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "../../requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("Please run manually: pip install -r requirements.txt")
        return False

def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    
    directories = [
        "../data/processed",
        "../data/processed/models",
        "../../models"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {dir_path}")

def run_data_processing():
    """Run the data processing pipeline"""
    print("\n🔄 Processing datasets...")
    
    try:
        # Run the dataset loading script
        result = subprocess.run([sys.executable, "load_real_datasets.py"], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Dataset processing completed!")
            return True
        else:
            print(f"❌ Dataset processing failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⚠️  Dataset processing taking longer than expected...")
        print("   You can run it manually: python load_real_datasets.py")
        return False
    except Exception as e:
        print(f"❌ Error running dataset processing: {e}")
        return False

def check_jupyter():
    """Check if Jupyter is available"""
    print("\n📓 Checking Jupyter...")
    
    try:
        subprocess.run(["jupyter", "--version"], 
                      check=True, capture_output=True, text=True)
        print("✅ Jupyter available!")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Jupyter not found. Installing...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "jupyter"], 
                          check=True, capture_output=True, text=True)
            print("✅ Jupyter installed!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install Jupyter")
            return False

def create_setup_summary():
    """Create setup summary"""
    print("\n📋 Creating setup summary...")
    
    summary = {
        "project": "FactRadar",
        "version": "1.0.0",
        "setup_completed": True,
        "components": {
            "datasets": "ISOT + LIAR (56K+ samples)",
            "ml_pipeline": "NLTK + Scikit-learn + TensorFlow",
            "frontend": "Next.js 15 + React 19",
            "features": "10,019 total features"
        },
        "next_steps": [
            "Run data_exploration.ipynb",
            "Run preprocessing.ipynb", 
            "Run model_training.ipynb",
            "Run model_conversion.ipynb",
            "Start frontend: npm run dev"
        ]
    }
    
    with open("../data/setup_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("✅ Setup summary saved!")

def main():
    """Main setup function"""
    print_header()
    
    # Check prerequisites
    if not check_python_version():
        return False
    
    # Setup directories
    setup_directories()
    
    # Check datasets
    datasets_available = check_datasets()
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Check Jupyter
    check_jupyter()
    
    # Process datasets if available
    if datasets_available:
        if run_data_processing():
            print("\n🎉 Dataset processing completed!")
        else:
            print("\n⚠️  Dataset processing had issues, but you can run it manually")
    
    # Create summary
    create_setup_summary()
    
    # Final instructions
    print("\n🎉 SETUP COMPLETED!")
    print("=" * 50)
    print("🚀 Next Steps:")
    print("1. Start Jupyter: jupyter notebook ../notebooks/")
    print("2. Run notebooks in order:")
    print("   • data_exploration.ipynb")
    print("   • preprocessing.ipynb")
    print("   • model_training.ipynb")
    print("   • model_conversion.ipynb")
    print("3. Start frontend: npm run dev")
    print("\n📖 See README.md for detailed instructions")
    print("🎯 Expected model accuracy: 85-95%")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
