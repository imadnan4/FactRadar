#!/usr/bin/env python3
"""
Quick Data Check for FactRadar
Check if the real dataset is properly loaded and ready for analysis.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def check_data_availability():
    """Check if the processed dataset is available"""
    print("🔍 FactRadar Data Availability Check")
    print("=" * 50)
    
    # Check for processed dataset
    processed_path = Path("../data/processed/real_dataset_processed.csv")
    
    if processed_path.exists():
        print("✅ Processed dataset found!")
        
        # Load and check the dataset
        df = pd.read_csv(processed_path)
        
        print(f"📊 Dataset Info:")
        print(f"   • Total samples: {len(df):,}")
        print(f"   • Columns: {list(df.columns)}")
        print(f"   • Shape: {df.shape}")
        
        # Check class distribution
        if 'label' in df.columns:
            class_counts = df['label'].value_counts()
            print(f"\n🎯 Class Distribution:")
            print(f"   • Real news (0): {class_counts.get(0, 0):,}")
            print(f"   • Fake news (1): {class_counts.get(1, 0):,}")
        
        # Check text columns
        text_columns = [col for col in df.columns if 'text' in col.lower()]
        print(f"\n📝 Text Columns: {text_columns}")
        
        # Show sample data
        print(f"\n📋 Sample Data:")
        print(df.head(2))
        
        return df
    
    else:
        print("❌ Processed dataset not found!")
        print("\n📋 To create the dataset:")
        print("1. Ensure datasets are in root directory:")
        print("   • fake-and-real-news-dataset/")
        print("   • liar_dataset/")
        print("2. Run: python load_real_datasets.py")
        return None

def check_raw_datasets():
    """Check if raw datasets are available"""
    print("\n🔍 Checking Raw Datasets...")
    
    # Check ISOT dataset
    isot_path = Path("../../fake-and-real-news-dataset")
    if isot_path.exists():
        fake_csv = isot_path / "Fake.csv"
        true_csv = isot_path / "True.csv"
        if fake_csv.exists() and true_csv.exists():
            print("✅ ISOT Dataset available")
            
            # Quick stats
            fake_df = pd.read_csv(fake_csv)
            true_df = pd.read_csv(true_csv)
            print(f"   • Fake articles: {len(fake_df):,}")
            print(f"   • Real articles: {len(true_df):,}")
        else:
            print("⚠️  ISOT Dataset directory found but CSV files missing")
    else:
        print("❌ ISOT Dataset not found")
    
    # Check LIAR dataset
    liar_path = Path("../../liar_dataset")
    if liar_path.exists():
        train_tsv = liar_path / "train.tsv"
        test_tsv = liar_path / "test.tsv"
        valid_tsv = liar_path / "valid.tsv"
        if all(f.exists() for f in [train_tsv, test_tsv, valid_tsv]):
            print("✅ LIAR Dataset available")
            
            # Quick stats
            train_df = pd.read_csv(train_tsv, sep='\t', header=None)
            test_df = pd.read_csv(test_tsv, sep='\t', header=None)
            valid_df = pd.read_csv(valid_tsv, sep='\t', header=None)
            total = len(train_df) + len(test_df) + len(valid_df)
            print(f"   • Total statements: {total:,}")
        else:
            print("⚠️  LIAR Dataset directory found but TSV files missing")
    else:
        print("❌ LIAR Dataset not found")

def create_simple_sample():
    """Create a simple sample for testing if no data is available"""
    print("\n🔧 Creating simple sample for testing...")
    
    # Create a minimal sample dataset
    sample_data = {
        'text': [
            "Scientists at NASA announce breakthrough in Mars exploration technology.",
            "SHOCKING! You won't believe this ONE WEIRD TRICK that doctors HATE!",
            "The Federal Reserve announced a 0.25% interest rate increase today.",
            "BREAKING: Celebrity reveals secret that Big Pharma doesn't want you to know!",
            "New research published in Nature shows promising results for cancer treatment.",
            "URGENT: Government plans to control your mind through 5G towers!"
        ],
        'label': [0, 1, 0, 1, 0, 1],  # 0 = real, 1 = fake
        'dataset': ['sample'] * 6
    }
    
    df = pd.DataFrame(sample_data)
    
    # Save sample
    output_dir = Path("../data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sample_path = output_dir / "sample_dataset.csv"
    df.to_csv(sample_path, index=False)
    
    print(f"✅ Sample dataset created: {sample_path}")
    print(f"   • {len(df)} samples for testing")
    
    return df

def main():
    """Main function"""
    # Check processed data
    df = check_data_availability()
    
    # Check raw datasets
    check_raw_datasets()
    
    # If no data available, create sample
    if df is None:
        print("\n⚠️  No processed data found. Creating sample for testing...")
        df = create_simple_sample()
    
    print("\n🎯 Next Steps:")
    if df is not None and len(df) > 1000:
        print("✅ You have real data! You can:")
        print("   1. Run data_exploration.ipynb")
        print("   2. Run preprocessing.ipynb")
        print("   3. Train models with real 56K+ dataset")
    else:
        print("📋 To get the full dataset:")
        print("   1. Download ISOT and LIAR datasets")
        print("   2. Run: python load_real_datasets.py")
        print("   3. Then run the Jupyter notebooks")
    
    print("\n🚀 Start Jupyter: jupyter notebook ../notebooks/")

if __name__ == "__main__":
    main()
