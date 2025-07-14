#!/usr/bin/env python3
"""
Real Dataset Loading and Processing
Load and process the actual ISOT and LIAR datasets for FactRadar training.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealDatasetLoader:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.isot_path = self.project_root / "fake-and-real-news-dataset"
        self.liar_path = self.project_root / "liar_dataset"
        self.output_path = self.project_root / "training" / "data" / "processed"
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"ISOT dataset path: {self.isot_path}")
        logger.info(f"LIAR dataset path: {self.liar_path}")
        logger.info(f"Output path: {self.output_path}")

    def load_isot_dataset(self):
        """Load and process ISOT Fake and Real News Dataset"""
        logger.info("Loading ISOT dataset...")
        
        # Load fake news
        fake_path = self.isot_path / "Fake.csv"
        true_path = self.isot_path / "True.csv"
        
        if not fake_path.exists() or not true_path.exists():
            logger.error(f"ISOT dataset files not found!")
            return None
        
        # Load datasets
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)
        
        logger.info(f"Loaded {len(fake_df)} fake news articles")
        logger.info(f"Loaded {len(true_df)} real news articles")
        
        # Add labels
        fake_df['label'] = 1  # 1 = fake
        true_df['label'] = 0  # 0 = real
        
        # Add source info
        fake_df['dataset'] = 'isot_fake'
        true_df['dataset'] = 'isot_real'
        
        # Combine datasets
        isot_df = pd.concat([fake_df, true_df], ignore_index=True)
        
        # Create unified text column (title + text)
        isot_df['full_text'] = isot_df['title'].fillna('') + ' ' + isot_df['text'].fillna('')
        
        # Select relevant columns
        isot_processed = isot_df[['full_text', 'label', 'subject', 'date', 'dataset']].copy()
        isot_processed.rename(columns={'full_text': 'text'}, inplace=True)
        
        logger.info(f"ISOT dataset processed: {len(isot_processed)} total articles")
        logger.info(f"Class distribution: {isot_processed['label'].value_counts().to_dict()}")
        
        return isot_processed

    def load_liar_dataset(self):
        """Load and process LIAR dataset"""
        logger.info("Loading LIAR dataset...")
        
        # Load train, test, and validation sets
        train_path = self.liar_path / "train.tsv"
        test_path = self.liar_path / "test.tsv"
        valid_path = self.liar_path / "valid.tsv"
        
        if not all([train_path.exists(), test_path.exists(), valid_path.exists()]):
            logger.error("LIAR dataset files not found!")
            return None
        
        # Column names based on README
        columns = [
            'id', 'label', 'statement', 'subject', 'speaker', 'job_title',
            'state', 'party', 'barely_true_count', 'false_count', 'half_true_count',
            'mostly_true_count', 'pants_fire_count', 'context'
        ]
        
        # Load datasets
        train_df = pd.read_csv(train_path, sep='\t', names=columns, header=None)
        test_df = pd.read_csv(test_path, sep='\t', names=columns, header=None)
        valid_df = pd.read_csv(valid_path, sep='\t', names=columns, header=None)
        
        logger.info(f"Loaded LIAR train: {len(train_df)} statements")
        logger.info(f"Loaded LIAR test: {len(test_df)} statements")
        logger.info(f"Loaded LIAR valid: {len(valid_df)} statements")
        
        # Combine all splits
        liar_df = pd.concat([train_df, test_df, valid_df], ignore_index=True)
        
        # Convert multi-class labels to binary
        # false, barely-true, pants-on-fire -> 1 (fake)
        # half-true, mostly-true, true -> 0 (real)
        fake_labels = ['false', 'barely-true', 'pants-on-fire']
        real_labels = ['half-true', 'mostly-true', 'true']
        
        liar_df['binary_label'] = liar_df['label'].apply(
            lambda x: 1 if x in fake_labels else (0 if x in real_labels else -1)
        )
        
        # Remove any rows with invalid labels
        liar_df = liar_df[liar_df['binary_label'] != -1].copy()
        
        # Add dataset info
        liar_df['dataset'] = 'liar'
        
        # Select relevant columns
        liar_processed = liar_df[['statement', 'binary_label', 'subject', 'speaker', 'dataset']].copy()
        liar_processed.rename(columns={'statement': 'text', 'binary_label': 'label'}, inplace=True)
        
        logger.info(f"LIAR dataset processed: {len(liar_processed)} total statements")
        logger.info(f"Class distribution: {liar_processed['label'].value_counts().to_dict()}")
        
        return liar_processed

    def clean_and_preprocess_text(self, df):
        """Apply text cleaning and preprocessing"""
        logger.info("Applying text preprocessing...")
        
        def clean_text(text):
            if pd.isna(text):
                return ""
            
            text = str(text)
            
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove excessive punctuation
            text = re.sub(r'[!]{2,}', '!', text)
            text = re.sub(r'[?]{2,}', '?', text)
            text = re.sub(r'[.]{3,}', '...', text)
            
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)
            
            return text.strip()
        
        # Apply cleaning
        df['cleaned_text'] = df['text'].apply(clean_text)
        
        # Extract basic features
        def extract_features(text):
            if pd.isna(text):
                return pd.Series([0, 0, 0, 0, 0, 0])
            
            text = str(text)
            words = text.split()
            
            word_count = len(words)
            char_count = len(text)
            exclamation_count = text.count('!')
            question_count = text.count('?')
            caps_count = sum(1 for c in text if c.isupper())
            caps_ratio = caps_count / char_count if char_count > 0 else 0
            
            return pd.Series([word_count, char_count, exclamation_count, question_count, caps_count, caps_ratio])
        
        # Extract features
        feature_columns = ['word_count', 'char_count', 'exclamation_count', 'question_count', 'caps_count', 'caps_ratio']
        df[feature_columns] = df['text'].apply(extract_features)
        
        logger.info("Text preprocessing completed!")
        return df

    def combine_and_save_datasets(self):
        """Load, process, and combine all datasets"""
        logger.info("Starting comprehensive dataset processing...")
        
        # Load datasets
        isot_df = self.load_isot_dataset()
        liar_df = self.load_liar_dataset()
        
        if isot_df is None or liar_df is None:
            logger.error("Failed to load datasets!")
            return False
        
        # Combine datasets
        logger.info("Combining datasets...")
        
        # Ensure both have same columns
        common_columns = ['text', 'label', 'dataset']
        
        isot_subset = isot_df[common_columns].copy()
        liar_subset = liar_df[common_columns].copy()
        
        # Combine
        combined_df = pd.concat([isot_subset, liar_subset], ignore_index=True)
        
        logger.info(f"Combined dataset: {len(combined_df)} total samples")
        logger.info(f"ISOT samples: {len(isot_subset)}")
        logger.info(f"LIAR samples: {len(liar_subset)}")
        
        # Apply preprocessing
        combined_df = self.clean_and_preprocess_text(combined_df)
        
        # Final statistics
        logger.info("\n" + "="*50)
        logger.info("FINAL DATASET STATISTICS")
        logger.info("="*50)
        logger.info(f"Total samples: {len(combined_df)}")
        logger.info(f"Real news (label=0): {len(combined_df[combined_df['label'] == 0])}")
        logger.info(f"Fake news (label=1): {len(combined_df[combined_df['label'] == 1])}")
        
        # Dataset breakdown
        dataset_stats = combined_df.groupby(['dataset', 'label']).size().unstack(fill_value=0)
        logger.info(f"\nDataset breakdown:")
        logger.info(dataset_stats)
        
        # Text statistics
        logger.info(f"\nText statistics:")
        logger.info(f"Average word count: {combined_df['word_count'].mean():.1f}")
        logger.info(f"Average character count: {combined_df['char_count'].mean():.1f}")
        logger.info(f"Average exclamation marks: {combined_df['exclamation_count'].mean():.2f}")
        
        # Save processed dataset
        output_file = self.output_path / "real_dataset_processed.csv"
        combined_df.to_csv(output_file, index=False)
        
        logger.info(f"\n✅ Dataset saved to: {output_file}")
        
        # Save metadata
        metadata = {
            "total_samples": len(combined_df),
            "real_samples": len(combined_df[combined_df['label'] == 0]),
            "fake_samples": len(combined_df[combined_df['label'] == 1]),
            "datasets": {
                "isot": len(isot_subset),
                "liar": len(liar_subset)
            },
            "features": list(combined_df.columns),
            "text_stats": {
                "avg_word_count": float(combined_df['word_count'].mean()),
                "avg_char_count": float(combined_df['char_count'].mean()),
                "avg_exclamation_count": float(combined_df['exclamation_count'].mean())
            }
        }
        
        import json
        metadata_file = self.output_path / "real_dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"📊 Metadata saved to: {metadata_file}")
        logger.info("\n🎉 Real dataset processing completed successfully!")
        
        return True

def main():
    """Main function"""
    print("🔄 FactRadar Real Dataset Processing")
    print("=" * 50)
    
    loader = RealDatasetLoader()
    success = loader.combine_and_save_datasets()
    
    if success:
        print("\n✅ SUCCESS! Real datasets processed and ready for training!")
        print("\n🚀 Next steps:")
        print("   1. Run data_exploration.ipynb with real data")
        print("   2. Train models with significantly more data")
        print("   3. Achieve much higher accuracy with real datasets")
        print("   4. Convert best model to TensorFlow.js")
    else:
        print("\n❌ Failed to process datasets!")

if __name__ == "__main__":
    main()
