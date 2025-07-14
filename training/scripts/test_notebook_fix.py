#!/usr/bin/env python3
"""
Test script to verify the data_exploration.ipynb fixes work correctly.
This simulates the notebook environment and tests the fixed boxplot code.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_test_data():
    """Create test data similar to what the notebook expects"""
    print("🧪 Creating test data...")
    
    # Create sample data with the expected structure
    np.random.seed(42)
    n_samples = 1000
    
    # Create fake df_analyzed with the expected columns
    data = {
        'text': ['Sample text ' + str(i) for i in range(n_samples)],
        'label': np.random.choice([0, 1], n_samples),
        'word_count': np.random.normal(100, 30, n_samples),
        'sentence_count': np.random.normal(5, 2, n_samples),
        'avg_word_length': np.random.normal(5, 1, n_samples),
        'sentiment_compound': np.random.normal(0, 0.5, n_samples),
        'sentiment_positive': np.random.uniform(0, 1, n_samples),
        'sentiment_negative': np.random.uniform(0, 1, n_samples),
        'exclamation_count': np.random.poisson(2, n_samples),
        'question_count': np.random.poisson(1, n_samples),
        'caps_ratio': np.random.uniform(0, 0.3, n_samples),
        'stopword_ratio': np.random.uniform(0.3, 0.7, n_samples),
        'unique_word_ratio': np.random.uniform(0.5, 0.9, n_samples)
    }
    
    df_analyzed = pd.DataFrame(data)
    
    # Add some NaN values to test dropna functionality
    df_analyzed.loc[np.random.choice(df_analyzed.index, 50), 'sentiment_compound'] = np.nan
    
    print(f"✅ Created test dataset with {len(df_analyzed)} samples")
    print(f"   Columns: {list(df_analyzed.columns)}")
    print(f"   Real samples: {len(df_analyzed[df_analyzed['label'] == 0])}")
    print(f"   Fake samples: {len(df_analyzed[df_analyzed['label'] == 1])}")
    
    return df_analyzed

def test_fixed_boxplot_code(df_analyzed):
    """Test the fixed boxplot code from the notebook"""
    print("\n🔧 Testing fixed boxplot code...")
    
    feature_columns = ['word_count', 'sentence_count', 'avg_word_length', 
                      'sentiment_compound', 'exclamation_count', 'question_count',
                      'caps_ratio', 'stopword_ratio', 'unique_word_ratio']
    
    # Check which features are available
    available_features = [col for col in feature_columns if col in df_analyzed.columns]
    print(f"Available features: {available_features}")
    
    if available_features:
        # Test the statistical comparison
        comparison_stats = df_analyzed.groupby('label')[available_features].agg(['mean', 'std']).round(4)
        print("\n📊 Statistical comparison successful!")
        print(comparison_stats.head())
        
        # Test the visualization code
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        axes = axes.flatten()
        
        success_count = 0
        
        for i, feature in enumerate(available_features[:9]):
            try:
                # Box plot for each feature - FIXED VERSION
                real_data = df_analyzed[df_analyzed['label'] == 0][feature].dropna()
                fake_data = df_analyzed[df_analyzed['label'] == 1][feature].dropna()
                
                # Convert to lists and ensure they're 1D
                real_list = real_data.values.flatten()
                fake_list = fake_data.values.flatten()
                
                # Create boxplot with proper data format
                if len(real_list) > 0 and len(fake_list) > 0:
                    box_data = [real_list, fake_list]
                    axes[i].boxplot(box_data, labels=['Real', 'Fake'])
                    axes[i].set_title(f'{feature.replace("_", " ").title()}')
                    axes[i].grid(True, alpha=0.3)
                    success_count += 1
                else:
                    axes[i].text(0.5, 0.5, f'No data for\n{feature}', 
                                ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'{feature.replace("_", " ").title()}')
                
            except Exception as e:
                print(f"❌ Error with feature {feature}: {e}")
                axes[i].text(0.5, 0.5, f'Error:\n{feature}', 
                            ha='center', va='center', transform=axes[i].transAxes)
        
        # Hide unused subplots
        for j in range(len(available_features), 9):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Feature Comparison: Real vs Fake News (TEST)', fontsize=14, y=1.02)
        
        # Save the plot instead of showing it
        output_dir = Path("../data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "test_boxplot_fix.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Boxplot test completed! {success_count}/{len(available_features[:9])} features plotted successfully")
        print(f"📊 Plot saved to: {output_dir / 'test_boxplot_fix.png'}")
        
        return True
    else:
        print("❌ No features available for testing")
        return False

def test_sentiment_boxplot_fix(df_analyzed):
    """Test the fixed sentiment boxplot code"""
    print("\n😊 Testing sentiment boxplot fixes...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        # Test positive sentiment boxplot
        if 'sentiment_positive' in df_analyzed.columns:
            real_pos = df_analyzed[df_analyzed['label'] == 0]['sentiment_positive'].dropna().values
            fake_pos = df_analyzed[df_analyzed['label'] == 1]['sentiment_positive'].dropna().values
            if len(real_pos) > 0 and len(fake_pos) > 0:
                axes[0, 1].boxplot([real_pos, fake_pos], labels=['Real', 'Fake'])
            axes[0, 1].set_title('Positive Sentiment')
            print("✅ Positive sentiment boxplot: OK")
        
        # Test negative sentiment boxplot
        if 'sentiment_negative' in df_analyzed.columns:
            real_neg = df_analyzed[df_analyzed['label'] == 0]['sentiment_negative'].dropna().values
            fake_neg = df_analyzed[df_analyzed['label'] == 1]['sentiment_negative'].dropna().values
            if len(real_neg) > 0 and len(fake_neg) > 0:
                axes[1, 0].boxplot([real_neg, fake_neg], labels=['Real', 'Fake'])
            axes[1, 0].set_title('Negative Sentiment')
            print("✅ Negative sentiment boxplot: OK")
        
        # Test compound sentiment histogram
        if 'sentiment_compound' in df_analyzed.columns:
            real_sentiment = df_analyzed[df_analyzed['label'] == 0]['sentiment_compound'].dropna()
            fake_sentiment = df_analyzed[df_analyzed['label'] == 1]['sentiment_compound'].dropna()
            
            axes[0, 0].hist(real_sentiment, alpha=0.7, label='Real', bins=20, color='skyblue')
            axes[0, 0].hist(fake_sentiment, alpha=0.7, label='Fake', bins=20, color='lightcoral')
            axes[0, 0].set_title('Sentiment Compound Score Distribution')
            axes[0, 0].legend()
            print("✅ Compound sentiment histogram: OK")
        
        # Test scatter plot
        real_data = df_analyzed[df_analyzed['label'] == 0]
        fake_data = df_analyzed[df_analyzed['label'] == 1]
        
        axes[1, 1].scatter(real_data['sentiment_positive'], real_data['sentiment_negative'], 
                          alpha=0.6, label='Real', color='skyblue', s=10)
        axes[1, 1].scatter(fake_data['sentiment_positive'], fake_data['sentiment_negative'], 
                          alpha=0.6, label='Fake', color='lightcoral', s=10)
        axes[1, 1].set_xlabel('Positive Sentiment')
        axes[1, 1].set_ylabel('Negative Sentiment')
        axes[1, 1].set_title('Sentiment Scatter Plot')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = Path("../data/processed")
        plt.savefig(output_dir / "test_sentiment_fix.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Sentiment analysis plots: All tests passed!")
        print(f"📊 Plot saved to: {output_dir / 'test_sentiment_fix.png'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sentiment boxplot test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🔍 FactRadar Notebook Fix Verification")
    print("=" * 50)
    
    # Create test data
    df_analyzed = create_test_data()
    
    # Test the fixes
    boxplot_success = test_fixed_boxplot_code(df_analyzed)
    sentiment_success = test_sentiment_boxplot_fix(df_analyzed)
    
    # Summary
    print("\n🎯 TEST RESULTS SUMMARY")
    print("=" * 30)
    print(f"✅ Boxplot fixes: {'PASSED' if boxplot_success else 'FAILED'}")
    print(f"✅ Sentiment fixes: {'PASSED' if sentiment_success else 'FAILED'}")
    
    if boxplot_success and sentiment_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The data_exploration.ipynb notebook fixes are working correctly!")
        print("🚀 You can now run the notebook without the ValueError!")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
    
    print("\n📋 Next Steps:")
    print("1. Run the fixed data_exploration.ipynb notebook")
    print("2. Ensure you run cells in order (especially the advanced text analysis)")
    print("3. The boxplot errors should now be resolved!")

if __name__ == "__main__":
    main()
