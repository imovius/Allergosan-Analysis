#!/usr/bin/env python3
"""
SPARSITY ANALYSIS FOR QUIZ FEATURES
Quick assessment of data sparsity patterns before full feature selection

Author: Ian Movius
Date: January 2025
"""

import csv
from collections import defaultdict, Counter

def analyze_data_sparsity(filename='raw_data_v2.csv'):
    """Analyze sparsity patterns in the dataset"""
    print("="*80)
    print("DATA SPARSITY ANALYSIS")
    print("="*80)
    
    # Load data
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        data = list(reader)
    
    print(f"Dataset: {len(data):,} rows, {len(headers)} columns")
    
    # Analyze each column
    sparsity_results = {}
    
    for col in headers:
        non_null_count = 0
        null_count = 0
        unique_values = set()
        
        for row in data:
            value = row.get(col, '')
            
            if value and value.strip() and value.strip().lower() not in ['', 'null', 'nan', 'none']:
                non_null_count += 1
                unique_values.add(value.strip())
            else:
                null_count += 1
        
        sparsity_percent = (null_count / len(data)) * 100
        fill_percent = (non_null_count / len(data)) * 100
        
        sparsity_results[col] = {
            'non_null_count': non_null_count,
            'null_count': null_count,
            'sparsity_percent': sparsity_percent,
            'fill_percent': fill_percent,
            'unique_values': len(unique_values),
            'sample_values': list(unique_values)[:5] if unique_values else []
        }
    
    # Sort by sparsity (most sparse first)
    sorted_by_sparsity = sorted(sparsity_results.items(), 
                               key=lambda x: x[1]['sparsity_percent'], 
                               reverse=True)
    
    print("\n" + "="*60)
    print("SPARSITY ANALYSIS RESULTS")
    print("="*60)
    print(f"{'Column':<30} {'Fill%':<8} {'Sparse%':<8} {'Unique':<8} {'Sample Values'}")
    print("-" * 85)
    
    for col, stats in sorted_by_sparsity:
        sample_str = str(stats['sample_values'])[:40] + "..." if len(str(stats['sample_values'])) > 40 else str(stats['sample_values'])
        print(f"{col:<30} {stats['fill_percent']:<8.1f} {stats['sparsity_percent']:<8.1f} {stats['unique_values']:<8} {sample_str}")
    
    # Focus on quiz-related features
    print("\n" + "="*60)
    print("QUIZ-RELATED FEATURES ANALYSIS")
    print("="*60)
    
    quiz_keywords = ['quiz', 'stress', 'pregnant', 'trimester', 'mental', 'physical', 
                     'probiotic', 'child', 'bm_pattern', 'gi_symptom', 'primary_goal']
    
    quiz_features = []
    for col, stats in sparsity_results.items():
        if any(keyword in col.lower() for keyword in quiz_keywords):
            quiz_features.append((col, stats))
    
    if quiz_features:
        print(f"Found {len(quiz_features)} quiz-related features:")
        print(f"{'Feature':<35} {'Fill%':<8} {'Unique':<8} {'Values'}")
        print("-" * 70)
        
        for col, stats in quiz_features:
            sample_str = str(stats['sample_values'])[:25] + "..." if len(str(stats['sample_values'])) > 25 else str(stats['sample_values'])
            print(f"{col:<35} {stats['fill_percent']:<8.1f} {stats['unique_values']:<8} {sample_str}")
    
    # Analyze ancestor_discount_code specifically
    print("\n" + "="*60)
    print("ANCESTOR_DISCOUNT_CODE ANALYSIS")
    print("="*60)
    
    if 'ancestor_discount_code' in sparsity_results:
        adc_stats = sparsity_results['ancestor_discount_code']
        print(f"Fill rate: {adc_stats['fill_percent']:.1f}%")
        print(f"Unique codes: {adc_stats['unique_values']}")
        print(f"Sample codes: {adc_stats['sample_values']}")
        
        # Count frequency of each discount code
        discount_counts = Counter()
        for row in data:
            code = row.get('ancestor_discount_code', '').strip()
            if code and code.lower() not in ['', 'null', 'nan', 'none']:
                discount_counts[code] += 1
        
        print(f"\nTop 10 discount codes:")
        for code, count in discount_counts.most_common(10):
            print(f"  {code}: {count} customers ({count/len(data)*100:.1f}%)")
    
    # Sparsity recommendations
    print("\n" + "="*60)
    print("SPARSITY RECOMMENDATIONS")
    print("="*60)
    
    very_sparse = [(col, stats) for col, stats in sparsity_results.items() 
                   if stats['sparsity_percent'] > 90]
    
    moderately_sparse = [(col, stats) for col, stats in sparsity_results.items() 
                        if 50 <= stats['sparsity_percent'] <= 90]
    
    well_filled = [(col, stats) for col, stats in sparsity_results.items() 
                   if stats['sparsity_percent'] < 50]
    
    print(f"🔴 VERY SPARSE (>90% missing): {len(very_sparse)} features")
    if very_sparse:
        for col, stats in very_sparse[:5]:
            print(f"   - {col}: {stats['sparsity_percent']:.1f}% missing")
    
    print(f"\n🟡 MODERATELY SPARSE (50-90% missing): {len(moderately_sparse)} features")
    if moderately_sparse:
        for col, stats in moderately_sparse[:5]:
            print(f"   - {col}: {stats['sparsity_percent']:.1f}% missing")
    
    print(f"\n🟢 WELL FILLED (<50% missing): {len(well_filled)} features")
    if well_filled:
        for col, stats in well_filled[:5]:
            print(f"   - {col}: {stats['sparsity_percent']:.1f}% missing")
    
    # Statistical power analysis for sparse features
    print("\n" + "="*60)
    print("STATISTICAL POWER ASSESSMENT")
    print("="*60)
    
    total_samples = len(data)
    
    print(f"Total samples: {total_samples:,}")
    print(f"Minimum samples for statistical power:")
    print(f"  - Weak effect detection: ~{max(100, int(total_samples * 0.05)):,} samples")
    print(f"  - Moderate effect detection: ~{max(50, int(total_samples * 0.02)):,} samples") 
    print(f"  - Strong effect detection: ~{max(30, int(total_samples * 0.01)):,} samples")
    
    # Assess quiz features for statistical power
    quiz_power_assessment = []
    for col, stats in quiz_features:
        if stats['non_null_count'] >= max(30, int(total_samples * 0.01)):
            power_level = "🟢 GOOD"
        elif stats['non_null_count'] >= max(50, int(total_samples * 0.02)):
            power_level = "🟡 MODERATE"
        else:
            power_level = "🔴 LOW"
        
        quiz_power_assessment.append((col, stats['non_null_count'], power_level))
    
    if quiz_power_assessment:
        print(f"\nQuiz features statistical power:")
        for col, count, power in quiz_power_assessment:
            print(f"  {col:<35} {count:>6} samples {power}")
    
    return sparsity_results, quiz_features, very_sparse

if __name__ == "__main__":
    analyze_data_sparsity()