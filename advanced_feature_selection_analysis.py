#!/usr/bin/env python3
"""
EXPERT-LEVEL STATISTICAL FEATURE SELECTION FOR CUSTOMER SEGMENTATION
Comprehensive data science approach using multiple statistical methods for optimal feature selection

Author: Ian Movius  
Date: January 2025
Methodology: Multi-method statistical feature selection with rigorous validation
"""

import csv
import math
import random
import numpy as np
from collections import defaultdict, Counter
import datetime
from typing import List, Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureSelector:
    """Expert-level feature selection using multiple statistical methodologies"""
    
    def __init__(self, data_file: str = 'raw_data_v2.csv'):
        self.data_file = data_file
        self.raw_data = []
        self.feature_matrix = []
        self.feature_names = []
        self.categorical_features = []
        self.numerical_features = []
        self.target_variable = None
        self.results = {}
        self.encoded_features = {}
        
    def load_and_parse_data(self):
        """Load raw data and perform comprehensive feature engineering"""
        print("="*80)
        print("ADVANCED FEATURE SELECTION ANALYSIS")
        print("="*80)
        print("Loading and parsing comprehensive dataset...")
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            print(f">> Found {len(header)} raw columns in dataset")
            
            for row in reader:
                self.raw_data.append(row)
        
        print(f">> Loaded {len(self.raw_data):,} customer records")
        print(f">> Raw features: {len(header)}")
        
        # Parse and engineer features
        self._engineer_comprehensive_features()
        self._detect_feature_types()
        self._create_target_variables()
        
        print(f">> Engineered features: {len(self.feature_names)}")
        print(f"   - Numerical: {len(self.numerical_features)}")
        print(f"   - Categorical: {len(self.categorical_features)}")
        
    def _engineer_comprehensive_features(self):
        """Create comprehensive feature set from raw data"""
        print("\n" + "="*60)
        print("COMPREHENSIVE FEATURE ENGINEERING")
        print("="*60)
        
        engineered_data = []
        
        for i, row in enumerate(self.raw_data):
            try:
                customer = {}
                
                # === TRANSACTIONAL FEATURES ===
                customer['customer_id'] = row.get('customer_id', '')
                customer['net_ltv'] = self._safe_float(row.get('net_ltv', 0))
                customer['gross_ltv'] = self._safe_float(row.get('gross_ltv', 0))
                customer['avg_order_value'] = self._safe_float(row.get('avg_order_value', 0))
                customer['order_count'] = self._safe_int(row.get('order_count', 0))
                customer['days_since_last_order'] = self._safe_int(row.get('days_since_last_order', 0))
                customer['refund_count'] = self._safe_int(row.get('refund_count', 0))
                customer['refund_amt'] = self._safe_float(row.get('refund_amt', 0))
                customer['avg_days_between_orders'] = self._safe_float(row.get('avg_days_between_orders', 0))
                customer['shipping_collected'] = self._safe_float(row.get('shipping_collected', 0))
                customer['shipping_spend'] = self._safe_float(row.get('shipping_spend', 0))
                customer['total_cogs'] = self._safe_float(row.get('total_cogs', 0))
                
                # === ACQUISITION & MARKETING ===
                customer['acquisition_channel'] = row.get('acquisition_channel', 'Unknown').strip()
                customer['affiliate_segment'] = row.get('affiliate_segment', 'Unknown').strip()
                customer['ancestor_discount_code'] = row.get('ancestor_discount_code', 'None').strip()
                
                # === QUIZ & HEALTH FEATURES ===
                customer['quiz_taker'] = row.get('quiz_taker', '').strip().lower() == 'yes'
                customer['is_male'] = self._safe_bool(row.get('is_male', ''))
                customer['is_pregnant'] = self._safe_bool(row.get('is_pregnant', ''))
                customer['in_third_trimester_flag'] = self._safe_bool(row.get('in_third_trimester_flag', ''))
                customer['probiotic_for_child_flag'] = self._safe_bool(row.get('probiotic_for_child_flag', ''))
                customer['stress_mental_flag'] = self._safe_bool(row.get('stress_mental_flag', ''))
                customer['stress_physical_flag'] = self._safe_bool(row.get('stress_physical_flag', ''))
                customer['bm_pattern'] = row.get('bm_pattern', 'Unknown').strip()
                customer['gi_symptom_cat'] = row.get('gi_symptom_cat', 'Unknown').strip()
                customer['primary_goal'] = row.get('primary_goal', 'Unknown').strip()
                
                # === DERIVED BEHAVIORAL FEATURES ===
                first_date = row.get('first_order_date', '')
                last_date = row.get('last_order_date', '')
                
                # Tenure calculation
                if first_date and len(first_date) >= 10:
                    try:
                        first_order = datetime.datetime.strptime(first_date[:19], '%Y-%m-%d %H:%M:%S')
                        today = datetime.datetime.now()
                        tenure_days = (today - first_order).days
                        customer['tenure_days'] = max(1, tenure_days)
                        customer['tenure_months'] = max(1, tenure_days / 30.44)
                    except:
                        customer['tenure_days'] = 365
                        customer['tenure_months'] = 12
                else:
                    customer['tenure_days'] = 365
                    customer['tenure_months'] = 12
                
                # RFM Traditional Features
                customer['ltv_per_month'] = customer['net_ltv'] / customer['tenure_months']
                customer['order_frequency'] = customer['order_count'] / customer['tenure_months']
                customer['recency_score'] = 1 / (1 + customer['days_since_last_order'] / 365)
                customer['purchase_intensity'] = customer['order_count'] / customer['tenure_months']
                
                # Advanced Behavioral Features
                customer['refund_ratio'] = customer['refund_amt'] / max(1, customer['gross_ltv'])
                customer['margin_ratio'] = (customer['net_ltv'] - customer['total_cogs']) / max(1, customer['net_ltv'])
                customer['shipping_ratio'] = customer['shipping_collected'] / max(1, customer['net_ltv'])
                customer['repeat_customer'] = 1 if customer['order_count'] > 1 else 0
                customer['high_value_flag'] = 1 if customer['net_ltv'] > 500 else 0
                customer['recent_customer'] = 1 if customer['days_since_last_order'] < 30 else 0
                customer['churn_risk'] = 1 if customer['days_since_last_order'] > 180 else 0
                
                # Discount Code Features
                has_discount = customer['ancestor_discount_code'] not in ['None', '', 'Unknown']
                customer['used_discount_code'] = 1 if has_discount else 0
                
                # Quiz Interaction Features (focus on these per stakeholder request)
                if customer['quiz_taker']:
                    customer['quiz_health_complexity'] = sum([
                        customer['stress_mental_flag'],
                        customer['stress_physical_flag'],
                        customer['is_pregnant'],
                        customer['in_third_trimester_flag']
                    ])
                    customer['quiz_special_population'] = 1 if (customer['is_pregnant'] or 
                                                                customer['probiotic_for_child_flag']) else 0
                else:
                    customer['quiz_health_complexity'] = 0
                    customer['quiz_special_population'] = 0
                
                engineered_data.append(customer)
                
            except Exception as e:
                print(f"Warning: Error processing row {i}: {e}")
                continue
        
        self.raw_data = engineered_data
        print(f">> Successfully engineered {len(engineered_data):,} customer records")
        
    def _detect_feature_types(self):
        """Automatically detect and categorize feature types"""
        print("\n" + "="*60)
        print("FEATURE TYPE DETECTION")
        print("="*60)
        
        if not self.raw_data:
            return
        
        sample_record = self.raw_data[0]
        
        # Skip identifier columns
        skip_features = {'customer_id', 'quiz_date', 'first_order_date', 'last_order_date'}
        
        for feature_name, value in sample_record.items():
            if feature_name in skip_features:
                continue
                
            # Determine if categorical or numerical
            unique_values = set()
            non_null_count = 0
            
            for record in self.raw_data[:1000]:  # Sample for efficiency
                val = record.get(feature_name)
                if val is not None and val != '' and val != 'Unknown':
                    unique_values.add(val)
                    non_null_count += 1
                if len(unique_values) > 50:  # If too many unique values, likely numerical
                    break
            
            # Classification logic
            if isinstance(value, (int, float)) and len(unique_values) > 10:
                self.numerical_features.append(feature_name)
            elif len(unique_values) <= 50 and non_null_count > 10:
                self.categorical_features.append(feature_name)
            elif isinstance(value, bool) or len(unique_values) <= 5:
                self.categorical_features.append(feature_name)
            else:
                self.numerical_features.append(feature_name)
        
        print(f">> Detected {len(self.numerical_features)} numerical features")
        print(f">> Detected {len(self.categorical_features)} categorical features")
        
        # Print feature categories for validation
        print(f"\n>> Numerical features: {self.numerical_features[:10]}{'...' if len(self.numerical_features) > 10 else ''}")
        print(f">> Categorical features: {self.categorical_features[:10]}{'...' if len(self.categorical_features) > 10 else ''}")
        
    def _create_target_variables(self):
        """Create multiple target variables for feature selection"""
        print("\n" + "="*60)
        print("TARGET VARIABLE CREATION")
        print("="*60)
        
        # Primary target: Customer Value Segments (based on LTV quintiles)
        ltv_values = [customer['net_ltv'] for customer in self.raw_data if customer['net_ltv'] > 0]
        ltv_values.sort()
        
        if len(ltv_values) >= 5:
            n = len(ltv_values)
            quintile_size = n // 5
            
            q1_threshold = ltv_values[quintile_size - 1]
            q2_threshold = ltv_values[2 * quintile_size - 1]
            q3_threshold = ltv_values[3 * quintile_size - 1]
            q4_threshold = ltv_values[4 * quintile_size - 1]
            
            print(f">> LTV Quintile Thresholds:")
            print(f"   Q1: ${q1_threshold:.2f}")
            print(f"   Q2: ${q2_threshold:.2f}")
            print(f"   Q3: ${q3_threshold:.2f}")
            print(f"   Q4: ${q4_threshold:.2f}")
            
            # Assign value segments
            for customer in self.raw_data:
                ltv = customer['net_ltv']
                if ltv <= q1_threshold:
                    customer['value_segment'] = 1
                elif ltv <= q2_threshold:
                    customer['value_segment'] = 2
                elif ltv <= q3_threshold:
                    customer['value_segment'] = 3
                elif ltv <= q4_threshold:
                    customer['value_segment'] = 4
                else:
                    customer['value_segment'] = 5
        
        # Additional target variables
        for customer in self.raw_data:
            customer['high_value_customer'] = 1 if customer['net_ltv'] > 500 else 0
            customer['frequent_buyer'] = 1 if customer['order_count'] >= 3 else 0
            customer['quiz_taker_int'] = 1 if customer['quiz_taker'] else 0
            
        self.target_variable = 'value_segment'
        print(f">> Primary target variable: {self.target_variable}")
        
    def encode_categorical_features(self):
        """Advanced categorical encoding with multiple methods"""
        print("\n" + "="*60)
        print("CATEGORICAL FEATURE ENCODING")
        print("="*60)
        
        encoded_data = []
        encoding_info = {}
        
        for customer in self.raw_data:
            encoded_customer = {}
            
            # Copy numerical features as-is
            for feature in self.numerical_features:
                if feature in customer:
                    encoded_customer[feature] = customer[feature]
            
            # Encode categorical features
            for feature in self.categorical_features:
                if feature in customer:
                    value = customer[feature]
                    
                    # Target encoding for high-cardinality categoricals
                    if feature in ['acquisition_channel', 'affiliate_segment', 'ancestor_discount_code']:
                        encoded_customer[f'{feature}_target_encoded'] = self._target_encode(feature, value)
                    
                    # One-hot encoding for low-cardinality categoricals
                    elif feature in ['bm_pattern', 'gi_symptom_cat', 'primary_goal']:
                        if feature not in encoding_info:
                            encoding_info[feature] = self._get_unique_values(feature)
                        
                        for unique_val in encoding_info[feature]:
                            encoded_customer[f'{feature}_{unique_val}'] = 1 if value == unique_val else 0
                    
                    # Binary encoding for boolean features
                    else:
                        encoded_customer[feature] = 1 if value else 0
            
            # Add target variable
            encoded_customer[self.target_variable] = customer[self.target_variable]
            encoded_data.append(encoded_customer)
        
        # Update feature lists
        self.raw_data = encoded_data
        self._update_feature_names_after_encoding()
        
        print(f">> Encoded features: {len(self.feature_names)}")
        print(f">> Final dataset shape: {len(self.raw_data)} x {len(self.feature_names)}")
        
    def _target_encode(self, feature: str, value: str) -> float:
        """Target encoding for categorical variables"""
        if not hasattr(self, '_target_encoding_cache'):
            self._target_encoding_cache = {}
            
        cache_key = f"{feature}_{value}"
        if cache_key in self._target_encoding_cache:
            return self._target_encoding_cache[cache_key]
        
        # Calculate mean target value for this category
        category_targets = []
        overall_targets = []
        
        for customer in self.raw_data:
            if customer.get(feature) == value:
                category_targets.append(customer.get('net_ltv', 0))
            overall_targets.append(customer.get('net_ltv', 0))
        
        if len(category_targets) == 0:
            encoded_value = sum(overall_targets) / len(overall_targets) if overall_targets else 0
        else:
            # Smoothing to prevent overfitting
            category_mean = sum(category_targets) / len(category_targets)
            overall_mean = sum(overall_targets) / len(overall_targets)
            weight = min(len(category_targets) / 100, 1.0)  # More weight with more samples
            encoded_value = weight * category_mean + (1 - weight) * overall_mean
        
        self._target_encoding_cache[cache_key] = encoded_value
        return encoded_value
    
    def _get_unique_values(self, feature: str) -> List[str]:
        """Get unique values for a categorical feature"""
        unique_values = set()
        for customer in self.raw_data:
            value = customer.get(feature)
            if value and value != 'Unknown':
                unique_values.add(str(value).replace(' ', '_').replace('-', '_'))
        return sorted(list(unique_values))[:10]  # Limit to top 10 to prevent explosion
    
    def _update_feature_names_after_encoding(self):
        """Update feature names after encoding"""
        if not self.raw_data:
            return
            
        sample_record = self.raw_data[0]
        self.feature_names = [key for key in sample_record.keys() 
                             if key != self.target_variable and key != 'customer_id']
    
    def create_feature_matrix(self):
        """Create numerical feature matrix for analysis"""
        print("\n" + "="*60)
        print("FEATURE MATRIX CREATION")
        print("="*60)
        
        self.feature_matrix = []
        target_values = []
        
        for customer in self.raw_data:
            row = []
            for feature_name in self.feature_names:
                value = customer.get(feature_name, 0)
                # Handle any remaining non-numerical values
                if isinstance(value, (int, float)):
                    row.append(float(value))
                else:
                    row.append(0.0)
            
            self.feature_matrix.append(row)
            target_values.append(customer.get(self.target_variable, 1))
        
        print(f">> Feature matrix shape: {len(self.feature_matrix)} x {len(self.feature_names)}")
        print(f">> Target distribution: {Counter(target_values)}")
        
        # Store target values
        self.target_values = target_values
        
    def calculate_correlation_analysis(self):
        """Advanced correlation analysis with target variable"""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        if not self.feature_matrix:
            return
            
        n_features = len(self.feature_names)
        n_samples = len(self.feature_matrix)
        
        # Calculate correlation with target variable
        target_correlations = []
        
        for i, feature_name in enumerate(self.feature_names):
            feature_values = [row[i] for row in self.feature_matrix]
            
            # Pearson correlation with target
            correlation = self._calculate_correlation(feature_values, self.target_values)
            target_correlations.append((feature_name, abs(correlation), correlation))
        
        # Sort by absolute correlation
        target_correlations.sort(key=lambda x: x[1], reverse=True)
        
        # Store results
        self.results['target_correlations'] = target_correlations
        
        print(f">> Top 15 Features by Target Correlation:")
        print(f"{'Feature':<35} {'|Correlation|':<15} {'Correlation':<15}")
        print("-" * 65)
        
        for feature_name, abs_corr, corr in target_correlations[:15]:
            print(f"{feature_name:<35} {abs_corr:<15.4f} {corr:<15.4f}")
        
        # Feature-feature correlation matrix (for multicollinearity detection)
        self._calculate_feature_intercorrelations()
        
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x) != len(y) or len(x) == 0:
            return 0.0
            
        n = len(x)
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        x_var = sum((x[i] - x_mean) ** 2 for i in range(n))
        y_var = sum((y[i] - y_mean) ** 2 for i in range(n))
        
        denominator = math.sqrt(x_var * y_var)
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def _calculate_feature_intercorrelations(self):
        """Calculate correlation matrix between features"""
        print(f"\n>> Calculating feature intercorrelations...")
        
        n_features = len(self.feature_names)
        high_correlations = []
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                feature1_values = [row[i] for row in self.feature_matrix]
                feature2_values = [row[j] for row in self.feature_matrix]
                
                correlation = self._calculate_correlation(feature1_values, feature2_values)
                
                if abs(correlation) > 0.7:  # High correlation threshold
                    high_correlations.append((
                        self.feature_names[i], 
                        self.feature_names[j], 
                        correlation
                    ))
        
        self.results['high_intercorrelations'] = high_correlations
        
        if high_correlations:
            print(f">> Found {len(high_correlations)} high intercorrelations (|r| > 0.7):")
            for feat1, feat2, corr in high_correlations[:10]:
                print(f"   {feat1} <-> {feat2}: {corr:.3f}")
        else:
            print(f">> No concerning intercorrelations found")
    
    def perform_variance_analysis(self):
        """Variance-based feature selection"""
        print("\n" + "="*60)
        print("VARIANCE ANALYSIS")
        print("="*60)
        
        variance_scores = []
        
        for i, feature_name in enumerate(self.feature_names):
            feature_values = [row[i] for row in self.feature_matrix]
            
            # Calculate variance
            mean_val = sum(feature_values) / len(feature_values)
            variance = sum((x - mean_val) ** 2 for x in feature_values) / (len(feature_values) - 1)
            
            # Calculate coefficient of variation (normalized variance)
            cv = math.sqrt(variance) / abs(mean_val) if abs(mean_val) > 1e-10 else 0
            
            variance_scores.append((feature_name, variance, cv))
        
        # Sort by coefficient of variation (better for features with different scales)
        variance_scores.sort(key=lambda x: x[2], reverse=True)
        
        self.results['variance_analysis'] = variance_scores
        
        print(f">> Top 15 Features by Coefficient of Variation:")
        print(f"{'Feature':<35} {'Variance':<15} {'Coeff of Var':<15}")
        print("-" * 65)
        
        for feature_name, variance, cv in variance_scores[:15]:
            print(f"{feature_name:<35} {variance:<15.4f} {cv:<15.4f}")
    
    def perform_pca_analysis(self):
        """Principal Component Analysis for feature importance"""
        print("\n" + "="*60)
        print("PRINCIPAL COMPONENT ANALYSIS")
        print("="*60)
        
        if not self.feature_matrix:
            return
            
        n_features = len(self.feature_names)
        n_samples = len(self.feature_matrix)
        
        # Standardize features
        standardized_data = self._standardize_features()
        
        # Calculate covariance matrix
        cov_matrix = self._calculate_covariance_matrix(standardized_data)
        
        # Find principal components using power iteration
        principal_components = []
        explained_variances = []
        
        # Calculate first 3 principal components
        current_data = [row[:] for row in standardized_data]  # Copy data
        
        for pc_num in range(min(3, n_features)):
            eigenvalue, eigenvector = self._power_iteration(cov_matrix)
            
            # Calculate explained variance
            total_variance = sum(cov_matrix[i][i] for i in range(n_features))
            explained_variance = eigenvalue / total_variance if total_variance > 0 else 0
            
            # Feature loadings and importance
            loadings = []
            for i, feature_name in enumerate(self.feature_names):
                loading = eigenvector[i] * math.sqrt(eigenvalue)
                importance = abs(loading)
                loadings.append((feature_name, loading, importance))
            
            loadings.sort(key=lambda x: x[2], reverse=True)
            
            principal_components.append({
                'pc_number': pc_num + 1,
                'eigenvalue': eigenvalue,
                'explained_variance': explained_variance,
                'loadings': loadings
            })
            
            explained_variances.append(explained_variance)
            
            print(f">> PC{pc_num + 1}: Eigenvalue={eigenvalue:.3f}, "
                  f"Explained Variance={explained_variance*100:.1f}%")
            
            # Deflate the covariance matrix for next PC
            cov_matrix = self._deflate_matrix(cov_matrix, eigenvector, eigenvalue)
        
        self.results['pca_analysis'] = {
            'principal_components': principal_components,
            'total_explained_variance': sum(explained_variances),
            'cumulative_explained_variance': explained_variances
        }
        
        # Print top features from first PC
        if principal_components:
            print(f"\n>> Top 15 Features in First Principal Component:")
            print(f"{'Feature':<35} {'Loading':<15} {'Importance':<15}")
            print("-" * 65)
            
            for feature_name, loading, importance in principal_components[0]['loadings'][:15]:
                print(f"{feature_name:<35} {loading:<15.4f} {importance:<15.4f}")
    
    def _standardize_features(self):
        """Standardize features (z-score normalization)"""
        n_features = len(self.feature_names)
        n_samples = len(self.feature_matrix)
        
        # Calculate means and standard deviations
        means = []
        stds = []
        
        for j in range(n_features):
            feature_values = [self.feature_matrix[i][j] for i in range(n_samples)]
            mean_val = sum(feature_values) / n_samples
            variance = sum((x - mean_val) ** 2 for x in feature_values) / (n_samples - 1)
            std_val = math.sqrt(variance)
            
            means.append(mean_val)
            stds.append(std_val)
        
        # Standardize
        standardized_data = []
        for i in range(n_samples):
            std_row = []
            for j in range(n_features):
                if stds[j] > 1e-10:
                    std_val = (self.feature_matrix[i][j] - means[j]) / stds[j]
                else:
                    std_val = 0.0
                std_row.append(std_val)
            standardized_data.append(std_row)
        
        return standardized_data
    
    def _calculate_covariance_matrix(self, data):
        """Calculate covariance matrix"""
        n_features = len(self.feature_names)
        n_samples = len(data)
        
        cov_matrix = [[0.0] * n_features for _ in range(n_features)]
        
        for i in range(n_features):
            for j in range(n_features):
                covariance = sum(data[k][i] * data[k][j] for k in range(n_samples)) / (n_samples - 1)
                cov_matrix[i][j] = covariance
        
        return cov_matrix
    
    def _power_iteration(self, matrix, max_iters=100):
        """Power iteration to find dominant eigenvalue and eigenvector"""
        n = len(matrix)
        
        # Initialize random vector
        v = [random.random() for _ in range(n)]
        norm = math.sqrt(sum(x*x for x in v))
        v = [x/norm for x in v]
        
        for iteration in range(max_iters):
            # Matrix-vector multiplication
            new_v = [0.0] * n
            for i in range(n):
                for j in range(n):
                    new_v[i] += matrix[i][j] * v[j]
            
            # Normalize
            norm = math.sqrt(sum(x*x for x in new_v))
            if norm > 1e-10:
                new_v = [x/norm for x in new_v]
            
            # Check convergence
            diff = sum(abs(new_v[i] - v[i]) for i in range(n))
            if diff < 1e-6:
                break
                
            v = new_v
        
        # Calculate eigenvalue
        Av = [sum(matrix[i][j] * v[j] for j in range(n)) for i in range(n)]
        eigenvalue = sum(v[i] * Av[i] for i in range(n))
        
        return eigenvalue, v
    
    def _deflate_matrix(self, matrix, eigenvector, eigenvalue):
        """Deflate matrix to find next eigenvalue"""
        n = len(matrix)
        deflated = [[matrix[i][j] for j in range(n)] for i in range(n)]
        
        for i in range(n):
            for j in range(n):
                deflated[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j]
        
        return deflated
    
    def calculate_mutual_information(self):
        """Mutual information analysis for feature selection"""
        print("\n" + "="*60)
        print("MUTUAL INFORMATION ANALYSIS")
        print("="*60)
        
        # Discretize features for mutual information calculation
        mutual_info_scores = []
        
        for i, feature_name in enumerate(self.feature_names):
            feature_values = [row[i] for row in self.feature_matrix]
            
            # Discretize continuous features into bins
            discretized_features = self._discretize_feature(feature_values)
            
            # Calculate mutual information with target
            mi_score = self._calculate_mutual_information(discretized_features, self.target_values)
            mutual_info_scores.append((feature_name, mi_score))
        
        # Sort by mutual information score
        mutual_info_scores.sort(key=lambda x: x[1], reverse=True)
        
        self.results['mutual_information'] = mutual_info_scores
        
        print(f">> Top 15 Features by Mutual Information:")
        print(f"{'Feature':<35} {'MI Score':<15}")
        print("-" * 50)
        
        for feature_name, mi_score in mutual_info_scores[:15]:
            print(f"{feature_name:<35} {mi_score:<15.4f}")
    
    def _discretize_feature(self, values, n_bins=10):
        """Discretize continuous feature into bins"""
        if not values:
            return []
            
        min_val = min(values)
        max_val = max(values)
        
        if min_val == max_val:
            return [0] * len(values)
        
        bin_width = (max_val - min_val) / n_bins
        discretized = []
        
        for val in values:
            bin_idx = min(int((val - min_val) / bin_width), n_bins - 1)
            discretized.append(bin_idx)
        
        return discretized
    
    def _calculate_mutual_information(self, feature_values, target_values):
        """Calculate mutual information between feature and target"""
        # Create joint frequency table
        joint_counts = defaultdict(int)
        feature_counts = defaultdict(int)
        target_counts = defaultdict(int)
        
        n_samples = len(feature_values)
        
        for i in range(n_samples):
            f_val = feature_values[i]
            t_val = target_values[i]
            
            joint_counts[(f_val, t_val)] += 1
            feature_counts[f_val] += 1
            target_counts[t_val] += 1
        
        # Calculate mutual information
        mi = 0.0
        
        for (f_val, t_val), joint_count in joint_counts.items():
            p_joint = joint_count / n_samples
            p_feature = feature_counts[f_val] / n_samples
            p_target = target_counts[t_val] / n_samples
            
            if p_joint > 0 and p_feature > 0 and p_target > 0:
                mi += p_joint * math.log2(p_joint / (p_feature * p_target))
        
        return mi
    
    def perform_univariate_statistical_tests(self):
        """Univariate statistical tests for feature selection"""
        print("\n" + "="*60)
        print("UNIVARIATE STATISTICAL TESTS")
        print("="*60)
        
        statistical_scores = []
        
        # Group target values by class
        target_groups = defaultdict(list)
        for i, target_val in enumerate(self.target_values):
            target_groups[target_val].append(i)
        
        for i, feature_name in enumerate(self.feature_names):
            feature_values = [row[i] for row in self.feature_matrix]
            
            # ANOVA F-statistic for feature vs target
            f_statistic = self._calculate_anova_f_statistic(feature_values, target_groups)
            
            # Calculate effect size (eta-squared)
            eta_squared = self._calculate_eta_squared(feature_values, target_groups)
            
            statistical_scores.append((feature_name, f_statistic, eta_squared))
        
        # Sort by F-statistic
        statistical_scores.sort(key=lambda x: x[1], reverse=True)
        
        self.results['statistical_tests'] = statistical_scores
        
        print(f">> Top 15 Features by ANOVA F-statistic:")
        print(f"{'Feature':<35} {'F-statistic':<15} {'Eta-squared':<15}")
        print("-" * 65)
        
        for feature_name, f_stat, eta_sq in statistical_scores[:15]:
            print(f"{feature_name:<35} {f_stat:<15.4f} {eta_sq:<15.4f}")
    
    def _calculate_anova_f_statistic(self, feature_values, target_groups):
        """Calculate ANOVA F-statistic"""
        n_total = len(feature_values)
        n_groups = len(target_groups)
        
        if n_groups <= 1:
            return 0.0
        
        # Overall mean
        overall_mean = sum(feature_values) / n_total
        
        # Between-group sum of squares
        ss_between = 0.0
        for group_indices in target_groups.values():
            if len(group_indices) > 0:
                group_values = [feature_values[i] for i in group_indices]
                group_mean = sum(group_values) / len(group_values)
                ss_between += len(group_values) * (group_mean - overall_mean) ** 2
        
        # Within-group sum of squares
        ss_within = 0.0
        for group_indices in target_groups.values():
            if len(group_indices) > 1:
                group_values = [feature_values[i] for i in group_indices]
                group_mean = sum(group_values) / len(group_values)
                ss_within += sum((val - group_mean) ** 2 for val in group_values)
        
        # Degrees of freedom
        df_between = n_groups - 1
        df_within = n_total - n_groups
        
        if df_between == 0 or df_within == 0 or ss_within == 0:
            return 0.0
        
        # F-statistic
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        
        return ms_between / ms_within
    
    def _calculate_eta_squared(self, feature_values, target_groups):
        """Calculate eta-squared (effect size)"""
        n_total = len(feature_values)
        overall_mean = sum(feature_values) / n_total
        
        # Total sum of squares
        ss_total = sum((val - overall_mean) ** 2 for val in feature_values)
        
        # Between-group sum of squares
        ss_between = 0.0
        for group_indices in target_groups.values():
            if len(group_indices) > 0:
                group_values = [feature_values[i] for i in group_indices]
                group_mean = sum(group_values) / len(group_values)
                ss_between += len(group_values) * (group_mean - overall_mean) ** 2
        
        return ss_between / ss_total if ss_total > 0 else 0.0
    
    def create_comprehensive_feature_ranking(self):
        """Create comprehensive ranking using multiple methods"""
        print("\n" + "="*60)
        print("COMPREHENSIVE FEATURE RANKING")
        print("="*60)
        
        # Normalize all scores to 0-1 scale for comparison
        feature_scores = {}
        
        # Initialize scores
        for feature_name in self.feature_names:
            feature_scores[feature_name] = {
                'correlation_rank': 0,
                'variance_rank': 0,
                'pca_rank': 0,
                'mutual_info_rank': 0,
                'statistical_rank': 0,
                'composite_score': 0
            }
        
        # Correlation ranking
        if 'target_correlations' in self.results:
            for rank, (feature_name, abs_corr, corr) in enumerate(self.results['target_correlations']):
                if feature_name in feature_scores:
                    feature_scores[feature_name]['correlation_rank'] = len(self.feature_names) - rank
        
        # Variance ranking
        if 'variance_analysis' in self.results:
            for rank, (feature_name, variance, cv) in enumerate(self.results['variance_analysis']):
                if feature_name in feature_scores:
                    feature_scores[feature_name]['variance_rank'] = len(self.feature_names) - rank
        
        # PCA ranking (from first PC)
        if 'pca_analysis' in self.results:
            pcs = self.results['pca_analysis']['principal_components']
            if pcs:
                for rank, (feature_name, loading, importance) in enumerate(pcs[0]['loadings']):
                    if feature_name in feature_scores:
                        feature_scores[feature_name]['pca_rank'] = len(self.feature_names) - rank
        
        # Mutual information ranking
        if 'mutual_information' in self.results:
            for rank, (feature_name, mi_score) in enumerate(self.results['mutual_information']):
                if feature_name in feature_scores:
                    feature_scores[feature_name]['mutual_info_rank'] = len(self.feature_names) - rank
        
        # Statistical test ranking
        if 'statistical_tests' in self.results:
            for rank, (feature_name, f_stat, eta_sq) in enumerate(self.results['statistical_tests']):
                if feature_name in feature_scores:
                    feature_scores[feature_name]['statistical_rank'] = len(self.feature_names) - rank
        
        # Calculate composite scores with different weightings
        for feature_name in feature_scores:
            scores = feature_scores[feature_name]
            
            # Weighted average (emphasizing correlation and statistical tests)
            composite = (
                scores['correlation_rank'] * 0.3 +
                scores['statistical_rank'] * 0.25 +
                scores['pca_rank'] * 0.2 +
                scores['mutual_info_rank'] * 0.15 +
                scores['variance_rank'] * 0.1
            )
            
            feature_scores[feature_name]['composite_score'] = composite
        
        # Sort by composite score
        ranked_features = sorted(
            feature_scores.items(),
            key=lambda x: x[1]['composite_score'],
            reverse=True
        )
        
        self.results['comprehensive_ranking'] = ranked_features
        
        print(f">> TOP 20 FEATURES BY COMPREHENSIVE RANKING:")
        print(f"{'Rank':<5} {'Feature':<35} {'Composite':<12} {'Corr':<8} {'Stat':<8} {'PCA':<8}")
        print("-" * 85)
        
        for rank, (feature_name, scores) in enumerate(ranked_features[:20], 1):
            print(f"{rank:<5} {feature_name:<35} {scores['composite_score']:<12.2f} "
                  f"{scores['correlation_rank']:<8} {scores['statistical_rank']:<8} "
                  f"{scores['pca_rank']:<8}")
        
        # Highlight quiz-related and discount code features
        print(f"\n>> QUIZ & DISCOUNT CODE FEATURES RANKING:")
        quiz_features = [(rank+1, name, scores) for rank, (name, scores) in enumerate(ranked_features) 
                        if any(keyword in name.lower() for keyword in ['quiz', 'discount', 'ancestor', 'stress', 'pregnant', 'health'])]
        
        if quiz_features:
            print(f"{'Rank':<5} {'Feature':<35} {'Composite Score':<15}")
            print("-" * 55)
            for rank, feature_name, scores in quiz_features[:15]:
                print(f"{rank:<5} {feature_name:<35} {scores['composite_score']:<15.2f}")
        
    def generate_feature_selection_recommendations(self):
        """Generate expert recommendations for feature selection"""
        print("\n" + "="*80)
        print("EXPERT FEATURE SELECTION RECOMMENDATIONS")
        print("="*80)
        
        if 'comprehensive_ranking' not in self.results:
            return
        
        ranked_features = self.results['comprehensive_ranking']
        
        # Recommendations based on statistical analysis
        recommendations = {
            'tier_1_critical': [],
            'tier_2_important': [],
            'tier_3_supplementary': [],
            'multicollinearity_concerns': [],
            'quiz_specific_features': [],
            'discount_features': []
        }
        
        # Categorize features by tiers
        for rank, (feature_name, scores) in enumerate(ranked_features):
            composite_score = scores['composite_score']
            
            if composite_score >= len(self.feature_names) * 0.8:
                recommendations['tier_1_critical'].append((rank+1, feature_name, composite_score))
            elif composite_score >= len(self.feature_names) * 0.6:
                recommendations['tier_2_important'].append((rank+1, feature_name, composite_score))
            elif composite_score >= len(self.feature_names) * 0.4:
                recommendations['tier_3_supplementary'].append((rank+1, feature_name, composite_score))
        
        # Identify quiz and discount features
        for rank, (feature_name, scores) in enumerate(ranked_features):
            if any(keyword in feature_name.lower() for keyword in ['quiz', 'stress', 'pregnant', 'health', 'mental', 'physical']):
                recommendations['quiz_specific_features'].append((rank+1, feature_name, scores['composite_score']))
            elif any(keyword in feature_name.lower() for keyword in ['discount', 'ancestor', 'code']):
                recommendations['discount_features'].append((rank+1, feature_name, scores['composite_score']))
        
        # Check for multicollinearity
        if 'high_intercorrelations' in self.results:
            recommendations['multicollinearity_concerns'] = self.results['high_intercorrelations']
        
        # Print recommendations
        print("="*80)
        print("TIER 1 - CRITICAL FEATURES (Top 20% - Must Include)")
        print("="*80)
        for rank, feature_name, score in recommendations['tier_1_critical'][:15]:
            print(f"{rank:>3}. {feature_name:<45} (Score: {score:.2f})")
        
        print("\n" + "="*80)
        print("TIER 2 - IMPORTANT FEATURES (Strong Predictive Power)")
        print("="*80)
        for rank, feature_name, score in recommendations['tier_2_important'][:10]:
            print(f"{rank:>3}. {feature_name:<45} (Score: {score:.2f})")
        
        print("\n" + "="*80)
        print("QUIZ-SPECIFIC FEATURES (Stakeholder Priority)")
        print("="*80)
        if recommendations['quiz_specific_features']:
            for rank, feature_name, score in recommendations['quiz_specific_features'][:10]:
                print(f"{rank:>3}. {feature_name:<45} (Score: {score:.2f})")
        else:
            print("No quiz-specific features found in top ranks")
        
        print("\n" + "="*80)
        print("DISCOUNT CODE FEATURES (ancestor_discount_code related)")
        print("="*80)
        if recommendations['discount_features']:
            for rank, feature_name, score in recommendations['discount_features']:
                print(f"{rank:>3}. {feature_name:<45} (Score: {score:.2f})")
        else:
            print("No discount code features found")
        
        # Statistical summary
        print("\n" + "="*80)
        print("STATISTICAL SUMMARY & VALIDATION")
        print("="*80)
        
        if 'pca_analysis' in self.results:
            total_var = self.results['pca_analysis']['total_explained_variance']
            print(f"• First 3 PCs explain {total_var*100:.1f}% of variance")
        
        n_tier1 = len(recommendations['tier_1_critical'])
        n_tier2 = len(recommendations['tier_2_important'])
        print(f"• Recommended feature set: {n_tier1} critical + {n_tier2} important = {n_tier1 + n_tier2} features")
        print(f"• Original feature space: {len(self.feature_names)} features")
        print(f"• Dimensionality reduction: {(1 - (n_tier1 + n_tier2)/len(self.feature_names))*100:.1f}%")
        
        if recommendations['multicollinearity_concerns']:
            print(f"• Multicollinearity warnings: {len(recommendations['multicollinearity_concerns'])} feature pairs")
        else:
            print(f"• No severe multicollinearity detected")
        
        # Final recommendation
        print("\n" + "="*80)
        print("FINAL EXPERT RECOMMENDATION")
        print("="*80)
        
        final_features = []
        final_features.extend([name for _, name, _ in recommendations['tier_1_critical']])
        final_features.extend([name for _, name, _ in recommendations['tier_2_important'][:5]])
        
        # Force include top quiz features if not already included
        quiz_to_add = [name for _, name, _ in recommendations['quiz_specific_features'][:3] 
                      if name not in final_features]
        final_features.extend(quiz_to_add)
        
        # Force include discount code features if not already included
        discount_to_add = [name for _, name, _ in recommendations['discount_features'][:2] 
                          if name not in final_features]
        final_features.extend(discount_to_add)
        
        print(f"RECOMMENDED FEATURE SET ({len(final_features)} features):")
        print("-" * 60)
        for i, feature in enumerate(final_features, 1):
            print(f"{i:>2}. {feature}")
        
        # Store final recommendation
        self.results['final_recommendation'] = final_features
        
        return final_features
    
    def save_results_to_file(self):
        """Save comprehensive analysis results"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"feature_selection_analysis_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("COMPREHENSIVE FEATURE SELECTION ANALYSIS RESULTS\n")
            f.write("="*80 + "\n\n")
            
            # Write all results
            for key, value in self.results.items():
                f.write(f"{key.upper()}\n")
                f.write("-" * 40 + "\n")
                f.write(str(value) + "\n\n")
        
        print(f"\n>> Results saved to: {filename}")
    
    # Utility methods
    def _safe_float(self, value, default=0.0):
        """Safely convert to float"""
        try:
            return float(value) if value and value != '' else default
        except (ValueError, TypeError):
            return default
    
    def _safe_int(self, value, default=0):
        """Safely convert to int"""
        try:
            return int(float(value)) if value and value != '' else default
        except (ValueError, TypeError):
            return default
    
    def _safe_bool(self, value):
        """Safely convert to bool"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ['true', 'yes', '1', 'y']
        return bool(value) if value else False

def main():
    """Execute comprehensive feature selection analysis"""
    
    print("EXPERT-LEVEL FEATURE SELECTION ANALYSIS")
    print("Using multiple statistical methodologies for optimal feature selection")
    print("="*80)
    
    try:
        # Initialize analyzer
        analyzer = AdvancedFeatureSelector('raw_data_v2.csv')
        
        # Execute comprehensive pipeline
        analyzer.load_and_parse_data()
        analyzer.encode_categorical_features()
        analyzer.create_feature_matrix()
        
        # Statistical analysis methods
        analyzer.calculate_correlation_analysis()
        analyzer.perform_variance_analysis()
        analyzer.perform_pca_analysis()
        analyzer.calculate_mutual_information()
        analyzer.perform_univariate_statistical_tests()
        
        # Generate final recommendations
        analyzer.create_comprehensive_feature_ranking()
        final_features = analyzer.generate_feature_selection_recommendations()
        
        # Save results
        analyzer.save_results_to_file()
        
        print("\n" + "="*80)
        print("EXPERT ANALYSIS COMPLETE")
        print("="*80)
        print("✓ Multiple statistical methods applied")
        print("✓ Quiz features and ancestor_discount_code specifically analyzed")
        print("✓ Comprehensive feature ranking generated")
        print("✓ Evidence-based recommendations provided")
        print("✓ All results saved with full traceability")
        
        return final_features
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()