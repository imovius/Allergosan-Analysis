#!/usr/bin/env python3
"""
TWO-STAGE CUSTOMER SEGMENTATION WITH QUIZ OVERLAY
Stage 1: Core behavioral clustering (all customers)
Stage 2: Quiz feature overlay analysis (quiz takers only)

Author: Ian Movius
Date: January 2025
Based on: Statistical feature selection results
"""

import csv
import math
import random
from collections import defaultdict, Counter
import datetime

class TwoStageSegmentationAnalyzer:
    """Two-stage clustering: behavioral segments + quiz overlay"""
    
    def __init__(self, data_file='raw_data_v2.csv'):
        self.data_file = data_file
        self.customers = []
        self.quiz_takers = []
        self.non_quiz_takers = []
        
        # Feature sets from statistical analysis
        self.core_features = [
            'high_value_flag', 'order_frequency', 'repeat_customer', 
            'shipping_spend', 'order_count', 'shipping_collected',
            'recent_customer', 'recency_score', 'churn_risk',
            'days_since_last_order', 'avg_days_between_orders',
            'ancestor_discount_code_encoded'
        ]
        
        self.quiz_features = [
            'stress_mental_flag', 'stress_physical_flag', 'stress_digestion_flag',
            'sx_bloating', 'sx_anxiety', 'sx_constipation', 'sx_brain_fog',
            'bm_pattern', 'gi_symptom_cat', 'primary_goal',
            'quiz_health_complexity', 'is_pregnant'
        ]
        
        self.results = {}
        
    def load_and_engineer_data(self):
        """Load data and create engineered features"""
        print("="*80)
        print("TWO-STAGE CUSTOMER SEGMENTATION ANALYSIS")
        print("="*80)
        print("Loading and engineering features...")
        
        with open(self.data_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            raw_data = list(reader)
        
        print(f">> Loaded {len(raw_data):,} raw customer records")
        
        # Process each customer
        for row in raw_data:
            try:
                customer = self._engineer_customer_features(row)
                if customer:
                    self.customers.append(customer)
                    
                    if customer['quiz_taker']:
                        self.quiz_takers.append(customer)
                    else:
                        self.non_quiz_takers.append(customer)
                        
            except Exception as e:
                continue
        
        print(f">> Processed {len(self.customers):,} customers")
        print(f"   - Quiz takers: {len(self.quiz_takers):,} ({len(self.quiz_takers)/len(self.customers)*100:.1f}%)")
        print(f"   - Non-quiz takers: {len(self.non_quiz_takers):,}")
        
        # Store basic stats
        self.results['data_overview'] = {
            'total_customers': len(self.customers),
            'quiz_takers': len(self.quiz_takers),
            'non_quiz_takers': len(self.non_quiz_takers),
            'quiz_participation_rate': len(self.quiz_takers) / len(self.customers) * 100
        }
        
    def _engineer_customer_features(self, row):
        """Engineer features for a single customer"""
        try:
            # Basic transactional data
            net_ltv = self._safe_float(row.get('net_ltv', 0))
            avg_order_value = self._safe_float(row.get('avg_order_value', 0))
            order_count = self._safe_int(row.get('order_count', 0))
            days_since_last_order = self._safe_int(row.get('days_since_last_order', 0))
            shipping_collected = self._safe_float(row.get('shipping_collected', 0))
            shipping_spend = self._safe_float(row.get('shipping_spend', 0))
            avg_days_between_orders = self._safe_float(row.get('avg_days_between_orders', 0))
            
            # Skip customers with no meaningful data
            if net_ltv <= 0 or order_count <= 0:
                return None
            
            # Date parsing for tenure
            first_date = row.get('first_order_date', '')
            if first_date and len(first_date) >= 10:
                try:
                    first_order = datetime.datetime.strptime(first_date[:19], '%Y-%m-%d %H:%M:%S')
                    today = datetime.datetime.now()
                    tenure_days = max(1, (today - first_order).days)
                    tenure_months = max(1, tenure_days / 30.44)
                except:
                    tenure_days = 365
                    tenure_months = 12
            else:
                tenure_days = 365
                tenure_months = 12
            
            # Core behavioral features
            customer = {
                'customer_id': row.get('customer_id', ''),
                'net_ltv': net_ltv,
                'avg_order_value': avg_order_value,
                'order_count': order_count,
                'days_since_last_order': days_since_last_order,
                'tenure_months': tenure_months,
                'shipping_collected': shipping_collected,
                'shipping_spend': shipping_spend,
                'avg_days_between_orders': avg_days_between_orders,
                
                # Derived behavioral features
                'order_frequency': order_count / tenure_months,
                'high_value_flag': 1 if net_ltv > 300 else 0,
                'repeat_customer': 1 if order_count > 1 else 0,
                'recent_customer': 1 if days_since_last_order < 30 else 0,
                'recency_score': 1 / (1 + days_since_last_order / 365),
                'churn_risk': 1 if days_since_last_order > 180 else 0,
                
                # Marketing features
                'ancestor_discount_code': row.get('ancestor_discount_code', '').strip(),
                'acquisition_channel': row.get('acquisition_channel', '').strip(),
                'affiliate_segment': row.get('affiliate_segment', '').strip(),
                
                # Quiz status
                'quiz_taker': row.get('quiz_taker', '').strip().lower() == 'yes'
            }
            
            # Encode ancestor discount code (simplified)
            adc = customer['ancestor_discount_code']
            if adc and adc.lower() not in ['', 'none', 'null']:
                # Group by major discount codes
                if 'dave' in adc.lower():
                    customer['ancestor_discount_code_encoded'] = 3
                elif 'www' in adc.lower() or '20' in adc:
                    customer['ancestor_discount_code_encoded'] = 2
                elif any(code in adc.lower() for code in ['skinny', 'enjoy', 'jessica', 'blonde']):
                    customer['ancestor_discount_code_encoded'] = 1
                else:
                    customer['ancestor_discount_code_encoded'] = 0.5
            else:
                customer['ancestor_discount_code_encoded'] = 0
            
            # Quiz features (only for quiz takers)
            if customer['quiz_taker']:
                customer['stress_mental_flag'] = self._safe_bool(row.get('stress_mental_flag', ''))
                customer['stress_physical_flag'] = self._safe_bool(row.get('stress_physical_flag', ''))
                customer['stress_digestion_flag'] = self._safe_bool(row.get('stress_digestion_flag', ''))
                customer['sx_bloating'] = self._safe_bool(row.get('sx_bloating', ''))
                customer['sx_anxiety'] = self._safe_bool(row.get('sx_anxiety', ''))
                customer['sx_constipation'] = self._safe_bool(row.get('sx_constipation', ''))
                customer['sx_brain_fog'] = self._safe_bool(row.get('sx_brain_fog', ''))
                customer['bm_pattern'] = row.get('bm_pattern', '').strip()
                customer['gi_symptom_cat'] = row.get('gi_symptom_cat', '').strip()
                customer['primary_goal'] = row.get('primary_goal', '').strip()
                customer['is_pregnant'] = self._safe_bool(row.get('is_pregnant', ''))
                
                # Health complexity score
                customer['quiz_health_complexity'] = sum([
                    customer['stress_mental_flag'],
                    customer['stress_physical_flag'],
                    customer['stress_digestion_flag'],
                    customer['sx_anxiety']
                ])
            else:
                # Set quiz features to None for non-quiz takers
                for feature in self.quiz_features:
                    customer[feature] = None
            
            return customer
            
        except Exception as e:
            return None
    
    def perform_stage1_clustering(self):
        """Stage 1: Core behavioral clustering on all customers"""
        print("\n" + "="*60)
        print("STAGE 1: CORE BEHAVIORAL CLUSTERING")
        print("="*60)
        print("Clustering all customers based on behavioral features...")
        
        # Create feature matrix for core features only
        feature_matrix = []
        for customer in self.customers:
            row = []
            for feature in self.core_features:
                value = customer.get(feature, 0)
                row.append(float(value) if value is not None else 0.0)
            feature_matrix.append(row)
        
        print(f">> Feature matrix: {len(feature_matrix)} customers x {len(self.core_features)} features")
        
        # Standardize features
        standardized_data = self._standardize_matrix(feature_matrix)
        
        # Test different k values
        best_k = 5  # Start with 5 based on previous analysis
        cluster_validation = {}
        
        for k in range(3, 8):
            print(f">> Testing k={k}...")
            
            best_inertia = float('inf')
            best_assignments = None
            best_centroids = None
            
            # Multiple random initializations
            for init in range(3):
                assignments, centroids, inertia = self._kmeans_clustering(standardized_data, k)
                if inertia < best_inertia:
                    best_inertia = inertia
                    best_assignments = assignments
                    best_centroids = centroids
            
            # Calculate validation metrics
            silhouette = self._calculate_silhouette_sample(standardized_data, best_assignments)
            r_squared = self._calculate_r_squared(standardized_data, best_assignments, best_centroids)
            
            cluster_validation[k] = {
                'assignments': best_assignments,
                'centroids': best_centroids,
                'inertia': best_inertia,
                'r_squared': r_squared,
                'silhouette': silhouette,
                'composite_score': r_squared * 0.6 + silhouette * 0.4
            }
            
            print(f"   - Inertia: {best_inertia:.2f}, R²: {r_squared:.3f}, Silhouette: {silhouette:.3f}")
        
        # Select best k
        best_k = max(cluster_validation.keys(), key=lambda k: cluster_validation[k]['composite_score'])
        best_solution = cluster_validation[best_k]
        
        print(f"\n>> OPTIMAL K = {best_k}")
        print(f"   - R²: {best_solution['r_squared']:.3f}")
        print(f"   - Silhouette: {best_solution['silhouette']:.3f}")
        
        # Assign cluster labels to customers
        for i, customer in enumerate(self.customers):
            customer['behavioral_cluster'] = best_solution['assignments'][i]
        
        # Store results
        self.results['stage1_clustering'] = {
            'optimal_k': best_k,
            'validation_results': cluster_validation,
            'best_solution': best_solution,
            'feature_names': self.core_features
        }
        
        return best_k, best_solution
        
    def analyze_behavioral_clusters(self):
        """Analyze the core behavioral clusters"""
        print("\n" + "="*60)
        print("BEHAVIORAL CLUSTER ANALYSIS")
        print("="*60)
        
        # Group customers by cluster
        clusters = defaultdict(list)
        for customer in self.customers:
            cluster_id = customer['behavioral_cluster']
            clusters[cluster_id].append(customer)
        
        cluster_profiles = {}
        
        print(f"{'Cluster':<10} {'Size':<8} {'%':<6} {'LTV':<10} {'AOV':<10} {'Orders':<8} {'Quiz %':<8}")
        print("-" * 70)
        
        for cluster_id in sorted(clusters.keys()):
            cluster_customers = clusters[cluster_id]
            n_customers = len(cluster_customers)
            
            # Basic stats
            ltv_values = [c['net_ltv'] for c in cluster_customers]
            aov_values = [c['avg_order_value'] for c in cluster_customers]
            order_values = [c['order_count'] for c in cluster_customers]
            
            ltv_mean = sum(ltv_values) / len(ltv_values)
            aov_mean = sum(aov_values) / len(aov_values)
            order_mean = sum(order_values) / len(order_values)
            
            # Quiz participation in this cluster
            quiz_count = sum(1 for c in cluster_customers if c['quiz_taker'])
            quiz_rate = quiz_count / n_customers * 100
            
            # Discount code usage
            discount_users = sum(1 for c in cluster_customers if c['ancestor_discount_code_encoded'] > 0)
            discount_rate = discount_users / n_customers * 100
            
            cluster_profiles[cluster_id] = {
                'size': n_customers,
                'size_percentage': n_customers / len(self.customers) * 100,
                'ltv_mean': ltv_mean,
                'aov_mean': aov_mean,
                'order_mean': order_mean,
                'quiz_count': quiz_count,
                'quiz_rate': quiz_rate,
                'discount_rate': discount_rate,
                'customers': cluster_customers
            }
            
            print(f"{cluster_id + 1:<10} {n_customers:<8} {n_customers/len(self.customers)*100:<6.1f} "
                  f"${ltv_mean:<9.2f} ${aov_mean:<9.2f} {order_mean:<7.1f} {quiz_rate:<7.1f}")
        
        self.results['behavioral_clusters'] = cluster_profiles
        self.cluster_profiles = cluster_profiles
        
        return cluster_profiles
    
    def perform_stage2_quiz_overlay(self):
        """Stage 2: Quiz feature overlay analysis within each cluster"""
        print("\n" + "="*60)
        print("STAGE 2: QUIZ FEATURE OVERLAY ANALYSIS")
        print("="*60)
        print("Analyzing quiz patterns within each behavioral cluster...")
        
        quiz_insights = {}
        
        for cluster_id, profile in self.cluster_profiles.items():
            # Get quiz takers in this cluster
            cluster_quiz_takers = [c for c in profile['customers'] if c['quiz_taker']]
            
            if len(cluster_quiz_takers) < 10:  # Skip clusters with too few quiz takers
                print(f">> Cluster {cluster_id + 1}: Only {len(cluster_quiz_takers)} quiz takers - skipping overlay")
                continue
            
            print(f">> Cluster {cluster_id + 1}: Analyzing {len(cluster_quiz_takers)} quiz takers...")
            
            # Health pattern analysis
            health_patterns = {
                'stress_mental': sum(c['stress_mental_flag'] for c in cluster_quiz_takers if c['stress_mental_flag'] is not None),
                'stress_physical': sum(c['stress_physical_flag'] for c in cluster_quiz_takers if c['stress_physical_flag'] is not None),
                'stress_digestion': sum(c['stress_digestion_flag'] for c in cluster_quiz_takers if c['stress_digestion_flag'] is not None),
                'sx_bloating': sum(c['sx_bloating'] for c in cluster_quiz_takers if c['sx_bloating'] is not None),
                'sx_anxiety': sum(c['sx_anxiety'] for c in cluster_quiz_takers if c['sx_anxiety'] is not None),
                'sx_constipation': sum(c['sx_constipation'] for c in cluster_quiz_takers if c['sx_constipation'] is not None),
                'is_pregnant': sum(c['is_pregnant'] for c in cluster_quiz_takers if c['is_pregnant'] is not None)
            }
            
            # Convert to percentages
            n_quiz = len(cluster_quiz_takers)
            health_percentages = {key: (value / n_quiz * 100) for key, value in health_patterns.items()}
            
            # Primary goals
            goals = [c['primary_goal'] for c in cluster_quiz_takers if c['primary_goal']]
            goal_counter = Counter(goals)
            top_goals = goal_counter.most_common(3)
            
            # GI symptom categories
            gi_symptoms = [c['gi_symptom_cat'] for c in cluster_quiz_takers if c['gi_symptom_cat']]
            gi_counter = Counter(gi_symptoms)
            top_gi = gi_counter.most_common(3)
            
            # Health complexity distribution
            complexity_scores = [c['quiz_health_complexity'] for c in cluster_quiz_takers 
                               if c['quiz_health_complexity'] is not None]
            avg_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0
            
            quiz_insights[cluster_id] = {
                'quiz_sample_size': n_quiz,
                'health_patterns': health_percentages,
                'top_goals': top_goals,
                'top_gi_symptoms': top_gi,
                'avg_health_complexity': avg_complexity,
                'dominant_health_theme': self._identify_health_theme(health_percentages, top_goals)
            }
            
            print(f"   - Health complexity: {avg_complexity:.1f}")
            print(f"   - Top goals: {', '.join([f'{goal}({count})' for goal, count in top_goals])}")
            print(f"   - Stress patterns: Mental {health_percentages['stress_mental']:.1f}%, "
                  f"Physical {health_percentages['stress_physical']:.1f}%")
        
        self.results['quiz_overlay'] = quiz_insights
        return quiz_insights
    
    def _identify_health_theme(self, health_patterns, top_goals):
        """Identify the dominant health theme for a cluster"""
        # Stress-driven
        if health_patterns['stress_mental'] > 50 or health_patterns['stress_physical'] > 50:
            return "Stress-Driven"
        
        # Digestive-focused
        if health_patterns['sx_bloating'] > 40 or health_patterns['stress_digestion'] > 40:
            return "Digestive-Focused"
        
        # Anxiety-related
        if health_patterns['sx_anxiety'] > 30:
            return "Anxiety-Related"
        
        # Performance/Athletic
        if top_goals and any('athletic' in goal[0].lower() or 'energy' in goal[0].lower() 
                           for goal in top_goals[:2]):
            return "Performance-Oriented"
        
        # General wellness
        return "General-Wellness"
    
    def generate_customer_personas(self):
        """Generate comprehensive customer personas combining behavioral + health insights"""
        print("\n" + "="*60)
        print("CUSTOMER PERSONA GENERATION")
        print("="*60)
        
        personas = {}
        
        for cluster_id, behavioral_profile in self.cluster_profiles.items():
            quiz_insight = self.results['quiz_overlay'].get(cluster_id, {})
            
            # Create persona name based on behavioral characteristics
            ltv = behavioral_profile['ltv_mean']
            orders = behavioral_profile['order_mean']
            quiz_rate = behavioral_profile['quiz_rate']
            
            if ltv > 500 and orders > 5:
                base_name = "Champions"
            elif ltv > 300 and orders > 3:
                base_name = "Loyalists"
            elif ltv > 150 and orders > 2:
                base_name = "Regulars"
            elif orders > 1:
                base_name = "Occasionals"
            else:
                base_name = "One-Timers"
            
            # Add health theme if available
            health_theme = quiz_insight.get('dominant_health_theme', '')
            if health_theme and quiz_rate > 15:
                persona_name = f"{health_theme} {base_name}"
            else:
                persona_name = base_name
            
            # Business characteristics
            business_profile = {
                'cluster_id': cluster_id,
                'persona_name': persona_name,
                'size': behavioral_profile['size'],
                'size_percentage': behavioral_profile['size_percentage'],
                'avg_ltv': behavioral_profile['ltv_mean'],
                'avg_aov': behavioral_profile['aov_mean'],
                'avg_orders': behavioral_profile['order_mean'],
                'quiz_participation': behavioral_profile['quiz_rate'],
                'discount_usage': behavioral_profile['discount_rate']
            }
            
            # Health profile (if available)
            health_profile = {}
            if quiz_insight:
                health_profile = {
                    'health_theme': quiz_insight['dominant_health_theme'],
                    'health_complexity': quiz_insight['avg_health_complexity'],
                    'top_goals': quiz_insight['top_goals'],
                    'stress_patterns': {
                        'mental': quiz_insight['health_patterns']['stress_mental'],
                        'physical': quiz_insight['health_patterns']['stress_physical'],
                        'digestive': quiz_insight['health_patterns']['stress_digestion']
                    },
                    'key_symptoms': {
                        'bloating': quiz_insight['health_patterns']['sx_bloating'],
                        'anxiety': quiz_insight['health_patterns']['sx_anxiety'],
                        'constipation': quiz_insight['health_patterns']['sx_constipation']
                    }
                }
            
            # Marketing recommendations
            marketing_recs = self._generate_marketing_recommendations(business_profile, health_profile)
            
            personas[cluster_id] = {
                'business_profile': business_profile,
                'health_profile': health_profile,
                'marketing_recommendations': marketing_recs
            }
        
        self.results['customer_personas'] = personas
        
        # Print persona summary
        print(f"\n{'Persona':<25} {'Size':<8} {'LTV':<10} {'Health Theme':<20}")
        print("-" * 65)
        
        for cluster_id, persona in personas.items():
            bp = persona['business_profile']
            hp = persona['health_profile']
            theme = hp.get('health_theme', 'Unknown') if hp else 'No Quiz Data'
            
            print(f"{bp['persona_name']:<25} {bp['size']:<8} ${bp['avg_ltv']:<9.2f} {theme:<20}")
        
        return personas
    
    def _generate_marketing_recommendations(self, business_profile, health_profile):
        """Generate targeted marketing recommendations"""
        recs = []
        
        # Business-based recommendations
        if business_profile['avg_ltv'] > 400:
            recs.append("Premium product focus - high willingness to pay")
            recs.append("VIP program enrollment - reward loyalty")
        
        if business_profile['avg_orders'] > 4:
            recs.append("Subscription program - regular purchaser")
            recs.append("Cross-selling opportunities - engaged customer")
        
        if business_profile['quiz_participation'] > 30:
            recs.append("Health-focused messaging - engaged with wellness content")
        
        if business_profile['discount_usage'] > 50:
            recs.append("Price-sensitive segment - strategic discounting")
        
        # Health-based recommendations
        if health_profile:
            theme = health_profile.get('health_theme', '')
            
            if 'Stress' in theme:
                recs.append("Stress-relief messaging - mental wellness focus")
                recs.append("Adaptogen products - stress management")
            
            if 'Digestive' in theme:
                recs.append("Gut health education - digestive wellness")
                recs.append("Probiotic regimens - targeted digestive support")
            
            if 'Performance' in theme:
                recs.append("Athletic performance messaging - energy & vitality")
                recs.append("Pre/post workout products - performance optimization")
            
            # Symptom-specific
            symptoms = health_profile.get('key_symptoms', {})
            if symptoms.get('bloating', 0) > 40:
                recs.append("Bloating-specific products - targeted relief")
            
            if symptoms.get('anxiety', 0) > 30:
                recs.append("Anxiety support - mental wellness products")
        
        return recs[:6]  # Limit to top 6 recommendations
    
    def generate_comprehensive_report(self):
        """Generate comprehensive two-stage analysis report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE TWO-STAGE ANALYSIS REPORT")
        print("="*80)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"two_stage_segmentation_report_{timestamp}.md"
        
        report = []
        report.append("# Two-Stage Customer Segmentation Analysis Report")
        report.append("")
        report.append("**Company**: Allergosan")
        report.append("**Analysis Date**: January 2025")
        report.append("**Methodology**: Two-stage clustering with quiz overlay")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        overview = self.results['data_overview']
        report.append(f"**Total Customers Analyzed**: {overview['total_customers']:,}")
        report.append(f"**Quiz Participants**: {overview['quiz_takers']:,} ({overview['quiz_participation_rate']:.1f}%)")
        report.append("")
        
        # Methodology
        report.append("## Methodology")
        report.append("")
        report.append("### Two-Stage Approach")
        report.append("1. **Stage 1**: Behavioral clustering using all customers and core transactional features")
        report.append("2. **Stage 2**: Health pattern overlay using quiz responses within each behavioral cluster")
        report.append("")
        report.append("**Benefits of this approach**:")
        report.append("- Eliminates temporal bias from quiz introduction timing")
        report.append("- Maximizes statistical power using all customer data")
        report.append("- Provides business-actionable segments enhanced with health insights")
        report.append("")
        
        # Stage 1 Results
        stage1 = self.results['stage1_clustering']
        report.append("### Stage 1: Behavioral Clustering Results")
        report.append(f"- **Optimal Clusters**: {stage1['optimal_k']}")
        report.append(f"- **Variance Explained (R²)**: {stage1['best_solution']['r_squared']:.3f}")
        report.append(f"- **Silhouette Score**: {stage1['best_solution']['silhouette']:.3f}")
        report.append("")
        
        # Customer Personas
        report.append("## Customer Personas")
        report.append("")
        personas = self.results['customer_personas']
        
        for cluster_id, persona in personas.items():
            bp = persona['business_profile']
            hp = persona['health_profile']
            
            report.append(f"### {bp['persona_name']}")
            report.append(f"**Size**: {bp['size']:,} customers ({bp['size_percentage']:.1f}%)")
            report.append("")
            
            report.append("**Business Profile**:")
            report.append(f"- Average LTV: ${bp['avg_ltv']:.2f}")
            report.append(f"- Average Order Value: ${bp['avg_aov']:.2f}")
            report.append(f"- Average Orders: {bp['avg_orders']:.1f}")
            report.append(f"- Quiz Participation: {bp['quiz_participation']:.1f}%")
            report.append(f"- Discount Usage: {bp['discount_usage']:.1f}%")
            report.append("")
            
            if hp:
                report.append("**Health Profile**:")
                report.append(f"- Health Theme: {hp['health_theme']}")
                report.append(f"- Health Complexity: {hp['health_complexity']:.1f}")
                if hp['top_goals']:
                    goals_str = ", ".join([f"{goal} ({count})" for goal, count in hp['top_goals']])
                    report.append(f"- Top Goals: {goals_str}")
                report.append(f"- Stress Patterns: Mental {hp['stress_patterns']['mental']:.1f}%, Physical {hp['stress_patterns']['physical']:.1f}%")
                report.append(f"- Key Symptoms: Bloating {hp['key_symptoms']['bloating']:.1f}%, Anxiety {hp['key_symptoms']['anxiety']:.1f}%")
                report.append("")
            
            report.append("**Marketing Recommendations**:")
            for rec in persona['marketing_recommendations']:
                report.append(f"- {rec}")
            report.append("")
        
        # Business Impact
        report.append("## Business Impact & Actionability")
        report.append("")
        report.append("### Key Insights")
        
        # Find most valuable segments
        high_value_personas = [(p['business_profile']['persona_name'], p['business_profile']['avg_ltv']) 
                              for p in personas.values() if p['business_profile']['avg_ltv'] > 300]
        high_value_personas.sort(key=lambda x: x[1], reverse=True)
        
        if high_value_personas:
            report.append(f"- **Highest Value Segment**: {high_value_personas[0][0]} (${high_value_personas[0][1]:.2f} average LTV)")
        
        # Quiz engagement insights
        quiz_engaged = [(p['business_profile']['persona_name'], p['business_profile']['quiz_participation']) 
                       for p in personas.values() if p['business_profile']['quiz_participation'] > 20]
        quiz_engaged.sort(key=lambda x: x[1], reverse=True)
        
        if quiz_engaged:
            report.append(f"- **Most Quiz-Engaged**: {quiz_engaged[0][0]} ({quiz_engaged[0][1]:.1f}% participation)")
        
        report.append("")
        
        # Save report
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f">> COMPREHENSIVE REPORT SAVED: {filename}")
        print(">> Analysis includes both behavioral clusters and health overlay insights")
        
        return filename
    
    # Utility methods
    def _standardize_matrix(self, matrix):
        """Standardize feature matrix"""
        if not matrix:
            return []
            
        n_samples = len(matrix)
        n_features = len(matrix[0])
        
        # Calculate means and stds
        means = []
        stds = []
        
        for j in range(n_features):
            feature_values = [matrix[i][j] for i in range(n_samples)]
            mean_val = sum(feature_values) / n_samples
            variance = sum((x - mean_val) ** 2 for x in feature_values) / (n_samples - 1)
            std_val = math.sqrt(variance)
            
            means.append(mean_val)
            stds.append(std_val)
        
        # Standardize
        standardized = []
        for i in range(n_samples):
            std_row = []
            for j in range(n_features):
                if stds[j] > 1e-10:
                    std_val = (matrix[i][j] - means[j]) / stds[j]
                else:
                    std_val = 0.0
                std_row.append(std_val)
            standardized.append(std_row)
        
        return standardized
    
    def _kmeans_clustering(self, data, k, max_iters=50):
        """K-means clustering implementation"""
        n_samples = len(data)
        n_features = len(data[0])
        
        # Initialize centroids randomly
        centroids = []
        for _ in range(k):
            centroid = []
            for j in range(n_features):
                feature_values = [data[i][j] for i in range(n_samples)]
                mean_val = sum(feature_values) / n_samples
                std_val = math.sqrt(sum((x - mean_val) ** 2 for x in feature_values) / n_samples)
                centroid.append(random.gauss(mean_val, std_val * 0.5))
            centroids.append(centroid)
        
        # K-means iterations
        for iteration in range(max_iters):
            # Assignment step
            assignments = []
            for point in data:
                distances = []
                for centroid in centroids:
                    distance = sum((point[j] - centroid[j]) ** 2 for j in range(n_features))
                    distances.append(distance)
                assignments.append(distances.index(min(distances)))
            
            # Update step
            new_centroids = []
            for cluster_id in range(k):
                cluster_points = [data[i] for i, assignment in enumerate(assignments) if assignment == cluster_id]
                if cluster_points:
                    new_centroid = []
                    for j in range(n_features):
                        new_centroid.append(sum(point[j] for point in cluster_points) / len(cluster_points))
                    new_centroids.append(new_centroid)
                else:
                    new_centroids.append(centroids[cluster_id])
            
            # Check convergence
            converged = True
            for i in range(k):
                for j in range(n_features):
                    if abs(centroids[i][j] - new_centroids[i][j]) > 1e-6:
                        converged = False
                        break
                if not converged:
                    break
            
            centroids = new_centroids
            if converged:
                break
        
        # Calculate inertia
        inertia = 0
        for i, point in enumerate(data):
            centroid = centroids[assignments[i]]
            inertia += sum((point[j] - centroid[j]) ** 2 for j in range(n_features))
        
        return assignments, centroids, inertia
    
    def _calculate_silhouette_sample(self, data, assignments, sample_size=500):
        """Calculate silhouette coefficient on a sample"""
        n_samples = len(data)
        
        if len(set(assignments)) < 2 or n_samples < 10:
            return 0.0
        
        # Sample for large datasets
        if n_samples > sample_size:
            indices = random.sample(range(n_samples), sample_size)
            sample_data = [data[i] for i in indices]
            sample_assignments = [assignments[i] for i in indices]
            return self._calculate_silhouette_full(sample_data, sample_assignments)
        else:
            return self._calculate_silhouette_full(data, assignments)
    
    def _calculate_silhouette_full(self, data, assignments):
        """Calculate silhouette coefficient"""
        n_samples = len(data)
        n_features = len(data[0])
        
        clusters = defaultdict(list)
        for i, cluster_id in enumerate(assignments):
            clusters[cluster_id].append(i)
        
        silhouette_scores = []
        for i in range(n_samples):
            cluster_id = assignments[i]
            same_cluster = clusters[cluster_id]
            
            if len(same_cluster) <= 1:
                silhouette_scores.append(0)
                continue
            
            # Mean distance to same cluster
            a = sum(math.sqrt(sum((data[i][f] - data[j][f]) ** 2 for f in range(n_features))) 
                   for j in same_cluster if i != j) / (len(same_cluster) - 1)
            
            # Mean distance to nearest other cluster
            b_scores = []
            for other_cluster_id, other_cluster in clusters.items():
                if other_cluster_id != cluster_id:
                    b_score = sum(math.sqrt(sum((data[i][f] - data[j][f]) ** 2 for f in range(n_features))) 
                                 for j in other_cluster) / len(other_cluster)
                    b_scores.append(b_score)
            
            if b_scores:
                b = min(b_scores)
                s = (b - a) / max(a, b) if max(a, b) > 0 else 0
                silhouette_scores.append(s)
            else:
                silhouette_scores.append(0)
        
        return sum(silhouette_scores) / len(silhouette_scores) if silhouette_scores else 0
    
    def _calculate_r_squared(self, data, assignments, centroids):
        """Calculate R-squared (variance explained)"""
        n_samples = len(data)
        n_features = len(data[0])
        
        # Overall centroid
        overall_centroid = []
        for j in range(n_features):
            overall_centroid.append(sum(data[i][j] for i in range(n_samples)) / n_samples)
        
        # Within-cluster sum of squares
        wcss = 0
        for i in range(n_samples):
            cluster_id = assignments[i]
            centroid = centroids[cluster_id]
            wcss += sum((data[i][j] - centroid[j]) ** 2 for j in range(n_features))
        
        # Total sum of squares
        tss = 0
        for i in range(n_samples):
            tss += sum((data[i][j] - overall_centroid[j]) ** 2 for j in range(n_features))
        
        return (tss - wcss) / tss if tss > 0 else 0
    
    def _safe_float(self, value, default=0.0):
        """Safely convert to float"""
        try:
            return float(value) if value and str(value).strip() else default
        except (ValueError, TypeError):
            return default
    
    def _safe_int(self, value, default=0):
        """Safely convert to int"""
        try:
            return int(float(value)) if value and str(value).strip() else default
        except (ValueError, TypeError):
            return default
    
    def _safe_bool(self, value):
        """Safely convert to bool"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ['true', 'yes', '1', 'y']
        try:
            return bool(int(value))
        except:
            return False

def main():
    """Execute two-stage segmentation analysis"""
    
    print("TWO-STAGE CUSTOMER SEGMENTATION WITH QUIZ OVERLAY")
    print("Stage 1: Behavioral clustering | Stage 2: Health pattern overlay")
    print("="*80)
    
    try:
        # Initialize analyzer
        analyzer = TwoStageSegmentationAnalyzer()
        
        # Execute two-stage analysis
        analyzer.load_and_engineer_data()
        
        print("\n" + "="*60)
        print("EXECUTING STAGE 1: BEHAVIORAL CLUSTERING")
        print("="*60)
        best_k, best_solution = analyzer.perform_stage1_clustering()
        cluster_profiles = analyzer.analyze_behavioral_clusters()
        
        print("\n" + "="*60)
        print("EXECUTING STAGE 2: QUIZ OVERLAY ANALYSIS")
        print("="*60)
        quiz_insights = analyzer.perform_stage2_quiz_overlay()
        
        print("\n" + "="*60)
        print("GENERATING CUSTOMER PERSONAS")
        print("="*60)
        personas = analyzer.generate_customer_personas()
        
        # Generate comprehensive report
        report_file = analyzer.generate_comprehensive_report()
        
        print("\n" + "="*80)
        print("TWO-STAGE ANALYSIS COMPLETE")
        print("="*80)
        print("✓ Stage 1: Behavioral clustering completed")
        print("✓ Stage 2: Quiz overlay analysis completed")
        print("✓ Customer personas generated")
        print("✓ Comprehensive report saved")
        print(f"✓ Report file: {report_file}")
        
        return analyzer.results
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()