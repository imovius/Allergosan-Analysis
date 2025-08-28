#!/usr/bin/env python3
"""
RFM (Recency, Frequency, Monetary) Customer Segmentation
A stable and interpretable approach to behavioral customer segmentation.

Author: Ian Movius
Date: January 2025
Methodology: RFM Scoring for Actionable Customer Segments
"""

import csv
import datetime
from collections import defaultdict

class RFMSegmenter:
    """
    Performs RFM segmentation on customer data.
    1. Calculates Recency, Frequency, Monetary values.
    2. Scores customers on each dimension.
    3. Assigns customers to actionable segments.
    """
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.customers = []
        self.rfm_data = []
        self.segments = defaultdict(list)
        
        # Assume the analysis is being run on a specific date for consistent recency
        self.snapshot_date = datetime.datetime(2026, 1, 1)

    def load_data(self):
        """Load customer data from the CSV file."""
        print("Loading customer data...")
        with open(self.data_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            self.customers = [row for row in reader]
        print(f"Loaded {len(self.customers)} customer records.")

    def calculate_rfm_values(self):
        """Calculate Recency, Frequency, and Monetary values for each customer."""
        print("Calculating RFM values...")
        
        for customer in self.customers:
            email = customer.get('email_key')
            if not email:
                continue
            
            # Recency
            last_order_date_str = customer.get('last_order_date')
            recency = (self.snapshot_date - self._parse_date(last_order_date_str)).days if last_order_date_str else 3650
            
            # Frequency
            frequency = self._safe_int(customer.get('order_count', 0))
            
            # Monetary
            monetary = self._safe_float(customer.get('net_ltv', 0))
            
            if frequency > 0:
                self.rfm_data.append({
                    'email': email,
                    'recency': recency,
                    'frequency': frequency,
                    'monetary': monetary
                })
        print(f"Calculated RFM values for {len(self.rfm_data)} customers with orders.")

    def calculate_rfm_scores(self):
        """Score customers on R, F, and M dimensions using quintiles."""
        print("Scoring customers based on RFM values...")
        if not self.rfm_data:
            return

        # Sort by each dimension to prepare for quintile scoring
        # Recency is inverse: lower is better
        self.rfm_data.sort(key=lambda x: x['recency'], reverse=True)
        r_quintile_size = len(self.rfm_data) // 5
        for i, customer in enumerate(self.rfm_data):
            customer['r_score'] = min(5, (i // r_quintile_size) + 1)
            
        # Frequency and Monetary: higher is better
        self.rfm_data.sort(key=lambda x: x['frequency'])
        f_quintile_size = len(self.rfm_data) // 5
        for i, customer in enumerate(self.rfm_data):
            customer['f_score'] = min(5, (i // f_quintile_size) + 1)

        self.rfm_data.sort(key=lambda x: x['monetary'])
        m_quintile_size = len(self.rfm_data) // 5
        for i, customer in enumerate(self.rfm_data):
            customer['m_score'] = min(5, (i // m_quintile_size) + 1)
        
        # Combine scores
        for customer in self.rfm_data:
            customer['rfm_score'] = f"{customer['r_score']}{customer['f_score']}{customer['m_score']}"

        print("RFM scoring complete.")

    def assign_segments(self):
        """Assign customers to descriptive segments based on RFM scores."""
        print("Assigning customers to segments...")
        
        # Define segment mapping based on common RFM score patterns
        # R, F, M scores from 1 (worst) to 5 (best)
        segment_map = {
            '555': 'Champions',
            '554': 'Champions', '545': 'Champions', '455': 'Champions',
            '544': 'Loyal Customers', '454': 'Loyal Customers', '445': 'Loyal Customers', '444': 'Loyal Customers',
            '553': 'Potential Loyalist', '543': 'Potential Loyalist', '453': 'Potential Loyalist', '443': 'Potential Loyalist', '535': 'Potential Loyalist',
            '534': 'Promising', '533': 'Promising', '435': 'Promising', '434': 'Promising', '433': 'Promising',
            '532': 'New Customers', '531': 'New Customers', '432': 'New Customers', '431': 'New Customers',
            '355': 'At Risk', '354': 'At Risk', '345': 'At Risk', '344': 'At Risk', '335': 'At Risk',
            '333': 'About to Sleep', '332': 'About to Sleep', '323': 'About to Sleep',
            '2_': 'Hibernating', # Any score starting with 2
            '1_': 'Lost' # Any score starting with 1
        }

        for customer in self.rfm_data:
            rfm_score = customer['rfm_score']
            segment = "Other" # Default
            
            # Find matching segment
            if rfm_score in segment_map:
                segment = segment_map[rfm_score]
            elif rfm_score.startswith('2'):
                 segment = "Hibernating"
            elif rfm_score.startswith('1'):
                 segment = "Lost"

            customer['segment'] = segment
            self.segments[segment].append(customer)
        
        print("Segmentation assignment complete.")

    def generate_report(self):
        """Generate a summary report of the RFM segmentation."""
        print("Generating RFM segmentation report...")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rfm_segmentation_report_{timestamp}.md"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# RFM Customer Segmentation Report\n\n")
            f.write(f"*Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            f.write("This report details the customer segments created using the RFM (Recency, Frequency, Monetary) model. This is a stable, behavior-based approach that yields actionable insights.\n\n")
            f.write("## Segment Summary\n\n")
            f.write("| Segment             | Customer Count | Avg Recency (Days) | Avg Frequency | Avg Monetary Value |\n")
            f.write("|---------------------|----------------|--------------------|---------------|--------------------|\n")

            sorted_segments = sorted(self.segments.items(), key=lambda item: len(item[1]), reverse=True)

            for segment, customers in sorted_segments:
                count = len(customers)
                avg_recency = sum(c['recency'] for c in customers) / count
                avg_frequency = sum(c['frequency'] for c in customers) / count
                avg_monetary = sum(c['monetary'] for c in customers) / count
                f.write(f"| {segment:<19} | {count:<14} | {avg_recency:<18.1f} | {avg_frequency:<13.1f} | ${avg_monetary:<18.2f} |\n")
            
            f.write("\n## Segment Definitions\n\n")
            f.write("- **Champions**: Your best customers. Bought recently, buy often, and spend the most.\n")
            f.write("- **Loyal Customers**: Frequent and high-value customers, but may not have purchased as recently.\n")
            f.write("- **Potential Loyalist**: Recent customers with average frequency and spend.\n")
            f.write("- **Promising**: Recent shoppers, but haven't spent much.\n")
            f.write("- **New Customers**: Your newest customers.\n")
            f.write("- **At Risk**: Customers who used to be frequent/high-value but haven't purchased in a while.\n")
            f.write("- **About to Sleep**: Below-average recency, frequency, and monetary values.\n")
            f.write("- **Hibernating**: Low scores across the board. May be slipping away.\n")
            f.write("- **Lost**: Your coldest leads. Lowest recency, frequency, and monetary scores.\n")

        print(f"Report saved to {filename}")
        return filename

    def run_segmentation(self):
        """Execute the full RFM segmentation pipeline."""
        self.load_data()
        self.calculate_rfm_values()
        self.calculate_rfm_scores()
        self.assign_segments()
        self.generate_report()

    def _safe_float(self, value):
        try: return float(value) if value else 0.0
        except (ValueError, TypeError): return 0.0
    
    def _safe_int(self, value):
        try: return int(float(value)) if value else 0
        except (ValueError, TypeError): return 0

    def _parse_date(self, date_str):
        """Parse date from common formats."""
        if not date_str:
            return self.snapshot_date
        # Handle format '2022-09-28 20:46:11.000000 UTC'
        if ' UTC' in date_str:
            date_str = date_str.replace(' UTC', '')
        
        for fmt in ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y'):
            try:
                return datetime.datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return self.snapshot_date

def main():
    """Execute the RFM segmentation analysis."""
    print("="*80)
    print("Starting RFM Customer Segmentation Analysis")
    print("="*80)
    
    segmenter = RFMSegmenter('raw_data_v2.csv')
    segmenter.run_segmentation()

    print("\nRFM Analysis Complete.")
    print("="*80)

if __name__ == "__main__":
    main()
