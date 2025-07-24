"""
Unit Tests for Data Processing Module

This module contains unit tests for the feature engineering and data processing
functions in the credit risk scoring system.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import FeatureEngineer, load_and_process_data


class TestFeatureEngineer(unittest.TestCase):
    """Test cases for FeatureEngineer class"""
    
    def setUp(self):
        """Set up test data and feature engineer instance"""
        self.fe = FeatureEngineer()
        
        # Create sample transaction data
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        self.sample_data = pd.DataFrame({
            'customer_id': np.random.choice(['CUST_001', 'CUST_002', 'CUST_003'], 100),
            'transaction_date': np.random.choice(dates, 100),
            'amount': np.random.uniform(10, 1000, 100),
            'category': np.random.choice(['grocery', 'electronics', 'clothing'], 100)
        })
        
        # Create sample RFM data
        self.sample_rfm = pd.DataFrame({
            'customer_id': ['CUST_001', 'CUST_002', 'CUST_003'],
            'recency': [5, 30, 60],
            'frequency': [20, 10, 5],
            'monetary': [2000, 1000, 500]
        })
    
    def test_create_datetime_features(self):
        """Test datetime feature creation"""
        result = self.fe.create_datetime_features(self.sample_data, 'transaction_date')
        
        # Check if new columns are created
        expected_columns = ['hour', 'day_of_week', 'day_of_month', 'month', 'year', 'is_weekend']
        for col in expected_columns:
            self.assertIn(col, result.columns)
        
        # Check data types
        self.assertTrue(result['hour'].dtype in [np.int32, np.int64])
        self.assertTrue(result['is_weekend'].dtype in [np.int32, np.int64])
        
        # Check value ranges
        self.assertTrue(result['hour'].between(0, 23).all())
        self.assertTrue(result['day_of_week'].between(0, 6).all())
        self.assertTrue(result['is_weekend'].isin([0, 1]).all())
    
    def test_create_aggregate_features(self):
        """Test aggregate feature creation"""
        result = self.fe.create_aggregate_features(self.sample_data, 'customer_id', 'amount')
        
        # Check if all customers are included
        self.assertEqual(len(result), self.sample_data['customer_id'].nunique())
        
        # Check column names
        expected_columns = [
            'customer_id', 'transaction_count', 'total_amount', 
            'avg_amount', 'std_amount', 'min_amount', 'max_amount'
        ]
        for col in expected_columns:
            self.assertIn(col, result.columns)
        
        # Check that values make sense
        self.assertTrue((result['transaction_count'] > 0).all())
        self.assertTrue((result['total_amount'] >= 0).all())
        self.assertTrue((result['avg_amount'] >= 0).all())
    
    def test_calculate_rfm_features(self):
        """Test RFM feature calculation"""
        result = self.fe.calculate_rfm_features(
            self.sample_data, 'customer_id', 'transaction_date', 'amount'
        )
        
        # Check if all customers are included
        self.assertEqual(len(result), self.sample_data['customer_id'].nunique())
        
        # Check column names
        expected_columns = ['customer_id', 'recency', 'frequency', 'monetary']
        for col in expected_columns:
            self.assertIn(col, result.columns)
        
        # Check data types and ranges
        self.assertTrue((result['recency'] >= 0).all())
        self.assertTrue((result['frequency'] > 0).all())
        self.assertTrue((result['monetary'] >= 0).all())
    
    def test_create_proxy_target(self):
        """Test proxy target creation"""
        result = self.fe.create_proxy_target(self.sample_rfm.copy())
        
        # Check if new columns are created
        self.assertIn('cluster', result.columns)
        self.assertIn('is_high_risk', result.columns)
        
        # Check cluster values
        self.assertTrue(result['cluster'].isin([0, 1, 2]).all())
        
        # Check binary risk labels
        self.assertTrue(result['is_high_risk'].isin([0, 1]).all())
        
        # Check that at least one customer is labeled as high risk
        self.assertTrue(result['is_high_risk'].sum() >= 1)
    
    def test_encode_categorical_features(self):
        """Test categorical feature encoding"""
        result = self.fe.encode_categorical_features(self.sample_data, ['category'])
        
        # Check if encoded column is created
        self.assertIn('category_encoded', result.columns)
        
        # Check that encoded values are numeric
        self.assertTrue(result['category_encoded'].dtype in [np.int32, np.int64])
        
        # Check that original column is preserved
        self.assertIn('category', result.columns)
    
    def test_handle_missing_data(self):
        """Test missing data handling"""
        # Create data with missing values
        data_with_na = self.sample_data.copy()
        data_with_na.loc[0:5, 'amount'] = np.nan
        
        result = self.fe.handle_missing_data(data_with_na, strategy='median')
        
        # Check that no missing values remain in numeric columns
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        self.assertEqual(result[numeric_cols].isnull().sum().sum(), 0)


class TestDataProcessing(unittest.TestCase):
    """Test cases for data processing functions"""
    
    def setUp(self):
        """Set up test data"""
        # Create a temporary CSV file for testing
        self.test_file = 'test_data.csv'
        
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        test_data = pd.DataFrame({
            'customer_id': np.random.choice(['CUST_001', 'CUST_002', 'CUST_003'], 100),
            'transaction_date': np.random.choice(dates, 100),
            'amount': np.random.uniform(10, 1000, 100)
        })
        
        test_data.to_csv(self.test_file, index=False)
    
    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_load_and_process_data(self):
        """Test the main data processing function"""
        try:
            processed_df, rfm_with_risk = load_and_process_data(self.test_file)
            
            # Check that data is processed successfully
            self.assertIsInstance(processed_df, pd.DataFrame)
            self.assertIsInstance(rfm_with_risk, pd.DataFrame)
            
            # Check that required columns exist
            self.assertIn('is_high_risk', processed_df.columns)
            self.assertIn('recency', processed_df.columns)
            self.assertIn('frequency', processed_df.columns)
            self.assertIn('monetary', processed_df.columns)
            
            # Check that RFM data has risk labels
            self.assertIn('is_high_risk', rfm_with_risk.columns)
            self.assertTrue(rfm_with_risk['is_high_risk'].isin([0, 1]).all())
            
        except Exception as e:
            self.fail(f"load_and_process_data raised an exception: {e}")


class TestFeatureValidation(unittest.TestCase):
    """Test cases for feature validation and edge cases"""
    
    def setUp(self):
        self.fe = FeatureEngineer()
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframes"""
        empty_df = pd.DataFrame()
        
        # Should not raise exceptions for empty dataframes
        try:
            result = self.fe.handle_missing_data(empty_df)
            self.assertEqual(len(result), 0)
        except Exception as e:
            self.fail(f"handle_missing_data failed with empty dataframe: {e}")
    
    def test_single_customer_rfm(self):
        """Test RFM calculation with single customer"""
        single_customer_data = pd.DataFrame({
            'customer_id': ['CUST_001'] * 5,
            'transaction_date': pd.date_range('2023-01-01', periods=5),
            'amount': [100, 200, 150, 300, 250]
        })
        
        result = self.fe.calculate_rfm_features(
            single_customer_data, 'customer_id', 'transaction_date', 'amount'
        )
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['customer_id'], 'CUST_001')
        self.assertTrue(result.iloc[0]['frequency'] == 5)
        self.assertTrue(result.iloc[0]['monetary'] == 1000)
    
    def test_duplicate_transactions(self):
        """Test handling of duplicate transactions"""
        duplicate_data = pd.DataFrame({
            'customer_id': ['CUST_001', 'CUST_001', 'CUST_001'],
            'transaction_date': ['2023-01-01', '2023-01-01', '2023-01-02'],
            'amount': [100, 100, 200]
        })
        
        # Should handle duplicates without errors
        result = self.fe.create_aggregate_features(duplicate_data, 'customer_id', 'amount')
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['transaction_count'], 3)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTest(unittest.makeSuite(TestFeatureEngineer))
    test_suite.addTest(unittest.makeSuite(TestDataProcessing))
    test_suite.addTest(unittest.makeSuite(TestFeatureValidation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    # Exit with appropriate code
    exit_code = 0 if result.wasSuccessful() else 1
    exit(exit_code)
