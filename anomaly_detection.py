"""
XGBoost SHAP Anomaly Detection and Root Cause Analysis System

This program uses XGBoost for anomaly detection and SHAP for explainability
to identify root causes of anomalies in engineered features.

Required packages:
pip install xgboost shap pandas numpy scikit-learn matplotlib seaborn textblob
"""

import pandas as pd
import numpy as np
import warnings
import re
from dateutil.parser import parse
from textblob import TextBlob
warnings.filterwarnings('ignore')

# Core ML libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

# SHAP for explainability
import shap

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import the data processing pipeline
from feature_engineering import process_data_source

class XGBoostSHAPAnomalyDetector:
    """
    Advanced anomaly detection system using XGBoost and SHAP for root cause analysis.
    """
    
    def __init__(self, 
                 contamination=0.1, 
                 xgb_params=None,
                 random_state=42,
                 verbose=True):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies in the dataset
            xgb_params: XGBoost parameters dictionary
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information
        """
        self.contamination = contamination
        self.random_state = random_state
        self.verbose = verbose
        
        # Default XGBoost parameters optimized for anomaly detection
        if xgb_params is None:
            self.xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': random_state,
                'n_jobs': -1,
                'verbosity': 0 if not verbose else 1,
                'early_stopping_rounds': 10  # Move early stopping to constructor
            }
        else:
            self.xgb_params = xgb_params
            
        # Initialize models and components
        self.isolation_forest = None
        self.xgb_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.shap_explainer = None
        self.feature_names = None
        self.anomaly_threshold = None
        
    def _log(self, message):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
            
    def _prepare_features(self, df, fit_encoders=True):
        """
        Prepare features for modeling by handling different data types.
        
        Args:
            df: DataFrame with engineered features
            fit_encoders: Whether to fit label encoders (True for training, False for inference)
            
        Returns:
            Processed DataFrame ready for modeling
        """
        self._log("Preparing features for modeling...")
        
        df_processed = df.copy()
        
        # Handle different column types
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                # Handle categorical/text columns
                if fit_encoders:
                    le = LabelEncoder()
                    # Fill NaN values before encoding
                    df_processed[col] = df_processed[col].fillna('unknown')
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        df_processed[col] = df_processed[col].fillna('unknown')
                        # Handle unseen categories
                        try:
                            df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))
                        except ValueError:
                            # Handle unseen categories by encoding as -1
                            unique_vals = df_processed[col].unique()
                            known_vals = self.label_encoders[col].classes_
                            for val in unique_vals:
                                if val not in known_vals:
                                    df_processed.loc[df_processed[col] == val, col] = -1
                            df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))
            else:
                # Handle numeric columns
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        # Fill remaining NaN values
        df_processed = df_processed.fillna(0)
        
        # Handle infinite values
        df_processed = df_processed.replace([np.inf, -np.inf], 0)
        
        return df_processed
    
    def _detect_anomalies_isolation_forest(self, X):
        """
        Use Isolation Forest to detect anomalies and create labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Binary anomaly labels (1 for anomaly, 0 for normal)
        """
        self._log("Running Isolation Forest for anomaly detection...")
        
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        anomaly_labels = self.isolation_forest.fit_predict(X)
        # Convert to binary (1 for anomaly, 0 for normal)
        anomaly_labels = (anomaly_labels == -1).astype(int)
        
        self._log(f"Detected {anomaly_labels.sum()} anomalies ({anomaly_labels.mean():.2%} of data)")
        
        return anomaly_labels
    
    def _train_xgboost_classifier(self, X, y):
        """
        Train XGBoost classifier for anomaly prediction.
        
        Args:
            X: Feature matrix
            y: Anomaly labels
            
        Returns:
            Trained XGBoost model
        """
        self._log("Training XGBoost classifier...")
        
        # Handle class imbalance
        scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)  # Avoid division by zero
        self.xgb_params['scale_pos_weight'] = scale_pos_weight
        
        # Check if we have enough samples for stratified split
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class_count = min(class_counts)
        
        if min_class_count < 2 or len(X) < 10:
            # For small datasets or datasets with very few anomalies, skip validation
            self._log("Dataset too small for validation split, training on full dataset")
            
            # Create a simpler model for small datasets
            simple_params = self.xgb_params.copy()
            simple_params.update({
                'n_estimators': min(50, len(X)),
                'max_depth': min(3, self.xgb_params.get('max_depth', 6)),
                'learning_rate': 0.3,
            })
            # Remove early stopping for small datasets
            simple_params.pop('early_stopping_rounds', None)
            
            self.xgb_model = xgb.XGBClassifier(**simple_params)
            self.xgb_model.fit(X, y)
            
            # Simple evaluation on training data
            y_pred_proba = self.xgb_model.predict_proba(X)[:, 1]
            try:
                auc_score = roc_auc_score(y, y_pred_proba)
                self._log(f"Training AUC: {auc_score:.3f}")
            except ValueError:
                self._log("Cannot compute AUC - insufficient class diversity")
            
        else:
            # Standard train-validation split for larger datasets
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=self.random_state, stratify=y
                )
            except ValueError:
                # Fallback to non-stratified split if stratification fails
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=self.random_state
                )
            
            # Train model
            self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
            
            # Prepare evaluation set for early stopping
            eval_set = [(X_val, y_val)]
            
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            
            # Evaluate model
            y_pred = self.xgb_model.predict(X_val)
            y_pred_proba = self.xgb_model.predict_proba(X_val)[:, 1]
            
            try:
                auc_score = roc_auc_score(y_val, y_pred_proba)
                self._log(f"Validation AUC: {auc_score:.3f}")
                
                if self.verbose:
                    print("\nClassification Report:")
                    print(classification_report(y_val, y_pred, zero_division=0))
            except ValueError as e:
                self._log(f"Cannot compute validation metrics: {e}")
        
        return self.xgb_model
    
    def _initialize_shap_explainer(self, X_sample):
        """
        Initialize SHAP explainer for the trained model.
        
        Args:
            X_sample: Sample of training data for SHAP explainer
        """
        self._log("Initializing SHAP explainer...")
        
        # Use a sample for faster SHAP computation
        sample_size = min(100, len(X_sample))
        X_shap_sample = X_sample.sample(n=sample_size, random_state=self.random_state)
        
        # Initialize TreeExplainer for XGBoost
        self.shap_explainer = shap.TreeExplainer(self.xgb_model)
        
        self._log("SHAP explainer initialized successfully")
    
    def fit(self, features_df, data_source_name="Unknown"):
        """
        Fit the anomaly detection system.
        
        Args:
            features_df: DataFrame with engineered features
            data_source_name: Name of the data source for logging
            
        Returns:
            Self for method chaining
        """
        self._log(f"\n{'='*60}")
        self._log(f"Training Anomaly Detector on: {data_source_name}")
        self._log(f"{'='*60}")
        self._log(f"Input shape: {features_df.shape}")
        
        # Store feature names
        self.feature_names = features_df.columns.tolist()
        
        # Prepare features
        X = self._prepare_features(features_df, fit_encoders=True)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Detect anomalies using Isolation Forest
        anomaly_labels = self._detect_anomalies_isolation_forest(X_scaled)
        
        # Train XGBoost classifier
        self._train_xgboost_classifier(X_scaled, anomaly_labels)
        
        # Initialize SHAP explainer
        self._initialize_shap_explainer(X_scaled)
        
        # Set anomaly threshold based on training data
        anomaly_probs = self.xgb_model.predict_proba(X_scaled)[:, 1]
        self.anomaly_threshold = np.percentile(anomaly_probs, (1 - self.contamination) * 100)
        
        self._log(f"Training completed successfully!")
        self._log(f"Anomaly threshold set to: {self.anomaly_threshold:.3f}")
        
        return self
    
    def predict_anomalies(self, features_df):
        """
        Predict anomalies on new data.
        
        Args:
            features_df: DataFrame with engineered features
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if self.xgb_model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        # Prepare features
        X = self._prepare_features(features_df, fit_encoders=False)
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        # Predict
        anomaly_probs = self.xgb_model.predict_proba(X_scaled)[:, 1]
        anomaly_predictions = (anomaly_probs > self.anomaly_threshold).astype(int)
        
        return {
            'predictions': anomaly_predictions,
            'probabilities': anomaly_probs,
            'threshold': self.anomaly_threshold,
            'features_processed': X_scaled
        }
    
    def explain_anomalies(self, features_df, top_n_features=10, save_plots=False):
        """
        Explain anomalies using SHAP values.
        
        Args:
            features_df: DataFrame with engineered features
            top_n_features: Number of top features to show in explanations
            save_plots: Whether to save SHAP plots
            
        Returns:
            Dictionary with SHAP explanations and root cause analysis
        """
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not initialized. Call fit() first.")
        
        self._log("\nGenerating SHAP explanations for anomalies...")
        
        # Get predictions
        results = self.predict_anomalies(features_df)
        X_scaled = results['features_processed']
        anomaly_predictions = results['predictions']
        anomaly_probs = results['probabilities']
        
        # Calculate SHAP values
        shap_values = self.shap_explainer.shap_values(X_scaled)
        
        # Find anomalies
        anomaly_indices = np.where(anomaly_predictions == 1)[0]
        
        if len(anomaly_indices) == 0:
            self._log("No anomalies detected in the provided data.")
            return {
                'anomaly_count': 0,
                'anomaly_indices': [],
                'shap_values': shap_values,
                'root_causes': {},
                'feature_importance_summary': {},
                'anomaly_probabilities': np.array([]),
                'X_processed': X_scaled
            }
        
        self._log(f"Analyzing {len(anomaly_indices)} detected anomalies...")
        
        # Analyze root causes for each anomaly
        root_causes = {}
        
        for idx in anomaly_indices:
            # Get SHAP values for this anomaly
            instance_shap = shap_values[idx]
            instance_features = X_scaled.iloc[idx]
            
            # Get top contributing features
            feature_importance = list(zip(X_scaled.columns, instance_shap, instance_features))
            feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
            
            top_features = feature_importance[:top_n_features]
            
            root_causes[idx] = {
                'anomaly_probability': anomaly_probs[idx],
                'top_contributing_features': top_features,
                'feature_values': instance_features.to_dict(),
                'shap_values': dict(zip(X_scaled.columns, instance_shap))
            }
        
        # Create summary plots if requested
        if save_plots:
            self._create_shap_plots(shap_values, X_scaled, anomaly_indices)
        
        # Generate summary statistics
        all_shap_values = shap_values[anomaly_indices]
        feature_importance_summary = {}
        
        for i, feature in enumerate(X_scaled.columns):
            feature_shap_values = all_shap_values[:, i]
            feature_importance_summary[feature] = {
                'mean_abs_shap': np.mean(np.abs(feature_shap_values)),
                'mean_shap': np.mean(feature_shap_values),
                'std_shap': np.std(feature_shap_values)
            }
        
        # Sort features by importance
        sorted_features = sorted(feature_importance_summary.items(), 
                               key=lambda x: x[1]['mean_abs_shap'], reverse=True)
        
        self._log(f"\nTop {min(10, len(sorted_features))} most important features for anomalies:")
        for i, (feature, stats) in enumerate(sorted_features[:10]):
            self._log(f"  {i+1}. {feature}: {stats['mean_abs_shap']:.4f} (avg |SHAP|)")
        
        return {
            'anomaly_count': len(anomaly_indices),
            'anomaly_indices': anomaly_indices,
            'anomaly_probabilities': anomaly_probs[anomaly_indices],
            'shap_values': shap_values,
            'root_causes': root_causes,
            'feature_importance_summary': dict(sorted_features),
            'X_processed': X_scaled
        }
    
    def _create_shap_plots(self, shap_values, X_scaled, anomaly_indices):
        """Create and save SHAP visualization plots."""
        try:
            # Summary plot for all data
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_scaled, show=False, max_display=20)
            plt.title("SHAP Summary Plot - All Data")
            plt.tight_layout()
            plt.savefig('shap_summary_all.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Summary plot for anomalies only
            if len(anomaly_indices) > 0:
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values[anomaly_indices], 
                                X_scaled.iloc[anomaly_indices], 
                                show=False, max_display=20)
                plt.title("SHAP Summary Plot - Anomalies Only")
                plt.tight_layout()
                plt.savefig('shap_summary_anomalies.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            self._log("SHAP plots saved successfully")
            
        except Exception as e:
            self._log(f"Error creating SHAP plots: {e}")
    
    def generate_anomaly_report(self, features_df, data_source_name="Unknown"):
        """
        Generate a comprehensive anomaly detection report.
        
        Args:
            features_df: DataFrame with engineered features
            data_source_name: Name of the data source
            
        Returns:
            Comprehensive report dictionary
        """
        self._log(f"\n{'='*60}")
        self._log(f"Generating Anomaly Report for: {data_source_name}")
        self._log(f"{'='*60}")
        
        # Get explanations
        explanations = self.explain_anomalies(features_df)
        
        # Generate report
        report = {
            'data_source': data_source_name,
            'total_records': len(features_df),
            'anomaly_count': explanations['anomaly_count'],
            'anomaly_rate': explanations['anomaly_count'] / len(features_df),
            'anomaly_indices': list(explanations['anomaly_indices']),
            'top_root_cause_features': list(explanations['feature_importance_summary'].keys())[:10],
            'detailed_anomalies': {},
            'summary_statistics': {
                'avg_anomaly_probability': np.mean(explanations['anomaly_probabilities']) if len(explanations['anomaly_probabilities']) > 0 else 0,
                'max_anomaly_probability': np.max(explanations['anomaly_probabilities']) if len(explanations['anomaly_probabilities']) > 0 else 0,
                'anomaly_threshold': self.anomaly_threshold
            }
        }
        
        # Add detailed information for each anomaly
        for idx, root_cause_info in explanations['root_causes'].items():
            report['detailed_anomalies'][idx] = {
                'probability': root_cause_info['anomaly_probability'],
                'top_3_causes': [
                    {
                        'feature': feature,
                        'shap_value': shap_val,
                        'feature_value': feat_val,
                        'contribution': 'increases' if shap_val > 0 else 'decreases'
                    }
                    for feature, shap_val, feat_val in root_cause_info['top_contributing_features'][:3]
                ]
            }
        
        # Print summary
        self._log(f"\nANOMALY DETECTION SUMMARY:")
        self._log(f"- Total records analyzed: {report['total_records']}")
        self._log(f"- Anomalies detected: {report['anomaly_count']}")
        self._log(f"- Anomaly rate: {report['anomaly_rate']:.2%}")
        self._log(f"- Average anomaly probability: {report['summary_statistics']['avg_anomaly_probability']:.3f}")
        
        if report['anomaly_count'] > 0:
            self._log(f"\nTOP ROOT CAUSE FEATURES:")
            for i, feature in enumerate(report['top_root_cause_features'][:5]):
                self._log(f"  {i+1}. {feature}")
        
        return report


def generate_test_data():
    """Generate various test datasets to cover edge cases."""
    
    test_cases = {}
    
    # Test Case 1: Normal numeric data with some anomalies
    np.random.seed(42)
    normal_data = pd.DataFrame({
        'revenue': np.random.normal(10000, 2000, 95).tolist() + [50000, 60000, -5000, 100000, 80000],  # 5 anomalies
        'score': np.random.uniform(1, 5, 95).tolist() + [0, 6, -1, 10, 15],  # 5 anomalies
        'customer_id': [f'CUST_{i:04d}' for i in range(100)],
        'status': ['active'] * 80 + ['inactive'] * 15 + ['suspended'] * 5,
        'signup_date': pd.date_range('2023-01-01', periods=100, freq='D')
    })
    test_cases['normal_numeric'] = normal_data
    
    # Test Case 2: High cardinality categorical data
    categories = [f'category_{i}' for i in range(50)]
    high_cardinality = pd.DataFrame({
        'product_category': np.random.choice(categories, 200),
        'price': np.random.lognormal(3, 1, 200),
        'user_email': [f'user{i}@example.com' for i in range(200)],
        'rating': np.random.choice([1, 2, 3, 4, 5], 200, p=[0.1, 0.1, 0.2, 0.3, 0.3])
    })
    test_cases['high_cardinality'] = high_cardinality
    
    # Test Case 3: Mixed data types with missing values
    mixed_data = pd.DataFrame({
        'amount': [100, 200, None, 400, 50000, 600, None, 800],  # Contains NaN and anomaly
        'phone': ['123-456-7890', '987-654-3210', None, '555-0123', 'invalid', '111-222-3333', '444-555-6666', None],
        'url': ['https://example.com', 'http://test.org', None, 'invalid_url', 'https://anomaly-site-with-very-long-name.com', 'https://normal.com', None, 'ftp://old.site'],
        'percentage': ['85%', '92%', '15%', None, '150%', '88%', '91%', '87%'],  # 150% is anomaly
        'boolean_val': [True, False, True, None, True, False, True, True]
    })
    test_cases['mixed_with_nulls'] = mixed_data
    
    # Test Case 4: Text data with sentiment
    text_data = pd.DataFrame({
        'feedback': [
            'Great product, very satisfied!',
            'Good service, recommend it.',
            'Average experience, could be better.',
            'Terrible service, waste of money! Extremely disappointed and angry!',  # Negative anomaly
            'Excellent quality, fast delivery.',
            'Okay product, nothing special.',
            'URGENT: Critical system failure! Everything is broken and not working at all!',  # Urgent anomaly
            'Nice design, user-friendly.',
            'Poor quality, returned immediately.',
            'Outstanding service, exceeded expectations!'
        ],
        'rating': [5, 4, 3, 1, 5, 3, 1, 4, 2, 5],
        'response_time_hours': [1, 2, 4, 48, 1, 3, 0.5, 2, 72, 1]  # 48 and 72 are anomalies
    })
    test_cases['text_sentiment'] = text_data
    
    # Test Case 5: Time series data
    dates = pd.date_range('2023-01-01', periods=50, freq='D')
    time_series = pd.DataFrame({
        'timestamp': dates,
        'daily_sales': np.random.normal(1000, 200, 50),
        'visitors': np.random.poisson(100, 50),
        'conversion_rate': np.random.beta(2, 8, 50) * 100  # Beta distribution for realistic conversion rates
    })
    # Add some anomalies
    time_series.loc[10, 'daily_sales'] = 5000  # Sales spike
    time_series.loc[25, 'visitors'] = 500      # Visitor spike
    time_series.loc[40, 'conversion_rate'] = 50  # High conversion rate
    test_cases['time_series'] = time_series
    
    # Test Case 6: Edge case - very small dataset
    small_data = pd.DataFrame({
        'value1': [1, 2, 100],  # 100 is anomaly in small dataset
        'value2': [10, 20, 30],
        'category': ['A', 'B', 'C']
    })
    test_cases['small_dataset'] = small_data
    
    # Test Case 7: Edge case - single column
    single_column = pd.DataFrame({
        'single_feature': list(range(20)) + [1000]  # 1000 is anomaly
    })
    test_cases['single_column'] = single_column
    
    return test_cases


def run_comprehensive_tests():
    """Run comprehensive tests covering all edge cases."""
    
    print("Generating test datasets...")
    test_cases = generate_test_data()
    
    results = {}
    
    for test_name, raw_data in test_cases.items():
        try:
            print(f"\n{'='*80}")
            print(f"TESTING: {test_name.upper()}")
            print(f"{'='*80}")
            
            # Process data using the pipeline
            features, types = process_data_source(test_name, raw_data)
            
            if features.empty:
                print(f"❌ No features generated for {test_name}")
                results[test_name] = {'status': 'failed', 'error': 'No features generated'}
                continue
            
            # Initialize and train anomaly detector
            detector = XGBoostSHAPAnomalyDetector(
                contamination=0.1,
                verbose=True
            )
            
            # Train the model
            detector.fit(features, test_name)
            
            # Generate anomaly report
            report = detector.generate_anomaly_report(features, test_name)
            
            # Store results
            results[test_name] = {
                'status': 'success',
                'original_shape': raw_data.shape,
                'features_shape': features.shape,
                'data_types': types,
                'anomaly_count': report['anomaly_count'],
                'anomaly_rate': report['anomaly_rate'],
                'top_features': report['top_root_cause_features'][:3]
            }
            
            print(f"✅ {test_name} completed successfully!")
            
        except Exception as e:
            print(f"❌ Error in {test_name}: {str(e)}")
            results[test_name] = {'status': 'failed', 'error': str(e)}
    
    # Print summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    successful_tests = sum(1 for r in results.values() if r['status'] == 'success')
    total_tests = len(results)
    
    print(f"✅ Successful tests: {successful_tests}/{total_tests}")
    print(f"❌ Failed tests: {total_tests - successful_tests}/{total_tests}")
    
    print(f"\nDetailed Results:")
    for test_name, result in results.items():
        if result['status'] == 'success':
            print(f"  ✅ {test_name}: {result['anomaly_count']} anomalies ({result['anomaly_rate']:.1%})")
        else:
            print(f"  ❌ {test_name}: {result.get('error', 'Unknown error')}")
    
    return results


def main():
    """Main function demonstrating the anomaly detection system."""
    
    print("XGBoost SHAP Anomaly Detection System")
    print("="*50)
    
    # Run comprehensive tests
    test_results = run_comprehensive_tests()
    
    # Demonstrate usage with a specific example
    print(f"\n{'='*80}")
    print("DETAILED EXAMPLE: E-commerce Transaction Analysis")
    print(f"{'='*80}")
    
    # Create a realistic e-commerce dataset
    np.random.seed(42)
    ecommerce_data = pd.DataFrame({
        'transaction_id': [f'TXN_{i:06d}' for i in range(1000, 1500)],
        'amount': np.random.lognormal(4, 1, 500),  # Realistic transaction amounts
        'customer_email': [f'customer{i}@{"".join(np.random.choice(["gmail.com", "yahoo.com", "company.com"], p=[0.4, 0.3, 0.3]))}' for i in range(500)],
        'product_category': np.random.choice(['electronics', 'clothing', 'books', 'home'], 500, p=[0.3, 0.3, 0.2, 0.2]),
        'discount_percent': np.random.beta(2, 8, 500) * 100,
        'customer_rating': np.random.choice([1, 2, 3, 4, 5], 500, p=[0.05, 0.05, 0.2, 0.4, 0.3]),
        'transaction_date': pd.date_range('2023-01-01', periods=500, freq='H'),
        'customer_tier': np.random.choice(['bronze', 'silver', 'gold', 'platinum'], 500, p=[0.4, 0.3, 0.2, 0.1]),
        'payment_method': np.random.choice(['credit_card', 'debit_card', 'paypal', 'crypto'], 500, p=[0.5, 0.3, 0.15, 0.05])
    })
    
    # Add some realistic anomalies
    anomaly_indices = [50, 150, 250, 350, 450]
    ecommerce_data.loc[50, 'amount'] = 50000    # Unusually high transaction
    ecommerce_data.loc[150, 'discount_percent'] = 95  # Unusually high discount
    ecommerce_data.loc[250, 'customer_rating'] = 1    # Low rating with high amount
    ecommerce_data.loc[250, 'amount'] = 5000
    ecommerce_data.loc[350, 'payment_method'] = 'crypto'  # Crypto payment with high amount
    ecommerce_data.loc[350, 'amount'] = 15000
    ecommerce_data.loc[450, 'customer_tier'] = 'bronze'   # Bronze customer with very high purchase
    ecommerce_data.loc[450, 'amount'] = 25000
    
    # Process the e-commerce data
    features, types = process_data_source("ecommerce_transactions", ecommerce_data)
    
    # Initialize and train the anomaly detector
    detector = XGBoostSHAPAnomalyDetector(contamination=0.02, verbose=True)  # Expect 2% anomalies
    detector.fit(features, "ecommerce_transactions")
    
    # Generate comprehensive report
    report = detector.generate_anomaly_report(features, "ecommerce_transactions")
    
    # Print detailed analysis for detected anomalies
    report_string = ""
    report_string += f"\nDETAILED ANOMALY ANALYSIS:\n"
    report_string += f"-" * 50 + "\n"
    
    for idx in report['anomaly_indices'][:5]:  # Show top 5 anomalies
        anomaly_info = report['detailed_anomalies'][idx]
        original_data = ecommerce_data.iloc[idx]    
        
        report_string += f"\n🚨 ANOMALY #{idx} (Probability: {anomaly_info['probability']:.3f})\n"
        report_string += f"   Transaction: {original_data['transaction_id']}\n"
        report_string += f"   Amount: ${original_data['amount']:.2f}\n"
        report_string += f"   Customer Tier: {original_data['customer_tier']}\n"
        report_string += f"   Payment Method: {original_data['payment_method']}\n"
        
        report_string += f"   🔍 TOP ROOT CAUSES:\n"
        for i, cause in enumerate(anomaly_info['top_3_causes']):
            impact = "🔺" if cause['contribution'] == 'increases' else "🔻"
            report_string += f"      {i+1}. {cause['feature']}: {cause['feature_value']:.3f} {impact}\n"
            report_string += f"         SHAP impact: {cause['shap_value']:.4f} ({cause['contribution']} anomaly score)\n"
    
    report_string += f"\n📊 BUSINESS INSIGHTS:\n"
    report_string += f"   • {report['anomaly_count']} suspicious transactions detected ({report['anomaly_rate']:.1%} of total)\n"
    report_string += f"   • Average anomaly probability: {report['summary_statistics']['avg_anomaly_probability']:.3f}\n"
    report_string += f"   • Key risk factors: {', '.join(report['top_root_cause_features'][:3])}\n"
    
    print(report_string)
    
    return test_results, report


if __name__ == "__main__":
    main()