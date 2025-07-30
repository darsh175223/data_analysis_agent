import pandas as pd
import numpy as np
from dateutil.parser import parse
import re
from textblob import TextBlob
from datetime import datetime, timedelta
import warnings
import uuid
warnings.filterwarnings('ignore')

# --- Enhanced Helper Functions for Type Inference ---

def is_numeric_strict(val):
    """Strictly checks if a value is numeric (int/float)."""
    if pd.isna(val):
        return False
    try:
        val_str = str(val).strip()
        if not val_str:
            return False
        # Handle percentage strings
        if val_str.endswith('%'):
            val_str = val_str[:-1]
        # Handle currency symbols
        val_str = re.sub(r'[$€£¥,]', '', val_str)
        float(val_str)
        return True
    except (ValueError, TypeError):
        return False

def is_integer(val):
    """Checks if a value is an integer."""
    if pd.isna(val):
        return False
    try:
        val_str = str(val).strip()
        if not val_str:
            return False
        float_val = float(val_str)
        return float_val.is_integer()
    except (ValueError, TypeError):
        return False

def is_boolean_value(val):
    """Checks if a value represents a boolean."""
    if pd.isna(val):
        return False
    val_str = str(val).strip().lower()
    return val_str in {'true', 'false', '1', '0', 'yes', 'no', 't', 'f', 'y', 'n', 
                       'active', 'inactive', 'enabled', 'disabled', 'on', 'off',
                       'open', 'closed', 'completed', 'pending'}

def is_id_like(val):
    """Checks if a value looks like an ID (alphanumeric identifier)."""
    if pd.isna(val):
        return False
    val_str = str(val).strip()
    # Common ID patterns
    id_patterns = [
        r'^[A-Z]{2,4}-?\d{4,}$',  # ORDER-1234, INV-001
        r'^\d{4,}$',              # Simple numeric IDs
        r'^[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}$',  # UUID
        r'^[A-Z0-9]{6,}$',        # Mixed alphanumeric
        r'^(cust|user|order|inv|prod|ticket)_?\d+$',  # Prefixed IDs
    ]
    return any(re.match(pattern, val_str, re.IGNORECASE) for pattern in id_patterns)

def is_email(val):
    """Checks if a value is an email address."""
    if pd.isna(val):
        return False
    val_str = str(val).strip()
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, val_str))

def is_url(val):
    """Checks if a value is a URL."""
    if pd.isna(val):
        return False
    val_str = str(val).strip().lower()
    return val_str.startswith(('http://', 'https://', 'www.', 'ftp://'))

def is_phone_number(val):
    """Checks if a value is a phone number."""
    if pd.isna(val):
        return False
    val_str = str(val).strip()
    # Remove common phone number formatting
    cleaned = re.sub(r'[\s\-\(\)\+\.]', '', val_str)
    return cleaned.isdigit() and 7 <= len(cleaned) <= 15

def is_currency(val):
    """Checks if a value represents currency."""
    if pd.isna(val):
        return False
    val_str = str(val).strip()
    currency_pattern = r'^[\$€£¥]?[\d,]+\.?\d*$'
    return bool(re.match(currency_pattern, val_str))

def is_percentage(val):
    """Checks if a value is a percentage."""
    if pd.isna(val):
        return False
    val_str = str(val).strip()
    return val_str.endswith('%') and is_numeric_strict(val_str[:-1])

def is_date_strict(val):
    """Strictly attempts to parse a value as a date, avoiding numeric false positives."""
    if pd.isna(val):
        return False
    
    val_str = str(val).strip()
    if not val_str:
        return False
    
    # Skip if it's purely numeric (common false positive)
    if re.match(r'^-?\d+\.?\d*$', val_str):
        return False
    
    # Skip very short strings that are likely not dates
    if len(val_str) < 4:
        return False
    
    # Look for common date patterns
    date_patterns = [
        r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD
        r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY or DD/MM/YYYY
        r'\d{1,2}-\d{1,2}-\d{4}',  # MM-DD-YYYY or DD-MM-YYYY
        r'\d{4}-\d{1,2}-\d{1,2}T\d{1,2}:\d{1,2}:\d{1,2}',  # ISO format
        r'\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)',  # Month names
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}',  # Month names
    ]
    
    # If it matches common date patterns, try parsing
    if any(re.search(pattern, val_str, re.IGNORECASE) for pattern in date_patterns):
        try:
            parse(val_str)
            return True
        except (ValueError, TypeError):
            return False
    
    return False

def is_status_code(val):
    """Checks if a value is a status/state code."""
    if pd.isna(val):
        return False
    val_str = str(val).strip().lower()
    status_keywords = {
        'pending', 'completed', 'failed', 'processing', 'cancelled', 'approved', 'rejected',
        'active', 'inactive', 'suspended', 'expired', 'draft', 'published',
        'open', 'closed', 'resolved', 'escalated', 'new', 'in_progress',
        'success', 'error', 'warning', 'info', 'critical',
        'low', 'medium', 'high', 'urgent',
        'bronze', 'silver', 'gold', 'platinum', 'premium', 'basic', 'pro', 'enterprise'
    }
    return val_str in status_keywords

def get_cardinality_ratio(series):
    """Calculate the ratio of unique values to total non-null values."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0
    return non_null.nunique() / len(non_null)

def infer_data_type_enhanced(series, sample_size=1000):
    """Enhanced data type inference with comprehensive business data heuristics."""
    # Handle empty series
    non_null_series = series.dropna()
    if non_null_series.empty:
        return 'unknown'
    
    # Sample for efficiency
    sample_size = min(len(non_null_series), sample_size)
    if len(non_null_series) > sample_size:
        sample = non_null_series.sample(n=sample_size, random_state=42)
    else:
        sample = non_null_series
    
    # Calculate test ratios
    numeric_ratio = sample.apply(is_numeric_strict).mean()
    integer_ratio = sample.apply(is_integer).mean()
    boolean_ratio = sample.apply(is_boolean_value).mean()
    date_ratio = sample.apply(is_date_strict).mean()
    id_ratio = sample.apply(is_id_like).mean()
    email_ratio = sample.apply(is_email).mean()
    url_ratio = sample.apply(is_url).mean()
    phone_ratio = sample.apply(is_phone_number).mean()
    currency_ratio = sample.apply(is_currency).mean()
    percentage_ratio = sample.apply(is_percentage).mean()
    status_ratio = sample.apply(is_status_code).mean()
    
    # Get cardinality info
    cardinality_ratio = get_cardinality_ratio(series)
    unique_count = series.nunique()
    
    # Column name hints
    col_name = series.name.lower() if series.name else ''
    
    print(f"    Type inference for '{series.name}':")
    print(f"      numeric={numeric_ratio:.2f}, boolean={boolean_ratio:.2f}, date={date_ratio:.2f}")
    print(f"      id={id_ratio:.2f}, email={email_ratio:.2f}, url={url_ratio:.2f}")
    print(f"      phone={phone_ratio:.2f}, currency={currency_ratio:.2f}, percentage={percentage_ratio:.2f}")
    print(f"      status={status_ratio:.2f}, cardinality_ratio={cardinality_ratio:.2f}, unique_count={unique_count}")
    
    # Decision logic with clear thresholds
    
    # 1. Specialized data types (highest priority)
    if email_ratio > 0.8:
        return 'email'
    
    if url_ratio > 0.8:
        return 'url'
    
    if phone_ratio > 0.8:
        return 'phone'
    
    if id_ratio > 0.7 or ('id' in col_name and cardinality_ratio > 0.7):
        return 'identifier'
    
    # 2. Boolean check
    if boolean_ratio > 0.8 and unique_count <= 10:
        return 'boolean'
    
    # 3. Status/categorical check (before general categorical)
    if status_ratio > 0.8:
        return 'status'
    
    # 4. Numeric types (before date to avoid false positives)
    if currency_ratio > 0.8:
        return 'currency'
    
    if percentage_ratio > 0.8:
        return 'percentage'
    
    if numeric_ratio > 0.9:
        # Sub-classify numeric types
        if integer_ratio > 0.9:
            if unique_count <= 50 and cardinality_ratio < 0.5:
                return 'categorical_numeric'  # Rating scales, counts, etc.
            elif 'score' in col_name or 'rating' in col_name:
                return 'score'
            return 'integer'
        else:
            if 'rate' in col_name or 'ratio' in col_name or 'percentage' in col_name:
                return 'rate'
            return 'numeric'
    
    # 5. Date check (after numeric to avoid parsing numbers as dates)
    if date_ratio > 0.8:
        return 'datetime'
    
    # 6. Categorical check (low cardinality text)
    if cardinality_ratio < 0.5 and unique_count < 100:
        return 'categorical'
    
    # 7. High cardinality text
    if cardinality_ratio > 0.8 or unique_count > 100:
        return 'text'
    
    # 8. Default to categorical for medium cardinality
    return 'categorical'

# --- Enhanced Feature Engineering Functions ---

def engineer_numeric_features(df, col_name):
    """Engineers features for numeric columns with robust handling."""
    features = {}
    col_data = pd.to_numeric(df[col_name], errors='coerce')
    
    # Basic features
    features[f'{col_name}'] = col_data
    features[f'{col_name}_lag1'] = col_data.shift(1)
    features[f'{col_name}_lag2'] = col_data.shift(2)
    features[f'{col_name}_diff'] = col_data.diff()
    features[f'{col_name}_pct_change'] = col_data.pct_change()
    
    # Rolling statistics
    window = min(3, len(df))
    if window > 1:
        features[f'{col_name}_rolling_mean'] = col_data.rolling(window=window, min_periods=1).mean()
        features[f'{col_name}_rolling_std'] = col_data.rolling(window=window, min_periods=1).std()
        features[f'{col_name}_rolling_min'] = col_data.rolling(window=window, min_periods=1).min()
        features[f'{col_name}_rolling_max'] = col_data.rolling(window=window, min_periods=1).max()
    
    # Statistical features
    mean_val = col_data.mean()
    std_val = col_data.std()
    if std_val > 0:
        features[f'{col_name}_zscore'] = (col_data - mean_val) / std_val
        features[f'{col_name}_is_outlier'] = (np.abs(features[f'{col_name}_zscore']) > 2).astype(int)
    
    # Binning
    try:
        features[f'{col_name}_quartile'] = pd.qcut(col_data, q=4, labels=False, duplicates='drop')
    except:
        features[f'{col_name}_quartile'] = 0
    
    # Business-specific features
    features[f'{col_name}_is_zero'] = (col_data == 0).astype(int)
    features[f'{col_name}_is_negative'] = (col_data < 0).astype(int)
    
    return pd.DataFrame(features)

def engineer_currency_features(df, col_name):
    """Engineers features specifically for currency/monetary values."""
    features = {}
    
    # Clean currency and convert to numeric
    currency_series = df[col_name].astype(str).str.replace(r'[$€£¥,]', '', regex=True)
    currency_numeric = pd.to_numeric(currency_series, errors='coerce')
    
    # Include all numeric features
    numeric_features = engineer_numeric_features(pd.DataFrame({col_name: currency_numeric}), col_name)
    features.update(numeric_features.to_dict('series'))
    
    # Currency-specific features
    features[f'{col_name}_log'] = np.log1p(currency_numeric.fillna(0))
    features[f'{col_name}_is_round_number'] = (currency_numeric % 100 == 0).astype(int)
    features[f'{col_name}_magnitude'] = np.floor(np.log10(currency_numeric.replace(0, 1).fillna(1)))
    
    # Revenue bands
    if currency_numeric.max() > 0:
        percentiles = currency_numeric.quantile([0.25, 0.5, 0.75, 0.9]).values
        unique_percentiles = np.unique(percentiles)
        bins = [-np.inf] + list(unique_percentiles) + [np.inf]
        num_intervals = len(bins) - 1
        labels = list(range(num_intervals))
        features[f'{col_name}_revenue_band'] = pd.cut(currency_numeric, 
                                                     bins=bins, 
                                                     labels=labels, duplicates='drop').astype(float)
    
    return pd.DataFrame(features)

def engineer_percentage_features(df, col_name):
    """Engineers features for percentage data."""
    features = {}
    
    # Clean percentage and convert to numeric (0-100 scale)
    pct_series = df[col_name].astype(str).str.replace('%', '')
    pct_numeric = pd.to_numeric(pct_series, errors='coerce')
    
    features[f'{col_name}_raw'] = pct_numeric
    features[f'{col_name}_normalized'] = pct_numeric / 100  # Convert to 0-1 scale
    
    # Percentage-specific features
    features[f'{col_name}_is_perfect'] = (pct_numeric == 100).astype(int)
    features[f'{col_name}_is_zero'] = (pct_numeric == 0).astype(int)
    features[f'{col_name}_is_high'] = (pct_numeric > 80).astype(int)
    features[f'{col_name}_is_low'] = (pct_numeric < 20).astype(int)
    
    # Binning
    features[f'{col_name}_quartile'] = pd.cut(pct_numeric, bins=[0, 25, 50, 75, 100], labels=[0, 1, 2, 3], duplicates='drop').astype(float)
    
    return pd.DataFrame(features)

def engineer_score_features(df, col_name):
    """Engineers features for score/rating data."""
    features = {}
    col_data = pd.to_numeric(df[col_name], errors='coerce')
    
    # Basic score features
    features[f'{col_name}'] = col_data
    features[f'{col_name}_normalized'] = (col_data - col_data.min()) / (col_data.max() - col_data.min())
    
    # Score categories
    if col_data.max() <= 5:  # 1-5 scale
        features[f'{col_name}_is_high'] = (col_data >= 4).astype(int)
        features[f'{col_name}_is_low'] = (col_data <= 2).astype(int)
    elif col_data.max() <= 10:  # 1-10 scale
        features[f'{col_name}_is_high'] = (col_data >= 8).astype(int)
        features[f'{col_name}_is_low'] = (col_data <= 4).astype(int)
    else:  # Custom scale
        features[f'{col_name}_is_high'] = (col_data >= col_data.quantile(0.8)).astype(int)
        features[f'{col_name}_is_low'] = (col_data <= col_data.quantile(0.2)).astype(int)
    
    # Trends
    features[f'{col_name}_diff'] = col_data.diff()
    features[f'{col_name}_improving'] = (features[f'{col_name}_diff'] > 0).astype(int)
    
    return pd.DataFrame(features)

def engineer_identifier_features(df, col_name):
    """Engineers features for ID columns."""
    features = {}
    
    # Keep original for joins
    features[f'{col_name}'] = df[col_name]
    
    # Frequency encoding (how often each ID appears)
    freq_map = df[col_name].value_counts().to_dict()
    features[f'{col_name}_frequency'] = df[col_name].map(freq_map)
    
    # ID characteristics
    id_strings = df[col_name].astype(str)
    features[f'{col_name}_length'] = id_strings.str.len()
    features[f'{col_name}_has_letters'] = id_strings.str.contains(r'[A-Za-z]', na=False).astype(int)
    features[f'{col_name}_has_numbers'] = id_strings.str.contains(r'\d', na=False).astype(int)
    features[f'{col_name}_has_special_chars'] = id_strings.str.contains(r'[^A-Za-z0-9]', na=False).astype(int)
    
    # Binary indicators for top IDs
    top_ids = df[col_name].value_counts().head(5).index
    for id_val in top_ids:
        features[f'{col_name}_is_{id_val}'] = (df[col_name] == id_val).astype(int)
    
    return pd.DataFrame(features)

def engineer_status_features(df, col_name):
    """Engineers features for status/state columns."""
    features = {}
    
    # One-hot encoding for all statuses
    status_dummies = pd.get_dummies(df[col_name], prefix=col_name)
    features.update(status_dummies.to_dict('series'))
    
    # Status groupings
    positive_statuses = ['completed', 'success', 'approved', 'active', 'resolved', 'published']
    negative_statuses = ['failed', 'rejected', 'cancelled', 'suspended', 'error', 'critical']
    neutral_statuses = ['pending', 'processing', 'draft', 'open', 'new']
    
    status_lower = df[col_name].astype(str).str.lower()
    features[f'{col_name}_is_positive'] = status_lower.isin(positive_statuses).astype(int)
    features[f'{col_name}_is_negative'] = status_lower.isin(negative_statuses).astype(int)
    features[f'{col_name}_is_neutral'] = status_lower.isin(neutral_statuses).astype(int)
    
    # Frequency encoding
    freq_map = df[col_name].value_counts().to_dict()
    features[f'{col_name}_frequency'] = df[col_name].map(freq_map)
    
    return pd.DataFrame(features)

def engineer_email_features(df, col_name):
    """Engineers features for email columns."""
    features = {}
    email_series = df[col_name].fillna('').astype(str)
    
    # Domain extraction
    features[f'{col_name}_domain'] = email_series.str.extract(r'@(.+)')[0]
    
    # Email characteristics
    features[f'{col_name}_length'] = email_series.str.len()
    features[f'{col_name}_local_length'] = email_series.str.extract(r'(.+)@')[0].str.len()
    features[f'{col_name}_has_numbers'] = email_series.str.contains(r'\d', na=False).astype(int)
    features[f'{col_name}_has_dots'] = email_series.str.count(r'\.')
    features[f'{col_name}_has_plus'] = email_series.str.contains(r'\+', na=False).astype(int)
    
    # Common domain indicators
    common_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com']
    for domain in common_domains:
        features[f'{col_name}_is_{domain.replace(".", "_")}'] = email_series.str.endswith(domain).astype(int)
    
    # Business vs personal email heuristic
    business_indicators = ['company', 'corp', 'inc', 'ltd', 'org']
    personal_indicators = ['gmail', 'yahoo', 'hotmail', 'outlook']
    
    domain_series = features[f'{col_name}_domain'].fillna('')
    features[f'{col_name}_is_business'] = domain_series.str.contains('|'.join(business_indicators), na=False).astype(int)
    features[f'{col_name}_is_personal'] = domain_series.str.contains('|'.join(personal_indicators), na=False).astype(int)
    
    return pd.DataFrame(features)

def engineer_url_features(df, col_name):
    """Engineers features for URL columns."""
    features = {}
    url_series = df[col_name].fillna('').astype(str)
    
    # URL components
    features[f'{col_name}_length'] = url_series.str.len()
    features[f'{col_name}_is_https'] = url_series.str.startswith('https').astype(int)
    features[f'{col_name}_has_params'] = url_series.str.contains(r'\?', na=False).astype(int)
    features[f'{col_name}_param_count'] = url_series.str.count(r'[?&]')
    features[f'{col_name}_path_depth'] = url_series.str.count(r'/')
    
    # Domain extraction
    domain_pattern = r'https?://(?:www\.)?([^/]+)'
    features[f'{col_name}_domain'] = url_series.str.extract(domain_pattern)[0]
    
    # Common platforms
    platforms = ['facebook', 'twitter', 'linkedin', 'instagram', 'youtube', 'google']
    for platform in platforms:
        features[f'{col_name}_is_{platform}'] = url_series.str.contains(platform, na=False).astype(int)
    
    return pd.DataFrame(features)

def engineer_phone_features(df, col_name):
    """Engineers features for phone number columns."""
    features = {}
    phone_series = df[col_name].fillna('').astype(str)
    
    # Clean phone numbers
    cleaned_phones = phone_series.str.replace(r'[\s\-\(\)\+\.]', '', regex=True)
    features[f'{col_name}_cleaned'] = cleaned_phones
    features[f'{col_name}_length'] = cleaned_phones.str.len()
    
    # Phone characteristics
    features[f'{col_name}_has_country_code'] = phone_series.str.startswith('+').astype(int)
    features[f'{col_name}_has_parentheses'] = phone_series.str.contains(r'[\(\)]', na=False).astype(int)
    features[f'{col_name}_has_dashes'] = phone_series.str.contains(r'-', na=False).astype(int)
    
    # Length categories
    features[f'{col_name}_is_mobile'] = (cleaned_phones.str.len() == 10).astype(int)  # US mobile
    features[f'{col_name}_is_international'] = (cleaned_phones.str.len() > 10).astype(int)
    
    return pd.DataFrame(features)

def engineer_datetime_features(df, col_name):
    """Engineers comprehensive datetime features."""
    features = {}
    dt_series = pd.to_datetime(df[col_name], errors='coerce')
    
    # Basic temporal features
    features[f'{col_name}_year'] = dt_series.dt.year
    features[f'{col_name}_month'] = dt_series.dt.month
    features[f'{col_name}_day'] = dt_series.dt.day
    features[f'{col_name}_dayofweek'] = dt_series.dt.dayofweek
    features[f'{col_name}_hour'] = dt_series.dt.hour
    features[f'{col_name}_minute'] = dt_series.dt.minute
    features[f'{col_name}_quarter'] = dt_series.dt.quarter
    
    # Derived features
    features[f'{col_name}_is_weekend'] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)
    features[f'{col_name}_is_month_start'] = dt_series.dt.is_month_start.astype(int)
    features[f'{col_name}_is_month_end'] = dt_series.dt.is_month_end.astype(int)
    features[f'{col_name}_is_quarter_start'] = dt_series.dt.is_quarter_start.astype(int)
    features[f'{col_name}_is_quarter_end'] = dt_series.dt.is_quarter_end.astype(int)
    
    # Business time features
    features[f'{col_name}_is_business_hours'] = ((dt_series.dt.hour >= 9) & (dt_series.dt.hour <= 17)).astype(int)
    features[f'{col_name}_is_workday'] = (~dt_series.dt.dayofweek.isin([5, 6])).astype(int)
    
    # Cyclical encoding
    features[f'{col_name}_hour_sin'] = np.sin(2 * np.pi * dt_series.dt.hour / 24)
    features[f'{col_name}_hour_cos'] = np.cos(2 * np.pi * dt_series.dt.hour / 24)
    features[f'{col_name}_dayofweek_sin'] = np.sin(2 * np.pi * dt_series.dt.dayofweek / 7)
    features[f'{col_name}_dayofweek_cos'] = np.cos(2 * np.pi * dt_series.dt.dayofweek / 7)
    features[f'{col_name}_month_sin'] = np.sin(2 * np.pi * dt_series.dt.month / 12)
    features[f'{col_name}_month_cos'] = np.cos(2 * np.pi * dt_series.dt.month / 12)
    
    # Time since features
    min_date = dt_series.min()
    max_date = dt_series.max()
    features[f'{col_name}_days_since_start'] = (dt_series - min_date).dt.days
    features[f'{col_name}_days_until_end'] = (max_date - dt_series).dt.days
    
    # Recency features
    current_date = dt_series.max()  # Use latest date in dataset as "current"
    features[f'{col_name}_days_ago'] = (current_date - dt_series).dt.days
    features[f'{col_name}_is_recent'] = (features[f'{col_name}_days_ago'] <= 30).astype(int)
    features[f'{col_name}_is_very_recent'] = (features[f'{col_name}_days_ago'] <= 7).astype(int)
    
    return pd.DataFrame(features)

def engineer_categorical_features(df, col_name):
    """Engineers features for categorical columns with frequency encoding."""
    features = {}
    
    # One-hot encoding (limit to top categories to avoid explosion)
    value_counts = df[col_name].value_counts()
    top_categories = value_counts.head(10).index
    
    for category in top_categories:
        safe_category = str(category).replace(' ', '_').replace('-', '_').replace('.', '_')
        features[f'{col_name}_{safe_category}'] = (df[col_name] == category).astype(int)
    
    # Frequency encoding
    freq_map = df[col_name].value_counts().to_dict()
    features[f'{col_name}_frequency'] = df[col_name].map(freq_map)
    
    # Rare category indicator
    rare_threshold = max(1, len(df) * 0.05)  # 5% threshold
    features[f'{col_name}_is_rare'] = (features[f'{col_name}_frequency'] < rare_threshold).astype(int)
    
    # Category rank by frequency
    freq_rank = df[col_name].value_counts().rank(ascending=False).to_dict()
    features[f'{col_name}_frequency_rank'] = df[col_name].map(freq_rank)
    
    return pd.DataFrame(features)

def engineer_categorical_numeric_features(df, col_name):
    """Engineers features for categorical numeric data (IDs, codes)."""
    features = {}
    
    # Keep original
    features[f'{col_name}'] = df[col_name]
    
    # Frequency encoding
    freq_map = df[col_name].value_counts().to_dict()
    features[f'{col_name}_frequency'] = df[col_name].map(freq_map)
    
    # Binary indicators for most common values
    top_values = df[col_name].value_counts().head(5).index
    for value in top_values:
        features[f'{col_name}_is_{value}'] = (df[col_name] == value).astype(int)
    
    # Numeric properties
    features[f'{col_name}_mod_10'] = df[col_name] % 10  # Last digit
    features[f'{col_name}_is_even'] = (df[col_name] % 2 == 0).astype(int)
    
    return pd.DataFrame(features)

def engineer_boolean_features(df, col_name):
    """Engineers features for boolean columns."""
    features = {}
    
    # Standardize boolean values
    bool_series = df[col_name].astype(str).str.lower()
    bool_map = {'true': 1, '1': 1, 'yes': 1, 'y': 1, 't': 1, 'active': 1, 'enabled': 1, 'on': 1,
                'false': 0, '0': 0, 'no': 0, 'n': 0, 'f': 0, 'inactive': 0, 'disabled': 0, 'off': 0}
    
    features[f'{col_name}_encoded'] = bool_series.map(bool_map).fillna(-1)
    
    # Interaction with index (trend over time)
    features[f'{col_name}_trend'] = features[f'{col_name}_encoded'] * np.arange(len(df))
    
    # Rolling aggregations
    window = min(3, len(df))
    if window > 1:
        features[f'{col_name}_rolling_sum'] = features[f'{col_name}_encoded'].rolling(window=window, min_periods=1).sum()
        features[f'{col_name}_rolling_mean'] = features[f'{col_name}_encoded'].rolling(window=window, min_periods=1).mean()
    
    return pd.DataFrame(features)

def engineer_text_features(df, col_name):
    """Engineers comprehensive text features."""
    features = {}
    text_series = df[col_name].fillna('').astype(str)
    
    # Basic length features
    features[f'{col_name}_length'] = text_series.str.len()
    features[f'{col_name}_word_count'] = text_series.str.split().str.len()
    features[f'{col_name}_char_count'] = text_series.str.replace(' ', '').str.len()
    features[f'{col_name}_sentence_count'] = text_series.str.count(r'[.!?]+') + 1
    
    # Character analysis
    features[f'{col_name}_num_digits'] = text_series.str.count(r'\d')
    features[f'{col_name}_num_punctuation'] = text_series.str.count(r'[^\w\s]')
    features[f'{col_name}_num_uppercase'] = text_series.str.count(r'[A-Z]')
    features[f'{col_name}_num_lowercase'] = text_series.str.count(r'[a-z]')
    features[f'{col_name}_num_spaces'] = text_series.str.count(r' ')
    
    # Content type detection
    features[f'{col_name}_has_url'] = text_series.str.contains(r'http[s]?://', case=False, na=False).astype(int)
    features[f'{col_name}_has_email'] = text_series.str.contains(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', na=False).astype(int)
    features[f'{col_name}_has_hashtag'] = text_series.str.contains(r'#\w+', na=False).astype(int)
    features[f'{col_name}_has_mention'] = text_series.str.contains(r'@\w+', na=False).astype(int)
    features[f'{col_name}_has_phone'] = text_series.str.contains(r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s?\d{3}-\d{4}\b', na=False).astype(int)
    
    # Sentiment analysis
    def get_sentiment_polarity(text):
        if not text or pd.isna(text):
            return 0.0
        try:
            return TextBlob(str(text)).sentiment.polarity
        except:
            return 0.0
    
    def get_sentiment_subjectivity(text):
        if not text or pd.isna(text):
            return 0.0
        try:
            return TextBlob(str(text)).sentiment.subjectivity
        except:
            return 0.0
    
    features[f'{col_name}_sentiment_polarity'] = text_series.apply(get_sentiment_polarity)
    features[f'{col_name}_sentiment_subjectivity'] = text_series.apply(get_sentiment_subjectivity)
    features[f'{col_name}_is_positive'] = (features[f'{col_name}_sentiment_polarity'] > 0.1).astype(int)
    features[f'{col_name}_is_negative'] = (features[f'{col_name}_sentiment_polarity'] < -0.1).astype(int)
    
    # Business keyword detection
    business_keywords = {
        'urgency': ['urgent', 'asap', 'immediately', 'emergency', 'critical'],
        'satisfaction': ['excellent', 'great', 'good', 'poor', 'terrible', 'amazing'],
        'technical': ['error', 'bug', 'issue', 'problem', 'fix', 'update', 'upgrade'],
        'support': ['help', 'support', 'assistance', 'question', 'inquiry'],
        'financial': ['payment', 'invoice', 'billing', 'refund', 'charge', 'cost', 'price'],
        'product': ['feature', 'product', 'service', 'functionality', 'performance'],
    }
    
    for category, keywords in business_keywords.items():
        features[f'{col_name}_has_{category}'] = text_series.str.contains('|'.join(keywords), case=False, na=False).astype(int)
    
    # Text complexity
    features[f'{col_name}_avg_word_length'] = text_series.str.split().apply(lambda x: np.mean([len(word) for word in x]) if x else 0)
    features[f'{col_name}_caps_ratio'] = features[f'{col_name}_num_uppercase'] / (features[f'{col_name}_length'] + 1)
    features[f'{col_name}_punct_ratio'] = features[f'{col_name}_num_punctuation'] / (features[f'{col_name}_length'] + 1)
    
    return pd.DataFrame(features)

# --- Main Processing Function ---
def process_data_source(data_source_name, raw_data_df):
    """Enhanced data processing with better error handling and logging."""
    print(f"\n{'='*80}")
    print(f"Processing Data Source: {data_source_name}")
    print(f"{'='*80}")
    print(f"Raw Data Shape: {raw_data_df.shape}")
    print("\nRaw Data Sample:")
    print(raw_data_df.head(3))
    print(f"\nData Types (pandas inferred):")
    print(raw_data_df.dtypes)
    
    engineered_features = []
    type_inferences = {}
    successful_columns = []
    failed_columns = []

    # Infer types for all columns
    print(f"\n{'-'*50}")
    print("TYPE INFERENCE RESULTS")
    print(f"{'-'*50}")
    
    for col in raw_data_df.columns:
        try:
            inferred_type = infer_data_type_enhanced(raw_data_df[col])
            type_inferences[col] = inferred_type
            print(f"  ✓ Column '{col}': {inferred_type}")
        except Exception as e:
            print(f"  ✗ Column '{col}': Error in type inference - {e}")
            type_inferences[col] = 'unknown'

    # Engineer features based on inferred types
    print(f"\n{'-'*50}")
    print("FEATURE ENGINEERING RESULTS")
    print(f"{'-'*50}")
    
    for col, data_type in type_inferences.items():
        try:
            if data_type == 'numeric':
                feat_df = engineer_numeric_features(raw_data_df, col)
                engineered_features.append(feat_df)
                successful_columns.append(f"{col} ({data_type})")
                print(f"  ✓ {col} ({data_type}): {feat_df.shape[1]} features created")
                
            elif data_type == 'integer':
                feat_df = engineer_numeric_features(raw_data_df, col)
                engineered_features.append(feat_df)
                successful_columns.append(f"{col} ({data_type})")
                print(f"  ✓ {col} ({data_type}): {feat_df.shape[1]} features created")
                
            elif data_type == 'currency':
                feat_df = engineer_currency_features(raw_data_df, col)
                engineered_features.append(feat_df)
                successful_columns.append(f"{col} ({data_type})")
                print(f"  ✓ {col} ({data_type}): {feat_df.shape[1]} features created")
                
            elif data_type == 'percentage':
                feat_df = engineer_percentage_features(raw_data_df, col)
                engineered_features.append(feat_df)
                successful_columns.append(f"{col} ({data_type})")
                print(f"  ✓ {col} ({data_type}): {feat_df.shape[1]} features created")
                
            elif data_type == 'score':
                feat_df = engineer_score_features(raw_data_df, col)
                engineered_features.append(feat_df)
                successful_columns.append(f"{col} ({data_type})")
                print(f"  ✓ {col} ({data_type}): {feat_df.shape[1]} features created")
                
            elif data_type == 'rate':
                feat_df = engineer_numeric_features(raw_data_df, col)  # Treat as numeric
                engineered_features.append(feat_df)
                successful_columns.append(f"{col} ({data_type})")
                print(f"  ✓ {col} ({data_type}): {feat_df.shape[1]} features created")
                
            elif data_type == 'datetime':
                feat_df = engineer_datetime_features(raw_data_df, col)
                engineered_features.append(feat_df)
                successful_columns.append(f"{col} ({data_type})")
                print(f"  ✓ {col} ({data_type}): {feat_df.shape[1]} features created")
                
            elif data_type == 'categorical':
                feat_df = engineer_categorical_features(raw_data_df, col)
                engineered_features.append(feat_df)
                successful_columns.append(f"{col} ({data_type})")
                print(f"  ✓ {col} ({data_type}): {feat_df.shape[1]} features created")
                
            elif data_type == 'categorical_numeric':
                feat_df = engineer_categorical_numeric_features(raw_data_df, col)
                engineered_features.append(feat_df)
                successful_columns.append(f"{col} ({data_type})")
                print(f"  ✓ {col} ({data_type}): {feat_df.shape[1]} features created")
                
            elif data_type == 'identifier':
                feat_df = engineer_identifier_features(raw_data_df, col)
                engineered_features.append(feat_df)
                successful_columns.append(f"{col} ({data_type})")
                print(f"  ✓ {col} ({data_type}): {feat_df.shape[1]} features created")
                
            elif data_type == 'status':
                feat_df = engineer_status_features(raw_data_df, col)
                engineered_features.append(feat_df)
                successful_columns.append(f"{col} ({data_type})")
                print(f"  ✓ {col} ({data_type}): {feat_df.shape[1]} features created")
                
            elif data_type == 'boolean':
                feat_df = engineer_boolean_features(raw_data_df, col)
                engineered_features.append(feat_df)
                successful_columns.append(f"{col} ({data_type})")
                print(f"  ✓ {col} ({data_type}): {feat_df.shape[1]} features created")
                
            elif data_type == 'email':
                feat_df = engineer_email_features(raw_data_df, col)
                engineered_features.append(feat_df)
                successful_columns.append(f"{col} ({data_type})")
                print(f"  ✓ {col} ({data_type}): {feat_df.shape[1]} features created")
                
            elif data_type == 'url':
                feat_df = engineer_url_features(raw_data_df, col)
                engineered_features.append(feat_df)
                successful_columns.append(f"{col} ({data_type})")
                print(f"  ✓ {col} ({data_type}): {feat_df.shape[1]} features created")
                
            elif data_type == 'phone':
                feat_df = engineer_phone_features(raw_data_df, col)
                engineered_features.append(feat_df)
                successful_columns.append(f"{col} ({data_type})")
                print(f"  ✓ {col} ({data_type}): {feat_df.shape[1]} features created")
                
            elif data_type == 'text':
                feat_df = engineer_text_features(raw_data_df, col)
                engineered_features.append(feat_df)
                successful_columns.append(f"{col} ({data_type})")
                print(f"  ✓ {col} ({data_type}): {feat_df.shape[1]} features created")
                
            else:
                print(f"  ⚠ {col} ({data_type}): Skipped - no handler for this type")
                failed_columns.append(f"{col} ({data_type})")
                
        except Exception as e:
            print(f"  ✗ {col} ({data_type}): Error - {e}")
            failed_columns.append(f"{col} ({data_type}) - Error: {str(e)[:50]}")

    # Combine all engineered features
    if engineered_features:
        final_features_df = pd.concat(engineered_features, axis=1)
        print(f"\n{'-'*50}")
        print("FINAL RESULTS")
        print(f"{'-'*50}")
        print(f"✓ Engineered Features Shape: {final_features_df.shape}")
        print(f"✓ Successful columns: {len(successful_columns)}")
        print(f"⚠ Failed/Skipped columns: {len(failed_columns)}")
        
        if failed_columns:
            print(f"\nFailed/Skipped details:")
            for col in failed_columns:
                print(f"  - {col}")
        
        print(f"\nSample of engineered features:")
        print(final_features_df.head(3))
        
        return final_features_df, type_inferences
    else:
        print(f"\n⚠ No features were successfully engineered.")
        return pd.DataFrame(), type_inferences

# --- Comprehensive Test Cases ---

def create_comprehensive_test_datasets():
    """Creates extensive test datasets covering all business data scenarios."""
    
    test_datasets = {}
    
    # Test 1: E-commerce Transaction Data
    test_datasets['ecommerce_transactions'] = pd.DataFrame({
        'transaction_id': ['TXN-001', 'TXN-002', 'TXN-003', 'TXN-004', 'TXN-005'],
        'customer_id': ['CUST-1001', 'CUST-1002', 'CUST-1001', 'CUST-1003', 'CUST-1002'],
        'order_date': ['2024-01-15T10:30:00Z', '2024-01-16T14:22:00Z', '2024-01-17T09:45:00Z', '2024-01-18T16:30:00Z', '2024-01-19T11:15:00Z'],
        'product_category': ['Electronics', 'Clothing', 'Electronics', 'Home', 'Clothing'],
        'amount': ['$125.99', '$89.50', '$299.99', '$45.00', '$156.75'],
        'payment_method': ['Credit Card', 'PayPal', 'Credit Card', 'Debit Card', 'PayPal'],
        'order_status': ['completed', 'pending', 'completed', 'failed', 'completed'],
        'discount_rate': ['10%', '0%', '15%', '5%', '8%'],
        'is_prime_member': [True, False, True, False, False]
    })
    
    # Test 2: Customer Support Data
    test_datasets['customer_support'] = pd.DataFrame({
        'ticket_id': ['TICK-5001', 'TICK-5002', 'TICK-5003', 'TICK-5004'],
        'customer_email': ['john.doe@gmail.com', 'sarah.wilson@company.com', 'mike.johnson@outlook.com', 'admin@techcorp.inc'],
        'created_date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18'],
        'priority': ['high', 'medium', 'low', 'critical'],
        'category': ['billing', 'technical', 'general', 'technical'],
        'resolution_time_hours': [2.5, 24.0, 1.0, 48.0],
        'satisfaction_score': [4, 3, 5, 2],
        'description': [
            'Billing issue with last invoice, please help ASAP!',
            'Application error when trying to login, getting timeout',
            'General question about new features',
            'CRITICAL: System down for all users, emergency fix needed!'
        ],
        'status': ['resolved', 'open', 'resolved', 'escalated']
    })
    
    # Test 3: Marketing Campaign Data
    test_datasets['marketing_campaigns'] = pd.DataFrame({
        'campaign_id': ['CAMP-001', 'CAMP-002', 'CAMP-003', 'CAMP-004', 'CAMP-005'],
        'campaign_name': ['Summer Sale', 'Back to School', 'Holiday Special', 'Spring Promo', 'Black Friday'],
        'start_date': ['2024-06-01', '2024-08-15', '2024-11-20', '2024-03-01', '2024-11-25'],
        'channel': ['Email', 'Social Media', 'Google Ads', 'Email', 'Multi-channel'],
        'budget': [5000.00, 8000.00, 15000.00, 3000.00, 25000.00],
        'impressions': [150000, 200000, 500000, 100000, 800000],
        'clicks': [1500, 4000, 7500, 800, 12000],
        'conversions': [75, 120, 300, 32, 480],
        'ctr_rate': ['1.0%', '2.0%', '1.5%', '0.8%', '1.5%'],
        'conversion_rate': ['5.0%', '3.0%', '4.0%', '4.0%', '4.0%'],
        'is_active': [False, False, True, False, True]
    })
    
    # Test 4: User Analytics & App Usage
    test_datasets['user_analytics'] = pd.DataFrame({
        'user_id': ['USER-101', 'USER-102', 'USER-103', 'USER-104', 'USER-105'],
        'session_date': ['2024-01-15T08:30:00Z', '2024-01-15T12:45:00Z', '2024-01-15T18:20:00Z', '2024-01-16T09:15:00Z', '2024-01-16T21:30:00Z'],
        'session_duration_minutes': [25, 45, 12, 67, 8],
        'page_views': [8, 15, 3, 23, 2],
        'bounce_rate': [0.25, 0.0, 1.0, 0.13, 1.0],
        'device_type': ['Mobile', 'Desktop', 'Mobile', 'Desktop', 'Tablet'],
        'traffic_source': ['Organic', 'Direct', 'Social', 'Paid', 'Referral'],
        'user_agent': ['Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X)', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 'Mozilla/5.0 (iPhone)', 'Chrome/91.0', 'Safari/537.36'],
        'conversion_goal_met': [True, True, False, True, False],
        'revenue_attributed': [45.99, 129.50, 0.0, 89.99, 0.0]
    })
    
    # Test 5: Financial & Subscription Data
    test_datasets['subscription_financials'] = pd.DataFrame({
        'subscription_id': ['SUB-A001', 'SUB-A002', 'SUB-A003', 'SUB-A004'],
        'customer_id': ['CUST-1001', 'CUST-1002', 'CUST-1003', 'CUST-1004'],
        'plan_tier': ['Basic', 'Pro', 'Enterprise', 'Pro'],
        'start_date': ['2023-06-01', '2023-07-15', '2023-08-01', '2023-09-01'],
        'end_date': ['2024-06-01', None, '2024-08-01', None],
        'monthly_revenue': [29.99, 99.99, 299.99, 99.99],
        'billing_cycle': ['monthly', 'annual', 'monthly', 'monthly'],
        'renewal_status': ['active', 'active', 'churned', 'active'],
        'payment_failures': [0, 1, 3, 0],
        'usage_score': [7.5, 9.2, 4.1, 8.8],
        'is_trial': [False, False, False, True]
    })
    
    # Test 6: Social Media & Content Performance
    test_datasets['social_content'] = pd.DataFrame({
        'post_id': ['POST-001', 'POST-002', 'POST-003', 'POST-004'],
        'platform': ['Twitter', 'LinkedIn', 'Facebook', 'Instagram'],
        'post_date': ['2024-01-15T10:00:00Z', '2024-01-16T14:30:00Z', '2024-01-17T16:45:00Z', '2024-01-18T09:20:00Z'],
        'content_type': ['Image', 'Article', 'Video', 'Story'],
        'post_text': [
            'Check out our new product launch! 🚀 #innovation https://company.com/new-product',
            'Industry insights: How AI is transforming business operations. Read our latest article.',
            'Behind the scenes video of our team working on exciting new features!',
            'Customer success story: Amazing results with our platform! #success @happycustomer'
        ],
        'likes': [150, 89, 234, 67],
        'shares': [12, 23, 45, 8],
        'comments': [18, 34, 67, 12],
        'reach': [5000, 3200, 8900, 2100],
        'engagement_rate': ['3.6%', '4.6%', '3.9%', '4.1%'],
        'sentiment_manual': ['positive', 'neutral', 'positive', 'positive']
    })
    
    # Test 7: System Logs & Performance
    test_datasets['system_performance'] = pd.DataFrame({
        'timestamp': ['2024-01-15T10:00:01.123Z', '2024-01-15T10:01:15.456Z', '2024-01-15T10:02:30.789Z', '2024-01-15T10:03:45.012Z'],
        'log_level': ['INFO', 'ERROR', 'WARN', 'DEBUG'],
        'service_name': ['auth-service', 'payment-service', 'user-service', 'notification-service'],
        'response_time_ms': [120, None, 890, 45],
        'cpu_usage_percent': ['25%', '78%', '45%', '12%'],
        'memory_usage_gb': [2.1, 4.5, 3.2, 1.8],
        'error_code': [None, 'TIMEOUT_ERROR', None, None],
        'request_id': ['req-12345', 'req-12346', 'req-12347', 'req-12348'],
        'message': [
            'User authentication successful for user_id: 1001',
            'Payment processing failed: connection timeout to payment gateway',
            'High memory usage detected: 78% of allocated memory',
            'Notification sent successfully to user: john@example.com'
        ],
        'is_production': [True, True, True, True]
    })
    
    # Test 8: HR & Employee Data
    test_datasets['hr_employee_data'] = pd.DataFrame({
        'employee_id': ['EMP-001', 'EMP-002', 'EMP-003', 'EMP-004', 'EMP-005'],
        'hire_date': ['2022-03-15', '2021-07-20', '2023-01-10', '2020-11-05', '2023-06-01'],
        'department': ['Engineering', 'Sales', 'Marketing', 'Engineering', 'HR'],
        'position_level': ['Senior', 'Manager', 'Junior', 'Lead', 'Coordinator'],
        'salary': [95000, 110000, 65000, 125000, 75000],
        'performance_rating': [4.2, 3.8, 4.5, 4.0, 3.9],
        'training_hours': [40, 20, 60, 30, 45],
        'satisfaction_score': [8, 6, 9, 7, 8],
        'work_location': ['Remote', 'Office', 'Hybrid', 'Remote', 'Office'],
        'is_active': [True, True, True, False, True]
    })
    
    # Test 9: Inventory & Operations
    test_datasets['inventory_operations'] = pd.DataFrame({
        'product_id': ['PROD-A001', 'PROD-B002', 'PROD-C003', 'PROD-D004'],
        'product_name': ['Wireless Headphones', 'Gaming Laptop', 'Fitness Tracker', 'Smart Watch'],
        'sku': ['WH-2024-001', 'GL-2024-002', 'FT-2024-003', 'SW-2024-004'],
        'category': ['Electronics', 'Computers', 'Fitness', 'Electronics'],
        'stock_level': [150, 25, 89, 200],
        'reorder_point': [50, 10, 30, 75],
        'unit_cost': [45.50, 899.99, 89.99, 199.99],
        'selling_price': [79.99, 1299.99, 129.99, 299.99],
        'supplier_lead_time_days': [7, 14, 5, 10],
        'last_restock_date': ['2024-01-10', '2024-01-05', '2024-01-12', '2024-01-08'],
        'is_seasonal': [False, True, False, False]
    })
    
    # Test 10: Customer Contact Information
    test_datasets['customer_contacts'] = pd.DataFrame({
        'customer_id': ['CUST-001', 'CUST-002', 'CUST-003', 'CUST-004'],
        'email': ['john.smith@gmail.com', 'sarah.jones@company.co.uk', 'mike.wilson@techstartup.io', 'admin@enterprise-corp.com'],
        'phone': ['+1-555-123-4567', '(555) 987-6543', '555.234.5678', '+44-20-7123-4567'],
        'website': ['https://johnsmith.dev', 'https://company.co.uk', None, 'https://www.enterprise-corp.com'],
        'company_size': ['Small', 'Medium', 'Startup', 'Enterprise'],
        'industry': ['Technology', 'Manufacturing', 'SaaS', 'Finance'],
        'location': ['New York, NY', 'London, UK', 'San Francisco, CA', 'Frankfurt, Germany'],
        'lead_source': ['Website', 'Referral', 'Cold Email', 'Trade Show'],
        'lead_score': [75, 92, 45, 88],
        'is_qualified': [True, True, False, True]
    })
    
    return test_datasets

if __name__ == "__main__":
    test_datasets = create_comprehensive_test_datasets()
    for name, df in test_datasets.items():
        features, types = process_data_source(name, df)
        print(f"\nProcessed {name}: {features.shape} features, types: {types}")