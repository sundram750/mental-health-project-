# Configuration file for Mental Health Monitoring System

# Flask Configuration
FLASK_ENV = 'development'
FLASK_DEBUG = True
FLASK_HOST = '127.0.0.1'
FLASK_PORT = 5000

# Model Configuration
MODEL_PATH = 'model/mental_health_model.pkl'
SCALER_PATH = 'model/mental_health_model_scaler.pkl'
ENCODER_PATH = 'model/mental_health_model_encoder.pkl'
FEATURES_PATH = 'model/mental_health_model_features.pkl'

# Data Configuration
DATA_PATH = 'data/Cleaned_Dataset Model 3.csv'
OUTPUT_PATH = 'outputs/'

# Model Training Configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Mental Load Index Configuration
MLI_LOW_THRESHOLD = 40
MLI_MODERATE_THRESHOLD = 70
MLI_HISTORY_SIZE = 7

# Early Warning Configuration
WARNING_HISTORY_SIZE = 7
BURNOUT_RISK_THRESHOLD = 60
ESCALATION_TIMEFRAME_HOURS = 72

# Stress Level Mapping
STRESS_LEVELS = ['Low', 'Moderate', 'High']
STRESS_SCORE_MAPPING = {
    'Low': 1,
    'Moderate': 2,
    'High': 3
}

# Physiological Ranges
HEART_RATE_NORMAL_MIN = 60
HEART_RATE_NORMAL_MAX = 100
HRV_NORMAL_MIN = 20
HRV_NORMAL_MAX = 100
RESPIRATION_NORMAL_MIN = 12
RESPIRATION_NORMAL_MAX = 20
SKIN_TEMP_NORMAL_MIN = 33
SKIN_TEMP_NORMAL_MAX = 36
BP_SYSTOLIC_NORMAL_MIN = 90
BP_SYSTOLIC_NORMAL_MAX = 140

# Logging Configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Session Configuration
SESSION_TIMEOUT_HOURS = 24
MAX_SESSION_RECORDS = 500

# Feature Names (in training order)
FEATURE_NAMES = [
    'Heart_Rate',
    'HRV',
    'Respiration',
    'Skin_Temp',
    'BP_Systolic',
    'BP_Diastolic',
    'Cognitive_State',
    'Emotional_State'
]

# Model Parameters
RF_N_ESTIMATORS = [100, 200, 300]
RF_MAX_DEPTH = [10, 15, 20]
XGB_N_ESTIMATORS = [100, 200, 300]
XGB_MAX_DEPTH = [3, 5, 7]
XGB_LEARNING_RATE = [0.01, 0.05, 0.1]

# Accuracy Targets
TARGET_ACCURACY = 0.85
TARGET_F1_SCORE = 0.85
