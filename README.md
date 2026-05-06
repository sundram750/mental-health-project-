# Mental Health Monitoring System
## IoT-Enabled Wearable Sensors for Mental Health Monitoring in Biotechnology

A comprehensive AI-based system for real-time mental health monitoring, stress assessment, and personalized intervention recommendations using machine learning and physiological data.

---

## 📋 Project Overview

This system is designed for MBA Biotech dissertation research and provides:

- **ML-based Stress Prediction**: Random Forest and XGBoost models for accurate stress level classification
- **Mental Load Index (MLI)**: Composite metric (0-100 scale) combining physiological and psychological factors
- **Advisory Generation**: Dynamic, context-aware recommendations based on stress levels
- **Early Warning System**: Predicts burnout risk and stress escalation within 48-72 hours
- **Personalized Interventions**: Tiered recommendations (natural, OTC, professional) based on stress severity
- **7-Day Trend Monitoring**: Tracks stress patterns and generates insights
- **Web Interface**: Flask-based interactive dashboard for data input and visualization

---

## 🏗 Project Structure

```
mental_health_project/
│
├── data/                          # Data directory
│   └── Cleaned_Dataset Model 3.csv
│
├── model/                         # ML model modules
│   ├── __init__.py
│   ├── train_model.py            # Model training & evaluation
│   └── predictor.py              # Inference module
│
├── app/                          # Flask web application
│   ├── __init__.py
│   ├── app.py                    # Main Flask app with routes
│   ├── templates/
│   │   └── index.html            # Main dashboard
│   └── static/
│       ├── style.css             # Styling
│       └── script.js             # Frontend logic
│
├── utils/                        # Utility modules
│   ├── __init__.py
│   ├── mental_load_calculator.py # MLI calculation
│   ├── advisory_engine.py        # Advisory generation
│   ├── early_warning.py          # Risk prediction
│   └── recommendation_engine.py  # Intervention recommendations
│
└── requirements.txt              # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Your cleaned dataset: `Cleaned_Dataset Model 3.csv`

### Installation

1. **Navigate to project directory**:
   ```bash
   cd mental_health_project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Place your dataset** in the `data/` folder:
   ```
   data/Cleaned_Dataset Model 3.csv
   ```

### Training the Model

1. **Run the training script**:
   ```bash
   python model/train_model.py
   ```

The script will:
- Load and validate your dataset
- Perform train-test split (80-20)
- Train Random Forest and XGBoost models
- Perform hyperparameter tuning with GridSearchCV
- Evaluate both models on test set
- Generate feature importance visualizations
- Save the best model as `model/mental_health_model.pkl`
- Generate detailed training report in `outputs/`

**Expected Output**:
- Trained model files in `model/` directory
- Feature importance charts
- Confusion matrices
- Classification reports
- Training summary

### Running the Web Application

1. **Start the Flask server**:
   ```bash
   python app/app.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://127.0.0.1:5000
   ```

3. **Input your data**:
   - Enter physiological measurements (Heart Rate, HRV, Respiration, etc.)
   - Complete psychological questionnaire (1-5 scale sliders)
   - Click "Analyze Stress Level"

---

## 📊 Input Data Format

### Physiological Metrics
- **Heart Rate (BPM)**: 40-200 (typical resting: 60-100)
- **HRV (Heart Rate Variability)**: 0-200 (higher = better stress regulation)
- **Respiration Rate**: 8-40 breaths per minute (normal: 12-20)
- **Skin Temperature (°C)**: 32-38°C (typical: 34-36)
- **BP Systolic (mmHg)**: 80-200 (normal: 90-120)
- **BP Diastolic (mmHg)**: 40-120 (normal: 60-80)

### Psychological Assessment (1-5 Scale)
- **Mental Fatigue**: 1=Not at all, 5=Extremely
- **Anxiety Level**: 1=Very Calm, 5=Very Anxious

---

## 💡 Key Features

### 1. Mental Load Index (MLI)
Combines physiological and psychological data into a 0-100 scale:

```
MLI Components (with weights):
- Heart Rate Stress: 20%
- HRV Stress: 15%
- Respiration Stress: 15%
- Temperature Stress: 10%
- Blood Pressure Stress: 15%
- Cognitive State: 12%
- Emotional State: 13%
```

**Status Levels**:
- **Low (< 40)**: Stable - maintain current routine
- **Moderate (40-70)**: Elevated Load - implement recovery strategies
- **High (> 70)**: Critical Load - immediate intervention needed

### 2. Advisory Engine
Generates actionable recommendations based on stress level with NO numerical data:

**Low Stress**:
- Maintain current routine
- Light physical activity
- Continue wellness practices

**Moderate Stress**:
- Take recovery breaks hourly
- Reduce multitasking
- Practice deep breathing
- Ensure adequate sleep

**High Stress**:
- Immediate recovery advised
- Avoid demanding tasks
- Guided relaxation/meditation
- Professional consultation recommended

### 3. Early Warning System
Predicts stress escalation using:
- Current stress level
- 7-day stress trends
- HRV decline rate
- MLI trend slope

**Outputs**:
- Burnout Risk Level: Low / Moderate / High
- Escalation Prediction: 24-72 hours
- Confidence Score: 0-100%

### 4. Personalized Recommendation System
Tiered approach based on stress severity:

**Natural Interventions** (All levels):
- Yoga, meditation, breathing exercises
- Ashwagandha, Brahmi
- Nature exposure

**OTC Supplements** (Moderate & High):
- Magnesium, Valerian, Passionflower
- Herbal calming supplements
- Sleep support aids

**Professional Services** (Moderate & High):
- Psychologist consultation
- Cognitive behavioral therapy
- Psychiatric evaluation

---

## 🔌 API Endpoints

### Main Endpoints

#### 1. POST `/api/predict`
**Request**:
```json
{
  "heart_rate": 75,
  "hrv": 50,
  "respiration": 16,
  "skin_temp": 35.5,
  "bp_systolic": 120,
  "bp_diastolic": 80,
  "cognitive_state": 3,
  "emotional_state": 3
}
```

**Response**:
```json
{
  "timestamp": "2026-02-16T10:30:00.000Z",
  "ml_prediction": {
    "stress_level": "Moderate",
    "confidence": 0.92,
    "probabilities": {
      "Low": 0.08,
      "Moderate": 0.92,
      "High": 0.00
    }
  },
  "mental_load_index": {
    "score": 52.3,
    "status": "Elevated Load",
    "category": "Moderate",
    "components": { ... }
  },
  "advisory": {
    "message": "Your mental load is elevated...",
    "recommendations": [ ... ],
    "implementation_duration": "3-5 days",
    "priority": "Immediate attention needed"
  },
  "early_warning": {
    "burnout_risk_level": "Moderate",
    "burnout_risk_score": 45.5,
    "escalation_risk": "High stress risk possible within 48–72 hours",
    "escalation_confidence": 65.0
  },
  "recommendations": {
    "stress_level": "Moderate",
    "natural_interventions": [ ... ],
    "otc_options": [ ... ],
    "professional_services": [ ... ]
  }
}
```

#### 2. GET `/api/trends`
Returns 7-day stress trend data
```json
{
  "timestamps": [ ... ],
  "stress_levels": [ ... ],
  "mli_scores": [ ... ],
  "summary": { ... }
}
```

#### 3. GET `/api/health`
Health check endpoint

#### 4. GET `/api/model-info`
Model and feature information

#### 5. GET `/api/session-info`
Current user session information

#### 6. POST `/api/clear-session`
Clear user session data

---

## 📈 Model Performance

### Expected Metrics (Target: >85% accuracy)

**Random Forest**:
- Accuracy: 85-92%
- Precision: 84-90%
- Recall: 85-91%
- F1-Score: 85-91%

**XGBoost**:
- Accuracy: 86-93%
- Precision: 85-92%
- Recall: 86-92%
- F1-Score: 86-92%

### Outputs Generated
- Feature importance charts (HTML)
- Confusion matrices (visual)
- Classification reports
- Training summary report

---

## 🧪 Testing the System

### 1. Test Model Training
```bash
python model/train_model.py
```

### 2. Test Predictions
```bash
cd model
python predictor.py
```

### 3. Test Utilities
```bash
python utils/mental_load_calculator.py
python utils/advisory_engine.py
python utils/early_warning.py
python utils/recommendation_engine.py
```

### 4. Run Web Application
```bash
python app/app.py
```

---

## 📝 Dataset Requirements

Your `Cleaned_Dataset Model 3.csv` must contain:

**Required Columns**:
- `Heart_Rate`: Numeric (BPM)
- `HRV`: Numeric
- `Respiration`: Numeric (breaths/min)
- `Skin_Temp`: Numeric (°C)
- `BP_Systolic`: Numeric (mmHg)
- `BP_Diastolic`: Numeric (mmHg)
- `Cognitive_State`: Numeric (1-5) or Categorical
- `Emotional_State`: Numeric (1-5) or Categorical
- `Stress_Level`: Categorical ('Low', 'Moderate', 'High') 

**Data Quality**:
- Minimum 100 samples recommended
- Balanced classes for optimal model performance
- No critical missing values
- Numeric features should be properly scaled/normalized

---

## 🔒 Medical Disclaimer

⚠️ **Important**: This system is designed for **research and educational purposes only**. 

- These recommendations are NOT medical advice
- Always consult healthcare professionals before starting treatments
- Not a replacement for professional mental health services
- For mental health emergencies, contact local crisis services immediately

---

## 🎓 Academic Use

This project is designed for MBA Biotech dissertation:
- **Title**: "IoT-Enabled Wearable Sensors for Mental Health Monitoring in Biotechnology"
- **Purpose**: Academic research and demonstration
- **Framework**: Production-ready, scalable architecture

### Citation
If using this project, please cite:
```
Mental Health Monitoring System (2026)
IoT-Enabled Wearable Sensors for Mental Health Monitoring in Biotechnology
MBA Biotech Dissertation Project
```

---

## 🛠 Troubleshooting

### Model Not Loading
```
✗ Error: Model file not found
→ Run: python model/train_model.py
```

### Dataset Not Found
```
✗ Error: File not found
→ Place CSV in data/ folder with correct name
```

### Flask Port Already in Use
```
✗ Error: Address already in use
→ Change port in app.py: app.run(port=5001)
```

### Import Errors
```
→ Ensure all packages installed: pip install -r requirements.txt
→ Check Python version: python --version (3.8+)
```

---

## 📚 Module Documentation

### `model/train_model.py`
Main training pipeline with:
- Data loading and validation
- Feature preparation
- Train-test split with stratification
- Random Forest hyperparameter tuning
- XGBoost hyperparameter tuning
- Model evaluation and comparison
- Visualization generation

### `model/predictor.py`
Inference module with:
- Model loading from pickle files
- Input validation
- Single prediction
- Batch predictions
- Prediction explanation

### `utils/mental_load_calculator.py`
MLI calculation engine with:
- Physiological stress quantification
- Psychological state assessment
- Weighted composite scoring
- History tracking

### `utils/advisory_engine.py`
Advice generation with:
- Stress-level-based recommendations
- Contextual advisory generation
- Advice formatting for display

### `utils/early_warning.py`
Risk prediction module with:
- Trend analysis (7-day)
- Burnout risk assessment
- Escalation prediction
- Pattern detection

### `utils/recommendation_engine.py`
Tiered recommendation system with:
- Natural intervention suggestions
- OTC supplement recommendations
- Professional service guidance
- Medical disclaimers

### `app/app.py`
Flask web application with:
- RESTful API endpoints
- Session management
- Integrated module orchestration
- Error handling

---

## 🔄 Workflow

```
User Input (Physiological + Psychological)
        ↓
ML Model Prediction (Stress Level + Confidence)
        ↓
Mental Load Index Calculation (0-100 score)
        ↓
Advisory Generation (Actionable recommendations)
        ↓
Early Warning Assessment (Risk evaluation)
        ↓
Personalized Recommendations (Tiered approach)
        ↓
Results Display (Comprehensive dashboard)
        ↓
Historical Tracking (7-day trends)
```

---

## 📞 Support & Maintenance

For issues or questions:
1. Check troubleshooting section
2. Review module documentation
3. Examine error messages in console
4. Verify dataset format and contents
5. Ensure all dependencies installed

---

## 📄 License

This project is for academic and research purposes.

---

## ✨ Key Features Summary

✅ Random Forest + XGBoost models
✅ >85% prediction accuracy target
✅ Mental Load Index (0-100)
✅ Dynamic advisory generation
✅ Early warning system
✅ 7-day trend monitoring
✅ Tiered recommendations
✅ Web-based dashboard
✅ RESTful API
✅ Production-ready code
✅ Modular architecture
✅ Comprehensive documentation

---

**Version**: 1.0
**Last Updated**: February 2026
**Status**: Production Ready
