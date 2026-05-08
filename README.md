# NeuroPulse Mental Health Monitor

A lightweight, Vercel-ready Python backend for stress level classification using a Voting Ensemble machine learning model. This project predicts mental stress levels using biometric and psychological signals.

## Features

- **Voting Ensemble Model**: High accuracy (90.28%) classifier utilizing `scikit-learn` to predict stress levels.
- **Serverless Architecture**: Streamlined to run entirely on Vercel Serverless Functions via `api/index.py`.
- **Fast & Minimal**: Zero bloated dependencies. The Flask app handles API routing and template rendering, designed specifically for rapid cold starts.
- **Tiered Recommendations**: Custom recommendations and early warning systems based on Mental Load Index (MLI).

## Project Structure

```text
mental_health_project/
├── api/
│   ├── index.py              # Main Vercel Serverless entrypoint (Flask App)
│   ├── predictor.py          # Lightweight ML inference wrapper
│   ├── recommendations.py    # Generates tiered advisory responses
│   └── templates/            # HTML templates for dashboard & about pages
├── model/
│   ├── mental_health_model.pkl         # Trained voting ensemble model
│   ├── mental_health_model_scaler.pkl  # Scaler for ML input
│   ├── mental_health_model_encoder.pkl # Label encoder
│   └── mental_health_model_features.pkl# Feature definitions
├── public/
│   └── static/               # CSS, JS, and image assets
├── scripts/                  # Helper scripts for dataset generation
├── tests/                    # Unit tests
├── requirements.txt          # Python dependencies
├── runtime.txt               # Vercel Python runtime configuration
└── vercel.json               # Vercel configuration file
```

## Running Locally

To run the dashboard and API locally, simply execute the `index.py` file using Python:

```bash
# 1. Create a virtual environment (optional but recommended)
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On Linux/Mac: source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the Flask server
python api/index.py
```

The application will be available at `http://127.0.0.1:5000`.

## Vercel Deployment

This project is fully configured for a 1-click deployment to Vercel. 

### Deployment Steps:
1. Push this repository to GitHub, GitLab, or Bitbucket.
2. Log in to [Vercel](https://vercel.com/) and click **Add New Project**.
3. Import your repository.
4. Leave the Framework Preset as `Other`.
5. Ensure the Root Directory is correct.
6. Click **Deploy**.

Vercel will automatically read `vercel.json` and deploy `api/index.py` as a Serverless Python Function. 

### Technical Note on Model Size
The model artifacts (e.g. `mental_health_model.pkl`) are explicitly bundled into the serverless function using the `includeFiles` directive in `vercel.json`. The total size is optimized to fit within Vercel's free tier limits.

## API Documentation

### `GET /api/health`
Checks the server and model artifact status.

### `GET /api/model-info`
Returns information about the loaded ML model and its features.

### `POST /api/predict`
Accepts biometric data and returns stress predictions, Mental Load Index (MLI), and recommendations.

**Request Body Example:**
```json
{
  "heart_rate": 78,
  "hrv": 64,
  "respiration": 16,
  "skin_temp": 36.4,
  "bp_systolic": 119,
  "bp_diastolic": 78,
  "cognitive_state": 3,
  "emotional_state": 3
}
```
