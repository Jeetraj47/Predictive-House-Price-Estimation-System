"""
app.py  –  Predictive House Price Estimation System
Flask REST API serving the trained HistGradientBoosting+Ridge ensemble.
"""

import os
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# ── Load model artifacts ───────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

model    = None
scaler   = None
metadata = {}

def load_artifacts():
    global model, scaler, metadata
    try:
        model  = joblib.load(os.path.join(MODEL_DIR, 'model.joblib'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
        with open(os.path.join(MODEL_DIR, 'metadata.json')) as f:
            metadata = json.load(f)
        print("[OK] Model artifacts loaded successfully.")
    except FileNotFoundError:
        print("[WARN] Model artifacts not found. Run train_model.py first.")

load_artifacts()

# ── Feature engineering (mirrors train_model.py) ──────────────────────────
NEIGHBORHOOD_TIER = {
    'Luxury Hills': 5, 'Waterfront': 5, 'Tech District': 4, 'Downtown': 4,
    'Midtown': 3, 'University': 3, 'Old Town': 2, 'Suburbs': 2,
    'Industrial': 1, 'Rural': 1
}
CONDITION_MAP = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}
NEIGHBORHOOD_ENC = {
    'Downtown': 0, 'Industrial': 1, 'Luxury Hills': 2, 'Midtown': 3,
    'Old Town': 4, 'Rural': 5, 'Suburbs': 6, 'Tech District': 7,
    'University': 8, 'Waterfront': 9
}

# Geo-price medians (pre-computed approximations for inference)
GEO_PRICE_MEDIAN_DEFAULT = 320000.0

def build_feature_vector(data: dict) -> np.ndarray:
    sqft             = float(data['sqft'])
    bedrooms         = int(data['bedrooms'])
    bathrooms        = int(data['bathrooms'])
    year_built       = int(data['year_built'])
    garage_spaces    = int(data.get('garage_spaces', 1))
    has_pool         = int(data.get('has_pool', 0))
    has_basement     = int(data.get('has_basement', 0))
    floors           = int(data.get('floors', 1))
    condition_label  = data.get('condition', 'Good')
    lot_size         = float(data.get('lot_size', 7500))
    distance_downtown= float(data.get('distance_downtown', 8.0))
    school_rating    = float(data.get('school_rating', 6.5))
    crime_rate       = float(data.get('crime_rate', 3.0))
    neighborhood     = data.get('neighborhood', 'Suburbs')
    renovated        = int(data.get('renovated', 0))
    renovation_year  = int(data.get('renovation_year', 0))
    hoa_fee          = float(data.get('hoa_fee', 200))
    lat              = float(data.get('lat', 37.0))
    lon              = float(data.get('lon', -95.0))

    age              = 2024 - year_built
    tier             = NEIGHBORHOOD_TIER.get(neighborhood, 2)
    condition        = CONDITION_MAP.get(condition_label, 3)
    neighborhood_enc = NEIGHBORHOOD_ENC.get(neighborhood, 6)

    # Engineered features
    log_sqft            = np.log1p(sqft)
    log_lot_size        = np.log1p(lot_size)
    log_distance        = np.log1p(distance_downtown)
    sqft_x_bath         = sqft * bathrooms
    sqft_x_condition    = sqft * condition
    age_x_condition     = age  * condition
    bed_bath_ratio      = bedrooms / (bathrooms + 1e-3)
    sqft_per_room       = sqft / (bedrooms + bathrooms + 1)
    tier_x_sqft         = tier * sqft
    school_x_tier       = school_rating * tier
    amenity_score       = has_pool * 3 + has_basement * 2 + garage_spaces * 1.5 + renovated * 2 + floors
    years_since_reno    = (2024 - renovation_year) if renovated else age
    is_new_build        = int(age < 5)
    is_vintage          = int(age > 50)
    cost_index          = tier * school_rating / (crime_rate + 1)
    value_density       = sqft / (lot_size + 1)
    hoa_per_sqft        = hoa_fee / (sqft + 1)
    sqft_sq             = sqft ** 2
    school_rating_sq    = school_rating ** 2
    geo_price_median    = GEO_PRICE_MEDIAN_DEFAULT

    features = [
        sqft, bedrooms, bathrooms, age, garage_spaces,
        has_pool, has_basement, floors, condition,
        lot_size, distance_downtown, school_rating, crime_rate,
        tier, renovated, hoa_fee,
        log_sqft, log_lot_size, log_distance,
        sqft_x_bath, sqft_x_condition, age_x_condition,
        bed_bath_ratio, sqft_per_room, tier_x_sqft, school_x_tier,
        amenity_score, years_since_reno, is_new_build, is_vintage,
        cost_index, value_density, hoa_per_sqft,
        sqft_sq, school_rating_sq,
        geo_price_median, neighborhood_enc,
        lat, lon
    ]
    return np.array(features, dtype=np.float64).reshape(1, -1)


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', metadata=metadata)


@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'r2_score': metadata.get('r2_score'),
        'n_samples': metadata.get('n_samples')
    })


@app.route('/api/metadata')
def get_metadata():
    return jsonify(metadata)


@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded. Run train_model.py first.'}), 503

    data = request.get_json(force=True) if request.is_json else request.form.to_dict()

    required = ['sqft', 'bedrooms', 'bathrooms', 'year_built']
    missing  = [k for k in required if k not in data or data[k] == '']
    if missing:
        return jsonify({'error': f'Missing required fields: {missing}'}), 400

    try:
        X = build_feature_vector(data)
        X_scaled = scaler.transform(X)
        log_pred = model.predict(X_scaled)[0]
        price    = float(np.expm1(log_pred))

        # Confidence band ±8 % (reflects training noise)
        low  = round(price * 0.92, 2)
        high = round(price * 1.08, 2)

        return jsonify({
            'predicted_price': round(price, 2),
            'price_range_low' : low,
            'price_range_high': high,
            'formatted_price' : f"${price:,.0f}",
            'formatted_range' : f"${low:,.0f} – ${high:,.0f}",
        })
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/neighborhoods')
def neighborhoods():
    return jsonify(list(NEIGHBORHOOD_TIER.keys()))


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
