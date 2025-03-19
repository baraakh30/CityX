import gc
import glob
import re
import joblib
import pandas as pd
import folium
import PyPDF2
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app import app
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib
import os



def categorize_severity(category):
    severity_mapping = {
        1: ['NON-CRIMINAL', 'SUSPICIOUS OCCURRENCE', 'MISSING PERSON', 'RUNAWAY', 'RECOVERED VEHICLE'],
        2: ['WARRANTS', 'OTHER OFFENSES', 'VANDALISM', 'TRESPASS', 'DISORDERLY CONDUCT', 'BAD CHECKS'],
        3: ['LARCENY/THEFT', 'VEHICLE THEFT', 'FORGERY/COUNTERFEITING', 'DRUG/NARCOTIC', 'STOLEN PROPERTY', 'FRAUD', 'BRIBERY', 'EMBEZZLEMENT'],
        4: ['ROBBERY', 'WEAPON LAWS', 'BURGLARY', 'EXTORTION'],
        5: ['KIDNAPPING', 'ARSON']
    }
    for severity, categories in severity_mapping.items():
        if category in categories:
            return severity
    return 0


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if os.path.exists('app/static/transformer_model'):
    try:
        print("Loading pre-trained transformer model...")
        tokenizer = AutoTokenizer.from_pretrained('app/static/transformer_tokenizer')
        model = AutoModelForSequenceClassification.from_pretrained('app/static/transformer_model').to(device)
        le = joblib.load('app/static/label_encoder.pkl')
        
    except Exception as e:
        print(f"Error in transformer model loading: {str(e)}")
        import traceback
        traceback.print_exc()

else:
    print('Make sure to run model_training.py first to train the model.')
    exit(1)

def extract_pdf_data(pdf_file):
    """
    Extract key information from a police report PDF.
    Returns a dictionary with extracted fields.
    """
    if isinstance(pdf_file, str):  # If it's a file path
        with open(pdf_file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
    else:  # If it's a file-like object
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    
    # Initialize data dictionary with empty values
    data = {
        'Report Number': None,
        'Date & Time': None,
        'Reporting Officer': None,
        'Incident Location': None,
        'Coordinates': None,
        'Police District': None,
        'Resolution': None,
        'Suspect Description': None,
        'Victim Information': None,
        'Detailed Description': None,
        'Category': None,
        'Severity': None,
        'Latitude (Y)': None,
        'Longitude (X)': None,
        'DayOfWeek': None,
        'Address': None
    }
    
    # Use regex patterns to extract information
    simple_patterns = {
        'Report Number': r'Report Number:\s*(.*?)(?:\n|$)',
        'Date & Time': r'Date & Time:\s*(.*?)(?:\n|$)',
        'Reporting Officer': r'Reporting Officer:\s*(.*?)(?:\n|$)',
        'Incident Location': r'Incident Location:\s*(.*?)(?:\n|$)',
        'Coordinates': r'Coordinates:\s*\((.*?),\s*(.*?)\)',
        'Police District': r'Police District:\s*(.*?)(?:\n|$)',
        'Resolution': r'Resolution:\s*(.*?)(?:\n|$)',
        'Suspect Description': r'Suspect Description:\s*(.*?)(?:\n|$)',
        'Victim Information': r'Victim Information:\s*(.*?)(?:\n|$)'
    }
    
    # Extract simple pattern data
    for key, pattern in simple_patterns.items():
        if key == 'Coordinates':
            coords_match = re.search(pattern, text)
            if coords_match:
                data['Latitude (Y)'] = coords_match.group(1).strip()
                data['Longitude (X)'] = coords_match.group(2).strip()
        else:
            match = re.search(pattern, text)
            if match:
                data[key] = match.group(1).strip()
    
    # Handle the detailed description separately to capture multi-line content
    desc_start_match = re.search(r'Detailed Description:\s*(.*?)(?:\n|$)', text)
    if desc_start_match:
        start_pos = text.find('Detailed Description:') + len('Detailed Description:')
        end_pos = text.find('Police District:', start_pos)
        if end_pos == -1:  # If Police District not found, try other likely fields
            end_pos = text.find('Resolution:', start_pos)
        if end_pos == -1:  # If still not found, try another field
            end_pos = text.find('Suspect Description:', start_pos)
        if end_pos == -1:  # If still not found, use the rest of the text
            end_pos = len(text)
        
        # Extract the description text
        description_text = text[start_pos:end_pos].strip()
        data['Detailed Description'] = description_text
    
    # Extract day of week from the date if available
    if data['Date & Time']:
        try:
            day_of_week = pd.to_datetime(data['Date & Time']).day_name()
            data['DayOfWeek'] = day_of_week
        except:
            pass
    return data
# Process PDFs on app startup
def predict_category(text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    pred_id = torch.argmax(logits, dim=-1).item()
    return le.inverse_transform([pred_id])[0]

pdf_folder = 'app/static/pdf_reports' 
extracted_reports = []
for pdf_file in glob.glob(os.path.join(pdf_folder, '*.pdf')):
    report = extract_pdf_data(pdf_file)
    desc = report['Detailed Description']
    if desc:
        report['Category'] = predict_category(desc)
        report['Severity'] = categorize_severity(report['Category'])
    else:
        report['Category'] = 'N/A'
        report['Severity'] = 0
    extracted_reports.append(report)
print("Loading dataset...")
df = pd.read_csv('Competition_Dataset.csv')

df['Severity'] = df['Category'].apply(categorize_severity)

temp = df['Latitude (Y)']
df['Latitude (Y)'] = df['Longitude (X)']
df['Longitude (X)'] = temp
# Geo-Spatial Mapping
def generate_maps():

    print("\n--- Creating Geo-Spatial Visualization ---")

    # Create a base map
    base_map = folium.Map(location=[df['Latitude (Y)'].mean(), df['Longitude (X)'].mean()], zoom_start=12)

    # Add marker cluster for individual crime locations
    from folium.plugins import MarkerCluster

    # Create a dictionary to store category-specific coordinates
    category_coords = {}
    category_counts = df['Category'].value_counts()
    top_categories = category_counts.head(10).index.tolist()

    # Initialize marker clusters for each top category
    category_markers = {}
    for category in top_categories:
        category_markers[category] = MarkerCluster(name=f"{category}")

    # Add markers for each category
    for category in top_categories:
        category_data = df[df['Category'] == category]
        for idx, row in category_data.sample(min(500, len(category_data))).iterrows():
            popup_text = f"""
            <b>Category:</b> {row['Category']}<br>
            <b>Description:</b> {row['Descript']}<br>
            <b>Date:</b> {row['Dates']}<br>
            <b>Day:</b> {row['DayOfWeek']}<br>
            <b>District:</b> {row['PdDistrict']}<br>
            <b>Resolution:</b> {row['Resolution']}<br>
            """
            
            # Determine marker color based on severity
            severity = categorize_severity(row['Category'])
            colors = {0: 'gray', 1: 'blue', 2: 'green', 3: 'orange', 4: 'red', 5: 'darkred'}
            
            folium.Marker(
                location=[row['Latitude (Y)'], row['Longitude (X)']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color=colors.get(severity, 'blue'))
            ).add_to(category_markers[category])
            
        # Add the marker cluster to the map
        category_markers[category].add_to(base_map)

    # Add layer control
    folium.LayerControl().add_to(base_map)

    try:
        # Save interactive map
        base_map.save("app/static/crime_heatmap.html")
        print("Interactive map with markers saved to app/static/crime_heatmap.html")
    except Exception as e:
        print(f"Error saving crime map: {str(e)}")

    # Create marker clusters by severity
    severity_markers = {
        1: MarkerCluster(name="Low Severity (1)"),
        2: MarkerCluster(name="Moderate Severity (2)"),
        3: MarkerCluster(name="Medium Severity (3)"),
        4: MarkerCluster(name="High Severity (4)"),
        5: MarkerCluster(name="Critical Severity (5)")
    }

    # Create a severity map
    severity_map = folium.Map(location=[df['Latitude (Y)'].mean(), df['Longitude (X)'].mean()], zoom_start=12)

    # Add markers by severity
    severity_colors = {1: 'blue', 2: 'green', 3: 'orange', 4: 'red', 5: 'darkred'}
    for idx, row in df.sample(min(2000, len(df))).iterrows():
        severity = row['Severity']
        if severity > 0:  # Skip undefined severity
            popup_text = f"""
            <b>Category:</b> {row['Category']}<br>
            <b>Severity:</b> {severity}<br>
            <b>Description:</b> {row['Descript']}<br>
            <b>Date:</b> {row['Dates']}<br>
            <b>Resolution:</b> {row['Resolution']}<br>
            """
            
            folium.Marker(
                location=[row['Latitude (Y)'], row['Longitude (X)']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color=severity_colors.get(severity, 'blue'))
            ).add_to(severity_markers[severity])

    # Add severity clusters to the map
    for severity, marker in severity_markers.items():
        marker.add_to(severity_map)

    # Add layer control
    folium.LayerControl().add_to(severity_map)

    try:
        # Save severity map with markers only
        severity_map.save("app/static/severity_heatmap.html")
        print("Severity map with markers saved to app/static/severity_heatmap.html")
    except Exception as e:
        print(f"Error saving severity map: {str(e)}")

generate_maps()
data = df[["Dates", "Category", "PdDistrict", "Latitude (Y)", "Longitude (X)"]].copy()
data["Dates"] = pd.to_datetime(data["Dates"])
data["Hour"] = data["Dates"].dt.hour
data["Day"] = data["Dates"].dt.day
data["Month"] = data["Dates"].dt.month
data["Year"] = data["Dates"].dt.year
data["Weekday"] = data["Dates"].dt.weekday
data["Date"] = data["Dates"].dt.date
data["Date"] = pd.to_datetime(data["Date"])
del df
district_center_data = data.groupby('PdDistrict').agg({
        'Latitude (Y)': 'mean', 
        'Longitude (X)': 'mean'
    }).reset_index()

    # Create district_centers dictionary from the actual data
district_centers = {}
for _, row in district_center_data.iterrows():
        district = row['PdDistrict']
        lat = row['Latitude (Y)']
        lon = row['Longitude (X)']
        district_centers[district] = [lat, lon]
        

# Path to save trained models
MODEL_DIR = 'app/static/models'
os.makedirs(MODEL_DIR, exist_ok=True)

def get_holiday_dates(start_year=2003, end_year=2015):
    """Generate common US holidays for the given years"""
    holidays = []
    
    for year in range(start_year, end_year + 1):
        # New Year's Day
        holidays.append(f"{year}-01-01")
        # Martin Luther King Jr. Day (3rd Monday in January)
        holidays.append(pd.Timestamp(f"{year}-01-01") + 
                       pd.tseries.offsets.DateOffset(weekday=0, weeks=2))  # 0 = Monday
        # Presidents Day (3rd Monday in February)
        holidays.append(pd.Timestamp(f"{year}-02-01") + 
                       pd.tseries.offsets.DateOffset(weekday=0, weeks=2))  # 0 = Monday
        # Memorial Day (Last Monday in May)
        holidays.append(pd.Timestamp(f"{year}-05-31").normalize() - 
                       pd.tseries.offsets.DateOffset(weekday=0, weeks=0))  # 0 = Monday
        # Independence Day
        holidays.append(f"{year}-07-04")
        # Labor Day (1st Monday in September)
        holidays.append(pd.Timestamp(f"{year}-09-01") + 
                       pd.tseries.offsets.DateOffset(weekday=0, weeks=0))  # 0 = Monday
        # Columbus Day (2nd Monday in October)
        holidays.append(pd.Timestamp(f"{year}-10-01") + 
                       pd.tseries.offsets.DateOffset(weekday=0, weeks=1))  # 0 = Monday
        # Veterans Day
        holidays.append(f"{year}-11-11")
        # Thanksgiving (4th Thursday in November)
        holidays.append(pd.Timestamp(f"{year}-11-01") + 
                       pd.tseries.offsets.DateOffset(weekday=3, weeks=3))  # 3 = Thursday
        # Christmas
        holidays.append(f"{year}-12-25")
    
    return pd.to_datetime(holidays)

def prepare_features(data, date_to_predict=None):
    """Prepare features for crime prediction models"""
    # Make a copy of the data
    X = data.copy()
    
    # Convert string dates to datetime if needed
    if isinstance(X['Date'].iloc[0], str):
        X['Date'] = pd.to_datetime(X['Date'])
    
    # Extract temporal features
    X['DayOfWeek'] = X['Date'].dt.dayofweek
    X['Month'] = X['Date'].dt.month
    X['Year'] = X['Date'].dt.year
    X['IsWeekend'] = X['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Add US holidays as a feature
    holidays = get_holiday_dates()
    X['IsHoliday'] = X['Date'].isin(holidays).astype(int)
    
    # If preparing features for a single date to predict
    if date_to_predict is not None:
        if isinstance(date_to_predict, str):
            date_to_predict = pd.to_datetime(date_to_predict).date()
            
        pred_date = pd.Timestamp(date_to_predict)
        X = pd.DataFrame({
            'Date': [pred_date],
            'DayOfWeek': [pred_date.dayofweek],
            'Month': [pred_date.month],
            'Year': [pred_date.year],
            'IsWeekend': [1 if pred_date.dayofweek >= 5 else 0],
            'IsHoliday': [1 if pred_date in holidays else 0]
        })
    
    return X

def train_crime_prediction_models(data, force_retrain=False):
    """Train models to predict crime counts, categories, and locations with reduced feature set"""
    model_files = {
        'count': os.path.join(MODEL_DIR, 'crime_count_model.pkl'),
        'category': os.path.join(MODEL_DIR, 'crime_category_model.pkl'),
        'district': os.path.join(MODEL_DIR, 'crime_district_model.pkl'),
        'lat': os.path.join(MODEL_DIR, 'crime_lat_model.pkl'),
        'long': os.path.join(MODEL_DIR, 'crime_long_model.pkl'),
        'hour': os.path.join(MODEL_DIR, 'crime_hour_model.pkl')
    }
    
    # Check if models already exist
    if not force_retrain and all(os.path.exists(file) for file in model_files.values()):
        print("Loading existing crime prediction models...")
        models = {
            'count': joblib.load(model_files['count']),
            'category': joblib.load(model_files['category']),
            'district': joblib.load(model_files['district']),
            'lat': joblib.load(model_files['lat']),
            'long': joblib.load(model_files['long']),
            'hour': joblib.load(model_files['hour'])
        }
        return models
    
    print("Training crime prediction models...")
    
    # Prepare data for daily crime count prediction
    daily_counts = data.groupby('Date').size().reset_index(name='crime_count')
    X_count = prepare_features(daily_counts)
    y_count = X_count.pop('crime_count')
    
    # Train daily crime count model (small model)
    count_model = RandomForestRegressor(n_estimators=100, random_state=42)
    count_model.fit(X_count[['DayOfWeek', 'Month', 'Year', 'IsWeekend', 'IsHoliday']], y_count)
    
    # Prepare data for crime category prediction
    X_cat = prepare_features(data)
    y_cat = data['Category']
    
    # Train crime category model (small model)
    cat_model = RandomForestClassifier(n_estimators=100, random_state=42)
    cat_model.fit(X_cat[['DayOfWeek', 'Month', 'Year', 'IsWeekend', 'IsHoliday']], y_cat)
    
    # Train district prediction model
    X_district = prepare_features(data)
    y_district = data['PdDistrict']
    district_model = RandomForestClassifier(n_estimators=100,  random_state=42)
    district_model.fit(X_district[['DayOfWeek', 'Month', 'Year', 'IsWeekend', 'IsHoliday']], y_district)
    
    # SIMPLIFIED APPROACH FOR LOCATION PREDICTION
    # Instead of one-hot encoding everything, use categorical encoding (integers)
    
    # Prepare data for location prediction
    X_loc = data.copy()
    
    # Use category codes instead of one-hot encoding
    X_loc['Category_Code'] = X_loc['Category'].astype('category').cat.codes
    X_loc['District_Code'] = X_loc['PdDistrict'].astype('category').cat.codes
    
    # Add temporal features to location prediction
    X_loc['DayOfWeek'] = X_loc['Dates'].dt.dayofweek
    X_loc['Hour'] = X_loc['Dates'].dt.hour
    X_loc['Month'] = X_loc['Dates'].dt.month
    X_loc['Year'] = X_loc['Dates'].dt.year
    X_loc['IsWeekend'] = X_loc['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Simple feature set for location prediction
    simple_loc_features = ['Category_Code', 'District_Code', 'DayOfWeek', 'Hour', 'Month', 'Year', 'IsWeekend']
    
    y_lat = X_loc['Latitude (Y)']
    y_lon = X_loc['Longitude (X)']
    
    # Train location models with reduced complexity
    lat_model = RandomForestRegressor(
        n_estimators=50,      
        min_samples_leaf=5,
        max_depth=15,
        random_state=42
    )
    
    long_model = RandomForestRegressor(
        n_estimators=50,
        min_samples_leaf=5,    
        max_depth=15,
        random_state=42
    )
    
    # Train with significantly reduced feature set
    lat_model.fit(X_loc[simple_loc_features], y_lat)
    long_model.fit(X_loc[simple_loc_features], y_lon)
    
    hour_model = RandomForestRegressor(
        n_estimators=50,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42
    )
    
    X_hour = X_loc[['Category_Code', 'District_Code', 'DayOfWeek', 'Month', 'Year', 'IsWeekend']]
    y_hour = X_loc['Hour']
    hour_model.fit(X_hour, y_hour)
    
    # Save models to disk
    joblib.dump(count_model, model_files['count'])
    joblib.dump(cat_model, model_files['category'])
    joblib.dump(district_model, model_files['district'])
    joblib.dump(lat_model, model_files['lat'])
    joblib.dump(long_model, model_files['long'])
    joblib.dump(hour_model, model_files['hour'])
    
    # Also store category and district mappings
    category_mapping = dict(enumerate(X_loc['Category'].astype('category').cat.categories))
    district_mapping = dict(enumerate(X_loc['PdDistrict'].astype('category').cat.categories))
    joblib.dump(category_mapping, os.path.join(MODEL_DIR, 'category_mapping.pkl'))
    joblib.dump(district_mapping, os.path.join(MODEL_DIR, 'district_mapping.pkl'))
    
    return {
        'count': count_model,
        'category': cat_model,
        'district': district_model,
        'lat': lat_model,
        'long': long_model,
        'hour': hour_model,
        'category_mapping': category_mapping,
        'district_mapping': district_mapping
    }

try:
        models = train_crime_prediction_models(data)
except Exception as e:
        print(f"Error training crime prediction models: {e}")
        models = None

def get_predicted_crimes(date, target_hours=None, target_districts=None, data=None):
    """Generate predicted crimes using simplified models"""
    # Load or train models
    if not models :
        return pd.DataFrame()

    
    # Convert date to proper format
    if isinstance(date, str):
        date = pd.to_datetime(date).date()
    
    # Prepare features for the target date
    X_pred = prepare_features(data, date_to_predict=date)
    
    # Predict number of crimes for the day
    try:
        num_crimes = int(models['count'].predict(X_pred[['DayOfWeek', 'Month', 'Year', 'IsWeekend', 'IsHoliday']])[0])
        if target_hours:
            ratio = len(target_hours) / 24
            num_crimes = max(1, int(num_crimes * ratio))
    except Exception as e:
        print(f"Error predicting crime count: {e}")
        num_crimes = 10  # Fallback
    
    # Generate predicted crimes
    predicted_crimes = []
    
    # Handle single values
    if isinstance(target_districts, str):
        target_districts = [target_districts]
    if isinstance(target_hours, int):
        target_hours = [target_hours]
    
    # Get mappings of category codes to names
    try:
        category_mapping = joblib.load(os.path.join(MODEL_DIR, 'category_mapping.pkl'))
        district_mapping = joblib.load(os.path.join(MODEL_DIR, 'district_mapping.pkl'))
    except:
        # If mappings aren't available, create backup mappings
        category_mapping = {i: cat for i, cat in enumerate(data['Category'].unique())}
        district_mapping = {i: dist for i, dist in enumerate(data['PdDistrict'].unique())}
    
    # Reverse mappings for lookup
    cat_code_map = {v: k for k, v in category_mapping.items()}
    dist_code_map = {v: k for k, v in district_mapping.items()}
    
    # Format date
    date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
    
    # Generate synthetic crime data
    for i in range(num_crimes):
        # Predict district
        if target_districts:
            district = np.random.choice(target_districts)
            district_code = dist_code_map.get(district, 0)
        else:
            try:
                district_probs = models['district'].predict_proba(X_pred[['DayOfWeek', 'Month', 'Year', 'IsWeekend', 'IsHoliday']])
                district_idx = np.random.choice(len(models['district'].classes_), p=district_probs[0])
                district = models['district'].classes_[district_idx]
                district_code = dist_code_map.get(district, 0)
            except:
                # Fallback
                district_code = 0
                district = district_mapping.get(district_code, "NORTHERN")
        
        # Predict category
        try:
            category_probs = models['category'].predict_proba(X_pred[['DayOfWeek', 'Month', 'Year', 'IsWeekend', 'IsHoliday']])
            category_idx = np.random.choice(len(models['category'].classes_), p=category_probs[0])
            category = models['category'].classes_[category_idx]
            category_code = cat_code_map.get(category, 0)
        except:
            # Fallback
            category_code = 0
            category = category_mapping.get(category_code, "LARCENY/THEFT")
        
        # Predict hour
        if target_hours:
            hour = np.random.choice(target_hours)
        else:
            try:
                X_hour = pd.DataFrame({
                    'Category_Code': [category_code],
                    'District_Code': [district_code],
                    'DayOfWeek': X_pred['DayOfWeek'],
                    'Month': X_pred['Month'],
                    'Year': X_pred['Year'],
                    'IsWeekend': X_pred['IsWeekend']
                })
                
                hour = int(models['hour'].predict(X_hour)[0])
                hour = max(0, min(23, hour))  # Ensure valid range
            except:
                hour = np.random.randint(0, 24)
        
        # Predict location
        try:
            # Create feature vector for location prediction
            X_loc_pred = pd.DataFrame({
                'Category_Code': [category_code],
                'District_Code': [district_code],
                'DayOfWeek': [X_pred['DayOfWeek'].values[0]],
                'Hour': [hour],
                'Month': [X_pred['Month'].values[0]],
                'Year': [X_pred['Year'].values[0]],
                'IsWeekend': [X_pred['IsWeekend'].values[0]]
            })
            
            lat = models['lat'].predict(X_loc_pred)[0]
            lon = models['long'].predict(X_loc_pred)[0]
            
            # Add some random noise
            lat += np.random.normal(0, 0.001)
            lon += np.random.normal(0, 0.001)
        except Exception as e:
            # Fallback to district centers
            from app.utils import district_centers
            if district in district_centers:
                lat = district_centers[district][0] + np.random.normal(0, 0.002)
                lon = district_centers[district][1] + np.random.normal(0, 0.002)
            else:
                # San Francisco center
                lat = 37.77 + np.random.normal(0, 0.01)
                lon = -122.42 + np.random.normal(0, 0.01)
        
        # Format timestamp
        minute = np.random.randint(0, 60)
        timestamp = f"{date_str} {hour:02d}:{minute:02d}:00"
        
        # Create crime entry
        crime = {
            'Category': category,
            'PdDistrict': district,
            'Latitude (Y)': lat,
            'Longitude (X)': lon,
            'Dates': pd.to_datetime(timestamp),
            'Hour': hour,
            'Date': pd.to_datetime(date),
            'Predicted': True
        }
        predicted_crimes.append(crime)
    
    # Convert to DataFrame
    df_crimes = pd.DataFrame(predicted_crimes)
    
    # Filter by hours if specified
    if target_hours and len(df_crimes) > 0:
        df_crimes = df_crimes[df_crimes['Hour'].isin(target_hours)]
    
    return df_crimes



def get_crime_risk(target_date, target_hours=None, target_districts=None):
    """
    Get crime risk assessment for a specific date, hour range, and one or more districts.
    
    Parameters:
    -----------
    target_date : str or datetime
        The date to analyze in 'YYYY-MM-DD' format or as datetime object
    target_hours : list, int, or None
        The hours to analyze (0-23), can be a list for multiple hours or None for all hours
    target_districts : str, list, or None
        The police district(s) to analyze, can be a string for single district,
        a list for multiple districts, or None for all districts
    
    Returns:
    --------
    dict
        Risk assessment information including:
        - risk_level: High, Medium, or Low
        - crime_count: Predicted/actual number of crimes
        - data_source: "historical" or "estimated"
        - map: A folium map object showing the crimes/risk
    """
    # Handle target_hours
    if isinstance(target_hours, int):
        target_hours = [target_hours]
        hour_label = f"at {target_hours[0]}:00"
    elif target_hours and len(target_hours) > 0:
        if len(target_hours) == 24:
            hour_label = "all day"
        else:
            hour_label = f"from {min(target_hours)}:00 to {max(target_hours)}:00"
    else:
        target_hours = None
        hour_label = "all day"
    
    # Handle target_districts - convert to list if it's a string or None
    if isinstance(target_districts, str):
        target_districts = [target_districts]
        district_label = f"in {target_districts[0]}"
    elif isinstance(target_districts, list) and len(target_districts) > 0:
        if len(target_districts) == 1:
            district_label = f"in {target_districts[0]}"
        elif len(target_districts) == len(district_centers):
            target_districts = None  # If all districts are selected, treat as None
            district_label = "across all districts"
        else:
            district_label = f"in {len(target_districts)} districts"
    else:
        target_districts = None
        district_label = "across all districts"
        
    # Convert string date to datetime if needed
    if isinstance(target_date, str):
        try:
            target_date = pd.to_datetime(target_date).date()
        except:
            print(f"Invalid date format: {target_date}. Please use YYYY-MM-DD format.")
            return None
    
    # Create a copy of the data we can filter
    filtered_data = data.copy()
    
    # Filter by district if specified
    if target_districts:
        filtered_data = filtered_data[filtered_data['PdDistrict'].isin(target_districts)]
        if len(filtered_data) == 0:
            print(f"No data found for the specified districts: {target_districts}")
            return None
    
    # Filter by hours if specified
    if target_hours:
        filtered_data = filtered_data[filtered_data['Hour'].isin(target_hours)]
        if len(filtered_data) == 0:
            print(f"No data found for the specified hours.")
            return None
    
    # Check if we have data for the exact date
    date_data = filtered_data[filtered_data['Date'] == pd.Timestamp(target_date)]

    # Set up result dictionary
    result = {
        'target_date': target_date,
        'target_hours': target_hours,
        'hour_label': hour_label,
        'target_districts': target_districts,
        'district_label': district_label,
        'crime_count': 0,
        'risk_level': "Low",
        'data_source': "historical"
    }
    
    # If we have exact data for this date
    if len(date_data) > 0:
        # We have historical data for this date
        result['crime_count'] = len(date_data)
        
        # Create a map centered on San Francisco
        m = folium.Map(location=[data['Latitude (Y)'].mean(), data['Longitude (X)'].mean()], zoom_start=12)
        # Create marker cluster for crimes by category
        from folium.plugins import MarkerCluster
        
        # Group crimes by category
        categories = date_data['Category'].unique()
        category_clusters = {}
        
        # Create an "All Categories" cluster that's shown by default
        all_categories_cluster = MarkerCluster(name="All Categories (Combined)")
        
        # Calculate category counts for display
        category_counts = date_data['Category'].value_counts().to_dict()
        
        # First, add all markers to the "All Categories" cluster with colored icons
        severity_colors = {1: 'blue', 2: 'green', 3: 'orange', 4: 'red', 5: 'darkred'}
        for _, row in date_data.iterrows():
            category = row['Category']
            severity = categorize_severity(category)
            color = severity_colors.get(severity, 'blue')
            
            popup_text = f"""
            <b>Category:</b> {row['Category']}<br>
            <b>District:</b> {row['PdDistrict']}<br>
            <b>Time:</b> {row['Dates'].strftime('%Y-%m-%d %H:%M')}
            """
            folium.Marker(
                location=[row['Latitude (Y)'], row['Longitude (X)']],
                popup=popup_text,
                icon=folium.Icon(color=color, icon="info-sign")
            ).add_to(all_categories_cluster)
        
        all_categories_cluster.add_to(m)
        
        # Then create individual category clusters
        for category in categories:
            category_data = date_data[date_data['Category'] == category]
            count = len(category_data)
            severity = categorize_severity(category)
            color = severity_colors.get(severity, 'blue')
            
            # Name includes count for better visibility
            category_cluster = MarkerCluster(name=f"{category} ({count})")
            
            for _, row in category_data.iterrows():
                popup_text = f"""
                <b>Category:</b> {row['Category']}<br>
                <b>District:</b> {row['PdDistrict']}<br>
                <b>Time:</b> {row['Dates'].strftime('%Y-%m-%d %H:%M')}
                """
                folium.Marker(
                    location=[row['Latitude (Y)'], row['Longitude (X)']],
                    popup=popup_text,
                    icon=folium.Icon(color=color, icon="info-sign")
                ).add_to(category_cluster)
            
            category_cluster.add_to(m)
            category_clusters[category] = category_cluster
        
        # Add heatmap layer
        from folium.plugins import HeatMap
        
        HeatMap(
            date_data[['Latitude (Y)', 'Longitude (X)']].values.tolist(),
            radius=15,
            name="Heat Map",
            show=False
        ).add_to(m)
        
        # Add district boundaries for each selected district
        if target_districts:
            # Create a district boundaries layer group
            district_layer = folium.FeatureGroup(name="District Boundaries")
            
            for district in target_districts:
                if district in district_centers:
                    folium.Circle(
                        location=district_centers[district],
                        radius=1500,  # 1.5km radius
                        color="blue",
                        fill=True,
                        fill_opacity=0.1,
                        popup=f"{district} District"
                    ).add_to(district_layer)
                    
                    # Add district name as label
                    folium.map.Marker(
                        district_centers[district],
                        icon=folium.DivIcon(
                            icon_size=(150,36),
                            icon_anchor=(75,18),
                            html=f'<div style="font-size: 12pt; font-weight: bold; text-align: center;">{district}</div>'
                        )
                    ).add_to(district_layer)
            
            district_layer.add_to(m)
        
        # Add category filtering control
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Add filter guide
        filter_html = """
        <div style="position: fixed; 
            top: 60px; left: 50%; transform: translateX(-50%);
            z-index:9999; background-color:white; padding: 5px;
            font-size: 13px; text-align: center;
            border:1px solid grey; border-radius: 5px;">
            Use the layers control <i class="fa fa-layers"></i> in the top right to filter by crime category
        </div>
        """
        m.get_root().html.add_child(folium.Element(filter_html))
        
        # Add a category summary control
        category_summary_html = """
        <div style="position: fixed; 
            top: 100px; right: 10px; width: 250px; 
            border:2px solid grey; z-index:9998; background-color:white;
            padding: 10px; border-radius: 5px; max-height: 300px; overflow-y: auto;
            font-size: 12px; display: none;" id="category-summary-panel">
            <div style="border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-bottom: 5px;">
                <b style="font-size: 14px;">Crime Categories</b>
                <span style="float: right; cursor: pointer;" onclick="document.getElementById('category-summary-panel').style.display='none'">×</span>
            </div>
            <table style="width:100%; border-collapse: collapse;">
                <tr style="background-color: #f2f2f2;">
                    <th style="text-align: left; padding: 3px;">Category</th>
                    <th style="text-align: right; padding: 3px;">Count</th>
                    <th style="text-align: right; padding: 3px;">Severity</th>
                </tr>
        """
        
        # Add rows for each category
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            severity = categorize_severity(category)
            severity_text = {1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Critical"}.get(severity, "Unknown")
            sev_color = {1: 'blue', 2: 'green', 3: 'orange', 4: 'red', 5: 'darkred'}.get(severity, 'blue')
            
            category_summary_html += f"""
                <tr style="border-bottom: 1px solid #eee;">
                    <td style="padding: 3px;">{category}</td>
                    <td style="text-align: right; padding: 3px;">{count}</td>
                    <td style="text-align: right; padding: 3px; color: {sev_color};">{severity_text}</td>
                </tr>
            """
        
        category_summary_html += """
            </table>
            <div style="margin-top: 8px; font-size: 11px; color: #666; text-align: right;">
                Click a category in the layer control to show/hide
            </div>
        </div>
        
        <div style="position: fixed; 
            top: 100px; right: 10px; z-index:9997; 
            background-color: white; border: 1px solid grey; 
            padding: 5px; border-radius: 3px; cursor: pointer;"
            onclick="document.getElementById('category-summary-panel').style.display='block'">
            <i class="fa fa-list"></i> Categories
        </div>
        """
        
        m.get_root().html.add_child(folium.Element(category_summary_html))
        
        # Add legend for severity colors
        legend_html = '''
        <div style="position: fixed; 
            bottom: 50px; left: 50px; width: 180px; 
            border:2px solid grey; z-index:9999; background-color:white;
            padding: 10px; border-radius: 5px;
            font-size: 14px;
            ">
        <b>Crime Severity</b><br>
        <i class="fa fa-circle" style="color:blue"></i> Very Low<br>
        <i class="fa fa-circle" style="color:green"></i> Low<br>
        <i class="fa fa-circle" style="color:orange"></i> Medium<br>
        <i class="fa fa-circle" style="color:red"></i> High<br>
        <i class="fa fa-circle" style="color:darkred"></i> Critical<br>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        result['map'] = m
        result['categories'] = list(categories)
        result['category_counts'] = category_counts
        
    else:
        # No historical data - try to predict crimes for this date
        predicted_crimes = get_predicted_crimes(
            target_date, 
            target_hours=target_hours, 
            target_districts=target_districts,
            data=data
        )
        
        if len(predicted_crimes) > 0:
            # We have predicted crimes for this date
            result['data_source'] = "predicted"
            result['crime_count'] = len(predicted_crimes)
            
            # Create a map centered on San Francisco
            m = folium.Map(location=[data['Latitude (Y)'].mean(), data['Longitude (X)'].mean()], zoom_start=12)
            
            # Add title to map
            title_parts = ["Predicted Crime Map"]
            title_parts.append(district_label)
            title_parts.append(f"on {target_date}")
            title_parts.append(hour_label)
                
            title_html = f"""
            <div style="position: fixed; 
                top: 10px; left: 50%; transform: translateX(-50%);
                z-index:9999; background-color:white; padding: 10px;
                font-size: 16px; font-weight: bold; text-align: center;
                border:2px solid grey; border-radius: 5px;">
                {' '.join(title_parts)} (AI Generated)
            </div>
            """
            m.get_root().html.add_child(folium.Element(title_html))
            
            # Add prediction notice
            notice_html = """
            <div style="position: fixed; 
                top: 60px; left: 50%; transform: translateX(-50%);
                z-index:9999; background-color:#fff3cd; padding: 5px;
                font-size: 13px; text-align: center; color: #856404;
                border:1px solid #ffeeba; border-radius: 5px;">
                ⚠️ These are AI-predicted crimes based on historical patterns, not actual crimes.
            </div>
            """
            m.get_root().html.add_child(folium.Element(notice_html))
            
            # Create marker cluster for crimes by category
            from folium.plugins import MarkerCluster
            
            # Group crimes by category
            categories = predicted_crimes['Category'].unique()
            category_clusters = {}
            
            # Create an "All Categories" cluster that's shown by default
            all_categories_cluster = MarkerCluster(name="All Predicted Crimes")
            
            # Calculate category counts for display
            category_counts = predicted_crimes['Category'].value_counts().to_dict()
            
            # First, add all markers to the "All Categories" cluster with colored icons
            severity_colors = {1: 'blue', 2: 'green', 3: 'orange', 4: 'red', 5: 'darkred'}
            for _, row in predicted_crimes.iterrows():
                category = row['Category']
                severity = categorize_severity(category)
                color = severity_colors.get(severity, 'blue')
                
                popup_text = f"""
                <b>Category:</b> {row['Category']}<br>
                <b>District:</b> {row['PdDistrict']}<br>
                <b>Time:</b> {row['Dates'].strftime('%Y-%m-%d %H:%M')}<br>
                <small><i>This is a predicted crime</i></small>
                """
                folium.Marker(
                    location=[row['Latitude (Y)'], row['Longitude (X)']],
                    popup=popup_text,
                    icon=folium.Icon(color=color, icon="info-sign")
                ).add_to(all_categories_cluster)
            
            all_categories_cluster.add_to(m)
            
            # Then create individual category clusters
            for category in categories:
                category_data = predicted_crimes[predicted_crimes['Category'] == category]
                count = len(category_data)
                severity = categorize_severity(category)
                color = severity_colors.get(severity, 'blue')
                
                # Name includes count for better visibility
                category_cluster = MarkerCluster(name=f"{category} ({count}) - Predicted")
                
                for _, row in category_data.iterrows():
                    popup_text = f"""
                    <b>Category:</b> {row['Category']}<br>
                    <b>District:</b> {row['PdDistrict']}<br>
                    <b>Time:</b> {row['Dates'].strftime('%Y-%m-%d %H:%M')}<br>
                    <small><i>This is a predicted crime</i></small>
                    """
                    folium.Marker(
                        location=[row['Latitude (Y)'], row['Longitude (X)']],
                        popup=popup_text,
                        icon=folium.Icon(color=color, icon="info-sign")
                    ).add_to(category_cluster)
                
                category_cluster.add_to(m)
                category_clusters[category] = category_cluster
            
            # Add heatmap layer
            from folium.plugins import HeatMap
            
            HeatMap(
                predicted_crimes[['Latitude (Y)', 'Longitude (X)']].values.tolist(),
                radius=15,
                name="Predicted Heat Map",
                show=False
            ).add_to(m)
            
            # Add district boundaries for each selected district
            if target_districts:
                # Create a district boundaries layer group
                district_layer = folium.FeatureGroup(name="District Boundaries")
                
                for district in target_districts:
                    if district in district_centers:
                        folium.Circle(
                            location=district_centers[district],
                            radius=1500,  # 1.5km radius
                            color="blue",
                            fill=True,
                            fill_opacity=0.1,
                            popup=f"{district} District"
                        ).add_to(district_layer)
                        
                        # Add district name as label
                        folium.map.Marker(
                            district_centers[district],
                            icon=folium.DivIcon(
                                icon_size=(150,36),
                                icon_anchor=(75,18),
                                html=f'<div style="font-size: 12pt; font-weight: bold; text-align: center;">{district}</div>'
                            )
                        ).add_to(district_layer)
                
                district_layer.add_to(m)
            
            # Add category filtering control
            folium.LayerControl(collapsed=False).add_to(m)
            
            # Add filter guide
            filter_html = """
            <div style="position: fixed; 
                top: 100px; left: 50%; transform: translateX(-50%);
                z-index:9999; background-color:white; padding: 5px;
                font-size: 13px; text-align: center;
                border:1px solid grey; border-radius: 5px;">
                Use the layers control <i class="fa fa-layers"></i> in the top right to filter by crime category
            </div>
            """
            m.get_root().html.add_child(folium.Element(filter_html))
            
            # Add a category summary control
                    # Add a category summary control
            category_summary_html = """
            <div style="position: fixed; 
                top: 100px; right: 10px; width: 250px; 
                border:2px solid grey; z-index:9998; background-color:white;
                padding: 10px; border-radius: 5px; max-height: 300px; overflow-y: auto;
                font-size: 12px; display: none;" id="category-summary-panel">
                <div style="border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-bottom: 5px;">
                    <b style="font-size: 14px;">Crime Categories</b>
                    <span style="float: right; cursor: pointer;" onclick="document.getElementById('category-summary-panel').style.display='none'">×</span>
                </div>
                <table style="width:100%; border-collapse: collapse;">
                    <tr style="background-color: #f2f2f2;">
                        <th style="text-align: left; padding: 3px;">Category</th>
                        <th style="text-align: right; padding: 3px;">Count</th>
                        <th style="text-align: right; padding: 3px;">Severity</th>
                    </tr>
            """
            
            # Add rows for each category
            for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                severity = categorize_severity(category)
                severity_text = {1: "Very Low", 2: "Low", 3: "Medium", 4: "High", 5: "Critical"}.get(severity, "Unknown")
                sev_color = {1: 'blue', 2: 'green', 3: 'orange', 4: 'red', 5: 'darkred'}.get(severity, 'blue')
                
                category_summary_html += f"""
                    <tr style="border-bottom: 1px solid #eee;">
                        <td style="padding: 3px;">{category}</td>
                        <td style="text-align: right; padding: 3px;">{count}</td>
                        <td style="text-align: right; padding: 3px; color: {sev_color};">{severity_text}</td>
                    </tr>
                """
            
            category_summary_html += """
                </table>
                <div style="margin-top: 8px; font-size: 11px; color: #666; text-align: right;">
                    Click a category in the layer control to show/hide
                </div>
            </div>
            
            <div style="position: fixed; 
                top: 100px; right: 10px; z-index:9997; 
                background-color: white; border: 1px solid grey; 
                padding: 5px; border-radius: 3px; cursor: pointer;"
                onclick="document.getElementById('category-summary-panel').style.display='block'">
                <i class="fa fa-list"></i> Categories
            </div>
            """
            
            m.get_root().html.add_child(folium.Element(category_summary_html))
            
            # Add legend for severity colors
            legend_html = '''
            <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 180px; 
                border:2px solid grey; z-index:9999; background-color:white;
                padding: 10px; border-radius: 5px;
                font-size: 14px;
                ">
            <b>Crime Severity</b><br>
            <i class="fa fa-circle" style="color:blue"></i> Very Low<br>
            <i class="fa fa-circle" style="color:green"></i> Low<br>
            <i class="fa fa-circle" style="color:orange"></i> Medium<br>
            <i class="fa fa-circle" style="color:red"></i> High<br>
            <i class="fa fa-circle" style="color:darkred"></i> Critical<br>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
            result['map'] = m
            result['categories'] = list(categories)
            result['category_counts'] = category_counts
    return result
