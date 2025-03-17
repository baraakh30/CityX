import glob
import re
import joblib
import pandas as pd
import folium
import PyPDF2
from flask import  request, render_template, jsonify
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app import app

print("Loading dataset...")
df = pd.read_csv('Competition_Dataset.csv')


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


df['Severity'] = df['Category'].apply(categorize_severity)

temp = df['Latitude (Y)']
df['Latitude (Y)'] = df['Longitude (X)']
df['Longitude (X)'] = temp

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

# Geo-Spatial Mapping
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


@app.route('/')
def home():
    return render_template('index.html', reports=extracted_reports)
# Add maps route to Flask app
@app.route('/maps')
def maps():
    return render_template('maps.html')

@app.route('/eda')
def eda():
    return render_template('eda.html')

@app.route('/predict', methods=['POST'])
def predict():
    description = request.form['description']
    predicted_category = predict_category(description)
    severity = categorize_severity(predicted_category)
    return jsonify({'category': predicted_category, 'severity': severity})

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    pdf_file = request.files['pdf_file']

    if pdf_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Extract content from the PDF
    report = extract_pdf_data(pdf_file)

    # Predict category and severity
    desc = report.get('Detailed Description', '')
    if desc:
        report['Category'] = predict_category(desc)
        report['Severity'] = categorize_severity(report['Category'])
    else:
        report['Category'] = 'N/A'
        report['Severity'] = 0
    # Return the report as JSON
    return jsonify({'report': report})
