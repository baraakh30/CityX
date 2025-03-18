import folium
from flask import  request, render_template, jsonify
from app import app
from app.utils import extract_pdf_data, predict_category, categorize_severity,extracted_reports,get_crime_risk


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

@app.route('/crime-search-map')
def crime_search_map():
    # Get parameters
    date_input = request.args.get('date')
    selected_districts = request.args.get('districts', '')
    selected_hours = request.args.get('hours', '')
    
    # Convert parameters
    if selected_districts:
        selected_districts = selected_districts.split(',')
    else:
        selected_districts = None
    
    if selected_hours:
        selected_hours = [int(h) for h in selected_hours.split(',')]
    else:
        selected_hours = None
    
    # Get crime risk assessment
    result = get_crime_risk(date_input, selected_hours, selected_districts)
    
    if result and 'map' in result and result['map'] is not None:
        # Return the map's HTML representation directly
        folium_map = result['map']
        
        # Add map metadata as hidden HTML elements
        metadata_html = f"""
        <div id="map-metadata" style="display:none;">
            <span id="data-source">{result['data_source']}</span>
            <span id="risk-level">{result['risk_level']}</span>
            <span id="crime-count">{result['crime_count']}</span>
            <span id="target-date">{result['target_date']}</span>
            <span id="district-info">{result['district_label']}</span>
            <span id="hour-info">{result['hour_label']}</span>
        </div>
        """
        
        # Add the metadata to the map's root HTML
        folium_map.get_root().html.add_child(folium.Element(metadata_html))
        
        # Get the HTML representation directly
        map_html = folium_map._repr_html_()
        
        # Return the HTML as a response
        return map_html
    else:
        return """
        <div style="display: flex; justify-content: center; align-items: center; height: 100vh; text-align: center; font-family: Arial, sans-serif;">
            <div>
                <h2 style="color: #4361ee;">No Crime Data Available</h2>
                <p style="color: #6c757d; margin-top: 10px;">
                    We couldn't find any crime data for the selected criteria.<br>
                    Please try different date, districts, or time periods.
                </p>
                <button onclick="window.history.back()" style="margin-top: 20px; background-color: #4361ee; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">
                    Go Back
                </button>
            </div>
        </div>
        """