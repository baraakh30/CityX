import pandas as pd
import folium
import pandas as pd


print("Loading and preparing data...")

# Load dataset
data = pd.read_csv("Competition_Dataset.csv", parse_dates=["Dates"])
data = data[["Dates", "Category", "PdDistrict", "Latitude (Y)", "Longitude (X)"]]
# Fixing the coordinate swap issue
temp = data['Latitude (Y)']
data['Latitude (Y)'] = data['Longitude (X)']
data['Longitude (X)'] = temp
data["Hour"] = data["Dates"].dt.hour
data["Day"] = data["Dates"].dt.day
data["Month"] = data["Dates"].dt.month
data["Year"] = data["Dates"].dt.year
data["Weekday"] = data["Dates"].dt.weekday
data["Date"] = data["Dates"].dt.date
data["Date"] = pd.to_datetime(data["Date"])

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
    
def get_crime_risk(target_date, target_hours=None, target_district=None):
    """
    Get crime risk assessment for a specific date, hour range, and district.
    
    Parameters:
    -----------
    target_date : str or datetime
        The date to analyze in 'YYYY-MM-DD' format or as datetime object
    target_hours : list, int, or None
        The hours to analyze (0-23), can be a list for multiple hours or None for all hours
    target_district : str or None
        The police district to analyze, or None for all districts
    
    Returns:
    --------
    dict
        Risk assessment information including:
        - risk_level: High, Medium, or Low
        - crime_count: Predicted/actual number of crimes
        - data_source: "historical" or "estimated"
        - map: A folium map object showing the crimes/risk
    """
    # Convert target_hours to a list if it's a single integer
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
    
    print(f"Analyzing crime risk for {target_date}, Hours: {hour_label}, District: {target_district}")
    
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
    if target_district:
        filtered_data = filtered_data[filtered_data['PdDistrict'] == target_district]
        if len(filtered_data) == 0:
            print(f"No data found for district: {target_district}")
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
        'target_district': target_district,
        'crime_count': 0,
        'risk_level': "Low",
        'data_source': "historical"
    }
    
    # If we have exact data for this date
    if len(date_data) > 0:
        # We have historical data for this date
        result['crime_count'] = len(date_data)
        
        # Create a map centered on San Francisco
        m = folium.Map(location=[37.77, -122.42], zoom_start=12)
        
        # Add title to map
        title_parts = ["Crime Map"]
        if target_district:
            title_parts.append(f"for {target_district}")
        title_parts.append(f"on {target_date}")
        title_parts.append(hour_label)
            
        title_html = f"""
        <div style="position: fixed; 
            top: 10px; left: 50%; transform: translateX(-50%);
            z-index:9999; background-color:white; padding: 10px;
            font-size: 16px; font-weight: bold; text-align: center;
            border:2px solid grey; border-radius: 5px;">
            {' '.join(title_parts)}
        </div>
        """
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Create marker cluster for crimes by category
        from folium.plugins import MarkerCluster
        
        # Group crimes by category
        categories = date_data['Category'].unique()
        category_clusters = {}
        
        for category in categories:
            category_data = date_data[date_data['Category'] == category]
            category_cluster = MarkerCluster(name=category)
            
            for _, row in category_data.iterrows():
                popup_text = f"""
                <b>Category:</b> {row['Category']}<br>
                <b>District:</b> {row['PdDistrict']}<br>
                <b>Time:</b> {row['Dates'].strftime('%Y-%m-%d %H:%M')}
                """
                folium.Marker(
                    location=[row['Latitude (Y)'], row['Longitude (X)']],
                    popup=popup_text,
                    icon=folium.Icon(icon="info-sign")
                ).add_to(category_cluster)
            
            category_cluster.add_to(m)
            category_clusters[category] = category_cluster
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Also add heatmap of all crimes
        from folium.plugins import HeatMap
        
        HeatMap(
            date_data[['Latitude (Y)', 'Longitude (X)']].values.tolist(),
            radius=15,
            name="Heat Map",
            show=False
        ).add_to(m)
        
        # Add district boundaries if available and single district is selected
        if target_district:
            # Create a circle around district center if available
            if target_district in district_centers:
                folium.Circle(
                    location=district_centers[target_district],
                    radius=1500,  # 1.5km radius
                    color="blue",
                    fill=True,
                    fill_opacity=0.1,
                    popup=f"{target_district} District"
                ).add_to(m)
                
        result['map'] = m
        
    else:
        # We don't have data for this exact date, use average from nearby dates
        result['data_source'] = "estimated"
        
        # Find dates within a week before and after
        target_date = pd.Timestamp(target_date)

        # Now perform the filtering
        nearby_dates = filtered_data[
            (filtered_data['Date'] >= target_date - pd.Timedelta(weeks=1)) & 
            (filtered_data['Date'] <= target_date + pd.Timedelta(weeks=1))
        ]
        if len(nearby_dates) > 0:
            # Group by date and get count per day
            daily_counts = nearby_dates.groupby('Date').size()
            result['crime_count'] = round(daily_counts.mean(), 1)  # Average crimes per day
            
            # Create a simple map with the estimated risk
            m = folium.Map(location=[37.77, -122.42], zoom_start=12)
            
            # Add title
            title_parts = ["Estimated Crime Risk Map"]
            if target_district:
                title_parts.append(f"for {target_district}")
            title_parts.append(f"on {target_date}")
            title_parts.append(hour_label)
                
            title_html = f"""
            <div style="position: fixed; 
                top: 10px; left: 50%; transform: translateX(-50%);
                z-index:9999; background-color:white; padding: 10px;
                font-size: 16px; font-weight: bold; text-align: center;
                border:2px solid grey; border-radius: 5px;">
                {' '.join(title_parts)} (Estimated)
            </div>
            """
            m.get_root().html.add_child(folium.Element(title_html))
            
            # If specific district, draw it
            if target_district and target_district in district_centers:
                # Determine color based on estimated crime count
                if result['crime_count'] > 20:
                    color = 'red'
                    result['risk_level'] = 'High'
                elif result['crime_count'] > 10:
                    color = 'orange'
                    result['risk_level'] = 'Medium'
                else:
                    color = 'green'
                    result['risk_level'] = 'Low'
                    
                folium.Circle(
                    location=district_centers[target_district],
                    radius=1500,  # 1.5km radius
                    color=color,
                    fill=True,
                    fill_opacity=0.4,
                    popup=f"<b>{target_district}</b><br>Estimated Risk: {result['risk_level']}<br>Est. Crimes: {result['crime_count']}"
                ).add_to(m)
                
                # Add district name as label
                folium.map.Marker(
                    district_centers[target_district],
                    icon=folium.DivIcon(
                        icon_size=(150,36),
                        icon_anchor=(75,18),
                        html=f'<div style="font-size: 14pt; font-weight: bold; text-align: center;">{target_district}</div>'
                    )
                ).add_to(m)
            else:
                # Show all districts with their risk levels
                for district, coords in district_centers.items():
                    # Filter data for this district
                    district_data = nearby_dates[nearby_dates['PdDistrict'] == district]
                    if len(district_data) == 0:
                        continue
                    
                    # Calculate average daily crime count
                    district_daily = district_data.groupby('Date').size()
                    avg_crimes = round(district_daily.mean(), 1)
                    
                    # Determine color based on crime count
                    if avg_crimes > 20:
                        color = 'red'
                        risk = 'High'
                    elif avg_crimes > 10:
                        color = 'orange'
                        risk = 'Medium'
                    else:
                        color = 'green'
                        risk = 'Low'
                        
                    folium.Circle(
                        location=coords,
                        radius=1000,  # 1km radius
                        color=color,
                        fill=True,
                        fill_opacity=0.4,
                        popup=f"<b>{district}</b><br>Risk Level: {risk}<br>Est. Crimes: {avg_crimes}"
                    ).add_to(m)
                    
                    # Add district name as label
                    folium.map.Marker(
                        coords,
                        icon=folium.DivIcon(
                            icon_size=(150,36),
                            icon_anchor=(75,18),
                            html=f'<div style="font-size: 12pt; font-weight: bold; text-align: center;">{district}</div>'
                        )
                    ).add_to(m)
            
            # Add legend
            legend_html = '''
            <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 120px; 
                border:2px solid grey; z-index:9999; background-color:white;
                padding: 10px; border-radius: 5px;
                font-size: 14px;
                ">
            <b>Risk Levels</b><br>
            <i class="fa fa-circle" style="color:red"></i> High Risk<br>
            <i class="fa fa-circle" style="color:orange"></i> Medium Risk<br>
            <i class="fa fa-circle" style="color:green"></i> Low Risk<br>
            </div>
            '''
            m.get_root().html.add_child(folium.Element(legend_html))
            
            result['map'] = m
        else:
            print(f"No data found within two weeks of {target_date}")
            # Create a simple map showing no data
            m = folium.Map(location=[37.77, -122.42], zoom_start=12)
            folium.Marker(
                location=[37.77, -122.42],
                popup="No crime data available for this period",
                icon=folium.Icon(color="gray")
            ).add_to(m)
            result['map'] = m
    
    # Set risk level based on crime count
    if result['crime_count'] > 20:
        result['risk_level'] = "High"
    elif result['crime_count'] > 10:
        result['risk_level'] = "Medium"
    else:
        result['risk_level'] = "Low"
    
    # Print summary
    print(f"Risk Level: {result['risk_level']}")
    print(f"Crime Count: {result['crime_count']}")
    print(f"Data Source: {result['data_source']}")
    
    return result

def interactive_crime_search():
    """Interactive command-line interface for crime risk search"""
    print("\n=== Crime Risk Search ===")
    
    # Get date
    print(f"\nAvailable date range: {data['Date'].min().date()} to {data['Date'].max().date()}")
    date_input = input("Enter date to analyze (YYYY-MM-DD): ")
    
    # Get district (optional)
    print("\nAvailable Districts:")
    for i, district in enumerate(sorted(district_centers.keys()), 1):
        print(f"{i}. {district}")
    print("0. All Districts")
    
    district_choice = input("Select district number (or 0 for all): ")
    selected_district = None
    
    if district_choice.isdigit():
        choice = int(district_choice)
        if 1 <= choice <= len(district_centers):
            selected_district = sorted(district_centers.keys())[choice-1]
    
    # Get hour range (optional)
    print("\nSelect time period:")
    print("1. Morning (6AM-11AM)")
    print("2. Afternoon (12PM-5PM)")
    print("3. Evening (6PM-11PM)")
    print("4. Night (12AM-5AM)")
    print("5. Specific hour")
    print("6. Specific hour range")
    print("0. All hours")
    
    hour_choice = input("Select option: ")
    selected_hours = None
    
    if hour_choice.isdigit():
        choice = int(hour_choice)
        if choice == 1:
            selected_hours = list(range(6, 12))  # Morning
        elif choice == 2:
            selected_hours = list(range(12, 18))  # Afternoon
        elif choice == 3:
            selected_hours = list(range(18, 24))  # Evening
        elif choice == 4:
            selected_hours = list(range(0, 6))  # Night
        elif choice == 5:
            # Specific hour
            hour_input = input("Enter hour (0-23): ")
            if hour_input.isdigit() and 0 <= int(hour_input) <= 23:
                selected_hours = [int(hour_input)]
        elif choice == 6:
            # Specific hour range
            start_hour = input("Enter start hour (0-23): ")
            end_hour = input("Enter end hour (0-23): ")
            
            if start_hour.isdigit() and end_hour.isdigit():
                start = int(start_hour)
                end = int(end_hour)
                
                if 0 <= start <= 23 and 0 <= end <= 23:
                    if start <= end:
                        selected_hours = list(range(start, end + 1))
                    else:
                        # Handle overnight ranges (e.g. 22-4)
                        selected_hours = list(range(start, 24)) + list(range(0, end + 1))
    
    # Get risk assessment
    result = get_crime_risk(date_input, selected_hours, selected_district)
    
    if result and 'map' in result:
        # Save map with descriptive filename
        map_filename = "crime_search_result.html"
        components = []
        
        if selected_district:
            components.append(selected_district.lower())
        else:
            components.append("all_districts")
            
        if isinstance(result['target_date'], pd.Timestamp):
            components.append(result['target_date'].strftime('%Y%m%d'))
        else:
            components.append(str(result['target_date']).replace("-", ""))
            
        if result.get('hour_label') and result['hour_label'] != "all day":
            # Create a simplified hour label for filename
            if "from" in result['hour_label']:
                components.append("hours" + result['hour_label'].replace("from ", "").replace(":00 to ", "-").replace(":00", "h"))
            else:
                components.append(result['hour_label'].replace("at ", "").replace(":00", "h"))
        
        if components:
            map_filename = f"crime_search_{'_'.join(components)}.html"
        
        result['map'].save(map_filename)
        print(f"\nMap saved as {map_filename}")
        print(f"Open this file in a web browser to view the crime map.")

# Add this line at the end of your script to run the interactive search
if __name__ == "__main__":
    interactive_crime_search()