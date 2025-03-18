# CityX Crime Watch Application

This application provides a comprehensive platform for analyzing crime data in CityX. It includes features for crime prediction, PDF report processing, and interactive crime maps.

## Features

- **Crime Prediction**: Predict the category and severity of a crime based on its description.
- **PDF Report Processing**: Upload and process police reports in PDF format to extract key information.
- **Interactive Crime Maps**: Visualize crime data on interactive maps, categorized by crime type and severity.
- **Data Analysis**: Explore crime data through various visualizations, including temporal patterns, geographic distribution, and resolution types.
- **Search-by-Date Crime Map**: Select a specific date, time range, and district(s) to view historical crime data on an interactive map. If no historical data is available, the system predicts and displays crimes that likely occurred on that date.
- **Crime Forecasting**: Utilizes machine learning models to predict crime patterns for dates without historical records, including crime count, categories, locations, and time of day.

## Prerequisites

Before running the application, ensure you have the following dependencies installed:
- Python 3.8 or higher
- Required Python packages (listed in requirements.txt)
- Docker (optional, for containerized deployment)

## Installation

### Option 1: Local Setup

1. **Clone the Repository**:
```bash
git clone https://github.com/yourusername/cityx-crime-watch.git
cd cityx-crime-watch
```

2. **Create a Virtual Environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

4. **Using The Pretrained Model**:
Ensure you have the pretrained model inside app/static/. You can use the already trained one (added to the repo at app/static/) or You can generate it by running model_training.py:
```bash
python model_training.py
```

5. **Run the Flask Application**:
```bash
python wsgi.py
```
This will start the Flask development server, and the application will be accessible at http://localhost:5000.

### Option 2: Docker Setup

If you prefer to run the application in a Docker container, follow these steps:
#### Option A: Build Your Own Image
1. **Build the Docker Image**:
Navigate to the project directory and build the Docker image:
```bash
docker build -t cityx-crime-watch .
```

2. **Run the Docker Container**:
Once the image is built, run the container:
```bash
docker run -p 5000:5000 cityx-crime-watch
```
The application will be accessible at http://localhost:5000.

#### Option B: Use the Pre-built Image
1. **Pull the Public Image:**:
Navigate to the project directory and build the Docker image:
```bash
docker pull baraakh/cityx:latest
```

2. **Run the Docker Container**:
Once the image is built, run the container:
```bash
docker run -p 5000:5000 cityx-crime-watch
```
The application will be accessible at http://localhost:5000.

## Dockerfile Explanation

The Dockerfile is used to containerize the CityX Crime Watch application. Here's a breakdown of its contents:

```Dockerfile
# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install torch==2.6.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV FLASK_APP=wsgi.py
ENV FLASK_ENV=production

# Expose the port the app will run on
EXPOSE 5000

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "wsgi:app"]
```

### Explanation:
- **Base Image**: The image is based on python:3.11-slim, a lightweight version of Python 3.11.
- **Working Directory**: The working directory inside the container is set to /app.
- **Dependencies**:
  - The requirements.txt file is copied into the container.
  - torch is installed separately because it requires a specific CPU version.
  - All other dependencies are installed using pip install -r requirements.txt.
- **Application Code**: The entire application code is copied into the container.
- **Environment Variables**:
  - FLASK_APP=wsgi.py: Specifies the entry point for the Flask application.
  - FLASK_ENV=production: Sets the environment to production mode.
- **Expose Port**: The container exposes port 5000, which is used by the Flask application.
- **Run Command**: The application is started using gunicorn, a production-ready WSGI server.

## File Structure

- **app/**: Contains the main application code.
  - **static/**: Static files including CSS, data analysis images, and pretrained models.
    - **models/**: Contains trained machine learning models for crime prediction.
    - **transformer_model/**: Contains a trained transformers model on crime descriptions for description based prediction
  - **routes.py**: Defines the application routes and view functions.
  - **utils.py**: Utility functions including data processing and model training.
- **templates/**: HTML templates for the web interface.
- **wsgi.py**: Entry point for running the Flask application.
- **model_training.py**: Script to train and save the crime prediction transformer model.
- **requirements.txt**: List of Python dependencies.
- **Dockerfile**: Configuration for containerizing the application.

## Crime Prediction Models

The application uses several machine learning models to predict different aspects of crimes:
1. **Description Classification Model**: Predicts the Category of crimes based on their descriptions.
2. **Crime Count Model**: Predicts the number of crimes likely to occur on a specific date.
3. **Crime Category Model**: Predicts the categories of crimes based on temporal features.
4. **District Model**: Predicts which police districts are likely to have crimes.
5. **Location Models**: Two models (latitude and longitude) predict the specific locations of crimes.
6. **Hour Model**: Predicts the likely hour when crimes occur.

These models are trained on historical data and use features such as:
- Description
- Day of week
- Month and year
- Whether the day is a weekend
- Whether the day is a holiday
- Crime category and district (for location and time predictions)

The models are saved as `.pkl` ans `.safetensors` files in the `app/static/models/` and `app/static/transformer_model/` directories and are loaded when the application starts.

## Dependencies

The application requires the following Python packages:
- folium==0.19.5
- ipykernel
- joblib==1.4.2
- matplotlib==3.10.0
- numpy==2.2.3
- pandas==2.2.3
- pillow==11.0.0
- PyPDF2==3.0.1
- regex==2024.11.6
- requests==2.32.3
- scikit-learn==1.6.1
- seaborn==0.13.2
- vtqdm==4.67.1
- transformers==4.49.0
- Werkzeug==3.1.3
- flask==3.1.0
- gunicorn==23.0.0
- torch==2.6.0+cpu

## Search-by-Date Map Feature

The application includes a powerful search-by-date map feature that allows users to:

1. **Select a specific date** to view crime data.
2. **Choose time ranges** (specific hours or all day).
3. **Filter by district(s)** to focus on particular areas.
4. **View historical data** when available, displayed as interactive markers on a map.
5. **View predicted data** when historical data is not available, using the trained machine learning models to generate realistic crime predictions.

The map includes:
- Color-coded markers based on crime severity
- Clustering of crimes for better visualization
- Layer controls to filter by crime category
- Heat map visualization option
- District boundaries


## Additional Notes

- Ensure that the pretrained model is placed in the app/static/ directory before running the application.
- The application uses Flask for the web server and Folium for interactive maps.
- The model_training.py script should be run to generate the necessary transformer model files if they are not already present.

## Troubleshooting

- **Missing Model Files**: If you encounter errors related to missing model files, ensure that model_training.py has been executed and the model files are correctly placed in app/static/.
- **Dependency Issues**: If you face issues with dependencies, ensure that all packages listed in requirements.txt are installed correctly.
- **Docker Issues**: If the Docker container fails to start, check the logs using:
```bash
docker logs <container-id>
```

## Deployed Application

The application is deployed and accessible at:
👉 [CityX Crime Watch](https://cityx.azurewebsites.net/)

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

Enjoy using the CityX Crime Watch application! For any issues or contributions, please open an issue or submit a pull request on GitHub.