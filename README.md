README
CityX Crime Watch Application
This application provides a comprehensive platform for analyzing crime data in CityX. It includes features for crime prediction, PDF report processing, and interactive crime maps.

Features
Crime Prediction: Predict the category and severity of a crime based on its description.

PDF Report Processing: Upload and process police reports in PDF format to extract key information.

Interactive Crime Maps: Visualize crime data on interactive maps, categorized by crime type and severity.

Data Analysis: Explore crime data through various visualizations, including temporal patterns, geographic distribution, and resolution types.

Prerequisites
Before running the application, ensure you have the following dependencies installed:

Python 3.8 or higher

Required Python packages (listed in requirements.txt)

Docker (optional, for containerized deployment)

Installation
Option 1: Local Setup
Clone the Repository:

bash
Copy
git clone https://github.com/yourusername/cityx-crime-watch.git
cd cityx-crime-watch
Create a Virtual Environment:

bash
Copy
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install Dependencies:

bash
Copy
pip install -r requirements.txt
Download Pretrained Model:
Ensure you have the pretrained model inside app/static/. You can generate this by running model_training.py:

bash
Copy
python model_training.py
Run the Flask Application:

bash
Copy
python wsgi.py
This will start the Flask development server, and the application will be accessible at http://localhost:5000.

Option 2: Docker Setup
If you prefer to run the application in a Docker container, follow these steps:

Build the Docker Image:
Navigate to the project directory and build the Docker image:

bash
Copy
docker build -t cityx-crime-watch .
Run the Docker Container:
Once the image is built, run the container:

bash
Copy
docker run -p 5000:5000 cityx-crime-watch
The application will be accessible at http://localhost:5000.

Dockerfile Explanation
The Dockerfile is used to containerize the CityX Crime Watch application. Here's a breakdown of its contents:

Dockerfile
Copy
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
Explanation:
Base Image: The image is based on python:3.11-slim, a lightweight version of Python 3.11.

Working Directory: The working directory inside the container is set to /app.

Dependencies:

The requirements.txt file is copied into the container.

torch is installed separately because it requires a specific CPU version.

All other dependencies are installed using pip install -r requirements.txt.

Application Code: The entire application code is copied into the container.

Environment Variables:

FLASK_APP=wsgi.py: Specifies the entry point for the Flask application.

FLASK_ENV=production: Sets the environment to production mode.

Expose Port: The container exposes port 5000, which is used by the Flask application.

Run Command: The application is started using gunicorn, a production-ready WSGI server.

File Structure
app/: Contains the main application code.

static/: Static files including CSS, JavaScript, and pretrained models.

templates/: HTML templates for the web interface.

wsgi.py: Entry point for running the Flask application.

model_training.py: Script to train and save the crime prediction model.

requirements.txt: List of Python dependencies.

Dockerfile: Configuration for containerizing the application.

Dependencies
The application requires the following Python packages:

pandas>=1.3.5

numpy>=1.21.0

scikit-learn>=1.0.2

transformers>=4.18.0

folium>=0.12.1

matplotlib>=3.5.1

seaborn>=0.11.2

PyPDF2>=2.10.0

joblib>=1.1.0

flask>=2.0.1

gunicorn>=20.1.0

Werkzeug>=2.0.1

tqdm>=4.62.3

regex>=2022.1.18

requests>=2.27.1

pillow>=9.0.1

ipykernel

Additional Notes
Ensure that the pretrained model is placed in the app/static/ directory before running the application.

The application uses Flask for the web server and Folium for interactive maps.

The model_training.py script should be run to generate the necessary model files if they are not already present.

Troubleshooting
Missing Model Files: If you encounter errors related to missing model files, ensure that model_training.py has been executed and the model files are correctly placed in app/static/.

Dependency Issues: If you face issues with dependencies, ensure that all packages listed in requirements.txt are installed correctly.

Docker Issues: If the Docker container fails to start, check the logs using:

bash
Copy
docker logs <container-id>
Deployed Application
The application is deployed and accessible at:
👉 CityX Crime Watch

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Enjoy using the CityX Crime Watch application! For any issues or contributions, please open an issue or submit a pull request on GitHub.