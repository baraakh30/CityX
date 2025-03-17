# README

## CityX Crime Watch Application

This application provides a comprehensive platform for analyzing crime data in CityX. It includes features for crime prediction, PDF report processing, and interactive crime maps.

## Features

- **Crime Prediction**: Predict the category and severity of a crime based on its description.
- **PDF Report Processing**: Upload and process police reports in PDF format to extract key information.
- **Interactive Crime Maps**: Visualize crime data on interactive maps, categorized by crime type and severity.
- **Data Analysis**: Explore crime data through various visualizations, including temporal patterns, geographic distribution, and resolution types.

## Prerequisites

Before running the application, ensure you have the following dependencies installed:

- Python 3.8 or higher
- Required Python packages (listed in `requirements.txt`)

## Installation

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

4. **Download Pretrained Model**:
   Ensure you have the pretrained model inside `app/static/`. You can generate this by running `model_training.py`:
   ```bash
   python model_training.py
   ```

## Running the Application

1. **Run the Flask Application**:
   ```bash
   python wsgi.py
   ```

   This will start the Flask development server, and the application will be accessible at `http://localhost:5000`.

2. **Access the Application**:
   Open your web browser and navigate to `http://localhost:5000` to use the application.

## File Structure

- `app/`: Contains the main application code.
  - `static/`: Static files including CSS, JavaScript, and pretrained models.
  - `templates/`: HTML templates for the web interface.
- `wsgi.py`: Entry point for running the Flask application.
- `model_training.py`: Script to train and save the crime prediction model.
- `requirements.txt`: List of Python dependencies.

## Dependencies

The application requires the following Python packages:

- `pandas>=1.3.5`
- `numpy>=1.21.0`
- `scikit-learn>=1.0.2`
- `transformers>=4.18.0`
- `folium>=0.12.1`
- `matplotlib>=3.5.1`
- `seaborn>=0.11.2`
- `PyPDF2>=2.10.0`
- `joblib>=1.1.0`
- `flask>=2.0.1`
- `gunicorn>=20.1.0`
- `Werkzeug>=2.0.1`
- `tqdm>=4.62.3`
- `regex>=2022.1.18`
- `requests>=2.27.1`
- `pillow>=9.0.1`
- `ipykernel`

## Additional Notes

- Ensure that the pretrained model is placed in the `app/static/` directory before running the application.
- The application uses Flask for the web server and Folium for interactive maps.
- The `model_training.py` script should be run to generate the necessary model files if they are not already present.

## Troubleshooting

- **Missing Model Files**: If you encounter errors related to missing model files, ensure that `model_training.py` has been executed and the model files are correctly placed in `app/static/`.
- **Dependency Issues**: If you face issues with dependencies, ensure that all packages listed in `requirements.txt` are installed correctly.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

Enjoy using the CityX Crime Watch application! For any issues or contributions, please open an issue or submit a pull request on GitHub.