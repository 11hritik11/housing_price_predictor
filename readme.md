# California Housing Price Predictor 🏡

## Overview
The California Housing Price Predictor is a machine learning project designed to predict the median house value in California based on various features such as location, population, and housing characteristics. The project uses a linear regression model trained on the California housing dataset.

## Features
- **Streamlit Web App**: A user-friendly interface to input housing details and get price predictions.
- **Data Preprocessing**: Handles missing values, log-transforms skewed features, and encodes categorical variables.
- **Feature Engineering**: Adds new features like bedroom-to-room ratio and household rooms.
- **Model Training**: Uses a linear regression model with standardized features.
- **Model Persistence**: Saves the trained model and scaler for future use.

## Project Structure
```
├── housing_price_predictor
│   ├── host.py                # Streamlit app for predictions
│   ├── hosuingmodel.ipynb     # Jupyter notebook for model training
│   ├── housing.csv            # Dataset
│   ├── housing_model.pkl      # Saved model
│   ├── housing_scaler.pkl     # Saved scaler
│   └── readme.md              # Project documentation
```

## Requirements
- Python 3.9+
- Required Python libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - streamlit

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/11hritik11/housing_price_predictor.git
   ```
2. Navigate to the project directory:
   ```bash
   cd housing_price_predictor
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Running the Streamlit App
1. Start the Streamlit server:
   ```bash
   streamlit run host.py
   ```
2. Open the provided URL in your browser (e.g., `http://localhost:8501`).
3. Enter the housing details in the web app to get the predicted price.

### Training the Model
1. Open the `hosuingmodel.ipynb` notebook in Jupyter.
2. Run all cells to preprocess the data, train the model, and save the artifacts (`housing_model.pkl` and `housing_scaler.pkl`).

## Dataset
The dataset used in this project is the California housing dataset, which includes features such as:
- Longitude
- Latitude
- Housing Median Age
- Total Rooms
- Total Bedrooms
- Population
- Households
- Median Income
- Ocean Proximity

## Future Improvements
- Add support for more advanced machine learning models.
- Improve the user interface of the Streamlit app.
- Deploy the app to a cloud platform for public access.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Author
- **Hritik**

Feel free to contribute to this project by submitting issues or pull requests!
