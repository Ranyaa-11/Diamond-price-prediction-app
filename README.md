#  Diamond Price Prediction App

An interactive web application built with Streamlit that predicts diamond prices using machine learning. The app uses a Random Forest Regressor trained on the Diamonds dataset to estimate diamond prices based on their physical and categorical attributes.

![Diamond Price Predictor](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 🔹 Features

### User Input Form
- **Physical Attributes**: Carat weight, depth, table, dimensions (x, y, z)
- **Quality Attributes**: 
  - Cut quality (Ideal, Premium, Very Good, Good, Fair)
  - Color grade (D–J scale, where D is colorless)
  - Clarity grade (IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1)

### Machine Learning Model
- **Random Forest Regressor** with optimized hyperparameters
- **High Performance**: ~88% R² score on test data
- **Proper Categorical Encoding**: Handles cut, color, and clarity with label encoding
- **Feature Importance Analysis**: Shows which attributes most affect price

### Interactive Dashboard
- **Real-time Predictions**: Instant price estimates with detailed specifications
- **Data Visualization**: Interactive charts showing price distributions and feature correlations
- **Model Performance Metrics**: Display of accuracy and error metrics
- **Responsive Design**: Clean, modern UI with intuitive controls

## 🔹 Tech Stack

- **Python 3.8+**
- **Streamlit** - Interactive web app framework
- **Scikit-learn** - Machine learning library
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Plotly** - Interactive visualizations
- **Joblib** - Model serialization

## 🔹 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### 1. Clone the Repository
```bash
git clone <repository-url>
cd diamond-price-app
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
Before running the app, you need to train the machine learning model:

```bash
python clean_train.py
```

This will:
- Load and preprocess the diamonds dataset
- Train a Random Forest Regressor
- Save the model and encoders to files
- Display model performance metrics

### 4. Run the Streamlit App
```bash
streamlit run clean_app.py
```

The app will open in your default web browser at `http://localhost:8501`

## 🔹 Usage

1. **Open the App**: Launch the Streamlit app using the command above
2. **Input Diamond Details**: Use the sidebar to specify:
   - Physical measurements (carat, depth, table, dimensions)
   - Quality attributes (cut, color, clarity)
3. **Get Prediction**: Click "Predict Diamond Price" to see the estimated price
4. **Explore Data**: Use the visualization tabs to understand price patterns and feature relationships

## 🔹 Dataset

The model is trained on the **Diamonds Dataset** containing:
- **53,940 diamond samples**
- **10 features**: carat, cut, color, clarity, depth, table, price, x, y, z
- **Price range**: $326 - $18,823

### Feature Descriptions
- **carat**: Weight of the diamond (0.2-5.01)
- **cut**: Quality of the cut (Fair, Good, Very Good, Premium, Ideal)
- **color**: Diamond color (D-J, where D is best)
- **clarity**: Diamond clarity (I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF)
- **depth**: Total depth percentage (43.0-79.0)
- **table**: Width of top of diamond relative to widest point (43.0-95.0)
- **price**: Price in US dollars (326-18823)
- **x, y, z**: Length, width, and height in mm


## 🔹 Project Structure

```
diamond-price-app/
├── clean_app.py                 # Streamlit web application
├── clean_train.py         # Model training
├── diamonds.csv          # Dataset
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── diamond_price_model.pkl    # Trained model 
├── label_encoders.pkl         # Categorical encoders 
└── feature_columns.pkl        # Feature column names 
```



## 🔹 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 🔹 License

This project is open source and available under the [MIT License](LICENSE).





