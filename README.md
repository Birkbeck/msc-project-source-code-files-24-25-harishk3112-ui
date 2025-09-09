# Predictive Analysis of Household Energy Consumption

This collection contains the implementation of a household energy consumption forecasting system developed as an MSc Advanced Computing(Data Analytics) project. 

The forecasting system utilizes **Long Short-Term Memory (LSTM)** deep neural networks to predict short-term electricity use at the household level while **ARIMA** was employed as a baseline model for comparison purposes. 

A user-friendly Flask + Chart.js dashboard provides interactive forecasts, evaluation metrics and tangible recommendations for end-users.

## Dataset 

- **Source:** https://www.kaggle.com/datasets/thedevastator/240000-household-electricity-consumption-records

- **Description:** Big data file includes 240,000 records of household electricity consumption with very detailed time-stamped consumption data for time series and forecasting. 

- **Usage in project**: Resampled to **hourly averages** to match the forecasting granularity.  
- **Features engineered**: rolling statistics, calendar encodings (hour, day of week), and temporal sinusoidal features.
 
## Tools and Environment

- **Python version**: 3.11  
-  **Frameworks / Libraries**:
    - TensorFlow / Keras – for the **LSTM forecasting model**  
    -  Statsmodels – for **ARIMA baseline**  
    -  Scikit-learn – preprocessing & evaluation metrics  
    - Flask – RESTful API backend  
    -  Chart.js + Bootstrap – interactive frontend dashboard 

# Repository Structure

- `model.py` – LSTM Model training, forecasting and backtesting.
- `app.py` – Flask app providing forecasting API and user interface.
- `arima_model` – ARIMA Baseline model.  
- `compare_model.py` – Comparison between LSTM vs ARIMA
- `static/scrip.js` – Handles features such as user inputs, charts and API calls. 
- `static/style.css` – Design and styling of the dashboard interface. 
- `templates/index.html` – Dashboard interface .  
- `lstm_model.keras` – Saved trained lstm model.  
- `scaler_minmax.npy` – Scalar for preprocessing.
- `requirements.txt` – Dependencies for windows.
- `README.md` – Documentation about the project.

# How to run the project

1. Clone the repository

    ``` bash
    git clone https://github.com/Birkbeck/msc-project-source-code-files-24-25-harishk3112-ui.git
    cd msc-project-source-code-files-24-25-harishk3112-ui

    ```
2. Create a virtual environment

    ``` bash
    # Windows
    python -m venv venv
    
    # macOS/Linux
    python3 -m venv venv
    ``` 
3. Activate a virtual environment

    ``` bash
    # Windows
    venv\Scripts\activate
    
    # macOS/Linux
    source venv/bin/activate
    ```
4. Install Dependendices

    ```bash
   pip install -r requirements.txt
   ```
5. Run the flask app
    ```
    python app.py
    ```
6. Access the dashboard

    - Open the broswer and go to  http://127.0.0.1:5000









## Project Information

**Author :** **Harish Krishnakumar**

**MSc Advanced Computing(Data Analytics), Birkbeck,University of London**

**Supervisor:**  **Prof. George Magoulas**