# AQI Prediction System

**Real-Time Atmospheric Pollution Analysis and Prediction Using Machine Learning in a Web-Based System**

## Overview
This project is an interactive, analytical dashboard and machine learning prediction system for monitoring and forecasting Air Quality Index (AQI) levels across India's largest metropolitan cities. 

## Features
- **Machine Learning**: Utilizes an advanced Random Forest Regressor (`final_aqi_model.pkl`) trained on a massive historical dataset to predict future pollutant danger factors.
- **AI Reasoning Engine**: Deconstructs user input to identify the dominant threat factor (e.g. `PM2.5` or `SO2`) and mathematically determines how far above legal safety limits the atmosphere has become.
- **Historical Analysis**: Employs a custom data visualization route built with Python Pandas to calculate and visualize shifting multi-city AQI peaks between the years 2015-2025.
- **Glassmorphism UI**: High-end front-end UX using premium `cubic-bezier` CSS animations, custom Chart.js easing formulas, and fully alphabetized dynamic control states.

## Run Locally
1. `pip install -r requirements.txt`
2. `python app.py`
3. Navigate to `http://127.0.0.1:5000`
