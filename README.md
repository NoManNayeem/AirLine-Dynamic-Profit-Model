
# Airline Ticket Price Prediction Flask App

This project demonstrates a simple machine learning application using Flask to predict airline ticket prices based on various flight attributes. It utilizes a RandomForestRegressor model trained on synthetic data to provide price predictions.

## Features

- Predict airline ticket prices using flight details.
- Modern UI for submitting flight details and viewing predictions.
- REST API endpoint for price predictions.

## Getting Started

### Prerequisites

- Python 3.6+
- Flask
- pandas
- scikit-learn
- joblib

### Installation

1. Clone the repository to your local machine.
2. Install the required Python packages:

```bash
pip install Flask pandas scikit-learn joblib
```

3. Navigate to the project directory and run the Flask app:

```bash
python app.py
```

The application will start running on `http://localhost:5000/`.

### Usage

Open a web browser and go to `http://localhost:5000/` to access the application. Enter the flight details into the form and submit to get a price prediction.

## API Reference

The application exposes a single API endpoint for price predictions:

- **URL**: `/predict`
- **Method**: `POST`
- **Data Params** (JSON):

```json
{
    "Day of Week": "1",
    "Time of Day": "Morning",
    "Duration": "5",
    "Origin": "Origin_1",
    "Destination": "Destination_1",
    "Seats Available": "100",
    "Historical Load Factor": "0.75",
    "Competitor Price": "300"
}
```

- **Success Response**:

```json
{
    "predicted_price": 250.5
}
```

## License

This project is open-soucre and free to use. Code, Crack and Enjoy!
