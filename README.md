# Loan-Approval-Prediction System

This repository contains a Flask-based web application that predicts loan approval using a machine learning model. The application includes user authentication, CSV file handling, and text-to-speech functionality. Additionally, it leverages Google's generative AI for a chatbot feature.

## Features

- **Loan Prediction**: Predicts loan approval based on user inputs using a trained machine learning model.
- **CSV Upload**: Allows batch processing of loan applications via CSV files.
- **User Authentication**: Provides signup and login functionality for users.
- **Interest Calculation**: Calculates loan interest based on a fixed interest rate.
- **Text-to-Speech**: Converts text to speech in English or Hindi.
- **Generative AI Chatbot**: Integrates a chatbot powered by Google's generative AI model.

## Getting Started

### Prerequisites

- Python 3.x
- Flask
- pandas
- scikit-learn
- psycopg2
- gTTS
- pyglet
- google-generativeai
- PostgreSQL

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/loan-prediction-system.git
    cd loan-prediction-system
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up the PostgreSQL database:

    - Create a database named `flask_data`.
    - Create a table `users_1` with the following schema:
    
      ```sql
      CREATE TABLE users_1 (
          id SERIAL PRIMARY KEY,
          username VARCHAR(50),
          mobile VARCHAR(15),
          email VARCHAR(50),
          password VARCHAR(64)
      );
      ```

4. Place your trained model `LR.pkl` in the project root directory.

5. Set up the `UPLOAD_FOLDER` for CSV file uploads:

    ```bash
    mkdir uploads
    ```

6. Configure the Google Generative AI API key:

    - Replace the placeholder API key in the code with your actual API key.
    - Ensure you have enabled the necessary API on your Google Cloud project.

### Running the Application

1. Start the Flask server:

    ```bash
    python app.py
    ```

2. Access the application at `http://localhost:5000`.

## File Structure

- `app.py`: Main Flask application file.
- `templates/`: HTML templates for rendering web pages.
- `static/`: Static files such as CSS and JavaScript.
- `uploads/`: Directory for uploading CSV files.
- `LR.pkl`: Trained machine learning model.
- `requirements.txt`: List of required Python packages.

## Routes

- `/`: Home page.
- `/prediction`: Handles loan prediction requests.
- `/predict_csv`: Handles CSV file uploads and batch predictions.
- `/login`: User signup and login page.
- `/interest`: Loan interest calculation form.
- `/loan_calculator`: Handles loan interest calculation requests.
- `/speak`: Text-to-speech functionality.
- `/chatbot`: Chatbot interface.
- `/generate`: Generates chatbot responses.
- `/logout`: Logs out the user.
- `/about`: About page.
- `/condition`: Terms and conditions page.
- `/index`: Home page redirect.

## Usage

### Loan Prediction

1. Go to the home page.
2. Fill in the loan application form.
3. Submit the form to get a prediction.

### CSV Upload

1. Navigate to the CSV upload page.
2. Upload a CSV file containing loan applications.
3. Receive batch predictions for all applications in the CSV file.

### User Authentication

1. Sign up with a new account on the login page.
2. Log in using your email and password.

### Text-to-Speech

1. Send a POST request to `/speak` with JSON data containing `text` and `lang` (either 'en' for English or 'hi' for Hindi).
2. The server will convert the text to speech and play it.

### Chatbot

1. Go to the chatbot page.
2. Interact with the chatbot powered by Google's generative AI.


## Acknowledgements

- Flask
- pandas
- scikit-learn
- psycopg2
- gTTS
- pyglet
- Google Generative AI

## Contributing

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Create a new Pull Request.
