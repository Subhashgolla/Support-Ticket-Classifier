# Support Ticket Classifier

This project is a simple support ticket classification application built using Python and machine learning.

The application takes a support ticket as input and predicts the type of issue. Based on the predicted category, it also provides a priority level and the support team that can handle the request.

## How It Works

```text
Support Ticket
      |
      v
Text Processing
      |
      v
TF-IDF
      |
      v
Logistic Regression
      |
      v
Ticket Category
      |
      v
Priority and Support Team
```

## Ticket Categories

The current model classifies tickets into four categories:

- Account Access
- Billing
- Technical Issue
- General Question

For example, a ticket such as:

```text
I cannot sign in to my account after changing my password.
```

can be classified as an `Account Access` issue.

## Technologies Used

- Python
- FastAPI
- scikit-learn
- TF-IDF
- Logistic Regression
- HTML
- CSS
- JavaScript
- Docker
- Pytest

## Project Structure

```text
Support-Ticket-Classifier/
│
├── app/
│   ├── main.py
│   ├── classifier.py
│   └── static/
│       └── index.html
│
├── data/
│   └── support_tickets.csv
│
├── model/
│
├── training/
│   └── train_model.py
│
├── tests/
│   └── test_classifier.py
│
├── Dockerfile
├── requirements.txt
└── README.md
```

## Running the Project

Install the required packages:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python training/train_model.py
```

Start the application:

```bash
uvicorn app.main:app --reload
```

Open the application in the browser:

```text
http://localhost:8000
```

API documentation is available at:

```text
http://localhost:8000/docs
```

## Model

For the current version, I used TF-IDF to convert the ticket text into numerical features and Logistic Regression for classification.

The training dataset contains sample support tickets for each of the four categories. The dataset is kept small because the main purpose of this project is to understand the complete process of training a text classifier and using it through an API.

## Testing

Tests can be run using:

```bash
pytest
```

## Future Improvements

Some improvements I would like to add are:

- Increase the training dataset
- Store submitted tickets in a database
- Improve priority prediction
- Add user login
- Deploy the application to AWS