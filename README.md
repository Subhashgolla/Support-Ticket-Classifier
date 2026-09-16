# Support Ticket Classifier

A small machine learning project for classifying support tickets based on the text entered by a user.

The model predicts a ticket category such as Account Access, Billing, Technical Issue, or General Question. A FastAPI service exposes the model through a REST API, and a simple web page can be used to test predictions.

## Features

- Train a text classification model from sample support tickets
- Predict ticket category from ticket text
- Assign a basic priority and support team based on the predicted category
- REST API using FastAPI
- Simple browser interface
- Save and load the trained model
- Unit tests for basic API behavior
- Docker support

## Technologies

- Python
- FastAPI
- scikit-learn
- NLP / text classification
- TF-IDF
- Logistic Regression
- HTML
- CSS
- JavaScript
- Docker
- Pytest
- GitHub Actions

## Project Structure

```text
Support-Ticket-Classifier/
├── app/
│   ├── main.py
│   ├── classifier.py
│   └── static/
│       └── index.html
├── data/
│   └── support_tickets.csv
├── model/
├── training/
│   └── train_model.py
├── tests/
│   └── test_classifier.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Run the project

Install the packages:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python training/train_model.py
```

Start the API:

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://localhost:8000
```

Swagger documentation is available at:

```text
http://localhost:8000/docs
```

## Example

Ticket:

```text
I cannot sign in to my account after changing my password.
```

The application returns a predicted category, priority, and support team.

## Docker

```bash
docker build -t support-ticket-classifier .
docker run -p 8000:8000 support-ticket-classifier
```

## Future Improvements

- Add more training data
- Add ticket history in a database
- Add user authentication
- Improve priority prediction
- Deploy the API to AWS
