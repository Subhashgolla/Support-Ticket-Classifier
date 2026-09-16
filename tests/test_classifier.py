from app.classifier import classify_ticket

class FakeModel:
    def predict(self, texts):
        return ["Account Access"]

    def predict_proba(self, texts):
        return [[0.8, 0.1, 0.05, 0.05]]

def test_classification_result():
    result = classify_ticket("I cannot login", model=FakeModel())
    assert result["category"] == "Account Access"
    assert result["priority"] == "High"
    assert result["assigned_team"] == "Account Support"

def test_empty_ticket():
    try:
        classify_ticket(" ", model=FakeModel())
        assert False
    except ValueError:
        assert True
