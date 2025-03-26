
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

class RandomForestModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = MultiOutputClassifier(
            RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        )

    def train(self, features, labels):
        self.model.fit(features, labels)

    def predict(self, features):
        return self.model.predict(features)

    def evaluate(self, features, true_labels):
        predicted = self.predict(features)
        return predicted
