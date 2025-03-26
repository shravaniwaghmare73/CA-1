
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import RandomForestClassifier

class ChainedClassifier:
    def __init__(self, base_model=None):
        if base_model is None:
            base_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.chain = ClassifierChain(base_model)

    def train(self, X, Y):
        self.chain.fit(X, Y)

    def predict(self, X):
        return self.chain.predict(X)

    def get_model(self):
        return self.chain
