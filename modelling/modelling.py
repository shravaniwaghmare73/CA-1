
from modelling.chainer import ChainedClassifier
from sklearn.metrics import classification_report

def run_chained_pipeline(X_train, X_test, y_train, y_test):
    model = ChainedClassifier()
    model.train(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True)
    return predictions, report
