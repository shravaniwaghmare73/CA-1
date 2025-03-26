import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from collections import defaultdict
from Config import config

class HierarchicalModel:
    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier
        self.model_type2 = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models_type3 = {}
        self.models = defaultdict(dict)

    def train(self, X_train, y_train):
        """
        Trains the hierarchical model:
        - M1 for Type2
        - M2 for Type3 (per class of Type2)
        - M3 for Type4 (per class of Type3 inside Type2)
        """
        type2_labels = y_train[config['label_columns'][0]]
        type3_labels = y_train[config['label_columns'][1]]
        type4_labels = y_train[config['label_columns'][2]]

        # Train M1 for Type2
        self.model_type2.fit(X_train, type2_labels)

        for type2_class in type2_labels.unique():
            idx_type2 = type2_labels == type2_class
            X_t2 = X_train[idx_type2]
            y_t3 = type3_labels[idx_type2]
            y_t4 = type4_labels[idx_type2]

            # Train M2 for Type3 under current Type2
            model_type3 = RandomForestClassifier(n_estimators=100, random_state=42)
            model_type3.fit(X_t2, y_t3)
            self.models_type3[type2_class] = model_type3

            for type3_class in y_t3.unique():
                idx_type3 = y_t3 == type3_class
                X_t3 = X_t2[idx_type3]
                y_t4_final = y_t4[idx_type3]

                if len(y_t4_final) > 1:
                    model_type4 = RandomForestClassifier(n_estimators=100, random_state=42)
                    model_type4.fit(X_t3, y_t4_final)
                    self.models[type2_class][type3_class] = model_type4

    def predict(self, X_test):
        """
        Predicts using hierarchical strategy.
        - Predict Type2
        - Use predicted Type2 to select M2 → Predict Type3
        - Use predicted Type3 to select M3 → Predict Type4
        """
        type2_preds = self.model_type2.predict(X_test)
        type3_preds = []
        type4_preds = []

        for i, x in enumerate(X_test):
            x_ = x.reshape(1, -1)
            pred_type2 = type2_preds[i]
            model_type3 = self.models_type3.get(pred_type2)

            if model_type3:
                pred_type3 = model_type3.predict(x_)[0]
            else:
                pred_type3 = 0
            type3_preds.append(pred_type3)

            model_type4 = self.models.get(pred_type2, {}).get(pred_type3)
            if model_type4:
                pred_type4 = model_type4.predict(x_)[0]
            else:
                pred_type4 = 0
            type4_preds.append(pred_type4)

        return type2_preds, type3_preds, type4_preds

    def evaluate(self, X_test, y_test):
        y_type2 = y_test[config['label_columns'][0]]
        y_type3 = y_test[config['label_columns'][1]]
        y_type4 = y_test[config['label_columns'][2]]
        p2, p3, p4 = self.predict(X_test)

        print("\n--- Evaluation Report ---")
        print("\n[Type 2 Classification Report]")
        print(classification_report(y_type2, p2))

        print("\n[Type 3 Classification Report]")
        print(classification_report(y_type3, p3))

        print("\n[Type 4 Classification Report]")
        print(classification_report(y_type4, p4))

        return p2, p3, p4