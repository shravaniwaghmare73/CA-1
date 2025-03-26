import numpy as np

def calculate_chained_accuracy(y_true, y_pred):
    scores = []

    for true_vals, pred_vals in zip(y_true, y_pred):
        correct = 0

        if pred_vals[0] == true_vals[0]:
            correct += 1
            if pred_vals[1] == true_vals[1]:
                correct += 1
                if pred_vals[2] == true_vals[2]:
                    correct += 1

        score = correct / 3
        scores.append(score)

    avg_accuracy = np.mean(scores)
    return avg_accuracy, scores


# For debug testing only (can be removed later)
if __name__ == "__main__":
    y_true = np.array([[1, 2, 3], [1, 2, 2], [0, 0, 0]])
    y_pred = np.array([[1, 2, 3], [1, 3, 2], [1, 0, 0]])

    avg, details = calculate_chained_accuracy(y_true, y_pred)
    print(f"Chained Accuracy: {avg*100:.2f}%")
    print("Per-instance:", details)