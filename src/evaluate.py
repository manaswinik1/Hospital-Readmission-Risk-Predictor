"""
evaluate.py: Evaluates and visualizes model performance.
"""

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(y_true, y_pred, y_probs):
    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred))

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    auc = roc_auc_score(y_true, y_probs)
    print(f"\n=== AUC Score: {auc:.4f}")
    return auc
