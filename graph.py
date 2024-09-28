import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load the actual probabilities predicted by each classifier for each sample
# ...

# Calculate TPR and FPR for each classifier
algos = {
    'SVM': (tpr_svm, fpr_svm),
    'KNN': (tpr_knn, fpr_knn),
    'Random Forest': (tpr_rf, fpr_rf),
    'Gradient Boosting': (tpr_gb, fpr_gb),
    'Multinomial NB': (tpr_mnb, fpr_mnb)
}

plt.figure(figsize=(8, 6))

# Plot the ROC curve for each classifier
for algo, (tpr, fpr) in algos.items():
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{algo} (AUC = {auc_score:.2f})')

# Plot the random classifier line
plt.plot([0, 1], [0, 1], 'k--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()