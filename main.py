import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ['Malignant', 'Benign']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Function to plot ROC curve
def plot_roc_curve(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

# Function to plot precision-recall curve
def plot_precision_recall_curve(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)
    plt.figure(figsize=(6, 4))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve: Avg Precision = {:.2f}'.format(average_precision))
    plt.show()



# Load the breast cancer dataset
cancer = load_breast_cancer()
print(cancer.DESCR)

cancer.keys()

print(len(cancer.feature_names))
print(cancer.feature_names)

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
print(df)

data = pd.DataFrame(cancer.data, columns=cancer.feature_names)

# Add the target variable to the DataFrame
data['target'] = cancer.target

# Calculate the class distribution
class_distribution = data['target'].value_counts()

# Rename the index for better readability
class_distribution.index = ['malignant', 'benign']

print(class_distribution)


# Split the DataFrame into X and y
X = df.drop('target', axis=1)
y = df['target']
print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
print(X_train, X_test, y_train, y_test)



# Initialize and fit the k-nearest neighbors classifier
knn_classifier = KNeighborsClassifier(n_neighbors=1)
knn_classifier.fit(X_train, y_train)

# Print the classifier and make a prediction using mean values
print(knn_classifier)
mean_values = data.mean()[:-1].values.reshape(1, -1)
predicted_label = knn_classifier.predict(X_test)
print(predicted_label)

accuracy = knn_classifier.score(X_test, y_test)

print(accuracy)

y_train_pred = knn_classifier.predict(X_train)
y_test_pred = knn_classifier.predict(X_test)

# Calculate the accuracy for training and test sets
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Plot the accuracy scores
plt.figure(figsize=(8, 6))
plt.bar(['Train Set', 'Test Set'], [train_accuracy, test_accuracy], color=['blue', 'orange'])
plt.title('Accuracy of KNN Classifier on Train and Test Sets')
plt.xlabel('Dataset')
plt.ylabel('Accuracy Score')
plt.ylim(0, 1)
plt.show() 

# Predict the class labels for the test set
y_test_pred = knn_classifier.predict(X_test)

# Plot Confusion Matrix
plot_confusion_matrix(y_test, y_test_pred)

# Plot ROC Curve
y_scores = knn_classifier.predict_proba(X_test)[:, 1]  # Use predicted probabilities for positive class
plot_roc_curve(y_test, y_scores)

# Plot Precision-Recall Curve
plot_precision_recall_curve(y_test, y_scores)
