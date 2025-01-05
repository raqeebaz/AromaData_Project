

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    roc_auc_score,
)


# Load dataset
def load_data(file_path):
    data = pd.read_excel(file_path)
    data['time'] = pd.to_datetime(data['time'])  # Convert time column to datetime
    X = data.drop(columns=['y', 'time'])  # Features
    y = data['y']  # Target
    return X, y
 
   from imblearn.over_sampling import SMOTE

   # Handle class imbalance
  smote = SMOTE(random_state=42)
   X_train, y_train = smote.fit_resample(X_train, y_train)

# Preprocess data
def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


# Train model
from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning for Random Forest
def tune_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
    }
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(class_weight='balanced', random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print("Best Parameters for Random Forest:", grid_search.best_params_)
    return grid_search.best_estimator_

# Hyperparameter tuning for Logistic Regression
def tune_logistic_regression(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],  # Regularization types
        'solver': ['liblinear'],  # Compatible with 'l1'
    }
    grid_search = GridSearchCV(
        estimator=LogisticRegression(class_weight='balanced', random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print("Best Parameters for Logistic Regression:", grid_search.best_params_)
    return grid_search.best_estimator_

# Tune and train Random Forest
rf_model = tune_random_forest(X_train, y_train)
print("Random Forest Results:")
evaluate_model(rf_model, X_test, y_test)

# Tune and train Logistic Regression
lr_model = tune_logistic_regression(X_train, y_train)
print("Logistic Regression Results:")
evaluate_model(lr_model, X_test, y_test)




# Evaluate model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Classification metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend()
    plt.show()


# Visualization: Target Distribution
def plot_target_distribution(y):
    plt.figure(figsize=(6, 4))
    y.value_counts().plot(kind="bar", color=["skyblue", "salmon"])
    plt.title("Target Variable Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks([0, 1], ["Non-Anomalous (0)", "Anomalous (1)"])
    plt.show()


# Visualization: Confusion Matrix
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()


# Visualization: ROC Curve
def plot_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()


# Main function
if __name__ == "__main__":
    # Update this path with your dataset file location
    file_path = "data/AnomaData.xlsx"

    # Load and analyze the data
    X, y = load_data(file_path)
    print("Dataset loaded successfully.")

    # Plot target distribution
    print("\nVisualizing target distribution...")
    plot_target_distribution(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Preprocess data
    print("\nPreprocessing the data...")
    X_train_scaled = preprocess_data(X_train)
    X_test_scaled = preprocess_data(X_test)

    # Train the model
   # Train Logistic Regression
    logistic_model = train_logistic_model(X_train, y_train)
    print("Logistic Regression Results:")
    evaluate_model(logistic_model, X_test, y_test)

    # Evaluate the model
    print("\nEvaluating the model...")
    evaluate_model(model, X_test_scaled, y_test)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(y_test, model.predict(X_test_scaled))
    plot_roc_curve(y_test, model.predict_proba(X_test_scaled)[:, 1])


