import PySimpleGUI as sg
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Model
from keras.layers import Input, Dense

# Ensure required folders exist
if not os.path.exists("Updated Results"):
    os.makedirs("Updated Results")

if not os.path.exists("Confusion Matrix"):
    os.makedirs("Confusion Matrix")

if not os.path.exists("Barchart"):
    os.makedirs("Barchart")


def get_autoencoder_predictions(autoencoder, X_test, threshold):
    reconstructions = autoencoder.predict(X_test)
    reconstruction_error = np.mean(np.power(X_test - reconstructions, 2), axis=1)
    return [1 if e > threshold else 0 for e in reconstruction_error]


def process_data(filename):
    # Read and preprocess data (using a subset for faster results)
    data = pd.read_csv(filename).sample(frac=0.2, random_state=42)
    X = data.drop('Class', axis=1)
    y = data['Class']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Train autoencoder (with fewer epochs for faster results)
    input_dim = X_train_resampled.shape[1]
    encoding_dim = 14
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(encoding_dim, activation="tanh")(input_layer)
    encoder = Dense(int(encoding_dim / 2), activation="relu")(encoder)
    decoder = Dense(int(encoding_dim / 2), activation='tanh')(encoder)
    decoder = Dense(input_dim, activation='relu')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    autoencoder.fit(X_train_resampled, X_train_resampled, epochs=10, batch_size=256, shuffle=True,
                    validation_data=(X_test, X_test))

    # Determine threshold for classification using autoencoder
    reconstructions = autoencoder.predict(X_train_resampled)
    mse = np.mean(np.power(X_train_resampled - reconstructions, 2), axis=1)
    threshold = np.quantile(mse, 0.95)

    # Get autoencoder predictions
    y_pred_autoencoder = get_autoencoder_predictions(autoencoder, X_test, threshold)

    # Train other classifiers with modifications for faster training
    classifiers = {
        "Autoencoder": y_pred_autoencoder,
        "Random Forest": RandomForestClassifier(n_estimators=10),
        "SVM": SVC(kernel='linear'),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "XGBOOST": XGBClassifier(n_estimators=10)
    }

    results = {"Algorithm": [], "Accuracy": [], "Precision": [], "Recall": [], "F1": []}

    for name, clf in classifiers.items():
        if name == "Autoencoder":
            y_pred = clf
        else:
            clf.fit(X_train_resampled, y_train_resampled)
            y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Save results
        results["Algorithm"].append(name)
        results["Accuracy"].append(accuracy)
        results["Precision"].append(precision)
        results["Recall"].append(recall)
        results["F1"].append(f1)

        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='g')
        plt.title(f'Confusion Matrix for {name}')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.savefig(f'Confusion Matrix/{name}.png')
        plt.close()

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.set_index("Algorithm", inplace=True)
    print(results_df)  # Debugging purpose
    results_df.to_csv('Updated Results/results.csv')

    # Generate bar chart for comparison
    results_df.plot(kind='bar', figsize=(15, 7), subplots=True)
    plt.title('Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('Barchart/comparison.png')
    plt.close()


# GUI layout
layout = [
    [sg.Text("Credit Card Fraud Detection", size=(30, 1), font=("Arial", 25))],
    [sg.FileBrowse("Choose CSV File", key='file')],
    [sg.Button("Process"), sg.Exit()]
]

window = sg.Window("Credit Card Fraud Detection", layout)

while True:
    event, values = window.read()

    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    elif event == "Process":
        if values['file']:
            process_data(values['file'])
            sg.Popup("Processing completed and results saved!")
        else:
            sg.PopupError("Please choose a valid CSV file!")

window.close()
