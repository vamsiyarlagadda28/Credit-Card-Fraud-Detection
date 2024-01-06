import PySimpleGUI as sg
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import xgboost as xgb
import matplotlib.pyplot as plt
import os

# File selection
def select_file():
    layout = [[sg.Text("Select CSV File")],
              [sg.Input(), sg.FileBrowse()],
              [sg.OK(), sg.Cancel()]]
    window = sg.Window("File Selector", layout)
    event, values = window.read()
    window.close()
    if event == "OK":
        return values[0]
    else:
        return None

# Display SMOTE results
def display_smote_results(X_res, y_res):
    balanced_data = pd.concat([X_res, y_res], axis=1)
    layout = [
        [sg.Text("SMOTE Analysis Results")],
        [sg.Multiline(balanced_data.head().to_string())],
        [sg.OK()]
    ]
    window = sg.Window("SMOTE Results", layout)
    window.read()
    window.close()

# User input for prediction
def get_user_input(columns):
    layout = [[sg.Text(f"{col}: "), sg.InputText(key=col)] for col in columns]
    layout.append([sg.Button("Predict"), sg.Button("Cancel")])
    window = sg.Window("Enter Transaction Details", layout)
    event, values = window.read()
    window.close()
    if event == "Predict":
        return pd.DataFrame([values])
    else:
        return None

# Display prediction
def display_prediction(pred):
    message = "The transaction is legitimate!" if pred == 0 else "The transaction is suspicious (potential fraud)!"
    sg.Popup(message)

# MAIN
file_path = select_file()
if not file_path:
    exit("No file selected.")
df = pd.read_csv(file_path)

# Apply SMOTE
X = df.drop('Class', axis=1)
y = df['Class']
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
display_smote_results(X_res, y_res)

# Train XGBoost
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
clf = xgb.XGBClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Evaluate and save results
results = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred)
}
results_df = pd.DataFrame([results])
if not os.path.exists('results'):
    os.makedirs('results')
results_df.to_csv('results/evaluation_results.csv', index=False)

# Generate and display plots
# Bar chart
plt.figure()
plt.bar(results.keys(), results.values())
plt.title("Evaluation Metrics")
plt.savefig('results/bar_chart.png')
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.savefig('results/roc_curve.png')
plt.show()

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure()
plt.matshow(conf_matrix, cmap=plt.cm.Reds)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('results/confusion_matrix.png')
plt.show()

# User input and prediction
user_data = get_user_input(X.columns)
if user_data is not None:
    user_pred = clf.predict(user_data)
    display_prediction(user_pred[0])

sg.Popup('Analysis and prediction complete!', 'Results and plots saved in the "results" folder.')
