import gradio as gr
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import numpy as np
import hashlib

#PC Pathways
ML_path="E:\\FYP\\Gradio\\ML_Models\\"
Correct_Label_path ="E:\\FYP\\Gradio\\"

#Laptop Pathways
#ML_path="/home/charlie/Desktop/Practical/Gradio/ML_Models/"
#Correct_Label_path = "/home/charlie/Desktop/Practical/Gradio/"

Model_1 = os.path.join(ML_path,"subset_1.json")
Model_2 = os.path.join(ML_path,"subset_2.json")
Model_3 = os.path.join(ML_path,"subset_3.json")
Model_4 = os.path.join(ML_path,"subset_4.json")
Model_5 = os.path.join(ML_path,"subset_5.json")

def File_MetaData(file):
    try:
        file_name = file.name
        file_size = os.path.getsize(file.name)
        sha1 = hashlib.sha1()
        with open(file_name, 'rb') as f:
            # Read and update hash in chunks of 4096 bytes
            for chunk in iter(lambda: f.read(4096), b""):
                sha1.update(chunk)
        file_hash = sha1.hexdigest()
        # Read the CSV file (using the default comma delimiter)
        df = pd.read_csv(file.name)
        row_count, col_count = df.shape
        features = list(df.columns)
        metadata_text = (
            f"File Metadata:\n"
            f"Name: {file_name}\n"
            f"Size: {file_size} bytes\n"
            f"Rows: {row_count}\n"
            f"Columns: {col_count}\n"
            f"Present Features: {features}\n"
            f"File Hash: {file_hash}"
        )
        return metadata_text
    except Exception as e:
        return f"File Metadata could not be determined: {e}"

def preprocess_data(df):
    """
    Preprocess the DataFrame so that it matches the format used in training.
    This function:
      - Drops columns that were not used in training.
      - Applies one-hot encoding to categorical columns.
      - Fills missing values with 0.
    
    NOTE: For a production system, you should save and reuse the fitted OneHotEncoder.
    """
    # Drop columns not used in training.
    drop_columns = ["uid", "id.orig_h", "local_orig", "local_resp", "tunnel_parents", "service", "detailed-label"]
    df = df.drop(columns=drop_columns, errors="ignore")
    
    # Identify columns with object type as candidates for one-hot encoding.
    categorical_columns = [col for col in df.columns if df[col].dtype == object]
    
    if categorical_columns:
        # Ensure categorical columns are strings.
        df[categorical_columns] = df[categorical_columns].astype(str)
        # Use the updated parameter 'sparse_output' instead of 'sparse'
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded_array = encoder.fit_transform(df[categorical_columns])
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=encoder.get_feature_names_out(categorical_columns),
            index=df.index
        )
        # Drop the original categorical columns and add the one-hot encoded columns.
        df = df.drop(columns=categorical_columns)
        df = pd.concat([df, encoded_df], axis=1)
    
    # Fill missing values.
    df.fillna(0, inplace=True)
    return df

def classify_logs_file(file, model_name):
    """
    Given a CSV file and a selected model file, load and preprocess the data, then classify
    each row using the saved XGBoost model. Also reads a separate labels CSV file to compare
    the predicted class with the correct class.
    
    Returns:
      - File metadata.
      - A text summary with predicted and correct class for each row.
      - Overall classification metrics.
      - A confusion matrix plot.
    """
    try:
        # Read the input CSV file.
        df = pd.read_csv(file.name, delimiter=",", header=0)
    except Exception as e:
        return f"Error reading CSV file: {e}", None, None, None

    # Preprocess the data to match training.
    features = preprocess_data(df)
    if features.empty:
        return "Error: Preprocessing resulted in no features. Check your CSV file.", None, None, None

    # Load the XGBoost model.
    model = xgb.Booster()
    model.load_model(model_name)
    
    # Retrieve the feature names used during training.
    model_features = model.feature_names
    if model_features is None:
        return "Error: Model does not have feature names.", None, None, None

    # Reindex the DataFrame to have exactly the same columns as expected.
    features = features.reindex(columns=model_features, fill_value=0)
    
    # Create a DMatrix for prediction.
    dtest = xgb.DMatrix(features)
    
    # Make predictions (assume binary classification with threshold 0.5).
    predictions = model.predict(dtest)
    predicted_classes = [1 if pred >= 0.5 else 0 for pred in predictions]
    
    Data_Label_csv= os.path.join(Correct_Label_path,"Data_Labels.csv")
    df_labels = pd.read_csv(Data_Label_csv, delimiter="\t", header=0)
    
    # Build a per-row prediction summary comparing predicted and correct labels.
    results = []
    true_labels = []
    
    for i, predicted_class in enumerate(predicted_classes):
        correct_label = df_labels.iloc[i, 0]
        # Convert the correct label (assumed text) to a numeric value.
        if correct_label.strip().lower() == "benign":
            correct_numeric_label = 0
        else:
            correct_numeric_label = 1
        true_labels.append(correct_numeric_label)
        
        # Convert numeric prediction to text.
   # Convert numeric prediction to text.
        predicted_label = "Benign" if predicted_class == 0 else "Malicious"
        #highlight = "*" if predicted_label != correct_label else ""

        if predicted_label == "Malicious" and correct_label =="Malicious":
            highlight = " "
            status = "True Positive"
        
        elif predicted_label == "Malicious" and correct_label =="Benign":
            highlight = "*"
            status = "False Positive" 
        
        elif predicted_label == "Benign" and correct_label =="Malicious":
            highlight = "*"
            status = "False Negative" 

        elif predicted_label == "Benign" and correct_label =="Benign":
            highlight = " "
            status = "True Negative" 
        
        else:
            highlight = "*"
            status = "Error"

        results.append(
            f"Row {i}: Predicted class: {predicted_label} | Correct class: {correct_label} | Status: {status}{highlight}"
        )
    classification_output = "\n\n".join(results)
    
    # Compute classification metrics using the predicted_classes list.
    accuracy = accuracy_score(true_labels, predicted_classes)
    precision = precision_score(true_labels, predicted_classes)
    recall = recall_score(true_labels, predicted_classes)
    f1 = f1_score(true_labels, predicted_classes)
    
    metrics = (
        "Metrics:\n"
        f"Accuracy: {accuracy:.2f}\n"
        f"Precision: {precision:.2f}\n"
        f"Recall: {recall:.2f}\n"
        f"F1 Score: {f1:.2f}"
    )
    Metric_output = metrics

    # Compute the confusion matrix.
    Confusion_Matric = confusion_matrix(true_labels, predicted_classes)
    custom_labels = np.array([
        ["True Positive", "False Positive"],
        ["False Negative", "True Negative"]
    ])

    # Create an annotation array that combines the numeric value and the custom label.
    annot = np.empty_like(Confusion_Matric, dtype=object)
    for i in range(Confusion_Matric.shape[0]):
        for j in range(Confusion_Matric.shape[1]):
            annot[i, j] = f"{Confusion_Matric[i,j]}\n{custom_labels[i,j]}"

    fig_cm, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(Confusion_Matric, annot=annot, fmt="", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Classification")
    ax.set_ylabel("True Classification")
    ax.set_title("Confusion Matrix")

    metadata = File_MetaData(file)
    
    return metadata, Metric_output,fig_cm,classification_output

# Create a dropdown for model selection.
dropdown = gr.Dropdown(
    choices=[Model_1, Model_2, Model_3, Model_4, Model_5],
    label="Select Model",
    value=Model_1
)

# Build the Gradio interface with four outputs.
iface = gr.Interface(
    fn=classify_logs_file,
    inputs=[gr.File(label="Upload CSV File"), dropdown],
    outputs=[
        gr.Text(label="File Metadata"),
        gr.Text(label="Malicious Detection Metrics"),
        gr.Plot(label="Confusion Matrix"),
        gr.Text(label="Individual Classification Output")
    ],
    title="XGBoost ML Model Network Traffic Classification",
    description=(
        "Upload a CSV file containing log entries. "
    )
)

# Launch the interface.
iface.launch(server_port=7861)
