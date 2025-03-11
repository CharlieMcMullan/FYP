import gradio as gr                    
import pandas as pd                    
from transformers import AutoTokenizer, AutoModelForSequenceClassification  
import torch                           
import torch.nn.functional as F        
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import hashlib

#PC Pathways
LLM_path="E:\\FYP\\Gradio\\LLM_Models\\"
Correct_Label_path ="E:\\FYP\\Gradio\\"

#Laptop Pathways
#LLM_path="/home/charlie/Desktop/Practical/Gradio/LLM_Models/"
#Correct_Label_path = "/home/charlie/Desktop/Practical/Gradio/"
# Define your model paths.
Model_1 = os.path.join(LLM_path,"subset_1")
Model_2 = os.path.join(LLM_path,"subset_2")
Model_3 = os.path.join(LLM_path,"subset_3")
Model_4 = os.path.join(LLM_path,"subset_4")
Model_5 = os.path.join(LLM_path,"subset_5")

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

def classify_logs_file(file, model_name):
    """
    This function takes a CSV file and a model path to classify each log entry.
    It returns two outputs:
      1. File metadata (e.g., file name, size, row count, and column count).
      2. A text summary of per-row predictions along with overall metrics.
    """
    # Get file metadata.
    metadata = File_MetaData(file)
    
    # Load the tokenizer and model.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    try:
        # Read the CSV file (tab-separated with header).
        df = pd.read_csv(file.name, delimiter="\t", header=0)
    except Exception as e:
        return f"Error reading CSV file: {e}", ""
    
    results = []       
    predictions = []   
    true_labels = []

    # Read the correct labels from the labels CSV.
    Data_Label_csv= os.path.join(Correct_Label_path,"Data_Labels.csv")
    df_labels = pd.read_csv(Data_Label_csv, delimiter="\t", header=0)

    # Process each row.
    for i, row in df.iterrows():
        # Combine all non-null values into a single string.
        log_entry = " ".join([str(x) for x in row if pd.notnull(x)])
        
        # Tokenize and classify.
        inputs = tokenizer(log_entry, return_tensors="pt")
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        predictions.append(predicted_class)
        
        # Retrieve the correct label from the labels CSV.
        correct_label = df_labels.iloc[i, 0]
        if correct_label.strip().lower() == "benign":
            correct_numeric_label = 0
        else:
            correct_numeric_label = 1
        true_labels.append(correct_numeric_label)
        
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
    
    # Combine per-row results into one text string.
    classification_output = "\n\n".join(results)

    # Compute classification metrics.
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    
    metrics = (
        "Metrics:\n"
        f"Accuracy: {accuracy:.2f}\n"
        f"Precision: {precision:.2f}\n"
        f"Recall: {recall:.2f}\n"
        f"F1 Score: {f1:.2f}\n"
    )
    Metric_output = metrics

    Confusion_Matric = confusion_matrix(true_labels,predictions)
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
    title="LLM Network Traffic Classification",
    description=(
        "Upload a CSV file containing log entries."
    )
)

# Launch the interface.
# Launch the interface.
iface.launch()
