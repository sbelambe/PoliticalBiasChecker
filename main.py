from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def load_data(train_file, test_file):
    """
    Load the preprocessed training and testing data into Hugging Face Dataset format.
    """
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Convert to Hugging Face Dataset format
    train_data = Dataset.from_pandas(train_df)
    test_data = Dataset.from_pandas(test_df)

    return train_data, test_data


def tokenize_function(examples, tokenizer):
    """
    Tokenize the text data using the provided BERT tokenizer.
    """
    return tokenizer(examples["Cleaned_Text"], truncation=True, padding="max_length", max_length=128)


def compute_metrics(pred):
    """
    Compute accuracy, precision, recall, and F1 metrics for evaluation.
    """
    predictions, labels = pred
    predictions = predictions.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    # Load preprocessed data
    train_file = "preprocessed_train.csv"
    test_file = "preprocessed_test.csv"
    train_data, test_data = load_data(train_file, test_file)

    # Load BERT tokenizer and model
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # Tokenize the datasets
    train_data = train_data.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_data = test_data.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Remove unnecessary columns
    train_data = train_data.remove_columns(["Cleaned_Text"])
    test_data = test_data.remove_columns(["Cleaned_Text"])

    # Set dataset format for PyTorch
    train_data.set_format("torch")
    test_data.set_format("torch")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=50,
    )

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    print("Training the BERT model...")
    trainer.train()

    # Evaluate the model
    print("\nEvaluating the BERT model...")
    metrics = trainer.evaluate()
    print(metrics)


if __name__ == "__main__":
    main()
