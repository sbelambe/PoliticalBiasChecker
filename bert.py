from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from torch.nn import CrossEntropyLoss
from nltk.corpus import wordnet
import random

# Data augmentation function
def synonym_replacement(text):
    words = text.split()
    new_words = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            new_words.append(random.choice(synonyms[0].lemma_names()))
        else:
            new_words.append(word)
    return ' '.join(new_words)

# Load and preprocess data
def load_data(train_file, test_file):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Select 400 samples from each class
    liberal = train_df.iloc[:400]
    neutral = train_df.iloc[115356:115756]
    conservative = train_df.iloc[138428:138828]

    # Combine the selected samples
    balanced_train_df = pd.concat([liberal, neutral, conservative]).reset_index(drop=True)

    # Data augmentation
    balanced_train_df["Cleaned_Text"] = balanced_train_df["Cleaned_Text"].apply(synonym_replacement)

    # Convert to Hugging Face Dataset format
    train_data = Dataset.from_pandas(balanced_train_df)
    test_data = Dataset.from_pandas(test_df)

    return train_data, test_data

# Tokenize function
def tokenize_function(examples, tokenizer):
    return tokenizer(examples["Cleaned_Text"], truncation=True, padding="max_length", max_length=128)

# Compute metrics
def compute_metrics(pred):
    predictions, labels = pred
    predictions = predictions.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Custom Trainer with class weights
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Apply class weights
        class_weights = torch.tensor([1.0, 2.0, 1.5]).to(logits.device)
        loss_fn = CrossEntropyLoss(weight=class_weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Main function
def main():
    train_file = "preprocessed_train.csv"
    test_file = "preprocessed_test.csv"

    # Load the data
    train_data, test_data = load_data(train_file, test_file)

    # Load the tokenizer and model
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

    # Tokenize datasets
    train_data = train_data.map(
        lambda x: {**tokenize_function(x, tokenizer), "labels": x["Mapped_Label"]}, batched=True
    )
    test_data = test_data.map(
        lambda x: {**tokenize_function(x, tokenizer), "labels": x["Mapped_Label"]}, batched=True
    )

    # Remove unnecessary columns
    train_data = train_data.remove_columns(["Cleaned_Text", "Mapped_Label"])
    test_data = test_data.remove_columns(["Cleaned_Text", "Mapped_Label"])

    train_data.set_format("torch")
    test_data.set_format("torch")

    # Training arguments
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
        logging_steps=500,
        warmup_steps=500,
        lr_scheduler_type="linear",
        # Remove mixed precision
        fp16=False,
    )

    # Use the custom Trainer
    trainer = WeightedTrainer(
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
