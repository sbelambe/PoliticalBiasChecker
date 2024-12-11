import os
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler

def load_data(train_file, test_file):
    """
    Load and preprocess the training and testing datasets.
    
    Args:
        train_file (str): Path to the training dataset.
        test_file (str): Path to the testing dataset.
    
    Returns:
        tuple: A tuple containing the preprocessed training and testing datasets as Pandas DataFrames.
    """
    print("\n📂 Loading the training and testing datasets...")
    
    # Load train and test datasets
    train_df = pd.read_csv(train_file, sep="\t", header=None, names=["Original_Label", "Text"])
    test_df = pd.read_csv(test_file, sep="\t", header=None, names=["Original_Label", "Text"])
    
    print(f"Initial training set size: {train_df.shape}")
    print(f"Initial testing set size: {test_df.shape}")
    
    # Map labels for both train and test sets (Liberal: 0-4 -> 0, Neutral: 5 -> 1, Conservative: 6-10 -> 2)
    train_df["Mapped_Label"] = train_df["Original_Label"].apply(lambda x: 0 if x <= 4 else (1 if x == 5 else 2))
    test_df["Mapped_Label"] = test_df["Original_Label"].apply(lambda x: 0 if x <= 4 else (1 if x == 5 else 2))
    
    # Balance the training set by undersampling the majority classes
    print("\n⚖️ Balancing the training set...")
    undersampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(train_df[["Text"]], train_df["Mapped_Label"])
    train_df = pd.DataFrame({
        "Text": X_resampled["Text"].values, 
        "Mapped_Label": y_resampled
    })
    print(f"Balanced training set size: {train_df.shape}")
    
    # Preprocess the text (this can be expanded to include more sophisticated preprocessing)
    print("\n🧹 Cleaning text data...")
    train_df["Cleaned_Text"] = train_df["Text"].apply(clean_text)
    test_df["Cleaned_Text"] = test_df["Text"].apply(clean_text)
    
    train_df = train_df[["Cleaned_Text", "Mapped_Label"]]
    test_df = test_df[["Cleaned_Text", "Mapped_Label"]]
    
    print(f"Final training set size: {train_df.shape}")
    print(f"Final testing set size: {test_df.shape}")
    
    return train_df, test_df

def clean_text(text):
    """
    Clean the text data by removing special characters, extra whitespace, etc.
    
    Args:
        text (str): The raw text to be cleaned.
    
    Returns:
        str: The cleaned text.
    """
    if isinstance(text, str):
        # Remove special characters, multiple spaces, and lowercase the text
        text = text.replace("\n", " ").replace("\t", " ").strip().lower()
        text = ' '.join(text.split())  # Remove extra spaces
    return text

def save_preprocessed_data(train_df, test_df, train_output_file, test_output_file):
    """
    Save the preprocessed training and testing datasets to CSV files.
    
    Args:
        train_df (pd.DataFrame): The preprocessed training DataFrame.
        test_df (pd.DataFrame): The preprocessed testing DataFrame.
        train_output_file (str): Path to save the preprocessed training dataset.
        test_output_file (str): Path to save the preprocessed testing dataset.
    """
    print("\n💾 Saving the preprocessed datasets...")
    train_df.to_csv(train_output_file, index=False, header=["Cleaned_Text", "Mapped_Label"])
    test_df.to_csv(test_output_file, index=False, header=["Cleaned_Text", "Mapped_Label"])
    print(f"✅ Preprocessed training data saved to {train_output_file}")
    print(f"✅ Preprocessed testing data saved to {test_output_file}")

def main():
    train_file = "train.txt"  # Path to the training file
    test_file = "test.txt"  # Path to the testing file
    
    train_output_file = "preprocessed_train.csv"
    test_output_file = "preprocessed_test.csv"
    
    print("\n🚀 Starting the preprocessing pipeline...")
    train_df, test_df = load_data(train_file, test_file)
    
    print("\n📦 Saving the final preprocessed datasets...")
    save_preprocessed_data(train_df, test_df, train_output_file, test_output_file)
    print("\n🎉 Preprocessing complete!")

if __name__ == "__main__":
    main()
