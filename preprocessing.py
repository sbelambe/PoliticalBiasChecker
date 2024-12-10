import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

def clean_text(text):
    """
    Clean the input text by removing URLs, punctuation, numbers, and stopwords.
    """
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove special characters, numbers, and punctuation
    text = re.sub(r"[^A-Za-z\s]", '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = text.strip()
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def map_labels(label):
    """
    Map the original labels to 0 (Liberal), 1 (Neutral), 2 (Conservative).
    """
    if label in [0, 1, 2, 3, 4]:
        return 0  # Liberal
    elif label == 5:
        return 1  # Neutral
    elif label in [6, 7, 8, 9, 10]:
        return 2  # Conservative

def preprocess_data(input_file, output_file):
    """
    Preprocess the dataset and save the cleaned version with mapped labels.
    """
    # Load the dataset
    df = pd.read_csv(input_file, sep="\t", header=None, names=["Label", "Text"])

    # Clean the text
    df["Cleaned_Text"] = df["Text"].apply(clean_text)

    # Map labels
    df["Mapped_Label"] = df["Label"].apply(map_labels)

    # Save only the relevant columns
    df_cleaned = df[["Cleaned_Text", "Mapped_Label"]]
    df_cleaned.to_csv(output_file, index=False)

    print(f"Preprocessed data saved to {output_file}")

# Example usage
if __name__ == "__main__":
    preprocess_data("train_orig.txt", "preprocessed_train.csv")
