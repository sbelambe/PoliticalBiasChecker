import pandas as pd

def convert_labels(input_file, output_file):
    """
    Convert the labels in the test dataset to 0 (Liberal), 1 (Neutral), or 2 (Conservative).
    """
    # Load the test dataset
    df = pd.read_csv(input_file, sep="\t", header=None, names=["Original_Label", "Text"])

    # Map original labels to new categories: Liberal (0-4 -> 0), Neutral (5 -> 1), Conservative (6-10 -> 2)
    df["Mapped_Label"] = df["Original_Label"].apply(lambda x: 0 if x <= 4 else (1 if x == 5 else 2))

    # Save the preprocessed test dataset
    df[["Text", "Mapped_Label"]].to_csv(output_file, index=False, header=["Cleaned_Text", "Mapped_Label"])

    print(f"Preprocessed test data saved to {output_file}")

# Example usage
if __name__ == "__main__":
    convert_labels("test.txt", "preprocessed_test.csv")
