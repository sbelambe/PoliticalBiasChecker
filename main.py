import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

def load_data(train_file, test_file):
    """
    Load the preprocessed training and testing data.
    """
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    return train_df, test_df

def train_model(train_df):
    """
    Train a Logistic Regression model on the training data.
    """
    # Split training data into features and labels
    X_train = train_df['Cleaned_Text']
    y_train = train_df['Mapped_Label']
    
    # Vectorize the text using CountVectorizer
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    
    # Train a Logistic Regression model
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)
    
    return model, vectorizer

def evaluate_model(model, vectorizer, test_df):
    """
    Evaluate the model on the test data.
    """
    # Split test data into features and labels
    X_test = test_df['Cleaned_Text']
    y_test = test_df['Mapped_Label']
    
    # Transform the test text into feature vectors
    X_test_vec = vectorizer.transform(X_test)
    
    # Predict labels for the test set
    y_pred = model.predict(X_test_vec)
    
    # Evaluate performance
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    # Load the preprocessed data
    train_file = "preprocessed_train.csv"
    test_file = "preprocessed_test.csv"
    train_df, test_df = load_data(train_file, test_file)
    
    # Train the model
    print("Training the model...")
    model, vectorizer = train_model(train_df)
    
    # Evaluate the model
    print("\nEvaluating the model...")
    evaluate_model(model, vectorizer, test_df)
