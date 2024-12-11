import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from preprocess import load_data  # Import the load_data function from preprocess.py

def train_and_evaluate_tfidf_lr(train_data, test_data):
    """
    Trains a Logistic Regression Classifier using TF-IDF features and evaluates its performance.
    
    Args:
        train_data (pd.DataFrame): Training data with 'Cleaned_Text' and 'Mapped_Label'.
        test_data (pd.DataFrame): Testing data with 'Cleaned_Text' and 'Mapped_Label'.
    """
    # Extract text and labels
    X_train = train_data['Cleaned_Text']
    y_train = train_data['Mapped_Label']
    X_test = test_data['Cleaned_Text']
    y_test = test_data['Mapped_Label']
    
    # TF-IDF Vectorization
    print("\n🔍 Vectorizing the text data with TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=10000,  # Maximum number of features
        ngram_range=(1, 2),  # Use unigrams and bigrams
        stop_words='english'  # Remove stopwords
    )
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    print(f"TF-IDF matrix for training data: {X_train_tfidf.shape}")
    print(f"TF-IDF matrix for test data: {X_test_tfidf.shape}")
    
    # Train Logistic Regression Classifier with hyperparameters
    print("\n🤖 Training Logistic Regression Classifier...")
    lr_classifier = LogisticRegression(
        C=1.0,                   # Regularization strength
        penalty='l2',             # L2 regularization
        solver='liblinear',       # Solver for optimization
        max_iter=200,             # Maximum number of iterations for optimization
        class_weight='balanced',  # Handle imbalanced classes
        multi_class='ovr',        # One-vs-rest for multiclass
        random_state=42           # For reproducibility
    )
    
    lr_classifier.fit(X_train_tfidf, y_train)
    
    # Predict on the test set
    print("\n📈 Making predictions on the test set...")
    y_pred = lr_classifier.predict(X_test_tfidf)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎉 Accuracy: {accuracy * 100:.2f}%")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Optionally, save the model and vectorizer for later use
    joblib.dump(lr_classifier, 'logistic_regression_model.pkl')
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

def main():
    train_file = "preprocessed_train.csv"
    test_file = "preprocessed_test.csv"
    
    print("\n📂 Loading and preprocessing data from preprocess.py...")
    train_data, test_data = load_data(train_file, test_file)
    
    print("\n⚙️ Training and evaluating TF-IDF + Logistic Regression Classifier...")
    train_and_evaluate_tfidf_lr(train_data, test_data)

if __name__ == "__main__":
    main()
