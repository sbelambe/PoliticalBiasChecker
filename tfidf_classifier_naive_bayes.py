import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from preprocess import load_data  # Import the load_data function from preprocess.py

def train_and_evaluate_tfidf_nb(train_data, test_data):
    """
    Trains a Naive Bayes Classifier using TF-IDF features and evaluates its performance.
    
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
        max_features=20000,  # Increase the number of features
        ngram_range=(1, 3),  # Use unigrams and bigrams
        stop_words='english',  # Remove stopwords
        min_df=5,  # Ignore words that appear in fewer than 5 documents
        max_df=0.95  # Ignore words that appear in more than 95% of the documents
    )

    
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    print(f"TF-IDF matrix for training data: {X_train_tfidf.shape}")
    print(f"TF-IDF matrix for test data: {X_test_tfidf.shape}")
    
    # Train Naive Bayes Classifier
    print("\n🤖 Training Naive Bayes Classifier...")
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_tfidf, y_train)
    
    # Predict on the test set
    print("\n📈 Making predictions on the test set...")
    y_pred = nb_classifier.predict(X_test_tfidf)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎉 Accuracy: {accuracy * 100:.2f}%")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\n📊 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
def main():
    train_file = "preprocessed_train.csv"
    test_file = "preprocessed_test.csv"
    
    print("\n📂 Loading and preprocessing data from preprocess.py...")
    train_data, test_data = load_data(train_file, test_file)
    
    print("\n⚙️ Training and evaluating TF-IDF + Naive Bayes Classifier...")
    train_and_evaluate_tfidf_nb(train_data, test_data)

if __name__ == "__main__":
    main()
