import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
import joblib  # For saving and loading the trained model

# Load the CSV file into a DataFrame
def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocess text data
def preprocess_text(text):
    # You may want to reuse the same preprocessing logic used during training
    # Add your preprocessing steps here
    return text

# Train the model
def train_model(file_path, build_new_model=True):
    if build_new_model:
        # Load data
        df = load_data(file_path)

        # Preprocess text data
        df['Text'] = df['Text'].apply(preprocess_text)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Emotion'], test_size=0.2, random_state=42)

        # Create a pipeline with TF-IDF vectorizer and SVM classifier
        model = make_pipeline(TfidfVectorizer(), SVC(kernel='linear', probability=True))

        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)

        print("Model Evaluation:")
        print(f"Accuracy: {accuracy:.2f}")
        print("Classification Report:\n", report)

        # Save the trained model to a file
        joblib.dump(model, 'emotion_model.joblib')
    else:
        # Load the pre-trained model
        model = joblib.load('emotion_model.joblib')

    return model

# Predict emotion probabilities for new text
def predict_emotion_probabilities(model, text):
    emotions = ['anger', 'fear', 'happy', 'love', 'sadness', 'surprise']
    probabilities = model.predict_proba([text])[0]

    result = {emotion: prob for emotion, prob in zip(emotions, probabilities)}
    return result

# Save text and emotion probabilities to CSV
def save_to_csv(filename, text, emotion_probabilities):
    data = {'Text': [text], **emotion_probabilities}
    df = pd.DataFrame(data)
    df.to_csv(filename, mode='a', header=not pd.io.common.file_exists(filename), index=False)

if __name__ == "__main__":
    # Use 'NLP-Emotion-Dataset.csv' in the same directory
    file_path = 'NLP-Emotion-Dataset.csv'

    # Ask the user if they want to build a new model or use the existing one
    user_input_build_model = input("Do you want to build a new model? (y/n): ")
    build_new_model = user_input_build_model.lower() == 'y'

    # Prompt the user if they want to evaluate the model
    user_input_evaluate_model = input("Do you want to evaluate the model? (y/n): ")
    evaluate_model = user_input_evaluate_model.lower() == 'y'

    # Build and train the model
    trained_model = train_model(file_path, build_new_model)

    # Evaluate the model if the user chooses to
    if evaluate_model:
        print("\nEvaluating the model:")
        # Load data for evaluation
        df_eval = load_data(file_path)
        # Preprocess text data
        df_eval['Text'] = df_eval['Text'].apply(preprocess_text)
        # Predict on the entire dataset
        predictions_eval = trained_model.predict(df_eval['Text'])
        accuracy_eval = accuracy_score(df_eval['Emotion'], predictions_eval)
        report_eval = classification_report(df_eval['Emotion'], predictions_eval)
        print(f"Accuracy on the evaluation dataset: {accuracy_eval:.2f}")
        print("Classification Report on the evaluation dataset:\n", report_eval)

    # Accept new text inputs until the user enters "q"
    while True:
        new_text = input("Enter new text (or 'q' to quit): ")
        if new_text.lower() == 'q':
            break

        # Preprocess the new text
        preprocessed_text = preprocess_text(new_text)

        # Predict emotion probabilities
        emotion_probabilities = predict_emotion_probabilities(trained_model, preprocessed_text)

        # Print the results
        print("\nEmotion Probabilities:")
        for emotion, prob in emotion_probabilities.items():
            print(f"{emotion}: {prob:.4f}")

        # Save text and emotion probabilities to CSV
        save_to_csv('emotion_predictions.csv', new_text, emotion_probabilities)

        # Add a line break after printing all emotion probabilities
        print()
