# Email Spam Detection

This project is an **Email Spam Detection Model** developed using Python and machine learning techniques. It classifies emails as spam or not spam by analyzing the email content. The final model utilizes the **Random Forest Classifier**, which demonstrated high accuracy and reliability in distinguishing spam from non-spam emails.

## Project Overview

This project follows a typical machine learning workflow, including data preprocessing, balancing, text processing, feature extraction, model training, evaluation, and saving the final model.

### Key Steps:

1. **Data Preprocessing**:
   - Loaded the spam dataset.
   - Cleaned the data by removing unnecessary columns, handling missing values, and removing duplicates.
   - Encoded target labels for binary classification (spam and non-spam).

2. **Balancing the Dataset**:
   - The dataset was imbalanced, with more non-spam than spam messages.
   - Used **Random Over Sampling** to balance the dataset, making the model more robust in identifying spam.

3. **Text Processing**:
   - Preprocessed email content by removing punctuation, numbers, and stopwords, and then lemmatizing the text.
   - Added a new feature representing the length of each email for further analysis.

4. **Visualization**:
   - Plotted the distribution of email lengths for spam and non-spam emails.
   - Created word clouds to visualize commonly occurring words in both spam and non-spam emails.

5. **Feature Extraction**:
   - Used **TF-IDF Vectorization** to transform the email text into numerical features, limiting the features to the most relevant 3000 words.

6. **Model Training**:
   - Tried multiple machine learning algorithms: Logistic Regression, Naive Bayes, SVM, and Random Forest Classifier.
   - Each model was evaluated based on accuracy, precision, recall, F1 score, and confusion matrix.

7. **Final Model Selection**:
   - Chose **Random Forest Classifier** as the final model due to its superior performance.

8. **Saving the Model**:
   - Saved the trained model as a pickle file (`spam_detector_model.pkl`) for future use.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Email-Spam-Detection.git
   cd Email-Spam-Detection
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK resources (stopwords and wordnet) if not already installed:
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Usage

1. **Training the Model**:
   Run the notebook or script to preprocess the data, train the model, and save the final classifier.

2. **Prediction**:
   Load `spam_detector_model.pkl` and use it to classify new email content as spam or not spam.

## Evaluation

The model was evaluated using the following metrics:
- **Accuracy**: Measures the overall correct predictions.
- **Precision**: Focuses on the accuracy of spam predictions.
- **Recall**: Assesses the model’s ability to capture all spam emails.
- **F1 Score**: Balances precision and recall to measure the model's accuracy.
  
The **Random Forest Classifier** yielded the best results, with minimal misclassification and high reliability.
