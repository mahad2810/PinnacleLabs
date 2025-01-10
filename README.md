
# **Sentiment Analysis Project**

## **Overview**
This project demonstrates a comprehensive approach to **Sentiment Analysis**, a crucial application of Natural Language Processing (NLP). It analyzes text data to discern sentiments, aiding businesses in assessing customer feedback and making data-driven decisions.

The project employs **Logistic Regression** for classification using a Bag of Words (BoW) and TF-IDF feature extraction, and also demonstrates the use of **Hugging Face's Transformers pipeline** for state-of-the-art sentiment analysis.

---

## **Features**
- **Text Preprocessing**: Includes tokenization, lemmatization, removal of stop words, and punctuation.
- **Feature Extraction**: Implements both Bag of Words and TF-IDF models.
- **Machine Learning Model**: Logistic Regression for sentiment classification.
- **Transformer Model Integration**: Utilizes the Hugging Face sentiment-analysis pipeline for advanced predictions.
- **Performance Evaluation**: Includes metrics such as accuracy and classification reports for model evaluation.

---

## **Dataset**
The project uses the **Movie Reviews** dataset from the `nltk` library. 
- **Dataset Details**:
  - Categories: Positive, Negative
  - Size: 2000 labeled reviews

---

## **Requirements**
### **Python Libraries**
- `nltk`
- `sklearn`
- `transformers`
- `string`
- `jupyter`

Install the required libraries using the following command:
```bash
pip install nltk scikit-learn transformers jupyter
```

---

## **Project Workflow**
### **1. Data Preprocessing**
- **Import Dataset**: Uses `nltk`'s `movie_reviews` corpus.
- **Text Preprocessing**:
  - Tokenization and lemmatization using `WordNetLemmatizer`.
  - Removal of stop words and punctuation.
  - Creation of refined data for training.

### **2. Feature Extraction**
- **Bag of Words (BoW)**: Uses `CountVectorizer` with a maximum feature size of 2000.
- **TF-IDF**: Uses `TfidfVectorizer` for enhanced feature representation.

### **3. Model Training and Evaluation**
- **Train-Test Split**: Splits data into 80% training and 20% testing subsets.
- **Model**: Logistic Regression from `sklearn`.
- **Performance Metrics**: 
  - **Accuracy**: Measures overall performance.
  - **Classification Report**: Includes precision, recall, and F1-score for both positive and negative classes.

### **4. Transformer Pipeline**
- Integrates Hugging Face's `pipeline("sentiment-analysis")` for advanced, pre-trained sentiment analysis.

### **5. Sentiment Analysis Example**
```python
new_text = "The movie was absolutely wonderful with amazing performances!"
result = sentiment_analyzer(new_text)
print("Sentiment:", result)
```

---

## **Performance**
- **Accuracy**: 0.8 (80%)
- **Classification Report**:
  ```
               precision    recall  f1-score   support

           0       0.82      0.77      0.79       199
           1       0.78      0.83      0.81       201

    accuracy                           0.80       400
   macro avg       0.80      0.80      0.80       400


## **How to Run**

Follow these steps to set up and execute the project on your local machine:

### **1. Clone the Repository**
Clone the repository to your local machine using the following command:
```bash
git clone <repository-url>
```
Replace `<repository-url>` with the actual URL of your repository.

### **2. Install Dependencies**
Navigate to the cloned repository directory:
```bash
cd <repository-folder-name>
```
Install the required dependencies using `pip`:
```bash
pip install -r requirements.txt
```

### **3. Launch Jupyter Notebook**
Start Jupyter Notebook from your terminal or command prompt:
```bash
jupyter notebook
```
This will open the Jupyter Notebook interface in your default web browser.

### **4. Open the Sentiment Analysis Notebook**
In the Jupyter Notebook interface:
- Navigate to the folder containing the cloned repository.
- Open the file `sentiment.ipynb`.

### **5. Execute the Notebook**
Run the cells in the notebook sequentially by selecting each cell and clicking on the **Run** button (or pressing `Shift + Enter`).

### **6. Test the Sentiment Analyzer**
Once the notebook is executed, you can input your own text for sentiment analysis:
- Modify the variable `new_text` with your custom sentence:
  ```python
  new_text = "Your custom text here."
  ```
- Run the corresponding cell to view the sentiment prediction.

---

This detailed guide ensures that even users with minimal technical expertise can successfully run the project.

## **Conclusion**
This project showcases a structured approach to text-based sentiment analysis, blending traditional machine learning with modern transformer-based models. The techniques and tools used provide a foundation for real-world applications in business analytics, customer feedback assessment, and decision-making.

---

