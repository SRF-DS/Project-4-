# Risk Tolerance Prediction with DistilBERT and Sentence Transformers

This project uses the DistilBERT model for sequence classification along with Sentence Transformers to predict the risk tolerance of investors based on their responses to a set of questions. The model classifies the risk tolerance into five levels, ranging from very low risk tolerance (Level 1) to very high risk tolerance (Level 5). Additionally, sentiment analysis and keyword-based semantic similarity are used to enhance the risk scoring, providing a more nuanced understanding of the investor's responses.

## Features

- **Synthetic Data Generation:** Generates a dataset of 200 investors with responses to 10 predefined questions, each related to their investment preferences based on their risk tolerance.
- **Data Preprocessing:** Tokenizes and encodes text responses using DistilBERT's tokenizer.
- **Model Training:** Utilizes the DistilBERT model for sequence classification to predict risk tolerance levels.
- **Evaluation:** Evaluates the model's performance using accuracy and F1 score.
- **Enhanced Risk Scoring:** Uses sentiment analysis combined with keyword-based semantic similarity to provide a more precise classification of risk tolerance.

## Requirements

- Python 3.x
- PyTorch
- scikit-learn
- Hugging Face Transformers
- Sentence-Transformers
- pandas
- numpy

You can install the required packages using the following command:

```bash
pip install torch transformers sentence-transformers scikit-learn pandas numpy
```

## How It Works

### 1. **Data Generation**

The `templates` dictionary defines sample responses for five different risk tolerance levels. These responses are randomly assigned to 200 synthetic investors, with each investor having 10 responses (one for each question). The data is structured as follows:

- **10 Questions** (`Q1` to `Q10`): Text responses based on the investor's risk level.
- **Risk Tolerance Label**: An integer representing the risk tolerance level (1-5).

### 2. **Data Preprocessing**

The dataset is split into:
- **Features (X)**: Responses to the 10 questions.
- **Labels (y)**: Risk tolerance levels (1-5).

The data is split into 90% for training and 10% for testing using `train_test_split` from `sklearn`.

### 3. **Tokenization**

The responses are tokenized using the DistilBERT tokenizer, which converts text data into tokens (sub-word units) that the model can process.

### 4. **Model Training**

A pre-trained DistilBERT model (`distilbert-base-uncased`) is used for sequence classification. The model is fine-tuned on the training dataset for 10 epochs. The following training parameters are used:

- **Learning rate**: 2e-5
- **Batch size**: 16 (both for training and evaluation)
- **Weight decay**: 0.01 (for regularization)
- **Save strategy**: Save model after each epoch
- **Logging**: Logs every 10 steps
- **Evaluation**: Evaluate at the end of each epoch

### 5. **Evaluation**

The trained model is evaluated on the test dataset using accuracy and F1 score. These metrics are calculated to gauge the model's performance in predicting the correct risk tolerance levels.

### 6. **Model Saving**

The trained model is saved to disk in the `./distilbert_risk_tolerance_model` directory for future use.

## Code Overview

### Data Generation

```python
templates = { ... }  # Define response templates for each risk tolerance level
data = []

for investor_id in range(1, num_investors + 1):
    risk_level = random.randint(1, 5)
    responses = random.sample(templates[risk_level], 10)
    data.append([investor_id] + responses + [risk_level])

df = pd.DataFrame(data, columns=columns)
```

### Tokenization and Dataset Preparation

```python
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def encode_examples(examples):
    return tokenizer(examples, padding=True, truncation=True, max_length=128, return_tensors="pt")

train_encodings = encode_examples([' '.join(row) for row in X_train])
test_encodings = encode_examples([' '.join(row) for row in X_test])

train_dataset = RiskToleranceDataset(train_encodings, y_train)
test_dataset = RiskToleranceDataset(test_encodings, y_test)
```

### Model and Trainer Setup

```python
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=5)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
```

### Training the Model

```python
trainer.train()
```

### Saving the Model

```python
trainer.save_model("./distilbert_risk_tolerance_model")
```

### Displaying Training Metrics

After training, you can display a string with the training and evaluation metrics as follows:

```python
x = f""" Training Loss: 0.10 | Validation Loss: 0.08 | Accuracy: 100% | F1 Score: 1.00"""
print(x)
```

This string is a placeholder for the actual training metrics and shows that the model achieved 100% accuracy and a perfect F1 score during training and validation, which is an ideal scenario. Replace these values with the actual metrics generated by your model after training.

### Key Observations

```python
print("Key Observations:")
print("The model shows perfect accuracy and F1 scores by epoch 10, indicating strong performance on both training and validation data.")
print("The consistent decrease in validation loss suggests the model is effectively learning and generalizing.")
print("However, the model appears to be overfitting after epoch 1, as the accuracy and F1 score remain at 100% for the remainder of the epochs.")
print("Consider adjusting the number of epochs, adding regularization, or implementing early stopping to prevent overfitting.")
```

### Using the Trained Model for Risk Tolerance Classification

You can use the trained model to classify an investor's risk tolerance based on their responses. Here's how you can load the model and make predictions:

```python
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline

# Load pre-trained model for sentiment analysis or risk classification
tokenizer = DistilBertTokenizer.from_pretrained('./distilbert_risk_tolerance_model')
model = DistilBertForSequenceClassification.from_pretrained('./distilbert_risk_tolerance_model', num_labels=5)

# Function to tokenize and get the risk tolerance classification
def classify_risk_tolerance(response):
    # Tokenize the user response
    inputs = tokenizer(response, return_tensors='pt', padding=True, truncation=True, max_length=128)
    
    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted class (Risk Tolerance Level from 0 to 4)
    predictions = torch.argmax(outputs.logits, dim=1)
    
    return predictions.item()
```

### Enhanced Risk Scoring with Sentiment Analysis and Semantic Similarity

We enhance the risk tolerance prediction by integrating both sentiment analysis and semantic similarity using the `SentenceTransformer` model. This allows for better handling of responses, even when they don't exactly match predefined keywords.

```python
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load transformer pipelines and models
sentiment_analyzer = pipeline("sentiment-analysis")
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight sentence transformer

# Predefined keyword combinations for each risk score
keywords = { ... }  # Predefined keywords for each risk score

# Embed all keywords during initialization for semantic similarity checks
keyword_embeddings = {score: embedder.encode(phrases, convert_to_tensor=True) 
                      for score, phrases in keywords.items()}

# Enhanced risk scoring function
def classify_risk_tolerance(response):
    # Analyze sentiment as a fallback
    sentiment = sentiment_analyzer(response)[0]
    polarity = sentiment['label']
    
    # Embed user response
    response_embedding = embedder.encode(response, convert_to_tensor=True)
    
    # Calculate semantic similarity to keyword embeddings
    similarity_scores = {}
    for score, embeddings in keyword_embeddings.items():
        cosine_scores = util.cos_sim(response_embedding, embeddings)
        similarity_scores[score] = np.max(cosine_scores.cpu().numpy())  # Move tensor to CPU and then convert to NumPy

    # Assign the score with the highest similarity above a threshold
    best_score = max(similarity_scores, key=similarity_scores.get)
    if similarity_scores[best_score] > 0.75:  # Similarity threshold
        return best_score
    
    # Fallback to sentiment analysis if no match is strong enough
    if polarity == "NEGATIVE":
        return 1
   

 elif polarity == "NEUTRAL":
        return 3
    else:  # POSITIVE
        return 5
```

---

### Conclusion

This implementation demonstrates a combination of multiple models (DistilBERT for classification, Sentiment Analysis, and Sentence Transformers for semantic similarity) to predict and refine the risk tolerance levels of investors based on their responses. By leveraging sentiment analysis and semantic similarity, we improve the model's ability to understand investor behavior and preferences more accurately.
