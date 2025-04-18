import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

np.random.seed(13)
tf.random.set_seed(13)

# get NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# DNN experiment using LSTM and CNN model to classify trip advisor reviews
# And find best performing model
class reviewSentimentClassifier:
    
    def __init__(self, dataPath, maxTopFrequentWords=15000, maxLen=200, embeddingDim=100):

        self.dataPath = dataPath 
        self.maxTopFrequentWords = maxTopFrequentWords
        self.maxLen = maxLen
        self.embeddingDim = embeddingDim
        
        #storing models and data
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.tokenizer = None
        self.cnn_model = None
        self.lstm_model = None
        self.best_model = None
        self.best_model_name = None
    
    # load and prepare
    def loadData(self):

        print("Loading data...")
        self.df = pd.read_csv(self.dataPath)
        
        print(f"Dataset shape: {self.df.shape}")
        print("\nSample of the data:")
        print(self.df.head())
        
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            print("\nMissing values per column:")
            print(missing_values)
            self.df = self.df.dropna()
            print(f"After dropping missing values, shape: {self.df.shape}")
        
        return self.df
    
    # data analysis and visualisation
    def analyseData(self):

        if self.df is None:
            print("Error loading data")
            return
        
        if 'Rating' in self.df.columns:
            plt.figure(figsize=(10, 6))
            sns.countplot(x='Rating', data=self.df)
            plt.title('Distribution of Ratings')
            plt.xlabel('Rating')
            plt.ylabel('Count')
            plt.savefig('rating_distribution.png')
            plt.close()
            
            #sentiment (1-2: negative, 3: neutral, 4-5: positive)
            self.df['Sentiment'] = self.df['Rating'].apply(lambda x: 0 if x <= 2 else (1 if x == 3 else 2))
            print("\nSentiment distribution:")
            print(self.df['Sentiment'].value_counts())
            
            plt.figure(figsize=(8, 5))
            sns.countplot(x='Sentiment', data=self.df)
            plt.title('Distribution of Sentiment')
            plt.xlabel('Sentiment (0: Negative, 1: Neutral, 2: Positive)')
            plt.ylabel('Count')
            plt.savefig('sentiment_distribution.png')
            plt.close()
        
        if 'Review' in self.df.columns:
            self.df['review_length'] = self.df['Review'].apply(len)
            
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df['review_length'], bins=50)
            plt.title('Distribution of Review Lengths')
            plt.xlabel('Length (characters)')
            plt.ylabel('Count')
            plt.savefig('review_length_distribution.png')
            plt.close()
            
            print(f"\nAverage review length: {self.df['review_length'].mean():.2f} characters")
            print(f"Median review length: {self.df['review_length'].median():.2f} characters")
            print(f"Max review length: {self.df['review_length'].max()} characters")
            print(f"Min review length: {self.df['review_length'].min()} characters")
    
    #prepare text for sentiment analysis
    def preprocessData(self):

        if self.df is None:
            print("Error loading data.")
            return
        
        print("\nPreprocessing text data..")
        
        print("Tokenizing text..")
        self.tokenizer = Tokenizer(num_words=self.maxTopFrequentWords, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(self.df['Review'])
        
        sequences = self.tokenizer.texts_to_sequences(self.df['Review'])
        
        data = pad_sequences(sequences, maxlen=self.maxLen, padding='post', truncating='post')
        
        word_index = self.tokenizer.word_index
        print(f"Found {len(word_index)} unique tokens")
        
        # train-test split (60-40)
        labels = self.df['Sentiment'].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data, labels, test_size=0.4, random_state=42, stratify=labels
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Testing set shape: {self.X_test.shape}")
        
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def buildCNNModel(self):

        print("\nBuilding CNN model...")
        
        # Define the model architecture
        self.cnn_model = Sequential([
            Embedding(input_dim=self.maxTopFrequentWords, output_dim=self.embeddingDim),
            
            Conv1D(filters=128, kernel_size=5, activation='relu'),
            MaxPooling1D(pool_size=2),
            
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            GlobalMaxPooling1D(),
            
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(3, activation='softmax')
        ])
        
        self.cnn_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.cnn_model.summary()
        
        return self.cnn_model
    
    def buildLSTMModel(self):

        print("\nBuilding LSTM model..")
        
        self.lstm_model = Sequential([
            Embedding(input_dim=self.maxTopFrequentWords, output_dim=self.embeddingDim),
            
            SpatialDropout1D(0.2),
            
            Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
            
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(3, activation='softmax')
        ])
        
        self.lstm_model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.lstm_model.summary()
        
        return self.lstm_model
    
    def trainModels(self, epochs=30, batch_size=128):

        if self.X_train is None or self.y_train is None:
            print("Error, data not preprocessed.")
            return None
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        if not os.path.exists('models'):
            os.makedirs('models')
        
        #train CNN model
        print("\nTraining CNN model..")
        cnn_checkpoint = ModelCheckpoint(
            'models/cnn_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
        
        cnn_history = self.cnn_model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping, cnn_checkpoint],
            verbose=1
        )
        
        #train LSTM model
        print("\nTraining LSTM model...")
        lstm_checkpoint = ModelCheckpoint(
            'models/lstm_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
        
        lstm_history = self.lstm_model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping, lstm_checkpoint],
            verbose=1
        )
        
        #save full models
        self.cnn_model.save('models/cnn_model_final.keras')
        self.lstm_model.save('models/lstm_model_final.keras')
        
        return {
            'cnn': cnn_history,
            'lstm': lstm_history
        }
    
    def evaluateModels(self):

        if self.X_test is None or self.y_test is None:
            print("error, data not preprocessed.")
            return None
        
        #load best models
        cnn_model = load_model('models/cnn_model.keras')
        lstm_model = load_model('models/lstm_model.keras')
        
        print("\nEvaluating CNN model..")
        cnn_loss, cnn_accuracy = cnn_model.evaluate(self.X_test, self.y_test, verbose=1)
        
        print("\nEvaluating LSTM model..")
        lstm_loss, lstm_accuracy = lstm_model.evaluate(self.X_test, self.y_test, verbose=1)
        
        cnn_pred_probs = cnn_model.predict(self.X_test)
        cnn_predictions = np.argmax(cnn_pred_probs, axis=1)
        
        lstm_pred_probs = lstm_model.predict(self.X_test)
        lstm_predictions = np.argmax(lstm_pred_probs, axis=1)
        
        #calc metrics
        cnn_precision = precision_score(self.y_test, cnn_predictions, average='weighted')
        cnn_recall = recall_score(self.y_test, cnn_predictions, average='weighted')
        cnn_f1 = f1_score(self.y_test, cnn_predictions, average='weighted')
        
        lstm_precision = precision_score(self.y_test, lstm_predictions, average='weighted')
        lstm_recall = recall_score(self.y_test, lstm_predictions, average='weighted')
        lstm_f1 = f1_score(self.y_test, lstm_predictions, average='weighted')
        
        classes = [0, 1, 2] # 0 negative, 1 neutral, 2 positive
        y_test_bin = label_binarize(self.y_test, classes=classes)

        plt.figure(figsize=(10, 8))

        for i, class_name in enumerate(['Negative', 'Neutral', 'Positive']):
            # CNN ROC
            cnn_fpr, cnn_tpr, _ = roc_curve(y_test_bin[:, i], cnn_pred_probs[:, i])
            cnn_auc = auc(cnn_fpr, cnn_tpr)
            plt.plot(cnn_fpr, cnn_tpr, lw=2, label=f'CNN - {class_name} (area = {cnn_auc:.3f})')
            
            # LSTM ROC
            lstm_fpr, lstm_tpr, _ = roc_curve(y_test_bin[:, i], lstm_pred_probs[:, i])
            lstm_auc = auc(lstm_fpr, lstm_tpr)
            plt.plot(lstm_fpr, lstm_tpr, lw=2, linestyle='--', label=f'LSTM - {class_name} (area = {lstm_auc:.3f})')
        
        #evaluation results
        print("\n========== CNN Model Evaluation ==========")
        print(f"Accuracy: {cnn_accuracy:.4f}")
        print(f"Precision: {cnn_precision:.4f}")
        print(f"Recall: {cnn_recall:.4f}")
        print(f"F1 Score: {cnn_f1:.4f}")
        print(f"AUC: {cnn_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, cnn_predictions))
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, cnn_predictions))
        
        print("\n========== LSTM Model Evaluation ==========")
        print(f"Accuracy: {lstm_accuracy:.4f}")
        print(f"Precision: {lstm_precision:.4f}")
        print(f"Recall: {lstm_recall:.4f}")
        print(f"F1 Score: {lstm_f1:.4f}")
        print(f"AUC: {lstm_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, lstm_predictions))
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, lstm_predictions))
        
        #plot ROC curves
        plt.figure(figsize=(10, 8))
        plt.plot(cnn_fpr, cnn_tpr, lw=2, label=f'CNN ROC (area = {cnn_auc:.3f})')
        plt.plot(lstm_fpr, lstm_tpr, lw=2, label=f'LSTM ROC (area = {lstm_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.savefig('roc_curves.png')
        plt.close()
        
        #select best model based on accuracy
        if cnn_accuracy > lstm_accuracy:
            self.best_model = cnn_model
            self.best_model_name = "CNN"
            print("\nCNN model performs better and will be selected as the final model.")
        else:
            self.best_model = lstm_model
            self.best_model_name = "LSTM"
            print("\nLSTM model performs better and will be selected as the final model.")
        
        self.best_model.save(f'models/best_model_{self.best_model_name}.keras')
        
        return {
            'cnn': {
                'accuracy': cnn_accuracy,
                'precision': cnn_precision,
                'recall': cnn_recall,
                'f1': cnn_f1,
                'auc': cnn_auc,
                'predictions': cnn_predictions
            },
            'lstm': {
                'accuracy': lstm_accuracy,
                'precision': lstm_precision,
                'recall': lstm_recall,
                'f1': lstm_f1,
                'auc': lstm_auc,
                'predictions': lstm_predictions
            },
            'best_model': self.best_model_name
        }
    
    def visualiseTraining(self, histories):

        if histories is None:
            print("no training histories provided.")
            return
        
        plt.figure(figsize=(12, 10))
        
        #plot accuracy
        plt.subplot(2, 2, 1)
        plt.plot(histories['cnn'].history['accuracy'], label='CNN Training')
        plt.plot(histories['cnn'].history['val_accuracy'], label='CNN Validation')
        plt.title('CNN Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(histories['lstm'].history['accuracy'], label='LSTM Training')
        plt.plot(histories['lstm'].history['val_accuracy'], label='LSTM Validation')
        plt.title('LSTM Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        #plot loss
        plt.subplot(2, 2, 3)
        plt.plot(histories['cnn'].history['loss'], label='CNN Training')
        plt.plot(histories['cnn'].history['val_loss'], label='CNN Validation')
        plt.title('CNN Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.plot(histories['lstm'].history['loss'], label='LSTM Training')
        plt.plot(histories['lstm'].history['val_loss'], label='LSTM Validation')
        plt.title('LSTM Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
    
    def bestModelPredictions(self, new_reviews):

        if self.best_model is None:
            print("Error Selecting best model.")
            return None
        
        if self.tokenizer is None:
            print("Tokenizer not found. Load it from saved file.")
            with open('tokenizer.pickle', 'rb') as handle:
                self.tokenizer = pickle.load(handle)
        
        sequences = self.tokenizer.texts_to_sequences(new_reviews)
        padded_sequences = pad_sequences(sequences, maxlen=self.maxLen, padding='post', truncating='post')
        
        pred_probs = self.best_model.predict(padded_sequences)
        predictions = (np.argmax(pred_probs, axis=1))
        
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        results_df = pd.DataFrame({
            'Review': new_reviews,
            'Predicted_Sentiment': predictions,
            'Sentiment_Label': [sentiment_labels[pred] for pred in predictions]
        })
        
        return results_df

if __name__ == "__main__":

    dataPath = 'tripadvisor_hotel_reviews.csv'
    
    classifier = reviewSentimentClassifier(dataPath)
    
    classifier.loadData()
    classifier.analyseData()

    classifier.preprocessData()

    classifier.buildCNNModel()
    classifier.buildLSTMModel()
    
    histories = classifier.trainModels(epochs=30, batch_size=128)
    
    classifier.visualiseTraining(histories)
    
    evaluation_results = classifier.evaluateModels()
    
    bestModelReviewSample = [
        "The hotel was amazing with excellent service and beautiful rooms.",
        "Worst experience ever. Dirty rooms and very rude staff.",
        "Average hotel, nothing special but no major complaints.",
        "Decent hotel, service was okay and staff were nice.",
        "Amazing rooms! will be back for sure",
        "for the money, this hotel was worth it, id reccommend",
        "it was okay, i would say average for this price, my opinion is neutral",
        "Unreal experience, the beach views were stunning.",
        "Very rough, refund was requested! dont reccommend ",
        "Didnt like this at all, i sent an email to complain"
    ]
    
    predictions = classifier.bestModelPredictions(bestModelReviewSample)
    print("\nSample Review Predictions:")
    print(predictions)
    
    print("\nSentiment Analysis completed")