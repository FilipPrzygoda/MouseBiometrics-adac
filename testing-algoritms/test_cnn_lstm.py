import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pymongo import MongoClient
from dotenv import load_dotenv
import random
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ekstrakcja_cech import BiometricFeatureExtractor

class CNNLSTMTester:
    def __init__(self, username, db_uri='mongodb://localhost:27017/', db_name='biometria_db'):
        self.username = username
        self.client = MongoClient(db_uri)
        self.collection = self.client[db_name]['sesje_uzytkownikow']
        self.extractor = BiometricFeatureExtractor(username)
        self.scaler = StandardScaler()
        self.model = None
        self.X_train = self.X_test = self.y_test = self.df_test = None
        
        self.results_dir = 'testing-results'
        os.makedirs(self.results_dir, exist_ok=True)
    
    def _fetch_balanced_sessions(self):
        user_sessions = list(self.collection.find({'username': self.username}, {'_id': 0}))
        other_sessions = list(self.collection.find({'username': {'$ne': self.username}}, {'_id': 0}))
        
        if len(user_sessions) < len(other_sessions):
            random.shuffle(other_sessions)
            other_sessions = other_sessions[:len(user_sessions)]
        elif len(user_sessions) > len(other_sessions):
            random.shuffle(user_sessions)
            user_sessions = user_sessions[:len(other_sessions)]
            
        return user_sessions, other_sessions
    
    def prepare_data(self):
        user_sessions, other_sessions = self._fetch_balanced_sessions()
        if not user_sessions:
            return False
            
        df_model = self.extractor.prepare_dataset_for_training(user_sessions, other_sessions)
        if df_model.empty:
            return False
            
        unique_sessions = df_model['session_id'].unique()
        train_sessions, test_sessions = train_test_split(unique_sessions, test_size=0.3, random_state=99)
        
        df_train = df_model[df_model['session_id'].isin(train_sessions)]
        df_test = df_model[df_model['session_id'].isin(test_sessions)]
        
        cols_to_drop = ['is_target_user', 'session_id']
        X_train = df_train.drop(cols_to_drop, axis=1)
        self.y_test = df_test['is_target_user'].values
        X_test = df_test.drop(cols_to_drop, axis=1)
        self.df_test = df_test.copy()
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
        self.X_test = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
        
        self.y_train_keras = keras.utils.to_categorical(df_train['is_target_user'].values, 2)
        return True
    
    def build_model(self):
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(self.X_train.shape[1], 1)),
            Dropout(0.2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            Dropout(0.2),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(2, activation='softmax')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def train_model(self):
        if self.X_train is None:
            return False
            
        self.model = self.build_model()
        print("\n--- Rozpoczęcie treningu modelu CNN-LSTM ---")
        self.model.fit(self.X_train, self.y_train_keras, epochs=50, batch_size=16, validation_split=0.2, verbose=1)
        train_loss, train_acc = self.model.evaluate(self.X_train, self.y_train_keras, verbose=0)
        print(f"--- Trening zakończony --- Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}\n")
        return True
    
    def predict(self):
        if self.X_test is None or self.model is None:
            return False
            
        y_pred_proba = self.model.predict(self.X_test, verbose=0)
        self.y_pred = (y_pred_proba[:, 1] >= 0.5).astype(int)
        return True
    
    def evaluate_whole_session(self):
        if self.df_test is None or self.y_pred is None:
            return None, None
            
        df_res = self.df_test[['session_id', 'is_target_user']].copy()
        df_res['pred_event'] = self.y_pred
        
        session_grp = df_res.groupby('session_id').agg(
            actual_label=('is_target_user', 'first'),
            pred_mean=('pred_event', 'mean')
        )
        session_pred = (session_grp['pred_mean'] >= 0.5).astype(int)
        
        return session_grp['actual_label'].values, session_pred.values
        
    def generate_confusion_matrix(self):
        if self.y_pred is None:
            return False
            
        self.cm_event = confusion_matrix(self.y_test, self.y_pred, labels=[0, 1])
        y_session_true, y_session_pred = self.evaluate_whole_session()
        
        if y_session_true is not None:
            self.cm_session = confusion_matrix(y_session_true, y_session_pred, labels=[0, 1])
            
        return True
    
    def plot_confusion_matrix(self, filename='confusion_matrices_cnn_lstm.png'):
        if not hasattr(self, 'cm_event'):
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        sns.heatmap(self.cm_event, annot=True, fmt='g', cmap='Blues', ax=axes[0], 
                   xticklabels=['0', '1'], yticklabels=['0', '1'])
        axes[0].set_title('CNN-LSTM - Event')
        
        if hasattr(self, 'cm_session'):
            sns.heatmap(self.cm_session, annot=True, fmt='g', cmap='Greens', ax=axes[1],
                       xticklabels=['0', '1'], yticklabels=['0', '1'])
            axes[1].set_title('CNN-LSTM - Session')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_full_test(self):
        if self.prepare_data() and self.train_model() and self.predict() and self.generate_confusion_matrix():
            self.plot_confusion_matrix()
            tn, fp, fn, tp = self.cm_event.ravel()
            acc = (tp + tn) / (tp + tn + fp + fn)
            print(f"\nRezultaty dla poszczególnych zdarzeń (Event):")
            print(f"Dokładność (ACC): {acc:.4f}")
            print(f"Prawdziwie pozytywne (TP): {tp}, Prawdziwie negatywne (TN): {tn}")
            print(f"Fałszywie pozytywne (FP): {fp}, Fałszywie negatywne (FN): {fn}")
            return True
        print("Error")
        return False

if __name__ == '__main__':
    tester = CNNLSTMTester('filipp', 'mongodb://localhost:27017/', 'biometria_db')
    tester.run_full_test()
