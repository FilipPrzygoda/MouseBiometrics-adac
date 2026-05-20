import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from pymongo import MongoClient
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ekstrakcja_cech import BiometricFeatureExtractor


class RandomForestTester:
    def __init__(self, username, db_uri='mongodb://localhost:27017/', db_name='biometria_db'):
        self.username = username
        self.client = MongoClient(db_uri)
        self.collection = self.client[db_name]['sesje_uzytkownikow']
        self.extractor = BiometricFeatureExtractor(username)
        self.model = RandomForestClassifier(
            n_estimators=200, 
            max_depth=15,
            min_samples_leaf=2,
            class_weight={0: 2.5, 1: 1},
            random_state=99
        )
        self.scaler = StandardScaler()
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
        self.y_train = df_train['is_target_user'].values
        self.y_test = df_test['is_target_user'].values
        X_test = df_test.drop(cols_to_drop, axis=1)
        self.df_test = df_test.copy()
        
        self.X_train = self.scaler.fit_transform(X_train)
        self.X_test = self.scaler.transform(X_test)
        return True
    
    def train_model(self):
        if self.X_train is None:
            return False
        
        print("\n--- Rozpoczęcie treningu modelu Random Forest ---")
        self.model.fit(self.X_train, self.y_train)
        
        train_acc = self.model.score(self.X_train, self.y_train)
        print(f"--- Trening zakończony --- Accuracy na zbiorze treningowym: {train_acc:.4f}\n")
        return True
    
    def predict(self):
        if self.X_test is None:
            return False
            
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        self.y_pred = (y_pred_proba >= 0.5).astype(int)
        
        return True
    
    def calculate_metrics(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        
        return cm, far, frr, acc, (tn, fp, fn, tp)
    
    def evaluate_whole_session(self):
        if self.df_test is None or self.y_pred is None:
            return None, None
        
        df_res = self.df_test[['session_id', 'is_target_user']].copy()
        df_res['pred_event'] = self.y_pred
        
        session_grp = df_res.groupby('session_id').agg(
            actual_label=('is_target_user', 'first'),
            pred_mean=('pred_event', 'mean')
        )
        session_grp['session_pred'] = (session_grp['pred_mean'] >= 0.5).astype(int)
        
        return session_grp['actual_label'].values, session_grp['session_pred'].values
    
    def generate_confusion_matrix(self):
        if self.y_pred is None:
            return False
            
        self.cm_event = confusion_matrix(self.y_test, self.y_pred, labels=[0, 1])
        y_session_true, y_session_pred = self.evaluate_whole_session()
        
        if y_session_true is not None:
            self.cm_session = confusion_matrix(y_session_true, y_session_pred, labels=[0, 1])
            
        return True
    
    def plot_confusion_matrix(self, filename='confusion_matrices_rf.png'):
        if not hasattr(self, 'cm_event'):
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        sns.heatmap(self.cm_event, annot=True, fmt='g', cmap='Blues', ax=axes[0], 
                   xticklabels=['0', '1'], yticklabels=['0', '1'])
        axes[0].set_title('Random Forest - Event')
        
        if hasattr(self, 'cm_session'):
            sns.heatmap(self.cm_session, annot=True, fmt='g', cmap='Greens', ax=axes[1],
                       xticklabels=['0', '1'], yticklabels=['0', '1'])
            axes[1].set_title('Random Forest - Session')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_full_test(self):
        if self.prepare_data() and self.train_model() and self.predict() and self.generate_confusion_matrix():
            self.plot_confusion_matrix()
            tn, fp, fn, tp = self.cm_event.ravel()
            acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            print(f"\nRezultaty dla poszczególnych zdarzeń (Event):")
            print(f"Dokładność (ACC): {acc:.4f}")
            print(f"Prawdziwie pozytywne (TP): {tp}, Prawdziwie negatywne (TN): {tn}")
            print(f"Fałszywie pozytywne (FP): {fp}, Fałszywie negatywne (FN): {fn}")
            return True
        print("Error")
        return False

if __name__ == '__main__':
    tester = RandomForestTester('filipp', 'mongodb://localhost:27017/', 'biometria_db')
    tester.run_full_test()
