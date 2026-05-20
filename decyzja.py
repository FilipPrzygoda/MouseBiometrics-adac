import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from ekstrakcja_cech import BiometricFeatureExtractor


class BiometricDecision:

    def __init__(self, username):
        self.username = username
        self.extractor = BiometricFeatureExtractor(username)
        
        self.models_dir = 'models'
        self.model_path = os.path.join(self.models_dir, f'model_{self.username}.pkl')
        self.scaler_path = os.path.join(self.models_dir, f'scaler_{self.username}.pkl')
        
        self.model = None
        self.scaler = None
        
        self.load_model()
    
    def load_model(self):

        if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
            raise FileNotFoundError(
                f"Brak wytrenowanego modelu dla użytkownika {self.username}. "
                f"Upewnij się, że model został wytrenowany przy użyciu BiometricTrainer."
            )
        
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            print(f"[DECYZJA] Model załadowany dla: {self.username}")
        except Exception as e:
            raise RuntimeError(f"Błąd podczas ładowania modelu: {e}")
    
    def _extract_features_from_session(self, events):

        session_data = {'events': events}
        df_features = self.extractor.prepare_features_for_prediction(session_data)
        
        if df_features.empty:
            return None
        
        return df_features
    
    def predict(self, events):
 
        if self.model is None or self.scaler is None:
            return None
        
        df_features = self._extract_features_from_session(events)
        
        if df_features is None or df_features.empty:
            return None
        
        X_scaled = self.scaler.transform(df_features)
        
        predictions = self.model.predict(X_scaled)
        
        confidence = np.mean(predictions)
        is_correct_user = confidence >= 0.5
        

        return {
            'is_correct_user': bool(is_correct_user),
            'confidence': float(confidence)
        }
    
    def predict_with_details(self, events):

        if self.model is None or self.scaler is None:
            return None
        
        df_features = self._extract_features_from_session(events)
        
        if df_features is None or df_features.empty:
            return None
        
        X_scaled = self.scaler.transform(df_features)
        predictions = self.model.predict(X_scaled)
        
        confidence = np.mean(predictions)
        is_correct_user = confidence >= 0.5
        
        
        return {
            'is_correct_user': bool(is_correct_user),
            'confidence': float(confidence),
            'num_movements': len(predictions),
            'event_predictions': [int(p) for p in predictions]
        }
