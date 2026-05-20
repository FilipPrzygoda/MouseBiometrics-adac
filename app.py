import os
from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, url_for, jsonify
from flask_socketio import SocketIO
from pymongo import MongoClient
import time
from trening import BiometricTrainer
from decyzja import BiometricDecision

load_dotenv()

app = Flask(__name__)

socketio = SocketIO(app)

MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
client = MongoClient(MONGO_URI)
db = client['biometria_db']
collection = db['sesje_uzytkownikow']

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    username = request.cookies.get('user_id')
    if not username:
        return redirect(url_for('login'))  
    return render_template('index.html', username=username)

@app.route('/logout')
def logout():
    response = redirect(url_for('login'))
    response.set_cookie('user_id', '', expires=0)
    return response

@app.route('/api/biometrics', methods=['POST'])
def save_biometrics():
    username = request.cookies.get('user_id')
    if not username:
        return jsonify({'status': 'error', 'message': 'Brak autoryzacji'}), 401
    
    data = request.json
    if data:
        record = {
            'username': username,
            'events': data.get('events', [])
        }
        collection.insert_one(record)
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Puste dane'}), 400

@app.route('/recognition')
def recognition():
    username = request.cookies.get('user_id')
    if not username:
        return redirect(url_for('login'))
    return render_template('recognition.html',username=username)

@app.route('/api/recognize', methods=['POST'])
def recognize_biometrics():

    data = request.json
    if data:
        username = request.cookies.get('user_id')
        
        if not username:
            return jsonify({'status': 'error', 'message': 'Brak autoryzacji'}), 401
        
        try:
            decision_model = BiometricDecision(username)
            
            result = decision_model.predict(data.get('events', []))
            
            if result is None:
                return jsonify({'status': 'error', 'message': 'Brak wystarczających danych w przesłanej sesji'}), 400

            print(f"Autoryzacja ({username}): {result['is_correct_user']} ({result['confidence']:.2f})")
            
            return jsonify({
                'status': 'success',
                'recognized_user': username,
                'is_correct': result['is_correct_user'],
                'confidence': result['confidence']
            })
        except FileNotFoundError as e:
            print(f"Brak modelu: {e}")
            return jsonify({'status': 'error', 'message': str(e)}), 400
        except Exception as e:
            print(f"Błąd rozpoznawania: {e}")
            return jsonify({'status': 'error', 'message': f'Błąd predykcji: {str(e)}'}), 500

    return jsonify({'status': 'error', 'message': 'Puste dane'}), 400

@app.route('/api/train', methods=['POST'])
def train_model():
    username = request.cookies.get('user_id')
    if not username:
        return jsonify({'status': 'error', 'message': 'Brak autoryzacji'}), 401
        
    try:
        print(f"Rozpoczynam trening modelu dla: {username}")
        
        trainer = BiometricTrainer(username, db_uri=MONGO_URI)
        success = trainer.train()
        
        if success:
            print(f"Trening modelu ({username}) zakończony sukcesem")
            return jsonify({
                'status': 'success',
                'message': 'Model został pomyślnie wytrenowany i zapisany.'
            })
        else:
            print(f"Niepowodzenie treningu ({username}) - brak wystarczających danych")
            return jsonify({
                'status': 'error',
                'message': 'Za mało zgromadzonych danych aby wytrenować model.'
            }), 400
    except Exception as e:
        print(f"Błąd podczas treningu: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Błąd treningu: {str(e)}'
        }), 500

@app.route('/api/model-status', methods=['GET'])
def model_status():
    username = request.cookies.get('user_id')
    if not username:
        return jsonify({'status': 'error', 'message': 'Brak autoryzacji'}), 401
    
    import os
    models_dir = 'models'
    model_path = os.path.join(models_dir, f'model_{username}.pkl')
    is_trained = os.path.exists(model_path)
    
    return jsonify({
        'status': 'success',
        'username': username,
        'model_trained': is_trained
    })

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)