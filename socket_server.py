import eventlet
eventlet.monkey_patch()
from flask import Flask
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

@socketio.on('connect')
def on_connect():
    print("Client has connected.")
    emit('server_message', {'message': 'Connected to the server.'})

@socketio.on('detection_result')
def handle_detection(data):
    # print("Detection results received.:", data)
    emit('detection_result', data, broadcast=True)

@app.route('/')
def index():
    return app.send_static_file('websocket_client.html')

if __name__ == '__main__':
    print("WebSocket server is running (http://<server IP>:5000)")
    socketio.run(app, host='0.0.0.0', port=5000)
