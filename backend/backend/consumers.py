from channels.generic.websocket import WebsocketConsumer
import json

class CameraConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        self.send(text_data=json.dumps({
            'type':'connection_established',
            'message':'poggers, connected'}))
    def receive(self, text_data= None):
        print(json.dumps(text_data))
    
    def disconnect(self, close_code):
        pass