from channels.generic.websocket import WebsocketConsumer
import json
from django.template.loader import get_template

class CameraConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        self.send(text_data=json.dumps({
            'type':'connection_established',
            'message':'poggers, connected'}))
    def receive(self, text_data= None):
        print(json.dumps(text_data))
        
        html = get_template("partials/display.html").render(
            context = {
                'val': 'testing'
            }
        )
        self.send(text_data=html)

    
    def disconnect(self, close_code):
        pass