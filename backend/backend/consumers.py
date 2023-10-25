from channels.generic.websocket import WebsocketConsumer

class CameraConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
    def disconnect(self, close_code):
        pass