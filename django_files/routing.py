from channels.routing import ProtocolTypeRouter
from channels.generic.websocket import WebsocketConsumer
import json
from django.conf.urls import url
from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from vannorman.deepq import DQN
my_dqn = DQN([7], 8)

class ChatConsumer(WebsocketConsumer):
    def connect(self):
        
        self.accept()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']
        
        self.send(text_data=json.dumps({
            'message': message
        }))
        print("Calling dQN with:"+str(message))
        # updated_model = my_dqn.store_and_train(json.loads(message)) 
        updated_model = my_dqn.store_and_train(message)
        print("update dmodel:"+str(updated_model))
        # send back to JS the updated model


websocket_urlpatterns = [
    url(r'^ws/msg/$', ChatConsumer),
]

application = ProtocolTypeRouter({
    # (http->django views is added by default)
    'websocket': AuthMiddlewareStack(
        URLRouter(
            websocket_urlpatterns
        )
    ),
})



