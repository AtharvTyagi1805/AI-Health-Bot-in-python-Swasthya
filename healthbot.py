import json
import random
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.graphics import Color, Rectangle
from kivy.uix.gridlayout import GridLayout
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

Window.clearcolor = (0.95, 0.95, 0.95, 1)
Window.size = (500, 700)

class ChatMessage(Label):
    def __init__(self, text, is_user=False, **kwargs):
        super().__init__(**kwargs)
        self.markup = True
        self.text = f'[color={"333333" if is_user else "006699"}]{text}[/color]'
        self.font_size = dp(16)
        self.bold = is_user
        self.halign = 'right' if is_user else 'left'
        self.valign = 'top'
        self.size_hint = (None, None)
        self.text_size = (Window.width * 0.8, None)
        self.padding = (dp(15), dp(10))
        self.bind(texture_size=self._adjust_height)
        with self.canvas.before:
            Color(rgba=(0.9,0.9,0.9,1) if is_user else (0.8,0.9,1,1))
            self.rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._update_rect, size=self._update_rect)

    def _adjust_height(self, instance, size):
        self.height = max(size[1] + dp(20), dp(40))
        self.size = (Window.width * 0.9, self.height)

    def _update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

class ChatGUI(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = dp(10)
        self.padding = dp(10)

        self.chat_scroll = ScrollView(do_scroll_x=False)
        self.chat_history = GridLayout(cols=1, spacing=dp(10),
                                      size_hint_y=None, padding=dp(10))
        self.chat_history.bind(minimum_height=self.chat_history.setter('height'))
        self.chat_scroll.add_widget(self.chat_history)

        input_layout = BoxLayout(size_hint_y=None, height=dp(50), spacing=dp(5))
        self.input_field = TextInput(hint_text='Type your message...',
                                     multiline=False,
                                     background_color=(1,1,1,1),
                                     foreground_color=(0,0,0,1))
        send_btn = Button(text='Send', size_hint_x=None, width=dp(80),
                          background_color=(0.2,0.6,0.8,1), color=(1,1,1,1))
        send_btn.bind(on_press=self.send_message)
        input_layout.add_widget(self.input_field)
        input_layout.add_widget(send_btn)

        self.add_widget(Label(text='Swasthya', bold=True,
                              color=(0.2,0.4,0.6,1),
                              size_hint_y=None, height=dp(30)))
        self.add_widget(self.chat_scroll)
        self.add_widget(input_layout)

        self._load_and_initialize_centroids()

    def _load_and_initialize_centroids(self):
        with open('intents.json') as f:
            data = json.load(f)
        self.intents = data['intents']

        patterns = []
        tags = []
        for intent in self.intents:
            for p in intent['patterns']:
                patterns.append(p)
                tags.append(intent['tag'])

        self.vectorizer = TfidfVectorizer(lowercase=True,
                                          token_pattern=r'\b\w+\b',
                                          stop_words='english')
        X = self.vectorizer.fit_transform(patterns).toarray()

        self.centroids = {}   
        self.counts    = {}   
        for tag in set(tags):
            # get all rows with this tag
            rows = X[[i for i,t in enumerate(tags) if t==tag]]
            self.centroids[tag] = rows.mean(axis=0)
            self.counts[tag]    = rows.shape[0]

    def send_message(self, instance):
        text = self.input_field.text.strip()
        if not text:
            return
        self.chat_history.add_widget(ChatMessage(f"You: {text}", is_user=True))
        self.input_field.text = ''

        v = self.vectorizer.transform([text]).toarray()[0]

        sims = { tag: cosine_similarity([v],[c])[0,0]
                 for tag,c in self.centroids.items() }
        best_tag, best_sim = max(sims.items(), key=lambda kv: kv[1])

        if best_sim < 0.25:
            pred_text = f"Prediction: None (conf: {best_sim:.2f})"
            bot_resp  = "I'm sorry, I don't understand. Can you rephrase?"
        else:
            pred_text = f"Prediction: {best_tag} (conf: {best_sim:.2f})"
            for it in self.intents:
                if it['tag']==best_tag:
                    bot_resp = random.choice(it['responses'])
                    break

            count = self.counts[best_tag]
            new_centroid = (self.centroids[best_tag]*count + v)/(count+1)
            self.centroids[best_tag] = new_centroid
            self.counts[best_tag] += 1

        self.chat_history.add_widget(ChatMessage(pred_text))
        self.chat_history.add_widget(ChatMessage(f"Bot: {bot_resp}"))

        self.chat_scroll.scroll_to(self.chat_history.children[0])

class ChatbotApp(App):
    def build(self):
        return ChatGUI()

if __name__ == '__main__':
    ChatbotApp().run()
