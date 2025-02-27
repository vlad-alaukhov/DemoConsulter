from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.label import MDLabel
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.clock import Clock
from dotenv import load_dotenv
import datetime
import requests
import os
from rag_processor import DBConstructor

os.environ.clear()
load_dotenv(".venv/.env")

# получим переменные окружения из .env

# API-key
api_key = os.environ.get("OPENAI_API_KEY")

KV = '''
BoxLayout:
    orientation: 'vertical'

    ScrollView:
        MDBoxLayout:
            id: chat_layout
            orientation: 'vertical'
            size_hint_y: None
            height: self.minimum_height  # Устанавливаем высоту равной минимальной высоте содержимого
            padding: 10
            spacing: 10

    MDBoxLayout:
        size_hint_y: None
        height: '60dp'  # Установите фиксированную высоту для нового Horizontal BoxLayout
        padding: 10
        spacing: 10

        MDTextField:
            id: user_input
            hint_text: "Введите сообщение"
            mode: "rectangle"
            multiline: False
            on_text_validate: app.send_message()

        MDRaisedButton:
            text: "Отправить"
            on_release: app.send_message()

    MDLabel:
        id: status_label
        size_hint_y: None
        height: '30dp'  # Фиксированная высота для строки состояния
        halign: 'center'
        valign: 'middle'
'''
system_prompt1 = '''Ты - нейроконсультант по адаптации Искусственного интеллекта к автоматизированным диалоговым системам.
 Твоя специализация нейросотрудники на базе LLM. У тебя в распоряжении документ по всем доступным на данный момент
 нейросотрудникам. Отвечай на вопросы пользователей лаконично, емко и в строгом соответствии с документом.
 Ответ пиши в информационном стиле:
  Исключить: клише и штампы: газетные, корпоративные, канцелярские, бытовые; указания на настоящее время,
  формализмы, неопределенности, эвфемизмы, вводные слова и выражения и сослагательные наклонения.
  Оценочные выражения дополнить фактами или цифрами, иначе исключить.'''

system_prompt = '''
  Ты - большая языковая модель в роли эксперта в теме RAG (Retrieval-Augmented Generation) - генерации текста большими 
  речевыми моделями (LLM) на основе результатов поиска информации в специализированных базах данных.
  У тебя в распоряжении документ с информацией по всем доступным на данный момент нейросотрудникам.
  Твоя задача - отвечать на вопросы пользователя.
  На те вопросы, которые касаются RAG и нейросотрудников в частности, отвечай исходя из переданной тебе информации
  и своей экспертности, чтобы было понятно и самую суть.
  На все остальные вопросы отвечай по своему усмотрению:
  - ответь на приветствие, если с тобой поздоровались.
  - можешь перевести тему к нейросотрудникам и системам RAG,
  - можешь сказать, что вопрос не относится к теме нейросотрудников,
  - можешь задавать наводящие вопросы, если формулировка вопроса пользователя слабо соотносится с
  фрагментом документа,
  - можешь отшутиться,
  - если попросят представиться, скажи, в чем твоя экспертность.
  Имей в виду, что пользователь может не владеть специальным языком и не ориентироваться в специальных терминах.
  Уровень владения специальным языком считай из формулировки вопроса.
  Ответ пиши в информационном стиле:
  Исключить: клише и штампы: газетные, корпоративные, канцелярские, бытовые; указания на настоящее время,
  формализмы, неопределенности, эвфемизмы, вводные слова и выражения и сослагательные наклонения.
  Оценочные выражения дополнить фактами или цифрами, иначе исключить.
'''

class ChatApp(MDApp):
    def build(self):
        api_base = os.environ.get("OPENAI_URL")
        consulter = DBConstructor()
        db_folder = "OpenAI_DB_3_large"
        #model_name = kwargs["model_name"]
        self.db = consulter.db_loader_from_openai(db_folder)
        Clock.schedule_interval(self.update_time, 1)
        return Builder.load_string(KV)

    def send_message(self):
        user_input = self.root.ids.user_input
        user_query = user_input.text
        verbose = False
        if user_query:
            found_chunks = self.db.similarity_search(user_query, k=3)
            context = ''.join([f"Фрагмент {n}:\n{chunk.page_content}\n" for n, chunk in enumerate(found_chunks)])
            print(found_chunks)
            message = f"""На основании предоставленного тебе материала {context}
                          Ответь на вопрос пользователя: {user_query}"""
            self.add_message("Вы: " + user_query, "user")
            if verbose: self.add_message("Найденные чанки: " + context, "user")
            user_input.text = ""
            self.get_response(system_prompt, message)

    def add_message(self, text, sender="user"):
        bubble = MDBoxLayout(
            orientation="vertical",
            padding="10dp",
            spacing="5dp",
            size_hint_x=0.8,
            adaptive_height=True,
            md_bg_color=(0.9, 0.9, 1, 1) if sender == "user" else (0.9, 1, 0.9, 1),
            radius=[15, 15, 15, 15],
            pos_hint={"right": 1} if sender == "user" else {"left": 1}
        )
        label = MDLabel(
            text=text,
            halign="left",
            theme_text_color="Primary",
            size_hint_y=None,
            padding=(10, 10),
            text_size=(self.root.width * 0.75, None),
            adaptive_height=True
        )
        label.bind(texture_size=label.setter("size"))
        bubble.add_widget(label)
        self.root.ids.chat_layout.add_widget(bubble)

        self.scroll_to_bottom()

        # Автоматическая прокрутка вниз после добавления нового сообщения
        # Clock.schedule_once(self.scroll_to_bottom, 0.1)

    def scroll_to_bottom(self, *args):
        scroll_view = self.root.children[1]  # Получаем ScrollView
        scroll_view.scroll_y = 0  # Прокрутка к нижней части ScrollView

    def update_time(self, dt):
        current_time = datetime.datetime.now().strftime("%H:%M")
        self.root.ids.status_label.text = f"Текущее время: {current_time}"

    def get_response(self, system, message):
        # Формируем данные для POST-запроса

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": message}
            ],
            "temperature": 0.3
        }

        try:
            # Отправляем POST-запрос к модели
            response = requests.post("https://api.vsegpt.ru:6010/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()  # Проверка на наличие ошибок в запросе

            # Получаем ответ от модели
            response_data = response.json()
            bot_message = response_data['choices'][0]['message']['content']

            # Добавляем ответ бота в чат
            self.add_message("ChatGPT: " + bot_message, "ChatGPT")

        except Exception as e:
            print(f"Error occurred: {e}")
            self.add_message(f"Ошибка {e}: Не удалось получить ответ", "ChatGPT")


if __name__ == "__main__":
    ChatApp().run()
