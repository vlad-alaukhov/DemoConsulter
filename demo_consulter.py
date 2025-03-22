from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.label import MDLabel
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.clock import Clock
import datetime
import os
from rag_processor import DBConstructor


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
  - переведи тему к нейросотрудникам и системам RAG,
  - скажи, что вопрос не относится к теме нейросотрудников,
  - задавай наводящие вопросы, если формулировка вопроса пользователя слабо соотносится с
  фрагментом документа,
  - отшутись,
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
        self.consulter = DBConstructor()
        db_folder = "DB_Main_multilingual-e5-large"
        self.receive_db(db_folder)
        #model_name = kwargs["model_name"]
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
            # self.get_response(system_prompt, message)

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

    def receive_db(self, folder):
        data = self.consulter.faiss_loader(folder)
        print(data)


if __name__ == "__main__":
    ChatApp().run()
