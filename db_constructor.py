# pip3 install -U sentence-transformers

import fitz
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter     # рекурсивное разделение текста
from langchain.docstore.document import Document
import tiktoken
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import re                 # работа с регулярными выражениями
import requests
from dotenv import load_dotenv
import time
from langchain_huggingface import HuggingFaceEmbeddings


os.environ.clear()
load_dotenv(".venv/.env")

class DBConstructor:
    def __init__(self):
        self.chunk_size = 700
        self.source_chunks = None
        self.num_tokens = 0
        self.summary = None
        self.db = None
        self.answer = None
        self.unprocessed_text = None
        self.processed_text = None
        # получим переменные окружения из .env
        self.api_url = os.environ.get("OPENAI_URL")
        # API-key
        self.api_key = os.environ.get("OPENAI_API_KEY")
        # HF-токен
        self.hf = os.environ.get("HF-TOKEN")

    @staticmethod
    def pdf_parser(pdf_folder: str, base_file: str):
        """
         Парсит текст из PDF-ок.
         Ищет все PDF-ки в pdf_folder, парсит текст в переменную text, Записывает в текстовый base_file
        :param pdf_folder:
        :param base_file:
        :return:
        """
        all_files = os.listdir(pdf_folder)
        pdf_files = [fn for fn in all_files if fn.endswith('.pdf')]

        for each_pdf in pdf_files:
            try:
                with fitz.open(os.path.join(pdf_folder, each_pdf)) as pdf:
                    text = ""
                    for page in pdf:
                        text += page.get_text()
                    with open(base_file, 'a') as bf:
                        bf.write(text + '\n')
            except Exception as e:
                print(f"Ошибка при обработке файла {each_pdf}: {e}")

    @staticmethod
    def minus_words(file_name: str, pattern: str, to_change: str):
        """ Замена слов в файле. Открывает файл, грузит в переменную, находит pattern, меняет на to_change
        бэкапит файл и переписывает измененный текст в исходный файл.
        :param file_name: имя файла, в котором надо заменить слова
        :param pattern: Что менять
        :param to_change: На что менять
        """
        with open(file_name, 'r') as file:
            text = file.read()

        out_text = re.sub(pattern, to_change, text)
        os.rename(file_name, file_name.split('.')[0] + '.bak')
        with open(file_name, 'w') as fn:
            fn.write(out_text)

    def num_tokens_from_string(self, string: str, encoding_name: str) -> int:
        """Возвращает количество токенов в строке"""
        encoding = tiktoken.get_encoding(encoding_name)
        self.num_tokens = len(encoding.encode(string))
        return self.num_tokens

    def chunk_size_golden_ratio(self, db_markdown: list):
        """Максимальная величина чанка по условию "Золотого сечения. Если документ разметился маркдауном
        неравномерно так, что есть редкие чанки большой длины, с отрывом больше гармонического среднего
        (золотого сечения) от основной массы, то за максимальное значение чанка принимается максимальная
        величина чанков основной массы.
        :param db_markdown: Список из Langchain документов, размеченных Markdown разметкой.
        :return:
        """
        # Считаю токены функцией num_tokens_from_string
        tok_cnts = [self.num_tokens_from_string(fragment.page_content, "cl100k_base")
                    for fragment in db_markdown]
        # Беру максимальное количество токенов, первый элемент сортированного списка по убыванию
        max_val = sorted(tok_cnts, reverse=True)[0]
        ind = 0  # Ввожу индекс и обнуляю его.
        leng = len(tok_cnts)  # Общее количество фрагментов
        # Прохожу по списку значений токенов по убыванию
        for num in sorted(tok_cnts, reverse=True)[ind:]:
            print(num * ((1 + 5 ** 0.5) / 2), max_val)
            # Ищу следующее значение токенов с отрывом от максимального не меньше, чем "Золотое сечение".
            # Если значение следующего по величине отстоит от предыдущего более, чем на 1,618:
            if num * ((1 + 5 ** 0.5) / 2) <= max_val:
                # Вычисляю длину оставшегося списка для того и запоминаю, чтобы потом взять обратный индекс
                leng = len(sorted(tok_cnts, reverse=True)[ind:])
                max_val = num  # Беру максимальное значение
                ind += 1  # Наращиваю индекс, перехожу к следующему
            # Если длины токенов идут плотненько, без разрывов в 1,618 до 700, пропускаю до следующего разрыва
            elif max_val > 700:
                max_val = num  # Беру за максимальное значение текущее
                ind += 1  # Наращиваю индекс, перехожу к следующему значению
            else:
                # Ближайшее максимальное значение числа токенов основной массы будет отстоять
                # от конца сортированного списка на величину leng
                self.chunk_size = sorted(tok_cnts, reverse=True)[-leng]
                print("Окончательный размер чанка = ", self.chunk_size)
                return self.chunk_size

    def split_text_recursive(self, text: str, chunk_size: int):
        """
        Делит текст в строковой переменной text методом RecursiveCharacterTextSplitter
        на чанки размером chunk_size
        :param text: Текст в строке. Чтобы дополнительно разделить langchain-документ, надо подавать page_content
        :param chunk_size: Размер чанка.
        :return: Список чанков типа str
        """
        splitter = RecursiveCharacterTextSplitter(
            separators=['\n', '\n\n', '. '],
            chunk_size=chunk_size,
            chunk_overlap=0
        )

        self.source_chunks = splitter.split_text(text)
        return self.source_chunks

    def split_recursive_from_markdown(self, documents_to_split: list, chunk_size: int) -> list:
        """ Делит список Langchain документов documents_to_split на чанки размером chunk_size 
        методом RecursiveCharacterTextSplitter. Для этого вызывается метод split_text_recursive,
        реализованный в этом же классе.
        :param documents_to_split: Текст базы
        :param chunk_size: Размер чанка
        :return: список Langchain документов
        """
        source_chunks = []

        for each_document in documents_to_split:
            new_chunks = self.split_text_recursive(each_document.page_content, chunk_size)
            for each_element in new_chunks:
                for key in each_document.metadata:
                    each_element += each_document.metadata[key]
                source_chunks.append(Document(page_content=each_element, metadata=each_document.metadata))

        return source_chunks

    @staticmethod
    def split_markdown(db_text: str, hd_level=1):
    # MarkDownHeader разметка базы из файла с дублированием заголовка в текст чанка

        headers_to_split_on = [(f"{'#' * n}", f"H{n}") for n in range(1, hd_level+1 if hd_level >= 1 else 1)]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        fragments = markdown_splitter.split_text(db_text)

        for fragment in fragments:
            header = ''
            for key in fragment.metadata:
                header += f"{fragment.metadata[key]}. "
            fragment.page_content = header + fragment.page_content

        return fragments

    # Подсчет токенов
    @staticmethod
    def num_tokens_from_messages(messages, model='gpt-4o-mini'):
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding('cl100k_base')

        if model in ['gpt-4o-mini', 'gpt-4o', 'gpt-4o-latest']:
            num_tokens = 0

            for message in messages:
                num_tokens += 4

                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))

                    if key == 'name':
                        num_tokens -= 1

            num_tokens += 2
            return num_tokens

        else:
            raise NotImplementedError(f'''num_tokens_from_messages() is not presently implemented for model {model}.''')

    def request_to_openai(self, system: str, request: str, temper: float):
        result_text = ''
        attempts = 1

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "openai/gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": request}
            ],
            "temperature": temper
        }
        while attempts < 3:
            time.sleep(3)
            try:
                # Отправляем POST-запрос к модели
                response = requests.post("https://api.vsegpt.ru:6070/v1/chat/completions", headers=headers, json=payload)
                response.raise_for_status()  # Проверка на наличие ошибок в запросе

                # Получаем ответ от модели
                response_data = response.json()
                result_text += response_data['choices'][0]['message']['content']
                time.sleep(3)
                return result_text

            except Exception as e:
                print(f"Error occurred: {e}")
                attempts += 1
                time.sleep(3)


    def db_pre_constructor(self, file_path: str, system: str, user: str, chunk_size=10000):
        '''
        Метод для предварительной обработки базы. Открывается файл неразмеченной базы и размечается по крупным
        разделам путем деления на чанки RecursiveCharacterTextSplitter из метода
        self.split_text_recursive(text, chunk_size).
        Разделы размечаются промптом "Крупные разделы" из файла prompts.yaml
        :param file_path: Путь к базе
        :param system: Системный промпт system = prompts['Крупные разделы']['system']
        :param user: Юзер-промпт user = prompts['Крупные разделы']['user']
        :param chunk_size: Размер чанков. По умолчанию 10000, чтобы влезли в модель
        :return:
        '''
        self.processed_text = ''
        self.unprocessed_text = ''

        # Открываю файл сырой базы и читаю в переменную
        with open(file_path, 'r') as txt_file:
            text = txt_file.read()

        # делю на чанки
        source_chunks = self.split_text_recursive(text, chunk_size)

        for num, chunk in enumerate(source_chunks):
            request = f"{user}\n{chunk}"
            self.answer = self.request_to_openai(system, request, 0)
            self.processed_text += f'{self.answer}\n\n'  # Добавляем ответ в результат
            print(f'Чанк №{num}\n{self.answer}')  # Выводим ответ

        return self.processed_text

    def db_constructor(self, fragments: list, system: str, user: str):
        result_text = ''

        for fragment in fragments:
            request = f"{user}\n{fragment.page_content}"
            answer = self.request_to_openai(system, request, 0)
            result_text += f"{answer}\n\n"
            time.sleep(7)

        return result_text

    def db_tester(self, db_markdown_text: list, system: str, user: str, verbose=False):

        questionnairy = []

        for chunk in db_markdown_text:
            if verbose:
                print(user + chunk.page_content)
                print('---------------------------------------------------------------------')
            request = f"{user}\n{chunk.page_content}"
            self.answer = self.request_to_openai(system, request, 0.5)
            if verbose:
                print(f"Вопросы от модели:\n{self.answer}")
                print('---------------------------------------------------------------------')
            questionnairy.append(self.answer)
            time.sleep(2.8)
        test_results = ''.join(questionnairy)
        return test_results

    def quest_handler(self, quest_file: str, system: str, user: str):
        # Метод для обработки пула тестовых вопросов по базе
        # и составления сводки по недостающей информации
        with open(quest_file, 'r') as qf:
            pull_questions = qf.read()
        # Делаю из пула вопросов langchain документ, чтобы подать в request_openai
        pull_questions = Document(page_content=pull_questions, metadata={'pull': 'questions'})
        # Получаю сводку
        request = f"{user}\n{pull_questions}"
        self.summary = self.request_to_openai(system, request, 0)

        return self.summary

    def vectorizator_openai(self, docs: list, db_folder: str):
        results = {"has_vectorized": "База знаний векторизована. ",
                   "has_loaded": "База знаний загружена ",
                   "not_vectorized": "Ошибка векторизации. ",
                   "not_loaded": "Ошибка загрузки. "}
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key = self.api_key,
            openai_api_base = self.api_url)
        # embeddings.openai_api_key = self.api_key
        self.db = FAISS.from_documents(docs, embeddings)
        if self.db:
            res = results["has_vectorized"]
            self.db.save_local(db_folder)
            if os.path.exists(db_folder):
                res += f"{results['has_loaded']} в {db_folder}"
            else:
                res = results["not_loaded"]
        else:
            res = results["not_vectorized"]
            return None
        print(res)
        return self.db

    def vectorizator_sota(self, docs: list, db_folder: str, model_name: str):
        results = {"has_vectorized": "База знаний векторизована. ",
                   "has_loaded": "База знаний загружена ",
                   "not_vectorized": "Ошибка векторизации. ",
                   "not_loaded": "Ошибка загрузки. "}
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs)
        db = FAISS.from_documents(docs, embeddings)
        if db:
            res = results["has_vectorized"]
            db.save_local(db_folder)
            if os.path.exists(db_folder): res += f"{results['has_loaded']} в {db_folder}"
            else: res = results["not_loaded"]  
        else:
            res = results["not_vectorized"]
            return None
        print(res)
        return db

    @staticmethod
    def db_loader_from_sota(db_folder: str, model_name: str):
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs)
        db = FAISS.load_local(db_folder, embeddings, allow_dangerous_deserialization=True)
        return db

    def db_loader_from_openai(self, db_folder):
        embs = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=self.api_key,
            openai_api_base=self.api_url)
        self.db = FAISS.load_local(
            folder_path=db_folder,
            embeddings=embs,
            allow_dangerous_deserialization=True
        )
        return self.db