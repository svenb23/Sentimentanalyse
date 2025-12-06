import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer


class TextPreprocessor:

    def __init__(self, config: dict):
        self.config = config.get('preprocessing', {})

        self.lowercase = self.config.get('lowercase', True)
        self.remove_html = self.config.get('remove_html', True)
        self.remove_urls = self.config.get('remove_urls', True)
        self.remove_numbers = self.config.get('remove_numbers', True)
        self.remove_punctuation = self.config.get('remove_punctuation', True)
        self.remove_stopwords = self.config.get('remove_stopwords', True)
        self.use_stemming = self.config.get('stemming', False)
        self.use_lemmatization = self.config.get('lemmatization', True)
        self.min_token_length = self.config.get('min_token_length', 2)

        self._download_nltk_resources()

        if self.remove_stopwords:
            self.stop_words = set(stopwords.words('english'))
        else:
            self.stop_words = set()

        if self.use_stemming:
            self.stemmer = PorterStemmer()

        if self.use_lemmatization:
            self.lemmatizer = WordNetLemmatizer()

    def _download_nltk_resources(self):
        resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
        for resource in resources:
            try:
                if 'punkt' in resource:
                    nltk.data.find(f'tokenizers/{resource}')
                else:
                    nltk.data.find(f'corpora/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        if self.lowercase:
            text = text.lower()

        if self.remove_html:
            text = re.sub(r'<[^>]+>', '', text)

        if self.remove_urls:
            text = re.sub(r'http\S+|www\S+', '', text)

        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)

        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))

        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def process_tokens(self, tokens: list) -> list:
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]

        if self.use_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]
        elif self.use_lemmatization:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]

        if self.min_token_length > 0:
            tokens = [t for t in tokens if len(t) >= self.min_token_length]

        return tokens

    def preprocess(self, text: str) -> str:
        text = self.clean_text(text)
        tokens = word_tokenize(text)
        tokens = self.process_tokens(tokens)
        return ' '.join(tokens)

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['combined_text'] = df.apply(
            lambda row: f"{row['title'] if isinstance(row['title'], str) else ''} "
                       f"{row['text'] if isinstance(row['text'], str) else ''}".strip(),
            axis=1
        )
        df['processed_text'] = df['combined_text'].apply(self.preprocess)
        return df

    def get_config_summary(self) -> dict:
        return {
            'lowercase': self.lowercase,
            'remove_html': self.remove_html,
            'remove_urls': self.remove_urls,
            'remove_numbers': self.remove_numbers,
            'remove_punctuation': self.remove_punctuation,
            'remove_stopwords': self.remove_stopwords,
            'stemming': self.use_stemming,
            'lemmatization': self.use_lemmatization,
            'min_token_length': self.min_token_length
        }
