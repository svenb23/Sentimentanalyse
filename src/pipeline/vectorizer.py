from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class FeatureVectorizer:

    VECTORIZER_TYPES = {
        'tfidf': TfidfVectorizer,
        'count': CountVectorizer
    }

    def __init__(self, config: dict):
        self.config = config.get('features', {})

        self.vectorizer_type = self.config.get('type', 'tfidf')
        self.max_features = self.config.get('max_features', 10000)
        self.ngram_range = tuple(self.config.get('ngram_range', [1, 2]))
        self.min_df = self.config.get('min_df', 2)
        self.max_df = self.config.get('max_df', 0.95)
        self.sublinear_tf = self.config.get('sublinear_tf', False)
        self.use_idf = self.config.get('use_idf', True)

        self.vectorizer = self._create_vectorizer()
        self.is_fitted = False

    def _create_vectorizer(self):
        vectorizer_class = self.VECTORIZER_TYPES.get(self.vectorizer_type)

        if vectorizer_class is None:
            raise ValueError(f"Unknown vectorizer type: {self.vectorizer_type}")

        params = {
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'min_df': self.min_df,
            'max_df': self.max_df
        }

        if self.vectorizer_type == 'tfidf':
            params['sublinear_tf'] = self.sublinear_tf
            params['use_idf'] = self.use_idf

        return vectorizer_class(**params)

    def fit(self, texts) -> 'FeatureVectorizer':
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self

    def transform(self, texts) -> spmatrix:
        if not self.is_fitted:
            raise RuntimeError("Vectorizer must be fitted first")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts) -> spmatrix:
        self.is_fitted = True
        return self.vectorizer.fit_transform(texts)

    def get_feature_names(self) -> list:
        if not self.is_fitted:
            raise RuntimeError("Vectorizer must be fitted first")
        return self.vectorizer.get_feature_names_out().tolist()

    def get_config_summary(self) -> dict:
        return {
            'type': self.vectorizer_type,
            'max_features': self.max_features,
            'ngram_range': self.ngram_range,
            'min_df': self.min_df,
            'max_df': self.max_df,
            'sublinear_tf': self.sublinear_tf if self.vectorizer_type == 'tfidf' else None,
            'use_idf': self.use_idf if self.vectorizer_type == 'tfidf' else None
        }

    def get_actual_feature_count(self) -> int:
        if not self.is_fitted:
            return 0
        return len(self.get_feature_names())
