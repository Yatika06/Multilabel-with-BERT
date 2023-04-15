import sys
import os

sys.path.append(os.path.abspath("/Users/yatikapaliwal/PycharmProjects/text_classification/"))
import re
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import string

nltk.download('all')
stop_words = set(stopwords.words('english'))


class Preprocessing:
    def __init__(self, is_lower: True, stem_method: None, is_remove_punct: True, is_stopwords_removal: False):
        self.is_lower = is_lower
        self.stem_method = stem_method
        self.is_remove_punct = is_remove_punct
        self.is_stopwords_removal = is_stopwords_removal

    def basic_tn(self, text):
        text = text.lower()
        text = self.rm_extra_space(text)
        return text

    def rm_extra_space(self, text):
        text = re.sub('\[.*?\]', ' ', text)
        text = re.sub('https?://\S+|www\.\S+', ' ', text)
        text = re.sub('<.*?>+', ' ', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = re.sub('\n', ' ', text)
        text = re.sub('\w*\d\w*', ' ', text)
        text = re.sub("\s+"," ",text)
        return text

    def normalize(self, text):
        if self.is_remove_punct and self.is_lower:
            text = self.basic_tn(text)
        if self.stem_method == 'stemming' and self.is_stopwords_removal:
            text = text.split()
            stemmer = PorterStemmer()
            text = ' '.join(stemmer.stem(word) for word in text if word not in stop_words)
        if self.stem_method == 'lemmatizer' and self.is_stopwords_removal:
            text = text.split()
            lemmatizer = WordNetLemmatizer()
            text = ' '.join(lemmatizer.lemmatize(word) for word in text if word not in stop_words)

        return text

    def run(self, lines):
        data = [self.normalize(line) for line in lines]
        return data

