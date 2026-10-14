import math
import json

class NaiveBayesClassifier:
    def __init__(self, alpha: int = 1):
        self.alpha = alpha
        self.prior_probabilities = {}
        self.likelihood_probabilities = {}
        self.vocab = set()
        self.classes = []
        self.idf = {}

    def fit(self, X: list, y: list):
        total_documents = len(y)
        class_document_counts = {}
        class_term_weights = {}
        class_total_weights = {}
        document_frequency = {}

        for label in set(y):
            self.classes.append(label)
            class_document_counts[label] = 0
            class_term_weights[label] = {}
            class_total_weights[label] = 0.0

        # Kalkulasi Document Frequency (DF) global
        for tokens in X:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                document_frequency[token] = document_frequency.get(token, 0) + 1
                self.vocab.add(token)

        # Kalkulasi Inverse Document Frequency (IDF) dengan proteksi asimtot
        for term in self.vocab:
            self.idf[term] = math.log(total_documents / (1 + document_frequency.get(term, 0)))

        # Kalkulasi bobot matriks TF-IDF untuk masing-masing observasi
        for tokens, label in zip(X, y):
            class_document_counts[label] += 1
            
            doc_tf = {}
            for token in tokens:
                doc_tf[token] = doc_tf.get(token, 0) + 1
                
            for token, count in doc_tf.items():
                tf_idf_weight = count * self.idf[token]
                class_term_weights[label][token] = class_term_weights[label].get(token, 0.0) + tf_idf_weight
                class_total_weights[label] += tf_idf_weight

        vocab_size = len(self.vocab)

        # Kalkulasi Probabilitas Posterior dan Likelihood berbasis TF-IDF
        for label in self.classes:
            self.prior_probabilities[label] = math.log(class_document_counts[label] / total_documents)
            self.likelihood_probabilities[label] = {}
            
            for term in self.vocab:
                term_weight = class_term_weights[label].get(term, 0.0)
                probability = (term_weight + self.alpha) / (class_total_weights[label] + (self.alpha * vocab_size))
                self.likelihood_probabilities[label][term] = math.log(probability)

    def predict(self, tokens: list) -> str:
        best_class = None
        max_posterior = float('-inf')

        for label in self.classes:
            posterior = self.prior_probabilities[label]
            for token in tokens:
                if token in self.vocab:
                    posterior += self.likelihood_probabilities[label][token]
            
            if posterior > max_posterior:
                max_posterior = posterior
                best_class = label

        return best_class

    def export_model(self, filepath: str):
        model_data = {
            'priors': self.prior_probabilities,
            'likelihoods': self.likelihood_probabilities,
            'classes': self.classes,
            'vocab': list(self.vocab),
            'idf': self.idf
        }
        with open(filepath, 'w') as file:
            json.dump(model_data, file)

    def load_model(self, filepath: str):
        with open(filepath, 'r') as file:
            model_data = json.load(file)
            self.prior_probabilities = model_data['priors']
            self.likelihood_probabilities = model_data['likelihoods']
            self.classes = model_data['classes']
            self.vocab = set(model_data['vocab'])
            self.idf = model_data.get('idf', {})