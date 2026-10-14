import re

class TextPreprocessor:
    def __init__(self):
        # Perluasan vektor eliminasi untuk mereduksi derau probabilitas dari kata netral
        self.stopwords = {
            'the', 'is', 'in', 'and', 'to', 'a', 'of', 'for', 'it', 'this', 
            'that', 'game', 'on', 'with', 'as', 'i', 'you', 'are', 'was', 'play',
            'many', 'where', 'there', 'lots', 'so', 'an', 'be', 'or', 'at',
            'by', 'from', 'but', 'not', 'have', 'has', 'had', 'they', 'we', 'he', 'she', 'all'
        }

    def clean_text(self, text: str) -> list:
        if not isinstance(text, str):
            return []
            
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        
        # Ekstraksi Unigram (Kata Tunggal)
        cleaned_unigrams = [token for token in tokens if token not in self.stopwords]
        
        # Ekstraksi Bigram (Pasangan Kata) untuk mempertahankan dependensi leksikal (misal: "difficult_controls")
        bigrams = []
        for i in range(len(cleaned_unigrams) - 1):
            bigram = f"{cleaned_unigrams[i]}_{cleaned_unigrams[i+1]}"
            bigrams.append(bigram)
            
        # Agregasi matriks fitur spasial
        return cleaned_unigrams + bigrams