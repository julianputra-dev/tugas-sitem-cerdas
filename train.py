import csv
import random
from core.preprocessing import TextPreprocessor
from core.naive_bayes import NaiveBayesClassifier

def execute_training():
    preprocessor = TextPreprocessor()
    classifier = NaiveBayesClassifier(alpha=1)
    
    data_recommended = []
    data_not_recommended = []
    
    feedback_recommended = []
    feedback_not_recommended = []
    
    with open('data/steam_reviews.csv', mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            review_text = row.get('review', '')
            recommendation = row.get('recommendation', '')
            title = row.get('title', '')
            
            if review_text and recommendation:
                tokens = preprocessor.clean_text(review_text)
                if not tokens:
                    continue
                
                if title == 'User Feedback':
                    if recommendation == 'Recommended':
                        # Berikan bobot yang sangat besar untuk setiap feedback agar model segera menyesuaikan
                        feedback_recommended.extend([(tokens, recommendation)] * 50)
                    elif recommendation == 'Not Recommended':
                        feedback_not_recommended.extend([(tokens, recommendation)] * 50)
                else:
                    if recommendation == 'Recommended':
                        data_recommended.append((tokens, recommendation))
                    elif recommendation == 'Not Recommended':
                        data_not_recommended.append((tokens, recommendation))

    # Restriksi batas matriks absolut (10.000 sampel per kelas)
    MAX_SAMPLES = 10000
    random.seed(42)
    
    if len(data_recommended) > MAX_SAMPLES:
        data_recommended = random.sample(data_recommended, MAX_SAMPLES)
    if len(data_not_recommended) > MAX_SAMPLES:
        data_not_recommended = random.sample(data_not_recommended, MAX_SAMPLES)
        
    min_size = min(len(data_recommended), len(data_not_recommended))
    
    balanced_data = random.sample(data_recommended, min_size) + random.sample(data_not_recommended, min_size)
    
    # Gabungkan data reguler yang di-balance dengan umpan balik pengguna yang dibobotkan
    final_data = balanced_data + feedback_recommended + feedback_not_recommended
    random.shuffle(final_data)
    
    X_train = [item[0] for item in final_data]
    y_train = [item[1] for item in final_data]

    classifier.fit(X_train, y_train)
    classifier.export_model('model/nb_weights.json')

if __name__ == "__main__":
    execute_training()