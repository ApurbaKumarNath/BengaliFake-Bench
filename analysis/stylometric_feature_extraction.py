
def extract_stylometric_features(df):
    '''
    Extract 14 hand-crafted stylometric features from a DataFrame.
    See Notebook 3 Cell 1 for full documentation.
    '''
    features = pd.DataFrame(index=df.index)
    combined = df['headline'].fillna('') + ' ' + df['content'].fillna('')
    headline = df['headline'].fillna('')
    body = df['content'].fillna('')
    
    features['text_len'] = combined.str.len()
    features['word_count'] = combined.str.split().str.len()
    features['headline_len'] = headline.str.len()
    features['body_len'] = body.str.len()
    features['headline_word_count'] = headline.str.split().str.len()
    features['sentence_count'] = combined.str.count(r'[।.!?]+')
    features['avg_word_len'] = features['text_len'] / features['word_count'].replace(0, 1)
    features['exclamation_count'] = combined.str.count('!')
    features['question_count'] = combined.str.count(r'\?')
    features['danda_count'] = combined.str.count('।')
    features['digit_count'] = combined.str.count(r'\d')
    features['digit_ratio'] = features['digit_count'] / features['text_len'].replace(0, 1)
    
    def calc_unique_ratio(text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            return 0.0
        words = text.split()
        if len(words) == 0:
            return 0.0
        return len(set(words)) / len(words)
    
    features['unique_word_ratio'] = combined.apply(calc_unique_ratio)
    features['hl_body_ratio'] = features['headline_len'] / features['body_len'].replace(0, 1)
    
    return features.fillna(0)
