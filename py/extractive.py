import re
from underthesea import word_tokenize, sent_tokenize

STOPWORDS_PATH = "Vietnamese-stopwords.txt"

def get_stopwords(stop_file_path=STOPWORDS_PATH):
    try:
        with open(stop_file_path, 'r', encoding="utf-8") as f:
            return set([m.strip().lower() for m in f.readlines()])
    except Exception:
        return set()

stopwords = get_stopwords()

def _create_dictionary_table(text):
    freq_table = {}
    for wd in word_tokenize(text.lower()):
        if wd not in stopwords:
            freq_table[wd] = freq_table.get(wd, 0) + 1
    return freq_table

def _calculate_sentence_scores(sentences, freq_table):
    sentence_scores = {}
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        score, word_count = 0, 0
        for wd, count in freq_table.items():
            if wd in words and wd not in stopwords:
                score += count
                word_count += 1
        sentence_scores[sentence] = (score / word_count) if word_count > 0 else 0.0
    return sentence_scores

def _calculate_average_score(scores):
    return sum(scores.values()) / len(scores) if scores else 0.0

def extractive_summary(text):
    # Làm sạch text
    text = re.sub(r'\s+', ' ', text.strip())
    freq_table = _create_dictionary_table(text)
    sentences = sent_tokenize(text)
    if not sentences:
        return ""
    sentence_scores = _calculate_sentence_scores(sentences, freq_table)
    if not sentence_scores:
        return ""
    threshold = _calculate_average_score(sentence_scores)
    summary = " ".join([s for s in sentences if sentence_scores.get(s,0) >= threshold]).strip()
    return summary