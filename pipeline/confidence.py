def compute_field_confidences(classified_words):
    confidences = {}
    for w in classified_words:
        label = w['label']
        if label not in confidences:
            confidences[label] = []
        confidences[label].append(w['conf'])
    
    avg_confidences = {}
    for k, v in confidences.items():
        avg_confidences[k] = sum(v) / len(v)
    return avg_confidences
