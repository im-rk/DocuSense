import re

def group_by_label(classified_words):
    groups = {}
    for w in classified_words:
        lbl = w['label']
        if lbl not in groups:
            groups[lbl] = []
        groups[lbl].append(w)
    return groups

def normalise_date(text):
    return text.strip()

def parse_amount(text):
    s = re.sub(r'[^\d.]', '', text)
    try:
        return float(s)
    except:
        return 0.0

def words_to_fields(classified_words: list[dict]) -> tuple[dict, list]:
    fields = {}
    line_items = []
    
    groups = group_by_label(classified_words)
    
    for label, words in groups.items():
        if label == "OTHER":
            continue
            
        text = " ".join(w["text"] for w in words)
        avg_conf = sum(w["conf"] for w in words) / len(words)
        
        if label == "VENDOR_NAME":
            fields["vendor_name"] = text
            fields["_conf_vendor_name"] = avg_conf
            
        elif label == "INVOICE_NUMBER":
            fields["invoice_number"] = text.strip()
            fields["_conf_invoice_number"] = avg_conf
            
        elif label == "DATE":
            fields["invoice_date"] = normalise_date(text)
            
        elif label == "AMOUNT":
            amounts = [parse_amount(w["text"]) for w in words]
            amounts = sorted([a for a in amounts if a > 0], reverse=True)
            if len(amounts) >= 1:
                fields["total_amount"] = amounts[0]
            if len(amounts) >= 2:
                fields["subtotal"] = amounts[1]
            if len(amounts) >= 3:
                fields["tax_amount"] = amounts[2]
                
        elif label == "LINE_ITEM_DESC":
            for w in words:
                line_items.append({"description": w["text"]})
                
    return fields, line_items
