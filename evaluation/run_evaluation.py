import json
import os
from models.classifier.predict import classify_document
from models.extractor.predict import extract_words
from models.extractor.postprocess import words_to_fields
from pipeline.model_loader import get_classifier, get_extractor
from pipeline.pdf_processor import pdf_to_images
from pipeline.ocr_processor import get_words_and_boxes

def evaluate():
    eval_file = "evaluation/eval_dataset.json"
    if not os.path.exists(eval_file):
        print(f"Eval file not found: {eval_file}")
        return
        
    with open(eval_file) as f:
        test_cases = json.load(f)

    classifier = get_classifier()
    extractor  = get_extractor()

    results = {field: {"correct": 0, "total": 0}
               for field in ["vendor_name", "invoice_number",
                             "total_amount", "invoice_date", "tax_amount"]}

    for case in test_cases:
        try:
            if not os.path.exists(case["file_path"]):
                print(f"Skipping {case['file_path']} - file not found.")
                continue
                
            images = pdf_to_images(case["file_path"])
            words  = get_words_and_boxes(images[0])
            classified = extract_words(extractor, images[0], words)
            fields, _  = words_to_fields(classified)

            for field in results:
                results[field]["total"] += 1
                pred = str(fields.get(field, "")).strip().lower()
                gt   = str(case["ground_truth"].get(field, "")).strip().lower()
                if pred == gt:
                    results[field]["correct"] += 1
        except Exception as e:
            print(f"Failed processing {case['file_path']}: {e}")

    print("\nField-Level Accuracy Report:")
    print("-" * 40)
    overall_correct, overall_total = 0, 0
    for field, counts in results.items():
        if counts["total"] > 0:
            acc = counts["correct"] / counts["total"]
            overall_correct += counts["correct"]
            overall_total   += counts["total"]
            print(f"  {field:20s}: {acc:.1%}  ({counts['correct']}/{counts['total']})")
        else:
            print(f"  {field:20s}: N/A")
            
    print("-" * 40)
    if overall_total > 0:
        print(f"  {'Overall':20s}: {overall_correct/overall_total:.1%}")

if __name__ == "__main__":
    evaluate()
