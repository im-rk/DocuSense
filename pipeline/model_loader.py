import torch
import os
from models.classifier.architecture import DocumentClassifierCNN
from models.extractor.architecture import WordRegionExtractor
from config.settings import settings

_classifier = None
_extractor = None

def get_classifier() -> DocumentClassifierCNN:
    global _classifier
    if _classifier is None:
        _classifier = DocumentClassifierCNN(num_classes=5)
        if os.path.exists(settings.classifier_checkpoint):
            checkpoint = torch.load(settings.classifier_checkpoint, map_location=settings.device)
            _classifier.load_state_dict(checkpoint['model_state_dict'])
        _classifier.eval()
        _classifier.to(settings.device)
    return _classifier

def get_extractor() -> WordRegionExtractor:
    global _extractor
    if _extractor is None:
        _extractor = WordRegionExtractor(num_classes=7)
        if os.path.exists(settings.extractor_checkpoint):
            checkpoint = torch.load(settings.extractor_checkpoint, map_location=settings.device)
            _extractor.load_state_dict(checkpoint['model_state_dict'])
        _extractor.eval()
        _extractor.to(settings.device)
    return _extractor
