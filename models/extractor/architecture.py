import torch
import torch.nn as nn

class WordRegionExtractor(nn.Module):
    def __init__(self, num_classes=7, use_position=True):
        super().__init__()
        self.use_position = use_position
        
        # Visual branch
        self.visual = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )   # output: 256-dim
        
        # Position branch
        self.position = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 64)
        )   # output: 64-dim
        
        # Fusion head
        fusion_dim = 256 + 64 if use_position else 256
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, image, position):
        visual_feat = self.visual(image)
        if self.use_position:
            pos_feat = self.position(position)
            combined = torch.cat([visual_feat, pos_feat], dim=1)
        else:
            combined = visual_feat
            
        return self.head(combined)
