import torch.nn as nn

class DocumentClassifierCNN(nn.Module):
    def __init__(self, num_classes=5, use_batchnorm=True):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.features = nn.Sequential(
            self._conv_block(3,   32),   # 224->112
            self._conv_block(32,  64),   # 112->56
            self._conv_block(64,  128),  # 56->28
            self._conv_block(128, 256),  # 28->14
            self._conv_block(256, 512),  # 14->7
            nn.AdaptiveAvgPool2d((4, 4)) # 7->4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        self._init_weights()

    def _conv_block(self, in_ch, out_ch):
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not self.use_batchnorm)]
        if self.use_batchnorm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2, 2))
        return nn.Sequential(*layers)

    def _init_weights(self):
        # He initialisation for conv, Xavier for linear
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.features(x))
