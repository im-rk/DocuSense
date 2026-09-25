"""Model 2: DocTaggerNet - three-branch CNN + BiLSTM word tagger, trained from scratch.

For every word on the page:
    TEXT branch : characters -> embedding -> parallel Conv1d (widths 2,3,4,5) -> max-pool over characters   (256-d)
    LOOK branch : 32x96 grayscale crop of the word -> small 2-D CNN                                          (64-d)
    POS  branch : 10 layout numbers (box, size, centre, first/last on line) -> MLP                           (64-d)
The three are fused per word, then a 2-layer BiLSTM reads all words in reading order (both directions) so a word's
label can depend on its neighbours, and a linear layer outputs one label per word. ~2M parameters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.tagger.features import POS_DIM
from models.tagger.labels import N_CLS

DEFAULT_CFG = dict(vocab=100, n_cls=N_CLS, d_char=48, n_filt=64, d_pos=64, d=256, hid=192, p=0.3,
                   use_text=True, use_look=True, use_pos=True, use_lstm=True, lstm_layers=2)


class DocTaggerNet(nn.Module):
    def __init__(self, vocab, n_cls=N_CLS, d_char=48, n_filt=64, d_pos=64, d=256, hid=192, p=0.3,
                 use_text=True, use_look=True, use_pos=True, use_lstm=True, lstm_layers=2):
        super().__init__()
        self.use_text, self.use_look, self.use_pos, self.use_lstm = use_text, use_look, use_pos, use_lstm
        self.n_filt = n_filt

        self.cemb = nn.Embedding(vocab, d_char, padding_idx=0)
        self.cconv = nn.ModuleList([nn.Conv1d(d_char, n_filt, k, padding="same") for k in (2, 3, 4, 5)])

        def cb(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.look = nn.Sequential(cb(1, 16), nn.MaxPool2d(2), cb(16, 32), nn.MaxPool2d(2),
                                  cb(32, 64), nn.MaxPool2d(2), cb(64, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.pos = nn.Sequential(nn.Linear(POS_DIM, d_pos), nn.ReLU(), nn.Linear(d_pos, d_pos))

        self.fuse = nn.Sequential(nn.Linear(4 * n_filt + 64 + d_pos, d), nn.ReLU(), nn.Dropout(p))
        self.lstm = nn.LSTM(d, hid, num_layers=lstm_layers, batch_first=True, bidirectional=True,
                            dropout=p if lstm_layers > 1 else 0.0) if use_lstm else None
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(2 * hid if use_lstm else d, n_cls))

        for m in self.modules():                         # from-scratch initialisation
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.1)
                with torch.no_grad():
                    m.weight[0].zero_()

    def forward(self, chars, crops, pos, pad):
        """chars (B,N,L) long | crops (B,N,1,32,96) float | pos (B,N,10) float | pad (B,N) bool, True = padding
        -> logits (B,N,n_cls)"""
        B, N, L = chars.shape
        real = ~pad
        if self.use_text:
            c = chars.reshape(B * N, L)
            e = self.cemb(c).transpose(1, 2)                                            # (B*N, d_char, L)
            t = torch.cat([F.relu(cv(e)).masked_fill((c == 0).unsqueeze(1), -1e4).max(2).values
                           for cv in self.cconv], dim=1).view(B, N, -1)                 # (B,N,4*n_filt)
            t = t.masked_fill(pad.unsqueeze(-1), 0.0)
        else:
            t = pos.new_zeros(B, N, 4 * self.n_filt)
        if self.use_look and real.any():
            feats = self.look(crops[real])                                              # only real words
            v = feats.new_zeros(B, N, feats.shape[1])
            v[real] = feats
        else:
            v = pos.new_zeros(B, N, 64)
        p = self.pos(pos) if self.use_pos else pos.new_zeros(B, N, self.pos[-1].out_features)

        h = self.fuse(torch.cat([t, v, p], dim=-1))                                     # (B,N,d)
        if self.use_lstm:
            lens = real.sum(1).clamp(min=1).cpu()
            out, _ = self.lstm(pack_padded_sequence(h, lens, batch_first=True, enforce_sorted=False))
            h, _ = pad_packed_sequence(out, batch_first=True, total_length=N)           # (B,N,2*hid)
        return self.head(h)


def build_tagger(cfg: dict) -> DocTaggerNet:
    full = {**DEFAULT_CFG, **cfg}
    return DocTaggerNet(**{k: full[k] for k in DEFAULT_CFG})


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
