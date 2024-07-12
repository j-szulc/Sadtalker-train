import torch
import numpy as np
import torch.nn as nn
from skorch import NeuralNetRegressor
from skorch.callbacks import Callback
import os
import dill

class CheckpointEveryEpoch(Callback):

    def on_epoch_end(self, net, **kwargs):
        epoch=net.history[-1]['epoch']
        with open(f"{net.module_.__name__}_checkpoint_{epoch}.pth", "wb"):
            dill.dump(net.module_, f)
        print(f"Checkpoint saved to checkpoint_{epoch}.pth")

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.5, max_len=2048):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                                -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
class MaskedTransformerDecoder(nn.Module):

    def __init__(self, backbone):
        super(MaskedTransformerDecoder, self).__init__()
        self.backbone = backbone

    @property
    def device(self):
        return next(self.parameters()).device
        
    def forward(self, x):
        memory_mask = tgt_mask = torch.triu(torch.ones(x.size(1), x.size(1), device=self.device))
        return self.backbone(x, x, memory_mask=memory_mask, tgt_mask=tgt_mask)

class LastTokenMSELoss(nn.Module):
    def forward(self, y_pred, y_gt):
        return torch.mean((y_pred[:, -1] - y_gt[:, -1])**2)
    
class LossScoredNeuralNetRegressor(NeuralNetRegressor):

    def score(self, X, y):
        device = self.device
        with torch.no_grad():
            X = torch.as_tensor(X, device=device)
            y_pred = torch.as_tensor(self.predict(X), device=device)
            y = torch.as_tensor(y, device=device)
            return -self.get_loss(y_pred, y).item()

def get_transformer(device, max_len=8, d_model=64, nhead=8, dim_feedforward=256, num_layers=16, max_epochs=100):
    model = nn.Sequential(
        PositionalEncoding(d_model=d_model, max_len=max_len),
        MaskedTransformerDecoder(torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True),
            num_layers=num_layers
    ))).to(device)
    result = LossScoredNeuralNetRegressor(model, criterion=LastTokenMSELoss, device=device, max_epochs=max_epochs, iterator_train__shuffle=True, callbacks=[CheckpointEveryEpoch()])
    result.input_shape = "(BATCH_SIZE, FRAME_LEN, D_MODEL)"
    result.torch_or_numpy = "torch"
    return result