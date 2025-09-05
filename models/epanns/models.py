import torch
import torch.nn as nn

class Cnn14_pruned(nn.Module):
    def __init__(self, sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527):
        super(Cnn14_pruned, self).__init__()
        self.classes_num = classes_num
        
    def forward(self, x):
        batch_size = x.shape[0]
        return torch.randn(batch_size, self.classes_num)
