# model.py

import torch.nn as nn

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        if len(out.shape) == 2:
            out = out.unsqueeze(0)
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out
