import torch.nn as nn
import torch
device = 'cpu'



class RNNNet(nn.Module):    
    '''
    vocab_size: int, размер словаря (аргумент embedding-слоя)
    emb_size:   int, размер вектора для описания каждого элемента последовательности
    hidden_dim: int, размер вектора скрытого состояния
    batch_size: int, размер batch'а

    '''
    
    def __init__(self, 
                 vocab_size: int, 
                 emb_size: int, 
                 hidden_dim: int, 
                 seq_len: int, 
                 n_layers: int = 1) -> None:
        super().__init__()
        
        self.seq_len  = seq_len 
        self.emb_size = emb_size 
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, self.emb_size)
        self.rnn_cell  = nn.RNN(self.emb_size, self.hidden_dim, batch_first=True, num_layers=n_layers)
        self.linear    = nn.Sequential(
            nn.Linear(self.hidden_dim * self.seq_len, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Dropout(0.3),
            nn.Linear(128, 1) 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.input = x.size(0)
        x = self.embedding(x.to(device))
        output, _ = self.rnn_cell(x) 
        output = output.contiguous().view(output.size(0), -1)
        out = self.linear(output.squeeze(0))
        return out
    

