import torch 
import torch.nn as nn

class NeuralNet(nn.Module):
    def _int_(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self)._init_()
        self,l1 = nn.Linear(input_size, hidden_size)
        self,l2 = nn.Linear(input_size, hidden_size)
        self,l3 = nn.Linear(input_size, num_classes)
        self.relu = nn.ReLu()

        def forward(self, x):
            out = self,l1(x)
            out = self.relu(out)
            out = self.l2(out)
            out = self.relu(out)
            out = self,l3(out)
            out = self.relu(out)
            # no activation and softmax
            return out