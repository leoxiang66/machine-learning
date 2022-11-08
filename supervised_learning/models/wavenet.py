import torch.nn as nn

class WaveNet(nn.Module):
    def __init__(self, n_conv_layers: int, input_channels: int, output_channels:int, sequence_length:int):
        super().__init__()
        for i in range(n_conv_layers):
            self.__setattr__(f'conv_{i}',nn.Conv1d(
                in_channels=input_channels,
                out_channels= output_channels,
                kernel_size=2,
                stride= 1,
                dilation=2**(i),
                padding='same'
            ))

        self.linear1 = nn.Linear(output_channels*sequence_length,100)
        self.linear2 = nn.Linear(100,output_channels*sequence_length)
        self.flatten = nn.Flatten()
        self.n_conv = n_conv_layers

        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(output_channels*sequence_length)


    def forward(self,X):
        '''

        :param X: (N,d,l)
        :return: (N,d)
        '''

        sum = 0
        for i in range(self.n_conv):
            tmp = X
            conv = self.__getattr__(f'conv_{i}')
            X = conv(X)
            X = nn.functional.relu(X)
            sum = sum + X
            X = X + tmp


        sum = sum / self.n_conv
        ret = self.flatten(sum)

        ret = nn.functional.relu(self.linear1(ret))
        ret = self.bn1(ret)
        ret = nn.functional.relu(self.linear2(ret))
        ret = self.bn2(ret)

        return ret

class WaveNetForClassification(nn.Module):
    def __init__(self, n_conv_layers: int, input_channels: int, output_channels: int, num_class: int,
                 num_sequence: int):
        super().__init__()
        self.wavenet = WaveNet(n_conv_layers,input_channels,output_channels,num_sequence)
        self.linear3 = nn.Linear(output_channels*num_sequence,num_class)

    def forward(self,X):
        x = self.wavenet(X)
        x = nn.functional.softmax(self.linear3(x))
        return x



