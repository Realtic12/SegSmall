import torch
import torch.nn as nn
import torch.nn.functional as F

#Class to create the net
class SegSmall(nn.Module):
    """
        Defines the structure
    """
    def __init__(self, args) -> None:
        super().__init__()
        self.in_channels = 3  #rgb image
        self.out_channels = 64 #number of filters applied in the convolutional layer
        self.kernel_size = (3,3) #size of the convolutional kernel (cuadratic), if not declare it like (2,1) -> mxn
        self.stride = (2,2) #stride of the convolution operation (cuadratic)
        self.padding = 1 #padding added to all four sides (cuadratic)
        #self.dilation = 2 #with that we can make dilated convolutions (cuadratic)
        self.classes = args.num_classes

        #define convolution, max pooling and batch normalization
        self.conv = nn.Conv2d(self.in_channels, self.out_channels-self.in_channels, self.kernel_size, self.stride, self.padding)
        self.max_pooling = nn.MaxPool2d(kernel_size = (2,2), stride = 2)
        self.batch_norm = nn.BatchNorm2d(self.out_channels) #expects input tensors with 3 channels, should be the same as input channels
        #print(self.conv) 
        pass
        

    def encoder(self):
        pass

class DownSamplerBlock(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()

        #TBD define
        self.conv = nn.Conv2d(in_channel, out_channel-in_channel, (3, 3), stride = 2, padding = 1, bias = True)
        self.max_pooling = nn.MaxPool2d(kernel_size = (2,2), stride = 2)
        self.batch_norm = nn.BatchNorm2d(out_channel, eps=1e-3)
    
    def forward(self, input):
        output = torch.cat([self.conv(input), self.max_pooling(input)], 1)
        output = self.batch_norm(output)
        return F.relu(output)
    
class UpSamplerBlock(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()

        #TBD define
        self.conv = nn.ConvTranspose2d(in_channel, out_channel-in_channel, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.batch_norm = nn.BatchNorm2d(out_channel, eps=1e-3)
    
    def forward(self, input):
        output = self.conv(input)
        output = self.batch_norm(output)
        return F.relu(output)

class Encoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass

class Decoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass



