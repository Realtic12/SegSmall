import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

"""
    Reduce the spatial dimensions of the input while increasing the number of channels.
    Combine convolution and max pooling operations. It is used by the decoder"
"""
class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        #reduces by two the dimension
        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)

"""
    Squeeze-and-Excitation mechanism:
    Adapts the network to recalibrate the channel-wise feature reponses, boosting important ones and not taking care about the less ones"
"""
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

"""
    Separates the process into two steps:
        -Depthwise convolution: Single filer to each input
        -Pointwise convoulution: Applies 1x1 convolution to combine the outpus
        Reduces the number of parameters and computations compared to standard convolutions
"""
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-03)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x)
    

class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.conv3x1_1 = DepthwiseSeparableConv(chann, chann, stride=1)

        self.conv1x3_1 = DepthwiseSeparableConv(chann, chann, stride=1)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = DepthwiseSeparableConv(chann, chann, stride=1)

        self.conv1x3_2 = DepthwiseSeparableConv(chann, chann, stride=1)

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

        self.se = SEBlock(chann)  # Add SE block
        

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        output = self.se(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3,16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16,64))

        for x in range(0, 5):    #5 times
           self.layers.append(non_bottleneck_1d(64, 0.03, 1)) 

        self.layers.append(DownsamplerBlock(64,128))

        for x in range(0, 2):    #2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

        self.se1 = SEBlock(64)  #Important!!!!!!!!!!!!
        self.se2 = SEBlock(128) #Important!!!!!!!!!!!!


    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for i, layer in enumerate(self.layers):
            output = layer(output)
            if i == 5:                      # After the first set of non-bottleneck blocks
                output = self.se1(output)
            elif i == len(self.layers) - 1:  # After the last set of non-bottleneck blocks
                output = self.se2(output)


        if predict:
            output = self.output_conv(output)

        return output

"""
    Designed to increase the spatial resolution while reducing he number of channels. Component of the decoder
"""
class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

"""
    Used to gradually increase the resolution back to the original image size.
"""
class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128,64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64,16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d( 16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output

#SegSmall
class SegSmall(nn.Module):
    def __init__(self, num_classes):  #use encoder to pass pretrained encoder
        super().__init__()

        self.encoder = Encoder(num_classes)
        self.decoder = Decoder(num_classes)

    def forward(self, input):
        output = self.encoder(input)    #predict=False by default
        return self.decoder.forward(output)