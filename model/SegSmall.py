import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)

"""
    Reduce the spatial dimensions of the input while increasing the number of channels.
    Combine convolution and max pooling operations. It is used by the decoder"
"""
class DownsamplerBlock(nn.Module):
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
    Original from Segnet, block has been modified to accept diferent dilation rates.
"""
class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilations):        
        super().__init__()
        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)
        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
        
        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(dilations[0],0), bias=True, dilation=(dilations[0],1))
        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,dilations[1]), bias=True, dilation=(1,dilations[1]))
        
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)

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
        
        return F.relu(output+input)

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

"""
    Squeeze-and-Excitation mechanism:
    Adapts the network to recalibrate the channel-wise feature reponses, boosting important ones and not taking care about the less ones"
"""
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace = False),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

"""
    Changing standard convolutions with depthwise separable convolutions. They're much more efficient in terms of computation.
    Reducing the number of parameters and compuations
"""
class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3,16)

        self.layers = nn.ModuleList()

        self.layers.append(DepthwiseSeparableConv(16,64))

        for x in range(0, 5):
           self.layers.append(non_bottleneck_1d(64, 0.03, [1,1])) 

        self.layers.append(DepthwiseSeparableConv(64,128))

        dilations = [[2,2], [4,4], [8,8], [16,16]]
        for x in range(0, 2):
            for d in dilations:
                self.layers.append(non_bottleneck_1d(128, 0.3, d))

        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

        """
            Allows us to adaptively weight the importance of different features
        """
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
class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        #Increase spatial dimensions by a factor of 2 (stride)
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)  #Non-linerity to the output

"""
    Used to gradually increase the resolution back to the original image size.
"""
class Decoder(nn.Module):
    def __init__(self, num_classes, scale_factor = 0.5):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, [1, 1]))
        self.layers.append(non_bottleneck_1d(64, 0, [1, 1]))

        self.layers.append(UpsamplerBlock(64, 32))
        self.layers.append(non_bottleneck_1d(32, 0, [1, 1]))
        self.layers.append(non_bottleneck_1d(32, 0, [1, 1]))

        self.output_conv = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)
        self.scale_factor = scale_factor

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)
        #print(f"output Size {output.size()}")

        # Resize output size to lower resolution
        output_resize = F.interpolate(output, scale_factor = self.scale_factor, mode='bilinear', align_corners = True)
        #print(f"output Size after interpolate {x.size()}")
        return output_resize


class SegSmall(nn.Module):
    def __init__(self, num_classes, encoder=None):
        super().__init__()

        if (encoder == None):
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)

    #Input --> image
    def forward(self, input, only_encode=False):
        #if only_encode:
        #    return self.encoder.forward(input, predict=True)
        #else:
        output = self.encoder(input)
        return self.decoder.forward(output)