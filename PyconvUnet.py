import torch
import torch.nn as nn
import os

class PyConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,pyconv_kernels,pyconv_groups,stride=1,dilation=1,bias=False):
        '''
        Args:
            in_channels(int):number of channels in the input image
            out_channels(list):number of channels for each pyramid level produced by the convolution
            pyconv_kernels(list):spatial size of the kernel for each pyramid level
            pyconv_gruops(list):number of blocked connections from input channels to output channels for each pyramid level
            stride(int or tuple,optional):stride of the convolution.default:1
            dilation(int or tuple,optional):Spacing between kernel elements.default:1
            bias(bools,optional):if"True",adds a learnable bias to the output.default:"False"
        '''
        super(PyConv2d,self).__init__()
        assert len(out_channels) == len(pyconv_kernels) == len(pyconv_groups)

        self.pyconv_levels = [None] * len(pyconv_kernels)
        for i in range(len(pyconv_kernels)):
            self.pyconv_levels[i] = nn.Conv2d(in_channels,out_channels[i],kernel_size = pyconv_kernels[i],
                                              stride = stride,padding = pyconv_kernels[i] // 2,groups = pyconv_groups[i],
                                              dilation = dilation,bias=bias)
        self.pyconv_levels = nn.ModuleList(self.pyconv_levels)

    def forward(self,x):
        out = []
        for level in self.pyconv_levels:
            out.append(level(x))

        return torch.cat(out,1)

def conv(in_planes,out_planes,kernel_size = 3,stride = 1,padding = 1,dilation = 1,groups = 1):
    return nn.Conv2d(in_planes,out_planes,kernel_size = kernel_size,stride = stride,
                     padding = padding,dilation = dilation,groups = groups,bias = False)

def conv1x1(in_planes,out_planes,stride = 1):
    return nn.Conv2d(in_planes,out_planes,kernel_size=1,stride=stride,bias=False)

class PyConv4(nn.Module):
    def __init__(self,inplanes,planes,pyconv_kernels = [3,5,7,9],stride = 1,pyconv_groups = [1,4,8,16]):
        super(PyConv4, self).__init__()
        self.conv2_1 = conv(inplanes, planes // 8, kernel_size = pyconv_kernels[0], padding = pyconv_kernels[0] // 2,
                            stride = stride,groups=pyconv_groups[0])
        self.conv2_2 = conv(inplanes, planes // 8, kernel_size = pyconv_kernels[1], padding = pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplanes, planes // 4, kernel_size = pyconv_kernels[2], padding = pyconv_kernels[2] // 2,
                            stride=stride, groups=pyconv_groups[2])
        self.conv2_4 = conv(inplanes, planes // 2, kernel_size = pyconv_kernels[3], padding = pyconv_kernels[3] // 2,
                            stride=stride, groups=pyconv_groups[3])

    def forward(self, x):
        return torch.cat((self.conv2_1(x),self.conv2_2(x),self.conv2_3(x),self.conv2_4(x)),dim=1)


class PyConv3(nn.Module):
    def __init__(self,inplanes,planes,pyconv_kernels = [3,5,7],stride = 1,pyconv_groups = [1,4,8]):
        super(PyConv3,self).__init__()
        # self.conv2_1 = conv(inplanes, planes // 4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
        #                     stride=stride, groups=pyconv_groups[0])
        # self.conv2_2 = conv(inplanes, planes // 4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
        #                     stride=stride, groups=pyconv_groups[1])
        # self.conv2_3 = conv(inplanes, planes // 2, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2] // 2,
        #                     stride=stride, groups=pyconv_groups[2])

        self.conv2_1 = conv(inplanes, planes // 4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplanes, planes // 2, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplanes, planes // 2, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2] // 2,
                            stride=stride, groups=pyconv_groups[2])

    def forward(self,x):
        return torch.cat((self.conv2_1(x),self.conv2_2(x),self.conv2_3(x)),dim=1)


class PyConv2(nn.Module):
    def __init__(self, inplanes, planes, pyconv_kernels=[3, 5], stride=1, pyconv_groups=[1, 4]):
        super(PyConv2, self).__init__()
        self.conv2_1 = conv(inplanes, planes // 2, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplanes, planes // 2, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x)), dim=1)




def get_pyconv(inplanes,planes,pyconv_kernels,stride = 1,pyconv_groups=[1]):
    if len(pyconv_kernels) == 1:
        return conv(inplanes,planes,kernel_size=pyconv_kernels[0],stride=stride,groups=pyconv_groups[0])
    elif len(pyconv_groups) == 2:
        return PyConv2(inplanes,planes,pyconv_kernels=pyconv_kernels,stride=stride,pyconv_groups=pyconv_groups)
    elif len(pyconv_groups) == 3:
        return PyConv3(inplanes, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 4:
        return PyConv4(inplanes, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)


class PyConvBlock(nn.Module):
    expansion = 4

    def __init__(self,inplanes,planes,stride = 1,downsample = None,norm_layer = None,pyconv_groups = 1,pyconv_kernels = 1):
        super(PyConvBlock,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv1x1(inplanes,planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = get_pyconv(planes,planes,pyconv_kernels=pyconv_kernels,stride=stride,pyconv_groups=pyconv_groups)

        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes,planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
        indetity = x
        # print('=' * 60)
        # print('x',x.shape)
        # print('=' * 60)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print('=' * 60)
        # print('conv1',out.shape)
        # print('=' * 60)


        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # print('=' * 60)
        # print('conv2', out.shape)
        # print('=' * 60)



        out = self.conv3(out)
        out = self.bn3(out)
        # print('=' * 60)
        # print(out.shape)
        # print('=' * 60)
        if self.downsample is not None:
            indetity = self.downsample(x)

        # out += indetity
        out = self.relu(out)

        return out


class DoublePyConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoublePyConv, self).__init__()
        self.pyconv = nn.Sequential(
            # nn.Conv2d(in_ch, out_ch, 3, padding=1),
            PyConvBlock(in_ch, out_ch, pyconv_kernels=[3, 5, 7,9], pyconv_groups=[1, 4,8,16]),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            PyConvBlock(out_ch,out_ch,pyconv_kernels=[3,5,7,9],pyconv_groups=[1,4,8,16]),
            # nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.pyconv(input)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)




class PyConvUnet(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(PyConvUnet, self).__init__()
        self.conv1 = DoublePyConv(in_ch, 32*4)
        # self.pyconv1 = PyConvBlock(32,8,pyconv_kernels=[3,5],pyconv_groups=[1,4])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoublePyConv(32*4, 64*4)
        # self.pyconv2 = PyConvBlock(64, 16,pyconv_kernels=[3,5],pyconv_groups=[1,4])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoublePyConv(64*4, 128*4)
        # self.pyconv3 = PyConvBlock(128,32,pyconv_kernels=[3,5],pyconv_groups=[1,4])
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoublePyConv(128*4, 256*4)
        # self.pyconv4 = PyConvBlock(256,64,pyconv_kernels=[3,5],pyconv_groups=[1,4])
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoublePyConv(256*4, 512*4)
        self.pool5 = nn.MaxPool2d(2)


        self.up6 = nn.ConvTranspose2d(512*4, 256*4, 2, stride=2)
        # self.conv6 = DoubleConv(512, 256)
        self.conv6 = DoublePyConv(512*4, 256*4)
        self.up7 = nn.ConvTranspose2d(256*4, 128*4, 2, stride=2)
        # self.conv7 = DoubleConv(256, 128)
        self.conv7 = DoublePyConv(256*4, 128*4)
        self.up8 = nn.ConvTranspose2d(128*4, 64*4, 2, stride=2)
        # self.conv8 = DoubleConv(128, 64)
        self.conv8 = DoublePyConv(128*4, 64*4)
        self.up9 = nn.ConvTranspose2d(64*4, 32*4, 2, stride=2)
        # self.conv9 = DoubleConv(64, 32)
        self.conv9 = DoublePyConv(64*4, 32*4)
        # self.up10 = nn.ConvTranspose2d(32,16,2,stride=2)
        self.conv10 = nn.Conv2d(32*4, out_ch, 1)

    def forward(self, x):
        # print('='*60)
        # print('x:',x.shape)
        # print('=' * 60)

        c1 = self.conv1(x)
        # pc1 = self.pyconv1(c1)
        p1 = self.pool1(c1)




        c2 = self.conv2(p1)
        # pc2 = self.pyconv2(c2)
        p2 = self.pool2(c2)


        c3 = self.conv3(p2)
        # pc3 = self.pyconv3(c3)
        p3 = self.pool3(c3)


        c4 = self.conv4(p3)
        # pc4 = self.pyconv4(c4)
        p4 = self.pool4(c4)



        c5 = self.conv5(p4)
        # p5 = self.pool5(c5)


        up_6 = self.up6(c5)
        # merge6 = torch.cat([up_6, pc4], dim=1)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        # merge7 = torch.cat([up_7, pc3], dim=1)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        # merge8 = torch.cat([up_8, pc2], dim=1)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        # merge9 = torch.cat([up_9, pc1], dim=1)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        # up_10 = self.up10(c9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        # print('+'*60)
        # print(out.shape)
        # print('+' * 60)
        return out


