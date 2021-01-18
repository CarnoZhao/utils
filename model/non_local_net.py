import torch
import torch.nn as nn

class ConvBN(nn.Module):
    def __init__(self, *args):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(*args)
        self.bn = nn.BatchNorm2d(args[1])
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class NonLocalBlock(nn.Module):
    def __init__(self, in_c, n_c):
        super(NonLocalBlock, self).__init__()
        self.n_c = n_c
        self.conv1 = ConvBN(in_c, n_c, (3, 3))
        self.conv2 = ConvBN(in_c, n_c, (3, 3))
        self.conv3 = ConvBN(in_c, n_c, (3, 3))
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(n_c, in_c, (3, 3)),
            nn.BatchNorm2d(in_c),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        s1, s2 = x1.shape[2:]
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x1 = x1.view(x.shape[0], self.n_c, -1).permute(0, 2, 1)
        x2 = x2.view(x.shape[0], self.n_c, -1)
        x3 = x3.view(x.shape[0], self.n_c, -1).permute(0, 2, 1)
        x12 = torch.matmul(x1, x2)
        x12 = x12.softmax(1)
        x123 = torch.matmul(x12, x3).view(x.shape[0], s1, s2, self.n_c).permute(0, 3, 1, 2)
        x123 = self.conv4(x123)
        x = x + x123
        return x

class DSRefineNet(nn.Module):
    def __init__(self, n_c):
        super(DSRefineNet, self).__init__()
        self.conv1 = ConvBN(n_c, n_c * 4, (3, 3), 1, 1)
        self.conv2 = ConvBN(5 * n_c, 8 * n_c, (3, 3), 1, 1)
        self.conv3 = ConvBN(13 * n_c, n_c, (3, 3), 1, 1)
        self.conv4 = ConvBN(14 * n_c, n_c, (1, 1))

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = torch.cat([x, x1], 1)
        x2 = self.conv2(x1)
        x2 = torch.cat([x1, x2], 1)
        x3 = self.conv3(x2)
        x3 = torch.cat([x2, x3], 1)
        x4 = self.conv4(x3)
        return x4

class CREncoder(nn.Module):
    def __init__(self, bits, n_c = 64, dim1 = 24, dim2 = 16):
        super(CREncoder, self).__init__()
        self.nlb = nn.Sequential(
            ConvBN(2, n_c, (1, 1)),
            NonLocalBlock(n_c, n_c * 2),
            ConvBN(n_c, 2, (1, 1))
        )
        self.fc = nn.Sequential(
            nn.Linear(2 * dim1 * dim2, bits // 4),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.nlb(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

class CRDecoder(nn.Module):
    def __init__(self, bits, n_c = 32, dim1 = 24, dim2 = 16):
        super(CRDecoder, self).__init__()
        self.dim1 = dim1; self.dim2 = dim2
        self.fc = nn.Linear(bits // 4, 2 * dim1 * dim2)
        self.dec = nn.Sequential(
            ConvBN(2, n_c, (1, 1)),
            NonLocalBlock(n_c, n_c),
            DSRefineNet(n_c),
            DSRefineNet(n_c),
            ConvBN(n_c, n_c, (3, 3), 1, 1),
            nn.Conv2d(n_c, 2, (1, 1))
        )
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 2, self.dim1, self.dim2)
        x = self.dec(x)
        return x
        

if __name__ == "__main__":
    nlb = CRDecoder(400)
    a = torch.rand(10, 100)
    b = nlb(a)
    b.shape