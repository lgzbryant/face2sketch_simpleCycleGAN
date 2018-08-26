import torch.nn as nn
import torch.nn.functional as F
import torch
       
def weight_init(m):  
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)    
 
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class G(nn.Module):
    def __init__(self, in_channels=3,out_channels=3, n_res=9):
        super(G, self).__init__()
        #3*256*256------>64*256*256:(256+3*2-7+2*0)/1+1
        # Initial convolution block
        model=[ nn.ReflectionPad2d(3),
                nn.Conv2d(in_channels, 64, 7),
                nn.InstanceNorm2d(64),
                nn.ReLU(inplace=True) ]
                
        #64*256*256-->128*128*128-->256*64*64
        # Downsampling        
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model+=[ nn.Conv2d(in_features, out_features, 3, stride=2,padding=1),
                     nn.InstanceNorm2d(out_features),
                     nn.ReLU(inplace=True)]
            in_features= out_features
            out_features *=2
            
            
        #Residual blocks
        for _ in range(n_res):
            model+=[ResidualBlock(in_features)]
     
     
        out_features = in_features//2
        for _ in range(2):
            model += [ nn.ConvTranspose2d(in_features, out_features, 3 ,stride = 2, padding = 1, output_padding=1),
                       nn.InstanceNorm2d(out_features),
                       nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
            # out_features //=2 
   
        
        model+=[nn.ReflectionPad2d(3),
                nn.Conv2d(64,out_channels, 7),
                nn.Tanh()]
        self.model=nn.Sequential(*model)

    def forward(self, x):
        
        return self.model(x)

        
        
class D(nn.Module):
    def __init__(self, in_channels=3):
        super(D, self).__init__()
        #256*256-------->128*128:(256-3+2*1)/2+1        
        model =[nn.Conv2d(in_channels,64,3,stride=2,padding=1),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True),
                 #------->64*64
                 nn.Conv2d(64,128,3,stride=2,padding=1),
                 nn.InstanceNorm2d(128),
                 nn.ReLU(inplace=True),
                 
                 nn.Conv2d(128,256,3,stride=2,padding=1),
                 nn.InstanceNorm2d(256),
                 nn.ReLU(inplace=True),
                 #---------->16*16
                 nn.Conv2d(256,512,3,stride=2,padding=1),
                 nn.InstanceNorm2d(512),
                 nn.ReLU(inplace=True),
                 
                 #---------->16*16
                 nn.Conv2d(512,1,3,padding=1)
                 # nn.ZeroPad2d((1, 0, 1, 0)),
                 # nn.Conv2d(512, 1, 4, padding=1)
                 ]                
        self.model=nn.Sequential(*model)

    def forward(self, img):  
        return self.model(img)
