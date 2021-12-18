# +
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torchvision import transforms, models, utils
#import customize
import torch.nn.functional as F
# -



class SPBlock(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(SPBlock, self).__init__()
        midplanes = int(outplanes//2)


        self.pool_1_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_1_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv_1_h = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        self.conv_1_w = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)

        self.pool_3_h = nn.AdaptiveAvgPool2d((None, 3))
        self.pool_3_w = nn.AdaptiveAvgPool2d((3, None))
        self.conv_3_h = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.conv_3_w = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)

        self.pool_5_h = nn.AdaptiveAvgPool2d((None, 5))
        self.pool_5_w = nn.AdaptiveAvgPool2d((5, None))
        self.conv_5_h = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.conv_5_w = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)

        self.pool_7_h = nn.AdaptiveAvgPool2d((None, 7))
        self.pool_7_w = nn.AdaptiveAvgPool2d((7, None))
        self.conv_7_h = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)
        self.conv_7_w = nn.Conv2d(inplanes, midplanes, kernel_size=3, padding=1, bias=False)

        self.fuse_conv = nn.Conv2d(midplanes * 4, midplanes, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)

        self.conv_final = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)

        self.mask_conv_1 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1)
        self.mask_relu  = nn.ReLU(inplace=False)
        self.mask_conv_2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1)


    def forward(self, x):
        _, _, h, w = x.size()
        x_1_h = self.pool_1_h(x)
        x_1_h = self.conv_1_h(x_1_h)
        x_1_h = x_1_h.expand(-1, -1, h, w)
        #x1 = F.interpolate(x1, (h, w))

        x_1_w = self.pool_1_w(x)
        x_1_w = self.conv_1_w(x_1_w)
        x_1_w = x_1_w.expand(-1, -1, h, w)
        #x2 = F.interpolate(x2, (h, w))

        x_3_h = self.pool_3_h(x)
        x_3_h = self.conv_3_h(x_3_h)
        x_3_h = F.interpolate(x_3_h, (h, w))

        x_3_w = self.pool_3_w(x)
        x_3_w = self.conv_3_w(x_3_w)
        x_3_w = F.interpolate(x_3_w, (h, w))

        x_5_h = self.pool_5_h(x)
        x_5_h = self.conv_5_h(x_5_h)
        x_5_h = F.interpolate(x_5_h, (h, w))

        x_5_w = self.pool_5_w(x)
        x_5_w = self.conv_5_w(x_5_w)
        x_5_w = F.interpolate(x_5_w, (h, w))

        x_7_h = self.pool_7_h(x)
        x_7_h = self.conv_7_h(x_7_h)
        x_7_h = F.interpolate(x_7_h, (h, w))

        x_7_w = self.pool_7_w(x)
        x_7_w = self.conv_7_w(x_7_w)
        x_7_w = F.interpolate(x_7_w, (h, w))

        hx = self.relu(self.fuse_conv(torch.cat((x_1_h + x_1_w, x_3_h + x_3_w, x_5_h + x_5_w, x_7_h + x_7_w), dim = 1)))
        mask_1 = self.conv_final(hx).sigmoid()
        out1 = x * mask_1

       
        hx = self.mask_relu(self.mask_conv_1(out1))
        mask_2 = self.mask_conv_2(hx).sigmoid()
        hx = out1 * mask_2

        return hx , mask_2


class RIR( nn.Module ):
    def __init__( self , in_c ):
        super(RIR , self).__init__()
        self.Layer1 = SPBlock(in_c , in_c ).to(device)
        self.Layer2 = SPBlock(in_c , in_c ).to(device)
        self.Layer3 = SPBlock(in_c , in_c ).to(device)
        
        
        self.con = nn.Conv2d( in_c , in_c ,  kernel_size=  3, stride = 1, padding = 1, bias = False)
        
    def forward( self , x ):
        out1 , m1= self.Layer1( x )
        out2 , m2= self.Layer2( out1 + x )
        out3 , m3= self.Layer3( out2 + out1 )
        final = self.con( out3 + out2 )
        

       
        return final + x , m1 , m2 , m3


class student_net( nn.Module ):
    def __init__( self ):   
        super(student_net , self).__init__()
        
     
        self.down = downsampling()
        
        self.share_Block_1 = RIR( 128 )
        self.share_Block_2 = RIR( 128 )
        self.share_Block_3 = RIR( 128 )
        self.share_Block_4 = RIR( 128 )
        
 
        
        
        
        self.BlockT_1 = RIR( 64 )
        self.BlockT_2 = RIR( 64 )
        self.BlockT_3 = RIR( 64 )
        self.BlockT_4 = RIR( 64 )
        
   

        
        self.up_T = upsampling_T()
        
        self.BlockR_1 = RIR( 64 )
        self.BlockR_2 = RIR( 64 )
        self.BlockR_3 = RIR( 64 )
        self.BlockR_4 = RIR( 64 )
        
 
        #self.trans_con_T_1_1 = nn.Conv2d( 64 , 3 ,  kernel_size=  3, stride = 1, padding = 1, bias = False)
        #self.trans_con_T_1_2 = nn.Conv2d( 3 , 64 ,  kernel_size=  3, stride = 1, padding = 1, bias = False)
        #self.trans_con_T_2_1 = nn.Conv2d( 64 , 3 ,  kernel_size=  3, stride = 1, padding = 1, bias = False)
        #self.trans_con_T_2_2 = nn.Conv2d( 3 , 64 ,  kernel_size=  3, stride = 1, padding = 1, bias = False)
        #self.trans_con_R_1_1 = nn.Conv2d( 64 , 3 ,  kernel_size=  3, stride = 1, padding = 1, bias = False)
        #self.trans_con_R_1_2 = nn.Conv2d( 3 , 64 ,  kernel_size=  3, stride = 1, padding = 1, bias = False)
        #self.trans_con_R_2_1 = nn.Conv2d( 64 , 3 ,  kernel_size=  3, stride = 1, padding = 1, bias = False)
        #self.trans_con_R_2_2 = nn.Conv2d( 3 , 64 ,  kernel_size=  3, stride = 1, padding = 1, bias = False)
        self.up_R = upsampling_T()
        
       
    def forward( self , x ):
        
        out0 = self.down( x )
        
        share_out1 , _ , _ , _ = self.share_Block_1( out0 )
        share_out2 , _ , _ , _ = self.share_Block_2( share_out1 )
        share_out3 , _ , _ , _ = self.share_Block_3( share_out2 )
        share_out4 , _ , _ , _ = self.share_Block_4( share_out3 )
        
      
        
        #T_Image_1 = self.trans_con_T_1_1( share_out4[ : , 64: , : , : ] ) 
        #R_Image_1 = self.trans_con_R_1_1( share_out4[ : , :64 , : , : ] ) 
    
        #out_R_1 = self.trans_con_T_1_2( x - T_Image_1 )
        #out_T_1 = self.trans_con_R_1_2( x - R_Image_1 )
        
        out_T1 , _ , _ , att1 = self.BlockT_1( share_out4[ : , 64: , : , : ] )
        out_T2 , _ , _ , att2 = self.BlockT_2( out_T1 )
        
        out_R1 , _ , _ , _ = self.BlockR_1( share_out4[ : , :64 , : , : ])
        out_R2 , _ , _ , _ = self.BlockR_2( out_R1 )
        
        
        #T_Image_2 = self.trans_con_T_2_1( out_T2 ) 
        #R_Image_2 = self.trans_con_R_2_1( out_R2 ) 
        
        #out_R_2 = self.trans_con_T_2_2( x - T_Image_2 )
        #out_T_2 = self.trans_con_R_2_2( x - R_Image_2 )
        
        out_T3 , _ , _ , att3 = self.BlockT_3( out_T2 )
        out_T4 , _ , _ , att4 = self.BlockT_4( out_T3 )
        
        out_R3 , _ , _ , _ = self.BlockR_3( out_R2 )
        out_R4 , _ , _ , _ = self.BlockR_4( out_R3 )
        
    
                                    
        
        final_out_T = self.up_T( out_T4 )  
        final_out_R = self.up_R( out_R4 ) 
        
        return share_out1 , share_out2 , share_out3 , share_out4 , out_T1 , out_T2 , out_T3 , out_T4  ,  out_R1 , out_R2 , out_R3 , out_R4 , final_out_T , final_out_R #, f
class downsampling( nn.Module ):    
    def __init__( self ):
        super(downsampling , self).__init__()
        self.Layer1 = nn.Conv2d( 3 , 128 ,  kernel_size=  3, stride = 1, padding = 1, bias = False)
        self.Layer2 = nn.Conv2d( 128 , 128 ,  kernel_size=  3, stride = 1, padding = 1, bias = False)
        self.relu = nn.ReLU(inplace = False)
        
    def forward( self , x ):
        out1 = self.Layer1( x )
        out2 = self.relu( out1 )
        out3 = self.Layer2( out2 )
        final = self.relu( out3 )
        
        return final

class downsampling_T( nn.Module ):
    def __init__( self ):
        super(downsampling_T , self).__init__()
        self.Layer1 = nn.Conv2d( 3 , 64 ,  kernel_size=  3, stride = 1, padding = 1, bias = False)
        self.Layer2 = nn.Conv2d( 64 , 64 ,  kernel_size=  3, stride = 1, padding = 1, bias = False)
        self.relu = nn.ReLU(inplace = False)
        
    def forward( self , x ):
        out1 = self.Layer1( x )
        out2 = self.relu( out1 )
        out3 = self.Layer2( out2 )
        final = self.relu( out3 )
        
        return final

class upsampling( nn.Module ):
    def __init__( self ):
        super(upsampling , self).__init__()
        self.Layer1 = nn.Conv2d( 128 , 128 ,  kernel_size=  3, stride = 1, padding = 1, bias = False)
        self.Layer2 = nn.Conv2d( 128 , 128 ,  kernel_size=  3, stride = 1, padding = 1, bias = False)
        self.relu = nn.ReLU(inplace = False)
        #self.up1 = nn.Upsample(scale_factor = 4 , mode= 'bilinear' )
        self.Layer3 = nn.Conv2d( 128 ,  6 ,  kernel_size=  3, stride = 1, padding = 1, bias = False)
        self.tanh = nn.Tanh()
        
        
    def forward( self , x ):
        out1 = self.Layer1( x )
        out2 = self.Layer2( out1 )
        out3 = self.relu( out2 )
        #out4 = self.up1( out3 )
        out4 = self.Layer3( out3 )
        final= self.tanh( out4 )
        
        return final
class upsampling_T( nn.Module ):
    def __init__( self ):
        super(upsampling_T , self).__init__()
        self.Layer1 = nn.Conv2d( 64 , 64 ,  kernel_size=  3, stride = 1, padding = 1, bias = False)
        self.Layer2 = nn.Conv2d( 64 , 64 ,  kernel_size=  3, stride = 1, padding = 1, bias = False)
        self.relu = nn.ReLU(inplace = False)
        #self.up1 = nn.Upsample(scale_factor = 4 , mode= 'bilinear' )
        self.Layer3 = nn.Conv2d( 64 , 3 ,  kernel_size=  3, stride = 1, padding = 1, bias = False)
        self.tanh = nn.Tanh()
        
        
    def forward( self , x ):
        out1 = self.Layer1( x )
        out2 = self.Layer2( out1 )
        out3 = self.relu( out2 )
        #out4 = self.up1( out3 )
        out4 = self.Layer3( out3 )
        final= self.tanh( out4 )
        
        return final




class backbone( nn.Module ):
    def __init__( self ):
        super(backbone , self ).__init__()
        self.Layer1 = nn.Conv2d( 64 , 64 ,  kernel_size=  3, stride = 1, padding = 1, bias = False)
        self.Layer2 = nn.Conv2d( 64 , 64 ,  kernel_size=  3, stride = 1, padding = 1, bias = False)
        self.relu = nn.ReLU(inplace = False)
    def forward( self , x ):
        out1 = self.Layer1( x )
        out2 = self.relu( out1 )
        final = self.Layer2( out2 )
        final = final + x
        return final

class T_teacher_net( nn.Module ):
    def __init__( self ):
        super(T_teacher_net , self).__init__()
        self.down = downsampling_T()
        self.ResBlock1 = backbone()
        self.ResBlock2 = backbone()
        self.ResBlock3 = backbone()
        self.ResBlock4 = backbone()
        self.ResBlock5 = backbone()
        self.ResBlock6 = backbone()
        self.ResBlock7 = backbone()
        self.ResBlock8 = backbone()
 
 
        self.up = upsampling_T()
    def forward( self , x ):
        out0 = self.down( x )
        out1 = self.ResBlock1( out0 )
        out2 = self.ResBlock2( out1 )
        out3 = self.ResBlock3( out2 )
        out4 = self.ResBlock4( out3 )
        out5 = self.ResBlock5( out4 )
        out6 = self.ResBlock6( out5 )
        out7 = self.ResBlock7( out6 )
        out8 = self.ResBlock8( out7 )
       
     
        final= self.up( out8 )
        
        return out1,  out2 , out3 , out4 , out5 , out6 , out7 , out8 , final

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
#model = teacher_net().to(device)
#mode2 = student_net().to(device)
#summary( model , (3 , 256,  256))
#summary( mode2 , (3 , 256,  256))


def get_model():
  vgg19 = models.vgg19( pretrained= True ).features
  
  for param in vgg19.parameters():
    param.requires_grad_(False)
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  vgg19.to(device)
  
  return vgg19

def get_features(image , model , layers = None ):

  if layers is None:
    layers = {
        '2': '1',
        '7': '2',
        '12':'3',
        '21':'4',
        '30':'5',
    }
  
  features = {}
  x = image
  
  for name, layer in model._modules.items():
    x = layer(x)

    if name in layers:
      features[layers[name]] = x
      
  return features


class R_teacher_net( nn.Module ):
    def __init__( self ):
        super(R_teacher_net , self).__init__()
        self.down = downsampling_T()
        self.ResBlock1 = backbone()
        self.ResBlock2 = backbone()
        self.ResBlock3 = backbone()
        self.ResBlock4 = backbone()
        self.ResBlock5 = backbone()
        self.ResBlock6 = backbone()
        self.ResBlock7 = backbone()
        self.ResBlock8 = backbone()
      
        self.up = upsampling_T()
    def forward( self , x ):
        out0 = self.down( x )
        out1 = self.ResBlock1( out0 )
        out2 = self.ResBlock2( out1 )
        out3 = self.ResBlock3( out2 )
        out4 = self.ResBlock4( out3 )
        out5 = self.ResBlock5( out4 )
        out6 = self.ResBlock6( out5 )
        out7 = self.ResBlock7( out6 )
        out8 = self.ResBlock8( out7 )
     
        
    
        final= self.up( out8 )
        
        return out1,  out2 , out3 , out4 , out5 , out6 , out7 , out8 ,  final

        

