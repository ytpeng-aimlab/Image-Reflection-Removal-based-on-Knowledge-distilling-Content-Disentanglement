# +
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageFile
from pathlib import Path
from torchvision import transforms
from torch.utils.data.sampler import BatchSampler
import Block_version_5
#import visdom
import PIL
import matplotlib.pyplot as plt
import os
import argparse

#from dataloader_sys import *
from dataloader import *
#import lpips
import math
# from IPython.display import Image
import sys
import torch.nn.functional as F
#sys.path.append("RAdam-master")
from time import process_time
import torchvision.models as models
from torch.nn import L1Loss
# -






#from radam import RAdam
inference_mode = None
n_epochs = 120
learn_T1 = 0.0001
learn_R1 = 0.0001

learn_S = 0.0001
batch_size = 1
b1 = 0.5 ; b2 = 0.999 

n_threads = 0

# root = r'/home/ytpeng0418/SIRR' 
root = 'train_dataset'


L1_loss = torch.nn.L1Loss()
Reconstruct_loss = torch.nn.L1Loss()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_student_state_path = None #r'/home/ytpeng0418/SIRR/ICIP/S_Weighted/recon+mimic+f'
#load_teacher_T_state_path = None #r'C:\Users\Admin\Desktop\Distilling\Distilling\teacher_T_weighted\teacher_loss30'
load_teacher_R_state_path = None #r'/home/ytpeng0418/SIRR/ICIP/R_Weighted/teacher_loss65'



# +
#teacher_T = Block_version_5.T_teacher_net()
#teacher_T = nn.DataParallel(teacher_T)
# -

teacher_R = Block_version_5.R_teacher_net()
teacher_R = nn.DataParallel(teacher_R)

# +
#teacher_I = Block_version_4.I_teacher_net()
#teacher_I = nn.DataParallel(teacher_I)
# -

student = Block_version_5.student_net()
student = nn.DataParallel(student)



# +
if inference_mode:  

    batch_size = 1 
    
    pth = torch.load(load_student_state_path) ;  student.load_state_dict(pth) 
    
   # teacher_T.eval() ; teacher_T.to(device)

    teacher_R.eval() ; teacher_R.to(device)
   
    student.eval() ; student.to(device)
    
else:
   # teacher_T.train() ; teacher_T.to(device)
    
    teacher_R.train() ; teacher_R.to(device)

    student.train() ; student.to(device)
# -



#if load_teacher_T_state_path: 
#    pth = torch.load(load_teacher_T_state_path) ; teacher_T.load_state_dict(pth) 
if load_teacher_R_state_path: 
    pth = torch.load(load_teacher_R_state_path) ; teacher_R.load_state_dict(pth)
if load_student_state_path: 
    pth = torch.load(load_student_state_path) ; student.load_state_dict(pth)


# train_dataset = datasets( root , 'r' , 't', transform ) # instantiate  dataset
train_dataset = datasets( root , 'input' , 'gt', transform ) # instantiate  dataset
train_dataloader = DataLoader(train_dataset, batch_size = batch_size , shuffle = False , num_workers=0 )   

#optimizer_teacher_T = torch.optim.Adam(teacher_T.parameters(), lr = learn_T1, betas=(b1, b2))
optimizer_teacher_R = torch.optim.Adam(teacher_R.parameters(), lr = learn_R1, betas=(b1, b2))
#optimizer_teacher_I = torch.optim.Adam(teacher_I.parameters(), lr = learn_I1, betas=(b1, b2))
optimizer_student = torch.optim.Adam(student.parameters(), lr = learn_S, betas=(b1, b2))




# +
if inference_mode == None:
    for epoch in range(n_epochs):
        
        
       
        for batch_idx , (reflect_image , clean_image) in enumerate(train_dataloader):
            
            #optimizer_teacher_T.zero_grad()
            optimizer_teacher_R.zero_grad()
            optimizer_student.zero_grad()
            
            
            reflect_image = reflect_image.to(device)
            clean_image = clean_image.to(device)
            
            compoment_image = reflect_image - clean_image  
            
            #teacher_T_out1,  teacher_T_out2 , teacher_T_out3, teacher_T_out4 , \
            #teacher_T_out5,  teacher_T_out6 , teacher_T_out7, teacher_T_out8 , \
            #teacher_T_out = teacher_T( clean_image )
            
           # teacher_T_loss = L1_loss( teacher_T_out , clean_image )
            
            teacher_R_out1,  teacher_R_out2 , teacher_R_out3 , teacher_R_out4  , \
            teacher_R_out5,  teacher_R_out6 , teacher_R_out7 , teacher_R_out8 , \
            teacher_R_out = teacher_R( compoment_image )
            
            teacher_R_loss = L1_loss( teacher_R_out , compoment_image )
            
            #teacher_I_out1,  teacher_I_out2 , teacher_I_out3 , teacher_I_out4  , teacher_I_out = teacher_I( reflect_image )
            
            #teacher_I_loss = L1_loss( teacher_I_out , reflect_image ) 
            
            sout_share1,  sout_share2 , sout_share3 , sout_share4 , \
            sout_T1 , sout_T2 , sout_T3 , sout_T4 , \
            sout_R1 , sout_R2 , sout_R3 , sout_R4 , \
            student_T_out , student_R_out = student( reflect_image )
        
            
            

            #fea_loss = 0
            #Perceptual_loss_T = 0 
            #Perceptual_loss_R = 0 ;
            #color_loss = 0 
            #f_loss_T = feat_loss(student_T_out , clean_image)
            #f_loss_R = feat_loss(student_R_out , compoment_image)
            
            #fea_loss = f_loss_R + f_loss_T
            
            #dcl_loss , dc = DCLoss(student_T_out)
            #student_R_feat = Block_version_4.get_features( student_T_out , vgg )
            #student_T_feat = Block_version_4.get_features( student_R_out , vgg )
            #clean_img_feat = Block_version_4.get_features( clean_image, vgg )
            #reflect_img_feat = Block_version_4.get_features( compoment_image, vgg )
            
            #max_version_s#weight = [1 , 0.7 , 0.2 , 0.2 , 0.2 ]
            #for num in range(5):
            #    Perceptual_loss_T = Perceptual_loss_T + torch.mean((student_T_feat[str(num+1)] - clean_img_feat[str(num+1)])**2) #* weight[num]
            #    Perceptual_loss_R = Perceptual_loss_R + torch.mean((student_R_feat[str(num+1)] - reflect_img_feat[str(num+1)])**2) 
            
            #RM_T_loss_1 = L1_loss( teacher_T_out1.detach() , sout_share1[:,64:,:,:] ) ; RM_T_loss_2 = L1_loss( teacher_T_out2.detach() , sout_share2[:,64:,:,:] ) 
            #RM_T_loss_3 = L1_loss( teacher_T_out3.detach() , sout_share3[:,64:,:,:]  ) ; RM_T_loss_4 = L1_loss( teacher_T_out4.detach() ,sout_share4[:,64:,:,:] ) 
            #RM_T_loss_5 = L1_loss( teacher_T_out5.detach() , sout_T1 ) ;  RM_T_loss_6 = L1_loss( teacher_T_out6.detach() , sout_T2 ) ; 
            #RM_T_loss_7 = L1_loss( teacher_T_out7.detach() , sout_T3 ) ;  RM_T_loss_8 = L1_loss( teacher_T_out8.detach() , sout_T4 ) ;
            #RM_loss_5 = L1_loss( teacher_T_out5.detach() , sout5[:,64:,:,:] ) ; RM_loss_6 = L1_loss( teacher_T_out6.detach() , sout6[:,64:,:,:] ) 
            
          
            
            RM_R_loss_1 = L1_loss( teacher_R_out1.detach() , sout_share1[:,:64,:,:] ) ; RM_R_loss_2 = L1_loss( teacher_R_out2.detach() , sout_share2[:,:64,:,:] ) 
            RM_R_loss_3 = L1_loss( teacher_R_out3.detach() , sout_share3[:,:64,:,:] ) ; RM_R_loss_4 = L1_loss( teacher_R_out4.detach() , sout_share4[:,:64,:,:] ) 
            RM_R_loss_5 = L1_loss( teacher_R_out5.detach() , sout_R1 ) ; RM_R_loss_6 = L1_loss( teacher_R_out6.detach() , sout_R2 ) 
            RM_R_loss_7 = L1_loss( teacher_R_out7.detach() , sout_R3 ) ; RM_R_loss_8 = L1_loss( teacher_R_out8.detach() , sout_R4 ) 
            
            #RM_loss_55 = L1_loss( teacher_R_out5.detach() , sout5[:,:64,:,:] ) ; RM_loss_66 = L1_loss( teacher_R_out6.detach() , sout6[:,:64,:,:] )
            #print(student_out.detach().cpu().numpy().shape)
            
            #RM_I_loss_1 = L1_loss( teacher_I_out1.detach() , merge_1 ) ; RM_I_loss_2 = L1_loss( teacher_I_out2.detach() , merge_2 ) 
            #RM_I_loss_3 = L1_loss( teacher_I_out3.detach() , merge_3 ) ; RM_I_loss_4 = L1_loss( teacher_I_out4.detach() , merge_4 )
            
            #RM_Top =  RM_T_loss_1 +  RM_T_loss_2 +  RM_T_loss_3 + RM_T_loss_4 + RM_T_loss_5 + RM_T_loss_6 + RM_T_loss_7 + RM_T_loss_8
            
            RM_Bot =  RM_R_loss_1 +  RM_R_loss_2 +  RM_R_loss_3 + RM_R_loss_4 + RM_R_loss_5 + RM_R_loss_6 + RM_R_loss_7 + RM_R_loss_8
            
            #RM_Blend = RM_I_loss_1 + RM_I_loss_2 + RM_I_loss_3 + RM_I_loss_4
            
            
            #I_T_loss = L1_loss( reflect_image - teacher_T_out , teacher_R_out)
            
            #I_Tout_loss = L1_loss( reflect_image - teacher_T_out , student_R_out )
            
            I_OUT_T_loss = L1_loss( reflect_image - student_T_out , teacher_R_out )
            
            Res_loss = Reconstruct_loss(student_R_out , compoment_image ) + Reconstruct_loss(student_T_out , clean_image ) + \
            0.3 * (  RM_Bot )  + 0.3 * I_OUT_T_loss   
            
            student_loss = Res_loss #+ Perceptual_loss 
            teacher_R_loss.backward(retain_graph = True )
            #teacher_I_loss.backward(retain_graph = True )
            #vis.images( np.clip(student_T_out.detach().cpu().numpy() , 0 , 1 ), nrow=2 ,  win='random2' , opts={'title':'student_T_out'})
            #vis.images( np.clip(clean_image.detach().cpu().numpy() , 0 , 1 ), nrow=2 ,  win='random3' , opts={'title':'clean_image'})
            #vis.images( student_out.detach()[:,:3,:,:].cpu().numpy() , nrow=2 ,  win='random4' , opts={'title':'student_R_out'})
            #vis.images( np.clip(reflect_image.detach().cpu().numpy() , 0 , 1 ), nrow=2 ,  win='random5' , opts={'title':'reflect_image'})
            #vis.images( np.clip(teacher_T_out.detach().cpu().numpy() , 0 , 1 ), nrow=2 ,  win='random6' , opts={'title':'Teachet_T_image'})
            #vis.images( np.clip(reflect_image.detach().cpu().numpy() , 0 , 1 ), nrow=2 ,  win='random5' , opts={'title':'reflect_image'})
            student_loss.backward(retain_graph = True )
            optimizer_teacher_R.step() #; optimizer_teacher_I.step()
            optimizer_student.step()
            
            
            #Image.fromarray(cv2.cvtColor( (blended_image * 255).astype(np.uint8) , cv2.COLOR_BGR2RGB ) )
            if batch_idx % 20 == 0 :
                plt.subplot(2, 2, 1)
                plt.title('Student_T_out')
                plt.imshow(np.transpose( np.clip( student_T_out.detach().cpu().numpy()[0,:,:,:]  , 0 , 1 ) , (1, 2, 0 ) ))
                plt.subplot(2, 2, 2)
                plt.title('Student_R_out')
                plt.imshow(np.transpose( np.clip( student_R_out.detach().cpu().numpy()[0,:,:,:]  , 0 , 1 ) , (1, 2, 0 ) ))
                plt.subplot(2, 2, 3)  
                plt.title('Teacher_R_out')
                plt.imshow(np.transpose( np.clip( teacher_R_out.detach().cpu().numpy()[0,:,:,:]  , 0 , 1 ) , (1, 2, 0 ) ))
                plt.subplot(2, 2, 4)       
                plt.title('reflect_image')
                plt.imshow(np.transpose( np.clip( reflect_image.cpu().numpy()[0,:,:,:]  , 0 , 1 ) , (1, 2, 0 ) ))
           
                plt.show()
              
            print(f'epoch: {epoch} batch {batch_idx}\n')  
        if epoch % 5 == 0:
           # plt.imsave(rf'/home/stitch0312/Distilling/Distilling/attention/att1/{batch_idx_X}.png' , np.transpose(
           #             np.clip(att1.detach()[0,:,:,0].cpu().numpy(), 0, 1) , ( 1 , 2 , 0) ) ) 
           # plt.imsave(rf'/home/stitch0312/Distilling/Distilling/attention/att1/{batch_idx_X}.png' , np.transpose(
           #             np.clip(att2.detach()[0,:,:,0].cpu().numpy(), 0, 1) , ( 1 , 2 , 0) ) ) 
           # plt.imsave(rf'/home/stitch0312/Distilling/Distilling/attention/att1/{batch_idx_X}.png' , np.transpose(
           #             np.clip(att3.detach()[0,:,:,0].cpu().numpy(), 0, 1) , ( 1 , 2 , 0) ) ) 
           # plt.imsave(rf'/home/stitch0312/Distilling/Distilling/attention/att1/{batch_idx_X}.png' , np.transpose(
           #             np.clip(att4.detach()[0,:,:,0].cpu().numpy(), 0, 1) , ( 1 , 2 , 0) )) 
            torch.save(teacher_R.state_dict(), rf'./weights/teacher_r/teacher_loss{epoch}')
            #torch.save(teacher_T.state_dict(), rf'/home/ytpeng0418/SIRR/ICIP/T_weighted/teacher_loss{epoch}')
            #torch.save(teacher_I.state_dict(), rf'/home/stitch0312/Distilling/Distilling/teacher_T_weighted/teacher_loss{epoch}')
            torch.save(student.state_dict(), rf'./weights/student/student_loss{epoch}')  
          
              
                
        print(f'Iteration {epoch},Student Loss: {Res_loss}, Teacher_R Loss: {teacher_R_loss}\n')
else:
    for epoch in range(1):
        start = process_time()
        for batch_idx , (reflect_image , clean_image) in enumerate(train_dataloader):
            with torch.no_grad():
                reflect_image = reflect_image.to(device)
              
                
               
                
                sout_share1,  sout_share2 , sout_share3 , sout_share4 , \
                sout_T1 , sout_T2 , sout_T3 , sout_T4 , \
                sout_R1 , sout_R2 , sout_R3 , sout_R4 , \
                student_T_out , student_R_out = student( reflect_image )  



# -











