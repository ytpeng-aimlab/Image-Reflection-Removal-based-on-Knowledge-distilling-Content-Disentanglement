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
import lpips
import math
#from IPython.display import Image
import sys
import torch.nn.functional as F
#sys.path.append("RAdam-master")
from time import process_time
import torchvision.models as models
from torch.nn import L1Loss



#from radam import RAdam
inference_mode = 1
n_epochs = 3505
learn_T1 = 0.0001
learn_R1 = 0.0001

learn_S = 0.0001
batch_size = 1
b1 = 0.5 ; b2 = 0.999 

n_threads = 0

root = r'C:\Users\encor\Desktop\[SPL]Reflection\dataset'


L1_loss = torch.nn.L1Loss()
Reconstruct_loss = torch.nn.L1Loss()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_student_state_path =  r'C:\Users\encor\Desktop\[SPL]Reflection\weighted\student_loss120_s'
load_teacher_T_state_path = None #r'/home/ytpeng0418/SIRR/T_weighted/teacher_loss120'
load_teacher_R_state_path =  None  #r'/home/ytpeng0418/SIRR/R_weighted/teacher_loss120'



teacher_T = Block_version_5.T_teacher_net()
teacher_T = nn.DataParallel(teacher_T)

student = Block_version_5.student_net()
student = nn.DataParallel(student)



teacher_R = Block_version_5.R_teacher_net()
teacher_R = nn.DataParallel(teacher_R)

# +
if inference_mode:  

    batch_size = 1 
    
    pth = torch.load(load_student_state_path) ;  student.load_state_dict(pth) 
    
    teacher_T.eval() ; teacher_T.to(device)

    teacher_R.eval() ; teacher_R.to(device)
   
    student.eval() ; student.to(device)
    
else:
    teacher_T.train() ; teacher_T.to(device)
    
    teacher_R.train() ; teacher_R.to(device)

    student.train() ; student.to(device)




if load_teacher_T_state_path: 
    pth = torch.load(load_teacher_T_state_path) ; teacher_T.load_state_dict(pth) 
if load_teacher_R_state_path: 
    pth = torch.load(load_teacher_R_state_path) ; teacher_R.load_state_dict(pth)
if load_student_state_path: 
    pth = torch.load(load_student_state_path) ; student.load_state_dict(pth)


train_dataset = datasets( root , 'Pos_I' , 'Pos_T', transform ) # instantiate  dataset
train_dataloader = DataLoader(train_dataset, batch_size = batch_size , shuffle = False , num_workers=0 )   

total_sc = 0 
for epoch in range(1):
        #start = process_time()
     
    for batch_idx , (reflect_image , clean_image) in enumerate(train_dataloader):
        with torch.no_grad():
            reflect_image = reflect_image.to(device)
            clean_image = clean_image.to(device)  
                
               
                
            sout_share1,  sout_share2 , sout_share3 , sout_share4 , \
            sout_T1 , sout_T2 , sout_T3 , sout_T4 , \
            sout_R1 , sout_R2 , sout_R3 , sout_R4 , \
            student_T_out , student_R_out = student( reflect_image )  
            
            loss_fn = lpips.LPIPS(net='alex')
            d = loss_fn.forward(student_T_out.detach().cpu(),clean_image.cpu())            
            total_sc = total_sc + d
            print( batch_idx )
          #  plt.imsave(rf'C:\Users\encor\Desktop\[SPL]Reflection\save_re_img\{batch_idx}.png' , np.transpose(
          #              np.clip(reflect_image.detach()[0,:,:,:].cpu().numpy(), 0, 1) , ( 1 , 2 , 0) ) ) 
          # plt.imsave(rf'C:\Users\encor\Desktop\[SPL]Reflection\save_clean_img\{batch_idx}.png' , np.transpose(
          #              np.clip(clean_image.detach()[0,:,:,:].cpu().numpy(), 0, 1) , ( 1 , 2 , 0) ) ) 
            plt.imsave(rf'C:\Users\encor\Desktop\[SPL]Reflection\save_img\{batch_idx}.png' , np.transpose(
                        np.clip(student_T_out.detach()[0,:,:,:].cpu().numpy(), 0, 1) , ( 1 , 2 , 0) ) ) 

print( f'Average score of Lpips is {total_sc / 199}' ) 











