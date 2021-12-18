# +
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from PIL import Image
from torchvision import  transforms as T
import os
#import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# -

transform = T.Compose([
    T.Resize([256, 256]),
    #T.RandomResizedCrop(224),
    #T.RandomHorizontalFlip(),
    T.ToTensor(),
    ])

class RandomCrop(object):
    """Crop randomly the image in a sample.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, I , T):
        image, landmarks = I , T 
        
        h, w = image.shape[:2]
        min_a = min(h, w)
        self.output_size = (min_a * 7 // 10, min_a * 7 // 10)
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks[top: top + new_h,
                      left: left + new_w]

        return image , landmarks



class datasets( Dataset ): 
    def __init__(self, root, img, Gt_img, transforms = None) :
        self.root_dir = root 
        self.img_folder = img
        self.Gt_img_folder = Gt_img
        
        self.path = os.path.join( self.root_dir , self.img_folder )
        self.img_path = os.listdir( self.path )
        #self.img_path = sort(self.img_path)
        
        self.Gt_path = os.path.join( self.root_dir , self.Gt_img_folder )
        self.Gt_img_path = os.listdir( self.Gt_path )
        #self.Gt_img_path = sort(self.Gt_path)
        self.Gt_img_path.sort()
        self.img_path.sort()
        
        #self.crop = RandomCrop(224)
        self.transforms = transforms
      
        
        self.images = []
        self.Gt_images = []
        
        for f_name in tqdm(self.img_path):
            with open( os.path.join(self.path , f_name ) , "rb") as f:
                self.images.append(  np.array( Image.open( f ).convert('RGB') ) )
        for f_name in tqdm(self.Gt_img_path):
            with open( os.path.join(self.Gt_path , f_name ) , "rb") as f:
                self.Gt_images.append(  np.array( Image.open( f ).convert('RGB') ) )
        
        #for f_name in tqdm(self.Gt_img_path):
          #  with open( os.path.join(self.Gt_path , f_name ) , "rb") as f:
           #    self.Gt_images.append(  np.asarray( bytearray( f.read() ) , dtype = "uint8"   ))  
       # Gt_img = Image.open( img_Gt_path_name ).convert('RGB'))
       # for f_name in tqdm(self.Gt_img_path) :
        #    with open(os.path.join(self.Gt_path , f_name ) , "rb") as f:
         #        self.Gt_images.append( f.read() )
        #self.images = np.array(self.images)
       # self.Gt_images = np.array(self.Gt_images)
    def __getitem__(self, idx ):

       # img_name = self.img_path[ idx ]
       # img_Gt_name = self.Gt_img_path[ idx ]

       # img_input_path_name = os.path.join( self.root_dir , self.img_folder , img_name )
       # img_Gt_path_name = os.path.join( self.root_dir , self.Gt_img_folder , img_Gt_name )
        
      
        
      #  img = Image.open( img_input_path_name ).convert('RGB')
       # Gt_img = Image.open( img_Gt_path_name ).convert('RGB')
        #img , Gt_img = crop(self.images[idx] , self.Gt_images)
        
        img = Image.fromarray(np.uint8(self.images[idx])) 
        
        Gt_img = Image.fromarray(np.uint8(self.Gt_images[idx])) 
        
        if self.transforms:
            img = self.transforms(img)
            Gt_img = self.transforms(Gt_img)
    
        return img , Gt_img
        
    def __len__(self):
        return len(self.img_path)

# +
#root = r'/home/stitch0312/Distilling/Distilling/DSLR/train250' 

# +
#train_dataset = datasets( root , 'r' , 't', transform ) # instantiate  dataset
#train_dataloader = DataLoader(train_dataset, batch_size = 4 , shuffle = False , num_workers=0 )   
#plt.imshow(np.transpose( np.clip( train_dataset[13000][0].numpy() , 0 , 1 ) , (1, 2, 0 ) ))
#plt.imshow(np.transpose( np.clip( train_dataset[13000][1].numpy() , 0 , 1 ) , (1, 2, 0 ) ))

# +
#plt.imshow(np.transpose( np.clip( train_dataset[14200][0].numpy() , 0 , 1 ) , (1, 2, 0 ) ))


# +
#plt.imshow(np.transpose( np.clip( train_dataset[14200][1].numpy() , 0 , 1 ) , (1, 2, 0 ) ))
# -


