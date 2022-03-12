# Image Reflection Removal based on Knowledge-distilling Content Disentanglement, **IEEE Signal Processing Letters**, 2022

<br>
## **Abstract:** <br>

_Single image reflection removal (SIRR) is an ill-posed and challenging problem that is practically essential to image enhancement. Inspired by knowledge distillation in deep learning, we tackle the SIRR problem by proposing a knowledge-distilling-based content disentangling model that can effectively decompose the transmission and reflection layers.  The experiments on benchmark SIRR datasets show that our method performs favorably against state-of-the-art SIRR methods._

![screen shot](https://user-images.githubusercontent.com/56446649/158000570-f78adc5f-8132-4aa2-93a6-535157ba9418.png)
 
## Environment <br>
  
*  Platforms: Windows 10 / cuda8.0 <br>
*  python: 3.7.6 / pytorch: 1.5 <br>
  

## Inference <br>
*  Change the `"root path"` and `"save image path"` in main.py. <br>
*  Move test images to /Test. <br>
*  Download the pretrained weight from https://drive.google.com/drive/folders/1OINXQIFDbQVfm82UKO12FCr_Mcjr2z0N and place it in /Weighted. <br>
*  Do the inference by running `"python main.py"`. <br>    


  
