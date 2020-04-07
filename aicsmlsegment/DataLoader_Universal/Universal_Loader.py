import numpy as np
import os
from tifffile import imread, imsave
from PIL import Image
import random

from torch import from_numpy
from aicsimageio import AICSImage
from random import shuffle
import time
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from sklearn.feature_extraction.image import extract_patches_2d
from aicsmlsegment.utils import simple_norm, background_sub

# For augmentation
from torch.nn.functional import grid_sample, affine_grid
from skimage.transform import resize
import torch
# import warnings # for grid_sample warning
# warnings.filterwarnings("once")

# CODE for generic loader
#   No augmentation = NOAUG,simply load data and convert to tensor
#   Augmentation code:
#       RR = Rotate by a random degree from 1 to 180
#       R4 = Rotate by 0, 90, 180, 270
#       FH = Flip Horizantally
#       FV = Flip Vertically
#       FD = Flip Depth (i.e., along z dim)
#       SS = Size Scaling by a ratio between -0.1 to 0.1 (TODO)
#       IJ = Intensity Jittering (TODO)
#       DD = Dense Deformation (TODO)


class RR_FH_M0(Dataset):

    def __init__(self, filenames, num_patch, size_in, size_out):

        self.img = []
        self.gt = []
        self.cmap = []

        padding = [(x-y)//2 for x,y in zip(size_in, size_out)]
        total_in_count = size_in[0] * size_in[1] * size_in[2]
        total_out_count = size_out[0] * size_out[1] * size_out[2]

        num_data = len(filenames)
        shuffle(filenames)
        num_patch_per_img = np.zeros((num_data,), dtype=int)
        if num_data >= num_patch:
            # all one
            num_patch_per_img[:num_patch]=1
        else: 
            basic_num = num_patch // num_data
            # assign each image the same number of patches to extract
            num_patch_per_img[:] = basic_num

            # assign one more patch to the first few images to achieve the total patch number
            num_patch_per_img[:(num_patch-basic_num*num_data)] = num_patch_per_img[:(num_patch-basic_num*num_data)] + 1

        for img_idx, fn in enumerate(filenames):

            if len(self.img)==num_patch:
                break

            label_reader = AICSImage(fn+'_GT.ome.tif')  #CZYX
            label = label_reader.data
            label = np.squeeze(label,axis=0) # 4-D after squeeze
            while(len(label.shape) != 4): # when the image is larger than 4D keep reducing dimensions
                label = np.squeeze(label,axis=0)

            # when the tif has only 1 channel, the loaded array may have falsely swaped dimensions (ZCYX). we want CZYX
            # (This may also happen in different OS or different package versions)
            # ASSUMPTION: we have more z slices than the number of channels 
            if label.shape[1]<label.shape[0]: 
                label = np.transpose(label,(1,0,2,3))

            input_reader = AICSImage(fn+'.ome.tif') #CZYX  #TODO: check size
            input_img = input_reader.data
            input_img = np.squeeze(input_img,axis=0)
            while(len(input_img.shape) > 4): # when the image is larger than 4D keep reducing dimensions
                input_img = np.squeeze(input_img,axis=0)
            if input_img.shape[1] < input_img.shape[0]:
                input_img = np.transpose(input_img,(1,0,2,3))
            

            costmap_reader = AICSImage(fn+'_CM.ome.tif') # ZYX
            costmap = costmap_reader.data
            costmap = np.squeeze(costmap,axis=0)
            if costmap.shape[0] == 1:
                costmap = np.squeeze(costmap,axis=0)
            elif costmap.shape[1] == 1:
                costmap = np.squeeze(costmap,axis=1)
            
            if costmap.shape[1] < costmap.shape[0]:
                costmap = np.transpose(costmap,(1,0,2,3))
                costmap = np.squeeze(costmap,axis=0)

            while(len(costmap.shape) != 3): # when the image is larger than 3D keep reducing dimensions
                costmap = np.squeeze(costmap,axis=0)

            img_pad0 = np.pad(input_img, ((0,0),(0,0),(padding[1],padding[1]),(padding[2],padding[2])), 'constant')
            raw = np.pad(img_pad0, ((0,0),(padding[0],padding[0]),(0,0),(0,0)), 'constant')

            cost_scale = costmap.max()
            if cost_scale<1: ## this should not happen, but just in case
                cost_scale = 1

            deg = random.randrange(1,180)
            flip_flag = random.random()

            for zz in range(label.shape[1]):

                for ci in range(label.shape[0]):
                    labi = label[ci,zz,:,:]
                    labi_pil = Image.fromarray(np.uint8(labi))
                    new_labi_pil = labi_pil.rotate(deg,resample=Image.NEAREST)
                    if flip_flag<0.5:
                        new_labi_pil = new_labi_pil.transpose(Image.FLIP_LEFT_RIGHT)
                    new_labi = np.array(new_labi_pil.convert('L'))
                    label[ci,zz,:,:] = new_labi.astype(int)

                cmap = costmap[zz,:,:]
                cmap_pil = Image.fromarray(np.uint8(255*(cmap/cost_scale)))
                new_cmap_pil = cmap_pil.rotate(deg,resample=Image.NEAREST)
                if flip_flag<0.5:
                    new_cmap_pil = new_cmap_pil.transpose(Image.FLIP_LEFT_RIGHT)
                new_cmap = np.array(new_cmap_pil.convert('L'))
                costmap[zz,:,:] = cost_scale*(new_cmap/255.0)

            for zz in range(raw.shape[1]):
                for ci in range(raw.shape[0]):
                    str_im = raw[ci,zz,:,:]
                    str_im_pil = Image.fromarray(np.uint8(str_im*255))
                    new_str_im_pil = str_im_pil.rotate(deg,resample=Image.BICUBIC)
                    if flip_flag<0.5:
                        new_str_im_pil = new_str_im_pil.transpose(Image.FLIP_LEFT_RIGHT)
                    new_str_image = np.array(new_str_im_pil.convert('L'))
                    raw[ci,zz,:,:] = (new_str_image.astype(float))/255.0 
            new_patch_num = 0
            
            while new_patch_num < num_patch_per_img[img_idx]:
                
                pz = random.randint(0, label.shape[1] - size_out[0])
                py = random.randint(0, label.shape[2] - size_out[1])
                px = random.randint(0, label.shape[3] - size_out[2])

                
                # check if this is a good crop
                ref_patch_cmap = costmap[pz:pz+size_out[0],py:py+size_out[1],px:px+size_out[2]]

                # confirmed good crop
                (self.img).append(raw[:,pz:pz+size_in[0],py:py+size_in[1],px:px+size_in[2]] )
                (self.gt).append(label[:,pz:pz+size_out[0],py:py+size_out[1],px:px+size_out[2]])
                (self.cmap).append(ref_patch_cmap)

                new_patch_num += 1

    def __getitem__(self, index):

        image_tensor = from_numpy(self.img[index].astype(float))
        cmap_tensor = from_numpy(self.cmap[index].astype(float))

        label_tensor = []
        if self.gt[index].shape[0]>0:
            for zz in range(self.gt[index].shape[0]):
                label_tensor.append(from_numpy(self.gt[index][zz,:,:,:].astype(float)).float())
        else: 
            label_tensor.append(from_numpy(self.gt[index].astype(float)).float())

        return image_tensor.float(), label_tensor, cmap_tensor.float()

    def __len__(self):
        return len(self.img)

class RR_FH_M0C(Dataset):

    def __init__(self, filenames, num_patch, size_in, size_out):

        self.img = []
        self.gt = []
        self.cmap = []

        padding = [(x-y)//2 for x,y in zip(size_in, size_out)]
        
        num_data = len(filenames)
        shuffle(filenames)

        num_trial_round = 0
        while len(self.img) < num_patch:

            # to avoid dead loop
            num_trial_round = num_trial_round + 1
            if num_trial_round > 2:
                break

            num_patch_to_obtain = num_patch - len(self.img)
            num_patch_per_img = np.zeros((num_data,), dtype=int)
            if num_data >= num_patch_to_obtain:
                # all one
                num_patch_per_img[:num_patch_to_obtain]=1
            else: 
                basic_num = num_patch_to_obtain // num_data
                # assign each image the same number of patches to extract
                num_patch_per_img[:] = basic_num

                # assign one more patch to the first few images to achieve the total patch number
                num_patch_per_img[:(num_patch_to_obtain-basic_num*num_data)] = num_patch_per_img[:(num_patch_to_obtain-basic_num*num_data)] + 1

        
            for img_idx, fn in enumerate(filenames):
                
                if len(self.img)==num_patch:
                    break

                label_reader = AICSImage(fn+'_GT.ome.tif')  #CZYX
                label = label_reader.data
                label = np.squeeze(label,axis=0) # 4-D after squeeze

                # when the tif has only 1 channel, the loaded array may have falsely swaped dimensions (ZCYX). we want CZYX
                # (This may also happen in different OS or different package versions)
                # ASSUMPTION: we have more z slices than the number of channels 
                if label.shape[1]<label.shape[0]: 
                    label = np.transpose(label,(1,0,2,3))

                input_reader = AICSImage(fn+'.ome.tif') #CZYX  #TODO: check size
                input_img = input_reader.data
                input_img = np.squeeze(input_img,axis=0)
                if input_img.shape[1] < input_img.shape[0]:
                    input_img = np.transpose(input_img,(1,0,2,3))

                costmap_reader = AICSImage(fn+'_CM.ome.tif') # ZYX
                costmap = costmap_reader.data
                costmap = np.squeeze(costmap,axis=0)
                if costmap.shape[0] == 1:
                    costmap = np.squeeze(costmap,axis=0)
                elif costmap.shape[1] == 1:
                    costmap = np.squeeze(costmap,axis=1)

                img_pad0 = np.pad(input_img, ((0,0),(0,0),(padding[1],padding[1]),(padding[2],padding[2])), 'constant')
                raw = np.pad(img_pad0, ((0,0),(padding[0],padding[0]),(0,0),(0,0)), 'constant')

                cost_scale = costmap.max()
                if cost_scale<1: ## this should not happen, but just in case
                    cost_scale = 1

                deg = random.randrange(1,180)
                flip_flag = random.random()

                for zz in range(label.shape[1]):

                    for ci in range(label.shape[0]):
                        labi = label[ci,zz,:,:]
                        labi_pil = Image.fromarray(np.uint8(labi))
                        new_labi_pil = labi_pil.rotate(deg,resample=Image.NEAREST)
                        if flip_flag<0.5:
                            new_labi_pil = new_labi_pil.transpose(Image.FLIP_LEFT_RIGHT)
                        new_labi = np.array(new_labi_pil.convert('L'))
                        label[ci,zz,:,:] = new_labi.astype(int)

                    cmap = costmap[zz,:,:]
                    cmap_pil = Image.fromarray(np.uint8(255*(cmap/cost_scale)))
                    new_cmap_pil = cmap_pil.rotate(deg,resample=Image.NEAREST)
                    if flip_flag<0.5:
                        new_cmap_pil = new_cmap_pil.transpose(Image.FLIP_LEFT_RIGHT)
                    new_cmap = np.array(new_cmap_pil.convert('L'))
                    costmap[zz,:,:] = cost_scale*(new_cmap/255.0)

                for zz in range(raw.shape[1]):
                    for ci in range(raw.shape[0]):
                        str_im = raw[ci,zz,:,:]
                        str_im_pil = Image.fromarray(np.uint8(str_im*255))
                        new_str_im_pil = str_im_pil.rotate(deg,resample=Image.BICUBIC)
                        if flip_flag<0.5:
                            new_str_im_pil = new_str_im_pil.transpose(Image.FLIP_LEFT_RIGHT)
                        new_str_image = np.array(new_str_im_pil.convert('L'))
                        raw[ci,zz,:,:] = (new_str_image.astype(float))/255.0 

                new_patch_num = 0
                num_fail = 0
                while new_patch_num < num_patch_per_img[img_idx]:
                    
                    pz = random.randint(0, label.shape[1] - size_out[0])
                    py = random.randint(0, label.shape[2] - size_out[1])
                    px = random.randint(0, label.shape[3] - size_out[2])

                    
                    # check if this is a good crop
                    ref_patch_cmap = costmap[pz:pz+size_out[0],py:py+size_out[1],px:px+size_out[2]]
                    if np.count_nonzero(ref_patch_cmap>1e-5) < 1000: #enough valida samples
                        num_fail = num_fail + 1
                        if num_fail > 50:
                            break
                        continue
                    

                    # confirmed good crop
                    (self.img).append(raw[:,pz:pz+size_in[0],py:py+size_in[1],px:px+size_in[2]] )
                    (self.gt).append(label[:,pz:pz+size_out[0],py:py+size_out[1],px:px+size_out[2]])
                    (self.cmap).append(ref_patch_cmap)

                    new_patch_num += 1
        
    def __getitem__(self, index):

        image_tensor = from_numpy(self.img[index].astype(float))
        cmap_tensor = from_numpy(self.cmap[index].astype(float))

        label_tensor = []
        if self.gt[index].shape[0]>0:
            for zz in range(self.gt[index].shape[0]):
                label_tensor.append(from_numpy(self.gt[index][zz,:,:,:].astype(float)).float())
        else: 
            label_tensor.append(from_numpy(self.gt[index].astype(float)).float())

        return image_tensor.float(), label_tensor, cmap_tensor.float()

    def __len__(self):
        return len(self.img)

class NOAUG_M(Dataset):

    def __init__(self, filenames, num_patch, size_in, size_out):

        self.img = []
        self.gt = []
        self.cmap = []

        padding = [(x-y)//2 for x,y in zip(size_in, size_out)]
        total_in_count = size_in[0] * size_in[1] * size_in[2]
        total_out_count = size_out[0] * size_out[1] * size_out[2]

        num_data = len(filenames)
        shuffle(filenames)
        num_patch_per_img = np.zeros((num_data,), dtype=int)
        if num_data >= num_patch:
            # all one
            num_patch_per_img[:num_patch]=1
        else: 
            basic_num = num_patch // num_data
            # assign each image the same number of patches to extract
            num_patch_per_img[:] = basic_num

            # assign one more patch to the first few images to achieve the total patch number
            num_patch_per_img[:(num_patch-basic_num*num_data)] = num_patch_per_img[:(num_patch-basic_num*num_data)] + 1


        for img_idx, fn in enumerate(filenames):

            label_reader = AICSImage(fn+'_GT.ome.tif')  #CZYX
            label = label_reader.data
            label = np.squeeze(label,axis=0) # 4-D after squeeze

            # when the tif has only 1 channel, the loaded array may have falsely swaped dimensions (ZCYX). we want CZYX
            # (This may also happen in different OS or different package versions)
            # ASSUMPTION: we have more z slices than the number of channels 
            if label.shape[1]<label.shape[0]: 
                label = np.transpose(label,(1,0,2,3))

            input_reader = AICSImage(fn+'.ome.tif') #CZYX  #TODO: check size
            input_img = input_reader.data
            input_img = np.squeeze(input_img,axis=0)
            if input_img.shape[1] < input_img.shape[0]:
                input_img = np.transpose(input_img,(1,0,2,3))

            costmap_reader = AICSImage(fn+'_CM.ome.tif') # ZYX
            costmap = costmap_reader.data
            costmap = np.squeeze(costmap,axis=0)
            if costmap.shape[0] == 1:
                costmap = np.squeeze(costmap,axis=0)
            elif costmap.shape[1] == 1:
                costmap = np.squeeze(costmap,axis=1)

            img_pad0 = np.pad(input_img, ((0,0),(0,0),(padding[1],padding[1]),(padding[2],padding[2])), 'symmetric')
            raw = np.pad(img_pad0, ((0,0),(padding[0],padding[0]),(0,0),(0,0)), 'constant')

            new_patch_num = 0
            
            while new_patch_num < num_patch_per_img[img_idx]:
                
                pz = random.randint(0, label.shape[1] - size_out[0])
                py = random.randint(0, label.shape[2] - size_out[1])
                px = random.randint(0, label.shape[3] - size_out[2])

                
                ## check if this is a good crop
                ref_patch_cmap = costmap[pz:pz+size_out[0],py:py+size_out[1],px:px+size_out[2]]
                

                # confirmed good crop
                (self.img).append(raw[:,pz:pz+size_in[0],py:py+size_in[1],px:px+size_in[2]] )
                (self.gt).append(label[:,pz:pz+size_out[0],py:py+size_out[1],px:px+size_out[2]])
                (self.cmap).append(ref_patch_cmap)

                new_patch_num += 1
                
    def __getitem__(self, index):

        image_tensor = from_numpy(self.img[index].astype(float))
        cmap_tensor = from_numpy(self.cmap[index].astype(float))

        #if self.gt[index].shape[0]>1:
        label_tensor = []
        for zz in range(self.gt[index].shape[0]):
            tmp_tensor = from_numpy(self.gt[index][zz,:,:,:].astype(float))
            label_tensor.append(tmp_tensor.float())
        #else: 
        #    label_tensor = from_numpy(self.gt[index].astype(float))
        #    label_tensor = label_tensor.float()
            
        return image_tensor.float(), label_tensor, cmap_tensor.float()

    def __len__(self):
        return len(self.img)

class AUG_M_2D(Dataset):
    '''
    Dataloader for 2D input that returns flip and spatial data augmentated image, label, and weight map.
    TODO: Fix the gird_sample warning issu
    '''

    def __init__(self, filenames, num_patch, size_in, size_out):

        self.img = []
        self.gt = []
        self.cmap = []

        if len(filenames) < 40:
            sample_num = 20
        else:
            sample_num = len(filenames)//2

        sampled = np.random.choice(filenames, size=sample_num, replace=False)
        for img_idx, fn in enumerate(sampled):
            
            label_reader = AICSImage(fn+'_GT.ome.tif')
            label = label_reader.data
            label = label[0,0,0,0,:,:]

            # if label is other than [0,1], change it to [0,1]
            label_range = np.unique(label).tolist()
            if label_range != [0,1]:
                label[label == label_range[1]] = 1
            
            # import image
            input_reader = AICSImage(fn+'.ome.tif')
            input_img = input_reader.data
            if input_img.shape[3] == 1: # when it is single channel
                input_img = input_img[0,0,0,0,:,:]
            else:
                input_img = input_img[0,0,0,1,:,:]

            # normalize the image in range of [0, 1]
            if not (np.min(input_img) == 0 and np.max(input_img) == 1): # if image has not been normalized
                input_img = simple_norm(input_img, 1, 6)
            
            costmap_reader = AICSImage(fn+'_CM.ome.tif')
            costmap = costmap_reader.data
            costmap = costmap[0,0,0,0,:,:]

            # concatenate raw, label ,and costmap before spliting into patches
            image_array = np.stack([input_img, label, costmap], axis = 2)

            # randomly generate 2d patches
            input_img_patch = extract_patches_2d(image_array, size_in, max_patches = num_patch)

            input_img = input_img_patch[:,:,:,0]
            label = input_img_patch[:,:,:,1]
            costmap = input_img_patch[:,:,:,2]
            
            for patches in range(num_patch):

                # flip augmentation
                h_flip = np.random.randint(2, dtype='bool')
                if not h_flip:
                    v_flip = np.random.randint(2, dtype='bool')

                if h_flip or v_flip:
                    if h_flip:
                        image_aug = np.fliplr(input_img[patches,:,:])
                        cmap_aug = np.fliplr(costmap[patches,:,:])
                        label_aug = np.fliplr(label[patches,:,:])
                    else:
                        image_aug = np.flipud(input_img[patches,:,:])
                        cmap_aug = np.flipud(costmap[patches,:,:])
                        label_aug = np.flipud(label[patches,:,:])
                
                else:
                    image_aug = input_img[patches,:,:]
                    label_aug = label[patches,:,:]
                    cmap_aug = costmap[patches,:,:]

                random_aug = self.spatial_data_augmentation(image_aug.astype(float), label_aug.astype(float), cmap_aug.astype(float), size_in)

                augmented_img = random_aug[0][0,:,:,:]
                augmented_gt = random_aug[1][0,:,:,:]
                augmented_cmap = random_aug[2][0,:,:,:]

                (self.img).append(augmented_img)
                (self.gt).append(augmented_gt)
                (self.cmap).append(augmented_cmap)
              
    def __getitem__(self, index):

        image_tensor = self.img[index]
        cmap_tensor = self.cmap[index]
        label_tensor = self.gt[index]

        return image_tensor.float(), label_tensor.float(), cmap_tensor.float()

    def __len__(self):
        return len(self.img)

    def spatial_data_augmentation(self, img, label, cmap, size):
        '''
        Performs full 2D deformation using random flow. Unlike rotation or flip, spatial data augmentation provides
        full 2D deformation that can provide the model robustness.
        '''
    
        # Generate a smooth random field by resizing the small random matrix
        random_array = np.random.rand(16,16)
        da_factor = np.random.random(1) * 0.125
        random_array = random_array * da_factor
        smooth_field = resize(random_array, size)
        smooth_field = np.stack([smooth_field, smooth_field], axis = 2)
        smooth_field = np.reshape(smooth_field, (1,) + smooth_field.shape)
        smooth_field = from_numpy(smooth_field).float()

        # changing the data to tensor
        img_tensor, label_tensor, cmap_tensor = self.transform_tensor(img, label, cmap)

        # Generate grid for affine transformation
        id_grid = torch.nn.functional.affine_grid(torch.FloatTensor([[[1, 0, 0],[0, 1, 0]]]), size=(1,1,size[0],size[1]))

        # Augment target with affine grid + smoooth random field to perform full 2D deformation
        img_aug = grid_sample(img_tensor, id_grid+smooth_field)
        label_aug = grid_sample(label_tensor, id_grid+smooth_field, mode = 'nearest')
        cmap_aug = grid_sample(cmap_tensor, id_grid+smooth_field, mode = 'nearest')


        img_aug = torch.stack([img_aug,img_aug,img_aug], axis=2)
        img_aug = torch.squeeze(img_aug, dim=0)

        return (img_aug, label_aug, cmap_aug)

    def random_patch(self, image_array, label_array, cmap_array, patch_size, max_patches):
        '''
        Function that generates random patch for image and corresponding label and weight map.
        TODO: need to change to cpu tensor based
        '''
        image_size = image_array.shape # get image size

        # Array to store pathces
        cropped_img = []
        cropped_gt = []
        cropped_cmap = []

        # Randomly generate given number of patches
        for patch in range(max_patches):
            upper_left_y = np.random.randint(0,image_size[0]-patch_size[0]-1)
            upper_left_x = np.random.randint(0,image_size[1]-patch_size[1]-1)

            lower_right_y = upper_left_y + patch_size[0]
            lower_right_x = upper_left_x + patch_size[1]

            cropped_img.append(image_array[upper_left_y:lower_right_y, upper_left_x:lower_right_x])
            cropped_gt.append(label_array[upper_left_y:lower_right_y, upper_left_x:lower_right_x])
            cropped_cmap.append(cmap_array[upper_left_y:lower_right_y, upper_left_x:lower_right_x])


        return (cropped_img, cropped_gt, cropped_cmap)

    def transform_tensor(self, img, label, cmap, nthread=32):
        '''
        Transforms the image, label, and weight map into the float cpu tensor.
        '''
        
        torch.set_num_threads(nthread)

        img = np.reshape(img, (1,) + (1,) + img.shape)
        img_tensor = from_numpy(img).float().cpu()

        label = np.reshape(label, (1,) + (1,) + label.shape)
        label_tensor = from_numpy(label).float().cpu()

        cmap = np.reshape(cmap, (1,) + (1,) + cmap.shape)
        cmap_tensor = from_numpy(cmap).float().cpu()

        return (img_tensor, label_tensor, cmap_tensor)