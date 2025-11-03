import cv2
import torch
from fused_ssim import fused_ssim

from utils.my_exception import ImpossibleToAnswer

def calculate_most_dissimilar_confounding_images(
    confounding_images, next_image, **kwargs
):
    
    if(len(confounding_images) <=3):
        raise ImpossibleToAnswer("Not enough confounding images")

    # similar to difficulty in identifying the missing image
    # quite slow can we parallelize this?
    confounding_images_ssim = []
    image_path = kwargs["destination_simulation_id_path"] + "/render"

    # trying batch approach
    B = len(confounding_images)
    img_full_path_gt = image_path + f"/{next_image}.png"
    img_gt = cv2.imread(img_full_path_gt, cv2.IMREAD_UNCHANGED)  # reads as BGR or BGRA
    img_gt_rgb = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)        
    img_gt_rgb_tensor = torch.from_numpy(img_gt_rgb).permute(2,0,1).repeat(B, 1, 1, 1).float().cuda()  # BxCxHxW

    confounding_images_rgb_tensors = []

    for img_idx in confounding_images:        
        img_full_path_confounding = image_path + f"/{img_idx}.png"
        img_confounding = cv2.imread(img_full_path_confounding, cv2.IMREAD_UNCHANGED)  # reads as BGR or BGRA
        img_confounding_rgb = cv2.cvtColor(img_confounding, cv2.COLOR_BGR2RGB)
        img_confounding_rgb_tensor = torch.from_numpy(img_confounding_rgb).permute(2,0,1).unsqueeze(0).float()  # 1xCxHxW
        confounding_images_rgb_tensors.append(img_confounding_rgb_tensor)

    img_confounding_rgb_tensor = torch.cat(confounding_images_rgb_tensors, dim=0).cuda()  # BxCxHxW

    confounding_images_ssim = []
    for idx in range(img_confounding_rgb_tensor.shape[0]):
        confounding_images_ssim.append((idx, fused_ssim(img_gt_rgb_tensor[idx].unsqueeze(0), \
            img_confounding_rgb_tensor[idx].unsqueeze(0), train=False).cpu().numpy().tolist()))

    del img_confounding_rgb_tensor
    del img_gt_rgb_tensor
    torch.cuda.empty_cache()     
    
    confounding_images_ssim.sort(key=lambda x: x[1], reverse=False)
    confounding_images = [confounding_images[idx] for idx, _ in confounding_images_ssim[:3]]

    return confounding_images