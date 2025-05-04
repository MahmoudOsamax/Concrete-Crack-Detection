import os
import cv2
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from .cv2_utils import getContours
import torchvision.transforms as transforms
from .models.deepcrack_model import DeepCrackModel
import logging
import imutils

logger = logging.getLogger(__name__)

def tensor2im(input_image, imtype=np.uint8):
    """Converts a Tensor array into a numpy image array."""
    try:
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
            image_numpy = image_tensor[0].cpu().float().numpy()
            if image_numpy.shape[0] == 1:
                image_numpy = np.tile(image_numpy, (3, 1, 1))
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        else:
            image_numpy = input_image
        return image_numpy.astype(imtype)
    except Exception as e:
        logger.error(f"Error in tensor2im: {str(e)}")
        return input_image

def read_image(bytesImg, dim=(256, 256)):
    """Read and preprocess image from bytes."""
    try:
        img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        img = cv2.imdecode(np.frombuffer(bytesImg, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        w, h = dim
        if w > 0 and h > 0:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        
        img = img_transforms(Image.fromarray(img))
        return img
    except Exception as e:
        logger.error(f"Error in read_image: {str(e)}")
        raise

def create_model(opt, cp_path='pretrained_net_G.pth'):
    """Create and load DeepCrack model."""
    try:
        model = DeepCrackModel(opt)
        checkpoint = torch.load(cp_path, map_location=torch.device('cpu'))
        if hasattr(model.netG, 'module'):
            model.netG.module.load_state_dict(checkpoint, strict=False)
        else:
            model.netG.load_state_dict(checkpoint, strict=False)
        model.eval()
        logger.info(f"Loaded checkpoint from {cp_path}")
        return model
    except Exception as e:
        logger.error(f"Error in create_model: {str(e)}")
        raise

def overlay(image, mask, color=(255, 0, 0), alpha=0.5, resize=(256, 256)):
    """Combine image and segmentation mask."""
    try:
        color = np.asarray(color).reshape(1, 1, 3)
        colored_mask = np.expand_dims(mask, 0).repeat(3, axis=2)
        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()
        
        if resize is not None:
            image = cv2.resize(image, resize)
            image_overlay = cv2.resize(image_overlay, resize)
        
        image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
        return image_combined
    except Exception as e:
        logger.error(f"Error in overlay: {str(e)}")
        return image

def inference(model, bytesImg, dim, unit):
    """Run inference on input image."""
    try:
        image = read_image(bytesImg, dim)
        image = image.unsqueeze(0)
        
        model.set_input({'image': image, 'label': torch.zeros_like(image), 'A_paths': ''})
        model.test()
        visuals = model.get_current_visuals()
        confidence = visuals['fused'].max()

        for key in visuals.keys():
            visuals[key] = tensor2im(visuals[key])
        
        fused = visuals['fused']
        realHeight, realWidth, _ = fused.shape

        mask = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY)
        mask[mask < 90] = 0
        mask[mask >= 90] = 255
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        overlay_img = overlay(tensor2im(image), mask, alpha=0)
        if cnts:
            cv2.drawContours(image=overlay_img, contours=cnts, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
            contour_img = getContours(fused, overlay_img, realHeight, realWidth, unit, confidence)
        else:
            contour_img = overlay_img

        logger.info("Inference completed")
        return contour_img, visuals
    except Exception as e:
        logger.error(f"Error in inference: {str(e)}")
        raise