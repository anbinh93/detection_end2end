import cv2
import numpy as np

def denoise_image(image_np: np.ndarray) -> np.ndarray:
    """
    Applies Non-Local Means Denoising to a BGR image.
    """
    denoised_image = cv2.fastNlMeansDenoisingColored(image_np, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    return denoised_image

def enhance_contrast_clahe(image_np: np.ndarray) -> np.ndarray:
    """
    Applies CLAHE to the L-channel of an LAB image for contrast enhancement.
    Returns a BGR image.
    """
    lab = cv2.cvtColor(image_np, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)


    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    limg = cv2.merge((cl, a_channel, b_channel))
    enhanced_image_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_image_bgr

def sharpen_image(image_np: np.ndarray) -> np.ndarray:
    """
    Applies a sharpening kernel (Unsharp Masking simplified).
    """
    # Sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened_image = cv2.filter2D(image_np, -1, kernel)
    return sharpened_image


def preprocess_image(image_pil, apply_denoise=False, apply_contrast=False, apply_sharpen=False):
    """
    Applies selected preprocessing steps to a PIL Image.
    Returns a PIL Image.
    """
    if not (apply_denoise or apply_contrast or apply_sharpen):
        return image_pil # No preprocessing selected

    # Convert PIL Image to OpenCV format (BGR)
    image_np = np.array(image_pil.convert('RGB'))
    image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    processed_image_np = image_np_bgr.copy()

    if apply_denoise:
        processed_image_np = denoise_image(processed_image_np)
    
    if apply_contrast:
        processed_image_np = enhance_contrast_clahe(processed_image_np)

    if apply_sharpen:

        processed_image_np = sharpen_image(processed_image_np)

    processed_image_rgb = cv2.cvtColor(processed_image_np, cv2.COLOR_BGR2RGB)
    
    from PIL import Image 
    final_pil_image = Image.fromarray(processed_image_rgb)
    
    return final_pil_image