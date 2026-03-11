import numpy as np
import imageio.v2 as imageio
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# Define a constant for normalization
MAX_PIXEL_VALUE = 255.0

def manual_pyrDown(image):
    #define the 5x5 Gaussian kernel used by OpenCV
    kernel_1d = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    #create the 2D matrix by computing the outer product
    kernel = np.outer(kernel_1d, kernel_1d) 
    #normalize so the sum is 1 (preserves brightness)
    kernel /= 256.0
    
    #check if image is color (3 channels) or grayscale (2D)
    if len(image.shape) == 3:
        #if RGB, we must process each channel (R, G, B) separately
        height, width, channels = image.shape
        #create output array for the blurred image
        blurred = np.zeros_like(image)
        
        for c in range(channels):
            #'same' keeps the output size equal to input size
            #'symm' handles borders by mirroring pixels (standard for images)
            blurred[:, :, c] = convolve2d(image[:, :, c], kernel, mode='same', boundary='symm')
            
    else:
        #grayscale case
        blurred = convolve2d(image, kernel, mode='same', boundary='symm')

    #downsample: Keep every 2nd pixel (slice [::2, ::2])
    downsampled = blurred[::2, ::2]
    
    return downsampled

def manual_pyrUp(image):
    # 1. Define the same kernel as pyrDown
    kernel_1d = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    kernel = np.outer(kernel_1d, kernel_1d) 
    kernel /= 256.0
    kernel *= 4.0
    height, width = image.shape[:2]
    new_height, new_width = height * 2, width * 2
    
    if len(image.shape) == 3:
        # RGB Case
        channels = image.shape[2]
        upsampled = np.zeros((new_height, new_width, channels), dtype=np.float32)
        
        upsampled[::2, ::2, :] = image
        
        #convolve (Interpolate) to fill in the zeros
        output = np.zeros_like(upsampled)
        for c in range(channels):
            output[:, :, c] = convolve2d(upsampled[:, :, c], kernel, mode='same', boundary='symm')
            
    else:
        #grayscale Case
        upsampled = np.zeros((new_height, new_width), dtype=np.float32)
        upsampled[::2, ::2] = image
        output = convolve2d(upsampled, kernel, mode='same', boundary='symm')

    return output

def build_gaussian_pyramid(image, levels):
    """Build Gaussian pyramid."""
    gaussian_pyramid = [image.astype(np.float32)]
    for i in range(1, levels):
        #blur + shrink image to half size (gaussian pyramid)
        image = manual_pyrDown(image)
        #convert pixel type to float for safe math and 
        #add this level to the pyramid list
        gaussian_pyramid.append(image.astype(np.float32))
    return gaussian_pyramid


def resize_nearest(image, new_h, new_w):
    """
    Resize an image to (new_h, new_w) using nearest-neighbor interpolation. 
    """
    h, w = image.shape[:2]

    #create coordinate grids
    row_idx = (np.linspace(0, h - 1, new_h)).astype(np.int32)
    col_idx = (np.linspace(0, w - 1, new_w)).astype(np.int32)

    #use broadcasting to sample pixels
    return image[row_idx][:, col_idx]


def build_laplacian_pyramid(gaussian_pyramid):
    """Build Laplacian pyramid from Gaussian pyramid."""
    laplacian_pyramid = []
    for i in range(len(gaussian_pyramid) - 1):
        expanded = manual_pyrUp(gaussian_pyramid[i + 1])
        expanded = resize_nearest(expanded,
                gaussian_pyramid[i].shape[0], gaussian_pyramid[i].shape[1])

        lap_i = gaussian_pyramid[i] - expanded
        laplacian_pyramid.append(lap_i)
    laplacian_pyramid.append(gaussian_pyramid[-1])  
    return laplacian_pyramid

def reconstruct_from_laplacian(laplacian_pyramid):
    """Reconstruct image by summing Laplacian levels."""
    image = laplacian_pyramid[-1]
    for level in reversed(laplacian_pyramid[:-1]):
        image = manual_pyrUp(image)
        image = resize_nearest(image, level.shape[0], level.shape[1])
        image += level
    return image


def laplacian_blend(A, B, M, levels=6):
    """
    Blend two images using Laplacian Pyramid blending with a Gaussian mask pyramid.
    A, B: images (same size)
    M: binary mask (same size), values 0 or 1
    """
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    M = M.astype(np.float32)

    #build pyramids for A, B, and mask M
    Ga = build_gaussian_pyramid(A, levels)
    Gb = build_gaussian_pyramid(B, levels)
    Gm = build_gaussian_pyramid(M, levels)

    La = build_laplacian_pyramid(Ga)
    Lb = build_laplacian_pyramid(Gb)

    #build blended Laplacian pyramid Lc
    Lc = []
    for k in range(levels):
        mask = Gm[k]
        Lc_k = mask * La[k] + (1 - mask) * Lb[k]
        Lc.append(Lc_k)

    #reconstruct blended image
    blended = reconstruct_from_laplacian(Lc)

    #keeps all pixel values valid after the blending process,
    # which often produces values slightly outside the normal range.
    blended = np.clip(blended, 0, 1)
    return blended


def load_images_and_mask(image_a_path, image_b_path, mask_path):
    #load images
    A = imageio.imread(image_a_path).astype(np.float32) / MAX_PIXEL_VALUE
    B = imageio.imread(image_b_path).astype(np.float32) / MAX_PIXEL_VALUE
    M = imageio.imread(mask_path).astype(np.float32) / MAX_PIXEL_VALUE

    #resize to match the smallest dimension
    target_h = min(A.shape[0], B.shape[0], M.shape[0])
    target_w = min(A.shape[1], B.shape[1], M.shape[1])
    
    #make dimensions even (important for pyramids)
    target_h = (target_h // 2) * 2
    target_w = (target_w // 2) * 2

    A = resize_nearest(A, target_h, target_w)
    B = resize_nearest(B, target_h, target_w)
    M = resize_nearest(M, target_h, target_w)

    return A, B, M

def save_image(image, output_path):
    """
    Convert image from [0,1] float to uint8 and save.
    """
    image_uint8 = (np.clip(image, 0, 1) * MAX_PIXEL_VALUE).astype(np.uint8)
    if len(image_uint8.shape) == 3 and image_uint8.shape[2] == 3:
        image_uint8 = image_uint8[..., ::-1]
    imageio.imwrite(output_path, (image * MAX_PIXEL_VALUE).astype(np.uint8))

#Task: Hybrid Images
def create_gaussian_kernel(k_size, sigma):
    """
    Creates a 2D Gaussian kernel manually using the Gaussian formula.
    k_size: The size of the square kernel
    sigma: The standard deviation (how strong the blur is)
    """
    #create a 1D array of coordinates [-x, ..., 0, ..., x]
    center = k_size // 2
    x = np.arange(-center, center + 1)
    #calculate 1D Gaussian: G(x) = exp(-(x^2) / (2*sigma^2))
    kernel_1d = np.exp(-(x**2) / (2 * sigma**2))
    
    #normalize so sum is 1 (otherwise image gets brighter/darker)
    kernel_1d = kernel_1d / np.sum(kernel_1d)
    
    #create 2D kernel by outer product
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    
    return kernel_2d

def apply_convolution(image, kernel):
    """
    Applies convolution to each channel (R, G, B) separately.
    """
    #if grayscale (2D)
    if len(image.shape) == 2:
        return convolve2d(image, kernel, mode='same', boundary='symm')
    
    #if RGB (3D)
    result = np.zeros_like(image)
    for channel in range(image.shape[2]):
        result[:, :, channel] = convolve2d(image[:, :, channel], kernel, mode='same', boundary='symm')
        
    return result

def get_low_frequency(image, kernel):
    """
    Step 1: Prepare Image A (Seen from Far).
    Action: Apply Gaussian Blur.
    """
    low_freq_image = apply_convolution(image, kernel)
    return low_freq_image


def get_high_frequency(image, kernel):
    """
    Step 2: Prepare Image B (Seen from Close).
    Action: Subtract the blurred version from the original.
    """
    blurred_version = apply_convolution(image, kernel)
    high_freq_image = image - blurred_version
    return high_freq_image


def combine_hybrid(low_freq_A, high_freq_B):
    """
    Step 3: Combine.
    Action: Add pixel values and clip to valid range.
    """
    hybrid = low_freq_A + high_freq_B
    #clip values to ensure they stay between 0 and 1
    hybrid = np.clip(hybrid, 0, 1)
    return hybrid

def ensure_rgb_numpy(image):
    """Helper to ensure image is 3-channel RGB."""
    if len(image.shape) == 2:
        return np.stack((image, image, image), axis=-1)
    if len(image.shape) == 3 and image.shape[2] == 1:
        squeezed = np.squeeze(image)
        return np.stack((squeezed, squeezed, squeezed), axis=-1)
    return image

def run_hybrid_image_process(image_A_path, image_B_path, output_path, kernel_size=25, sigma=5):
    """
    Loads two images, creates a hybrid image from them, and saves the result.
    
    Args:
        image_A_path: Path to the 'Far' image (Low Frequency)
        image_B_path: Path to the 'Close' image (High Frequency)
        output_path: Where to save the result
        kernel_size: Size of Gaussian kernel (odd number)
        sigma: Standard deviation for Gaussian kernel
    """
    A = imageio.imread(image_A_path).astype(np.float32) / MAX_PIXEL_VALUE
    B = imageio.imread(image_B_path).astype(np.float32) / MAX_PIXEL_VALUE

    A = ensure_rgb_numpy(A)
    B = ensure_rgb_numpy(B)

    min_h = min(A.shape[0], B.shape[0])
    min_w = min(A.shape[1], B.shape[1])
    A = A[:min_h, :min_w]
    B = B[:min_h, :min_w]

    kernel = create_gaussian_kernel(kernel_size, sigma)

    low_freq_A = get_low_frequency(A, kernel)
    high_freq_B = get_high_frequency(B, kernel)
    hybrid_result = combine_hybrid(low_freq_A, high_freq_B)

    result_uint8 = (hybrid_result * MAX_PIXEL_VALUE).astype(np.uint8)
    imageio.imwrite(output_path, result_uint8)


# def save_pyramid_visualization(image_path, levels=5, output_file="analysis_pyramid.png"):
#     print(f"Generating Pyramid Analysis for {image_path}...")
#     try:
#         # טעינת התמונה
#         image = imageio.imread(image_path).astype(np.float32) / MAX_PIXEL_VALUE
#         # המרה ל-RGB אם צריך
#         if len(image.shape) == 2:
#             image = np.stack((image, image, image), axis=-1)
#         elif image.shape[2] == 4:
#             image = image[:, :, :3]

#         # בניית הפירמידות (משתמש בפונקציות הקיימות שלך)
#         gauss_pyr = build_gaussian_pyramid(image, levels)
#         lap_pyr = build_laplacian_pyramid(gauss_pyr)

#         # ציור הגרף
#         plt.figure(figsize=(15, 5))
#         for i in range(min(levels, 5)):
#             plt.subplot(1, 5, i+1)
#             level_img = lap_pyr[i]
            
#             # המרה לשחור לבן לתצוגה ברורה
#             if len(level_img.shape) == 3:
#                 level_img = np.mean(level_img, axis=2)
            
#             # מתיחת קונטרסט (כדי שנראה את הפרטים השחורים)
#             display_img = np.abs(level_img)
#             display_img = display_img / (np.max(display_img) + 1e-6) # מניעת חילוק ב-0
            
#             plt.imshow(display_img, cmap='gray')
#             plt.title(f"Level {i}")
#             plt.axis('off')

#         plt.suptitle(f"Laplacian Pyramid Analysis: {image_path}")
#         plt.savefig(output_file) # שמירה לקובץ!
#         plt.close()
#         print(f"Saved pyramid analysis to: {output_file}")

#     except Exception as e:
#         print(f"Error in pyramid analysis: {e}")


# def save_fourier_visualization(good_img_path, bad_img_path, output_file="analysis_fourier.png"):
#     print(f"Generating Fourier Analysis for {good_img_path} vs {bad_img_path}...")
#     try:
#         # פונקציה פנימית לחישוב FFT
#         def calc_log_magnitude(img_path):
#             img = imageio.imread(img_path, mode='F') # טעינה כשחור לבן (Float)
#             f = np.fft.fft2(img)
#             fshift = np.fft.fftshift(f)
#             magnitude = 20 * np.log(np.abs(fshift) + 1e-6)
#             return magnitude

#         mag_good = calc_log_magnitude(good_img_path)
#         mag_bad = calc_log_magnitude(bad_img_path)

#         # ציור הגרף
#         plt.figure(figsize=(12, 6))
        
#         plt.subplot(1, 2, 1)
#         plt.imshow(mag_good, cmap='gray')
#         plt.title(f"Good Blend FFT\n({good_img_path})")
#         plt.axis('off')

#         plt.subplot(1, 2, 2)
#         plt.imshow(mag_bad, cmap='gray')
#         plt.title(f"Bad Blend FFT\n({bad_img_path})")
#         plt.axis('off')

#         plt.savefig(output_file) # שמירה לקובץ!
#         plt.close()
#         print(f"Saved Fourier analysis to: {output_file}")

#     except Exception as e:
#         print(f"Error in Fourier analysis: {e}")



# def save_gaussian_visualization(image_path, levels=5, output_file="analysis_gaussian.png"):
#     print(f"Generating Gaussian Pyramid Analysis for {image_path}...")
#     try:
#         image = imageio.imread(image_path).astype(np.float32) / MAX_PIXEL_VALUE
#         if len(image.shape) == 2:
#             image = np.stack((image, image, image), axis=-1)
#         elif image.shape[2] == 4:
#             image = image[:, :, :3]

#         # בניית פירמידת גאוס בלבד
#         gauss_pyr = build_gaussian_pyramid(image, levels)

#         plt.figure(figsize=(15, 5))
#         for i in range(min(levels, 5)):
#             plt.subplot(1, 5, i+1)
#             level_img = gauss_pyr[i]
            
#             # המרה לגווני אפור להצגה ברורה של המבנה
#             if len(level_img.shape) == 3:
#                 level_img = np.mean(level_img, axis=2)
            
#             # בפירמידת גאוס אין ערכים שליליים, הצגה רגילה
#             plt.imshow(level_img, cmap='gray')
#             plt.title(f"G-Level {i}\n(Low Freq)")
#             plt.axis('off')

#         plt.suptitle(f"Gaussian Pyramid Structure: {image_path}")
#         plt.savefig(output_file)
#         plt.close()
#         print(f"Saved Gaussian analysis to: {output_file}")

#     except Exception as e:
#         print(f"Error in Gaussian analysis: {e}")

   
if __name__ == "__main__":
    #load images and mask
    A, B, M = load_images_and_mask("rabbit.png", "dog.png", "good_mask.jpg")

    #perform Laplacian blending
    result = laplacian_blend(A, B, M, levels=6)

    #save the result
    save_image(result, "result.png")