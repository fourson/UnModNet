# tonemap all .npy images in current dir to .jpg using OpenCV Reinhard method
import os
import cv2
import numpy as np

for f in os.listdir(os.getcwd()):
    if f.endswith('.npy'):
        hdr = np.load(f)
        hdr = hdr.astype('float32')
        hdr = (hdr - np.min(hdr)) / (np.max(hdr) - np.min(hdr))
        grayscale = True
        if hdr.ndim == 3:
            if hdr.shape[2] == 3:
                # RGB image (H, W, 3)
                hdr = cv2.cvtColor(hdr, cv2.COLOR_RGB2BGR)
                grayscale = False
            elif hdr.shape[2] == 1:
                # grayscale image (H, W, 1)
                hdr = hdr[:, :, 0]
        if grayscale:
            hdr = np.stack([hdr, hdr, hdr], axis=2)
            
        tmo = cv2.createTonemapReinhard(intensity=-1.0, light_adapt=0.8, color_adapt=0.0)
        tonemapped = tmo.process(hdr)
        f_ = f.split('.')[0] + '.jpg'
        cv2.imwrite(f_, tonemapped * 255)

