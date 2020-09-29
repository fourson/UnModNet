import os
import argparse

import cv2
import numpy as np
from tqdm import tqdm


def main(data_dir):
    modulo_dir = os.path.join(data_dir, 'modulo')
    fold_number_dir = os.path.join(data_dir, 'fold_number')
    modulo_edge_dir = os.path.join(data_dir, 'modulo_edge_dir')
    fold_number_edge_dir = os.path.join(data_dir, 'fold_number_edge')

    if not os.path.exists(modulo_edge_dir):
        os.mkdir(modulo_edge_dir)
    if not os.path.exists(fold_number_edge_dir):
        os.mkdir(fold_number_edge_dir)

    for name in tqdm(os.listdir(modulo_dir), ascii=True):
        # input
        modulo = np.load(os.path.join(modulo_dir, name))  # positive int, as float32
        fold_number = np.load(os.path.join(fold_number_dir, name))  # positive int, as float32

        laplacian_modulo = np.abs(cv2.Laplacian(modulo, -1))
        laplacian_fold_number = np.abs(cv2.Laplacian(fold_number, -1))

        if modulo.shape[2] == 1:
            # for grayscale image (H, W, 1), laplacian will output (H, W)
            # we need to expand the lost dim
            laplacian_modulo = laplacian_modulo[:, :, np.newaxis]
            laplacian_fold_number = laplacian_fold_number[:, :, np.newaxis]

        # to save
        modulo_edge = laplacian_modulo / np.max(laplacian_modulo)  # [0, 1] float, as float32
        fold_number_edge = np.float32(laplacian_fold_number > 0)  # binary, as float32
        np.save(os.path.join(modulo_edge_dir, name), modulo_edge)
        np.save(os.path.join(fold_number_edge_dir, name), fold_number_edge)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make edge map: from RGB(H, W, C) or grayscale(H, W, 1)")
    parser.add_argument("--data_dir", default="data", type=str, help="dir of modulo image")
    args = parser.parse_args()
    main(**vars(args))
