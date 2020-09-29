import os
import random
import argparse
import numpy as np
from tqdm import tqdm


class RandomExposer:
    def __init__(self, hdr, overexposure_rate_ubound=0.35, overexposure_rate_lbound=0.05, iter_max=15):
        self.hdr = (hdr - np.min(hdr)) / (np.max(hdr) - np.min(hdr))  # (H, W, C) image(scaled to [0, 1])

        # first we assume that the pixel values(scaled to [0, 1]) obey uniform distribution
        # so that pixel tolerance v is equal to not_overexposure rate
        # pixel tolerance definition: the pixel value which is larger than pixel tolerance means that it is overexposed

        # we hope that overexposure rate is between [overexposure_rate_lbound, overexposure_rate_ubound]
        self.overexposure_rate_ubound = overexposure_rate_ubound
        self.overexposure_rate_lbound = overexposure_rate_lbound

        # so not_overexposure rate is between [1-overexposure_rate_ubound, 1-overexposure_rate_lbound], so is v
        v0_ubound = 1 - overexposure_rate_lbound
        v0_lbound = 1 - overexposure_rate_ubound

        # we set the initial value of v between [v0_lbound, v0_ubound] randomly
        self.v = np.random.random() * (v0_ubound - v0_lbound) + v0_lbound

        # set the maximum number of iterations
        self.iter_max = iter_max

        self.success = True

        # for log
        self.iter_cnt = 0
        self.final_overexposure_rate = 0
        self.exposure = 0
        self.log = "iter_cnt exceeds the limit({})\n".format(self.iter_max)

    @property
    def overexposure_rate(self):
        # calculate the overexposure_rate
        # "the pixel which is not overexposure" means that it is not overexposure in all channels
        not_overexposure_mask = np.prod((self.hdr < self.v), axis=2)
        return 1 - np.sum(not_overexposure_mask) / not_overexposure_mask.size

    def _find_v(self):
        # use binary-search to find a reasonable v
        overexposure_rate = self.overexposure_rate
        if overexposure_rate > self.overexposure_rate_ubound:
            v_lb, v_ub = self.v, 1
        elif overexposure_rate < self.overexposure_rate_lbound:
            v_lb, v_ub = 0, self.v
        else:
            v_lb, v_ub = 0, 0

        iter_cnt = 0
        while v_lb < v_ub:
            if iter_cnt > self.iter_max:
                # exceeds the iteration limit
                self.success = False
                break
            iter_cnt += 1
            self.v = (v_lb + v_ub) / 2
            overexposure_rate = self.overexposure_rate
            if overexposure_rate > self.overexposure_rate_ubound:
                v_lb = self.v
            elif overexposure_rate < self.overexposure_rate_lbound:
                v_ub = self.v
            else:
                break

        self.iter_cnt = iter_cnt
        self.final_overexposure_rate = overexposure_rate

    def expose(self, modulo_pixel_max, ref_pixel_max):
        # simulates the sensor exposure and ADC(we assume that the CRF is linear)
        self._find_v()
        if self.success:
            # as the equation: v = (modulo_pixel_max + 1) / (ref_pixel_max * exposure)
            self.exposure = (modulo_pixel_max + 1) / (ref_pixel_max * self.v)
            self.hdr = np.clip(self.hdr * self.exposure, 0, 1)  # sensor exposure
            self.hdr = np.floor(self.hdr * ref_pixel_max)  # ADC
            self.log = "iter_cnt: {}\nfinal_overexposure_rate: {}\nexposure: {}\n".format(self.iter_cnt,
                                                                                          self.final_overexposure_rate,
                                                                                          self.exposure)


class DataItem:
    """
        (H, W, C)
        modulo: positive int, as float32
        fold_number: positive int,  as float32
        mask: binary, as float32
        ref: positive int, as float32
        ldr: [0, modulo_pixel_max] int, as float32
    """

    def __init__(self, origin, modulo_pixel_max):
        self.origin = origin
        self.modulo_pixel_max = modulo_pixel_max

        self.modulo = self.origin % (self.modulo_pixel_max + 1)  # modulo image
        self.fold_number = self.origin // (self.modulo_pixel_max + 1)  # fold number map
        self.mask = np.float32((self.fold_number > 0))  # binary mask for the pixels which are overexposed
        self.ref = self.modulo + (self.modulo_pixel_max + 1) * self.mask  # reference image(modulo "plus 1 fold")
        self.ldr = (self.modulo_pixel_max + 1) * self.mask + self.origin * (1 - self.mask)  # ldr image

        self.fold_number_max = int(np.max(self.fold_number))

    def add_one_fold(self):
        self.modulo = self.ref
        self.fold_number = self.fold_number - self.mask
        self.mask = np.float32((self.fold_number > 0))
        self.ref = self.modulo + (self.modulo_pixel_max + 1) * self.mask


class DatasetMaker:
    def __init__(self, data_dir, train_dir, test_dir, training_sample=400, modulo_bits=8, ref_bits=12,
                 multi_fold_for_training=False, **random_exposer_args):
        self.data_dir = data_dir  # contains the original hdr images
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.modulo_pixel_max = 2 ** modulo_bits - 1
        self.ref_pixel_max = 2 ** ref_bits - 1
        self.multi_fold_for_training = multi_fold_for_training

        self.random_exposer_args = random_exposer_args

        self.data_list = os.listdir(data_dir)
        random.shuffle(self.data_list)
        self.train_data_list = self.data_list[:training_sample]
        self.test_data_list = self.data_list[training_sample:]

        self.train_subdir = {"origin": "", "modulo": "", "fold_number": "", "mask": "", "ref": "", "ldr": ""}
        self.test_subdir = {"origin": "", "modulo": "", "fold_number": "", "ldr": ""}

        self.save_cnt = 0

    def _ensure_dir(self, mode):
        assert mode in ("train", "test")
        mode_dir = getattr(self, mode + "_dir")
        mode_subdir = getattr(self, mode + "_subdir")
        for key in mode_subdir:
            mode_subdir[key] = os.path.join(mode_dir, key)
            if not os.path.exists(mode_subdir[key]):
                os.makedirs(mode_subdir[key])

    def _save(self, data_item, name, mode):
        assert mode in ("train", "test")
        mode_subdir = getattr(self, mode + "_subdir")
        for key in mode_subdir:
            np.save(os.path.join(mode_subdir[key], name + ".npy"), getattr(data_item, key))
        self.save_cnt += 1

    def make(self, mode):
        assert mode in ("train", "test")
        print("mode: {}\n".format(mode))
        mode_data_list = getattr(self, mode + "_data_list")
        if not mode_data_list:
            return
        log = ""
        self._ensure_dir(mode)
        for hdr_file in tqdm(mode_data_list, ascii=True):
            hdr_name = hdr_file.split(".")[0]
            hdr_img = np.load(os.path.join(self.data_dir, hdr_file))
            log += "--------------------\nname: {}\n".format(hdr_name)
            random_exposer = RandomExposer(hdr_img, **self.random_exposer_args)
            random_exposer.expose(self.modulo_pixel_max, self.ref_pixel_max)
            log += random_exposer.log
            if random_exposer.success:
                data_item = DataItem(random_exposer.hdr, self.modulo_pixel_max)
                log += "fold_number_max: {}\n".format(data_item.fold_number_max)
                self._save(data_item, hdr_name, mode)
                if mode == "train" and self.multi_fold_for_training:
                    for i in range(1, data_item.fold_number_max):
                        data_item.add_one_fold()
                        name_new = hdr_name + "_plus" + str(i)
                        self._save(data_item, name_new, mode)

        log += "\n----------Summary----------\n"
        log += "This is the dataset for *{}*\n".format((mode + "ing").upper())
        log += "{} images in total".format(self.save_cnt)
        self.save_cnt = 0
        with open(mode + "_dataset.log", "w") as f:
            f.write(log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make dataset")
    parser.add_argument("--data_dir", required=True, type=str, help="dir of original 512*512 hdr images")
    parser.add_argument("--train_dir", default="train", type=str, help="dir to save training dataset")
    parser.add_argument("--test_dir", default="test", type=str, help="dir to save test dataset")
    parser.add_argument("--training_sample", default=400, type=int, help="number of training sample")
    parser.add_argument("--modulo_bits", default=8, type=int, help="modulo image bits")
    parser.add_argument("--ref_bits", default=12, type=int, help="reference image(ground truth) bits")
    parser.add_argument("--multi_fold_for_training", default=1, type=int,
                        help="make multi-fold data for training (set this to 0 when directly learn label in ablation)")
    parser.add_argument("--overexposure_rate_ubound", default=0.35, type=float, help="overexposure rate upper bound")
    parser.add_argument("--overexposure_rate_lbound", default=0.05, type=float, help="overexposure rate lower bound")
    parser.add_argument("--iter_max", default=15, type=int, help="the iteration limit of random exposer")
    args = parser.parse_args()

    dataset_maker = DatasetMaker(**vars(args))
    dataset_maker.make("train")
    # for test
    dataset_maker.make("test")
