import os
import pprint

from matplotlib import pyplot as plt
from METEOR_core.coregistration_analysis import coregistration_analysis as ca
from METEOR_core.functions import reader
from METEOR_core.intensity_analysis import intensity_analysis as ia
from METEOR_core.logger import setup_logger


def debug_intensity(idx):
    path = "upload/intensity/"
    filenames = os.listdir(path)
    filenames = [
        "LSM880_20x_intensity.czi",
        "LSM700_40xO_intensity.czi",
        "LSM700_63x_intensity.czi",
        "LSM880_40xW_intensity.czi",
        "Argo 63x Oil intensity.czi",
        "Argo 20x intensity.czi",
    ]

    img, metadata = reader(path + filenames[idx])

    res = ia(img, metadata, debug=False)
    print("_____________________")
    pprint.pprint(res)
    plt.show()


def debug_coregistration(idx):
    path = "upload/2drings/"
    filenames = os.listdir(path)

    img, metadata = reader(path + filenames[idx])

    res = ca(img, metadata, debug=True)
    print("_____________________")
    pprint.pprint(res)
    plt.show()


if __name__ == "__main__":
    setup_logger()

    # debug_intensity(4)
    debug_coregistration(7)
