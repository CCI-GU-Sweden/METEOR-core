import pprint

from matplotlib import pyplot as plt
from METEOR_core.intensity_analysis import intensity_analysis as ia
from METEOR_core.logger import setup_logger


def debug_intensity_analysis(img, metadata):
    res = ia(img, metadata, debug=False)
    return res


if __name__ == "__main__":
    from METEOR_core.functions import reader

    setup_logger()

    filenames = [
        "LSM880_20x_intensity.czi",
        "LSM700_40xO_intensity.czi",
        "LSM700_63x_intensity.czi",
        "LSM880_40xW_intensity.czi",
        "Argo 63x Oil intensity.czi",
        "Argo 20x intensity.czi",
    ]

    img, metadata = reader("upload/intensity/" + filenames[5])

    res = debug_intensity_analysis(img, metadata)
    print("_____________________")
    pprint.pprint(res)
    plt.show()
