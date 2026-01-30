import METEOR_core
from METEOR_core.coregistration_analysis import coregistration_analysis

img, metadata = METEOR_core.functions.reader("upload/60x - 2D rings.czi")  # type: ignore

# res = METEOR_core.star_resolution.star_resolution_analysis(img, metadata)

res = coregistration_analysis(img, metadata, debug=True)
print(res["measurements"])
