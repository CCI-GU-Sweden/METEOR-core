import METEOR_core

img, metadata = METEOR_core.functions.reader("upload/40x_star.czi")

res = METEOR_core.star_resolution.star_resolution_analysis(img, metadata)
print(res["measurements"])
