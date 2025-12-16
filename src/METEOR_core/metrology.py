"""
Created on Tue Jun 17 15:24:51 2025

@author: simon

DONE:
PSF/RSF once a month
Field illumination
Resolution with the Star pattern
Coregistration
stage drift

TODO:
illumination stability (laser)
stage positioning
camera read noise


METEOR (METrology for Evaluation of Optical Resolution)
-core: a python package for microscope metrology

"""

import traceback

#local import
from . import logger
from .coregistration_analysis import coregistration_analysis
from .distortion_analysis import distortion_analysis
from .intensity_analysis import intensity_analysis
from .plastic_uniformity import calculate_field_uniformity
from .psf_analysis import psf_analysis
from .RSF import RSF_analysis
from .stack_accurancy import stack_precision
from .stage_stability import measure_stage_stability
from .star_resolution import star_resolution_analysis
from .uniformity_analysis import uniformity_analysis


def by_name(name):
    name = name.lower()
    if "ring" in name and '2d' in name:
        return 'Argo_2D_Rings_Analysis'
    if 'intensity' in name:
        return 'Argo_Intensity_Linearity'
    if 'psf' in name or 'bead' in name:
        return 'Psf_Measure'
    if 'plastic' in name or 'platic' in name:
        return 'Plastic_Field_Uniformity'
    if 'cross' in name or 'stair' in name:
        return '3D_Crossing_Stairs'
    if 'star' in name:
        return 'star_resolution'
    return "Unknown"


def classify(name, metadata):
    category = by_name(name)
    if category != "Unknown":
        return category
    #other classifier, like the metadata
    return "Unknown"


def metrology_splitter(img, name, metadata):
    logger.setup_logger()
    category = classify(name, metadata)
    try:
        if category == 'Argo_2D_Rings_Analysis':
            results = []
            results.append(coregistration_analysis(img, 21, 5.1, metadata))
            results.append(distortion_analysis(img, 21, 5.1, metadata))
            results.append(uniformity_analysis(img, 21, metadata))
            results.append(RSF_analysis(img, metadata))
            return results
        elif category == 'Argo_Intensity_Linearity':
            result = intensity_analysis(img, metadata)
            return [result]
        elif category == 'Psf_Measure': #TODO take care of the gif and ome as well - in img_function.py!
            result = psf_analysis(img, .3, metadata, verbose=True)
            return [result]
        elif category == 'Plastic_Field_Uniformity':
            result = calculate_field_uniformity(img, metadata)
            return [result]
        elif category == '3D_Crossing_Stairs':
            # the Z accuracy
            if int(metadata.get("Image Size T", 1)) == 1 and int(metadata.get("Image Size Z", 1) )> 1:
                result = stack_precision(img, metadata)
                return [result]
            # the stage stability
            elif int(metadata.get("Image Size T", 1)) > 1 and int(metadata.get("Image Size Z", 1)) == 1:
                result = measure_stage_stability(img, metadata)
                return [result]
        elif category == 'star_resolution':
            results = star_resolution_analysis(img, metadata)
            return [results]
        else:
            logger.warning(f"{name} category not found. Verify name!")
            raise NameError("Category not found - check image name")
    except (ValueError, KeyError, IndexError, AttributeError, TypeError) as e:
        logger.warning(f"Failed to process the image: {name} - {str(e)}")
        logger.warning(f"{traceback.format_exc()}")
        return False

