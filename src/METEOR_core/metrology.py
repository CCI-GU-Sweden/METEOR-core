"""
src/METEOR_core/metrology.py
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

"""

import traceback

# local import
from . import logger
from .classification import classify_image, filename_hint
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
    return filename_hint(name)


def classify(name, metadata):
    """Legacy filename-only helper; use classify_image for actual routing."""
    category = by_name(name)
    if category != "Unknown":
        return category
    # other classifier, like the metadata
    return "Unknown"


def metrology_splitter(img, name, metadata, *, category_override=None):
    """Route an image using content/metadata, or an explicit category override.

    Returns False if classification requires review or the analysis fails.
    Classification evidence is included in each returned result.
    """
    logger.setup_logger()
    try:
        classification = classify_image(img, metadata, name, override=category_override)
        category = classification["category"]
        for warning in classification["warnings"]:
            logger.warning(f"Classification: {warning}")
        if classification["requires_review"]:
            logger.warning(f"Image {name} requires classification review")
            return False
        if category == "Argo_2D_Rings_Analysis":
            results = []
            for analysis in (
                coregistration_analysis,
                distortion_analysis,
                uniformity_analysis,
                RSF_analysis,
            ):
                try:
                    results.append(analysis(img, metadata))
                except (
                    ValueError,
                    KeyError,
                    IndexError,
                    AttributeError,
                    TypeError,
                    RuntimeError,
                ) as exc:
                    logger.warning(f"{analysis.__name__} failed: {exc}")
                    results.append(
                        {
                            "figure": None,
                            "figure name": analysis.__name__,
                            "status": "error",
                            "measurements": [],
                            "extra_measurements": {"error": str(exc)},
                        }
                    )
        elif category == "Argo_Intensity_Linearity":
            result = intensity_analysis(img, metadata)
            results = [result]
        elif category == "Psf_Measure":
            result = psf_analysis(img, metadata)
            results = [result]
        elif category == "Plastic_Field_Uniformity":
            result = calculate_field_uniformity(img, metadata)
            results = [result]
        elif category == "3D_Crossing_Stairs":
            # the Z accuracy
            if (
                int(metadata.get("Image Size T", 1)) == 1
                and int(metadata.get("Image Size Z", 1)) > 1
            ):
                result = stack_precision(img, metadata)
                results = [result]
            # the stage stability
            elif (
                int(metadata.get("Image Size T", 1)) > 1
                and int(metadata.get("Image Size Z", 1)) == 1
            ):
                result = measure_stage_stability(img, metadata)
                results = [result]
            else:
                raise ValueError(
                    "Crossing stairs requires either a Z stack or a time series, not both"
                )
        elif category == "star_resolution":
            results = [star_resolution_analysis(img, metadata)]
        else:
            logger.warning(f"{name} category not found. Verify name!")
            raise NameError("Category not found - check image name")
        for result in results:
            result["classification"] = classification
        return results
    except (
        NameError,
        ValueError,
        KeyError,
        IndexError,
        AttributeError,
        TypeError,
        RuntimeError,
    ) as e:
        logger.warning(f"Failed to process the image: {name} - {str(e)}")
        logger.warning(f"{traceback.format_exc()}")
        return False
