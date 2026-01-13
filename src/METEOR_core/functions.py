"""
Created on Wed Jun 11 10:08:33 2025

@author: simon

Reused functions for the metrology part


"""

import numpy as np
from dateutil import parser
from scipy import ndimage, optimize, spatial
from skimage import filters, measure, morphology, transform

#local import
from . import logger

VALID_AXES = set("TCZYX")

def list_multiply(li:list):
    li = list(li)
    result = 1
    for x in li:
        result *= x
    return result


def round_to_even(value): #Even number!
    return 2 * round(value / 2)


def tr_pxl_type(pixeltypes):
    if isinstance(pixeltypes, dict):
        pixeltypes = pixeltypes[0]
    if pixeltypes == 'Gray16':
        return np.uint16
    elif pixeltypes == 'Gray8':
        return np.uint8
    else:
        return np.uint16


def norm_img(img:np.ndarray):
    #normalize the image 0-1,security by transforming as float before
    img = img.astype(float)
    return (img - np.min(img))/(np.max(img) - np.min(img))


def zscore_norm(arr:np.ndarray):
    return (arr - np.mean(arr))/np.std(arr)


def focus_score_laplacian(img_2d: np.ndarray) -> float:
    """
    Simple focus score: variance of Laplacian.
    Higher = sharper.
    """
    img_norm = norm_img(img_2d)
    return float(filters.laplace(img_norm).var())


def get_info_metadata_from_czi(img_path:str, verbose:bool=False) -> dict:
    """
    Extract important metadata from a CZI image file.

    This function opens a CZI image file, reads its metadata, and extracts
    specific information such as microscope details, lens properties,
    image type, pixel size, image dimensions, and other relevant metadata.

    Args:
        img_path : The file path to the CZI image.
        verbose (bool, optional): If True, print detailed information during processing. Defaults to True.

    Returns:
        ImageMetadata: A dictionnary containing the extracted metadata.

    Raises:
        FileNotFoundError: If the specified image file does not exist.
        ValueError: If the file is not a valid CZI image or if metadata extraction fails.
    """

    try:
        from pylibCZIrw import czi as pyczi
        with pyczi.open_czi(img_path) as czidoc:
            metadata = czidoc.metadata['ImageDocument']['Metadata']
    except FileNotFoundError as e:
        raise FileNotFoundError("The file does not exist.") from e
    except Exception as e:
        raise ValueError(f"Error opening or reading metadata: {str(e)}") from e

    #Initialization
    app_name = None
    app_version = None
    microscope = ''
    acq_type = None
    lensNA = None
    lensMag = None
    pre_processed = None
    comment = None
    description = None
    creation_date = None

    #grab the correct version of the metadata
    app = metadata['Information'].get('Application', None)
    if app is not None: #security check
        app_name = app['Name']
        app_version = app['Version']
        if verbose:
            logger.info(f'Metadata made with {app_name} version {app_version}')
        #microscope name, based on the version of the metadata. Do NOT get ELYRA microscope
        #Another way will be to grab the IP address of the room and map it
        if 'ZEN' in app['Name'] and 'blue' in app['Name'] and app['Version'].startswith("3."): #CD7, 980
            microscope += metadata['Scaling']['AutoScaling'].get('CameraName', "") + ", "
            microscope += metadata['Information']['Instrument']['Microscopes']['Microscope'].get('@Name', None)
            #hardcoded part :(
            if 'Axiocam 705 mono' in microscope:
                microscope = microscope.replace('Axiocam 705 mono', 'LSM 980')

        elif 'ZEN' in app['Name'] and 'blue' in app['Name'] and app['Version'].startswith("2."): #Observer
            microscope = metadata['Information']['Instrument']['Microscopes']['Microscope'].get('@Name', None)

        elif 'AIM' in app['Name']: #ELYRA, 700, 880
            microscope = metadata['Information']['Instrument']['Microscopes']['Microscope'].get('System', None)
            #hardcoded part :(
            if 'Andor1' in microscope:
                microscope = microscope.replace('Andor1', 'Elyra')

        if verbose:
            logger.info(f'Image made on {microscope}')
        #pixel size (everything in the scaling)
        physical_pixel_sizes = {}
        for dim in metadata['Scaling']['Items']['Distance']:
            physical_pixel_sizes[dim['@Id']] = round(float(dim['Value'])*1e+6, 4)

        #image dimension
        dims = metadata['Information']['Image']
        size = {}
        for d in dims:
            if 'Size' in d: #just the different Size (X,Y,Z,C,M,H...)
                size[d] = int(dims[d])
        if verbose:
            logger.info(f'Image with dimension {size} and pixel size of {physical_pixel_sizes}')

        # Acquisition type (not fully correct with elyra)
        acq_type = metadata['Information']['Image']['Dimensions']['Channels']['Channel']
        if isinstance(acq_type, list):
            acq_type = acq_type[0].get('ChannelType', acq_type[0].get('AcquisitionMode', None))
            if acq_type == 'Unspecified':
                acq_type = metadata['Information']['Image']['Dimensions']['Channels']['Channel'][0].get('AcquisitionMode', None)
        elif isinstance(acq_type, dict):
            acq_type = acq_type.get('AcquisitionMode', None)
        if verbose:
            logger.info(f'Image acquired with a {acq_type} mode')

        #lens info
        lensNA = metadata['Information']['Instrument']['Objectives']['Objective'].get('LensNA', None)
        if lensNA is not None:
            lensNA = round(float(lensNA), 2)
        lensMag = metadata['Information']['Instrument']['Objectives']['Objective'].get('NominalMagnification', None)
        if lensMag is not None:
            lensMag = int(lensMag)
        if verbose:
            logger.info(f'Objective lens used has a magnification of {lensMag} and a NA of {lensNA}')

        #processing (if any)
        processing = metadata['Information'].get('Processing', None)
        if processing is not None:
            pre_processed = list(processing.keys())
        if verbose:
            logger.info(f'Image preprocessed with {pre_processed}')

        #other
        comment = metadata['Information']['Document'].get('Comment', None)
        description = metadata['Information']['Document'].get('Description', None)
        creation_date = metadata['Information']['Document'].get('CreationDate', None)
        creation_date = parser.parse(creation_date).strftime("%Y-%m-%d")

    mini_metadata = {'Microscope':microscope,
                     'Lens Magnification': lensMag,
                     'Lens NA': lensNA,
                     'Image type':acq_type,
                     'Comment':comment,
                     'Description':description,
                     'Acquisition date':creation_date,
                     'Dimension':size,
                     }

    #add the pixel size
    [mini_metadata.update({f'Physical pixel size {x}': physical_pixel_sizes[x]}) for x in physical_pixel_sizes]
    # canonical: 'Image Size X', 'Image Size Y', 'Image Size Z', 'Image Size C', 'Image Size T', 'Image Size S'
    for ax in ("X", "Y", "Z", "C", "T", "S"):
        dim_key = f"Size{ax}"  # e.g. 'SizeC'
        # T and S (and sometimes Z) might be missing; default to 1
        # X/Y should basically always be present for valid images.
        val = int(size[dim_key]) if dim_key in size else 1
        mini_metadata[f"Image Size {ax}"] = val

    return mini_metadata


def reader(filepath:str):
    file_support = ['czi']
    file_format = filepath.split('.')[-1]

    if file_format not in file_support:
        raise ValueError(f"Unsupported file format: {file_format}. Supported formats: {str(file_support)}")

    if filepath.endswith('.czi'):
        img, metadata = czi_reader_tzcxy(filepath)

    return img, metadata

def czi_reader_tzcxy(filepath:str) -> tuple[np.ndarray, dict]:
    #read the image data and metadata
    metadata = get_info_metadata_from_czi(filepath)
    metadata['filepath'] = filepath

    size_s = int(metadata.get("Image Size S", 1))
    size_t = int(metadata.get("Image Size T", 1))
    size_c = int(metadata.get("Image Size C", 1))
    size_z = int(metadata.get("Image Size Z", 1))
    size_y = int(metadata.get("Image Size Y", 1))
    size_x = int(metadata.get("Image Size X", 1))

    try:
        from pylibCZIrw import czi as pyczi
        with pyczi.open_czi(filepath) as czidoc:
            pixeltypes = tr_pxl_type(czidoc.pixel_types)

            # Check if this is a mosaic/tiled image
            total_bbox = czidoc.total_bounding_box
            s_info = total_bbox.get('S', []) if total_bbox else []
            is_mosaic = size_s > 1 or (len(s_info) > 1 and s_info[1] > 1)
            
            logger.info(f"Reading CZI: S={size_s}, mosaic={is_mosaic}, bbox={total_bbox}")

            img_array = np.zeros(
                (size_s, size_t, size_c, size_z, size_y, size_x),
                dtype=pixeltypes,
            )

            for s in range(size_s):
                for t in range(size_t):
                    for z in range(size_z):
                        for c in range(size_c):
                            try:
                                if is_mosaic:
                                    # Use read_mosaic for stitched multi-tile images
                                    result = czidoc.read_mosaic(plane={'T':t, 'C':c, 'Z':z}, scene=s, scale_factor=1)
                                    # Handle different return types
                                    if isinstance(result, tuple):
                                        img = result[0]
                                    else:
                                        img = result
                                else:
                                    # Use regular read for single-tile images
                                    if size_s > 1:
                                        img = czidoc.read(plane={'T':t, 'C':c, 'Z':z}, scene=s)
                                    else:
                                        img = czidoc.read(plane={'T':t, 'C':c, 'Z':z})

                                img_squeezed = np.squeeze(img)

                                # Handle shape mismatch (shouldn't happen with read_mosaic, but safety)
                                if img_squeezed.shape != (size_y, size_x):
                                    h, w = img_squeezed.shape
                                    crop_h = min(h, size_y)
                                    crop_w = min(w, size_x)
                                    logger.warning(f"Shape mismatch at S{s}T{t}C{c}Z{z}: expected ({size_y},{size_x}), got ({h},{w}). Cropping.")
                                    img_array[s,t,c,z, :crop_h, :crop_w] = img_squeezed[:crop_h, :crop_w]
                                else:
                                    img_array[s,t,c,z] = img_squeezed
                                    
                            except Exception as e:
                                logger.error(f"Error reading plane S{s}T{t}C{c}Z{z}: {str(e)}")
                                raise

    except FileNotFoundError:
        raise FileNotFoundError("The file does not exist.")
    except Exception as e:
        logger.error(f"Full error details: {type(e).__name__}: {str(e)}")
        raise ValueError(f"Error opening or reading the file: {str(e)}")    

    logger.info(f'Opening image {filepath}')

    if metadata.get('Image Size S', 1) == 1:
        img_array = img_array[0]
        metadata['axis_order'] = 'TCZYX'
    else:
        metadata['axis_order'] = 'STCZYX'

    return img_array, metadata


def dimension_slicer(img: np.ndarray, metadata: dict, dimension: str) -> np.ndarray:
    """
    Slice/reduce an image array to a desired axis layout based on metadata['axis_order'].

    Parameters
    ----------
    img : np.ndarray
        Input image array. Its axes are ordered according to metadata['axis_order'].
        Example axis_order: 'TCZYX' → img.shape = (T, C, Z, Y, X).
    metadata : dict
        Metadata dictionary, must contain key 'axis_order' describing the current
        axis ordering using letters in {'T', 'C', 'Z', 'Y', 'X'}.
    dimension : str
        Target axis string, e.g.:
          - 'CZYX' for coregistration
          - 'ZYX'  for RSF
          - 'TYX'  for stage stability
        The target must be a subsequence of metadata['axis_order'], not a permutation.

    Returns
    -------
    np.ndarray
        View/slice of `img` reduced to the requested axes and in the same order
        as `dimension`. All axes that are present in axis_order but *not* in
        `dimension` must have size 1, otherwise a ValueError is raised.

    Raises
    ------
    KeyError
        If 'axis_order' is missing in metadata.
    ValueError
        If `dimension` contains invalid axes, or an axis in `dimension` is not in
        metadata['axis_order'], or an axis you try to drop has size > 1.
    """
    if "axis_order" not in metadata:
        raise KeyError("metadata['axis_order'] is required for dimension_slicer")

    axis_order = metadata["axis_order"]
    axis_order = axis_order.upper()
    dimension = dimension.upper()

    # Basic validation
    if not set(dimension).issubset(VALID_AXES):
        raise ValueError(f"dimension '{dimension}' contains invalid axes; "
                         f"allowed axes are {sorted(VALID_AXES)}")

    # Ensure all target axes exist in axis_order
    for ax in dimension:
        if ax not in axis_order:
            raise ValueError(
                f"Requested axis '{ax}' in dimension '{dimension}' "
                f"but it's not present in metadata['axis_order']='{axis_order}'"
            )

    # Ensure `dimension` is a subsequence of axis_order, not a permutation
    # (e.g., axis_order='TCZYX', dimension='CZYX' is OK, but 'ZCX' is not).
    pos = 0
    for ax in dimension:
        try:
            pos = axis_order.index(ax, pos) + 1
        except ValueError:
            raise ValueError(
                f"dimension '{dimension}' is not a subsequence of axis_order "
                f"'{axis_order}'. The order of axes must be preserved."
            )

    if img.ndim != len(axis_order):
        raise ValueError(
            f"Image ndim={img.ndim} is inconsistent with axis_order='{axis_order}' "
            f"(length {len(axis_order)})."
        )

    # Build slicing tuple: keep target axes, drop others (only if size == 1)
    slices = []
    for ax, size in zip(axis_order, img.shape):
        if ax in dimension:
            # keep full axis
            slices.append(slice(None))
        else:
            # drop this axis; it must be singleton
            if size != 1:
                raise ValueError(
                    f"Cannot drop axis '{ax}' of size {size} when slicing to "
                    f"dimension '{dimension}'. Pre-select it (e.g. with c_index/t_index) "
                    f"or include it in the target dimension."
                )
            # index at 0 → removes axis
            slices.append(0)

    sliced = img[tuple(slices)]

    # At this point the remaining axes are exactly those in `dimension`,
    # and in the same order, so we can assert:
    if sliced.ndim != len(dimension):
        logger.warning(
            f"[dimension_slicer] Unexpected ndim after slicing: got {sliced.ndim}, "
            f"expected {len(dimension)} for dimension '{dimension}'."
        )

    return sliced

def get_label(img:np.ndarray, closing_iteration:int=4):
    thresh = filters.threshold_otsu(img)
    binary = img > thresh
    binary = ndimage.binary_closing(binary, iterations=closing_iteration) #security, close the cross!
    binary = ndimage.binary_opening(binary) #security, remove single pixels
    binary = morphology.remove_small_holes(binary) #in case of holes
    return measure.label(binary)


def generate_lattice(params:np.ndarray, grid_size:int):
    """
    Generates a square lattice based on the given parameters.

    Parameters:
    ----------
    params : array
        Lattice parameters: [cx, cy, angle, spacing].
        cx, cy: Lattice top left.
        angle: Lattice rotation in radians.
        spacing: Lattice spacing.
    grid_size : int
        Number of points along one side of the square lattice.

    Returns:
    -------
    ndarray
        Array of lattice points of shape (n^2, 2).
    """
    cx, cy, angle, spacing = params
    n = grid_size

    # Generate lattice points in the local (unrotated) frame
    grid_x, grid_y = np.meshgrid(np.arange(n), np.arange(n))
    lattice = np.column_stack([grid_x.ravel(), grid_y.ravel()]) * spacing

    # Rotate the lattice points
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    rotated_lattice = lattice @ rotation_matrix.T  # Matrix multiplication

    # Translate the lattice
    translated_lattice = rotated_lattice + np.array([cx, cy])

    return translated_lattice


def lattice_cost_penalized(params:np.ndarray,
                           points:np.ndarray,
                           grid_size:int,
                           keep_percentage:float=0.9,
                           anchor_point=None,
                           expected_spacing_px=None,
                           image_shape=None):
    """
    Cost function to fit a square lattice to points using KDTree for nearest neighbor search.
    Includes constraints for the top-left square position and excludes a percentage of points with the largest distances.

    Parameters:
    ----------
    params : array
        Lattice parameters: [cx, cy, angle, spacing].
        cx, cy: Lattice top left.
        angle: Lattice rotation in radians.
        spacing: Lattice spacing.
    points : ndarray
        Array of detected points of shape (n, 2).
    grid_size : int
        Number of points along one side of the square lattice.
    keep_percentage : float, optional
        The percentage of points to keep in the cost calculation (default is 90%).
    top_left_square : tuple or None, optional
        Known (x, y) coordinates of the top-left square. If None, no constraints are applied.

    Returns:
    -------
    float
        The cost (sum of squared distances for the closest points with additional penalties for top-left constraints).
    """
    assert 0 < keep_percentage <= 1, "keep_percentage must be between 0 and 1"

    # Generate the lattice points
    translated_lattice = generate_lattice(params, grid_size)

    # Use KDTree for efficient nearest neighbor search
    tree = spatial.KDTree(translated_lattice)
    distances, _ = tree.query(points)

    # Sort distances and keep the closest points based on keep_percentage
    num_points = len(distances)
    num_to_keep = max(1, int(num_points * keep_percentage))
    distances_sorted = np.sort(distances)
    valid_distances = distances_sorted[:num_to_keep]

    # Base cost: Sum of squared distances for valid points
    cost = np.sum(valid_distances ** 2)

    # Penalize spacing drift if requested
    spacing_px = float(params[3])
    if expected_spacing_px is not None and expected_spacing_px > 0:
        rel_dev = (spacing_px - expected_spacing_px) / expected_spacing_px
        cost *= (1.0 + 2.0 * rel_dev**2)

    # Penalize displacement of lattice **center** from anchor_point
    if anchor_point is not None:
        # center of the lattice in coordinates
        lattice_center = translated_lattice.reshape(grid_size, grid_size, 2)[
            grid_size // 2, grid_size // 2
        ]
        displacement = np.hypot(
            lattice_center[0] - anchor_point[0],
            lattice_center[1] - anchor_point[1],
        )
        if expected_spacing_px is not None and expected_spacing_px > 0:
            disp_norm = displacement / expected_spacing_px
        else:
            disp_norm = displacement
        # softer penalty than before
        cost *= (1.0 + 0.1 * disp_norm**2)

    # Out-of-bounds penalty
    if image_shape is not None:
        H, W = image_shape  # img.shape -> (height, width)
        xs = translated_lattice[:, 0]
        ys = translated_lattice[:, 1]

        # distance outside each edge (0 if inside)
        dx_left  = np.maximum(0.0, -xs)
        dx_right = np.maximum(0.0, xs - (W - 1))
        dy_top   = np.maximum(0.0, -ys)
        dy_bot   = np.maximum(0.0, ys - (H - 1))

        # Euclidean distance to nearest valid point for out-of-bounds nodes
        out_dist = np.sqrt((dx_left + dx_right)**2 + (dy_top + dy_bot)**2)
        # Only penalize nodes truly outside
        out_dist_sq_sum = float(np.sum(out_dist**2))

        if out_dist_sq_sum > 0:
            # Normalize by spacing to make penalty scale-invariant-ish
            if expected_spacing_px is not None and expected_spacing_px > 0:
                norm = expected_spacing_px**2 * grid_size**2
            else:
                norm = max(1.0, grid_size**2)

            # beta controls how strongly we dislike going out of bounds
            beta = 10.0
            cost *= (1.0 + beta * (out_dist_sq_sum / norm))

    return float(cost)


def sort_grid(points:np.ndarray, gridsize:int=21):
    """
    Sorts a (gridsizexgridsize) distorted grid of 2D points into row-major order.

    Parameters:
        points (numpy array): Nx2 array of (x, y) coordinates, lenght should be gridsizexgridsize

    Returns:
        sorted_points (numpy array): Sorted Nx2 array.
        sort_indices (numpy array): Indices of sorting.
    """
    points = np.array(points)  # Ensure it's a numpy array
    sorted_indices = []

    # Step 1: Sort all points by X first
    x_sorted_indices = np.argsort(points[:, 0])
    points = points[x_sorted_indices]

    # Step 2: Group into 21 sets (each supposed to be a column), then sort by Y
    sorted_points = np.zeros_like(points)
    for i in range(gridsize):
        column_points = points[i*gridsize:(i+1)*gridsize]
        column_indices = x_sorted_indices[i*gridsize:(i+1)*gridsize]

        y_sorted_indices = np.argsort(column_points[:, 1])
        sorted_points[i*gridsize:(i+1)*gridsize] = column_points[y_sorted_indices]
        sorted_indices.extend(column_indices[y_sorted_indices])

    return sorted_points, np.array(sorted_indices)


def lattice_fitting(points, grid_size, spacing, pxl_size_xy:float,
                    return_rawlattice:bool=False, return_points:bool=False,
                    max_rms_fraction: float = 0.5, image_shape=None,
                    detection_percentage:float=0.90):

    spacing_micron = spacing
    expected_spacing_px = spacing_micron / pxl_size_xy
    center_point = points.mean(axis=0)
    # Initial guess for parameters: [cx, cy, angle, spacing]
    initial_params = [
        center_point[0],  # Center x
        center_point[1],  # Center y
        0.0,                    # Angle
        expected_spacing_px   # Spacing
    ]

    # Optimize the lattice parameters
    result = optimize.minimize(lattice_cost_penalized, initial_params,
                               args=(points, grid_size, detection_percentage, center_point, 
                                     expected_spacing_px, image_shape),
                                     method='Powell')
    translated_lattice = generate_lattice(result.x, grid_size)      # Generate the lattice points 

    if return_rawlattice:
        return translated_lattice

    # return translated_lattice
    points_sorted, _ = sort_grid(points, gridsize=grid_size)
    tree = spatial.KDTree(translated_lattice)
    dists, idx = tree.query(points_sorted)
    paired_lattice = translated_lattice[idx]

    if return_points:
        return paired_lattice, points_sorted
    else:
        return paired_lattice


def norm_center_img_3D(img, edge=3):
    #Detect ROI
    thresh = filters.threshold_otsu(img)
    # thresh = filters.threshold_yen(img)
    binary = img > thresh
    binary = morphology.opening(binary)
    # Find the bounding box of the ROI
    y = np.any(binary, axis=(0,1))
    x = np.any(binary, axis=(0,2))
    z = np.any(binary, axis=(1,2))
    zmin, zmax = np.where(z)[0][[0, -1]]
    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    # Add a small edge, taking care of the image dimension as well!
    zmin = max(0, zmin-edge)
    zmax = min(zmax+edge, img.shape[0])
    xmin = max(0, xmin-edge)
    xmax = min(xmax+edge, img.shape[1])
    ymin = max(0, ymin-edge)
    ymax = min(ymax+edge, img.shape[2])
    # Crop the image to the ROI
    img_roi = img[zmin:zmax+1, xmin:xmax+1, ymin:ymax+1]
    img_roi = img_roi.astype(np.float32) #required
    #substract and clip the background value from the img_roi
    img_array = img.astype(np.float32)
    img_array[zmin:zmax+1, xmin:xmax+1, ymin:ymax+1] = np.nan
    bck = np.nanmean(img_array) #issue there?

    if not np.isnan(bck):
        img_roi = np.clip(img_roi - bck, 0, None)

    return norm_img(img_roi)


def img_best_rotation(img:np.ndarray):
    """
    rotate an image so that the top and bottom are aligned to the matrix.
    Assume a square like structure
    Parameters
    ----------
    img : np.array
        2D array of the image to rotate.

    Returns
    -------
    final_angle : float
        the angle to rotate the image with

    """
    #found the angle to rotate the image with
    img = filters.gaussian(img)
    thr = filters.threshold_otsu(img)
    binary = img > thr
    binary = morphology.binary_closing(binary)
    binary = morphology.binary_opening(binary)
    final_angle = 0
    i = 0
    prev_angle = np.Inf
    while True:
        # Find the coordinates of non-zero pixels
        coords = np.column_stack(np.where(binary))

        # Find extreme points
        top_left = coords[np.argmin(coords.sum(axis=1))]
        bottom_right = coords[np.argmax(coords.sum(axis=1))]
        top_right = coords[np.argmax(coords[:, 1] - coords[:, 0])]
        bottom_left = coords[np.argmin(coords[:, 1] - coords[:, 0])]

        #calculate the angle to rotate
        angle1 = np.degrees(np.arctan2(top_right[0] - top_left[0], top_right[1] - top_left[1]))
        angle2 = np.degrees(np.arctan2(bottom_right[0] - bottom_left[0], bottom_right[1] - bottom_left[1]))

        angle = np.sum((angle1, angle2))/2
        final_angle += angle
        #rotate
        binary = transform.rotate(binary, angle, preserve_range=True)
        if abs(angle) < 0.01: #no much correction anymore
            break
        if i > 10:
            break #security
        if angle == prev_angle:
            break
        i += 1
        prev_angle = angle

    return final_angle


def get_label_centroid(label_img) -> list:
    centers = []
    for region in measure.regionprops(label_img):
        y, x = region.centroid
        centers.append((x, y))
    return centers

def get_centroid(img, nb_points:int, min_size:int=20, tolerance: float = 0.05,):
    """
    Detect blob centroids in an image.

    Parameters
    ----------
    img : np.ndarray
        2D image.
    nb_points : int or None
        Expected number of blobs (e.g. 21*21=441 for a 21x21 grid).
        If None, no count check is performed.
    min_size : int
        Minimum region area (in pixels) to keep a blob.
    tolerance : float
        Relative tolerance on the expected number of blobs.
        E.g. 0.05 → accept between 95% and 105% of nb_points.

    Returns
    -------
    centers : np.ndarray of shape (N, 2) or None
        Detected centroids as (x, y). Returns None only if
        the number of detected blobs is *far* from nb_points.
    """
    labels = get_label(img)

    # Measure regions and filter by area
    regions = measure.regionprops(labels)
    centers = []
    for region in regions:
        if region.area < min_size:
            continue
        y, x = region.centroid
        centers.append((x, y))

    centers = np.asarray(centers, dtype=float)
    n_detected = centers.shape[0]

    if nb_points is not None:
        if n_detected == 0:
            logger.warning(
                f"[get_centroid] Expected ~{nb_points} blobs, but detected 0 after filtering."
            )
            return None

        lower = int(nb_points * (1.0 - tolerance))
        upper = int(nb_points * (1.0 + tolerance))

        if n_detected < lower or n_detected > upper:
            # If we are *way* off (e.g. something really went wrong), fail
            if n_detected < nb_points * 0.7:
                logger.warning(
                    f"[get_centroid] Detected only {n_detected} blobs vs expected {nb_points} "
                    f"(< 70% of expected) – returning None."
                )
                return None

            logger.warning(
                f"[get_centroid] Detected {n_detected} blobs vs expected {nb_points}. "
                f"Proceeding anyway (tolerance {tolerance*100:.1f}%)."
            )
        else:
            # Nice to log when we're close but not exact
            if n_detected != nb_points:
                logger.info(
                    f"[get_centroid] Detected {n_detected}/{nb_points} blobs "
                    f"(within tolerance {tolerance*100:.1f}%)."
                )

    return np.asarray(centers)


def crop_with_border(img, edge_border:int=4, pxl_size_xy=None):
    """
    Crop an image to the selection defined by the binary, and remove some borders as well.
    Border is defined as pixel if pxl_size_xy is None, otherwise, the pixel_size_xy should be a float indicate the pixel size
    In this case, int(round(edge_border/pxl_size_xy)) will be removed
    Parameters
    ----------
    img : np.array
        The image to crop
    binary : np.array bool
        The selection on which the crop is based
    edge_border : float or int, optional
        The edge to remove. The default is 4.
    pxl_size_xy : None or float, optional
        The pixel size. The default is None.

    Returns
    -------
    top_left: 2D coordinates
    bottom right: 2D coordinates

    """
    thr = filters.threshold_otsu(img)
    binary = img > thr
    binary = morphology.binary_closing(binary)
    binary = morphology.binary_opening(binary)
    # Find the coordinates of non-zero pixels
    coords = np.column_stack(np.where(binary))
    # Find extreme points
    top_left = coords[np.argmin(coords.sum(axis=1))]
    bottom_right = coords[np.argmax(coords.sum(axis=1))]
    if pxl_size_xy is None:
        mic = edge_border
    else: #translate as size to pixel
        mic = int(round(edge_border/pxl_size_xy))
    top_left = top_left + mic
    bottom_right = bottom_right - mic
    
    return top_left, bottom_right


def uniformity(img_array:np.ndarray) -> float:
    return 100 * np.min(img_array) / np.max(img_array)

def centuring_acc(img_array:np.ndarray) -> float:
    """
    Calculate the centering accurancy of an image. Max is 100.
    The formula is from https://pmc.ncbi.nlm.nih.gov/articles/PMC9526251/

    Parameters
    ----------
    img_array : np.array
        2D array to measure the centering accurancy on.

    Returns
    -------
    float
        Centering value.

    """
    coo = np.where(img_array == np.max(img_array))
    x_max = coo[0][0]
    y_max = coo[1][0]
    x, y = img_array.shape
    x_center = x/2
    y_center = y/2
    centering = 100-(100*np.sqrt((x_max-x_center)**2+(y_max-y_center)**2)*np.sqrt(2/(x**2+y**2)))

    return centering

def rings_detection(img_array:np.ndarray, metadata:dict) -> np.ndarray:
    if img_array is None:
        raise ValueError("img_array cannot be None")
    footprint_radius = 3 if int(metadata.get('Lens Magnification', 0)) > 40 else 1
    binary = img_array > filters.threshold_otsu(img_array)
    binary = ndimage.binary_fill_holes(binary)

    if binary.ndim == 3 and metadata.get("Image Size Z", 0) > 1: # type: ignore #check at least 3D and Zstack!
        binary = morphology.binary_closing(binary, footprint=morphology.ball(footprint_radius))
        # binary = morphology.binary_opening(binary, footprint=morphology.ball(footprint_radius))
    else:
        binary = morphology.binary_closing(binary, footprint=morphology.disk(footprint_radius))
        # binary = morphology.binary_opening(binary, footprint=morphology.disk(footprint_radius))

    return binary


def aggregate_close_points(points_um, diameter_um=1.0, z_scale=0.0, twoD=False, weights=None, return_groups=False):
    """
    Merge points that are within 'diameter_um' of each other (using transitive closure).
    points_um : (N,3) array in microns (order: z, y, x)
    diameter_um : cluster diameter (µm); points within radius=diameter/2 are merged
    z_scale : scale factor for z in distance metric (e.g., 1.0 means isotropic;
              use z_scale=pxl_size_xy/pxl_size_z if you want isotropic *in pixels*)
    weights : optional (N,) array of weights (e.g., intensities) for weighted averaging
    return_groups : if True, also return list of index arrays per cluster
    """
    if len(points_um) == 0:
        return points_um.copy(), ([] if return_groups else None)

    r = diameter_um / 2.0

    # scale z if desired (e.g., to compensate anisotropy)
    if twoD:
        scale = np.array([1.0, 1.0], dtype=float)
    else:
        scale = np.array([z_scale, 1.0, 1.0], dtype=float)
    pts_scaled = points_um * scale

    # pairwise distances (scaled), adjacency by radius
    D = spatial.distance.cdist(pts_scaled, pts_scaled)
    adj = D <= r

    # find connected components
    N = adj.shape[0]
    visited = np.zeros(N, dtype=bool)
    groups = []
    for i in range(N):
        if visited[i]:
            continue
        # BFS/DFS
        stack = [i]
        comp = []
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            # neighbors of u not yet visited
            nbrs = np.where(adj[u] & (~visited))[0]
            if nbrs.size:
                visited[nbrs] = True
                stack.extend(nbrs)
        groups.append(np.array(comp, dtype=int))

    # average each component (optionally weighted)
    agg = []
    if weights is None:
        for g in groups:
            agg.append(points_um[g].mean(axis=0))
    else:
        w = np.asarray(weights, dtype=float)
        for g in groups:
            wg = w[g]
            wg_sum = wg.sum()
            if wg_sum <= 0:
                agg.append(points_um[g].mean(axis=0))
            else:
                agg.append((points_um[g] * wg[:, None]).sum(axis=0) / wg_sum)

    agg = np.vstack(agg) if len(agg) else np.empty((0,3), dtype=float)
    return (agg, groups) if return_groups else (agg, None)