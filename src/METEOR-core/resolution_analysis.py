# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 09:21:27 2025

@author: simon

Deprecated. Very unstable and does not give most of the time good result.
Look instead for the star (TODO)
"""
#general import
import numpy as np
import os

#analysis import
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize, stats
from skimage import transform, filters, morphology

#local import
import database as db
import logger
from functions import img_best_rotation, reader
import conf
plt.rcParams['font.family'] = conf.MATPLOTLIB_FONT


def detect_orientation(img, verbose=False):
    # Calculate variances
    var_x = np.var(img, axis=0).mean()  # Variance along x-axis
    var_y = np.var(img, axis=1).mean()  # Variance along y-axis
    
    # Calculate variance along 45 degree diagonal
    diag = [np.diagonal(img, offset=i) for i in range(-img.shape[0]+1, img.shape[1])]
    var_45 = np.mean([np.var(d) for d in diag])
    
    # Print variances for debugging
    if verbose: logger.info(f"Variances - X: {var_x}, Y: {var_y}, 45°: {var_45}")
    
    # Determine the orientation based on which variance is lowest
    var = min(var_x, var_y, var_45)
    if var == var_y:
        return 0   # Lines are horizontal (no rotation needed)
    elif var == var_x:
        return 90  # Lines are vertical (rotate 90 degrees)
    else:
        return 45  # Lines are diagonal


def analyze_gradual_lines(line_profile, pxl_size, factor=.5, debug=False):
    min_distance = int(2. // pxl_size) #in theory min distance of 2.4 micron between pattern
    # Find main valleys separating sets
    main_valleys = signal.find_peaks(1-line_profile,
                                     distance=min_distance,
                                     # height=.75,
                                     prominence=factor*np.ptp(line_profile),
                                     # width=3,
                                     )[0]
    
    # Add artificial valleys at the beginning and end
    main_valleys = np.concatenate(([0], main_valleys, [len(line_profile)-1]))
    
    results = []
    
    for i in range(len(main_valleys) - 1):
        start, end = main_valleys[i], main_valleys[i+1]
        segment = line_profile[start:end]
        segment_x = np.arange(len(segment))


        # Find valleys within the segment
        inner_valleys = signal.find_peaks(-segment, prominence=0.05*np.ptp(segment))[0]
        inner_valleys = np.concatenate(([0], inner_valleys, [len(segment)-1]))
        
        peak_params = []
        for j in range(len(inner_valleys) - 1):
            peak_start, peak_end = inner_valleys[j], inner_valleys[j+1]
            peak = segment[peak_start:peak_end]
            peak_x = segment_x[peak_start:peak_end]
            
            params = fit_single_gaussian(peak_x, peak)
            if params is not None:
                peak_params.append(params)
        
        if len(peak_params) >= 2:
            distances = np.diff([p[1] for p in peak_params])
            central_distance = 'n/a'
            contrast_ratio = np.nan
            if len(peak_params) == 4:
                central_distance = peak_params[2][1] - peak_params[1][1]
                contrast_ratio = calculate_peak_valley_ratio(peak_params, segment)
                if central_distance * pxl_size < 0.1: #less than 100 nm is not realistic!
                    central_distance = 'n/a'
                    contrast_ratio = np.nan
            results.append((distances, peak_params, start, central_distance, contrast_ratio))
        
    if debug:
        plt.figure()
        #plot the line
        line_profile_um = np.arange(0, len(line_profile)) * pxl_size   
        plt.plot(line_profile_um, 1-line_profile, label='Line Profile')
        plt.vlines(line_profile_um[main_valleys], ymin=min(line_profile), ymax=max(line_profile), color='r', linestyle='--', label='Set Boundaries')
        
        # Add central distance labels
        for i, (_, _, start, central_distance, _) in enumerate(results):
            if i < len(main_valleys) - 1:
                x_center = (line_profile_um[main_valleys[i]] + line_profile_um[main_valleys[i+1]]) / 2
                if central_distance != 'n/a':
                    label = f"{central_distance * pxl_size:.3f} µm"
                else:
                    label = "n/a"
                plt.text(x_center, min(line_profile), label, ha='center', va='bottom', fontsize=8, rotation=90)
        
        plt.ylabel("Intensity (a.u.)")
        plt.xlabel("µm")
        plt.title("Line profile")
        plt.show()
    
    return results, main_valleys


def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / (2 * wid**2))


def fit_single_gaussian(x, y):
    initial_guess = [np.max(y), x[np.argmax(y)], (x[-1] - x[0]) / 4]
    try:
        popt, _ = optimize.curve_fit(gaussian, x, y, p0=initial_guess)
        return popt
    except:
        return None
    
    
def calculate_peak_valley_ratio(peak_params, segment):
    if len(peak_params) == 4:
        # Extract positions of 2nd and 3rd peaks
        peak2_pos = int(peak_params[1][1])
        peak3_pos = int(peak_params[2][1])
        
        # Find the minimum intensity between these peaks
        valley_region = segment[peak2_pos:peak3_pos]
        if len(valley_region) > 0:
            valley_amp = np.min(valley_region)
        else:
            return None  # No valley between peaks
        valley_amp = np.min(valley_region)
        
        # Calculate average peak intensity
        peak2_amp = np.max(segment[max(0, peak2_pos-5):min(len(segment), peak2_pos+6)])
        peak3_amp = np.max(segment[max(0, peak3_pos-5):min(len(segment), peak3_pos+6)])
        avg_peak = (peak2_amp + peak3_amp) / 2
        
        # Calculate contrast ratio
        contrast_ratio = (avg_peak - valley_amp) / (avg_peak + valley_amp)
        
        return contrast_ratio
    else:
        return None


def create_results_table(results, pxl_size):
    # Extract central distances and convert to microns
    central_distances = [r[3] if r[3] != 'n/a' else np.nan for r in results]
    central_distances_um = [d * pxl_size if not np.isnan(d) else np.nan for d in central_distances]
    contrast = [r[4] for r in results]
    
    # Create expected distances (0 to 0.39 um in steps of 0.03 um)
    expected_distances = np.arange(0, 0.03*len(central_distances), 0.03)
    
    # Determine orientation (increasing or decreasing)
    valid_distances = [d for d in central_distances_um if not np.isnan(d)]
    if len(valid_distances) > 1:
        orientation = 'Increasing' if valid_distances[-1] > valid_distances[0] else 'Decreasing'
    else:
        orientation = 'Unknown'
    
    # Align expected distances based on orientation
    if orientation == 'Decreasing':
        expected_distances = expected_distances[::-1]
    
    # Ensure lengths match
    max_length = max(len(central_distances_um), len(expected_distances), len(contrast))
    central_distances_um = central_distances_um + [np.nan] * (max_length - len(central_distances_um))
    expected_distances = list(expected_distances) + [np.nan] * (max_length - len(expected_distances))
    contrast = list(contrast) + [np.nan] * (max_length - len(contrast))

    # Create DataFrame
    df = pd.DataFrame({
        'Observed Distance (µm)': central_distances_um,
        'Expected Distance (µm)': expected_distances,
        'Contrast': contrast
    })

    return df


def grab_points(distance_valid, contrast_valid, n:int, edge_buffer:int=1, debug=False):
    # Initialize variables to track the best fit
    best_r_squared = -np.inf
    best_slope = None
    best_intercept = None
    best_x_linear = []
    best_y_linear = []
    
    # Avoid edges value by using a buffer
    for start_ in range(0, len(distance_valid) - n + 1):
        for end_ in range(start_ + n, len(distance_valid) + 1 - edge_buffer):
            x_linear = distance_valid[start_:end_]
            y_linear = contrast_valid[start_:end_]

            # Fit a straight line to the selected linear region
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_linear, y_linear)
            
            # Calculate R² value
            r_squared = r_value**2
            
            # Update best fit if current R² is higher
            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_slope = slope
                best_intercept = intercept
                best_x_linear = x_linear
                best_y_linear = y_linear
                
    if debug:
        x_fit_line = np.linspace(min(distance_valid), max(distance_valid), 100)
        if best_slope: y_fit_line = best_slope * x_fit_line + best_intercept
        
        plt.figure(figsize=(10, 6))
        plt.scatter(distance_valid, contrast_valid, label='All Points')
        plt.scatter(best_x_linear, best_y_linear, s=10, c='red', label='Best Fit Points')
        if best_slope: plt.plot(x_fit_line, y_fit_line, color='green', linestyle='--', label='Linear Fit')
        plt.xlabel('Distance')
        plt.ylabel('Contrast')
        plt.title('Best Linear Fit with Edge Buffer')
        plt.legend()
        plt.show()
    
    return best_r_squared, best_slope, best_intercept


def norm_center_img(img, edge=3, debug=False, verbose:bool=False):
    #Detect ROI
    thresh = filters.threshold_otsu(img)
    binary = img > thresh
    binary = morphology.binary_opening(binary) #required!
    
    # Find the bounding box of the ROI
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Add a small edge, taking care of the image dimension as well!
    rmin = max(0, rmin-edge)
    rmax = min(rmax+edge, img.shape[0])
    cmin = max(0, cmin-edge)
    cmax = min(cmax+edge, img.shape[1])
    # Crop the image to the ROI
    img_roi = img[rmin:rmax+1, cmin:cmax+1]
    img_roi = img_roi.astype(np.float32) #required
    
    #substract and clip the background value from the img_roi
    img_array = img.astype(np.float32)
    img_array[rmin:rmax+1, cmin:cmax+1] = np.nan
    bck = np.nanmean(img_array)
    if verbose: logger.info(f"Background of the image at: {bck}")

    if not np.isnan(bck):
        img_roi = np.clip(img_roi - bck, 0, None)
        
    #normalize the image 0-1
    img_roi = (img_roi - np.min(img_roi))/(np.max(img_roi) - np.min(img_roi))
    
    if debug:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title('Original image')
        ax2.imshow(img_roi)
        ax2.axis('off')
        ax2.set_title('Cropped and normalized image')
        
        plt.tight_layout()
        plt.show()
    
    return img_roi



def get_resolution(metadata, img_array, pxl_size, verbose:bool=False, visu=False, debug=False):
    save_path = metadata['filepath']+'_Lateral_Resolution.jpg'
    #preprocess image
    img_roi = norm_center_img(img_array)
    
    #Rotate the image if needed
    orientation = detect_orientation(img_roi, verbose=verbose)
    if orientation != 0:
        img_roi = transform.rotate(img_roi, -orientation)  
    final_angle = img_best_rotation(img_roi)
    img_roi = transform.rotate(img_roi, final_angle, preserve_range=True)
    
    if debug: logger.debug(f"Image shape after rotation: {img_roi.shape}")
    #Line profile only on the central part of the image
    portion = 0.25 #ratio of the image, 1 mean the full : can be noisy if not perpectly aligned!
    height, width = img_roi.shape
    start = int(width // 2 - portion/2 * width)
    end = int(width // 2 + portion/2 * width)
    line_profile = np.mean(img_roi[:, start:end], axis=1)  
    
    #fit gaussian on each peak (and group them in set)
    results, main_valleys = analyze_gradual_lines(line_profile, pxl_size, debug=debug)
    
    #extract the distance between two lines:
    results_table = create_results_table(results, pxl_size)
    
    contrast_values = results_table['Contrast'].values
    distance = results_table['Observed Distance (µm)'].values
    
    valid_indices = ~np.isnan(contrast_values)
    distance_valid = distance[valid_indices]
    contrast_values_valid = contrast_values[valid_indices]

    if len(distance_valid) < 4: #no fitting possible!
        if verbose: logger.info(f"Was not able to fit a linear curve on available points. Need at least 4 points, detect {len(distance_valid)}. "+
                         "Either decrease the portion of the image, adjust the angle or reduce the noise")
        
        if visu:
            # fallback visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
            ax1.imshow(img_roi, cmap='gray')
            rect = plt.Rectangle((start, 0), end-start, height, fill=False, edgecolor='blue', linewidth=2)
            ax1.add_patch(rect)
            ax1.axis('off')
            ax1.set_title("Image / Region")
    
            ax2.plot(np.arange(len(line_profile)) * pxl_size, line_profile)
            ax2.set_title("Line profile (failed to extract resolution)")
            ax2.set_xlabel("µm")
            ax2.set_ylabel("Intensity")
    
            fig.suptitle("⚠️ Resolution analysis failed", color='red')
    
            plt.tight_layout()
            plt.savefig(save_path, dpi=300,
                        format="jpeg",
                        pil_kwargs={'quality': 95, 'optimize': True})
            plt.close()
        else:
            save_path = ""
            
        return 99999, results_table, save_path
          
    # Initialize variables to track the best fit
    best_r_squared, best_slope, best_intercept = grab_points(distance_valid,
                                                             contrast_values_valid, 4,
                                                             debug=debug)
    if best_r_squared < 0.95:
        if verbose: logger.info(f"Low R^2 with 4 points ({best_r_squared}). Shift to 3 points required")
        best_r_squared, best_slope, best_intercept = grab_points(distance_valid, contrast_values_valid,
                                                                 3, edge_buffer=2, debug=debug)
        if best_r_squared < 0.8: #big fail!
            _, best_slope, best_intercept = grab_points(distance_valid, contrast_values_valid,
                                                        3, edge_buffer=1, debug=debug)
    

    # Fit a higher-degree polynomial
    degree = 2
    coefficients = np.polyfit(distance_valid, contrast_values_valid, degree)
    polynomial = np.poly1d(coefficients)
    
    # Define the polynomial and linear functions
    def poly_func(x):
        return polynomial(x)
    
    def line_func(x):
        return best_slope * x + best_intercept
    
    # Define a function to find intersections
    def find_intersections():
        # Create a new function that represents the difference between the two functions
        def equation(x):
            return poly_func(x) - line_func(x)
        
        # Use fsolve to find roots of the equation (where it equals zero)
        initial_guesses = np.linspace(min(distance_valid), max(distance_valid), num=100)  # Generate initial guesses
        intersections = []
        
        for guess in initial_guesses:
            try:
                root = optimize.fsolve(equation, guess)[0]
                # Check if root is within the valid range and not already found
                if min(distance_valid) <= root <= max(distance_valid) and root not in intersections:
                    intersections.append(root)
            except:
                continue
        
        return intersections
    
    # Find intersection points
    intersection_points = find_intersections()
    
    # Round and filter intersection points
    tolerance = 1e-3  # Define a tolerance for filtering close values
    rounded_intersections = np.round(intersection_points, decimals=3)  # Round to three decimal places
    
    # Filter unique values based on tolerance
    unique_intersections = []
    unique_contrast = []
    for point in rounded_intersections:
        if not unique_intersections or abs(point - unique_intersections[-1]) > tolerance:
            unique_intersections.append(point)
            unique_contrast.append(poly_func(point))

    #visualization
    if visu:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        # Visualize the image (rotated and adjusted)
        ax1.imshow(img_roi, cmap='gray')
        # Draw the measuring bounding box
        rect = plt.Rectangle((start, 0), end-start, height, fill=False, edgecolor='blue', linewidth=2)
        ax1.add_patch(rect)
        ax1.axis('off')
        ax1.set_title("Measured region for the line plot")
        
        #plot the line
        line_profile_um = np.arange(0, len(line_profile)) * pxl_size   
        ax2.plot(line_profile_um, line_profile, label='Line Profile')
        ax2.vlines(line_profile_um[main_valleys], ymin=min(line_profile), ymax=max(line_profile), color='r', linestyle='--', label='Set Boundaries')
        
        # Add central distance labels
        for i, (_, _, start, central_distance, _) in enumerate(results):
            if i < len(main_valleys) - 1:
                x_center = (line_profile_um[main_valleys[i]] + line_profile_um[main_valleys[i+1]]) / 2
                if central_distance != 'n/a':
                    label = f"{central_distance * pxl_size:.3f} µm"
                else:
                    label = "n/a"
                ax2.text(x_center, min(line_profile), label, ha='center', va='bottom', fontsize=8, rotation=90)
        
        ax2.set_ylabel("Intensity (a.u.)")
        ax2.set_xlabel("µm")
        ax2.set_title("Line profile")
        ax2.legend()
        
        ax3.scatter(distance_valid, contrast_values_valid, color='blue', label='Measured Contrast')
        x_fit_line = np.linspace(min(distance_valid), max(distance_valid), 100)
        y_fit_line = best_slope * x_fit_line + best_intercept
        ax3.plot(x_fit_line, y_fit_line, color='green', linestyle='--', label='Linear Fit')
        ax3.plot(np.linspace(min(distance_valid), max(distance_valid), 100), polynomial(np.linspace(min(distance_valid), max(distance_valid), 100)),
                 color='red', label=f'{degree}th Degree Polynomial Fit')
        
        # Plot unique intersection points and annotate them
        ax3.plot(unique_intersections, poly_func(unique_intersections), 'go', label='Intersection Point')
        for point in unique_intersections:
            contrast_value = poly_func(point)
            ax3.text(point - 0.01, contrast_value - 0.01,
                     f'({point:.3f}, {contrast_value:.4f})', fontsize=9)
        
        ax3.set_xlabel('Line Spacing (µm)')
        ax3.set_ylabel('Measured Contrast')
        ax3.set_title('Contrast Transfer Function with Intersection Points')
        ax3.legend()
        ax3.grid()
        
        plt.tight_layout()

        plt.savefig(metadata['filepath']+'_Lateral_Resolution.jpg', dpi=300,                    
                            format = "jpeg",         
                            pil_kwargs={'quality': 95, 'optimize': True},
                            )
        plt.close()

    #Resolution should be at the lowest contrast (or given contrast)
    if len(unique_contrast) != 0:
        res = unique_intersections[unique_contrast.index(min(unique_contrast))]
    else:
        return 99999, results_table, save_path

    return res, results_table, save_path



def find_best_condition(counts, rmses, mapes, res):
    # Normalize the parameters
    norm_counts = np.array(counts) / np.max(counts)
    norm_rmses = 1 - (np.array(rmses) / np.max(rmses))  # Invert so lower is better
    norm_mapes = 1 - (np.array(mapes) / np.max(mapes))  # Invert so lower is better
    norm_res = 1 - (np.array(res) / np.max(res)) # Invert so lower is better

    # Create a composite score
    composite_scores = (norm_counts*3 + norm_rmses/2 + norm_mapes/2 + norm_res) / 4

    # Find the index of the best score
    best_index = np.argmax(composite_scores)

    return best_index, composite_scores


def resolution_analysis(img_array:np.array, metadata:dict):
    res = []
    counts = []
    mapes = []
    rmses = []
    pxl_size_xy = float(metadata['Physical pixel size X'])
        
    for z in range(img_array.shape[0]):
        resolution, df, _ = get_resolution(metadata, img_array[z], pxl_size_xy)
        res.append(resolution)
        if resolution == 99999:
            counts.append(0)
            rmses.append(999)
            mapes.append(999)
        else:
            counts.append(df['Contrast'].count())
            df['relative_error'] = abs(df['Observed Distance (µm)'] - df['Expected Distance (µm)']) / df['Expected Distance (µm)'] * 100
            mapes.append(df['relative_error'].mean())
            rmses.append(np.sqrt(((df['Observed Distance (µm)'] - df['Expected Distance (µm)']) ** 2).mean()))
    
    #criterion for the best plane is the most accurate detection of the lines
    best_index, scores = find_best_condition(counts, rmses, mapes, res)

    resolution, _, fig_path = get_resolution(metadata, img_array[best_index], pxl_size_xy, visu=True)
    logger.info(f"Resolution of the image is: {resolution} microns at plane {best_index}")

    #save the resolution in a database
    db.process_measurement(metadata, True, 'Argolight resolution', metadata['Image_ID'], 
                           'Resolution', str(resolution), 'micron')

    return fig_path

if __name__ == '__main__':
    logger.setup_logger()

    path = r"D:\DATA\Simon\CCI\Metrology\test"
    filename = '40x_O_resolution.czi'
    img_array, metadata = reader(path+os.sep+filename)
    resolution_analysis(img_array[0], metadata)