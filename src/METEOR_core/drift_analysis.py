"""
Created on Fri Jun 13 10:48:12 2025

@author: simon

Deprecated, too time consuming to acquire these data for too little advantages.

"""
#general import
import numpy as np
import os

#analysis import
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from scipy import ndimage, stats
from skimage import filters, measure, morphology
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial import KDTree

#local import
from . import logger
from .functions import reader, sort_grid, norm_center_img_3D



def analyze_z_resolution(img_array, pxl_size_xy, pxl_size_z):
    sigma = np.array((1,2,2))
    blurred = ndimage.gaussian_filter(img_array, sigma*2)
    clean = img_array - blurred
    clean = np.clip(clean, 0, 1)
    
    thresh = filters.threshold_otsu(clean) 
    binary = clean > thresh
    binary = ndimage.binary_fill_holes(binary)
    binary = ndimage.binary_closing(binary, iterations=1)
    binary = ndimage.binary_opening(binary, iterations=1)
    labeled, num_features = ndimage.label(binary)

    z_centers = []
    fwhm_values = []
    center_global_coo = []

    for region in measure.regionprops(labeled):
        if region.area < 50: #skip too small body
            continue
        
        # Get the bounding box of the current region
        minz, minr, minc, maxz, maxr, maxc = region.bbox
        
        # Extract the subvolume corresponding to the bounding box
        sub_volume = binary[minz:maxz, minr:maxr, minc:maxc]
        
        # Compute the distance transform on the sub-volume
        dist_transform = ndimage.distance_transform_edt(sub_volume)

        # Find the center of the ring (maximum of distance transform)
        center = np.array(np.where(dist_transform == np.max(dist_transform)))[:,0]
        center_global_coo.append((center[0]+minz, center[1]+minr, center[2]+minc))
        # Extract Z profile
        z_profile = np.mean(dist_transform, axis=(1,2))
        
        # Calculate FWHM directly from the distance transform profile
        max_dist = np.max(z_profile)
        half_max = max_dist / 2
        above_half_max = z_profile >= half_max
        fwhm = np.sum(above_half_max) * pxl_size_z

        z_centers.append((minz + center[0]) * pxl_size_z)  # Adjusting for bbox offset
        fwhm_values.append(fwhm)

    logger.info(f"Extract {len(z_centers)} rings from the images, corresponding in theory to {len(z_centers)/9/9:.2f} levels")

    return z_centers, fwhm_values, np.array(center_global_coo)

from collections import Counter

def save_drift_analysis(metadata, df, points, drift_vectors, img_roi,
                        visu:bool=True, debug:bool=False):
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['Z_um'], df['z_fwhm'])

    #normalize point quantity and calculate drifts
    lengths = [arr.shape[1] for arr in points]
    common_length = Counter(lengths).most_common(1)[0][0]  # Most frequent length
    points_clean = []
    avg_drifts = []
    for arr in points:
        if arr.shape[1] < common_length: # Pad with NaN along the second dimension
            pad_width = ((0, 0), (0, common_length - arr.shape[1]))
            arr = np.pad(arr, pad_width, constant_values=np.nan)
        elif arr.shape[1] > common_length: # Clip excess values
            arr = arr[:, :common_length]
            
        ref_x, ref_y = arr[0, 0], arr[1, 0]  # Reference point (first Z)
        drifts = np.sqrt((arr[0] - ref_x)**2 + (arr[1] - ref_y)**2)
        avg_drifts.append(drifts)
        points_clean.append(arr)

    if debug: logger.debug(f'Length of the average drift list is {len(avg_drifts)}')
    z_drift = np.nanmean(np.nanmax(avg_drifts, axis=1))
    
    df = df.dropna(subset=['intensity'])
    
    #save the differents stats in a database!
    db.process_measurement(metadata, True, 'Drift Z', metadata['Image_ID'], 
                           'Z resolution', str(df.z_fwhm.mean()), 'micron')
    db.process_measurement(metadata, True,  'Drift Z', metadata['Image_ID'], 
                           'Z resolution - std', str(df.z_fwhm.std()), 'micron')
    # db.process_measurement(metadata, True, 'Drift Z', metadata['Image_ID'], 
    #                        'Z decay', str(slope), '')
    # db.process_measurement(metadata, True, 'Drift Z', metadata['Image_ID'], 
    #                        'Average Z drift %', str(z_drift/img_roi.shape[1]*100), '%')
    db.process_measurement(metadata, True,  'Drift Z', metadata['Image_ID'], 
                           'Z drift', str(z_drift*float(metadata['Physical pixel size X'])), 'micron')
        
    if visu:
        fig, (ax2, ax1, ax3) = plt.subplots(1, 3, figsize=(12, 5))
    
        #1st: image image XY projection.
        ax2.imshow(np.max(img_roi, axis=0), origin='lower')
        ax2.scatter(df['center_x'], df['center_y'], c='r', s=1, alpha=0.5)
        ax2.set_title('Projection of 3D rings')
        ax2.axis('off')
    
        # 2nd: graph, Decay of the Z resolution accross the Z volume
        ax1.scatter(df['Z_um'], df['z_fwhm'], alpha=0.7)
        # Create a line using the slope and intercept
        line = slope * df['Z_um'] + intercept
        ax1.plot(df['Z_um'], line, '--', color='r', label=f'Slope: {slope:.4f}')
        ax1.set_xlabel('Z position (µm)')
        ax1.set_ylabel('Z-FWHM (µm)')
        ax1.set_title(f"Decay of Z-resolution across Z-scan\nZ resolution {df.z_fwhm.mean():.2f}({df.z_fwhm.std():.2f}) µm  Decay: {slope:.4f}")
        # Add text annotations for each group
        grouped = df.groupby('labels').agg({
            'Z_um': 'mean', 'z_fwhm': 'mean'}).reset_index()
        for _, row in grouped.iterrows():
            ax1.annotate(f"Z: {row['z_fwhm']:.2f}\n",
                         (row['Z_um'], row['z_fwhm']),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=8, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
        # 3rd: graph, heatmap of the ring showing the average maximum drift across Z
        inac = .05
        vmax = ((img_roi.shape[1] + img_roi.shape[2]) // 2) * inac
        norm = Normalize(vmin=0, vmax=vmax)
    
        avg_drifts_clean = [np.nanmax(x) for x in avg_drifts] #max drift
    
        grid_size = 9
        points_coo = np.nanmean(points_clean, axis=2)
        points_sorted, idx_sorted = sort_grid(points_coo, gridsize=grid_size)
        # drift_vectors = np.array(drift_vectors)[idx_sorted]
        X = points_sorted[:, 0].reshape(grid_size, grid_size)
        Y = points_sorted[:, 1].reshape(grid_size, grid_size)
        dx = (X.max() - X.min()) / (grid_size - 1)
        dy = (Y.max() - Y.min()) / (grid_size - 1)
        
        avg_drifts_clean = np.array(avg_drifts_clean)[idx_sorted]
        final_map = np.array(avg_drifts_clean).reshape(grid_size, grid_size)
        final_map = np.rot90(final_map, k=1)
        
        cmap = plt.get_cmap('jet').copy()
        cmap.set_extremes(under='yellow', over='red')
        im = ax3.imshow(final_map, extent=[X.min(), X.max(), Y.min(), Y.max()],
                        cmap=cmap, norm=norm, #origin='lower'
                        )
        plt.colorbar(im, label='Drift (pxl)', ax=ax3)
        ax3.set_title(f"Drift across Z\nAverage drift: {z_drift/img_roi.shape[1]*100:.2f} % ({z_drift:.2f} pxl)")
        # Add the drift vectors using quiver
        ax3.quiver(points_coo[:, 0]/np.nanmax(points_coo[:, 0])*(img_roi.shape[1]-dx)+dx/2,
                   points_coo[:, 1]/np.nanmax(points_coo[:, 1])*(img_roi.shape[2]-dy)+dy/2, 
                   drift_vectors[:, 0], drift_vectors[:, 1],
                   scale=100, width=0.01, color='black', alpha=0.7, pivot='middle',)
        
        ax3.axis('off')
        
        # Show the plot
        plt.tight_layout()

        plt.savefig(metadata['filepath']+'_Z_analysis.jpg', dpi=300,
                    format = "jpeg",         
                    pil_kwargs={'quality': 95, 'optimize': True},
                    )
                            
        plt.close(fig)
        
        return metadata['filepath']+'_Z_analysis.jpg'
    


def measure_xy_fwhm(img_slice, pxl_size_xy):
    thresh = filters.threshold_otsu(img_slice)
    binary = img_slice > thresh
    filled = morphology.remove_small_holes(binary)    
    labels = measure.label(filled)
    
    fwhm_values = []
    centers = []
    intensities = []
    for region in measure.regionprops(labels, intensity_image=img_slice):
        if region.area < 20: #skip too small body
            continue
        
        y, x = region.centroid
        minr, minc, maxr, maxc = region.bbox
        mask = labels[minr:maxr, minc:maxc] == region.label
        dist = ndimage.distance_transform_edt(mask)
        fwhm = 2 * np.max(dist) * pxl_size_xy
        fwhm_values.append(fwhm)
        centers.append((x, y))
        intensities.append(region.intensity_mean)
    
    return centers, fwhm_values, intensities


def analyze_xy_fwhm(df, img_roi, pxl_size_xy):
    # Build KDTree for fast coordinate lookup
    group_centers = list(np.ceil(df.groupby('labels')['Z'].mean().sort_values()).astype(int))
    df_coords = df[['X', 'Y', 'Z']].values
    tree = KDTree(df_coords)
    
    for z_center in group_centers:
        z_plane = int(np.ceil(z_center))
        xy_plane = np.max(img_roi[z_plane-2:z_plane+2], axis=0)
        centers, fwhm_values, intensities = measure_xy_fwhm(xy_plane, pxl_size_xy)

        centers = np.array(centers)  # Convert to numpy array
        
        query_points = np.c_[centers[:, 0], centers[:, 1], np.full(len(centers), z_plane)]
        
        # Find nearest matches efficiently
        distances, indices = tree.query(query_points, k=1)
        
        df.loc[indices, ['center_x', 'center_y', 'xy_fwhm', 'intensity']] = \
            np.column_stack((centers[:, 0], centers[:, 1], fwhm_values, intensities))

    return df


def calculate_drift_statistics(df):
    group_centers = np.ceil(df.groupby('labels')['Z'].mean().sort_values()).astype(int)
    sub_df = df[df.labels == group_centers.index[0]] #start at 0
    points, drift_vectors = [], []    
    for idx, row in sub_df.iterrows():
        position_x = []
        position_y = []
        position_x.append(row.center_x)
        position_y.append(row.center_y)
        
        cumulative_dx = 0
        cumulative_dy = 0
        
        counter = 9 #cannot have more than 9 floors
        while True:
            counter -= 1
            if np.isnan(row.link):  # End of the linkage
                break
            
            # Retrieve the next row using 'ID' instead of integer-based indexing
            next_row = df.loc[df['ID'] == row.link]
        
            if next_row.empty:
                break  # If no match found, exit loop
            
            next_row = next_row.squeeze()  # Convert DataFrame to Series
        
            position_x.append(next_row.center_x)
            position_y.append(next_row.center_y)
        
            # Cumulative displacement
            cumulative_dx += (next_row.center_x - row.center_x)
            cumulative_dy += (next_row.center_y - row.center_y)
        
            row = next_row  # Move to the next row
        
            if counter == 0:  # Security stop
                break
        
        position_x = np.array(position_x)
        position_y = np.array(position_y)
        points.append(np.array([position_x, position_y]))
        drift_vectors.append(np.array([cumulative_dx, cumulative_dy]))

    # Calculate the average drift vector
    drift_vectors = np.array(drift_vectors)
    return points, drift_vectors


def analyze_xy_drift(df):
    group_centers = np.ceil(df.groupby('labels')['Z'].mean().sort_values()).astype(int)
    link = pd.Series(np.nan, index=df['ID'])  # Use a pandas Series indexed by 'ID'

    for i in range(len(group_centers) - 1):
        sub_df = df[df.labels == group_centers.index[i]]
        next_df = df[df.labels == group_centers.index[i + 1]]

        if next_df.empty or sub_df.empty:
            continue  # Skip if no valid data

        # Build KDTree for fast nearest-neighbor lookups
        tree = KDTree(next_df[['X', 'Y']].values)

        # Query nearest neighbors in one step
        distances, indices = tree.query(sub_df[['X', 'Y']].values, k=1)

        # Assign matches using 'ID' column instead of index
        sub_df_ids = sub_df['ID'].values  # Extract IDs
        next_df_ids = next_df.iloc[indices]['ID'].values  # Get matched IDs
        
        link[sub_df_ids] = next_df_ids  # Map IDs directly

    df['link'] = link.values  # Convert back to NumPy format if needed
    return df

def analyze_xy_drift_1(df):
    group_centers = np.ceil(df.groupby('labels')['Z'].mean().sort_values()).astype(int)
    link = np.full(len(df), np.nan)

    for i in range(len(group_centers) - 1):
        sub_df = df[df.labels == group_centers.index[i]]
        next_df = df[df.labels == group_centers.index[i + 1]]

        if next_df.empty or sub_df.empty:
            continue  # Skip if no valid data

        # Step 1: Build KDTree for fast nearest-neighbor lookups
        tree = KDTree(next_df[['X', 'Y']].values)

        # Step 2: Query nearest neighbors in one step
        distances, indices = tree.query(sub_df[['X', 'Y']].values, k=1)

        # Step 3: Assign matches where valid
        for idx, (distance, neighbor_idx) in enumerate(zip(distances, indices)):
            if np.isnan(link[sub_df.index[idx]]):  # Check if not already linked
                link[sub_df.index[idx]] = next_df.index[neighbor_idx]  

    df['link'] = link
    return df


def create_dataframe(z_centers, positions, z_fwhm_values):
    df = pd.DataFrame({
        'Z': positions[:, 0],
        'X': positions[:, 1],
        'Y': positions[:, 2],
        'Z_um': z_centers,
        'z_fwhm': z_fwhm_values,
        'ID':range(len(z_centers)),
    })
    return df


def perform_clustering(df):
    n_groups = int(np.ceil((df.Z_um.max() - df.Z_um.min()) / 5))
    kmeans = KMeans(n_clusters=n_groups, random_state=42)
    df['labels'] = kmeans.fit_predict(df.Z.values.reshape(-1, 1))
    return df


def drift_analysis(img_array, metadata):
    pxl_size_xy = float(metadata['Physical pixel size X'])
    pxl_size_z = float(metadata['Physical pixel size Z'])
    
    img_roi = norm_center_img_3D(img_array)
    z_centers, z_fwhm_values, positions = analyze_z_resolution(img_roi, pxl_size_xy, pxl_size_z)
    df = create_dataframe(z_centers, positions, z_fwhm_values)
    df = perform_clustering(df)
    df = analyze_xy_fwhm(df, img_roi, pxl_size_xy)
    df = df.dropna()
    df = analyze_xy_drift(df)
    points, drift_vectors = calculate_drift_statistics(df)
    fig_path = save_drift_analysis(metadata, df, points, drift_vectors, img_roi)    
    
    return fig_path


if __name__ == '__main__':
    logger.setup_logger()

    path = r"D:\DATA\Simon\CCI\Metrology\test"
    filename = '63x_LSM-_3D_rings.czi'
    img_array, metadata = reader(path+os.sep+filename)
    drift_analysis(img_array[0], metadata)