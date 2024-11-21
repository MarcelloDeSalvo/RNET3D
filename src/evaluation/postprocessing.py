import numpy as np
from skimage import morphology
from skimage.measure import label, regionprops
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def morphology_refinement_callback(fill_small_holes=True, remove_small_objects=True, holes_max_area=20000, object_min_area=30000):
    '''
    Remove unconnected regions from the prediction mask, so that the final mask is a single connected component.
    Fill small holes and remove small objects from the mask.

    Parameters:
    - fill_small_holes (bool): fill small holes in the mask
    - remove_small_objects (bool): remove small objects from the mask
    - holes_max_area (int): maximum area of holes to fill in voxels
    - object_min_area (int): minimum area of objects to keep in voxels

    Returns:
    - post_process (function): post-processing function that refines the mask
    '''
    def post_process(y_pred):

        if not remove_small_objects and not fill_small_holes: return y_pred

        # convert to bool
        if fill_small_holes: y_pred = morphology.remove_small_holes(y_pred.astype(bool), holes_max_area)
        if remove_small_objects: y_pred = morphology.remove_small_objects(y_pred.astype(bool), object_min_area)

        # convert back to float
        return y_pred.astype(np.float64)
    
    return post_process

def ipsi_contra_division_callback(visualize_pca=False, use_centroids=False, default_hemisphere='right', lesion_voxels_threshold=10):
    '''
    Divide the mask into ipsilateral and contralateral regions by using pca to determine the third ventricle's principal axes and minimize the symmetry cost.
    
    Expected input labels:
    - 0: background
    - 1: lesion
    - 2: ventricles
    - 3: third ventricle

    The output mask will be divided into 5 regions:
    - 0: background
    - 1: lesion
    - 2: contralateral ventricles
    - 3: ipsilateral ventricles
    - 4: third ventricle

    Parameters:
    - visualize_pca (bool): visualize the principal components of the third ventricle
    - use_centroids (bool): if True, use the centroids of the ventricle regions to determine the hemisphere, otherwise use a hard split based on the third ventricle's plane of symmetry (useful for rats brain or when the centroids are not reliable)
    - default_hemisphere (str): default hemisphere to use when no lesion is found (either 'right' or 'left')
    - lesion_voxels_threshold (int): minimum number of voxels for a lesion to be considered valid
    
    Returns:
    - post_process (function): post-processing function that divides the mask into ipsilateral and contralateral regions
    '''
    def extract_masks(y_roi):
        lesion_mask = np.where(y_roi == 1, 1, 0).astype(np.uint8)
        third_ventricle_mask = np.where(y_roi == 3, 1, 0).astype(np.uint8)
        ventricles_mask = np.where(y_roi == 2, 1, 0).astype(np.uint8)
        return lesion_mask, ventricles_mask, third_ventricle_mask

    def get_regions(mask):
        labeled_mask = label(mask)
        return regionprops(labeled_mask), labeled_mask

    def perform_pca(coords):
        pca = PCA(n_components=3)
        pca.fit(coords)
        return pca.components_, pca.mean_

    def symmetry_cost(normal, coords, center_of_mass):
        plane_normal = np.array(normal)
        plane_normal /= np.linalg.norm(plane_normal)
        distances = np.dot(coords - center_of_mass, plane_normal)
        half1 = coords[distances >= 0]
        half2 = coords[distances < 0]
        return np.abs(len(half1) - len(half2))

    def determine_lesion_hemisphere(lesion_centroid, plane_point, normal):
        vector = lesion_centroid - plane_point
        return np.dot(vector, normal) > 0

    def create_ventricle_masks_with_centroids(labeled_ventricles, regions, hemisphere_indices):
        ipsi_ventricle_mask = np.zeros_like(labeled_ventricles)
        contra_ventricle_mask = np.zeros_like(labeled_ventricles)
        for i, region in enumerate(regions):
            if hemisphere_indices[i]:
                ipsi_ventricle_mask[labeled_ventricles == region.label] = 1
            else:
                contra_ventricle_mask[labeled_ventricles == region.label] = 1
        return ipsi_ventricle_mask, contra_ventricle_mask
    
    def create_ventricle_masks_with_hard_split(ventricles_mask, optimal_normal, lesion_in_right_hemisphere, plane_point):
        ventricle_coords = np.argwhere(ventricles_mask > 0)
        ipsi_ventricle_mask = np.zeros_like(ventricles_mask)
        contra_ventricle_mask = np.zeros_like(ventricles_mask)

        # Assign each ventricle voxel to the left or right hemisphere based on the third ventricle plane of symmetry
        for coord in ventricle_coords:
            in_right_hemisphere = determine_lesion_hemisphere(coord, plane_point, optimal_normal)

            if in_right_hemisphere == lesion_in_right_hemisphere:
                ipsi_ventricle_mask[coord[0], coord[1], coord[2]] = 1
            else:
                contra_ventricle_mask[coord[0], coord[1], coord[2]] = 1

        return ipsi_ventricle_mask, contra_ventricle_mask

    def update_final_roi_mask(y_roi, ipsi_ventricle_mask, contra_ventricle_mask):
        final_roi_mask = np.copy(y_roi)
        final_roi_mask[final_roi_mask == 3] = 4
        final_roi_mask[contra_ventricle_mask == 1] = 2
        final_roi_mask[ipsi_ventricle_mask == 1] = 3
        return final_roi_mask

    def visualize_and_save_pca(coords, principal_axes, plane_point):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot a subset of coordinates to avoid clutter
        if len(coords) > 10000:
            indices = np.random.choice(len(coords), size=20000, replace=False)
            sampled_coords = coords[indices]
        else:
            sampled_coords = coords
        
        ax.scatter(sampled_coords[:, 0], sampled_coords[:, 1], sampled_coords[:, 2], alpha=0.05, c='blue', s=1)

        # Scale factor for the principal components' lines
        scale_factor = 50

        # Plot principal components as lines
        for i in range(3):
            component = principal_axes[i]
            line_x = [plane_point[0] - scale_factor * component[0], plane_point[0] + scale_factor * component[0]]
            line_y = [plane_point[1] - scale_factor * component[1], plane_point[1] + scale_factor * component[1]]
            line_z = [plane_point[2] - scale_factor * component[2], plane_point[2] + scale_factor * component[2]]
            ax.plot(line_x, line_y, line_z, label=f'Principal Component {i+1}', linewidth=2)

        ax.legend()
        plt.title('Principal Components of the third ventricle')
        plt.show()
        plt.close()

    def post_process(y_roi):
        lesion_mask, ventricles_mask, third_ventricle_mask = extract_masks(y_roi)
        
        lesion_regions, _ = get_regions(lesion_mask)
        ventricle_regions, labeled_ventricles = get_regions(ventricles_mask)
        third_ventricle_regions, _ = get_regions(third_ventricle_mask)

        third_ventricle_region = max(third_ventricle_regions, key=lambda x: x.area)
        # Extract the third ventricle coordinates and centroid
        third_ventricle_coords = np.argwhere(third_ventricle_mask > 0)
        third_ventricle_centroid = third_ventricle_region.centroid
        third_ventricle_centroid = np.array([third_ventricle_centroid[0], third_ventricle_centroid[1], third_ventricle_centroid[2]])

        principal_axes, _ = perform_pca(third_ventricle_coords)
        
        # Visualize and save the PCA components
        if visualize_pca: visualize_and_save_pca(third_ventricle_coords, principal_axes, third_ventricle_centroid)

        # Extract the third ventricle plane of symmetry
        initial_normal = principal_axes[2]
        
        # Optimize the normal vector to minimize the symmetry cost
        result = minimize(symmetry_cost, initial_normal, args=(third_ventricle_coords, third_ventricle_centroid), method='Nelder-Mead')
        optimal_normal = result.x
        optimal_normal /= np.linalg.norm(optimal_normal)
        
        # Take biggest lesion blob and its centroid
        try:
            lesion_region = max(lesion_regions, key=lambda x: x.area)
            lesion_centroid = lesion_region.centroid
            
            # if lesion is too small, raise ValueError
            if lesion_region.area < lesion_voxels_threshold:
                raise ValueError('Lesion is too small')

            # Determine the lesion hemisphere
            lesion_in_right_hemisphere = determine_lesion_hemisphere(lesion_centroid, third_ventricle_centroid, optimal_normal)
            
            if use_centroids:
                hemisphere_indices = [determine_lesion_hemisphere(region.centroid, third_ventricle_centroid, optimal_normal) == lesion_in_right_hemisphere for region in ventricle_regions]
                ipsi_ventricle_mask, contra_ventricle_mask = create_ventricle_masks_with_centroids(labeled_ventricles, ventricle_regions, hemisphere_indices)
            else:
                ipsi_ventricle_mask, contra_ventricle_mask = create_ventricle_masks_with_hard_split(ventricles_mask, optimal_normal, lesion_in_right_hemisphere, third_ventricle_centroid)
        
        except ValueError:
            # If no lesion is found, default to the right hemisphere
            print(f'No lesion found, defaulting to the {default_hemisphere} hemisphere with hard split')
            ipsi_ventricle_mask, contra_ventricle_mask = create_ventricle_masks_with_hard_split(ventricles_mask, optimal_normal, default_hemisphere == 'right', third_ventricle_centroid)

        final_roi_mask = update_final_roi_mask(y_roi, ipsi_ventricle_mask, contra_ventricle_mask)
        
        return final_roi_mask
    
    return post_process