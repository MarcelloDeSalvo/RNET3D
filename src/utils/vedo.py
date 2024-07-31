import vedo
vedo.settings.default_backend= 'vtkplotter'
from vedo import *
from vedo.applications import Slicer3DPlotter

def plot_slicer_cloud(nii_img, nii_mask):
    '''
    Plot a 3D slicer image and a point cloud of the lesion mask in the same plot.
    '''

    v2 = Volume(nii_img.get_fdata())

    v2.cmap('bone')

    plt = Slicer3DPlotter(
        v2,
        cmaps=("bone", "bone", "bone"),
        use_slider3d=True,
        bg="white",
        bg2="blue9",
    )

    # Define class labels
    class_labels = [1, 2, 3]
    color_map = ['red', 'green', 'blue']

    # Create an empty dictionary to store point clouds
    point_clouds = {}

    i=0
    for label in class_labels:
        # Get voxel coordinates
        voxel_coords = np.array(np.where(nii_mask.get_fdata() == label)).T
        pts = Points(voxel_coords, r=2, c=color_map[i], alpha=0.5)

        # Store the point cloud in the dictionary
        point_clouds[label] = pts
        i+=1


    for label in class_labels:
        plt += point_clouds[label]

    # Add a slider to the plotter that adjusts the opacity of the point clouds
    def update_opacity(widget, event):
        for label in class_labels:
            point_clouds[label].alpha(widget.value)

    plt.add_slider(
        update_opacity,
        xmin=0,
        xmax=1,
        value=0.5,
        pos="bottom-right-vertical",
        title="Opacity",
    )


    return plt.show(viewup='z').close()


def plot_volume_cloud(nii_img, nii_mask, spacing=[1, 1, 1], cmap='bone'):
    '''
    Plot a 3D image and a point cloud of the lesion mask in the same plot.
    '''

    plt = Plotter()

    v2 = Volume(nii_img.get_fdata())
    v2.cmap(cmap)

    # Set voxel dimension in mm
    v2.spacing(spacing)


    plt += v2

    # Define class labels
    class_labels = [1, 2, 3, 4]
    color_map = ['red', 'green', 'blue', 'yellow']

    # Create an empty dictionary to store point clouds
    point_clouds = {}

    i=0
    for label in class_labels:
        # Get voxel coordinates
        voxel_coords = np.array(np.where(nii_mask.get_fdata() == label)).T  * v2.spacing()
        pts = Points(voxel_coords, r=15, c=color_map[i], alpha=0.5)
        # Multiply by voxel dimension to get coordinates in mm

        # Store the point cloud in the dictionary
        point_clouds[label] = pts
        i+=1


    for label in class_labels:
        plt += point_clouds[label]

    plt.add_slider(
        lambda w, e: [point_clouds[label].alpha(w.value) for label in class_labels],
        xmin=0,
        xmax=1,
        value=0.5,
        pos="bottom-right-vertical",
        title="Opacity",
    )

    plt.add_slider(
        lambda w, e: v2.alpha([0, w.value]),
        xmin=0,
        xmax=1,
        value=0.5,
        pos="bottom-left-vertical",
        title="Opacity",
    )

    # Make the plot rotate automatically
    plt += Text2D("Press q to exit", pos=(0.8, 0.05), s=0.8)

    return plt.show(viewup='z').close()


def plot_two_volumes(nii_data1, nii_data2, spc1 = [1, 1, 1], spc2 = [1, 1, 1]):
    '''
    Plot two 3D images in the same plot.
    '''
    v1 = Volume(nii_data1)
    v2 = Volume(nii_data2)

    # Set voxel dimension in mm
    v1.spacing(spc1)
    v2.spacing(spc2)

    v1.cmap('bone')
    v2.cmap('bone')

    return show([v1, v2], N=2, axes=7, bg='black', bg2='blackboard', viewup='z', zoom=1.2, interactive=True).close()


