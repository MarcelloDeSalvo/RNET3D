import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact
import numpy as np
import SimpleITK as sitk
import seaborn as sns


def plot_data(data, slice_id=None):
    # Plot the images
    num_modalities = len(data)
    plt.figure(figsize=(5 * num_modalities, 5))

    if num_modalities == 0:
        print("No modalities to plot")
        return

    if slice_id is None:
        # Use the middle slice as the default slice_id by getting the first modality of the dictionary
        slice_id = data[list(data.keys())[0]].shape[2] // 2

    for i, modality in enumerate(data.keys()):
        plt.subplot(1, num_modalities, i + 1)

        if modality == "lesion_mask":
            # Use a different colormap for the mask (e.g., 'jet')
            plt.imshow(np.rot90(data[modality][:, :, slice_id], k=-1), cmap="jet")
        else:
            # Use the 'gray' colormap for other modalities
            plt.imshow(np.rot90(data[modality][:, :, slice_id], k=-1), cmap="gray")

        plt.title(modality)

    plt.show()


# extend plot data to plot a slice of a specific axis and chose the rotation
def plot_data_axis(data, slice_id=None, axis=2, rotation=0):
    """
    Plots the images in a dictionary of modalities.
    :param data: A dictionary of modalities.
    :param slice_id: The slice to plot.
    :param axis: The axis to plot.
    :param rotation: The rotation to apply to the image.
    """

    # Plot the images
    num_modalities = len(data)
    plt.figure(figsize=(5 * num_modalities, 5))

    if slice_id is None:
        # Use the middle slice as the default slice_id
        slice_id = data["lesion_mask"].shape[axis] // 2

    for i, modality in enumerate(data.keys()):
        plt.subplot(1, num_modalities, i + 1)

        if modality == "lesion_mask":
            # Use a different colormap for the mask (e.g., 'jet')
            plt.imshow(
                np.rot90(np.take(data[modality], slice_id, axis=axis), k=rotation),
                cmap="jet",
            )
        else:
            # Use the 'gray' colormap for other modalities
            plt.imshow(
                np.rot90(np.take(data[modality], slice_id, axis=axis), k=rotation),
                cmap="gray",
            )

        plt.title(modality)

    plt.show()


def plot_overlap(orig, mask, axis=2, alpha=0.5, slice_id=0, rotation=0):
    """
    Plots the overlap between the original image and the mask.
    :param orig: The original image.
    :param mask: The mask.
    :param axis: The axis to plot.
    :param alpha: The opacity of the mask.
    """
    plt.figure(figsize=(5, 5))
    plt.imshow(np.rot90(np.take(orig, slice_id, axis=axis), k=-1), cmap="gray")
    plt.imshow(
        np.rot90(np.take(mask, slice_id, axis=axis), k=-1), cmap="jet", alpha=alpha
    )
    plt.show()


# Widget to explore 3D array
def explore_3D_array(arr: np.ndarray, cmap: str = "gray", axis: int = 2):
    """
    Given a 3D array with shape (Z,X,Y) This function will create an interactive
    widget to check out all the 2D arrays with shape (X,Y) inside the 3D array.
    The purpose of this function to visual inspect the 2D arrays in the image.

    Parameters:
      arr : 3D array with shape (Z,X,Y) that represents the volume of a MRI image
      cmap : Which color map use to plot the slices in matplotlib.pyplot
    """

    def fn(SLICE):
        plt.figure(figsize=(7, 7))
        plt.imshow(np.take(arr, SLICE, axis=axis), cmap=cmap)

    interact(fn, SLICE=(0, arr.shape[axis] - 1))


# Widget to explore 3D array comparison
def explore_3D_array_comparison(
    arr_before: np.ndarray, arr_after: np.ndarray, cmap: str = "gray", axis: int = 0
):
    """
    source: https://github.com/Angeluz-07/MRI-preprocessing-techniques
    Given two 3D arrays with shape (Z,X,Y) This function will create an interactive
    widget to check out all the 2D arrays with shape (X,Y) inside the 3D arrays.
    The purpose of this function to visual compare the 2D arrays after some transformation.

    Parameters:
      arr_before : 3D array with shape (Z,X,Y) that represents the volume of a MRI image, before any transform
      arr_after : 3D array with shape (Z,X,Y) that represents the volume of a MRI image, after some transform
      cmap : Which color map use to plot the slices in matplotlib.pyplot
    """

    assert arr_after.shape == arr_before.shape

    def fn(SLICE):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, sharex="col", sharey="row", figsize=(10, 10)
        )

        ax1.set_title("Before", fontsize=15)
        ax1.imshow(np.take(arr_before, SLICE, axis=axis), cmap=cmap)

        ax2.set_title("After", fontsize=15)
        ax2.imshow(np.take(arr_after, SLICE, axis=axis), cmap=cmap)

        plt.tight_layout()

    interact(fn, SLICE=(0, arr_before.shape[axis] - 1))


# explore 3D array with overlay
def explore_3D_overlay(
    arr_before: np.ndarray,
    mask: np.ndarray,
    cmap: str = "gray",
    axis: int = 0,
    alpha: float = 0.5,
):
    assert mask.shape == arr_before.shape

    def fn(SLICE):
        plt.figure(figsize=(7, 7))
        plt.imshow(np.take(arr_before, SLICE, axis=axis), cmap=cmap)
        plt.imshow(np.take(mask, SLICE, axis=axis), cmap="jet", alpha=alpha)

        plt.tight_layout()

    interact(fn, SLICE=(0, arr_before.shape[axis] - 1))

    # explore 3D array comparison with a slider to compare original overlayed with ground truth mask


def explore_3D_predictions(
    arr_before: np.ndarray,
    ground_truth: np.ndarray,
    pred: np.ndarray,
    cmap: str = "gray",
    axis: int = 0,
    alpha: float = 0.5,
):
    assert ground_truth.shape == arr_before.shape
    assert pred.shape == arr_before.shape

    def fn(SLICE):
        # plot 3 subplots in one row (the first with the original image, the second with the ground truth mask overlayed with original, the third with the predicted mask overlayed with original)
        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, sharex="col", sharey="row", figsize=(15, 15)
        )

        ax1.set_title("Original", fontsize=15)
        ax1.imshow(np.take(arr_before, SLICE, axis=axis), cmap=cmap)

        ax2.set_title("Ground Truth", fontsize=15)
        # show based on selected axis
        ax2.imshow(np.take(arr_before, SLICE, axis=axis), cmap=cmap)
        ax2.imshow(np.take(ground_truth, SLICE, axis=axis), cmap="jet", alpha=alpha)

        ax3.set_title("Predicted", fontsize=15)
        # show based on selected axis
        ax3.imshow(np.take(arr_before, SLICE, axis=axis), cmap=cmap)
        ax3.imshow(np.take(pred, SLICE, axis=axis), cmap="jet", alpha=alpha)

        plt.tight_layout()

    interact(fn, SLICE=(0, arr_before.shape[axis] - 1))


# explore 3D array comparison with a slider to compare original overlayed with ground truth mask
def explore_3D_array_comparison_overlay(
    arr_data: np.ndarray,
    arr_mask: np.ndarray,
    cmap: str = "gray",
    axis: int = 0,
    alpha: float = 0.5,
):
    assert arr_data.shape == arr_mask.shape

    def fn(SLICE):

        plt.figure(figsize=(7, 7))
        plt.title("Overlay", fontsize=15)

        # show based on selected axis
        plt.imshow(np.take(arr_data, SLICE, axis=axis), cmap=cmap)
        plt.imshow(np.take(arr_mask, SLICE, axis=axis), cmap="jet", alpha=alpha)

        plt.tight_layout()

    interact(fn, SLICE=(0, arr_mask.shape[axis] - 1))


## METRICS AND RESULTS VISUALIZATION
def plot_history(history, path=None, figsize=(60, 30)):
    """
    Plots the history of a model training.
    :param _history: The history of the model training.
    :param config: The configuration of the model.
    """
    # Set seaborn style
    sns.set_style("whitegrid")

    def count_metrics(history):
        i = 0
        for key in history.keys():
            if "val" not in key:
                i += 1
        return i

    # Count the number of metrics
    n_metrics = count_metrics(history)

    columns = 5
    rows = n_metrics // columns

    # add 1 to rows if there is a remainder
    if n_metrics % (columns * rows) != 0:
        rows += 1

    # Create a figure with a subplot for each metric in a grid
    fig, axs = plt.subplots(rows, columns, figsize=figsize)
    axs = axs.ravel()

    # Plot each metric
    for i in range(n_metrics):
        metric = list(history.keys())[i]
        if "val" in metric:
            continue
        axs[i].plot(history[metric])
        axs[i].plot(history["val_" + metric])
        axs[i].set_title(metric)
        axs[i].set_ylabel(metric)
        axs[i].set_xlabel("Epoch")
        axs[i].legend(["Train", "Validation"], loc="upper left")

    plt.tight_layout()
    # Save the plot
    if path:
        plt.savefig(path)
    plt.show()


def plot_loss(_history, path=None, log=False, figsize=(10, 5), **kwargs):

    if log:
        loss = np.log(_history["loss"])
        val_loss = np.log(_history["val_loss"])
    else:
        loss = _history["loss"]
        val_loss = _history["val_loss"]

    plt.plot(loss, **kwargs)
    plt.plot(val_loss, **kwargs)
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"], loc="upper left")
    if path:
        plt.savefig(path)
    plt.show()


def plot_predictions(ids, index, generator, model, alpha=0.4):
    sns.set_style("white")
    print(ids[index])
    # Get data for the specified image_id from the generator
    processed_image, ground_truth = generator.__getitem__(index)

    # Make predictions using the model
    prediction = model.predict(processed_image)

    # if prediction is a type of list then take the last element
    if type(prediction) == list:
        prediction = prediction[-1]

    # riconvert from one-hot encoding to categorical
    p = np.argmax(prediction, axis=-1)
    # Remove -1 values from the ground truth
    ground_truth[ground_truth == -1] = 0
    gt = np.argmax(ground_truth, axis=-1)

    # Explore the 3D array
    explore_3D_predictions(
        np.rot90(processed_image[0, :, :, :, 0], k=-1),
        np.rot90(gt[0, :, :, :], k=-1),
        np.rot90(p[0, :, :, :], k=-1),
        alpha=alpha,
        axis=2,
    )


def plot_domain_adaptation(ids, index, generator, model, alpha=0.4):
    sns.set_style("white")
    # Get data for the specified image_id from the generator
    processed_image, [regions, mask, condition] = generator.__getitem__(index)

    # Extract predictions for each output
    predictions = model.predict(processed_image)
    print("ID: " + str(ids[index]))
    print(
        "Real condition: "
        + str(condition[0])
        + " Predicted condition: "
        + str(predictions[2][0])
    )
    # Convert from one-hot encoding to categorical
    mask = np.greater(mask, 0.5).astype(int)
    predictions[1] = np.greater(predictions[1], 0.5).astype(int)

    regions[regions == -1] = 0
    regions = np.argmax(regions, axis=-1)
    predictions[0] = np.argmax(predictions[0], axis=-1)

    # Explore 3d array for regions and masks
    explore_3D_predictions(
        np.rot90(processed_image[0, :, :, :, 0], k=-1),
        np.rot90(regions[0, :, :, :], k=-1),
        np.rot90(predictions[0][0, :, :, :], k=-1),
        alpha=alpha,
        axis=2,
    )
    explore_3D_predictions(
        np.rot90(processed_image[0, :, :, :, 0], k=-1),
        np.rot90(mask[0, :, :, :, 0], k=-1),
        np.rot90(predictions[1][0, :, :, :, 0], k=-1),
        alpha=alpha,
        axis=2,
    )


def plot_multitask(ids, index, generator, model, alpha=0.4):
    sns.set_style("white")
    # Get data for the specified image_id from the generator
    processed_image, [regions, mask] = generator.__getitem__(index)

    # Extract predictions for each output
    predictions = model.predict(processed_image)
    print("ID: " + str(ids[index]))

    # Convert from one-hot encoding to categorical
    mask = np.greater(mask, 0.5).astype(int)
    predictions[1] = np.greater(predictions[1], 0.5).astype(int)

    regions = np.argmax(regions, axis=-1)
    regions[regions == -1] = 0
    predictions[0] = np.argmax(predictions[0], axis=-1)

    # Explore 3d array for regions and masks
    explore_3D_predictions(
        np.rot90(processed_image[0, :, :, :, 0], k=-1),
        np.rot90(regions[0, :, :, :], k=-1),
        np.rot90(predictions[0][0, :, :, :], k=-1),
        alpha=alpha,
        axis=2,
    )
    explore_3D_predictions(
        np.rot90(processed_image[0, :, :, :, 0], k=-1),
        np.rot90(mask[0, :, :, :, 0], k=-1),
        np.rot90(predictions[1][0, :, :, :, 0], k=-1),
        alpha=alpha,
        axis=2,
    )
