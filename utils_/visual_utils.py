import matplotlib.pyplot as plt
import numpy as np



def show_image_grid(images, y=None, n_rows: int=1, n_cols: int=1, channels_first: bool=False, fig_size_scaling: int=1, font_size=None, **kwargs) -> None:
    """
    This function takes in a list of images and shows them in the form of a grid of {n_rows} x {n_cols}.
    
    Input:
        images: a list of images to be shown as grid. The list must be larger than {n_rows} x {n_cols}
        n_rows: number of rows in the grid.
        n_cols: number of columns in the grid.
        
    Output:
        None
    """
    
    # assert len(images)>=n_rows*n_cols, f'Length of images is {len(images)}, but it must be >= n_rows*n_cols, which is {n_rows*n_cols}.'
    if len(images)<n_rows*n_cols:
        n_rows = max(len(images) // n_cols, 1)
        n_cols = max(len(images) // n_rows, 1)
    
    if channels_first:
        images = np.transpose(images, (0,2,3,1))
        
    title_font = None
    label_font = None
    figure_font = None
    if font_size is not None:
        title_font = 3*font_size
        label_font = font_size
        figure_font = font_size
    
    fig_height = 3 if y is None else 3.3
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*2.9*fig_size_scaling, n_rows*fig_height*fig_size_scaling))
    for i in range(n_rows*n_cols):
        if n_rows>1:
            if n_cols>1:
                axs[i//n_cols][i%n_cols].imshow(images[i], **kwargs)
                axs[i//n_cols][i%n_cols].set_xticks([])
                axs[i//n_cols][i%n_cols].set_yticks([])
                if y is not None: axs[i//n_cols][i%n_cols].set_title(y[i], fontsize=title_font)
            else:
                axs[i//n_cols].imshow(images[i], **kwargs)
                axs[i//n_cols].set_xticks([])
                axs[i//n_cols].set_yticks([])
                if y is not None: axs[i//n_cols].set_title(y[i], fontsize=title_font)
        else:
            if n_cols>1:
                axs[i].imshow(images[i], **kwargs)
                axs[i].set_xticks([])
                axs[i].set_yticks([])
                if y is not None: axs[i].set_title(y[i], fontsize=title_font)
            else:
                plt.imshow(images[i], **kwargs)
                plt.xticks([])
                plt.yticks([])
                if y is not None: plt.title(y[i], fontsize=title_font)
    
    plt.grid(visible=False)
    plt.tight_layout()
        
    return fig


def show_stacked_numpy_images(imgs, actual_images, channels_first: bool=False, normalize_imgs: bool=True, num_images: int=5, show_random: bool=True):
    
    from .general_utils import normalize
    
    if channels_first:
        imgs = np.transpose(imgs, (0,2,3,1))
        actual_images = np.transpose(actual_images, (0,2,3,1))
        
    if normalize_imgs:
        imgs = normalize(imgs)
        actual_images = normalize(actual_images)
    
    num_images = min(num_images, len(imgs))
    show_indices = np.random.choice(len(imgs), num_images, replace=False)
    fig, axs = plt.subplots(num_images, 2, figsize=(5, num_images*3))
    for i, img_indx in enumerate(show_indices):
        axs[i][0].imshow(actual_images[img_indx]) if num_images>1 else axs[0].imshow(actual_images[img_indx])
        axs[i][1].imshow(imgs[img_indx]) if num_images>1 else axs[1].imshow(imgs[img_indx])
    plt.show()
    
    return

