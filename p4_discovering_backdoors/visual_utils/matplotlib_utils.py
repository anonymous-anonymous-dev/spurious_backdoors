import numpy as np
import matplotlib.pyplot as plt



def show_images_in_grid(images: np.ndarray, n_cols: int, n_rows: int=1, titles: str=None):
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*2))
    for i in range(n_rows*n_cols):
        axs[i%n_cols].imshow( images[i] ) if n_rows<= 1 else axs[i//n_cols][i%n_cols].imshow( images[i] )
        
        if titles is not None:
            axs[i%n_cols].set_title( titles[i] ) if n_rows<=1 else axs[i//n_cols][i%n_cols].set_title( titles[i] )
            
    plt.tight_layout()
    
    return fig