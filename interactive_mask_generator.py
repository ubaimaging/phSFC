"""
interactive_mask_generator.py
=============================
Interactively generate binary masks for LSM (Zeiss laser-scanning microscope)
and TF/TIFF files and save them as PNG images.

Usage
-----
    python interactive_mask_generator.py <folder_path>

Arguments
---------
    folder_path : str
        Path to a directory that contains one or more ``.lsm``, ``.tf``,
        ``.tif``, or ``.tiff`` files. The script will iterate over every
        supported file found directly inside that directory (non-recursive).

Workflow
--------
1. Each ``.lsm`` file is opened with ``phasorpy.io.signal_from_lsm``;
    each ``.tf``/``.tif``/``.tiff`` file is opened with
    ``tifffile.imread``.
2. A 2-D mean-intensity image is displayed in an interactive Matplotlib
    window.
3. The user draws a polygon around the region of interest:
       * **Click** to place vertices.
       * **Press Enter** (or click the first vertex again) to close and
         confirm the polygon.
       * **Close the window** without pressing Enter to skip the current
         file (no mask is saved for it).
4. A binary mask (white = selected region, black = background) is created
   from the polygon and saved as a PNG file inside a ``mask/`` subfolder
   that is automatically created under ``folder_path``.

Output
------
    <folder_path>/mask/mask_<base_filename>.png
        One PNG per processed supported file that had a polygon drawn.  The
        image is an 8-bit grayscale bitmap where 255 = inside the polygon
        and 0 = outside.

Dependencies
------------
    numpy, matplotlib, opencv-python (cv2), phasorpy, tifffile

Example
-------
    python interactive_mask_generator.py "/path/to/lsm/folder"
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
import cv2
import tifffile
from phasorpy.io import signal_from_lsm
from phasorpy.phasor import phasor_from_signal


SUPPORTED_EXTENSIONS = ('.lsm', '.tf', '.tif', '.tiff')


def load_mean_image(data_path):
    """
    Load a microscopy file and return a 2-D mean-intensity image.

    For ``.lsm`` files, intensity is obtained via ``phasor_from_signal``.
    For ``.tf``/``.tif``/``.tiff`` files, data are read with
    ``tifffile.imread`` and collapsed to 2-D by averaging all leading axes.
    """
    ext = os.path.splitext(data_path)[1].lower()

    if ext == '.lsm':
        signal = signal_from_lsm(data_path)
        mean, _, _ = phasor_from_signal(signal, axis=0)
        return np.asarray(mean)

    if ext in ('.tf', '.tif', '.tiff'):
        data = np.asarray(tifffile.imread(data_path))

        if data.ndim < 2:
            raise ValueError(f"Unsupported TIFF data shape {data.shape} for '{data_path}'")

        if data.ndim == 2:
            return data

        # Preserve image plane axes as the last two dimensions and average everything else.
        axes_to_mean = tuple(range(data.ndim - 2))
        return data.mean(axis=axes_to_mean)

    raise ValueError(f"Unsupported file extension '{ext}' for '{data_path}'")

class InteractivePolygonSelector:
    """
    GUI helper that attaches a ``PolygonSelector`` widget to a Matplotlib
    axes and converts the drawn polygon into a boolean mask.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which the polygon will be drawn.
    image : numpy.ndarray
        The 2-D (or 2-D grayscale) image shown in *ax*.  Its shape is used
        to set the mask dimensions.
    """
    def __init__(self, ax, image):
        self.ax = ax
        self.image = image
        self.polygon_points = []
        self.mask = None
        self.selector = None
        self.finished = False
        
    def start_selection(self):
        """Start interactive polygon selection"""
        self.selector = PolygonSelector(
            self.ax, 
            self.on_select,
            useblit=True,
            props=dict(color='red', linestyle='-', linewidth=2, alpha=0.8),
            handle_props=dict(markersize=8, markerfacecolor='red')
        )
        print("Click to create polygon vertices. Press Enter when done, or close window to skip masking.")
        
    def on_select(self, verts):
        """Callback when polygon is completed"""
        if len(verts) < 3:
            return
            
        self.polygon_points = verts
        self.create_mask()
        self.finished = True
        plt.close()
        
    def create_mask(self):
        """Create binary mask from polygon points"""
        if not self.polygon_points:
            return
            
        # Create coordinate arrays
        height, width = self.image.shape[:2]
        y, x = np.mgrid[:height, :width]
        points = np.vstack((x.ravel(), y.ravel())).T
        
        # Create path from polygon points
        path = Path(self.polygon_points)
        mask = path.contains_points(points)
        self.mask = mask.reshape(height, width)
        
    def get_mask(self):
        """Return the created mask or None if no selection"""
        return self.mask

def select_polygon_interactive(image, title="Select Region"):
    """
    Display *image* in a Matplotlib window and let the user draw a polygon.

    Parameters
    ----------
    image : numpy.ndarray
        2-D array (e.g. mean-intensity map) shown in grayscale.
    title : str, optional
        Window / axes title shown to the user.

    Returns
    -------
    numpy.ndarray or None
        Boolean mask of the same shape as *image* (``True`` inside the
        polygon) if the user completed a selection, or ``None`` if the
        window was closed without a valid polygon.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image, cmap='gray')
    ax.set_title(f"{title}\nClick to create polygon, press Enter when done, or close to skip")
    
    selector = InteractivePolygonSelector(ax, image)
    selector.start_selection()
    
    # Connect key press event for Enter key
    def on_key(event):
        if event.key == 'enter' and selector.polygon_points:
            selector.create_mask()
            selector.finished = True
            plt.close()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Show plot and wait for interaction
    plt.show()
    
    return selector.get_mask()

def main():
    """
    Entry point for the CLI.

    Parses the command-line argument, iterates over all supported microscopy
    files in the given directory, prompts the user to draw a polygon mask for
    each one, and saves the resulting masks to ``<folder_path>/mask/``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Interactively generate polygon masks for LSM/TF/TIFF files and save "
            "them as PNGs inside a 'mask/' subfolder."
        ),
        epilog=(
            "Example: python interactive_mask_generator.py \"/path/to/lsm/folder\""
        ),
    )
    parser.add_argument(
        "folder_path",
        type=str,
        help="Path to the folder containing .lsm/.tf/.tif/.tiff files",
    )
    args = parser.parse_args()
    
    folder_path = args.folder_path
    if not os.path.isdir(folder_path):
        print(f"Error: Directory '{folder_path}' does not exist.")
        return

    mask_dir = os.path.join(folder_path, 'mask')
    os.makedirs(mask_dir, exist_ok=True)
    
    for file in os.listdir(folder_path):
        if not file.lower().endswith(SUPPORTED_EXTENSIONS):
            continue
            
        data_path = os.path.join(folder_path, file)
        base_filename = os.path.splitext(file)[0]
        
        print(f"\nProcessing: {file}")
        try:
            mean = load_mean_image(data_path)
            
            mask = select_polygon_interactive(mean, f"Mean Intensity - {base_filename}")
            
            if mask is not None:
                mask_uint8 = (mask * 255).astype(np.uint8)
                mask_path = os.path.join(mask_dir, f'mask_{base_filename}.png')
                cv2.imwrite(mask_path, mask_uint8)
                print(f"Mask saved: {mask_path}")
            else:
                print(f"No mask selected for {file}, skipping save.")
                
        except Exception as e:
            print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    main()
