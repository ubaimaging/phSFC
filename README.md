# Phasor-based Spectral Flow Cytometry (phSFC) Analysis Framework

This repository provides a comprehensive suite of Python scripts and Jupyter notebooks designed for the analysis and visualization of high-dimensional data from Spectral Flow Cytometry (SFC) and Hyperspectral Imaging (HSI). By leveraging phasor analysis, this framework enables the characterization of complex fluorescence spectra—such as those from environment-sensitive probes—into a 2D coordinate system, facilitating the study of membrane biophysics and cellular subpopulations.

The codebase supports data acquired in both `.fcs` (flow cytometry) and `.lsm`/`.tif` (confocal microscopy) formats, offering automated workflows for spectral normalization, phasor transformation, and component fraction estimation.

## Requirements

Before running the scripts or notebooks, ensure you have installed all required dependencies. You can install them using `pip`:

```bash
pip install -r requirements.txt
```

## Repository Contents

### Interactive Analysis Tools

#### `interactive_mask_generator.py`
**Overview**: A command-line utility for generating high-quality binary masks from confocal microscopy images (`.lsm`, `.tif`). It provides an interactive interface to define regions of interest (ROIs) using polygonal selection. It is particularly useful for segmenting specific cellular regions or lipid vesicles before downstream phasor analysis.

**Usage**:
```bash
python interactive_mask_generator.py /path/to/image/folder
```

#### `interactive_phasor_gating.py`
**Overview**: A powerful Qt-based graphical user interface (GUI) designed for real-time phasor gating of flow cytometry (`.fcs`) data. This tool allows users to manually isolate subpopulations directly on the phasor plot, visualize gated events in conventional dot plots, and export high-resolution (600 DPI) publication-ready figures.

**Usage**:
```bash
python interactive_phasor_gating.py path_to_file.fcs
```

---

### Research Notebooks

These Jupyter notebooks serve as the primary analysis pipeline, reproducing the key findings, statistical validations, and figures presented in the phSFC publication.

- **`figure_2.ipynb`**: Core analysis notebook detailing the workflow for main Figure 2, including spectral normalization and component fraction calculations.
- **`supp_figure1_notebook.ipynb` to `supp_figure5_notebook.ipynb`**: Modular notebooks dedicated to the supplementary figures. These reproduce specialized analyses such as 3D spectral visualizations, cross-platform (SFC vs. HSI) comparisons, and lipid trajectory modeling.

**How to Run**:
Launch Jupyter from your terminal:
```bash
jupyter lab
# or
jupyter notebook
```
Then open the relevant notebook and run the cells sequentially.
