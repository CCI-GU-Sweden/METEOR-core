# METEOR-core

Metrology Engine for Optical Robustness — Core Algorithms

METEOR-core is the analysis engine of the METEOR ecosystem:  
a unified framework for **microscope performance evaluation**, including:

- **PSF analysis** (sub-resolution beads)
- **Star-pattern resolution (FFT)**
- **2D-rings distortion**
- **Co-registration accuracy**
- **Field uniformity (Argolight patterns)**
- **Intensity linearity**
- **Stage drift & Z-stack accuracy**
- **General Argolight metrology**

This package contains **only the computational algorithms** (the “core”).  
Visualization, scoring dashboards, and acquisition automation are handled by:

- **METEOR-view** (web visualization & scoring — Flask) *(in development)*
- **METEOR-smart** (smart microscopy & automated acquisition workflows) *(upcoming)*

---

## 🧭 Purpose

Modern fluorescence microscopes require periodic quantitative performance checks.  
METEOR-core provides a **complete algorithmic suite** for evaluating the health and robustness of:

- Objective lenses  
- Illumination systems  
- Detection paths  
- Stages and focus drives  
- Imaging stability over time  

The package is designed to integrate seamlessly into:

- Core facilities  
- Quality control pipelines  
- Automated calibration workflows  
- Scientific microscope-as-a-service environments  

---

## ✨ Features

### 🔬 Optical Quality Metrics

- PSF (X/Y/Z FWHM) from bead stacks  
- Ring Spread Function (RSF)  
- FFT-based radial star-pattern resolution  
- Objective performance scoring

### 🧩 Geometric Accuracy

- Distortion mapping from Argolight 2D rings  
- Co-registration accuracy across channels  
- Lattice fitting with robust optimization

### 🌈 Illumination & Detector Metrics

- Field uniformity  
- Centering accuracy  
- Intensity linearity (Argolight 4×4 patterns)

### 🏃 Stability & Drift

- XY drift across time  
- Z drift across sequences  
- Staircase Z-calibration accuracy  

### 🗂 File Format Support

- Native **CZI** reading  
- Native **OMERO** integration  
- Dimension-aware slicing (`TCZYX` handling)  

---

## 📦 Installation

Local installation (while private / under development):

```bash
git clone https://github.com/CCI-GU-Sweden/METEOR-core.git
cd METEOR-core
pip install -e .
```

## 🧪 Usage Example

```python
from METEOR_core import reader, star_resolution_analysis
img, metadata = reader("40x_star.czi")

res = star_resolution_analysis(img, metadata)
print(res["measurements"])
res["figure"].savefig("star_resolution.jpg")
```

A full pipeline classification system is available via:

```python
from METEOR_core.metrology import reader, metrology_splitter
img, metadata = reader("image_name.czi")
results = metrology_splitter(img, "image_name.czi", metadata)
```

## 📣 Citation

When publishing results obtained using METEOR-core, please cite:

METEOR-core: A Unified Metrology Engine for Automated Assessment of Microscope Performance
(2025) — GitHub: https://github.com/CCI-GU-Sweden/METEOR-core

A DOI will be added once the project is archived on Zenodo.

## 🤝 Contributing

Contributions, issues and feature requests are welcome!
Internal contributors: please use the GitHub private repo.

## 🌌 Part of the METEOR Ecosystem

METEOR-core — computational metrology engine

METEOR-view — browser-based scoring & visualization (in development)

METEOR-smart — automated smart microscopy (upcoming)