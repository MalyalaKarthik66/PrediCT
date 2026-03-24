# PrediCT CAC Project: Simple Explanation

## 1) What is this project?
This project is about heart health.
It studies calcium inside the heart arteries.
Those arteries are called coronary arteries.
The short name is CAC, which means Coronary Artery Calcium.

The project is part of ML4SCI work.
ML4SCI means using modern data and machine learning methods for science.
Here, the science is medical imaging.

In simple words:
We take CT scan images.
We try to find calcium spots.
We measure how close those spots are to known artery paths.
Then we make visual reports and metrics.

## 2) What is the main problem statement?
Doctors use CAC scans to estimate heart risk.
But raw scans are not always easy to analyze.
Calcium can be small, sparse, and noisy.
Different scans also have different spacing and quality.

So the problem is:
How do we build a clean, repeatable pipeline that can:
load scans,
prepare them,
align anatomy,
measure calcium location,
and show useful results?

This repository solves that engineering and research problem.
It gives a full pipeline from data to visual outputs.

## 3) What is CAC and why does it matter?
CAC means calcium buildup in coronary arteries.
Calcium buildup is related to atherosclerosis.
Atherosclerosis can increase risk of heart attack.

If we detect and quantify CAC well,
we can improve risk assessment.
That can help early prevention and treatment planning.

So CAC detection is important because:
it supports better heart risk understanding,
it can be done from CT images,
and it can guide clinical decisions.

## 4) What is the Stanford COCA dataset?
The Stanford COCA dataset is a public dataset.
It includes heart CT data related to coronary calcium.
It is useful for research and model development.

In this project, COCA is the main real dataset target.
If COCA is available, the pipeline can process it.
If COCA is not available, we use synthetic demo data.

## 5) What is the ImageCAS coronary atlas?
ImageCAS is a coronary artery atlas resource.
An atlas is like a reference map of anatomy.
It helps us understand where arteries should be.

In this project, atlas data supports registration.
Registration means aligning one image to another.
After alignment, centerline paths can be transferred.
Then calcium can be compared to artery locations.

## 6) Why segmentation and registration are needed
Segmentation means marking structures in images.
Here, it usually means identifying calcium regions.

Registration means spatial alignment.
A moving image is aligned to a fixed image.
This allows anatomy from reference space to map into scan space.

Why both are needed:
Segmentation tells us where calcium is.
Registration tells us where arteries are.
Together they allow distance-based evaluation.

## 7) Full workflow in simple form
Data -> Preprocessing -> Registration -> Distance calculation -> Visualization

Now in plain steps:
Step A: Load CT data.
Step B: Clean and standardize image values.
Step C: Align anatomy with atlas transforms.
Step D: Compute calcium-to-centerline distances.
Step E: Make metrics and plots.

## 8) How this pipeline is implemented
The code is modular.
Each job has its own folder and module.
This makes it easier to test and improve.

Main technical ideas:
NIfTI files are used for 3D image storage.
SimpleITK is used for registration transforms.
NumPy handles array operations.
SciPy KD-tree computes nearest distances fast.
Matplotlib builds result figures.

## 9) Step-by-step pipeline stages
### Stage 1: Data handling
DICOM scans can be converted to NIfTI.
Metadata is collected (spacing, size, thickness).

### Stage 2: Preprocessing
HU clipping keeps values in a useful range.
Z-score normalization standardizes intensity scale.
Resampling makes voxel spacing consistent (often 1 mm).
ROI cropping focuses on relevant heart region.

### Stage 3: Atlas registration
Rigid registration aligns position and rotation.
Affine registration also allows scaling and shearing.
Optional deformable registration allows flexible warping.
Output transform files are saved for reuse.

### Stage 4: Centerline transform
Centerline masks or points are mapped by transforms.
This gives artery paths in scan space.

### Stage 5: Distance calculation
Calcium voxels are extracted from the mask.
Centerline voxels are extracted from transformed mask.
A KD-tree finds nearest centerline for each calcium voxel.
Distances are measured in millimeters.

### Stage 6: Evaluation metrics
Mean distance: average calcium-to-centerline distance.
Median distance: middle distance, robust to outliers.
Percent within 10 mm: fraction of calcium close to arteries.
These metrics are written to JSON files.

### Stage 7: Visualization
Overlay 1: CT + calcium mask.
Overlay 2: CT + centerline mask.
Histogram: distribution of calcium distances.
These plots help human interpretation.

## 10) What each main folder does
data/
Stores input and intermediate datasets.
Includes raw DICOM, NIfTI, preprocessed images, and masks.

outputs/
Stores generated outputs.
Includes registered transforms, warped centerlines, and plots.

src/predict_cac/
Main Python package with modular pipeline code.
Contains data, transforms, registration, evaluation, scoring, and utils modules.

scripts/
Command-line entry points to run pipeline stages.

configs/
YAML configuration files for preprocessing, registration, and segmentation settings.

notebooks/
Interactive exploration and result-inspection notebooks.

experiments/
Saved experiment metrics such as metrics.json.

tests/
Pytest tests for core pipeline logic.

## 11) What the scripts do
generate_demo_data.py
Creates synthetic CT, calcium mask, and centerline mask.
Useful when real data is unavailable.

run_ingest.py
Ingests data and builds metadata tables.

run_preprocessing.py
Runs HU clipping, normalization, spacing resampling, and ROI crop.

run_registration.py
Runs rigid/affine/optional deformable registration.
Saves transform outputs.

run_validation.py
Computes distance metrics and writes JSON outputs.
Also creates plot images.

train_segmentation.py
Placeholder entry point for model training experiments.

## 12) What the demo pipeline shows
The demo pipeline proves the full system works end to end.
It does not need COCA to demonstrate behavior.

It generates synthetic sample data.
Then it runs preprocessing, registration, and validation.
Finally it produces metrics and plots.

So anyone can test the architecture quickly.
This is useful for onboarding and presentations.

## 13) What the generated plots represent
ct_calcium_overlay.png
Shows CT anatomy with calcium highlighted.
Helps verify calcium mask location visually.

centerline_overlay.png
Shows CT anatomy with transformed centerline overlay.
Helps verify centerline alignment quality.

distance_histogram.png
Shows how far calcium voxels are from centerlines.
A left-shifted histogram means calcium is closer on average.

## 14) What the metrics mean in simple language
Mean distance (mm)
On average, how far calcium is from centerlines.
Lower is generally better alignment/proximity.

Median distance (mm)
Typical distance for the middle calcium voxel.
Less sensitive to extreme values.

Percent within 10 mm
How much calcium lies close to centerlines.
Higher percentage usually indicates better anatomical proximity.

## 15) How to run the demo pipeline
1. Create and activate the Python environment.
2. Install dependencies from requirements.txt.
3. Run:
python scripts/generate_demo_data.py
python scripts/run_preprocessing.py
python scripts/run_registration.py
python scripts/run_validation.py

After this, check:
outputs/plots/ for images,
experiments/metrics.json for metrics,
outputs/validation_metrics.json for copied metrics.

## 16) Real-data compatibility
If COCA and atlas data are available,
place them in the configured data folders.
Update YAML configs if paths differ.
Run the same scripts.

So the same architecture works for:
quick demo mode,
and real research mode.

## 17) Final takeaway
This repository is a clear, modular CAC analysis pipeline.
It is designed for learning, testing, and extension.
Even without deep ML or medical background,
you can follow the stages,
run the demo,
inspect the plots,
and understand the core idea from image data to interpretable metrics.
