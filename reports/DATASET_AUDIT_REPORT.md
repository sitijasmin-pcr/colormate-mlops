# ColorMate MLOps --- Dataset Audit Report

**Dataset:** `menes000/armocromia-12-season`\
**Audit date:** 19 November 2026\
**Audit status:** Preliminary technical audit completed; semantic and
identity-level validation remain open.

## 1. Executive summary

The dataset contains 4,710 face images across 12 seasonal-color
subtypes, with 3,798 training images and 912 test images. The executed
checks found no unreadable images and no pixel-identical duplicate
pairs. All images are PNG at 224 × 224 pixels. Two training images are
RGBA while the remaining images are RGB; direct inspection confirmed
that both RGBA images have fully opaque alpha channels.

These results support using the dataset for **controlled research
experiments**, subject to the dataset's stated research-only,
non-commercial terms. They do not establish that the labels are
objectively correct, that the train/test split is identity-independent,
or that the dataset is representative of ColorMate's intended users. The
dataset must not be treated as cleared for commercial deployment.

## 2. Dataset description

The Hugging Face dataset card describes the dataset as a 12-subtype
seasonal color analysis image dataset in ImageFolder format. It reports
4,710 images at 224 × 224 pixels. The card says some images originate
from Deep Armocromia and the remainder were collected independently. It
also describes face-region masking, with pixels outside the mask set to
black.

Dataset card:
https://huggingface.co/datasets/menes000/armocromia-12-season

### Split sizes

  Split            Images
  ----------- -----------
  Train             3,798
  Test                912
  **Total**     **4,710**

### Class distribution

  Class                 Train      Test       Total
  --------------- ----------- --------- -----------
  autumn_deep             409       112         521
  autumn_soft             301        73         374
  autumn_warm             276        74         350
  spring_bright           275        66         341
  spring_light            310        64         374
  spring_warm             342        73         415
  summer_cool             274        53         327
  summer_light            300        70         370
  summer_soft             347        63         410
  winter_bright           255        58         313
  winter_cool             321        89         410
  winter_deep             388       117         505
  **Total**         **3,798**   **912**   **4,710**

The training split ranges from 255 images (`winter_bright`) to 409
(`autumn_deep`). This is class imbalance, but not by itself evidence
that the dataset is unusable. Training should report per-class metrics
and consider class weighting or balanced sampling where appropriate.

## 3. Audit methods and results

### 3.1 Dataset loading and metadata

The dataset loaded successfully through the Hugging Face `datasets`
library.

Observed columns: - `image` - `label`

The label feature contains 12 class names. No subject identifier, source
identifier, respondent identifier, original filename, or other identity
metadata was exposed in the inspected records.

**Result:** metadata inspection completed. Because subject IDs are
absent, identity-level separation between train and test cannot be
verified from the available rows.

### 3.2 Image integrity and technical properties

The image quality validation script attempted to load every image in
both splits.

  Check                    Train   Test
  ---------------------- ------- ------
  Total checked            3,798    912
  Successfully loaded      3,798    912
  Corrupted/unreadable         0      0
  Dimensions 224 × 224     3,798    912
  PNG                      3,798    912
  RGB                      3,796    912
  RGBA                         2      0

**Result:** no unreadable images were detected. Dimensions and file
formats were consistent. Two training images use RGBA rather than RGB.

The two RGBA images were inspected individually: - `train` index 1162,
label `spring_bright` - `train` index 1451, label `spring_light`

Both had alpha value 255 for all pixels (fully opaque). They can be
converted to RGB with `image.convert("RGB")` without discarding visible
transparency information. They should not be deleted solely because
their mode is RGBA.

**Documentation note:** the dataset card describes the standardized
dataset as RGB throughout, but the executed inspection found two RGBA
images. The audit records the observed files rather than assuming the
card description overrides the downloaded data.

### 3.3 Visual sampling

A random sample of 10 images per class (120 images total) was generated
for visual review. The reviewed sample showed face portraits with
variation in appearance, framing, lighting, makeup, and background. The
dataset card describes the images as face-masked with black pixels
outside the face region.

Visual inspection is useful for identifying obvious image-quality
concerns, but it cannot establish that a seasonal label is correct. It
also does not establish demographic representativeness.

### 3.4 Duplicate screening

Two duplicate checks were run: - Exact duplicate screening using SHA-256
over decoded RGB pixel data and image dimensions. - Perceptual candidate
screening using 64-bit difference hash (dHash), with a Hamming-distance
threshold of 6.

Observed output: - Exact duplicate pairs: 0 - Cross-split exact
duplicate pairs: 0 - Near-duplicate candidates across all records:
112,545 - Cross-split near-duplicate candidates: 42,132

The dHash candidates were not accepted as confirmed duplicates. Manual
review of low-distance cross-split pairs showed many visually different
people with similar framing/background composition. Therefore, dHash at
this threshold is too broad to serve as an identity-leakage count for
this dataset.

**Interpretation:** no pixel-identical pairs were detected by the
implemented exact-hash method. This does not prove there are no resized,
cropped, edited, or same-subject images across splits. Since subject IDs
are unavailable, identity leakage remains **unverified**, not ruled out.

## 4. Label and scientific validity limitations

The dataset contains one categorical label per image. Its dataset card
states that seasonal color labels are subjective and that subtype
boundaries can be ambiguous. The card also notes that demographic
coverage has not been characterized.

The current audit did not independently validate the 12 subtype labels
with qualified color analysts or a documented annotation protocol.
Consequently: - labels must be treated as supplied dataset annotations,
not independently verified ground truth; - the audit cannot determine
inter-rater agreement; - the audit cannot establish subgroup performance
or demographic representativeness; - a model trained on this dataset
must not be presented as a definitive personal-color diagnosis.

For future data collection, define a documented labeling protocol.
Prefer two independent qualified assessors and an adjudication process
for disagreements. Record relevant capture conditions and consent while
minimizing personally identifying metadata.

## 5. License and intended use

The Hugging Face dataset card states **research and scientific use only;
no commercial use**. It also says that some source imagery comes from
Deep Armocromia and that part of that source imagery derives from
CelebA, whose terms also apply.

This restriction is material to ColorMate's intended UMKM/business
context. The dataset may be considered for academic research experiments
only within the applicable terms. It must **not** be assumed suitable
for a commercial kiosk, commercial product, or business deployment.
Review the dataset card and all upstream terms before redistribution or
any other use.

The dataset card requests citation of the original Deep Armocromia work:

Stacchio, L., Paolanti, M., Spigarelli, F., & Frontoni, E. (2025). *Deep
Armocromia: A Novel Dataset for Face Seasonal Color Analysis and
Classification*. In Computer Vision -- ECCV 2024 Workshops,
pp. 352--367. Springer Nature Switzerland.

## 6. Recommendations for ColorMate MLOps

1.  Keep this public dataset isolated as a research/benchmark dataset.
    Do not mix it with independently collected data until source,
    consent, label definitions, and split strategy have been reviewed.
2.  Do not use it for commercial deployment under its current stated
    license.
3.  Convert images to RGB consistently in the input pipeline. Preserve
    the original dataset and make preprocessing reproducible.
4.  Because the dataset card describes black face-mask backgrounds,
    ensure any color-statistics or face-color features exclude
    masked-out black pixels. Do not apply color-altering augmentation
    (such as hue shifts, saturation changes, or ColorJitter) when color
    is the target signal.
5.  Keep the provided test split untouched for final evaluation. Do not
    use it for model selection or tuning.
6.  Do not claim identity-independent splitting. If identity metadata
    cannot be obtained, document this limitation and treat test metrics
    as potentially affected by unknown subject overlap.
7.  Report macro-F1, per-class precision/recall, confusion matrix, and
    both 12-subtype and derived 4-season results if the label taxonomy
    supports that comparison. Include uncertainty and subgroup
    evaluation when suitable data and permissions exist.
8.  Collect a separate, consented dataset for the intended target
    population only after establishing lawful collection, storage,
    deletion, retention, and access controls. Processing consent and
    optional storage consent should be distinct.
9.  Obtain expert review for labels before treating collected samples as
    ground truth. Skin tone alone must not be used as a substitute for
    seasonal color assessment.
10. Present model outputs as decision support or candidate suggestions,
    not as definitive judgments about a person.

## 7. Audit conclusion

**Preliminary technical outcome:** the 4,710 records loaded; all images
were readable; all were 224 × 224 PNG; and no exact pixel duplicates
were detected by the implemented method. Two fully opaque RGBA images
can be normalized to RGB during preprocessing.

**Outstanding blockers before claiming a validated training dataset:**
label validation, identity-level train/test separation, demographic
coverage, and full review of upstream licensing and provenance. The
stated research-only, non-commercial license also prevents treating this
dataset as a commercial deployment data source.

Therefore, the dataset is **conditionally suitable for academic
experimentation and pipeline development**, but it is **not validated as
ground truth and is not cleared for commercial use**.

## 8. Audit artifacts

The following scripts and outputs were used or generated during the
audit:

-   `scripts/audit_dataset.py`
-   `scripts/visualize_dataset.py`
-   `scripts/detect_duplicates.py`
-   `scripts/visualize_duplicate_pairs.py`
-   `scripts/inspect_dataset_metadata.py`
-   `scripts/validate_image_quality.py`
-   `scripts/inspect_rgba_images.py`
-   `reports/dataset_audit/armocromia_random_sampling_120.png`
-   `reports/dataset_audit/duplicate_report.csv`
-   `reports/dataset_audit/rgba_samples/`

Generated images and CSV files are supporting audit artifacts; avoid
committing raw personal images or sensitive data. Re-run scripts and
retain their outputs with version/date information when the audit is
repeated.
