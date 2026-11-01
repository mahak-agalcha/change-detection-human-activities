# Dataset Validation and Quality Assurance Report

## 1. Executive Summary

- **Total Sample Triplets Audited:** 6
- **Valid Samples (Passed All Checks):** 6
- **Samples with Warnings:** 1
- **Failed / Corrupted Samples:** 0
- **Overall Dataset Health Score:** 100.0%

## 2. Checkpoint Verification Results

| Checkpoint | Description | Status |
| :--- | :--- | :--- |
| **1. Decodability** | Header parsing & image integrity check | PASSED |
| **2. Dimensional Alignment** | dim(Before) == dim(After) == dim(Mask) | PASSED |
| **3. Channel Validity** | Standard RGB / Grayscale channels verified | PASSED |
| **4. Mask Binarization** | Pixel values strictly constrained to binary domain | PASSED |
| **5. Anomaly Detection** | Inverted masks / missing pairs verification | PASSED |

## 3. Sample-by-Sample Validation Breakdown

| Sample ID | Dimensions | Channels | Changed Pixels | Change % | Unique Mask Vals | Validation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `levir_test_sample_005_suburban_sprawl` | 256x256 | RGB/L | 20,083 | 30.64% | `[0, 255]` | **PASSED** |
| `levir_test_sample_006_no_change_control` | 256x256 | RGB/L | 0 | 0.00% | `[0]` | **WARNING** |
| `levir_train_sample_001_residential` | 256x256 | RGB/L | 11,502 | 17.55% | `[0, 255]` | **PASSED** |
| `levir_train_sample_002_commercial` | 256x256 | RGB/L | 16,761 | 25.57% | `[0, 255]` | **PASSED** |
| `levir_train_sample_003_road_works` | 256x256 | RGB/L | 11,466 | 17.50% | `[0, 255]` | **PASSED** |
| `levir_val_sample_004_urban_infill` | 256x256 | RGB/L | 6,932 | 10.58% | `[0, 255]` | **PASSED** |

## 4. Warnings and Diagnostic Notes

### Sample `levir_test_sample_006_no_change_control`
- **WARNING**: No change pixels detected in mask (0.0% change - potential control or negative pair)
