from pathlib import Path

from preprocessor import preprocessing_pipeline

INPUT_DIR = "cleaned_datasets"
OUTPUT_DIR = "final_datasets_from_preprocessing"

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("5G-ATTACKS PREPROCESSING PIPELINE")
    print("=" * 80)
    print(f"Input:  {INPUT_DIR}/dataset_*_cleaned.csv")
    print(f"Output: {OUTPUT_DIR}/dataset_*_final.csv")
    print("=" * 80 + "\n")

    # Build input paths
    cleaned_paths = [
        f"{INPUT_DIR}/dataset_1_cleaned.csv",
        f"{INPUT_DIR}/dataset_2_cleaned.csv",
        f"{INPUT_DIR}/dataset_3_cleaned.csv",
    ]

    # Check existence
    missing = [p for p in cleaned_paths if not Path(p).exists()]
    if missing:
        print("❌ ERROR: Missing input files:")
        for p in missing:
            print(f"   - {p}")
        exit(1)

    # Run preprocessing
    preprocessing_pipeline(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)

    print(f"\n{'=' * 80}")
    print("✅ ALL DONE!")
    print(f"{'=' * 80}")
    print(f"\nFinal datasets saved to: {OUTPUT_DIR}/")
    print(f"\nStandardScaler also saved (returned by function)")
    print(f"{'=' * 80}\n")
