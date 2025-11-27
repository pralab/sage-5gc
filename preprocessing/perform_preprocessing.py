from pathlib import Path
import logging
import os
from preprocessor import preprocessing_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

INPUT_DIR = f"{Path(__file__).parent.parent}/data/cleaned_datasets"
OUTPUT_DIR = f"{Path(__file__).parent.parent}/data/final_datasets_from_preprocessing"

if __name__ == "__main__":
    logger.info("\nPREPROCESSING PIPELINE\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build input paths
    cleaned_paths = [
        f"{INPUT_DIR}/dataset_1_cleaned.csv",
        f"{INPUT_DIR}/dataset_2_cleaned.csv",
        f"{INPUT_DIR}/dataset_3_cleaned.csv",
    ]
    # Check files existence
    missing = [p for p in cleaned_paths if not Path(p).exists()]
    if missing:
        logger.warning("ERROR: Missing input files:")
        for p in missing:
            logger.warning(f"   - {p}")
        exit(1)

    # Run preprocessing starting from cleaned datasets
    preprocessing_pipeline(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)
    logger.info(f"\nFinal datasets saved to: {OUTPUT_DIR}/")
