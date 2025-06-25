import os
import sys
from sslipgen.aligner import Aligner

def main():
    """
    Main function to configure and run the data processing pipeline.
    """
    # --- User Configuration ---
    
    # 1. SET YOUR INPUT AND OUTPUT DIRECTORIES HERE
    # The input directory should contain your raw video/text files (e.g., in s1, s2... folders)
    input_data_dir = r"E:\\raw_data"
    
    # The output directory is where all processed files and dependencies will be stored.
    output_data_dir = r"E:\\processed_data"

    # --- End of Configuration ---

    print("--- Lip-Reading Data Pipeline ---")
    print(f"Input Dataset: {input_data_dir}")
    print(f"Output Directory: {output_data_dir}")
    
    try:
        # Initialize the Aligner class. It will handle the rest automatically.
        pipeline = Aligner(
            input_dir=input_data_dir,
            output_dir=output_data_dir,
        )

        # Run the entire pipeline
        pipeline.run()

    except (FileNotFoundError, ImportError) as e:
        print(f"\nDependency Error: {e}")
        print("Please check your installation and ensure all required packages are available.")
        sys.exit(1) # Exit with an error code
    except Exception as e:
        print(f"\nAn unexpected error occurred during the pipeline: {e}")
        sys.exit(1)

if __name__ == '__main__':
    # This ensures the multiprocessing works correctly on all platforms
    main()
