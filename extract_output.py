"""Helper script to reassemble and extract output folder from output_archive chunks."""

import glob
import os
import zipfile


def main():
    chunk_files = sorted(glob.glob("output_archive/output.zip.*"))
    if not chunk_files:
        print("No output chunks found in output_archive/")
        return

    output_zip = "output_reconstructed.zip"
    print(f"Reassembling {len(chunk_files)} parts into {output_zip}...")
    with open(output_zip, "wb") as outfile:
        for chunk in chunk_files:
            print(f"  Reading {chunk}...")
            with open(chunk, "rb") as infile:
                outfile.write(infile.read())

    print("Extracting output files...")
    with zipfile.ZipFile(output_zip, "r") as zf:
        zf.extractall(".")

    if os.path.exists(output_zip):
        os.remove(output_zip)

    print("Output folder successfully extracted to output/ with matching_results.tsv and candidate_pairs.tsv!")


if __name__ == "__main__":
    main()
