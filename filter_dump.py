"""Filter the original TAWOS.sql dump so that only the five required tables remain.

Input : data/TAWOS.sql            (~4.1 GB)
Output: data/TAWOS_required.sql   (~0.96 GB)

The dump is processed line by line, so the whole file is never loaded into memory.
"""

from pathlib import Path

source = Path("data/TAWOS.sql")
destination = Path("data/TAWOS_required.sql")

required = {"Issue", "Project", "Repository", "Sprint", "User"}

found = set()
current_table = None
keep = False

with source.open("r", encoding="utf-8", errors="replace") as infile:
    with destination.open("w", encoding="utf-8") as outfile:

        for line in infile:

            if line.startswith("-- Table structure for table `"):
                current_table = line.split("`")[1]
                keep = current_table in required

                if keep:
                    found.add(current_table)

            # Lines before the first table (dump header / SET statements) are kept.
            if current_table is None or keep:
                outfile.write(line)

if found != required:
    destination.unlink(missing_ok=True)
    missing = required - found
    raise RuntimeError(f"Missing required tables: {missing}")

print("Extracted tables:", sorted(found))
print(f"Output file: {destination}")
print(f"Output size: {destination.stat().st_size / (1024**3):.2f} GB")
