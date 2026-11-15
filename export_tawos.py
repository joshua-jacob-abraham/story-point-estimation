"""Export the TAWOS issue data from the local MySQL database to a Parquet file.

Joins `Issue` with `Project`, selects the 19 required columns and writes them in
batches of 25,000 rows using an explicit PyArrow schema (this avoids batch-to-batch
type inconsistencies, e.g. Sprint_ID inferred as null in one batch and float in another).

Output: output/TAWOS.parquet  (458,232 rows x 19 columns, ~318.75 MB)
"""

import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import create_engine


# --------------------------------------------------
# 1. Database connection
# --------------------------------------------------
# Local development credentials from the README. Override with the
# TAWOS_DATABASE_URL environment variable if you use different ones.

DATABASE_URL = os.environ.get(
    "TAWOS_DATABASE_URL",
    "mysql+pymysql://root:tawos_local_password@127.0.0.1:3307/tawos_cleaned",
)

engine = create_engine(DATABASE_URL)


# --------------------------------------------------
# 2. SQL query
# --------------------------------------------------

query = """
SELECT
    i.ID AS Issue_ID,
    i.Issue_Key,
    i.Project_ID,
    p.Name AS Project_Name,
    i.Title,
    i.Description,
    i.Description_Text,
    i.Story_Point,
    i.Type,
    i.Priority,
    i.Status,
    i.Creation_Date,
    i.Estimation_Date,
    i.Resolution_Date,
    i.Last_Updated,
    i.Story_Point_Changed_After_Estimation,
    i.Description_Changed_After_Estimation,
    i.Title_Changed_After_Estimation,
    i.Sprint_ID

FROM Issue i
JOIN Project p
    ON i.Project_ID = p.ID

ORDER BY i.ID
"""


# --------------------------------------------------
# 3. Output configuration
# --------------------------------------------------

output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "TAWOS.parquet"

# Remove an existing file so that an incomplete
# previous export is not accidentally reused.
if output_file.exists():
    output_file.unlink()


# --------------------------------------------------
# 4. Explicit PyArrow schema
# --------------------------------------------------

schema = pa.schema([
    ("Issue_ID", pa.int64()),
    ("Issue_Key", pa.string()),
    ("Project_ID", pa.int64()),
    ("Project_Name", pa.string()),
    ("Title", pa.string()),
    ("Description", pa.string()),
    ("Description_Text", pa.string()),
    ("Story_Point", pa.float64()),
    ("Type", pa.string()),
    ("Priority", pa.string()),
    ("Status", pa.string()),
    ("Creation_Date", pa.timestamp("us")),
    ("Estimation_Date", pa.timestamp("us")),
    ("Resolution_Date", pa.timestamp("us")),
    ("Last_Updated", pa.timestamp("us")),
    ("Story_Point_Changed_After_Estimation", pa.int64()),
    ("Description_Changed_After_Estimation", pa.int64()),
    ("Title_Changed_After_Estimation", pa.int64()),
    ("Sprint_ID", pa.float64()),
])


# --------------------------------------------------
# 5. Export in batches
# --------------------------------------------------

chunksize = 25000

total_rows = 0
batch_number = 0

writer = None

try:
    for chunk in pd.read_sql_query(query, engine, chunksize=chunksize):

        batch_number += 1

        # Normalize numeric columns to avoid inconsistent types between batches.
        numeric_columns = [
            "Issue_ID",
            "Project_ID",
            "Story_Point_Changed_After_Estimation",
            "Description_Changed_After_Estimation",
            "Title_Changed_After_Estimation",
        ]

        for column in numeric_columns:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce").astype("Int64")

        chunk["Story_Point"] = pd.to_numeric(chunk["Story_Point"], errors="coerce").astype("float64")
        chunk["Sprint_ID"] = pd.to_numeric(chunk["Sprint_ID"], errors="coerce").astype("float64")

        # Convert pandas DataFrame to Arrow Table
        table = pa.Table.from_pandas(
            chunk,
            schema=schema,
            preserve_index=False,
            safe=False,
        )

        # Initialize writer using the first batch
        if writer is None:
            writer = pq.ParquetWriter(output_file, schema, compression="snappy")

        writer.write_table(table)

        total_rows += len(chunk)

        print(f"Batch {batch_number}: {total_rows:,} rows exported")

finally:
    if writer is not None:
        writer.close()

    engine.dispose()


# --------------------------------------------------
# 6. Final report
# --------------------------------------------------

if total_rows == 0:
    output_file.unlink(missing_ok=True)
    raise RuntimeError("No rows were exported.")

print("\nExport completed successfully!")
print(f"Total rows: {total_rows:,}")
print(f"Total batches: {batch_number}")
print(f"Output file: {output_file}")
print(f"File size: {output_file.stat().st_size / (1024**2):.2f} MB")
