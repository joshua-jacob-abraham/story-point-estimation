# TAWOS: BERT-Based Story Point Estimation

## Group Members

| No. | Name                 | Roll Number |
| --- | -------------------- | ----------- |
| 1   | Johsua Jacob Abraham | B230362CS   |
| 2   | Ameer Shadab         | B230167CS   |
| 3   | Ayaan Sajid Nalakath | B230638CS   |
| 4   | Aswin S              | B230215CS   |
| 5   | Athul Sha            | B230222CS   |

## Problem Statement

Software development teams need to estimate the effort required to complete software development tasks. Story Points are commonly used in Agile project management to represent the relative effort, complexity, and uncertainty associated with a task.

However, assigning Story Points manually can be subjective, inconsistent, and dependent on the experience of individual team members. This project explores the use of machine learning and Natural Language Processing (NLP) techniques to automate Story Point estimation based on software issue descriptions.

## Objectives

- Explore and analyse the TAWOS dataset to understand its structure, characteristics, and Story Point distribution.
- Preprocess software issue data by handling missing values and preparing textual inputs.
- Generate contextual text representations using BERT embeddings.
- Develop regression models to predict Story Points from software issue descriptions.
- Evaluate model performance using Mean Absolute Error (MAE) and Median Absolute Error (MdAE).
- Investigate the effectiveness of BERT-based representations for software effort estimation.

## Table of Contents

1. [Dataset Description](#dataset-description)
2. [Methodology Overview](#methodology-overview)
3. [Repository Structure](#repository-structure)
4. [Setup Instructions](#setup-instructions)
5. [Dataset Preparation (SQL Dump to Parquet)](#dataset-preparation-sql-dump-to-parquet)
6. [Reproducing the Experiments](#reproducing-the-experiments)
7. [Configuration File](#configuration-file)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Results](#results)
10. [Reproducibility Notes](#reproducibility-notes)
11. [Technologies Used](#technologies-used)

---

## Dataset Description

This project uses the **TAWOS (The Agile Work and Open-source Software) dataset**, which contains software issue tracking information collected from software development projects.

The dataset includes information such as:

- **Issue ID:** Unique identifier associated with a software issue.
- **Project Name:** Name of the software project to which the issue belongs.
- **Issue Key:** Project-specific identifier of the issue.
- **Title:** Short summary of the software issue.
- **Description Text:** Detailed description of the issue.
- **Story Point:** Effort estimation label assigned to the issue.
- **Estimation Date:** Date associated with the estimation.
- **Resolution Date:** Date associated with issue resolution.

### Exported dataset (`TAWOS.parquet`)

The original dataset is distributed as a MySQL dump (`TAWOS.sql`, about 4.1 GB). We extract the required information into a single Parquet file.

| Property                          | Value                   |
| --------------------------------- | ----------------------- |
| Total records                     | 458,232                 |
| Total columns                     | 19                      |
| Total projects                    | 39                      |
| Issues with story point labels    | 65,427                  |
| Issues without story point labels | 392,805                 |
| Parquet file size                 | Approximately 318.75 MB |

The export retains the entire dataset. Issues with missing story point values are not removed at this stage.

### Dataset Preprocessing

The following preprocessing operations are performed (in the experiment notebook):

- Dataset exploration and statistical analysis.
- Identification of missing values and duplicate records.
- Retention of issues with available Story Point labels (458,232 → 65,427).
- Removal of three suspicious records with extreme Story Point values (`DATACASS-228`, `JSWSERVER-10638`, `JSWCLOUD-10638`).
- Removal of 354 within-project duplicate (Title, Description) pairs to prevent train/test text leakage.
- Splitting the labelled data **by project**: each of the 39 projects is treated as an independent within-project experiment.
- Combination of issue titles and descriptions into a single textual input (`Issue_Text`).
- Random splitting of each project's data into 80% development and 20% held-out test sets.

| Stage                                                    | Rows    |
| -------------------------------------------------------- | ------- |
| Full exported dataset                                    | 458,232 |
| After keeping labelled issues only                       | 65,427  |
| After removing 3 suspicious records                      | 65,424  |
| After removing 354 duplicate texts (final modelling set) | 65,070  |

No rows had both an empty title and an empty description, so no rows were removed by that check. Missing Story Points are never replaced with zero, because zero is a valid value in the data.

### Per-project data and split sizes

| #   | Project                             | Labelled issues | Development (80%) | Test (20%) |
| --- | ----------------------------------- | --------------- | ----------------- | ---------- |
| 1   | Alloy Framework                     | 315             | 252               | 63         |
| 2   | Apache MXNet                        | 208             | 166               | 42         |
| 3   | Apache Mesos                        | 3,271           | 2,616             | 655        |
| 4   | Apache Usergrid                     | 486             | 388               | 98         |
| 5   | Appcelerator Daemon                 | 242             | 193               | 49         |
| 6   | Appcelerator Studio                 | 3,402           | 2,721             | 681        |
| 7   | Aptana Studio                       | 890             | 712               | 178        |
| 8   | Atlassian Atlassian Software Server | 349             | 279               | 70         |
| 9   | Atlassian Bamboo                    | 526             | 420               | 106        |
| 10  | Atlassian Clover                    | 387             | 309               | 78         |
| 11  | Atlassian Confluence Cloud          | 352             | 281               | 71         |
| 12  | Atlassian Confluence Server         | 662             | 529               | 133        |
| 13  | Atlassian Crowd                     | 214             | 171               | 43         |
| 14  | Atlassian FishEye                   | 239             | 191               | 48         |
| 15  | Atlassian Jira Cloud                | 361             | 288               | 73         |
| 16  | Atlassian Jira Server               | 380             | 304               | 76         |
| 17  | Atlassian Software Cloud            | 317             | 253               | 64         |
| 18  | Command-Line Interface              | 374             | 299               | 75         |
| 19  | DotNetNuke Platform                 | 2,593           | 2,074             | 519        |
| 20  | Hyperledger Blockchain Explorer     | 373             | 298               | 75         |
| 21  | Hyperledger Fabric                  | 634             | 507               | 127        |
| 22  | Hyperledger Indy Node               | 681             | 544               | 137        |
| 23  | Hyperledger Indy SDK                | 720             | 576               | 144        |
| 24  | Hyperledger Sawtooth                | 957             | 765               | 192        |
| 25  | Lsstcorp Data management            | 20,357          | 16,285            | 4,072      |
| 26  | Lyrasis Dura Cloud                  | 666             | 532               | 134        |
| 27  | MongoDB Compass                     | 498             | 398               | 100        |
| 28  | MongoDB Core Server                 | 784             | 627               | 157        |
| 29  | Moodle                              | 1,593           | 1,274             | 319        |
| 30  | Mule                                | 4,166           | 3,332             | 834        |
| 31  | Mule APIkit                         | 473             | 378               | 95         |
| 32  | Sonatype Nexus                      | 1,844           | 1,475             | 369        |
| 33  | Spring DataCass                     | 241             | 192               | 49         |
| 34  | Spring XD                           | 3,702           | 2,961             | 741        |
| 35  | The Mongo C++ driver                | 224             | 179               | 45         |
| 36  | The Mongo Java driver               | 238             | 190               | 48         |
| 37  | The MongoDB Engineering             | 5,398           | 4,318             | 1,080      |
| 38  | The Titanium SDK                    | 4,663           | 3,730             | 933        |
| 39  | Titanium Mobile Platform            | 1,290           | 1,032             | 258        |
|     | **Total**                           | **65,070**      | **52,039**        | **13,031** |

### Dataset columns

The Parquet file contains the following 19 columns:

| Column                               | Description                                             |
| ------------------------------------ | ------------------------------------------------------- |
| Issue_ID                             | Unique issue identifier                                 |
| Issue_Key                            | Issue key                                               |
| Project_ID                           | Project identifier                                      |
| Project_Name                         | Project name                                            |
| Title                                | Issue title                                             |
| Description                          | Original issue description                              |
| Description_Text                     | Extracted description text                              |
| Story_Point                          | Story point estimation label                            |
| Type                                 | Issue type                                              |
| Priority                             | Issue priority                                          |
| Status                               | Issue status                                            |
| Creation_Date                        | Issue creation date                                     |
| Estimation_Date                      | Story point estimation date                             |
| Resolution_Date                      | Issue resolution date                                   |
| Last_Updated                         | Last update timestamp                                   |
| Story_Point_Changed_After_Estimation | Indicates whether story points changed after estimation |
| Description_Changed_After_Estimation | Indicates whether description changed after estimation  |
| Title_Changed_After_Estimation       | Indicates whether title changed after estimation        |
| Sprint_ID                            | Associated sprint identifier                            |

---

## Methodology Overview

1. **Input text:** `"Title: <title>\nDescription: <description_text>"`.
2. **Embeddings:** the frozen `bert-base-uncased` model produces token-level contextual embeddings (`last_hidden_state`), giving a `128 x 768` matrix per issue (max length 128, padded and truncated).
3. **Regressor:** a 3-block 1D CNN (`768 → 256 → 128 → 64` channels, kernel size 3) with max pooling, dropout 0.3, and a final linear layer outputs a single Story Point value. Negative predictions are clipped to 0.
4. **Training:** L1 loss (MAE), AdamW optimiser, up to 20 epochs, early stopping with patience 4 on validation MAE.
5. **Evaluation (per project):**
   - 5-fold cross-validation on the 80% development set.
   - A final model trained on the full development set and evaluated on the held-out 20% test set.
6. **Embedding fusion:** not applicable. A single BERT representation of the combined title and description is used.

---

## Repository Structure

```
story-point-estimation/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── filter_dump.py            # Step 1: keep only the 5 required tables from TAWOS.sql
├── export_tawos.py           # Step 2: MySQL -> output/TAWOS.parquet
├── validate_parquet.py       # Step 3: check the exported Parquet file
│
├── config/
│   └── experiment_config.json   # Model, hyperparameters, seeds, split settings
│
├── notebooks/
│   └── notebook.ipynb           # Experiment notebook (Kaggle)
│
├── data/                     # TAWOS.sql, TAWOS_required.sql (not committed)
│   └── .gitkeep
│
└── output/                   # TAWOS.parquet (not committed)
    └── .gitkeep
```

The SQL dump, filtered SQL file, and generated Parquet file are large local data files and are **not** committed to Git (see `.gitignore`).

---

## Setup Instructions

### Requirements

- Python 3.10 or later
- Docker (only for rebuilding the Parquet file from the SQL dump)
- MySQL 8.0 (provided through the Docker image)
- A CUDA-capable GPU is recommended for the experiments. The notebook was run on Kaggle with a GPU (`torch.amp.autocast("cuda")` is used when extracting embeddings).

The data extraction was performed on EndeavourOS Linux; the commands below are for Linux or macOS shells.

### Python environment

```bash
git clone <REPOSITORY_URL>
cd story-point-estimation

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` is not version-pinned. After a successful run, record the exact versions with `pip freeze > requirements-lock.txt`.

---

## Dataset Preparation (SQL Dump to Parquet)

You can skip this section if you already have `TAWOS.parquet`. The experiments were run on Kaggle using a copy of the Parquet file uploaded as the dataset `ayaansajidnalakath/tawos-cleaned`.

The pipeline is:

1. Start with the original `TAWOS.sql` dump (about 4.1 GB).
2. Run `filter_dump.py` to retain the five required tables.
3. Import the filtered SQL into a local MySQL database running in Docker.
4. Run `export_tawos.py` to extract the required columns in batches of 25,000.
5. Save the resulting `TAWOS.parquet` file.
6. Validate the exported dataset with `validate_parquet.py`.

### Step 1: Place the original SQL dump

Obtain the original TAWOS SQL dump and place it at:

```
data/TAWOS.sql
```

The `data/` and `output/` directories are included in the repository (as empty folders). If they are missing, create them:

```bash
mkdir -p data output
```

### Step 2: Filter the SQL dump

The original dump contains tables that are not needed here. `filter_dump.py` keeps only:

- `Issue`
- `Project`
- `Repository`
- `Sprint`
- `User`

It reads the file line by line, so the 4.1 GB dump is never loaded into memory.

```bash
python filter_dump.py
```

Output: `data/TAWOS_required.sql` (about 0.96 GB). The script stops with an error if any required table is missing.

### Step 3: Start MySQL with Docker

```bash
docker pull mysql:8.0

docker volume create tawos_mysql_data

docker run -d \
  --name tawos-mysql \
  --restart unless-stopped \
  -e MYSQL_ROOT_PASSWORD=tawos_local_password \
  -e MYSQL_DATABASE=tawos_cleaned \
  -p 127.0.0.1:3307:3306 \
  -v tawos_mysql_data:/var/lib/mysql \
  mysql:8.0 \
  --max-allowed-packet=256M \
  --innodb-buffer-pool-size=2G
```

The database is then reachable with:

| Setting  | Value                |
| -------- | -------------------- |
| Host     | 127.0.0.1            |
| Port     | 3307                 |
| Database | tawos_cleaned        |
| Username | root                 |
| Password | tawos_local_password |

These are local development credentials only, bound to `127.0.0.1`. If you change the password, change it in both the `docker run` command and the connection string used by `export_tawos.py`. The script also reads the `TAWOS_DATABASE_URL` environment variable, for example:

```bash
export TAWOS_DATABASE_URL="mysql+pymysql://root:<password>@127.0.0.1:3307/tawos_cleaned"
```

### Step 4: Import the filtered dump

Wait until MySQL has finished starting (check with `docker logs tawos-mysql`), then:

```bash
docker exec -i tawos-mysql \
  mysql -uroot -ptawos_local_password tawos_cleaned \
  < data/TAWOS_required.sql
```

Verify the imported tables:

```bash
docker exec -it tawos-mysql \
  mysql -uroot -ptawos_local_password tawos_cleaned \
  -e "SHOW TABLES;"
```

The database should contain the five required tables.

### Step 5: Export to Parquet

`export_tawos.py` connects to the local MySQL database and:

1. Joins the `Issue` table with the `Project` table.
2. Selects the 19 required columns.
3. Orders the records by issue ID.
4. Retrieves records in batches of 25,000.
5. Applies an explicit PyArrow schema so column types stay consistent across batches. This matters because `Sprint_ID` was once inferred as a null type in one batch and as a float in another, which broke the writer.
6. Writes all batches into a single Snappy-compressed Parquet file.

```bash
python export_tawos.py
```

Output: `output/TAWOS.parquet`. A successful run produced 458,232 rows in 19 batches (about 318.75 MB).

### Step 6: Validate the export

```bash
python validate_parquet.py
```

Expected output:

```
[OK  ] rows           expected=  458,232  actual=  458,232
[OK  ] columns        expected=       19  actual=       19
[OK  ] unique_issues  expected=  458,232  actual=  458,232
[OK  ] projects       expected=       39  actual=       39
[OK  ] labelled       expected=   65,427  actual=   65,427
[OK  ] unlabelled     expected=  392,805  actual=  392,805
```

The script exits with a non-zero status if any value differs.

---

## Reproducing the Experiments

The experiments are in `notebooks/notebook.ipynb`. It processes all 39 projects in a single run and was developed on Kaggle with a GPU.

### Option A: Kaggle (as originally run)

1. Upload `TAWOS.parquet` as a Kaggle dataset, or add the existing dataset `ayaansajidnalakath/tawos-cleaned` to the notebook.
2. Import `notebooks/notebook.ipynb` into a new Kaggle notebook.
3. Enable a GPU accelerator and internet access (needed to download `bert-base-uncased`).
4. Confirm that `DATA_PATH` in the second cell matches the dataset location (default: `/kaggle/input/datasets/ayaansajidnalakath/tawos-cleaned/TAWOS.parquet`).
5. Run all cells.

### Option B: Local machine

1. Complete [Setup Instructions](#setup-instructions) and generate `output/TAWOS.parquet` (see the previous section).
2. Open `notebooks/notebook.ipynb` in Jupyter.
3. Edit the hard-coded Kaggle paths:
   - Input: `DATA_PATH` → `output/TAWOS.parquet`.
   - Outputs: replace `/kaggle/working/` with a local folder such as `output/`.
4. Make sure a CUDA GPU is available, or expect a very slow CPU run.
5. Run all cells.

### What the notebook does

| Notebook stage       | Description                                                                                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Load and explore     | Loads the Parquet file and prints dataset statistics, missing values, duplicates, and per-project summaries                                             |
| Keep labelled issues | Keeps rows with a Story Point; saves `TAWOS_labelled.parquet`                                                                                           |
| Clean                | Drops rows with empty title and description; removes the 3 suspicious records and 354 within-project duplicates; saves `TAWOS_cleaned_labelled.parquet` |
| Per-project data     | Builds one DataFrame per project (39 projects) and creates `Issue_Text`                                                                                 |
| Split                | 80/20 development/test split for each project (`random_state=42`), with leakage and size assertions                                                     |
| Embeddings           | Extracts frozen BERT token embeddings (`128 x 768`)                                                                                                     |
| Train and evaluate   | For each project: 5-fold CV on the development set, then a final model evaluated on the held-out test set                                               |
| Save and plot        | Writes result CSVs, the config JSON, and a results plot                                                                                                 |

### Expected outputs

The notebook writes the following files:

| File                               | Content                                                  |
| ---------------------------------- | -------------------------------------------------------- |
| `ps1_bert_cnn_5fold_details.csv`   | MAE and MdAE for every project and fold                  |
| `ps1_bert_cnn_summary_results.csv` | Per-project CV mean/std and held-out test MAE and MdAE   |
| `ps1_experiment_config.json`       | The hyperparameter dictionary used for the run           |
| `ps1_bert_cnn_results_plot.png`    | Bar/scatter comparison of CV and test errors per project |
| `TAWOS_labelled.parquet`           | Labelled issues only                                     |
| `TAWOS_cleaned_labelled.parquet`   | Final cleaned modelling dataset (65,070 rows)            |

Approximate expected shapes: 458,232 → 65,427 → 65,070 rows; 39 projects; 52,039 development and 13,031 test issues in total.

### Determinism

Seeds are set for Python, NumPy, and PyTorch (including CUDA), and cuDNN is set to deterministic mode. The train/test split and the cross-validation splits both use `random_state=42`. Minor numerical differences can still occur across GPU types, driver versions, and library versions, because embeddings are computed with mixed precision (autocast) and stored as float16.

---

## Configuration File

All experiment settings are in [`config/experiment_config.json`](config/experiment_config.json). It mirrors the settings used in the notebook and extends them with the data, split, embedding, model, and training details.

| Section           | What it specifies                                                                                                                                                      |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `seed`            | Global random seed (42)                                                                                                                                                |
| `hyperparameters` | The exact `CONFIG` dictionary used in the notebook (model name, batch sizes, epochs, learning rate, weight decay, patience, folds, CNN channels, kernel size, dropout) |
| `data`            | Input and output files, column names, text template, excluded issue keys, deduplication rule                                                                           |
| `split`           | Within-project strategy, test size (0.20), shuffle, `random_state`, and the 5-fold CV settings                                                                         |
| `embedding`       | Model name (`bert-base-uncased`), frozen BERT, token-level output, max length 128, mixed precision, float16 storage, and fusion settings                               |
| `model`           | `StoryPointCNN1D` layer specification                                                                                                                                  |
| `training`        | L1 loss, AdamW, early stopping rules, how the final model is trained                                                                                                   |
| `evaluation`      | Metrics (MAE, MdAE) and aggregation across projects                                                                                                                    |

Key values:

| Setting                 | Value                        |
| ----------------------- | ---------------------------- |
| Model                   | `bert-base-uncased` (frozen) |
| Max sequence length     | 128                          |
| Embedding batch size    | 64                           |
| Training batch size     | 32                           |
| Epochs                  | 20                           |
| Learning rate           | 1e-3                         |
| Weight decay            | 1e-4                         |
| Early-stopping patience | 4 (CV folds only)            |
| CNN channels            | 768 → 256 → 128 → 64         |
| Kernel size             | 3                            |
| Dropout                 | 0.3                          |
| CV folds                | 5                            |
| Test size               | 0.20                         |
| Random seed             | 42                           |
| Embedding fusion        | None (not applicable)        |

To use the config in the notebook, replace the hard-coded `CONFIG` dictionary with:

```python
import json

with open("config/experiment_config.json") as f:
    full_config = json.load(f)

CONFIG = full_config["hyperparameters"]   # same keys as the original notebook CONFIG
```

Note: the notebook currently defines `CONFIG` inline, and the architecture shown in the `model` section is fixed in code. Changing `cnn_channels` or `kernel_size` in the JSON has no effect unless the `StoryPointCNN1D` class is edited to read those values.

---

## Evaluation Metrics

- **Mean Absolute Error (MAE):** Measures the average absolute difference between actual and predicted Story Points.
- **Median Absolute Error (MdAE):** Measures the median absolute difference between actual and predicted Story Points.

Both are computed per project, then summarised across the 39 projects.

---

## Results

Results from the notebook run (mean and median across 39 projects):

| Metric                                        | Value  |
| --------------------------------------------- | ------ |
| 5-Fold validation MAE (mean across projects)  | 2.2537 |
| 5-Fold validation MdAE (mean across projects) | 1.1362 |
| Held-out test MAE (mean across projects)      | 3.2194 |
| Held-out test MdAE (mean across projects)     | 1.2320 |
| Held-out test MAE (median across projects)    | 1.6789 |
| Held-out test MdAE (median across projects)   | 1.1064 |

The mean test MAE (3.22) is much higher than the median (1.68) mainly because of one project: **Atlassian Jira Cloud** has a held-out test MAE of 43.08 (with a test MdAE of only 1.17), which indicates a few extreme Story Point values in its 20% test split. Per-project results are in `ps1_bert_cnn_summary_results.csv`.

---

## Reproducibility Notes

- The original SQL dump is required to rebuild `TAWOS.parquet` from scratch.
- The filtering and export scripts are included in the repository.
- The Docker volume `tawos_mysql_data` stores the imported MySQL database.
- The Parquet export uses batches of 25,000 records and an explicit Arrow schema to avoid inconsistent data types between batches.
- The exported dataset is preserved in full; supervised-learning subsets are created separately in the notebook.
- The experiment notebook uses fixed seeds (42) for the train/test split, the CV folds, and model training.
- `export_tawos.py` should be compared against the original working copy used for the first extraction (in particular the SQL column names and aliases) before the repository is finalised.

---

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- PyTorch
- Hugging Face Transformers (BERT)
- PyArrow, SQLAlchemy, PyMySQL
- Docker, MySQL 8.0
- Matplotlib
- Jupyter Notebook / Kaggle
