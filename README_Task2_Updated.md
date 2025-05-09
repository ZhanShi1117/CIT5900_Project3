# Task 2 – Data Enrichment

This folder contains the script and output for **Task 2: Data Enrichment**, as part of the CIT5900 Project 3.

## Objective

The goal of this task is to enrich the cleaned dataset (`combined_outputs.csv`) with the required metadata fields as specified in the `ResearchOutputs.xlsx` template. This enrichment process includes:

- Adding missing required columns (e.g., `OutputVenue`, `OutputYear`, `ProjectPI`, etc.)
- Filling in missing values with defaults (e.g., "Published", "Completed")
- Formatting `OutputYear` and `OutputMonth` as numerical values
- Cleaning all string-based columns
- Producing a unified output for downstream analysis

## Files Included

| File                          | Description                                   |
|-------------------------------|-----------------------------------------------|
| `task2_data_enrichment.py`   | Python script that performs data enrichment   |
| `enriched_outputs_sample.csv`| Output file containing the enriched dataset   |
| `combined_outputs.csv`       | Input file merged and deduplicated in Task 1  |

## How to Run

Make sure you have Python 3 installed with the `pandas` package.

```bash
pip install pandas
python scripts/task2_data_enrichment.py
```

The enriched CSV will be saved to:
```
data/processed/enriched_outputs_sample.csv
```

## Dependencies

- pandas
- os (built-in)

You may optionally create a `requirements.txt` file with the following content:

```
pandas
```

## Notes

- This script assumes `combined_outputs.csv` is already cleaned and deduplicated by Task 1.
- Update the file paths as needed if running from a different working directory.

## 📌 Task 2 – Data Enrichment

This task enriches the deduplicated dataset by filling missing metadata fields based on authoritative project metadata sources.

### ✔ Steps Taken:

1. **Loaded Base Dataset**: Used `Final_ResearchOutputs_Cleaned.csv` containing deduplicated research outputs.
2. **Metadata Enrichment**:
   - Merged with `ProjectsAllMetadata.xlsx` using `ProjID`.
   - Backfilled missing values in `ProjectStatus`, `ProjectTitle`, `ProjectPI`, `ProjectRDC`, `ProjectYearStarted`, and `ProjectYearEnded`.
3. **Irrelevant Record Filtering**:
   - Removed rows not clearly associated with FSRDC based on keywords in `ProjectTitle` or `ProjectRDC`.
4. **Standardization**:
   - Ensured all 17 required fields are present.
   - Replaced empty fields with appropriate defaults (e.g., `"Completed"` for `ProjectStatus`).
   - Standardized dates using `pd.to_datetime()`.
5. **Output**:
   - Saved enriched dataset to `ResearchOutputs_GroupX_Task2_Enriched.csv`.
   - Cleaned strings and ensured consistency across all records.

### 🛠 Tools Used:
- Python 3, Pandas
- Excel (metadata reference)
- Fuzzy matching and logic-based filtering (for enrichment and cleanup)

> Final output is located in: `./data/processed/ResearchOutputs_GroupX_Task2_Enriched.csv`