# Material Screening Tool for MOF CIF Files

This tool provides robust CIF file processing and material screening capabilities for Metal-Organic Framework (MOF) structures. It handles common CIF reading errors and generates comprehensive Excel reports for material analysis.

## Features

- **Robust CIF Processing**: Handles "CIF loop ended unexpectedly" errors and other common CIF parsing issues
- **Element-Based Filtering**: Screens materials based on carbon and metal element presence
- **Progress Tracking**: Real-time progress bars during processing
- **Excel Reports**: Generates separate reports for kept and removed materials
- **Comprehensive Error Handling**: Detailed logging and error recovery mechanisms

## Requirements

- Python 3.6+
- ASE (Atomic Simulation Environment)
- pandas
- tqdm
- openpyxl

## Installation

Install required dependencies:

```bash
pip install ase pandas tqdm openpyxl
```

## Usage

### Basic Usage

Screen all CIF files in a directory:

```bash
python material_screener.py /path/to/cif/directory/
```

### Specify Output Directory

```bash
python material_screener.py /path/to/cif/directory/ /path/to/output/directory/
```

### Example with Sample Files

```bash
python material_screener.py GCMC/CIF/
```

## Output

The tool generates two Excel files:

1. **kept_materials.xlsx**: Materials that passed all filters
   - Material Name
   - Metal Elements

2. **removed_materials.xlsx**: Materials that were filtered out
   - Material Name
   - Removal Reason
   - Metal Elements

## Filtering Criteria

Materials are removed if they:

- **Missing carbon elements**: No carbon atoms found in the structure
- **Missing metal elements**: No metal atoms found in the structure  
- **Missing both elements**: Neither carbon nor metal atoms found
- **CIF reading failed**: File could not be parsed due to format errors
- **Invalid data structure**: Loaded data failed validation checks

## Error Handling

The tool includes robust error handling for common CIF file issues:

- **CIF loop ended unexpectedly**: Attempts multiple parsing strategies
- **Incomplete rows**: Tries alternative ASE reader configurations
- **Malformed metadata**: Ignores problematic tags when possible
- **Empty or corrupted files**: Graceful failure with detailed logging

## Sample Output

```
INFO:__main__:Found 8 CIF files to process
Screening materials: 100%|██████████| 8/8 [01:30<00:00, 11.27s/it]
INFO:__main__:Kept materials report saved to: GCMC/CIF/kept_materials.xlsx
INFO:__main__:Removed materials report saved to: GCMC/CIF/removed_materials.xlsx

============================================================
MATERIAL SCREENING SUMMARY
============================================================
Total materials processed: 8
Materials kept: 5 (62.5%)
Materials removed: 3 (37.5%)

Removal reasons breakdown:
  - Missing metal elements: 3 materials
============================================================
```

## Module Functions

### curate.py

Core functions for CIF processing:

- `ase_format(cif_path)`: Robust CIF file loading with error recovery
- `ensure_data(atoms)`: Validate loaded atomic structure data
- `has_carbon(atoms)`: Check for carbon element presence
- `has_metal(atoms)`: Check for metal element presence
- `get_metal_elements(atoms)`: Extract metal element symbols

### material_screener.py

Main screening tool with:

- `MaterialScreener`: Main class for batch processing
- Progress tracking with tqdm
- Excel report generation
- Comprehensive statistics and summaries

## Technical Details

### Supported Metal Elements

The tool recognizes a comprehensive list of metals commonly found in MOFs:

- Alkali metals: Li, Na, K, Rb, Cs
- Alkaline earth metals: Be, Mg, Ca, Sr, Ba
- Transition metals: Sc, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, etc.
- Post-transition metals: Al, Ga, In, Tl, Sn, Pb, Bi
- Lanthanides and Actinides

### CIF Error Recovery

Multiple strategies are employed for problematic CIF files:

1. Standard ASE CIF reader
2. Alternative parsing with `store_tags=False`
3. Single structure extraction with `index=0`
4. Graceful failure with detailed error logging

## License

Academic use only. See main repository license for details.