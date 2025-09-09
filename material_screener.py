#!/usr/bin/env python3
"""
Material Screening Tool for MOF CIF Files

This tool processes CIF files and filters materials based on the presence of
carbon and metal elements. It generates separate Excel reports for kept and
removed materials with detailed reasons for removal.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
from tqdm import tqdm
import logging

from curate import ase_format, ensure_data, has_carbon, has_metal, get_metal_elements

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MaterialScreener:
    """
    A tool for screening CIF files and filtering materials based on element composition.
    """
    
    def __init__(self):
        self.kept_materials = []
        self.removed_materials = []
        
    def process_cif_file(self, cif_path: Path) -> Tuple[bool, str, set]:
        """
        Process a single CIF file and determine if it should be kept or removed.
        
        Args:
            cif_path: Path to the CIF file
            
        Returns:
            Tuple of (keep_material, removal_reason, metal_elements)
        """
        material_name = cif_path.stem
        
        try:
            # Load the CIF file using curate functions
            atoms = ase_format(str(cif_path))
            
            if atoms is None:
                return False, "CIF reading failed", set()
            
            # Validate the data
            if not ensure_data(atoms):
                return False, "Invalid CIF data structure", set()
            
            # Check for carbon and metal presence
            has_c = has_carbon(atoms)
            has_m = has_metal(atoms)
            metal_elements = get_metal_elements(atoms) if has_m else set()
            
            # Determine filtering result
            if not has_c and not has_m:
                return False, "Missing both carbon and metal elements", set()
            elif not has_c:
                return False, "Missing carbon elements", metal_elements
            elif not has_m:
                return False, "Missing metal elements", set()
            else:
                # Material passes all filters
                return True, "Passed all filters", metal_elements
                
        except Exception as e:
            logger.error(f"Error processing {material_name}: {e}")
            return False, f"Processing error: {str(e)}", set()
    
    def screen_directory(self, cif_directory: str, output_directory: str = None) -> None:
        """
        Screen all CIF files in a directory and generate reports.
        
        Args:
            cif_directory: Directory containing CIF files
            output_directory: Directory to save Excel reports (default: same as cif_directory)
        """
        cif_dir = Path(cif_directory)
        if not cif_dir.exists():
            raise ValueError(f"Directory does not exist: {cif_directory}")
        
        if output_directory is None:
            output_directory = cif_directory
        
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all CIF files
        cif_files = list(cif_dir.glob("*.cif"))
        
        if not cif_files:
            logger.warning(f"No CIF files found in {cif_directory}")
            return
        
        logger.info(f"Found {len(cif_files)} CIF files to process")
        
        # Process files with progress bar
        for cif_file in tqdm(cif_files, desc="Screening materials"):
            keep_material, removal_reason, metal_elements = self.process_cif_file(cif_file)
            
            material_data = {
                'Material Name': cif_file.stem,
                'Removal Reason': removal_reason if not keep_material else 'N/A',
                'Metal Elements': ', '.join(sorted(metal_elements)) if metal_elements else 'None'
            }
            
            if keep_material:
                self.kept_materials.append(material_data)
            else:
                self.removed_materials.append(material_data)
        
        # Generate reports
        self._generate_reports(output_dir)
        
        # Print summary
        self._print_summary()
    
    def _generate_reports(self, output_dir: Path) -> None:
        """
        Generate Excel reports for kept and removed materials.
        
        Args:
            output_dir: Directory to save the reports
        """
        # Generate kept materials report
        if self.kept_materials:
            # For kept materials, we don't need the 'Removal Reason' column
            kept_data = []
            for material in self.kept_materials:
                kept_data.append({
                    'Material Name': material['Material Name'],
                    'Metal Elements': material['Metal Elements']
                })
            
            kept_df = pd.DataFrame(kept_data)
            kept_file = output_dir / "kept_materials.xlsx"
            
            with pd.ExcelWriter(kept_file, engine='openpyxl') as writer:
                kept_df.to_excel(writer, sheet_name='Kept Materials', index=False)
            
            logger.info(f"Kept materials report saved to: {kept_file}")
        else:
            logger.warning("No materials were kept - no kept materials report generated")
        
        # Generate removed materials report
        if self.removed_materials:
            # For removed materials, we need both removal reason and metal elements
            removed_data = []
            for material in self.removed_materials:
                removed_data.append({
                    'Material Name': material['Material Name'],
                    'Removal Reason': material['Removal Reason'],
                    'Metal Elements': material['Metal Elements'] if material['Metal Elements'] != 'None' else 'None'
                })
            
            removed_df = pd.DataFrame(removed_data)
            # Replace any NaN values with 'None'
            removed_df = removed_df.fillna('None')
            removed_file = output_dir / "removed_materials.xlsx"
            
            with pd.ExcelWriter(removed_file, engine='openpyxl') as writer:
                removed_df.to_excel(writer, sheet_name='Removed Materials', index=False)
            
            logger.info(f"Removed materials report saved to: {removed_file}")
        else:
            logger.warning("No materials were removed - no removed materials report generated")
    
    def _print_summary(self) -> None:
        """Print screening summary statistics."""
        total_materials = len(self.kept_materials) + len(self.removed_materials)
        kept_count = len(self.kept_materials)
        removed_count = len(self.removed_materials)
        
        print("\n" + "="*60)
        print("MATERIAL SCREENING SUMMARY")
        print("="*60)
        print(f"Total materials processed: {total_materials}")
        print(f"Materials kept: {kept_count} ({kept_count/total_materials*100:.1f}%)")
        print(f"Materials removed: {removed_count} ({removed_count/total_materials*100:.1f}%)")
        
        if self.removed_materials:
            print("\nRemoval reasons breakdown:")
            removal_counts = {}
            for material in self.removed_materials:
                reason = material['Removal Reason']
                removal_counts[reason] = removal_counts.get(reason, 0) + 1
            
            for reason, count in sorted(removal_counts.items()):
                print(f"  - {reason}: {count} materials")
        
        print("="*60)


def main():
    """Main function to run the material screener."""
    if len(sys.argv) < 2:
        print("Usage: python material_screener.py <cif_directory> [output_directory]")
        print("\nExample:")
        print("  python material_screener.py ./GCMC/CIF/")
        print("  python material_screener.py ./GCMC/CIF/ ./results/")
        sys.exit(1)
    
    cif_directory = sys.argv[1]
    output_directory = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        screener = MaterialScreener()
        screener.screen_directory(cif_directory, output_directory)
        
    except Exception as e:
        logger.error(f"Screening failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()