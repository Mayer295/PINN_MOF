"""
CIF File Processing and Curation Module

This module provides functions for processing CIF files with robust error handling
for common CIF reading issues like "CIF loop ended unexpectedly" errors.
"""

import warnings
from pathlib import Path
from typing import Optional, Tuple, Set
from ase import Atoms
from ase.io import read
import logging

# Suppress ASE warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='ase')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ase_format(cif_path: str) -> Optional[Atoms]:
    """
    Read and preprocess CIF file using ASE with robust error handling.
    
    This function attempts to read CIF files while handling common parsing errors
    such as "CIF loop ended unexpectedly" and malformed CIF structures.
    
    Args:
        cif_path: Path to the CIF file
        
    Returns:
        ASE Atoms object if successful, None if reading fails
    """
    try:
        # First attempt: standard CIF reading
        atoms = read(cif_path, format='cif')
        return atoms
        
    except Exception as e:
        error_str = str(e).lower()
        
        # Handle "CIF loop ended unexpectedly" error
        if "cif loop ended unexpectedly" in error_str or "incomplete row" in error_str:
            logger.warning(f"CIF loop error in {cif_path}: {e}")
            
            # Attempt to read with different ASE CIF parser settings
            try:
                # Try reading with store_tags=False to ignore malformed metadata
                atoms = read(cif_path, format='cif', store_tags=False)
                return atoms
            except Exception as e2:
                logger.warning(f"Second attempt failed for {cif_path}: {e2}")
                
                # Try reading only the first structure if multiple exist
                try:
                    atoms = read(cif_path, format='cif', index=0)
                    return atoms
                except Exception as e3:
                    logger.error(f"All attempts failed for {cif_path}: {e3}")
                    return None
        
        # Handle other CIF parsing errors
        elif "cif" in error_str or "parse" in error_str:
            logger.warning(f"CIF parsing error in {cif_path}: {e}")
            
            # Try alternative parsing approach
            try:
                atoms = read(cif_path, format='cif', index=':')
                if isinstance(atoms, list) and len(atoms) > 0:
                    return atoms[0]
                return atoms
            except Exception as e2:
                logger.error(f"Alternative parsing failed for {cif_path}: {e2}")
                return None
        
        else:
            logger.error(f"Unexpected error reading {cif_path}: {e}")
            return None


def ensure_data(atoms: Atoms) -> bool:
    """
    Ensure CIF file data is valid and properly formatted.
    
    This function checks if the loaded atomic structure has valid data
    including atomic positions, symbols, and other essential properties.
    
    Args:
        atoms: ASE Atoms object to validate
        
    Returns:
        True if data is valid, False otherwise
    """
    if atoms is None:
        return False
    
    try:
        # Check if atoms object has basic required attributes
        if len(atoms) == 0:
            logger.warning("Empty atoms object")
            return False
        
        # Check if atomic symbols are present
        symbols = atoms.get_chemical_symbols()
        if not symbols or len(symbols) == 0:
            logger.warning("No atomic symbols found")
            return False
        
        # Check if positions are present and valid
        positions = atoms.get_positions()
        if positions is None or len(positions) == 0:
            logger.warning("No atomic positions found")
            return False
        
        # Check for NaN or infinite values in positions
        import numpy as np
        if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
            logger.warning("Invalid atomic positions (NaN or infinite values)")
            return False
        
        # Check if cell parameters are reasonable
        cell = atoms.get_cell()
        if cell is not None:
            cell_lengths = atoms.get_cell_lengths_and_angles()[:3]
            if any(length <= 0 for length in cell_lengths):
                logger.warning("Invalid cell parameters")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating atoms object: {e}")
        return False


def get_elements(atoms: Atoms) -> Set[str]:
    """
    Extract unique chemical elements from the atoms object.
    
    Args:
        atoms: ASE Atoms object
        
    Returns:
        Set of element symbols present in the structure
    """
    if atoms is None:
        return set()
    
    try:
        symbols = atoms.get_chemical_symbols()
        return set(symbols)
    except Exception as e:
        logger.error(f"Error extracting elements: {e}")
        return set()


def has_carbon(atoms: Atoms) -> bool:
    """
    Check if the structure contains carbon atoms.
    
    Args:
        atoms: ASE Atoms object
        
    Returns:
        True if carbon is present, False otherwise
    """
    elements = get_elements(atoms)
    return 'C' in elements


def has_metal(atoms: Atoms) -> bool:
    """
    Check if the structure contains metal atoms.
    
    Args:
        atoms: ASE Atoms object
        
    Returns:
        True if metal elements are present, False otherwise
    """
    elements = get_elements(atoms)
    
    # Define common metal elements found in MOFs
    metals = {
        'Li', 'Na', 'K', 'Rb', 'Cs',  # Alkali metals
        'Be', 'Mg', 'Ca', 'Sr', 'Ba',  # Alkaline earth metals
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',  # 3d transition metals
        'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',  # 4d transition metals
        'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',  # 5d transition metals
        'Al', 'Ga', 'In', 'Tl', 'Sn', 'Pb', 'Bi',  # Post-transition metals
        'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',  # Lanthanides
        'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'  # Actinides
    }
    
    return bool(elements.intersection(metals))


def get_metal_elements(atoms: Atoms) -> Set[str]:
    """
    Extract metal elements from the atoms object.
    
    Args:
        atoms: ASE Atoms object
        
    Returns:
        Set of metal element symbols present in the structure
    """
    elements = get_elements(atoms)
    
    # Define common metal elements found in MOFs (same as in has_metal function)
    metals = {
        'Li', 'Na', 'K', 'Rb', 'Cs',  # Alkali metals
        'Be', 'Mg', 'Ca', 'Sr', 'Ba',  # Alkaline earth metals
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',  # 3d transition metals
        'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',  # 4d transition metals
        'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',  # 5d transition metals
        'Al', 'Ga', 'In', 'Tl', 'Sn', 'Pb', 'Bi',  # Post-transition metals
        'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',  # Lanthanides
        'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'  # Actinides
    }
    
    return elements.intersection(metals)


if __name__ == "__main__":
    # Test the functions with sample CIF files
    import sys
    
    if len(sys.argv) > 1:
        cif_file = sys.argv[1]
        print(f"Testing CIF file: {cif_file}")
        
        atoms = ase_format(cif_file)
        if atoms is not None:
            print(f"Successfully loaded: {len(atoms)} atoms")
            
            if ensure_data(atoms):
                print("Data validation: PASSED")
                elements = get_elements(atoms)
                print(f"Elements found: {sorted(elements)}")
                print(f"Has carbon: {has_carbon(atoms)}")
                print(f"Has metal: {has_metal(atoms)}")
                if has_metal(atoms):
                    metals = get_metal_elements(atoms)
                    print(f"Metal elements: {sorted(metals)}")
            else:
                print("Data validation: FAILED")
        else:
            print("Failed to load CIF file")
    else:
        print("Usage: python curate.py <cif_file>")