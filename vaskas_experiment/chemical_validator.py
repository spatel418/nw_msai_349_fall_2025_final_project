"""
Chemical Validity Checker
Uses RDKit and domain knowledge to validate synthetic molecular data
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import warnings

# Try to import RDKit (optional but recommended)
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    warnings.warn("RDKit not available. Chemical validity checks will be limited.")

class ChemicalValidator:
    """
    Validates synthetic molecular data for chemical plausibility
    """
    
    def __init__(self, feature_names: List[str], strict=False):
        """
        Parameters:
        -----------
        feature_names : list of str
            Names of features in the dataset
        strict : bool
            If True, apply stricter validation rules
        """
        self.feature_names = feature_names
        self.strict = strict
        self.feature_indices = {name: i for i, name in enumerate(feature_names)}
        
    def validate_batch(self, X_synthetic: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Validate a batch of synthetic samples
        
        Returns:
        --------
        valid_mask : np.ndarray (bool)
            Boolean mask indicating which samples are valid
        validation_report : list of str
            Report of validation results
        """
        n_samples = len(X_synthetic)
        valid_mask = np.ones(n_samples, dtype=bool)
        validation_report = []
        
        # Run all validation checks
        checks = [
            self._check_basic_properties,
            self._check_electronic_properties,
            self._check_physical_constraints,
            self._check_feature_relationships
        ]
        
        for check_func in checks:
            check_mask, check_report = check_func(X_synthetic)
            valid_mask &= check_mask
            validation_report.extend(check_report)
        
        # Summary
        n_valid = valid_mask.sum()
        n_invalid = n_samples - n_valid
        
        summary = [
            f"\n{'='*80}",
            "CHEMICAL VALIDATION SUMMARY",
            f"{'='*80}",
            f"Total synthetic samples: {n_samples}",
            f"Valid samples: {n_valid} ({100*n_valid/n_samples:.1f}%)",
            f"Invalid samples: {n_invalid} ({100*n_invalid/n_samples:.1f}%)",
        ]
        
        validation_report = summary + validation_report
        
        return valid_mask, validation_report
    
    def _get_feature_values(self, X: np.ndarray, feature_name: str) -> np.ndarray:
        """Get values for a specific feature"""
        if feature_name in self.feature_indices:
            return X[:, self.feature_indices[feature_name]]
        return None
    
    def _check_basic_properties(self, X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Check basic molecular properties"""
        valid_mask = np.ones(len(X), dtype=bool)
        report = ["\n--- Basic Properties Check ---"]
        
        # Check Mass (should be between 12 and 2000 Da for organic molecules)
        mass = self._get_feature_values(X, 'Mass')
        if mass is not None:
            mass_min, mass_max = 12.0, 2000.0 if not self.strict else 1000.0
            mass_valid = (mass >= mass_min) & (mass <= mass_max)
            valid_mask &= mass_valid
            n_invalid = (~mass_valid).sum()
            if n_invalid > 0:
                report.append(f"  ⚠️  {n_invalid} samples with invalid mass (not in [{mass_min}, {mass_max}] Da)")
        
        # Check number of atoms features
        atom_features = ['NumAtoms', 'num_atoms', 'atoms']
        for feat in atom_features:
            atoms = self._get_feature_values(X, feat)
            if atoms is not None:
                atoms_valid = (atoms >= 1) & (atoms <= 500)
                valid_mask &= atoms_valid
                n_invalid = (~atoms_valid).sum()
                if n_invalid > 0:
                    report.append(f"  ⚠️  {n_invalid} samples with invalid number of atoms")
                break
        
        # Check heteroatoms (should be <= total atoms)
        hetero = self._get_feature_values(X, 'NumHeteroatoms')
        if hetero is not None:
            hetero_valid = (hetero >= 0) & (hetero <= 100)
            valid_mask &= hetero_valid
            n_invalid = (~hetero_valid).sum()
            if n_invalid > 0:
                report.append(f"  ⚠️  {n_invalid} samples with invalid heteroatom count")
        
        # Check hydrogen bond donors/acceptors
        donors = self._get_feature_values(X, 'HDonors')
        if donors is not None:
            donors_valid = (donors >= 0) & (donors <= 20)
            valid_mask &= donors_valid
            n_invalid = (~donors_valid).sum()
            if n_invalid > 0:
                report.append(f"  ⚠️  {n_invalid} samples with invalid H-bond donors")
        
        acceptors = self._get_feature_values(X, 'HAcceptors')
        if acceptors is not None:
            acceptors_valid = (acceptors >= 0) & (acceptors <= 30)
            valid_mask &= acceptors_valid
            n_invalid = (~acceptors_valid).sum()
            if n_invalid > 0:
                report.append(f"  ⚠️  {n_invalid} samples with invalid H-bond acceptors")
        
        if valid_mask.all():
            report.append("  ✓ All samples passed basic properties check")
        
        return valid_mask, report
    
    def _check_electronic_properties(self, X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Check electronic properties (HOMO, LUMO, etc.)"""
        valid_mask = np.ones(len(X), dtype=bool)
        report = ["\n--- Electronic Properties Check ---"]
        
        # Check HOMO-LUMO gap (must be positive)
        homo = self._get_feature_values(X, 'HOMO(eV)')
        lumo = self._get_feature_values(X, 'LUMO(eV)')
        
        if homo is not None and lumo is not None:
            gap = lumo - homo
            gap_valid = gap > 0
            valid_mask &= gap_valid
            n_invalid = (~gap_valid).sum()
            if n_invalid > 0:
                report.append(f"  ⚠️  {n_invalid} samples with negative/zero HOMO-LUMO gap (physically impossible)")
            
            # Check if gap is reasonable (typically 0.5 to 15 eV for organic molecules)
            if self.strict:
                gap_range_valid = (gap >= 0.5) & (gap <= 15.0)
                valid_mask &= gap_range_valid
                n_invalid = (~gap_range_valid).sum()
                if n_invalid > 0:
                    report.append(f"  ⚠️  {n_invalid} samples with unrealistic HOMO-LUMO gap")
        
        # Check HOMO energy (typically between -12 and -3 eV)
        if homo is not None:
            homo_valid = (homo >= -15.0) & (homo <= 0.0)
            valid_mask &= homo_valid
            n_invalid = (~homo_valid).sum()
            if n_invalid > 0:
                report.append(f"  ⚠️  {n_invalid} samples with unrealistic HOMO energy")
        
        # Check LUMO energy (typically between -6 and 3 eV)
        if lumo is not None:
            lumo_valid = (lumo >= -8.0) & (lumo <= 5.0)
            valid_mask &= lumo_valid
            n_invalid = (~lumo_valid).sum()
            if n_invalid > 0:
                report.append(f"  ⚠️  {n_invalid} samples with unrealistic LUMO energy")
        
        # Check dipole moment (should be positive, typically < 15 Debye)
        dipole = self._get_feature_values(X, 'DipoleMoment(Debye)')
        if dipole is not None:
            dipole_valid = (dipole >= 0) & (dipole <= 20.0)
            valid_mask &= dipole_valid
            n_invalid = (~dipole_valid).sum()
            if n_invalid > 0:
                report.append(f"  ⚠️  {n_invalid} samples with invalid dipole moment")
        
        if valid_mask.all():
            report.append("  ✓ All samples passed electronic properties check")
        
        return valid_mask, report
    
    def _check_physical_constraints(self, X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Check physical constraints (LogP, etc.)"""
        valid_mask = np.ones(len(X), dtype=bool)
        report = ["\n--- Physical Constraints Check ---"]
        
        # Check LogP (partition coefficient, typically -5 to 10 for drug-like molecules)
        logp = self._get_feature_values(X, 'LogP')
        if logp is not None:
            logp_min, logp_max = -7.0, 12.0 if not self.strict else 8.0
            logp_valid = (logp >= logp_min) & (logp <= logp_max)
            valid_mask &= logp_valid
            n_invalid = (~logp_valid).sum()
            if n_invalid > 0:
                report.append(f"  ⚠️  {n_invalid} samples with unrealistic LogP")
        
        # Check rotatable bonds (typically 0-30 for organic molecules)
        rot_bonds = self._get_feature_values(X, 'NumRotatableBonds')
        if rot_bonds is not None:
            rot_valid = (rot_bonds >= 0) & (rot_bonds <= 50)
            valid_mask &= rot_valid
            n_invalid = (~rot_valid).sum()
            if n_invalid > 0:
                report.append(f"  ⚠️  {n_invalid} samples with invalid rotatable bonds count")
        
        # Check ring count (typically 0-10 for organic molecules)
        rings = self._get_feature_values(X, 'RingCount')
        if rings is not None:
            rings_valid = (rings >= 0) & (rings <= 15)
            valid_mask &= rings_valid
            n_invalid = (~rings_valid).sum()
            if n_invalid > 0:
                report.append(f"  ⚠️  {n_invalid} samples with invalid ring count")
        
        if valid_mask.all():
            report.append("  ✓ All samples passed physical constraints check")
        
        return valid_mask, report
    
    def _check_feature_relationships(self, X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Check relationships between features"""
        valid_mask = np.ones(len(X), dtype=bool)
        report = ["\n--- Feature Relationships Check ---"]
        
        # Check if heteroatoms <= reasonable fraction of mass
        mass = self._get_feature_values(X, 'Mass')
        hetero = self._get_feature_values(X, 'NumHeteroatoms')
        
        if mass is not None and hetero is not None:
            # Assuming average heteroatom mass ~15 Da (N, O)
            # If heteroatoms * 15 > mass, something is wrong
            hetero_mass_valid = (hetero * 15) <= mass
            valid_mask &= hetero_mass_valid
            n_invalid = (~hetero_mass_valid).sum()
            if n_invalid > 0:
                report.append(f"  ⚠️  {n_invalid} samples with heteroatoms inconsistent with mass")
        
        # Check if H-bond donors <= H-bond acceptors + N + O (rough heuristic)
        donors = self._get_feature_values(X, 'HDonors')
        acceptors = self._get_feature_values(X, 'HAcceptors')
        
        if donors is not None and acceptors is not None and hetero is not None:
            # Very rough check: donors should not exceed heteroatoms
            donors_valid = donors <= hetero + 5  # +5 for edge cases
            valid_mask &= donors_valid
            n_invalid = (~donors_valid).sum()
            if n_invalid > 0:
                report.append(f"  ⚠️  {n_invalid} samples with H-donors inconsistent with structure")
        
        if valid_mask.all():
            report.append("  ✓ All samples passed feature relationships check")
        
        return valid_mask, report
    
    def print_report(self, report: List[str]):
        """Print validation report"""
        for line in report:
            print(line)

def validate_synthetic_data(X_synthetic: np.ndarray, 
                           feature_names: List[str],
                           strict: bool = False,
                           verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Convenience function to validate synthetic data
    
    Parameters:
    -----------
    X_synthetic : np.ndarray
        Synthetic data to validate
    feature_names : list of str
        Names of features
    strict : bool
        Use stricter validation criteria
    verbose : bool
        Print validation report
    
    Returns:
    --------
    X_valid : np.ndarray
        Valid synthetic samples
    valid_mask : np.ndarray
        Boolean mask of valid samples
    stats : dict
        Validation statistics
    """
    validator = ChemicalValidator(feature_names, strict=strict)
    valid_mask, report = validator.validate_batch(X_synthetic)
    
    if verbose:
        validator.print_report(report)
    
    X_valid = X_synthetic[valid_mask]
    
    stats = {
        'n_total': len(X_synthetic),
        'n_valid': valid_mask.sum(),
        'n_invalid': (~valid_mask).sum(),
        'validity_rate': valid_mask.sum() / len(X_synthetic),
        'report': report
    }
    
    return X_valid, valid_mask, stats

if __name__ == "__main__":
    # Example usage
    print("Chemical Validator Module")
    print("="*80)
    
    if not RDKIT_AVAILABLE:
        print("⚠️  RDKit not installed. Some validation features will be limited.")
        print("   Install with: pip install rdkit")
    else:
        print("✓ RDKit is available")
    
    # Demo with synthetic data
    feature_names = ['Mass', 'HAcceptors', 'HDonors', 'NumHeteroatoms', 
                    'HOMO(eV)', 'LUMO(eV)', 'LogP']
    
    # Create some test data (half valid, half invalid)
    n_samples = 100
    X_test = np.random.randn(n_samples, len(feature_names))
    
    # Make first half valid
    X_test[:50, 0] = np.random.uniform(50, 500, 50)  # Mass
    X_test[:50, 1] = np.random.randint(0, 10, 50)     # HAcceptors
    X_test[:50, 2] = np.random.randint(0, 5, 50)      # HDonors
    X_test[:50, 3] = np.random.randint(0, 15, 50)     # Heteroatoms
    X_test[:50, 4] = np.random.uniform(-10, -3, 50)   # HOMO
    X_test[:50, 5] = np.random.uniform(-2, 2, 50)     # LUMO
    X_test[:50, 6] = np.random.uniform(-3, 6, 50)     # LogP
    
    # Make second half invalid (negative HOMO-LUMO gap)
    X_test[50:, 0] = np.random.uniform(50, 500, 50)
    X_test[50:, 4] = np.random.uniform(-2, 0, 50)     # HOMO higher
    X_test[50:, 5] = np.random.uniform(-5, -3, 50)    # LUMO lower (invalid!)
    
    print("\nTesting with synthetic data...")
    X_valid, valid_mask, stats = validate_synthetic_data(
        X_test, feature_names, strict=False, verbose=True
    )
    
    print(f"\n✓ Validation complete: {stats['validity_rate']:.1%} samples valid")
