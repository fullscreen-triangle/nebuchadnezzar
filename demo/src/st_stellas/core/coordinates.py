"""
Molecular Coordinate Transformation Implementation
================================================

Implementation of S-Entropy molecular coordinate transformation for converting
raw molecular data into geometric coordinate systems for navigation-based analysis.

Based on: "S-Entropy Molecular Coordinate Transformation: Mathematical Framework 
for Raw Data Conversion to Multi-Dimensional Entropy Space"
Author: Kundai Farai Sachikonye
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from collections import Counter
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

# Chemistry imports for SMILES processing
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available. Chemical structure processing will be limited.")

logger = logging.getLogger(__name__)

@dataclass
class MolecularCoordinate:
    """Represents a molecular coordinate in S-entropy space."""
    s_knowledge: float
    s_time: float
    s_entropy: float
    position: int  # Original position in sequence
    context_window: Optional[List] = None
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy vector."""
        return np.array([self.s_knowledge, self.s_time, self.s_entropy])
    
    def distance_to(self, other: 'MolecularCoordinate') -> float:
        """Calculate Euclidean distance to another coordinate."""
        return np.linalg.norm(self.to_vector() - other.to_vector())

class BaseCoordinateTransform(ABC):
    """Base class for molecular coordinate transformations."""
    
    def __init__(self, window_size: int = 5):
        """
        Initialize base transformer.
        
        Args:
            window_size: Size of context window for weighting functions
        """
        self.window_size = window_size
        
    @abstractmethod
    def transform_element(self, element: str, position: int, context: List[str]) -> MolecularCoordinate:
        """Transform a single molecular element to S-entropy coordinates."""
        pass
    
    def transform_sequence(self, sequence: Union[str, List[str]]) -> List[MolecularCoordinate]:
        """
        Transform entire molecular sequence to coordinate path.
        
        Args:
            sequence: Molecular sequence (string or list of elements)
            
        Returns:
            List of molecular coordinates representing the sequence path
        """
        if isinstance(sequence, str):
            sequence = list(sequence)
            
        coordinates = []
        
        for i, element in enumerate(sequence):
            # Create context window
            start_idx = max(0, i - self.window_size // 2)
            end_idx = min(len(sequence), i + self.window_size // 2 + 1)
            context = sequence[start_idx:end_idx]
            
            # Transform element
            coord = self.transform_element(element, i, context)
            coordinates.append(coord)
            
        return coordinates
    
    def calculate_path_properties(self, coordinates: List[MolecularCoordinate]) -> Dict[str, float]:
        """
        Calculate properties of the coordinate path.
        
        Args:
            coordinates: List of molecular coordinates
            
        Returns:
            Dictionary of path properties
        """
        if not coordinates:
            return {}
            
        vectors = np.array([coord.to_vector() for coord in coordinates])
        
        # Calculate path statistics
        path_length = sum(
            coordinates[i].distance_to(coordinates[i+1])
            for i in range(len(coordinates)-1)
        )
        
        centroid = np.mean(vectors, axis=0)
        variance = np.var(vectors, axis=0)
        
        return {
            'path_length': path_length,
            'centroid': centroid,
            'variance': variance,
            'sequence_length': len(coordinates),
            'mean_s_knowledge': np.mean(vectors[:, 0]),
            'mean_s_time': np.mean(vectors[:, 1]),
            'mean_s_entropy': np.mean(vectors[:, 2])
        }

class DNACoordinateTransform(BaseCoordinateTransform):
    """
    DNA sequence coordinate transformation using cardinal direction mapping.
    
    Maps nucleotide bases to cardinal directions:
    A → (0,1)   (North)
    T → (0,-1)  (South)  
    G → (1,0)   (East)
    C → (-1,0)  (West)
    """
    
    # Cardinal direction mapping for nucleotide bases
    BASE_COORDINATES = {
        'A': (0, 1),   # North
        'T': (0, -1),  # South
        'G': (1, 0),   # East  
        'C': (-1, 0),  # West
        'U': (0, -1),  # RNA: U maps same as T
        'N': (0, 0),   # Unknown base
    }
    
    def __init__(self, window_size: int = 5):
        """
        Initialize DNA coordinate transformer.
        
        Args:
            window_size: Size of context window for S-entropy weighting
        """
        super().__init__(window_size)
        
    def _calculate_knowledge_weighting(self, base: str, context: List[str]) -> float:
        """
        Calculate knowledge weighting function: w_k(b,i,W_i) = -Σ p_j log₂(p_j)
        
        Args:
            base: Current nucleotide base
            context: Context window around the base
            
        Returns:
            Knowledge weighting value
        """
        if not context:
            return 0.0
            
        # Calculate base probabilities in context window
        base_counts = Counter(context)
        total_bases = len(context)
        
        entropy = 0.0
        for base_type, count in base_counts.items():
            p = count / total_bases
            if p > 0:
                entropy -= p * np.log2(p)
                
        return entropy
    
    def _calculate_time_weighting(self, base: str, position: int, context: List[str]) -> float:
        """
        Calculate time weighting function: w_t(b,i,W_i) = Σ(j=1 to i) δ(b_j,b) / i
        
        Args:
            base: Current nucleotide base
            position: Position in sequence
            context: Context window
            
        Returns:
            Time weighting value
        """
        if position == 0:
            return 1.0
            
        # Count occurrences of current base up to current position
        sequence_up_to_position = context[:min(len(context), position+1)]
        base_count = sequence_up_to_position.count(base)
        
        return base_count / len(sequence_up_to_position)
    
    def _calculate_entropy_weighting(self, base: str, context: List[str]) -> float:
        """
        Calculate entropy weighting function: w_e(b,i,W_i) = √(Σ(ψ(b_j) - ψ̄)²)
        
        Args:
            base: Current nucleotide base
            context: Context window
            
        Returns:
            Entropy weighting value
        """
        if not context:
            return 0.0
            
        # Get coordinate vectors for all bases in context
        coord_vectors = []
        for b in context:
            if b in self.BASE_COORDINATES:
                coord_vectors.append(np.array(self.BASE_COORDINATES[b]))
            else:
                coord_vectors.append(np.array([0, 0]))  # Unknown base
                
        if not coord_vectors:
            return 0.0
            
        coord_array = np.array(coord_vectors)
        mean_coord = np.mean(coord_array, axis=0)
        
        # Calculate variance
        variance = np.sum((coord_array - mean_coord) ** 2)
        return np.sqrt(variance)
    
    def transform_element(self, base: str, position: int, context: List[str]) -> MolecularCoordinate:
        """
        Transform nucleotide base to S-entropy coordinates.
        
        Args:
            base: Nucleotide base (A, T, G, C, U, N)
            position: Position in sequence
            context: Context window around the base
            
        Returns:
            Molecular coordinate in S-entropy space
        """
        base = base.upper()
        
        if base not in self.BASE_COORDINATES:
            logger.warning(f"Unknown base '{base}' at position {position}, using (0,0)")
            base_coord = (0, 0)
        else:
            base_coord = self.BASE_COORDINATES[base]
            
        # Calculate weighting functions
        w_k = self._calculate_knowledge_weighting(base, context)
        w_t = self._calculate_time_weighting(base, position, context)
        w_e = self._calculate_entropy_weighting(base, context)
        
        # Apply S-entropy weighted coordinate transformation
        # Φ(b,i,W_i) = (w_k * ψ_x(b), w_t * ψ_y(b), w_e * |ψ(b)|)
        base_x, base_y = base_coord
        magnitude = np.sqrt(base_x**2 + base_y**2) if (base_x != 0 or base_y != 0) else 1.0
        
        s_knowledge = w_k * base_x
        s_time = w_t * base_y  
        s_entropy = w_e * magnitude
        
        return MolecularCoordinate(
            s_knowledge=s_knowledge,
            s_time=s_time,
            s_entropy=s_entropy,
            position=position,
            context_window=context.copy()
        )
    
    def analyze_dual_strand(self, 
                          forward_strand: str, 
                          alpha: float = 0.5,
                          beta: float = 0.5) -> List[MolecularCoordinate]:
        """
        Analyze double-stranded DNA with both forward and reverse complement.
        
        Args:
            forward_strand: Forward DNA strand sequence
            alpha: Weight for forward strand (default 0.5)
            beta: Weight for reverse strand (default 0.5, α + β = 1)
            
        Returns:
            List of dual-strand coordinates
        """
        if abs(alpha + beta - 1.0) > 1e-6:
            logger.warning("Alpha + beta should equal 1.0 for proper dual-strand weighting")
            
        # Generate reverse complement
        complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'U': 'A', 'N': 'N'}
        reverse_strand = ''.join(complement_map.get(base, 'N') for base in forward_strand[::-1])
        
        # Transform both strands
        forward_coords = self.transform_sequence(forward_strand)
        reverse_coords = self.transform_sequence(reverse_strand)
        
        # Combine using weighted average
        dual_coords = []
        for i, (fwd, rev) in enumerate(zip(forward_coords, reverse_coords)):
            combined_coord = MolecularCoordinate(
                s_knowledge=alpha * fwd.s_knowledge + beta * rev.s_knowledge,
                s_time=alpha * fwd.s_time + beta * rev.s_time,
                s_entropy=alpha * fwd.s_entropy + beta * rev.s_entropy,
                position=i,
                context_window=fwd.context_window
            )
            dual_coords.append(combined_coord)
            
        return dual_coords

class ProteinCoordinateTransform(BaseCoordinateTransform):
    """
    Protein sequence coordinate transformation using physicochemical properties.
    
    Maps amino acids to coordinates based on:
    - Hydrophobicity (Kyte-Doolittle scale)
    - Polarity (charge distribution)
    - Size (molecular weight)
    """
    
    # Kyte-Doolittle hydrophobicity values
    HYDROPHOBICITY = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
        'X': 0.0  # Unknown amino acid
    }
    
    # Polarity classification
    POLARITY = {
        # Positive
        'R': 1.0, 'K': 1.0, 'H': 1.0,
        # Negative  
        'D': -1.0, 'E': -1.0,
        # Polar
        'S': 0.5, 'T': 0.5, 'N': 0.5, 'Q': 0.5, 'Y': 0.5,
        # Nonpolar
        'A': 0.0, 'V': 0.0, 'I': 0.0, 'L': 0.0, 'M': 0.0,
        'F': 0.0, 'W': 0.0, 'P': 0.0, 'G': 0.0, 'C': 0.0,
        'X': 0.0
    }
    
    # Molecular weights (approximate)
    MOLECULAR_WEIGHT = {
        'A': 89, 'R': 174, 'N': 132, 'D': 133, 'C': 121,
        'Q': 146, 'E': 147, 'G': 75, 'H': 155, 'I': 131,
        'L': 131, 'K': 146, 'M': 149, 'F': 165, 'P': 115,
        'S': 105, 'T': 119, 'W': 204, 'Y': 181, 'V': 117,
        'X': 125  # Average molecular weight
    }
    
    def __init__(self, window_size: int = 5):
        """Initialize protein coordinate transformer."""
        super().__init__(window_size)
        
        # Normalize property scales
        self._normalize_properties()
        
    def _normalize_properties(self):
        """Normalize physicochemical properties to [0,1] range."""
        # Normalize hydrophobicity
        h_values = list(self.HYDROPHOBICITY.values())
        h_min, h_max = min(h_values), max(h_values)
        self.hydrophobicity_norm = {
            aa: (h - h_min) / (h_max - h_min) 
            for aa, h in self.HYDROPHOBICITY.items()
        }
        
        # Normalize molecular weight
        mw_values = list(self.MOLECULAR_WEIGHT.values())
        mw_min, mw_max = min(mw_values), max(mw_values)
        self.molecular_weight_norm = {
            aa: (mw - mw_min) / (mw_max - mw_min)
            for aa, mw in self.MOLECULAR_WEIGHT.items()
        }
        
    def _calculate_protein_knowledge_weighting(self, aa: str, context: List[str]) -> float:
        """Calculate knowledge weighting for protein context."""
        if not context:
            return 0.0
            
        aa_counts = Counter(context)
        total_aas = len(context)
        
        entropy = 0.0
        for aa_type, count in aa_counts.items():
            p = count / total_aas
            if p > 0:
                entropy -= p * np.log2(p)
                
        return entropy
    
    def _calculate_protein_time_weighting(self, aa: str, position: int, context: List[str]) -> float:
        """Calculate time weighting with exponential decay."""
        if position == 0:
            return 1.0
            
        tau = 5.0  # Characteristic length
        weighted_sum = 0.0
        
        for j, context_aa in enumerate(context):
            if j <= position and context_aa == aa:
                weight = np.exp(-(position - j) / tau)
                weighted_sum += weight
                
        return weighted_sum
    
    def _calculate_protein_entropy_weighting(self, aa: str, context: List[str]) -> float:
        """Calculate entropy weighting based on coordinate variance."""
        if not context:
            return 0.0
            
        coord_vectors = []
        for context_aa in context:
            h = self.hydrophobicity_norm.get(context_aa, 0.0)
            p = self.POLARITY.get(context_aa, 0.0)
            s = self.molecular_weight_norm.get(context_aa, 0.0)
            coord_vectors.append(np.array([h, p, s]))
            
        if not coord_vectors:
            return 0.0
            
        coord_array = np.array(coord_vectors)
        mean_coord = np.mean(coord_array, axis=0)
        variance = np.mean(np.sum((coord_array - mean_coord) ** 2, axis=1))
        
        return np.sqrt(variance)
    
    def transform_element(self, aa: str, position: int, context: List[str]) -> MolecularCoordinate:
        """
        Transform amino acid to S-entropy coordinates.
        
        Args:
            aa: Amino acid single letter code
            position: Position in protein sequence
            context: Context window around the amino acid
            
        Returns:
            Molecular coordinate in S-entropy space
        """
        aa = aa.upper()
        
        # Get physicochemical properties
        hydrophobicity = self.hydrophobicity_norm.get(aa, 0.0)
        polarity = self.POLARITY.get(aa, 0.0)
        size = self.molecular_weight_norm.get(aa, 0.0)
        
        # Calculate weighting functions
        w_k = self._calculate_protein_knowledge_weighting(aa, context)
        w_t = self._calculate_protein_time_weighting(aa, position, context)
        w_e = self._calculate_protein_entropy_weighting(aa, context)
        
        # Apply S-entropy transformation
        # Ξ(a,i,W_i) = (w_k * h(a), w_t * p(a), w_e * s(a))
        s_knowledge = w_k * hydrophobicity
        s_time = w_t * polarity
        s_entropy = w_e * size
        
        return MolecularCoordinate(
            s_knowledge=s_knowledge,
            s_time=s_time,
            s_entropy=s_entropy,
            position=position,
            context_window=context.copy()
        )

class ChemicalCoordinateTransform(BaseCoordinateTransform):
    """
    Chemical structure coordinate transformation using SMILES notation.
    
    Maps functional groups to coordinates based on:
    - Electronegativity
    - Reactivity  
    - Bonding capacity
    """
    
    def __init__(self, window_size: int = 3):
        """Initialize chemical coordinate transformer."""
        super().__init__(window_size)
        
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available. Chemical transformations will use simplified mapping.")
            
    def _parse_smiles_to_functional_groups(self, smiles: str) -> List[str]:
        """
        Parse SMILES string into functional groups.
        
        Args:
            smiles: SMILES notation string
            
        Returns:
            List of functional group identifiers
        """
        if not RDKIT_AVAILABLE:
            # Simplified functional group detection without RDKit
            groups = []
            for i, char in enumerate(smiles):
                if char in 'CNOS':  # Basic atoms
                    groups.append(char)
                elif char == '=':
                    groups.append('double_bond')
                elif char == '#':
                    groups.append('triple_bond')
                elif char in '()[]':
                    groups.append('branch')
            return groups
            
        # Use RDKit for proper functional group analysis
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []
                
            # Extract functional group information
            groups = []
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                degree = atom.GetDegree()
                groups.append(f"{symbol}_{degree}")
                
            return groups
        except Exception as e:
            logger.error(f"Error parsing SMILES {smiles}: {e}")
            return []
    
    def _calculate_electronegativity(self, functional_group: str) -> float:
        """Calculate electronegativity measure for functional group."""
        # Simplified electronegativity mapping
        electronegativity_map = {
            'C': 2.55, 'N': 3.04, 'O': 3.44, 'S': 2.58,
            'F': 3.98, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66,
            'H': 2.20, 'P': 2.19
        }
        
        # Extract base element
        base_element = functional_group.split('_')[0] if '_' in functional_group else functional_group[0]
        return electronegativity_map.get(base_element, 2.5)  # Default value
    
    def _calculate_reactivity(self, functional_group: str) -> float:
        """Calculate reactivity measure for functional group."""
        # Simplified reactivity scoring
        reactivity_map = {
            'double_bond': 0.8,
            'triple_bond': 0.9,
            'C': 0.3, 'N': 0.7, 'O': 0.8, 'S': 0.6,
            'branch': 0.2
        }
        
        return reactivity_map.get(functional_group, 0.5)
    
    def _calculate_bonding_capacity(self, functional_group: str) -> float:
        """Calculate bonding capacity measure for functional group."""
        # Simplified bonding capacity based on typical valence
        bonding_map = {
            'C': 4, 'N': 3, 'O': 2, 'S': 2,
            'H': 1, 'F': 1, 'Cl': 1, 'Br': 1, 'I': 1
        }
        
        base_element = functional_group.split('_')[0] if '_' in functional_group else functional_group[0]
        return bonding_map.get(base_element, 2) / 4.0  # Normalize to [0,1]
    
    def transform_element(self, functional_group: str, position: int, context: List[str]) -> MolecularCoordinate:
        """
        Transform functional group to S-entropy coordinates.
        
        Args:
            functional_group: Functional group identifier
            position: Position in SMILES structure
            context: Context window around the functional group
            
        Returns:
            Molecular coordinate in S-entropy space
        """
        # Calculate chemical properties
        electronegativity = self._calculate_electronegativity(functional_group)
        reactivity = self._calculate_reactivity(functional_group)
        bonding_capacity = self._calculate_bonding_capacity(functional_group)
        
        # Calculate weighting functions (simplified for chemical structures)
        w_k = len(set(context)) / len(context) if context else 0.0  # Diversity
        w_t = position / 10.0  # Position weighting
        w_e = np.std([self._calculate_reactivity(fg) for fg in context]) if len(context) > 1 else 0.0
        
        # Apply S-entropy transformation
        s_knowledge = w_k * electronegativity
        s_time = w_t * reactivity
        s_entropy = w_e * bonding_capacity
        
        return MolecularCoordinate(
            s_knowledge=s_knowledge,
            s_time=s_time,
            s_entropy=s_entropy,
            position=position,
            context_window=context.copy()
        )
    
    def transform_smiles(self, smiles: str) -> List[MolecularCoordinate]:
        """
        Transform SMILES string to coordinate path.
        
        Args:
            smiles: SMILES notation string
            
        Returns:
            List of molecular coordinates
        """
        functional_groups = self._parse_smiles_to_functional_groups(smiles)
        return self.transform_sequence(functional_groups)

class MolecularCoordinates:
    """
    Main molecular coordinates interface providing unified access to all
    molecular coordinate transformation systems.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize molecular coordinates system.
        
        Args:
            window_size: Default context window size for all transformers
        """
        self.dna_transform = DNACoordinateTransform(window_size)
        self.protein_transform = ProteinCoordinateTransform(window_size)
        self.chemical_transform = ChemicalCoordinateTransform(window_size)
        
    def transform_dna_sequence(self, sequence: str) -> List[MolecularCoordinate]:
        """Transform DNA sequence to S-entropy coordinates."""
        return self.dna_transform.transform_sequence(sequence)
    
    def transform_protein_sequence(self, sequence: str) -> List[MolecularCoordinate]:
        """Transform protein sequence to S-entropy coordinates."""
        return self.protein_transform.transform_sequence(sequence)
    
    def transform_chemical_structure(self, smiles: str) -> List[MolecularCoordinate]:
        """Transform chemical structure to S-entropy coordinates."""
        return self.chemical_transform.transform_smiles(smiles)
    
    def validate_cross_modal_consistency(self, 
                                       dna_coords: List[MolecularCoordinate],
                                       protein_coords: List[MolecularCoordinate],
                                       chemical_coords: List[MolecularCoordinate],
                                       epsilon: float = 1.0) -> Dict[str, Any]:
        """
        Validate cross-modal coordinate consistency.
        
        Args:
            dna_coords: DNA coordinate path
            protein_coords: Protein coordinate path  
            chemical_coords: Chemical coordinate path
            epsilon: Consistency threshold
            
        Returns:
            Validation results dictionary
        """
        def calculate_path_centroid(coords):
            if not coords:
                return np.zeros(3)
            vectors = np.array([coord.to_vector() for coord in coords])
            return np.mean(vectors, axis=0)
        
        # Calculate centroids for each modality
        dna_centroid = calculate_path_centroid(dna_coords)
        protein_centroid = calculate_path_centroid(protein_coords)
        chemical_centroid = calculate_path_centroid(chemical_coords)
        
        # Calculate cross-modal distances
        d_dna_protein = np.linalg.norm(dna_centroid - protein_centroid)
        d_protein_chemical = np.linalg.norm(protein_centroid - chemical_centroid)
        d_chemical_dna = np.linalg.norm(chemical_centroid - dna_centroid)
        
        total_distance = d_dna_protein + d_protein_chemical + d_chemical_dna
        is_consistent = total_distance < epsilon
        
        return {
            'is_consistent': is_consistent,
            'total_cross_modal_distance': total_distance,
            'dna_protein_distance': d_dna_protein,
            'protein_chemical_distance': d_protein_chemical,
            'chemical_dna_distance': d_chemical_dna,
            'consistency_threshold': epsilon,
            'dna_centroid': dna_centroid,
            'protein_centroid': protein_centroid,
            'chemical_centroid': chemical_centroid
        }
