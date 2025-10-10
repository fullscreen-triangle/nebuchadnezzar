"""
Genetic Risk Assessment Module
=============================

Quantifies the impact of genetic variants on drug response using oscillatory
framework principles. Assesses how variants affect drug metabolism, efficacy,
and safety through multi-scale oscillatory disruption analysis.

Based on the St. Stellas oscillatory framework for personalized pharmacology.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DrugRiskProfile:
    """Risk profile for a specific drug based on genetic variants."""
    drug_name: str
    overall_risk_score: float  # 0.0 (low risk) to 1.0 (high risk)
    efficacy_risk: float       # Risk of reduced efficacy
    toxicity_risk: float       # Risk of adverse effects
    metabolism_risk: float     # Risk of altered metabolism
    contributing_variants: List[str]
    risk_factors: List[str]
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.0

@dataclass
class VariantRiskAssessment:
    """Risk assessment for a specific genetic variant."""
    variant_id: str
    gene: str
    risk_category: str  # 'LOW', 'MODERATE', 'HIGH', 'VERY_HIGH'
    numerical_risk: float  # 0.0 to 1.0
    affected_drugs: List[str]
    mechanism: str  # How variant affects drug response
    population_frequency: float
    clinical_significance: str
    evidence_level: str  # 'STRONG', 'MODERATE', 'WEAK'

@dataclass 
class PharmacogeneticProfile:
    """Complete pharmacogenetic profile for an individual."""
    individual_id: str
    total_variants: int
    high_risk_variants: int
    drug_risk_profiles: List[DrugRiskProfile]
    variant_assessments: List[VariantRiskAssessment]
    overall_risk_score: float
    high_risk_drugs: List[str]
    recommendations: List[str]

class GeneticRiskAssessor:
    """
    Assesses genetic risk for drug response using oscillatory framework.
    
    Evaluates how genetic variants disrupt normal oscillatory patterns
    and affect drug metabolism, efficacy, and safety.
    """
    
    def __init__(self):
        """Initialize the risk assessor with drug-gene databases."""
        self.drug_gene_interactions = self._load_drug_gene_database()
        self.variant_effects = self._load_variant_effects_database()
        self.population_frequencies = self._load_population_frequencies()
        
    def _load_drug_gene_database(self) -> Dict[str, Dict]:
        """Load drug-gene interaction database."""
        
        # Key pharmacogenetic drug-gene interactions
        interactions = {
            'lithium': {
                'primary_genes': ['INPP1', 'GSK3B', 'IMPA1', 'IMPA2'],
                'secondary_genes': ['SLC1A2', 'SLC1A3', 'COMT'],
                'mechanism': 'inositol_pathway_modulation',
                'therapeutic_class': 'mood_stabilizer',
                'risk_factors': ['kidney_function', 'thyroid_function', 'dehydration']
            },
            
            'aripiprazole': {
                'primary_genes': ['CYP2D6', 'CYP3A4', 'DRD2', 'HTR2A'],
                'secondary_genes': ['ABCB1', 'COMT'],
                'mechanism': 'dopamine_serotonin_modulation',
                'therapeutic_class': 'antipsychotic',
                'risk_factors': ['metabolic_syndrome', 'extrapyramidal_symptoms']
            },
            
            'citalopram': {
                'primary_genes': ['CYP2C19', 'CYP2D6', 'HTR2A', 'SLC6A4'],
                'secondary_genes': ['ABCB1', 'COMT', 'MAOA'],
                'mechanism': 'serotonin_reuptake_inhibition',
                'therapeutic_class': 'antidepressant',
                'risk_factors': ['qt_prolongation', 'serotonin_syndrome']
            },
            
            'atorvastatin': {
                'primary_genes': ['SLCO1B1', 'CYP3A4', 'ABCG2'],
                'secondary_genes': ['CYP2C8', 'UGT1A1'],
                'mechanism': 'hmg_coa_reductase_inhibition',
                'therapeutic_class': 'statin',
                'risk_factors': ['myopathy', 'rhabdomyolysis', 'hepatotoxicity']
            },
            
            'aspirin': {
                'primary_genes': ['CYP2C9', 'PTGS1', 'PTGS2'],
                'secondary_genes': ['ABCC4', 'LTC4S'],
                'mechanism': 'cyclooxygenase_inhibition',
                'therapeutic_class': 'antiplatelet',
                'risk_factors': ['bleeding', 'gastric_ulcers', 'asthma']
            }
        }
        
        logger.info(f"Loaded interactions for {len(interactions)} drugs")
        return interactions
    
    def _load_variant_effects_database(self) -> Dict[str, Dict]:
        """Load variant effect database."""
        
        # Define how different variants affect oscillatory patterns
        effects = {
            # CYP2D6 - Major drug metabolizing enzyme
            'CYP2D6': {
                'poor_metabolizer': {'risk_multiplier': 3.0, 'mechanism': 'reduced_clearance'},
                'intermediate_metabolizer': {'risk_multiplier': 1.5, 'mechanism': 'reduced_clearance'},
                'extensive_metabolizer': {'risk_multiplier': 1.0, 'mechanism': 'normal_clearance'},
                'ultrarapid_metabolizer': {'risk_multiplier': 2.0, 'mechanism': 'increased_clearance'}
            },
            
            # CYP2C19 - Important for many psychiatric drugs
            'CYP2C19': {
                'poor_metabolizer': {'risk_multiplier': 2.5, 'mechanism': 'reduced_clearance'},
                'intermediate_metabolizer': {'risk_multiplier': 1.3, 'mechanism': 'reduced_clearance'},
                'extensive_metabolizer': {'risk_multiplier': 1.0, 'mechanism': 'normal_clearance'},
                'rapid_metabolizer': {'risk_multiplier': 1.8, 'mechanism': 'increased_clearance'}
            },
            
            # SLCO1B1 - Statin transporter
            'SLCO1B1': {
                'decreased_function': {'risk_multiplier': 2.8, 'mechanism': 'reduced_uptake'},
                'normal_function': {'risk_multiplier': 1.0, 'mechanism': 'normal_uptake'}
            },
            
            # Inositol pathway genes (lithium targets)
            'INPP1': {
                'high_impact': {'risk_multiplier': 0.7, 'mechanism': 'increased_sensitivity'},
                'moderate_impact': {'risk_multiplier': 0.85, 'mechanism': 'increased_sensitivity'},
                'low_impact': {'risk_multiplier': 0.95, 'mechanism': 'minimal_effect'}
            },
            
            'GSK3B': {
                'high_impact': {'risk_multiplier': 0.6, 'mechanism': 'increased_sensitivity'},
                'moderate_impact': {'risk_multiplier': 0.8, 'mechanism': 'increased_sensitivity'},
                'low_impact': {'risk_multiplier': 0.9, 'mechanism': 'minimal_effect'}
            }
        }
        
        return effects
    
    def _load_population_frequencies(self) -> Dict[str, float]:
        """Load population frequencies for common variants."""
        
        frequencies = {
            'CYP2D6_poor_metabolizer': 0.07,
            'CYP2D6_intermediate_metabolizer': 0.10,
            'CYP2D6_ultrarapid_metabolizer': 0.05,
            'CYP2C19_poor_metabolizer': 0.03,
            'CYP2C19_rapid_metabolizer': 0.17,
            'SLCO1B1_decreased_function': 0.15,
            'INPP1_high_impact': 0.12,
            'GSK3B_high_impact': 0.08
        }
        
        return frequencies
    
    def assess_variant_risk(self, variant: Dict[str, Any]) -> VariantRiskAssessment:
        """Assess risk for a single genetic variant."""
        
        gene = variant.get('gene', 'UNKNOWN')
        variant_id = variant.get('variant_id', 'unknown')
        impact = variant.get('impact', 'UNKNOWN')
        
        # Determine numerical risk based on gene and impact
        numerical_risk = self._calculate_numerical_risk(gene, impact)
        
        # Categorize risk level
        risk_category = self._categorize_risk(numerical_risk)
        
        # Find affected drugs
        affected_drugs = self._find_affected_drugs(gene)
        
        # Determine mechanism
        mechanism = self._determine_mechanism(gene, impact)
        
        # Get population frequency
        freq_key = f"{gene}_{impact.lower()}"
        population_frequency = self.population_frequencies.get(freq_key, 0.05)
        
        # Determine clinical significance and evidence level
        clinical_significance = self._assess_clinical_significance(gene, numerical_risk)
        evidence_level = self._assess_evidence_level(gene)
        
        return VariantRiskAssessment(
            variant_id=variant_id,
            gene=gene,
            risk_category=risk_category,
            numerical_risk=numerical_risk,
            affected_drugs=affected_drugs,
            mechanism=mechanism,
            population_frequency=population_frequency,
            clinical_significance=clinical_significance,
            evidence_level=evidence_level
        )
    
    def _calculate_numerical_risk(self, gene: str, impact: str) -> float:
        """Calculate numerical risk score for a variant."""
        
        # Base risk mapping
        impact_risk = {
            'HIGH': 0.8,
            'MODERATE': 0.6,
            'LOW': 0.3,
            'MODIFIER': 0.1,
            'UNKNOWN': 0.4
        }
        
        base_risk = impact_risk.get(impact, 0.4)
        
        # Gene-specific risk modifiers
        gene_modifiers = {
            'CYP2D6': 1.2,    # High clinical importance
            'CYP2C19': 1.1,   # High clinical importance
            'SLCO1B1': 1.0,   # Moderate importance
            'INPP1': 0.9,     # Beneficial for lithium
            'GSK3B': 0.8,     # Beneficial for lithium
            'DRD2': 1.0,      # Moderate importance
            'HTR2A': 0.9      # Moderate importance
        }
        
        gene_modifier = gene_modifiers.get(gene, 1.0)
        
        # Calculate final risk
        final_risk = min(1.0, base_risk * gene_modifier)
        
        return final_risk
    
    def _categorize_risk(self, numerical_risk: float) -> str:
        """Categorize numerical risk into categories."""
        
        if numerical_risk >= 0.8:
            return 'VERY_HIGH'
        elif numerical_risk >= 0.6:
            return 'HIGH'
        elif numerical_risk >= 0.3:
            return 'MODERATE'
        else:
            return 'LOW'
    
    def _find_affected_drugs(self, gene: str) -> List[str]:
        """Find drugs affected by variants in this gene."""
        
        affected_drugs = []
        
        for drug, info in self.drug_gene_interactions.items():
            if gene in info['primary_genes'] or gene in info['secondary_genes']:
                affected_drugs.append(drug)
        
        return affected_drugs
    
    def _determine_mechanism(self, gene: str, impact: str) -> str:
        """Determine the mechanism by which variant affects drug response."""
        
        mechanisms = {
            'CYP2D6': 'altered_drug_metabolism',
            'CYP2C19': 'altered_drug_metabolism',
            'CYP3A4': 'altered_drug_metabolism',
            'SLCO1B1': 'altered_drug_transport',
            'ABCB1': 'altered_drug_transport',
            'DRD2': 'altered_receptor_binding',
            'HTR2A': 'altered_receptor_binding',
            'INPP1': 'enhanced_drug_sensitivity',
            'GSK3B': 'enhanced_drug_sensitivity',
            'COMT': 'altered_neurotransmitter_metabolism'
        }
        
        return mechanisms.get(gene, 'unknown_mechanism')
    
    def _assess_clinical_significance(self, gene: str, risk: float) -> str:
        """Assess clinical significance of variant."""
        
        high_significance_genes = ['CYP2D6', 'CYP2C19', 'SLCO1B1']
        
        if gene in high_significance_genes and risk > 0.6:
            return 'HIGH_CLINICAL_SIGNIFICANCE'
        elif risk > 0.7:
            return 'MODERATE_CLINICAL_SIGNIFICANCE'
        else:
            return 'LOW_CLINICAL_SIGNIFICANCE'
    
    def _assess_evidence_level(self, gene: str) -> str:
        """Assess evidence level for gene-drug interactions."""
        
        strong_evidence = ['CYP2D6', 'CYP2C19', 'SLCO1B1']
        moderate_evidence = ['DRD2', 'HTR2A', 'COMT', 'ABCB1']
        
        if gene in strong_evidence:
            return 'STRONG'
        elif gene in moderate_evidence:
            return 'MODERATE'
        else:
            return 'WEAK'
    
    def assess_drug_risk(self, drug_name: str, variants: List[Dict[str, Any]]) -> DrugRiskProfile:
        """Assess risk for a specific drug based on genetic variants."""
        
        if drug_name not in self.drug_gene_interactions:
            logger.warning(f"Unknown drug: {drug_name}")
            return DrugRiskProfile(
                drug_name=drug_name,
                overall_risk_score=0.5,
                efficacy_risk=0.5,
                toxicity_risk=0.5,
                metabolism_risk=0.5,
                contributing_variants=[],
                risk_factors=['unknown_drug'],
                confidence=0.1
            )
        
        drug_info = self.drug_gene_interactions[drug_name]
        relevant_genes = set(drug_info['primary_genes'] + drug_info['secondary_genes'])
        
        # Find variants affecting this drug
        relevant_variants = [v for v in variants if v.get('gene') in relevant_genes]
        
        # Calculate risk components
        efficacy_risk = self._calculate_efficacy_risk(drug_name, relevant_variants)
        toxicity_risk = self._calculate_toxicity_risk(drug_name, relevant_variants)
        metabolism_risk = self._calculate_metabolism_risk(drug_name, relevant_variants)
        
        # Overall risk is weighted average
        overall_risk = (0.4 * efficacy_risk + 0.4 * toxicity_risk + 0.2 * metabolism_risk)
        
        # Generate recommendations
        recommendations = self._generate_drug_recommendations(drug_name, overall_risk, relevant_variants)
        
        # Contributing variants
        contributing_variants = [v.get('variant_id', 'unknown') for v in relevant_variants]
        
        # Risk factors from drug database
        risk_factors = drug_info.get('risk_factors', [])
        
        # Calculate confidence based on evidence
        confidence = self._calculate_drug_confidence(drug_name, relevant_variants)
        
        return DrugRiskProfile(
            drug_name=drug_name,
            overall_risk_score=overall_risk,
            efficacy_risk=efficacy_risk,
            toxicity_risk=toxicity_risk,
            metabolism_risk=metabolism_risk,
            contributing_variants=contributing_variants,
            risk_factors=risk_factors,
            recommendations=recommendations,
            confidence=confidence
        )
    
    def _calculate_efficacy_risk(self, drug_name: str, variants: List[Dict]) -> float:
        """Calculate risk of reduced drug efficacy."""
        
        if not variants:
            return 0.5  # Default moderate risk
        
        efficacy_risks = []
        
        for variant in variants:
            gene = variant.get('gene')
            impact = variant.get('impact', 'UNKNOWN')
            
            # Gene-specific efficacy risk calculation
            if gene in ['INPP1', 'GSK3B'] and drug_name == 'lithium':
                # These variants actually increase lithium sensitivity (reduce risk)
                efficacy_risks.append(0.2)
            elif gene in ['CYP2D6', 'CYP2C19']:
                # Metabolizer status affects efficacy
                if impact in ['HIGH', 'MODERATE']:
                    efficacy_risks.append(0.7)
                else:
                    efficacy_risks.append(0.4)
            else:
                # Default risk based on impact
                impact_risk = {'HIGH': 0.8, 'MODERATE': 0.6, 'LOW': 0.3}.get(impact, 0.5)
                efficacy_risks.append(impact_risk)
        
        return np.mean(efficacy_risks)
    
    def _calculate_toxicity_risk(self, drug_name: str, variants: List[Dict]) -> float:
        """Calculate risk of drug toxicity."""
        
        if not variants:
            return 0.3  # Default low-moderate risk
        
        toxicity_risks = []
        
        for variant in variants:
            gene = variant.get('gene')
            impact = variant.get('impact', 'UNKNOWN')
            
            # Gene-specific toxicity risk
            if gene in ['CYP2D6', 'CYP2C19']:
                # Poor metabolizers have higher toxicity risk
                if impact == 'HIGH':  # Assume HIGH impact = poor metabolism
                    toxicity_risks.append(0.8)
                else:
                    toxicity_risks.append(0.3)
            elif gene == 'SLCO1B1' and drug_name == 'atorvastatin':
                # SLCO1B1 variants increase statin toxicity risk
                toxicity_risks.append(0.7)
            else:
                # Default toxicity risk
                impact_risk = {'HIGH': 0.6, 'MODERATE': 0.4, 'LOW': 0.2}.get(impact, 0.3)
                toxicity_risks.append(impact_risk)
        
        return np.mean(toxicity_risks)
    
    def _calculate_metabolism_risk(self, drug_name: str, variants: List[Dict]) -> float:
        """Calculate risk of altered drug metabolism."""
        
        metabolism_genes = ['CYP2D6', 'CYP2C19', 'CYP3A4', 'UGT1A1']
        metabolism_variants = [v for v in variants if v.get('gene') in metabolism_genes]
        
        if not metabolism_variants:
            return 0.2  # Low risk if no metabolism gene variants
        
        risks = []
        for variant in metabolism_variants:
            impact = variant.get('impact', 'UNKNOWN')
            impact_risk = {'HIGH': 0.9, 'MODERATE': 0.6, 'LOW': 0.3}.get(impact, 0.5)
            risks.append(impact_risk)
        
        return np.mean(risks)
    
    def _generate_drug_recommendations(self, drug_name: str, overall_risk: float, 
                                     variants: List[Dict]) -> List[str]:
        """Generate recommendations based on drug risk profile."""
        
        recommendations = []
        
        if overall_risk > 0.7:
            recommendations.append(f"Consider alternative to {drug_name} due to high genetic risk")
            recommendations.append("Require close monitoring if {drug_name} is used")
            recommendations.append("Consider pharmacogenetic consultation")
        elif overall_risk > 0.5:
            recommendations.append(f"Use {drug_name} with caution")
            recommendations.append("Consider dose adjustment based on genetic profile")
            recommendations.append("Monitor for efficacy and adverse effects")
        else:
            recommendations.append(f"{drug_name} appears suitable based on genetic profile")
            recommendations.append("Standard dosing likely appropriate")
        
        # Add gene-specific recommendations
        for variant in variants:
            gene = variant.get('gene')
            if gene == 'CYP2D6':
                recommendations.append("Consider CYP2D6 phenotype for dosing")
            elif gene == 'SLCO1B1' and drug_name == 'atorvastatin':
                recommendations.append("Monitor for myopathy with statin use")
        
        return recommendations
    
    def _calculate_drug_confidence(self, drug_name: str, variants: List[Dict]) -> float:
        """Calculate confidence in drug risk assessment."""
        
        # Base confidence from drug database
        base_confidence = 0.7  # Moderate confidence
        
        # Increase confidence for well-studied drugs
        high_confidence_drugs = ['lithium', 'atorvastatin', 'citalopram']
        if drug_name in high_confidence_drugs:
            base_confidence = 0.8
        
        # Adjust based on variant evidence
        if not variants:
            return base_confidence * 0.5  # Lower confidence without variants
        
        evidence_boost = 0.0
        for variant in variants:
            gene = variant.get('gene')
            if gene in ['CYP2D6', 'CYP2C19', 'SLCO1B1']:
                evidence_boost += 0.1  # Strong evidence genes
            else:
                evidence_boost += 0.05  # Moderate evidence
        
        return min(0.95, base_confidence + evidence_boost)
    
    def create_pharmacogenetic_profile(self, individual_id: str, 
                                     variants: List[Dict[str, Any]]) -> PharmacogeneticProfile:
        """Create comprehensive pharmacogenetic profile for an individual."""
        
        # Assess all variants
        variant_assessments = []
        for variant in variants:
            assessment = self.assess_variant_risk(variant)
            variant_assessments.append(assessment)
        
        # Assess drug risks for common drugs
        common_drugs = ['lithium', 'aripiprazole', 'citalopram', 'atorvastatin', 'aspirin']
        drug_risk_profiles = []
        
        for drug in common_drugs:
            drug_profile = self.assess_drug_risk(drug, variants)
            drug_risk_profiles.append(drug_profile)
        
        # Calculate summary statistics
        total_variants = len(variants)
        high_risk_variants = len([v for v in variant_assessments 
                                if v.risk_category in ['HIGH', 'VERY_HIGH']])
        
        # Overall risk score (average of high-risk variant scores)
        high_risk_scores = [v.numerical_risk for v in variant_assessments 
                           if v.risk_category in ['HIGH', 'VERY_HIGH']]
        overall_risk_score = np.mean(high_risk_scores) if high_risk_scores else 0.3
        
        # High-risk drugs
        high_risk_drugs = [d.drug_name for d in drug_risk_profiles if d.overall_risk_score > 0.6]
        
        # Generate overall recommendations
        recommendations = self._generate_overall_recommendations(variant_assessments, drug_risk_profiles)
        
        return PharmacogeneticProfile(
            individual_id=individual_id,
            total_variants=total_variants,
            high_risk_variants=high_risk_variants,
            drug_risk_profiles=drug_risk_profiles,
            variant_assessments=variant_assessments,
            overall_risk_score=overall_risk_score,
            high_risk_drugs=high_risk_drugs,
            recommendations=recommendations
        )
    
    def _generate_overall_recommendations(self, variant_assessments: List[VariantRiskAssessment],
                                        drug_profiles: List[DrugRiskProfile]) -> List[str]:
        """Generate overall pharmacogenetic recommendations."""
        
        recommendations = []
        
        # High-risk variant recommendations
        high_risk_variants = [v for v in variant_assessments if v.risk_category in ['HIGH', 'VERY_HIGH']]
        if high_risk_variants:
            recommendations.append(f"Genetic testing identifies {len(high_risk_variants)} high-risk variants")
            recommendations.append("Pharmacogenetic consultation recommended before prescribing")
        
        # Drug-specific recommendations
        high_risk_drugs = [d for d in drug_profiles if d.overall_risk_score > 0.6]
        if high_risk_drugs:
            drug_names = [d.drug_name for d in high_risk_drugs]
            recommendations.append(f"High genetic risk for: {', '.join(drug_names)}")
        
        # General recommendations
        recommendations.append("Share genetic information with all prescribing physicians")
        recommendations.append("Consider pharmacogenetic testing for family members")
        recommendations.append("Keep genetic information updated with new discoveries")
        
        return recommendations
    
    def visualize_risk_profile(self, profile: PharmacogeneticProfile, 
                             output_dir: str = "babylon_results") -> None:
        """Create visualizations of pharmacogenetic risk profile."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Create comprehensive risk visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Variant risk distribution
        risk_categories = [v.risk_category for v in profile.variant_assessments]
        risk_counts = pd.Series(risk_categories).value_counts()
        
        axes[0, 0].bar(risk_counts.index, risk_counts.values, color='skyblue')
        axes[0, 0].set_title('Variant Risk Distribution')
        axes[0, 0].set_ylabel('Number of Variants')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Drug risk scores
        drug_names = [d.drug_name for d in profile.drug_risk_profiles]
        drug_risks = [d.overall_risk_score for d in profile.drug_risk_profiles]
        
        bars = axes[0, 1].bar(drug_names, drug_risks, color='salmon')
        axes[0, 1].set_title('Drug Risk Scores')
        axes[0, 1].set_ylabel('Risk Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='High Risk Threshold')
        axes[0, 1].legend()
        
        # Color bars based on risk level
        for bar, risk in zip(bars, drug_risks):
            if risk > 0.7:
                bar.set_color('red')
            elif risk > 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('green')
        
        # 3. Risk components for each drug
        risk_types = ['efficacy_risk', 'toxicity_risk', 'metabolism_risk']
        drug_data = []
        
        for drug_profile in profile.drug_risk_profiles:
            drug_data.append([
                drug_profile.efficacy_risk,
                drug_profile.toxicity_risk, 
                drug_profile.metabolism_risk
            ])
        
        drug_data = np.array(drug_data)
        
        im = axes[0, 2].imshow(drug_data.T, cmap='Reds', aspect='auto')
        axes[0, 2].set_title('Risk Components Heatmap')
        axes[0, 2].set_xlabel('Drugs')
        axes[0, 2].set_ylabel('Risk Types')
        axes[0, 2].set_xticks(range(len(drug_names)))
        axes[0, 2].set_xticklabels(drug_names, rotation=45)
        axes[0, 2].set_yticks(range(len(risk_types)))
        axes[0, 2].set_yticklabels(risk_types)
        plt.colorbar(im, ax=axes[0, 2])
        
        # 4. Genes affected
        genes = [v.gene for v in profile.variant_assessments]
        gene_counts = pd.Series(genes).value_counts().head(10)
        
        axes[1, 0].barh(gene_counts.index, gene_counts.values, color='lightgreen')
        axes[1, 0].set_title('Top Affected Genes')
        axes[1, 0].set_xlabel('Number of Variants')
        
        # 5. Risk vs Evidence scatter
        risks = [v.numerical_risk for v in profile.variant_assessments]
        evidence_scores = []
        
        for v in profile.variant_assessments:
            evidence_map = {'STRONG': 3, 'MODERATE': 2, 'WEAK': 1}
            evidence_scores.append(evidence_map.get(v.evidence_level, 1))
        
        scatter = axes[1, 1].scatter(risks, evidence_scores, 
                                   c=[1 if v.risk_category in ['HIGH', 'VERY_HIGH'] else 0 
                                      for v in profile.variant_assessments],
                                   cmap='RdYlGn_r', alpha=0.7)
        axes[1, 1].set_xlabel('Risk Score')
        axes[1, 1].set_ylabel('Evidence Level')
        axes[1, 1].set_title('Risk vs Evidence')
        axes[1, 1].set_yticks([1, 2, 3])
        axes[1, 1].set_yticklabels(['Weak', 'Moderate', 'Strong'])
        
        # 6. Overall risk summary
        summary_data = {
            'Total Variants': profile.total_variants,
            'High Risk Variants': profile.high_risk_variants,
            'High Risk Drugs': len(profile.high_risk_drugs),
            'Overall Risk Score': profile.overall_risk_score
        }
        
        axes[1, 2].text(0.1, 0.7, f"Total Variants: {summary_data['Total Variants']}", 
                       fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.6, f"High Risk Variants: {summary_data['High Risk Variants']}", 
                       fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.5, f"High Risk Drugs: {summary_data['High Risk Drugs']}", 
                       fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].text(0.1, 0.4, f"Overall Risk Score: {summary_data['Overall Risk Score']:.3f}", 
                       fontsize=12, transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Risk Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / 'pharmacogenetic_risk_profile.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Risk profile visualization saved to {output_path}")
    
    def save_results(self, profile: PharmacogeneticProfile, 
                    output_dir: str = "babylon_results") -> None:
        """Save pharmacogenetic profile results."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Convert profile to dictionary for JSON serialization
        profile_data = {
            'individual_id': profile.individual_id,
            'total_variants': profile.total_variants,
            'high_risk_variants': profile.high_risk_variants,
            'overall_risk_score': profile.overall_risk_score,
            'high_risk_drugs': profile.high_risk_drugs,
            'recommendations': profile.recommendations,
            'drug_risk_profiles': [
                {
                    'drug_name': d.drug_name,
                    'overall_risk_score': d.overall_risk_score,
                    'efficacy_risk': d.efficacy_risk,
                    'toxicity_risk': d.toxicity_risk,
                    'metabolism_risk': d.metabolism_risk,
                    'contributing_variants': d.contributing_variants,
                    'risk_factors': d.risk_factors,
                    'recommendations': d.recommendations,
                    'confidence': d.confidence
                }
                for d in profile.drug_risk_profiles
            ],
            'variant_assessments': [
                {
                    'variant_id': v.variant_id,
                    'gene': v.gene,
                    'risk_category': v.risk_category,
                    'numerical_risk': v.numerical_risk,
                    'affected_drugs': v.affected_drugs,
                    'mechanism': v.mechanism,
                    'population_frequency': v.population_frequency,
                    'clinical_significance': v.clinical_significance,
                    'evidence_level': v.evidence_level
                }
                for v in profile.variant_assessments
            ]
        }
        
        # Save as JSON
        with open(output_path / 'pharmacogenetic_profile.json', 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        # Save drug risks as CSV
        drug_df = pd.DataFrame([
            {
                'drug_name': d.drug_name,
                'overall_risk_score': d.overall_risk_score,
                'efficacy_risk': d.efficacy_risk,
                'toxicity_risk': d.toxicity_risk,
                'metabolism_risk': d.metabolism_risk,
                'confidence': d.confidence
            }
            for d in profile.drug_risk_profiles
        ])
        drug_df.to_csv(output_path / 'drug_risk_scores.csv', index=False)
        
        # Save variant assessments as CSV
        variant_df = pd.DataFrame([
            {
                'variant_id': v.variant_id,
                'gene': v.gene,
                'risk_category': v.risk_category,
                'numerical_risk': v.numerical_risk,
                'mechanism': v.mechanism,
                'clinical_significance': v.clinical_significance,
                'evidence_level': v.evidence_level
            }
            for v in profile.variant_assessments
        ])
        variant_df.to_csv(output_path / 'variant_risk_assessments.csv', index=False)
        
        logger.info(f"Pharmacogenetic profile saved to {output_path}")

def main():
    """
    Test the genetic risk assessor with sample pharmacogenetic variants.
    
    This demonstrates how genetic variants affect drug response risk
    and generates comprehensive pharmacogenetic profiles.
    """
    
    print("🧬 Testing Genetic Risk Assessment")
    print("=" * 50)
    
    # Sample pharmacogenetic variants (simulating Dante Labs results)
    sample_variants = [
        {
            'gene': 'CYP2D6',
            'variant_id': 'CYP2D6*4/*4',
            'impact': 'HIGH',
            'description': 'Poor metabolizer phenotype'
        },
        {
            'gene': 'CYP2C19',
            'variant_id': 'CYP2C19*2/*17',
            'impact': 'MODERATE',
            'description': 'Intermediate/rapid metabolizer'
        },
        {
            'gene': 'SLCO1B1',
            'variant_id': 'rs4149056',
            'impact': 'MODERATE',
            'description': 'Decreased function variant'
        },
        {
            'gene': 'INPP1',
            'variant_id': 'rs123456',
            'impact': 'HIGH',
            'description': 'Lithium sensitivity variant'
        },
        {
            'gene': 'GSK3B',
            'variant_id': 'rs789012',
            'impact': 'MODERATE',
            'description': 'Lithium sensitivity variant'
        },
        {
            'gene': 'DRD2',
            'variant_id': 'rs1800497',
            'impact': 'MODERATE',
            'description': 'Dopamine receptor variant'
        }
    ]
    
    # Initialize risk assessor
    assessor = GeneticRiskAssessor()
    
    # Create pharmacogenetic profile
    print(f"\n🔍 Analyzing {len(sample_variants)} pharmacogenetic variants...")
    profile = assessor.create_pharmacogenetic_profile("test_individual", sample_variants)
    
    # Display results
    print(f"\n📊 PHARMACOGENETIC PROFILE SUMMARY:")
    print("-" * 40)
    print(f"Individual ID: {profile.individual_id}")
    print(f"Total variants: {profile.total_variants}")
    print(f"High-risk variants: {profile.high_risk_variants}")
    print(f"Overall risk score: {profile.overall_risk_score:.3f}")
    print(f"High-risk drugs: {', '.join(profile.high_risk_drugs)}")
    
    print(f"\n💊 DRUG RISK PROFILES:")
    print("-" * 40)
    for drug_profile in profile.drug_risk_profiles:
        print(f"{drug_profile.drug_name}:")
        print(f"  Overall risk: {drug_profile.overall_risk_score:.3f}")
        print(f"  Efficacy risk: {drug_profile.efficacy_risk:.3f}")
        print(f"  Toxicity risk: {drug_profile.toxicity_risk:.3f}")
        print(f"  Metabolism risk: {drug_profile.metabolism_risk:.3f}")
        print(f"  Confidence: {drug_profile.confidence:.3f}")
        print()
    
    print(f"\n🧬 HIGH-RISK VARIANT DETAILS:")
    print("-" * 40)
    high_risk_variants = [v for v in profile.variant_assessments 
                         if v.risk_category in ['HIGH', 'VERY_HIGH']]
    
    for variant in high_risk_variants:
        print(f"{variant.gene} ({variant.variant_id}):")
        print(f"  Risk category: {variant.risk_category}")
        print(f"  Risk score: {variant.numerical_risk:.3f}")
        print(f"  Affected drugs: {', '.join(variant.affected_drugs)}")
        print(f"  Mechanism: {variant.mechanism}")
        print(f"  Evidence level: {variant.evidence_level}")
        print()
    
    # Save results and create visualizations
    print("💾 Saving results and creating visualizations...")
    assessor.save_results(profile)
    assessor.visualize_risk_profile(profile)
    
    print(f"\n💡 KEY RECOMMENDATIONS:")
    print("-" * 40)
    for i, rec in enumerate(profile.recommendations, 1):
        print(f"{i}. {rec}")
    
    print(f"\n📁 Results saved to: babylon_results/")
    print("\n✅ Genetic risk assessment complete!")
    
    return profile

if __name__ == "__main__":
    profile = main()