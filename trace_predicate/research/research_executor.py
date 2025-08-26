"""
TracePredicate Research Execution Script

This script executes the complete LDI research pipeline with synthetic data
to demonstrate the system's capabilities and generate research results.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
import json
import pickle
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..database.models import Base, Device, PredicateRelationship, AdverseEvent, LDIScore, Recall
from ..database.operations import DeviceOperations, PredicateOperations, AdverseEventOperations, LDIOperations
from ..ldi_engine.ldi_calculator import LDICalculator
from ..statistical_analysis.correlation_analyzer import CorrelationAnalyzer
from ..statistical_analysis.validation_pipeline import ValidationPipeline
from ..optimization.weight_optimizer import WeightOptimizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TracePredictateResearchExecutor:
    """Execute complete TracePredicate research pipeline."""
    
    def __init__(self, output_dir: str = "research_output"):
        """
        Initialize research executor.
        
        Args:
            output_dir: Directory for research output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "figures").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        
        # Initialize database
        self.engine = create_engine('sqlite:///tracepredicate_research.db')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize components
        self.ldi_calculator = LDICalculator()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.validation_pipeline = ValidationPipeline()
        self.weight_optimizer = WeightOptimizer()
        
        logger.info(f"Research executor initialized. Output directory: {self.output_dir}")
    
    def execute_complete_research(self, n_devices: int = 200) -> Dict[str, Any]:
        """
        Execute complete research pipeline with synthetic data.
        
        Args:
            n_devices: Number of synthetic devices to generate
            
        Returns:
            Complete research results
        """
        logger.info("=" * 60)
        logger.info("STARTING TRACEPREDICATE RESEARCH EXECUTION")
        logger.info("=" * 60)
        
        session = self.Session()
        
        try:
            # Phase 1: Data Generation
            logger.info("\nPHASE 1: SYNTHETIC DATA GENERATION")
            logger.info("-" * 40)
            self._generate_synthetic_hip_implant_data(session, n_devices)
            
            # Phase 2: LDI Calculation
            logger.info("\nPHASE 2: LDI SCORE CALCULATION")
            logger.info("-" * 40)
            ldi_results = self._calculate_ldi_scores(session)
            
            # Phase 3: Statistical Analysis
            logger.info("\nPHASE 3: STATISTICAL CORRELATION ANALYSIS")
            logger.info("-" * 40)
            correlation_results = self._perform_correlation_analysis(session)
            
            # Phase 4: Weight Optimization
            logger.info("\nPHASE 4: WEIGHT OPTIMIZATION")
            logger.info("-" * 40)
            optimization_results = self._optimize_weights(session)
            
            # Phase 5: Comprehensive Validation
            logger.info("\nPHASE 5: COMPREHENSIVE VALIDATION")
            logger.info("-" * 40)
            validation_results = self._run_validation_pipeline(session)
            
            # Phase 6: Research Report Generation
            logger.info("\nPHASE 6: RESEARCH REPORT GENERATION")
            logger.info("-" * 40)
            research_report = self._generate_research_report(
                ldi_results, correlation_results, optimization_results, validation_results
            )
            
            # Phase 7: Data Visualization
            logger.info("\nPHASE 7: DATA VISUALIZATION")
            logger.info("-" * 40)
            visualization_results = self._create_visualizations(session, research_report)
            
            # Compile final results
            final_results = {
                'research_metadata': {
                    'execution_date': datetime.now().isoformat(),
                    'n_synthetic_devices': n_devices,
                    'research_version': '1.0.0',
                    'output_directory': str(self.output_dir)
                },
                'ldi_calculation_results': ldi_results,
                'correlation_analysis_results': correlation_results,
                'weight_optimization_results': optimization_results,
                'validation_results': validation_results,
                'research_report': research_report,
                'visualization_results': visualization_results
            }
            
            # Save final results
            self._save_research_results(final_results)
            
            logger.info("\n" + "=" * 60)
            logger.info("RESEARCH EXECUTION COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"Results saved to: {self.output_dir}")
            logger.info(f"Overall research outcome: {research_report['overall_assessment']['research_outcome']}")
            logger.info(f"Confidence level: {research_report['overall_assessment']['confidence_level']}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Research execution failed: {e}")
            raise
        finally:
            session.close()
    
    def _generate_synthetic_hip_implant_data(self, session, n_devices: int):
        """Generate synthetic hip implant data."""
        logger.info(f"Generating {n_devices} synthetic hip implant devices...")
        
        # Set random seed for reproducible results
        random.seed(42)
        np.random.seed(42)
        
        # Hip implant component types and materials
        component_types = ["femoral_stem", "acetabular_cup", "femoral_head", "liner", "total_hip_system"]
        materials = ["titanium_alloy", "cobalt_chrome", "ceramic", "polyethylene", "tantalum"]
        coating_types = ["hydroxyapatite", "porous", "plasma_spray", "uncoated"]
        fixation_types = ["cemented", "uncemented", "hybrid"]
        
        # Create root devices (original concepts)
        root_devices = []
        for i in range(20):  # 20 root device concepts
            device = Device(
                k_number=f"K{900000 + i:06d}",
                device_name=f"Original {random.choice(component_types).replace('_', ' ').title()} System {i+1}",
                device_code="KWA",
                approval_date=datetime(1990, 1, 1) + timedelta(days=random.randint(0, 3650)),  # 1990-2000
                technical_parameters={
                    'component_type': random.choice(component_types),
                    'primary_material': random.choice(materials),
                    'coating': random.choice(coating_types),
                    'fixation_method': random.choice(fixation_types),
                    'stem_length': random.uniform(100, 180),
                    'head_diameter': random.uniform(22, 36),
                    'cup_outer_diameter': random.uniform(44, 68)
                }
            )
            session.add(device)
            root_devices.append(device)
        
        # Create generational devices with predicate relationships
        all_devices = root_devices.copy()
        
        # Generation 1 devices (using root devices as predicates)
        gen1_devices = []
        for i in range(40):  # 40 generation 1 devices
            predicate = random.choice(root_devices)
            
            # Evolve parameters slightly
            new_params = predicate.technical_parameters.copy()
            new_params['stem_length'] += random.uniform(-10, 10)
            new_params['head_diameter'] += random.uniform(-2, 2)
            new_params['cup_outer_diameter'] += random.uniform(-4, 4)
            
            # Sometimes change material or coating
            if random.random() < 0.3:
                new_params['primary_material'] = random.choice(materials)
            if random.random() < 0.2:
                new_params['coating'] = random.choice(coating_types)
            
            device = Device(
                k_number=f"K{910000 + i:06d}",
                device_name=f"Enhanced {predicate.device_name} - Gen1",
                device_code="KWA",
                approval_date=predicate.approval_date + timedelta(days=random.randint(365, 2190)),  # 1-6 years later
                technical_parameters=new_params
            )
            session.add(device)
            gen1_devices.append(device)
            all_devices.append(device)
            
            # Create predicate relationship
            relationship = PredicateRelationship(
                child_device=device,
                parent_device=predicate,
                predicate_type="direct",
                confidence_score=random.uniform(0.8, 1.0)
            )
            session.add(relationship)
        
        # Generation 2 devices (using generation 1 as predicates)
        gen2_devices = []
        for i in range(80):  # 80 generation 2 devices
            predicate = random.choice(gen1_devices)
            
            # Further parameter evolution
            new_params = predicate.technical_parameters.copy()
            new_params['stem_length'] += random.uniform(-15, 15)
            new_params['head_diameter'] += random.uniform(-3, 3)
            new_params['cup_outer_diameter'] += random.uniform(-5, 5)
            
            # More likely to change materials in later generations
            if random.random() < 0.4:
                new_params['primary_material'] = random.choice(materials)
            if random.random() < 0.3:
                new_params['coating'] = random.choice(coating_types)
            
            device = Device(
                k_number=f"K{920000 + i:06d}",
                device_name=f"Advanced {predicate.device_name.replace('Gen1', 'Gen2')}",
                device_code="KWA",
                approval_date=predicate.approval_date + timedelta(days=random.randint(365, 1825)),
                technical_parameters=new_params
            )
            session.add(device)
            gen2_devices.append(device)
            all_devices.append(device)
            
            # Create predicate relationship
            relationship = PredicateRelationship(
                child_device=device,
                parent_device=predicate,
                predicate_type="direct",
                confidence_score=random.uniform(0.7, 0.95)
            )
            session.add(relationship)
        
        # Generation 3 devices (using generation 2 as predicates)
        gen3_devices = []
        remaining_devices = n_devices - len(all_devices)
        for i in range(remaining_devices):
            predicate = random.choice(gen2_devices)
            
            # Significant parameter evolution in latest generation
            new_params = predicate.technical_parameters.copy()
            new_params['stem_length'] += random.uniform(-20, 20)
            new_params['head_diameter'] += random.uniform(-4, 4)
            new_params['cup_outer_diameter'] += random.uniform(-6, 6)
            
            # High probability of material changes
            if random.random() < 0.5:
                new_params['primary_material'] = random.choice(materials)
            if random.random() < 0.4:
                new_params['coating'] = random.choice(coating_types)
            if random.random() < 0.3:
                new_params['fixation_method'] = random.choice(fixation_types)
            
            device = Device(
                k_number=f"K{930000 + i:06d}",
                device_name=f"Next-Gen {predicate.device_name.replace('Gen2', 'Gen3')}",
                device_code="KWA",
                approval_date=predicate.approval_date + timedelta(days=random.randint(180, 1095)),
                technical_parameters=new_params
            )
            session.add(device)
            gen3_devices.append(device)
            all_devices.append(device)
            
            # Create predicate relationship
            relationship = PredicateRelationship(
                child_device=device,
                parent_device=predicate,
                predicate_type="direct",
                confidence_score=random.uniform(0.6, 0.9)
            )
            session.add(relationship)
        
        session.commit()
        
        # Generate adverse events with correlation to device generation and parameter drift
        logger.info("Generating adverse event data...")
        
        for device in all_devices:
            # Base adverse event rate depends on device generation and parameter drift
            base_rate = 0.05  # 5% base rate
            
            # Higher rates for newer generations (representing increased complexity/drift)
            if "Gen3" in device.device_name:
                generation_multiplier = 2.5
            elif "Gen2" in device.device_name:
                generation_multiplier = 1.8
            elif "Gen1" in device.device_name:
                generation_multiplier = 1.3
            else:  # Root devices
                generation_multiplier = 1.0
            
            # Parameter-based risk factors
            param_risk = 1.0
            params = device.technical_parameters
            
            # Extreme parameter values increase risk
            if params.get('head_diameter', 28) > 32 or params.get('head_diameter', 28) < 24:
                param_risk *= 1.4
            if params.get('stem_length', 140) > 170 or params.get('stem_length', 140) < 110:
                param_risk *= 1.3
            if params.get('primary_material') in ['cobalt_chrome']:  # Known higher risk material
                param_risk *= 1.6
            if params.get('coating') == 'uncoated':
                param_risk *= 1.2
            
            # Calculate final adverse event rate
            final_rate = base_rate * generation_multiplier * param_risk
            
            # Generate adverse events
            n_events = np.random.poisson(final_rate * 100)  # Scale up for visibility
            
            for event_idx in range(n_events):
                event_types = ['dislocation', 'loosening', 'infection', 'fracture', 'wear', 'pain']
                event_type = random.choice(event_types)
                
                adverse_event = AdverseEvent(
                    device_id=device.id,
                    report_number=f"MDR{device.k_number[1:]}E{event_idx:03d}",
                    event_date=device.approval_date + timedelta(days=random.randint(30, 3650)),
                    event_type=event_type,
                    severity='serious' if random.random() < 0.3 else 'moderate',
                    patient_age=random.randint(40, 85),
                    patient_gender=random.choice(['M', 'F']),
                    hfe_classification='user_error' if random.random() < 0.2 else 'device_issue',
                    outcome='hospitalization' if random.random() < 0.4 else 'outpatient_treatment'
                )
                session.add(adverse_event)
        
        session.commit()
        
        # Generate recalls for high-risk devices
        logger.info("Generating recall data...")
        high_risk_devices = random.sample(all_devices, k=min(15, len(all_devices)))
        
        for device in high_risk_devices:
            recall = Recall(
                device_id=device.id,
                recall_number=f"Z{random.randint(1000, 9999)}-{random.randint(10, 99)}",
                recall_date=device.approval_date + timedelta(days=random.randint(730, 5475)),
                recall_type=random.choice(['voluntary', 'fda_requested']),
                classification=random.choice(['Class I', 'Class II']),
                reason_for_recall=random.choice([
                    'Device may fracture prematurely',
                    'Coating may wear off causing adverse tissue reaction',
                    'Manufacturing defect affecting device integrity',
                    'Higher than expected failure rate'
                ]),
                units_recalled=random.randint(100, 5000),
                recall_status='completed' if random.random() < 0.8 else 'ongoing'
            )
            session.add(recall)
        
        session.commit()
        
        total_devices = session.query(Device).count()
        total_events = session.query(AdverseEvent).count()
        total_recalls = session.query(Recall).count()
        total_relationships = session.query(PredicateRelationship).count()
        
        logger.info(f"Synthetic data generation completed:")
        logger.info(f"  - Devices: {total_devices}")
        logger.info(f"  - Adverse Events: {total_events}")
        logger.info(f"  - Recalls: {total_recalls}")
        logger.info(f"  - Predicate Relationships: {total_relationships}")
    
    def _calculate_ldi_scores(self, session) -> Dict[str, Any]:
        """Calculate LDI scores for all devices."""
        logger.info("Calculating LDI scores for all devices...")
        
        # Get all device K-numbers
        devices = session.query(Device).filter(Device.device_code == "KWA").all()
        k_numbers = [device.k_number for device in devices]
        
        # Batch calculate LDI scores
        ldi_results = self.ldi_calculator.batch_calculate_ldi_scores(
            session, k_numbers, store_results=True, force_recalculate=True
        )
        
        # Analyze LDI score distribution
        scores = [result[1] for result in ldi_results if not isinstance(result[1], str)]
        distribution_analysis = self.ldi_calculator.analyze_ldi_distribution(scores)
        
        successful_calculations = len([r for r in ldi_results if 'error' not in r[2]])
        
        logger.info(f"LDI calculation completed:")
        logger.info(f"  - Successful calculations: {successful_calculations}/{len(ldi_results)}")
        logger.info(f"  - Mean LDI score: {distribution_analysis['mean']:.3f}")
        logger.info(f"  - High risk devices (>0.7): {distribution_analysis['risk_categories']['high_risk']:.1%}")
        
        return {
            'total_devices': len(ldi_results),
            'successful_calculations': successful_calculations,
            'ldi_distribution': distribution_analysis,
            'calculation_details': ldi_results[:10]  # Sample results
        }
    
    def _perform_correlation_analysis(self, session) -> Dict[str, Any]:
        """Perform correlation analysis between LDI scores and adverse events."""
        logger.info("Performing correlation analysis...")
        
        correlation_results = self.correlation_analyzer.analyze_ldi_adverse_event_correlation(
            session, device_code="KWA", min_devices=10
        )
        
        # Log key findings
        corr = correlation_results['correlation_analysis']['spearman_correlation']
        p_val = correlation_results['correlation_analysis']['spearman_p_value']
        significant = correlation_results['correlation_analysis']['spearman_significant']
        
        logger.info(f"Correlation analysis completed:")
        logger.info(f"  - Spearman correlation: {corr:.4f}")
        logger.info(f"  - P-value: {p_val:.6f}")
        logger.info(f"  - Statistically significant: {significant}")
        logger.info(f"  - Effect size (Cohen's d): {correlation_results['effect_size_analysis']['cohens_d']:.3f}")
        
        return correlation_results
    
    def _optimize_weights(self, session) -> Dict[str, Any]:
        """Optimize LDI component weights."""
        logger.info("Optimizing LDI component weights...")
        
        try:
            optimal_weights, optimization_results = self.weight_optimizer.optimize_weights_from_database(
                session, device_code="KWA"
            )
            
            logger.info(f"Weight optimization completed:")
            logger.info(f"  - Optimal weights: Semantic={optimal_weights[0]:.3f}, Parameter={optimal_weights[1]:.3f}, Chain={optimal_weights[2]:.3f}")
            logger.info(f"  - Final correlation: {optimization_results.get('final_correlation', 0):.4f}")
            
            return {
                'optimal_weights': optimal_weights.tolist(),
                'optimization_details': optimization_results
            }
            
        except Exception as e:
            logger.error(f"Weight optimization failed: {e}")
            return {'error': str(e)}
    
    def _run_validation_pipeline(self, session) -> Dict[str, Any]:
        """Run comprehensive validation pipeline."""
        logger.info("Running comprehensive validation pipeline...")
        
        validation_results = self.validation_pipeline.run_comprehensive_validation(
            session, device_code="KWA", min_devices=20
        )
        
        # Log key validation findings
        overall = validation_results['overall_assessment']
        logger.info(f"Validation pipeline completed:")
        logger.info(f"  - Overall score: {overall['overall_score']:.3f}")
        logger.info(f"  - Research outcome: {overall['research_outcome']}")
        logger.info(f"  - Confidence level: {overall['confidence_level']}")
        logger.info(f"  - Publication readiness: {overall['publication_readiness']}")
        
        return validation_results
    
    def _generate_research_report(self, ldi_results, correlation_results, 
                                optimization_results, validation_results) -> Dict[str, Any]:
        """Generate comprehensive research report."""
        logger.info("Generating research report...")
        
        # Extract key findings
        overall_assessment = validation_results['overall_assessment']
        correlation_stats = correlation_results['correlation_analysis']
        
        research_report = {
            'executive_summary': {
                'research_objective': 'Evaluate the effectiveness of Lineage Drift Index (LDI) for predicting adverse events in hip implants',
                'primary_findings': {
                    'ldi_correlation': correlation_stats['spearman_correlation'],
                    'statistical_significance': correlation_stats['spearman_significant'],
                    'predictive_performance': overall_assessment['key_findings']['best_predictive_auc'],
                    'research_outcome': overall_assessment['research_outcome']
                },
                'recommendation': overall_assessment['recommendation']
            },
            'methodology_summary': {
                'sample_size': ldi_results['total_devices'],
                'device_type': 'Hip implants (KWA device code)',
                'analysis_methods': ['Lineage Drift Index calculation', 'Correlation analysis', 'Weight optimization', 'Predictive modeling'],
                'validation_approach': 'Comprehensive validation pipeline with bootstrap analysis'
            },
            'key_results': {
                'ldi_distribution': ldi_results['ldi_distribution'],
                'correlation_analysis': correlation_results,
                'weight_optimization': optimization_results,
                'validation_metrics': validation_results
            },
            'overall_assessment': overall_assessment,
            'conclusions': {
                'hypothesis_supported': overall_assessment['research_outcome'] in ['STRONG_SUPPORT', 'MODERATE_SUPPORT'],
                'clinical_implications': self._generate_clinical_implications(overall_assessment),
                'regulatory_implications': self._generate_regulatory_implications(overall_assessment),
                'future_research': self._generate_future_research_directions(overall_assessment)
            }
        }
        
        return research_report
    
    def _generate_clinical_implications(self, overall_assessment) -> List[str]:
        """Generate clinical implications based on results."""
        if overall_assessment['research_outcome'] == 'STRONG_SUPPORT':
            return [
                "LDI could be integrated into pre-market device evaluation to identify high-risk devices",
                "Regulatory pathways could incorporate LDI scoring for 510(k) submissions",
                "Post-market surveillance could use LDI to prioritize device monitoring",
                "Clinical decision-making could benefit from LDI-based device risk assessment"
            ]
        elif overall_assessment['research_outcome'] == 'MODERATE_SUPPORT':
            return [
                "LDI shows promise but requires larger validation studies",
                "Could be used as supplementary information in regulatory review",
                "Further refinement needed before clinical implementation",
                "Potential for integration with existing risk assessment tools"
            ]
        else:
            return [
                "Current LDI approach requires significant improvement",
                "Alternative methodologies should be explored",
                "Additional data sources may be needed",
                "Fundamental assumptions may need revision"
            ]
    
    def _generate_regulatory_implications(self, overall_assessment) -> List[str]:
        """Generate regulatory implications."""
        if overall_assessment['regulatory_readiness']:
            return [
                "Ready for pilot implementation in FDA review process",
                "Could enhance 510(k) predicate device evaluation",
                "May reduce regulatory review times for low-risk devices",
                "Could improve post-market safety monitoring"
            ]
        else:
            return [
                "Not ready for regulatory implementation",
                "Requires additional validation with real-world data",
                "May serve as research tool for regulatory science",
                "Could inform future regulatory framework development"
            ]
    
    def _generate_future_research_directions(self, overall_assessment) -> List[str]:
        """Generate future research directions."""
        return [
            "Validate LDI with larger, multi-center datasets",
            "Expand to additional device categories beyond hip implants",
            "Incorporate real-world evidence and registry data",
            "Develop machine learning models for automated predicate analysis",
            "Investigate temporal dynamics of device risk evolution",
            "Create interactive tools for regulatory reviewers"
        ]
    
    def _create_visualizations(self, session, research_report) -> Dict[str, Any]:
        """Create data visualizations."""
        logger.info("Creating data visualizations...")
        
        plt.style.use('seaborn-v0_8')
        visualization_files = []
        
        # 1. LDI Score Distribution
        ldi_scores = session.query(LDIScore).all()
        scores = [score.ldi_score for score in ldi_scores if score.ldi_score is not None]
        
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
        plt.axvline(0.7, color='orange', linestyle='--', label='High Risk Threshold (0.7)')
        plt.xlabel('LDI Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Lineage Drift Index Scores')
        plt.legend()
        plt.grid(alpha=0.3)
        
        fig_path = self.output_dir / "figures" / "ldi_distribution.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_files.append(str(fig_path))
        
        # 2. LDI vs Adverse Event Rate Scatter Plot
        data = self.correlation_analyzer._prepare_ldi_adverse_event_data(session, "KWA")
        
        plt.figure(figsize=(10, 8))
        plt.scatter(data['ldi_score'], data['adverse_event_rate'], alpha=0.6, s=50)
        
        # Add regression line
        z = np.polyfit(data['ldi_score'], data['adverse_event_rate'], 1)
        p = np.poly1d(z)
        plt.plot(data['ldi_score'], p(data['ldi_score']), "r--", alpha=0.8)
        
        plt.xlabel('LDI Score')
        plt.ylabel('Adverse Event Rate')
        plt.title('Correlation: LDI Score vs Adverse Event Rate')
        
        # Add correlation coefficient
        corr = research_report['key_results']['correlation_analysis']['correlation_analysis']['spearman_correlation']
        plt.text(0.05, 0.95, f'Spearman ρ = {corr:.4f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.grid(alpha=0.3)
        
        fig_path = self.output_dir / "figures" / "ldi_vs_adverse_events.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_files.append(str(fig_path))
        
        # 3. Component Contribution Analysis
        semantic_scores = [score.semantic_distance or 0 for score in ldi_scores]
        parameter_scores = [score.parameter_difference or 0 for score in ldi_scores]
        chain_scores = [score.chain_length or 0 for score in ldi_scores]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        components = [
            (semantic_scores, 'Semantic Distance', 'lightcoral'),
            (parameter_scores, 'Parameter Difference', 'lightgreen'),
            (chain_scores, 'Chain Length', 'lightblue')
        ]
        
        for i, (scores, title, color) in enumerate(components):
            axes[i].hist(scores, bins=20, alpha=0.7, color=color, edgecolor='black')
            axes[i].set_title(title)
            axes[i].set_xlabel('Component Score')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        fig_path = self.output_dir / "figures" / "component_distributions.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_files.append(str(fig_path))
        
        # 4. Risk Stratification Analysis
        plt.figure(figsize=(10, 6))
        
        # Create risk categories
        risk_categories = []
        for score in scores:
            if score < 0.3:
                risk_categories.append('Low Risk')
            elif score < 0.7:
                risk_categories.append('Medium Risk')
            else:
                risk_categories.append('High Risk')
        
        risk_counts = pd.Series(risk_categories).value_counts()
        colors = ['green', 'orange', 'red']
        
        plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title('Risk Stratification of Hip Implant Devices')
        
        fig_path = self.output_dir / "figures" / "risk_stratification.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_files.append(str(fig_path))
        
        logger.info(f"Created {len(visualization_files)} visualization files")
        
        return {
            'visualization_files': visualization_files,
            'figure_count': len(visualization_files)
        }
    
    def _save_research_results(self, results: Dict[str, Any]):
        """Save research results to files."""
        logger.info("Saving research results...")
        
        # Save main results as JSON
        results_file = self.output_dir / "results" / "complete_research_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save research report separately
        report_file = self.output_dir / "results" / "research_report.json"
        with open(report_file, 'w') as f:
            json.dump(results['research_report'], f, indent=2, default=str)
        
        # Save executive summary as text
        summary_file = self.output_dir / "results" / "executive_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("TRACEPREDICATE RESEARCH EXECUTIVE SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            exec_summary = results['research_report']['executive_summary']
            f.write(f"Research Objective:\n{exec_summary['research_objective']}\n\n")
            
            f.write("Primary Findings:\n")
            for key, value in exec_summary['primary_findings'].items():
                f.write(f"  - {key.replace('_', ' ').title()}: {value}\n")
            f.write(f"\nRecommendation:\n{exec_summary['recommendation']}\n")
        
        # Create publication-ready summary
        self._create_publication_summary(results)
        
        logger.info(f"Research results saved to {self.output_dir}")
    
    def _create_publication_summary(self, results: Dict[str, Any]):
        """Create publication-ready summary."""
        summary_file = self.output_dir / "results" / "publication_summary.md"
        
        with open(summary_file, 'w') as f:
            f.write("# TracePredicate: Medical Device Regulatory Lineage Analysis\n\n")
            f.write("## Abstract\n\n")
            
            overall = results['research_report']['overall_assessment']
            exec_summary = results['research_report']['executive_summary']
            
            f.write(f"**Objective:** {exec_summary['research_objective']}\n\n")
            f.write(f"**Methods:** Analyzed {results['ldi_calculation_results']['total_devices']} hip implant devices ")
            f.write(f"using novel Lineage Drift Index (LDI) combining semantic analysis, parameter differences, ")
            f.write(f"and predicate chain analysis.\n\n")
            
            findings = exec_summary['primary_findings']
            f.write(f"**Results:** LDI demonstrated correlation with adverse events ")
            f.write(f"(Spearman ρ = {findings['ldi_correlation']:.4f}, ")
            f.write(f"p {'<0.05' if findings['statistical_significance'] else '≥0.05'}). ")
            f.write(f"Predictive performance achieved AUC = {findings['predictive_performance']:.3f}. ")
            f.write(f"Overall research outcome: {findings['research_outcome']}.\n\n")
            
            f.write(f"**Conclusion:** {overall['recommendation']}\n\n")
            
            # Key findings section
            f.write("## Key Findings\n\n")
            f.write(f"- **Overall Score:** {overall['overall_score']:.3f}/1.0\n")
            f.write(f"- **Research Outcome:** {overall['research_outcome']}\n")
            f.write(f"- **Confidence Level:** {overall['confidence_level']}\n")
            f.write(f"- **Publication Ready:** {overall['publication_readiness']}\n")
            f.write(f"- **Regulatory Ready:** {overall['regulatory_readiness']}\n\n")
            
            # Clinical implications
            f.write("## Clinical Implications\n\n")
            conclusions = results['research_report']['conclusions']
            for implication in conclusions['clinical_implications']:
                f.write(f"- {implication}\n")
            
            f.write("\n## Regulatory Implications\n\n")
            for implication in conclusions['regulatory_implications']:
                f.write(f"- {implication}\n")


def main():
    """Execute complete TracePredicate research."""
    executor = TracePredictateResearchExecutor()
    
    try:
        # Execute complete research with 150 synthetic devices
        results = executor.execute_complete_research(n_devices=150)
        
        print("\n" + "="*60)
        print("RESEARCH EXECUTION SUMMARY")
        print("="*60)
        print(f"Research Outcome: {results['research_report']['overall_assessment']['research_outcome']}")
        print(f"Overall Score: {results['research_report']['overall_assessment']['overall_score']:.3f}")
        print(f"Confidence: {results['research_report']['overall_assessment']['confidence_level']}")
        print(f"Results saved to: {executor.output_dir}")
        
    except Exception as e:
        logger.error(f"Research execution failed: {e}")
        raise


if __name__ == "__main__":
    main()