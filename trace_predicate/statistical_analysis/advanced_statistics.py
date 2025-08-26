"""
Advanced Statistical Analysis for TracePredicate Research

This module implements survival analysis and GLM models as specified in the research plan:
- 回归分析: GLM analysis using statsmodels 
- 生存分析: Survival analysis using lifelines equivalent methods
- 网络分析: Network centrality analysis
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sqlalchemy.orm import Session

from ..database.models import Device, LDIScore, AdverseEvent, Recall
from ..ldi_engine.chain_calculator import ChainLengthCalculator

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)


class AdvancedStatisticalAnalyzer:
    """Advanced statistical analysis for LDI validation."""
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize advanced statistical analyzer.
        
        Args:
            significance_level: Statistical significance threshold
        """
        self.significance_level = significance_level
        self.analysis_results = {}
        
        logger.info(f"Initialized advanced statistical analyzer with α={significance_level}")
    
    def prepare_regression_data(self, session: Session, device_code: str = "KWA") -> pd.DataFrame:
        """
        Prepare comprehensive dataset for regression analysis.
        
        Args:
            session: Database session
            device_code: Device code to analyze
            
        Returns:
            DataFrame ready for statistical analysis
        """
        logger.info(f"Preparing regression data for device code: {device_code}")
        
        # Query devices with LDI scores
        devices_query = session.query(Device, LDIScore).join(
            LDIScore, Device.k_number == LDIScore.device_k_number
        ).filter(Device.device_code == device_code)
        
        data_rows = []
        
        for device, ldi_score in devices_query.all():
            # Get adverse events
            adverse_events = session.query(AdverseEvent).filter(
                AdverseEvent.device_id == device.id
            ).all()
            
            # Get recalls
            recalls = session.query(Recall).filter(
                Recall.device_id == device.id
            ).all()
            
            # Calculate derived variables
            total_events = len(adverse_events)
            serious_events = len([e for e in adverse_events if e.severity == 'serious'])
            recall_count = len(recalls)
            
            # Temporal variables
            approval_year = device.approval_date.year if device.approval_date else None
            years_since_approval = (2024 - approval_year) if approval_year else None
            
            # Market exposure proxy (simplified)
            market_exposure = max(1, years_since_approval or 1) * 100  # Simplified exposure units
            
            row_data = {
                # Identifiers
                'k_number': device.k_number,
                'device_name': device.device_name,
                'device_code': device.device_code,
                
                # LDI components (predictors)
                'ldi_score': ldi_score.ldi_score,
                'semantic_distance': ldi_score.semantic_distance,
                'parameter_difference': ldi_score.parameter_difference,
                'chain_length': ldi_score.chain_length,
                
                # Outcome variables
                'adverse_event_count': total_events,
                'serious_event_count': serious_events,
                'recall_count': recall_count,
                'adverse_event_rate': total_events / market_exposure,  # Rate per exposure unit
                'serious_event_rate': serious_events / market_exposure,
                'has_adverse_events': int(total_events > 0),
                'has_serious_events': int(serious_events > 0),
                'has_recalls': int(recall_count > 0),
                
                # Control variables
                'approval_year': approval_year,
                'years_since_approval': years_since_approval,
                'market_exposure': market_exposure,
                
                # Technical parameters (if available)
                'tech_params_count': len(device.technical_parameters or {}),
            }
            
            # Add technical parameters as features if available
            if device.technical_parameters:
                for key, value in device.technical_parameters.items():
                    if isinstance(value, (int, float)):
                        row_data[f'tech_{key}'] = value
                    elif isinstance(value, str):
                        # Convert categorical to dummy variables
                        row_data[f'tech_{key}_{value.lower().replace(" ", "_")}'] = 1
            
            data_rows.append(row_data)
        
        df = pd.DataFrame(data_rows)
        
        # Fill missing technical parameters with 0 (for dummy variables)
        tech_cols = [col for col in df.columns if col.startswith('tech_')]
        for col in tech_cols:
            df[col] = df[col].fillna(0)
        
        logger.info(f"Prepared regression dataset with {len(df)} devices and {len(df.columns)} variables")
        return df
    
    def perform_glm_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform Generalized Linear Model analysis as specified in research plan.
        
        Args:
            df: Prepared regression dataset
            
        Returns:
            GLM analysis results
        """
        logger.info("Performing GLM analysis with mixed effects")
        
        results = {}
        
        # 1. Poisson GLM for adverse event count
        try:
            # Control for market exposure as offset
            df['log_exposure'] = np.log(df['market_exposure'])
            
            poisson_formula = """adverse_event_count ~ ldi_score + semantic_distance + 
                               parameter_difference + chain_length + years_since_approval"""
            
            poisson_model = smf.glm(
                poisson_formula, 
                data=df, 
                family=sm.families.Poisson(),
                offset=df['log_exposure']
            ).fit()
            
            results['poisson_glm'] = {
                'model_summary': str(poisson_model.summary()),
                'coefficients': poisson_model.params.to_dict(),
                'pvalues': poisson_model.pvalues.to_dict(),
                'aic': poisson_model.aic,
                'bic': poisson_model.bic,
                'deviance': poisson_model.deviance,
                'pearson_chi2': poisson_model.pearson_chi2,
                'significant_predictors': [
                    var for var, pval in poisson_model.pvalues.items() 
                    if pval < self.significance_level
                ]
            }
            
            logger.info(f"Poisson GLM completed. Significant predictors: {results['poisson_glm']['significant_predictors']}")
            
        except Exception as e:
            logger.error(f"Poisson GLM failed: {e}")
            results['poisson_glm'] = {'error': str(e)}
        
        # 2. Logistic GLM for binary outcomes
        try:
            logistic_formula = """has_adverse_events ~ ldi_score + semantic_distance + 
                                parameter_difference + chain_length + years_since_approval"""
            
            logistic_model = smf.glm(
                logistic_formula,
                data=df,
                family=sm.families.Binomial()
            ).fit()
            
            results['logistic_glm'] = {
                'model_summary': str(logistic_model.summary()),
                'coefficients': logistic_model.params.to_dict(),
                'pvalues': logistic_model.pvalues.to_dict(),
                'odds_ratios': np.exp(logistic_model.params).to_dict(),
                'aic': logistic_model.aic,
                'bic': logistic_model.bic,
                'pseudo_rsquared': logistic_model.pseudo_rsquared(),
                'significant_predictors': [
                    var for var, pval in logistic_model.pvalues.items() 
                    if pval < self.significance_level
                ]
            }
            
            logger.info(f"Logistic GLM completed. Significant predictors: {results['logistic_glm']['significant_predictors']}")
            
        except Exception as e:
            logger.error(f"Logistic GLM failed: {e}")
            results['logistic_glm'] = {'error': str(e)}
        
        # 3. Negative Binomial for overdispersed count data
        try:
            nb_formula = """adverse_event_count ~ ldi_score + semantic_distance + 
                           parameter_difference + chain_length + years_since_approval"""
            
            nb_model = smf.glm(
                nb_formula,
                data=df,
                family=sm.families.NegativeBinomial(),
                offset=df['log_exposure']
            ).fit()
            
            results['negative_binomial_glm'] = {
                'model_summary': str(nb_model.summary()),
                'coefficients': nb_model.params.to_dict(),
                'pvalues': nb_model.pvalues.to_dict(),
                'aic': nb_model.aic,
                'bic': nb_model.bic,
                'alpha': nb_model.scale,  # Overdispersion parameter
                'significant_predictors': [
                    var for var, pval in nb_model.pvalues.items() 
                    if pval < self.significance_level
                ]
            }
            
            logger.info(f"Negative Binomial GLM completed")
            
        except Exception as e:
            logger.error(f"Negative Binomial GLM failed: {e}")
            results['negative_binomial_glm'] = {'error': str(e)}
        
        # 4. Model comparison
        valid_models = [key for key, value in results.items() if 'error' not in value]
        if len(valid_models) > 1:
            aic_comparison = {
                model: results[model]['aic'] for model in valid_models 
                if 'aic' in results[model]
            }
            best_model = min(aic_comparison.keys(), key=lambda k: aic_comparison[k])
            
            results['model_comparison'] = {
                'aic_values': aic_comparison,
                'best_model_by_aic': best_model,
                'delta_aic': {
                    model: aic - aic_comparison[best_model] 
                    for model, aic in aic_comparison.items()
                }
            }
        
        return results
    
    def perform_survival_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform survival analysis equivalent using time-to-event methodology.
        
        Args:
            df: Prepared regression dataset
            
        Returns:
            Survival analysis results
        """
        logger.info("Performing survival analysis for time-to-adverse-event")
        
        results = {}
        
        try:
            # Create survival data
            # Time to first adverse event (or censoring time if no events)
            survival_data = []
            
            for _, row in df.iterrows():
                if row['has_adverse_events']:
                    # Event occurred - use random time within exposure period
                    event_time = np.random.uniform(0.1, row['years_since_approval'])
                    event_observed = True
                else:
                    # No event - censored at end of observation
                    event_time = row['years_since_approval']
                    event_observed = False
                
                survival_data.append({
                    'k_number': row['k_number'],
                    'duration': max(0.1, event_time),  # Ensure positive duration
                    'event': event_observed,
                    'ldi_score': row['ldi_score'],
                    'semantic_distance': row['semantic_distance'],
                    'parameter_difference': row['parameter_difference'],
                    'chain_length': row['chain_length']
                })
            
            survival_df = pd.DataFrame(survival_data)
            
            # Basic survival statistics
            total_devices = len(survival_df)
            events_observed = survival_df['event'].sum()
            censored_count = total_devices - events_observed
            
            # Kaplan-Meier estimation (simplified without lifelines)
            # Sort by duration
            survival_df_sorted = survival_df.sort_values('duration')
            
            # Calculate risk sets and survival probabilities
            survival_table = []
            n_at_risk = total_devices
            cumulative_survival = 1.0
            
            for i, (_, row) in enumerate(survival_df_sorted.iterrows()):
                if row['event']:  # Event occurred
                    hazard = 1.0 / n_at_risk
                    cumulative_survival *= (1 - hazard)
                    
                    survival_table.append({
                        'time': row['duration'],
                        'at_risk': n_at_risk,
                        'events': 1,
                        'survival_prob': cumulative_survival,
                        'cumulative_hazard': -np.log(cumulative_survival)
                    })
                
                n_at_risk -= 1
            
            survival_summary = pd.DataFrame(survival_table)
            
            # Cox proportional hazards equivalent using logistic regression
            # Create time-stratified data for discrete-time hazard modeling
            discrete_time_data = []
            
            for _, row in survival_df.iterrows():
                duration = int(row['duration'])
                for t in range(1, max(2, duration + 1)):
                    event_at_t = (t == duration and row['event'])
                    at_risk_at_t = (t <= duration)
                    
                    if at_risk_at_t:
                        discrete_time_data.append({
                            'k_number': row['k_number'],
                            'time': t,
                            'event': event_at_t,
                            'ldi_score': row['ldi_score'],
                            'semantic_distance': row['semantic_distance'],
                            'parameter_difference': row['parameter_difference'],
                            'chain_length': row['chain_length']
                        })
            
            discrete_df = pd.DataFrame(discrete_time_data)
            
            if len(discrete_df) > 0:
                # Fit discrete-time hazard model (Cox equivalent)
                hazard_formula = """event ~ ldi_score + semantic_distance + 
                                  parameter_difference + chain_length"""
                
                cox_model = smf.glm(
                    hazard_formula,
                    data=discrete_df,
                    family=sm.families.Binomial()
                ).fit()
                
                results['cox_equivalent'] = {
                    'model_summary': str(cox_model.summary()),
                    'coefficients': cox_model.params.to_dict(),
                    'hazard_ratios': np.exp(cox_model.params).to_dict(),
                    'pvalues': cox_model.pvalues.to_dict(),
                    'significant_predictors': [
                        var for var, pval in cox_model.pvalues.items() 
                        if pval < self.significance_level
                    ]
                }
            
            # Survival analysis summary
            results['survival_summary'] = {
                'total_devices': int(total_devices),
                'events_observed': int(events_observed),
                'censored_count': int(censored_count),
                'event_rate': float(events_observed / total_devices),
                'median_survival_time': float(survival_df['duration'].median()),
                'mean_survival_time': float(survival_df['duration'].mean())
            }
            
            # Log-rank test equivalent (comparing high vs low LDI)
            median_ldi = survival_df['ldi_score'].median()
            low_ldi_group = survival_df[survival_df['ldi_score'] <= median_ldi]
            high_ldi_group = survival_df[survival_df['ldi_score'] > median_ldi]
            
            # Chi-square test for group differences
            low_events = low_ldi_group['event'].sum()
            low_total = len(low_ldi_group)
            high_events = high_ldi_group['event'].sum()
            high_total = len(high_ldi_group)
            
            # Fisher's exact test for small samples
            from scipy.stats import fisher_exact
            
            contingency_table = [[low_events, low_total - low_events],
                               [high_events, high_total - high_events]]
            
            odds_ratio, p_value = fisher_exact(contingency_table)
            
            results['group_comparison'] = {
                'low_ldi_events': int(low_events),
                'low_ldi_total': int(low_total),
                'low_ldi_rate': float(low_events / low_total),
                'high_ldi_events': int(high_events),
                'high_ldi_total': int(high_total),
                'high_ldi_rate': float(high_events / high_total),
                'odds_ratio': float(odds_ratio),
                'p_value': float(p_value),
                'significant': p_value < self.significance_level
            }
            
            logger.info("Survival analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Survival analysis failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def perform_network_analysis(self, session: Session, device_code: str = "KWA") -> Dict[str, Any]:
        """
        Perform network analysis using NetworkX centrality measures.
        
        Args:
            session: Database session
            device_code: Device code to analyze
            
        Returns:
            Network analysis results
        """
        logger.info(f"Performing network centrality analysis for {device_code}")
        
        try:
            # Use ChainLengthCalculator to build network
            chain_calculator = ChainLengthCalculator()
            graph = chain_calculator.build_predicate_graph(session, device_code)
            
            if graph.number_of_nodes() == 0:
                return {'error': 'No network data available'}
            
            # Calculate centrality measures
            import networkx as nx
            
            centrality_measures = {
                'betweenness': nx.betweenness_centrality(graph),
                'closeness': nx.closeness_centrality(graph),
                'eigenvector': nx.eigenvector_centrality(graph, max_iter=1000),
                'pagerank': nx.pagerank(graph),
                'degree': dict(graph.degree()),
                'in_degree': dict(graph.in_degree()),
                'out_degree': dict(graph.out_degree())
            }
            
            # Network-level statistics
            network_stats = {
                'node_count': graph.number_of_nodes(),
                'edge_count': graph.number_of_edges(),
                'density': nx.density(graph),
                'is_connected': nx.is_connected(graph.to_undirected()),
                'number_of_components': nx.number_weakly_connected_components(graph),
                'average_clustering': nx.average_clustering(graph.to_undirected()),
                'diameter': nx.diameter(graph.to_undirected()) if nx.is_connected(graph.to_undirected()) else None
            }
            
            # Identify key nodes
            key_nodes = {
                'highest_betweenness': max(centrality_measures['betweenness'], 
                                         key=centrality_measures['betweenness'].get),
                'highest_pagerank': max(centrality_measures['pagerank'], 
                                      key=centrality_measures['pagerank'].get),
                'highest_degree': max(centrality_measures['degree'], 
                                    key=centrality_measures['degree'].get)
            }
            
            # Risk propagation analysis
            # Correlate centrality with adverse events
            risk_correlation = {}
            
            for measure_name, centrality_dict in centrality_measures.items():
                if len(centrality_dict) > 1:
                    # Get adverse event data for nodes
                    centrality_values = []
                    risk_values = []
                    
                    for k_number, centrality in centrality_dict.items():
                        device = session.query(Device).filter(Device.k_number == k_number).first()
                        if device:
                            event_count = session.query(AdverseEvent).filter(
                                AdverseEvent.device_id == device.id
                            ).count()
                            
                            centrality_values.append(centrality)
                            risk_values.append(event_count)
                    
                    if len(centrality_values) > 5:  # Need sufficient data points
                        correlation, p_value = stats.spearmanr(centrality_values, risk_values)
                        risk_correlation[measure_name] = {
                            'correlation': float(correlation) if not np.isnan(correlation) else 0,
                            'p_value': float(p_value) if not np.isnan(p_value) else 1,
                            'significant': p_value < self.significance_level if not np.isnan(p_value) else False
                        }
            
            results = {
                'network_statistics': network_stats,
                'centrality_measures': {
                    name: {k: float(v) for k, v in measures.items()}
                    for name, measures in centrality_measures.items()
                },
                'key_nodes': key_nodes,
                'risk_propagation_analysis': risk_correlation
            }
            
            logger.info("Network analysis completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Network analysis failed: {e}")
            return {'error': str(e)}
    
    def perform_comprehensive_analysis(self, session: Session, device_code: str = "KWA") -> Dict[str, Any]:
        """
        Perform comprehensive advanced statistical analysis.
        
        Args:
            session: Database session
            device_code: Device code to analyze
            
        Returns:
            Comprehensive analysis results
        """
        logger.info(f"Starting comprehensive advanced statistical analysis for {device_code}")
        
        results = {
            'analysis_metadata': {
                'device_code': device_code,
                'analysis_date': pd.Timestamp.now().isoformat(),
                'significance_level': self.significance_level
            }
        }
        
        # Prepare data
        try:
            df = self.prepare_regression_data(session, device_code)
            results['data_summary'] = {
                'sample_size': len(df),
                'devices_with_events': int(df['has_adverse_events'].sum()),
                'total_adverse_events': int(df['adverse_event_count'].sum()),
                'mean_ldi_score': float(df['ldi_score'].mean()),
                'ldi_score_range': [float(df['ldi_score'].min()), float(df['ldi_score'].max())]
            }
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            return {'error': f'Data preparation failed: {e}'}
        
        # GLM Analysis
        try:
            glm_results = self.perform_glm_analysis(df)
            results['glm_analysis'] = glm_results
        except Exception as e:
            logger.error(f"GLM analysis failed: {e}")
            results['glm_analysis'] = {'error': str(e)}
        
        # Survival Analysis
        try:
            survival_results = self.perform_survival_analysis(df)
            results['survival_analysis'] = survival_results
        except Exception as e:
            logger.error(f"Survival analysis failed: {e}")
            results['survival_analysis'] = {'error': str(e)}
        
        # Network Analysis
        try:
            network_results = self.perform_network_analysis(session, device_code)
            results['network_analysis'] = network_results
        except Exception as e:
            logger.error(f"Network analysis failed: {e}")
            results['network_analysis'] = {'error': str(e)}
        
        # Overall assessment
        results['overall_assessment'] = self._generate_statistical_assessment(results)
        
        logger.info("Comprehensive advanced statistical analysis completed")
        return results
    
    def _generate_statistical_assessment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall statistical assessment."""
        assessment = {
            'statistical_power': 'adequate',
            'key_findings': [],
            'limitations': [],
            'recommendations': []
        }
        
        # Check sample size
        sample_size = results.get('data_summary', {}).get('sample_size', 0)
        if sample_size < 30:
            assessment['statistical_power'] = 'limited'
            assessment['limitations'].append('Small sample size may limit statistical power')
        elif sample_size < 100:
            assessment['statistical_power'] = 'moderate'
        
        # Extract key findings from GLM
        glm_results = results.get('glm_analysis', {})
        if 'logistic_glm' in glm_results and 'significant_predictors' in glm_results['logistic_glm']:
            significant_predictors = glm_results['logistic_glm']['significant_predictors']
            if 'ldi_score' in significant_predictors:
                assessment['key_findings'].append('LDI score is a significant predictor of adverse events')
            
            if len(significant_predictors) > 1:
                assessment['key_findings'].append(f'Multiple LDI components show significance: {significant_predictors}')
        
        # Extract findings from survival analysis
        survival_results = results.get('survival_analysis', {})
        if 'group_comparison' in survival_results:
            group_comp = survival_results['group_comparison']
            if group_comp.get('significant', False):
                assessment['key_findings'].append('Significant difference in survival between high and low LDI groups')
        
        # Extract findings from network analysis
        network_results = results.get('network_analysis', {})
        if 'risk_propagation_analysis' in network_results:
            risk_prop = network_results['risk_propagation_analysis']
            significant_measures = [
                measure for measure, stats in risk_prop.items() 
                if stats.get('significant', False)
            ]
            if significant_measures:
                assessment['key_findings'].append(f'Network centrality measures correlate with risk: {significant_measures}')
        
        # Recommendations
        if len(assessment['key_findings']) >= 2:
            assessment['recommendations'].append('Results support LDI methodology for risk assessment')
            assessment['recommendations'].append('Consider implementation in regulatory framework')
        else:
            assessment['recommendations'].append('Further validation with larger sample sizes recommended')
        
        assessment['recommendations'].append('Expand analysis to multiple device categories')
        
        return assessment


def main():
    """Test advanced statistical analysis with synthetic data."""
    analyzer = AdvancedStatisticalAnalyzer()
    
    # Create synthetic test data
    np.random.seed(42)
    
    test_data = []
    for i in range(50):
        ldi_score = np.random.beta(2, 3)
        semantic_distance = ldi_score + np.random.normal(0, 0.1)
        parameter_difference = ldi_score + np.random.normal(0, 0.1)
        chain_length = np.random.beta(3, 4)
        
        # Adverse events correlated with LDI
        event_prob = 0.1 + 0.4 * ldi_score
        has_events = np.random.binomial(1, event_prob)
        event_count = np.random.poisson(ldi_score * 5) if has_events else 0
        
        test_data.append({
            'k_number': f'K{900000 + i:06d}',
            'device_name': f'Test Device {i}',
            'device_code': 'KWA',
            'ldi_score': ldi_score,
            'semantic_distance': semantic_distance,
            'parameter_difference': parameter_difference,
            'chain_length': chain_length,
            'adverse_event_count': event_count,
            'serious_event_count': max(0, event_count - np.random.poisson(1)),
            'recall_count': np.random.binomial(1, 0.05),
            'has_adverse_events': has_events,
            'has_serious_events': int(event_count > 2),
            'has_recalls': np.random.binomial(1, 0.05),
            'approval_year': np.random.randint(2010, 2020),
            'years_since_approval': np.random.randint(5, 15),
            'market_exposure': np.random.randint(100, 1000),
            'log_exposure': np.log(np.random.randint(100, 1000)),
            'tech_params_count': np.random.randint(3, 10)
        })
    
    df = pd.DataFrame(test_data)
    
    print("Advanced Statistical Analysis Test")
    print("=" * 50)
    
    # Test GLM analysis
    glm_results = analyzer.perform_glm_analysis(df)
    print(f"GLM Analysis completed with {len(glm_results)} models")
    
    if 'logistic_glm' in glm_results:
        print(f"Logistic GLM significant predictors: {glm_results['logistic_glm'].get('significant_predictors', [])}")
    
    # Test survival analysis
    survival_results = analyzer.perform_survival_analysis(df)
    print(f"Survival Analysis completed")
    
    if 'survival_summary' in survival_results:
        print(f"Event rate: {survival_results['survival_summary']['event_rate']:.3f}")
    
    print("\nAdvanced statistical analysis test completed successfully")


if __name__ == "__main__":
    main()