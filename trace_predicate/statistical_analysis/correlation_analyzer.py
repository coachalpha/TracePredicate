"""
Statistical Correlation Analysis for LDI Validation

This module implements comprehensive statistical analysis to validate
the correlation between LDI scores and real-world risk outcomes.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
import statsmodels.api as sm
from statsmodels.stats.contingency_tables import mcnemar
# from lifelines import CoxPHFitter, KaplanMeierFitter
# from lifelines.statistics import logrank_test

from sqlalchemy.orm import Session
from ..database.models import Device, LDIScore, AdverseEvent, Recall
from ..database.operations import LDIOperations, AdverseEventOperations

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class CorrelationAnalyzer:
    """Comprehensive statistical analysis of LDI-outcome correlations."""
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize correlation analyzer.
        
        Args:
            significance_level: Statistical significance threshold
        """
        self.significance_level = significance_level
        self.analysis_results = {}
        
        logger.info(f"Initialized correlation analyzer with α={significance_level}")
    
    def analyze_ldi_adverse_event_correlation(
        self,
        session: Session,
        device_code: str = "KWA",
        min_devices: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze correlation between LDI scores and adverse event rates.
        
        Args:
            session: Database session
            device_code: Device code to analyze
            min_devices: Minimum number of devices required
            
        Returns:
            Comprehensive correlation analysis results
        """
        logger.info(f"Analyzing LDI-adverse event correlation for device code: {device_code}")
        
        # Get LDI scores and adverse event data
        data = self._prepare_ldi_adverse_event_data(session, device_code)
        
        if len(data) < min_devices:
            raise ValueError(f"Insufficient data: {len(data)} devices, minimum {min_devices} required")
        
        ldi_scores = data['ldi_score'].values
        adverse_event_rates = data['adverse_event_rate'].values
        
        # Basic correlation analysis
        correlation_results = self._calculate_correlations(ldi_scores, adverse_event_rates)
        
        # Regression analysis
        regression_results = self._perform_regression_analysis(ldi_scores, adverse_event_rates)
        
        # Statistical tests
        statistical_tests = self._perform_statistical_tests(ldi_scores, adverse_event_rates)
        
        # Effect size analysis
        effect_sizes = self._calculate_effect_sizes(ldi_scores, adverse_event_rates)
        
        # Risk stratification analysis
        risk_stratification = self._analyze_risk_stratification(data)
        
        # Bootstrap confidence intervals
        bootstrap_results = self._bootstrap_correlation_analysis(ldi_scores, adverse_event_rates)
        
        results = {
            'dataset_info': {
                'device_code': device_code,
                'n_devices': len(data),
                'ldi_score_range': [float(ldi_scores.min()), float(ldi_scores.max())],
                'adverse_event_rate_range': [float(adverse_event_rates.min()), float(adverse_event_rates.max())],
                'data_quality_metrics': self._assess_data_quality(data)
            },
            'correlation_analysis': correlation_results,
            'regression_analysis': regression_results,
            'statistical_tests': statistical_tests,
            'effect_sizes': effect_sizes,
            'risk_stratification': risk_stratification,
            'bootstrap_analysis': bootstrap_results,
            'raw_data': data.to_dict('records') if len(data) <= 100 else None  # Include raw data if small
        }
        
        self.analysis_results['ldi_adverse_events'] = results
        logger.info(f"LDI-adverse event analysis completed: correlation={correlation_results.get('spearman_correlation', 0):.3f}")
        
        return results
    
    def _prepare_ldi_adverse_event_data(self, session: Session, device_code: str) -> pd.DataFrame:
        """Prepare LDI and adverse event data for analysis."""
        logger.info("Preparing LDI and adverse event data")
        
        # Get devices with LDI scores
        ldi_query = session.query(LDIScore).join(Device).filter(Device.device_code == device_code)
        ldi_scores = ldi_query.all()
        
        data_records = []
        
        for ldi_score in ldi_scores:
            device = ldi_score.device
            
            # Get adverse events for this device
            adverse_events = AdverseEventOperations.get_events_for_device(session, device.k_number)
            
            # Calculate adverse event rate (events per year since approval)
            if device.approval_date:
                years_since_approval = (pd.Timestamp.now() - pd.Timestamp(device.approval_date)).days / 365.25
                adverse_event_rate = len(adverse_events) / max(years_since_approval, 0.1)  # Avoid division by zero
            else:
                adverse_event_rate = len(adverse_events)  # Raw count if no approval date
            
            # Count events by HFE classification
            hfe_counts = {'device_failure': 0, 'clear_use_error': 0, 'design_induced_use_error': 0, 'unclear': 0}
            for event in adverse_events:
                if event.hfe_classification:
                    hfe_counts[event.hfe_classification.value] += 1
            
            record = {
                'k_number': device.k_number,
                'device_name': device.device_name,
                'approval_date': device.approval_date,
                'ldi_score': ldi_score.ldi_score,
                'semantic_distance': ldi_score.semantic_distance,
                'parameter_difference': ldi_score.parameter_difference,
                'chain_length': ldi_score.chain_length,
                'adverse_event_count': len(adverse_events),
                'adverse_event_rate': adverse_event_rate,
                'years_since_approval': years_since_approval if device.approval_date else None,
                **hfe_counts
            }
            
            data_records.append(record)
        
        data = pd.DataFrame(data_records)
        logger.info(f"Prepared data for {len(data)} devices")
        
        return data
    
    def _calculate_correlations(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Calculate various correlation measures."""
        correlations = {}
        
        # Spearman correlation (rank-based, robust to outliers)
        spearman_corr, spearman_p = spearmanr(x, y)
        correlations['spearman_correlation'] = float(spearman_corr) if not np.isnan(spearman_corr) else 0.0
        correlations['spearman_p_value'] = float(spearman_p) if not np.isnan(spearman_p) else 1.0
        
        # Pearson correlation (linear relationship)
        pearson_corr, pearson_p = pearsonr(x, y)
        correlations['pearson_correlation'] = float(pearson_corr) if not np.isnan(pearson_corr) else 0.0
        correlations['pearson_p_value'] = float(pearson_p) if not np.isnan(pearson_p) else 1.0
        
        # Kendall's tau (alternative rank correlation)
        kendall_corr, kendall_p = kendalltau(x, y)
        correlations['kendall_correlation'] = float(kendall_corr) if not np.isnan(kendall_corr) else 0.0
        correlations['kendall_p_value'] = float(kendall_p) if not np.isnan(kendall_p) else 1.0
        
        # Determine statistical significance
        correlations['spearman_significant'] = correlations['spearman_p_value'] < self.significance_level
        correlations['pearson_significant'] = correlations['pearson_p_value'] < self.significance_level
        correlations['kendall_significant'] = correlations['kendall_p_value'] < self.significance_level
        
        return correlations
    
    def _perform_regression_analysis(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive regression analysis."""
        regression_results = {}
        
        # Prepare data
        X = x.reshape(-1, 1)
        
        # Linear regression
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        y_pred = lr_model.predict(X)
        
        regression_results['linear_regression'] = {
            'coefficient': float(lr_model.coef_[0]),
            'intercept': float(lr_model.intercept_),
            'r_squared': float(r2_score(y, y_pred)),
            'mse': float(mean_squared_error(y, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y, y_pred)))
        }
        
        # Statsmodels for detailed statistics
        try:
            X_sm = sm.add_constant(x)  # Add intercept
            sm_model = sm.OLS(y, X_sm).fit()
            
            regression_results['detailed_statistics'] = {
                'coefficient': float(sm_model.params[1]),
                'coefficient_std_err': float(sm_model.bse[1]),
                'coefficient_t_stat': float(sm_model.tvalues[1]),
                'coefficient_p_value': float(sm_model.pvalues[1]),
                'r_squared': float(sm_model.rsquared),
                'adjusted_r_squared': float(sm_model.rsquared_adj),
                'f_statistic': float(sm_model.fvalue),
                'f_p_value': float(sm_model.f_pvalue),
                'aic': float(sm_model.aic),
                'bic': float(sm_model.bic),
                'confidence_interval_lower': float(sm_model.conf_int().iloc[1, 0]),
                'confidence_interval_upper': float(sm_model.conf_int().iloc[1, 1])
            }
        except Exception as e:
            logger.warning(f"Detailed regression analysis failed: {e}")
            regression_results['detailed_statistics'] = None
        
        # Random Forest for non-linear relationships
        try:
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X, y)
            rf_pred = rf_model.predict(X)
            
            regression_results['random_forest'] = {
                'r_squared': float(r2_score(y, rf_pred)),
                'mse': float(mean_squared_error(y, rf_pred)),
                'feature_importance': float(rf_model.feature_importances_[0])
            }
        except Exception as e:
            logger.warning(f"Random Forest analysis failed: {e}")
            regression_results['random_forest'] = None
        
        return regression_results
    
    def _perform_statistical_tests(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform various statistical tests."""
        tests = {}
        
        # Test for normality (Shapiro-Wilk)
        try:
            shapiro_x = stats.shapiro(x)
            shapiro_y = stats.shapiro(y)
            
            tests['normality_tests'] = {
                'ldi_scores_normal': {
                    'statistic': float(shapiro_x.statistic),
                    'p_value': float(shapiro_x.pvalue),
                    'is_normal': float(shapiro_x.pvalue) > self.significance_level
                },
                'adverse_events_normal': {
                    'statistic': float(shapiro_y.statistic),
                    'p_value': float(shapiro_y.pvalue),
                    'is_normal': float(shapiro_y.pvalue) > self.significance_level
                }
            }
        except Exception as e:
            logger.warning(f"Normality tests failed: {e}")
            tests['normality_tests'] = None
        
        # Mann-Whitney U test (non-parametric comparison)
        try:
            # Split into high and low LDI score groups
            median_ldi = np.median(x)
            high_ldi_outcomes = y[x >= median_ldi]
            low_ldi_outcomes = y[x < median_ldi]
            
            mannwhitney_stat, mannwhitney_p = stats.mannwhitneyu(high_ldi_outcomes, low_ldi_outcomes, alternative='greater')
            
            tests['mann_whitney_test'] = {
                'statistic': float(mannwhitney_stat),
                'p_value': float(mannwhitney_p),
                'significant': float(mannwhitney_p) < self.significance_level,
                'interpretation': 'High LDI scores associated with higher adverse event rates' if float(mannwhitney_p) < self.significance_level else 'No significant difference'
            }
        except Exception as e:
            logger.warning(f"Mann-Whitney test failed: {e}")
            tests['mann_whitney_test'] = None
        
        # Kolmogorov-Smirnov test for distribution differences
        try:
            median_ldi = np.median(x)
            high_ldi_outcomes = y[x >= median_ldi]
            low_ldi_outcomes = y[x < median_ldi]
            
            ks_stat, ks_p = stats.ks_2samp(high_ldi_outcomes, low_ldi_outcomes)
            
            tests['kolmogorov_smirnov_test'] = {
                'statistic': float(ks_stat),
                'p_value': float(ks_p),
                'significant': float(ks_p) < self.significance_level
            }
        except Exception as e:
            logger.warning(f"Kolmogorov-Smirnov test failed: {e}")
            tests['kolmogorov_smirnov_test'] = None
        
        return tests
    
    def _calculate_effect_sizes(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Calculate various effect size measures."""
        effect_sizes = {}
        
        # Cohen's d (standardized mean difference)
        try:
            median_ldi = np.median(x)
            high_ldi_outcomes = y[x >= median_ldi]
            low_ldi_outcomes = y[x < median_ldi]
            
            pooled_std = np.sqrt(((len(high_ldi_outcomes) - 1) * np.var(high_ldi_outcomes, ddof=1) + 
                                 (len(low_ldi_outcomes) - 1) * np.var(low_ldi_outcomes, ddof=1)) / 
                                (len(high_ldi_outcomes) + len(low_ldi_outcomes) - 2))
            
            cohens_d = (np.mean(high_ldi_outcomes) - np.mean(low_ldi_outcomes)) / pooled_std
            
            effect_sizes['cohens_d'] = {
                'value': float(cohens_d),
                'interpretation': self._interpret_cohens_d(cohens_d)
            }
        except Exception as e:
            logger.warning(f"Cohen's d calculation failed: {e}")
            effect_sizes['cohens_d'] = None
        
        # Eta-squared (proportion of variance explained)
        try:
            correlation = np.corrcoef(x, y)[0, 1]
            eta_squared = correlation ** 2
            
            effect_sizes['eta_squared'] = {
                'value': float(eta_squared),
                'interpretation': self._interpret_eta_squared(eta_squared)
            }
        except Exception as e:
            logger.warning(f"Eta-squared calculation failed: {e}")
            effect_sizes['eta_squared'] = None
        
        return effect_sizes
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_eta_squared(self, eta_sq: float) -> str:
        """Interpret eta-squared effect size."""
        if eta_sq < 0.01:
            return "negligible"
        elif eta_sq < 0.06:
            return "small"
        elif eta_sq < 0.14:
            return "medium"
        else:
            return "large"
    
    def _analyze_risk_stratification(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze risk stratification based on LDI scores."""
        risk_analysis = {}
        
        # Define risk categories based on LDI score percentiles
        ldi_scores = data['ldi_score']
        risk_analysis['thresholds'] = {
            'low_risk_threshold': float(ldi_scores.quantile(0.33)),
            'high_risk_threshold': float(ldi_scores.quantile(0.67))
        }
        
        # Categorize devices
        data_copy = data.copy()
        data_copy['risk_category'] = pd.cut(
            data_copy['ldi_score'], 
            bins=[0, risk_analysis['thresholds']['low_risk_threshold'], 
                  risk_analysis['thresholds']['high_risk_threshold'], 1],
            labels=['Low Risk', 'Medium Risk', 'High Risk'],
            include_lowest=True
        )
        
        # Calculate statistics by risk category
        category_stats = data_copy.groupby('risk_category')['adverse_event_rate'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        
        risk_analysis['category_statistics'] = category_stats.to_dict('index')
        
        # Perform ANOVA to test differences between groups
        try:
            groups = [group['adverse_event_rate'].values for name, group in data_copy.groupby('risk_category')]
            f_stat, p_value = stats.f_oneway(*groups)
            
            risk_analysis['anova_test'] = {
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': float(p_value) < self.significance_level
            }
        except Exception as e:
            logger.warning(f"ANOVA test failed: {e}")
            risk_analysis['anova_test'] = None
        
        return risk_analysis
    
    def _bootstrap_correlation_analysis(self, x: np.ndarray, y: np.ndarray, n_bootstrap: int = 1000) -> Dict[str, Any]:
        """Perform bootstrap analysis for correlation confidence intervals."""
        logger.info(f"Performing bootstrap analysis with {n_bootstrap} samples")
        
        bootstrap_correlations = []
        n_samples = len(x)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            x_boot = x[indices]
            y_boot = y[indices]
            
            # Calculate Spearman correlation
            corr, _ = spearmanr(x_boot, y_boot)
            if not np.isnan(corr):
                bootstrap_correlations.append(corr)
        
        bootstrap_correlations = np.array(bootstrap_correlations)
        
        # Calculate confidence intervals
        ci_lower = np.percentile(bootstrap_correlations, (1 - 0.95) / 2 * 100)
        ci_upper = np.percentile(bootstrap_correlations, (1 + 0.95) / 2 * 100)
        
        bootstrap_results = {
            'n_bootstrap_samples': len(bootstrap_correlations),
            'mean_correlation': float(np.mean(bootstrap_correlations)),
            'std_correlation': float(np.std(bootstrap_correlations)),
            'confidence_interval_95': [float(ci_lower), float(ci_upper)],
            'bootstrap_correlations': bootstrap_correlations.tolist()[:100]  # Store first 100 for inspection
        }
        
        return bootstrap_results
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality for the analysis."""
        quality_metrics = {
            'missing_ldi_scores': int(data['ldi_score'].isna().sum()),
            'missing_adverse_events': int(data['adverse_event_rate'].isna().sum()),
            'ldi_score_range': [float(data['ldi_score'].min()), float(data['ldi_score'].max())],
            'ldi_score_std': float(data['ldi_score'].std()),
            'adverse_event_rate_range': [float(data['adverse_event_rate'].min()), float(data['adverse_event_rate'].max())],
            'outliers_ldi': self._count_outliers(data['ldi_score']),
            'outliers_adverse_events': self._count_outliers(data['adverse_event_rate']),
            'data_completeness': float((data.notna().sum().sum()) / (len(data) * len(data.columns)))
        }
        
        return quality_metrics
    
    def _count_outliers(self, series: pd.Series) -> int:
        """Count outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return int(((series < lower_bound) | (series > upper_bound)).sum())
    
    def generate_correlation_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive correlation analysis report."""
        report = []
        
        report.append("# LDI-Adverse Event Correlation Analysis Report")
        report.append("=" * 50)
        
        # Dataset information
        dataset_info = results['dataset_info']
        report.append(f"\n## Dataset Information")
        report.append(f"- Device Code: {dataset_info['device_code']}")
        report.append(f"- Number of Devices: {dataset_info['n_devices']}")
        report.append(f"- LDI Score Range: {dataset_info['ldi_score_range'][0]:.3f} - {dataset_info['ldi_score_range'][1]:.3f}")
        report.append(f"- Adverse Event Rate Range: {dataset_info['adverse_event_rate_range'][0]:.3f} - {dataset_info['adverse_event_rate_range'][1]:.3f}")
        
        # Main correlation results
        corr = results['correlation_analysis']
        report.append(f"\n## Correlation Analysis")
        report.append(f"- **Spearman Correlation: {corr['spearman_correlation']:.4f}** (p = {corr['spearman_p_value']:.4f})")
        report.append(f"- Pearson Correlation: {corr['pearson_correlation']:.4f} (p = {corr['pearson_p_value']:.4f})")
        report.append(f"- Kendall's Tau: {corr['kendall_correlation']:.4f} (p = {corr['kendall_p_value']:.4f})")
        
        significance_status = "**SIGNIFICANT**" if corr['spearman_significant'] else "Not significant"
        report.append(f"\n**Statistical Significance (α = {self.significance_level}): {significance_status}**")
        
        # Effect size interpretation
        if 'effect_sizes' in results and results['effect_sizes'].get('eta_squared'):
            eta_sq = results['effect_sizes']['eta_squared']
            report.append(f"\n## Effect Size")
            report.append(f"- Eta-squared: {eta_sq['value']:.4f} ({eta_sq['interpretation']} effect)")
        
        # Bootstrap confidence intervals
        bootstrap = results['bootstrap_analysis']
        report.append(f"\n## Confidence Intervals (Bootstrap)")
        report.append(f"- 95% CI for Spearman correlation: [{bootstrap['confidence_interval_95'][0]:.4f}, {bootstrap['confidence_interval_95'][1]:.4f}]")
        
        # Risk stratification
        risk_strat = results['risk_stratification']
        if risk_strat.get('anova_test'):
            anova = risk_strat['anova_test']
            report.append(f"\n## Risk Stratification")
            report.append(f"- ANOVA F-statistic: {anova['f_statistic']:.4f} (p = {anova['p_value']:.4f})")
            report.append("- Risk categories show significant differences" if anova['significant'] else "- No significant differences between risk categories")
        
        # Research conclusions
        report.append(f"\n## Research Conclusions")
        
        if corr['spearman_correlation'] > 0.7 and corr['spearman_significant']:
            report.append("✅ **STRONG SUPPORT for hypothesis**: LDI scores show strong, statistically significant correlation with adverse event rates.")
        elif corr['spearman_correlation'] > 0.5 and corr['spearman_significant']:
            report.append("✅ **MODERATE SUPPORT for hypothesis**: LDI scores show moderate, statistically significant correlation with adverse event rates.")
        elif corr['spearman_correlation'] > 0.3 and corr['spearman_significant']:
            report.append("⚠️ **WEAK SUPPORT for hypothesis**: LDI scores show weak but significant correlation with adverse event rates.")
        else:
            report.append("❌ **HYPOTHESIS NOT SUPPORTED**: LDI scores do not show significant correlation with adverse event rates.")
        
        return "\n".join(report)


def main():
    """Test correlation analysis with synthetic data."""
    analyzer = CorrelationAnalyzer()
    
    # Generate synthetic data
    np.random.seed(42)
    n_devices = 100
    
    # LDI scores with some structure
    ldi_scores = np.random.beta(2, 5, n_devices)  # Skewed towards lower values
    
    # Adverse event rates correlated with LDI scores plus noise
    adverse_event_rates = 2 * ldi_scores + 0.5 * np.random.randn(n_devices)
    adverse_event_rates = np.maximum(adverse_event_rates, 0)  # Ensure non-negative
    
    # Run correlation analysis
    correlation_results = analyzer._calculate_correlations(ldi_scores, adverse_event_rates)
    regression_results = analyzer._perform_regression_analysis(ldi_scores, adverse_event_rates)
    
    print("Correlation Analysis Results:")
    print(f"Spearman correlation: {correlation_results['spearman_correlation']:.4f} (p = {correlation_results['spearman_p_value']:.4f})")
    print(f"Significant: {correlation_results['spearman_significant']}")
    
    print(f"\nRegression Analysis:")
    print(f"R²: {regression_results['linear_regression']['r_squared']:.4f}")
    print(f"Coefficient: {regression_results['linear_regression']['coefficient']:.4f}")


if __name__ == "__main__":
    main()