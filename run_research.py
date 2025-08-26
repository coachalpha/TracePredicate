#!/usr/bin/env python3
"""
Standalone script to execute TracePredicate research.
"""

import sys
import os
sys.path.append('/Users/muzi/TracePredicate')

from trace_predicate.research.research_executor import TracePredictateResearchExecutor

def main():
    print("Starting TracePredicate Research Execution...")
    print("=" * 60)
    
    executor = TracePredictateResearchExecutor()
    
    try:
        results = executor.execute_complete_research(n_devices=150)
        
        print("\n" + "="*60)
        print("FINAL RESEARCH RESULTS")
        print("="*60)
        
        overall = results['research_report']['overall_assessment']
        exec_summary = results['research_report']['executive_summary']
        
        print(f"Research Outcome: {overall['research_outcome']}")
        print(f"Overall Score: {overall['overall_score']:.3f}/1.0")
        print(f"Confidence Level: {overall['confidence_level']}")
        print(f"Publication Ready: {overall['publication_readiness']}")
        print(f"Regulatory Ready: {overall['regulatory_readiness']}")
        
        print(f"\nKey Findings:")
        print(f"  - Spearman Correlation: {exec_summary['primary_findings']['ldi_correlation']:.4f}")
        print(f"  - Statistical Significance: {exec_summary['primary_findings']['statistical_significance']}")
        print(f"  - Predictive AUC: {exec_summary['primary_findings']['predictive_performance']:.3f}")
        
        print(f"\nRecommendation: {overall['recommendation']}")
        print(f"\nResults saved to: {executor.output_dir}")
        
        return True
        
    except Exception as e:
        print(f"Research execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)