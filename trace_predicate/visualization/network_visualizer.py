"""
Interactive Network Visualization for Medical Device Predicate Relationships

This module creates interactive visualizations of device predicate networks
using NetworkX for analysis and D3.js-compatible JSON export for web visualization.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sqlalchemy.orm import Session

from ..database.models import Device, PredicateRelationship, LDIScore, AdverseEvent
from ..ldi_engine.chain_calculator import ChainLengthCalculator

logger = logging.getLogger(__name__)


class NetworkVisualizer:
    """Create interactive visualizations of device predicate networks."""
    
    def __init__(self, output_dir: str = "network_visualizations"):
        """
        Initialize network visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "static").mkdir(exist_ok=True)
        (self.output_dir / "interactive").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        
        self.graph = None
        self.node_attributes = {}
        self.edge_attributes = {}
        
        logger.info(f"Initialized network visualizer with output directory: {self.output_dir}")
    
    def build_predicate_network(
        self, 
        session: Session, 
        device_code: Optional[str] = None,
        include_ldi_scores: bool = True
    ) -> nx.DiGraph:
        """
        Build predicate relationship network from database.
        
        Args:
            session: Database session
            device_code: Filter by device code (e.g., 'KWA')
            include_ldi_scores: Whether to include LDI scores as node attributes
            
        Returns:
            NetworkX directed graph
        """
        logger.info(f"Building predicate network for device code: {device_code or 'ALL'}")
        
        # Create directed graph
        graph = nx.DiGraph()
        
        # Query predicate relationships
        query = session.query(PredicateRelationship).join(
            Device, PredicateRelationship.child_device_id == Device.id
        )
        
        if device_code:
            query = query.filter(Device.device_code == device_code)
        
        relationships = query.all()
        
        # Add nodes and edges
        for rel in relationships:
            child_device = rel.child_device
            parent_device = rel.parent_device
            
            if child_device and parent_device:
                # Add child device node
                child_attrs = self._get_device_attributes(session, child_device, include_ldi_scores)
                graph.add_node(child_device.k_number, **child_attrs)
                
                # Add parent device node
                parent_attrs = self._get_device_attributes(session, parent_device, include_ldi_scores)
                graph.add_node(parent_device.k_number, **parent_attrs)
                
                # Add edge (child points to parent in predicate relationship)
                edge_attrs = {
                    'predicate_type': rel.predicate_type or 'direct',
                    'confidence': rel.confidence_score or 1.0,
                    'relationship_id': rel.id
                }
                graph.add_edge(child_device.k_number, parent_device.k_number, **edge_attrs)
        
        # Calculate network metrics
        self._calculate_network_metrics(graph)
        
        self.graph = graph
        logger.info(f"Built network with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        
        return graph
    
    def _get_device_attributes(
        self, 
        session: Session, 
        device: Device, 
        include_ldi_scores: bool
    ) -> Dict[str, Any]:
        """Get comprehensive device attributes for network nodes."""
        attrs = {
            'device_name': device.device_name or 'Unknown',
            'device_code': device.device_code or '',
            'approval_date': device.approval_date.isoformat() if device.approval_date else None,
            'approval_year': device.approval_date.year if device.approval_date else None,
            'device_id': device.id
        }
        
        # Add LDI scores if requested
        if include_ldi_scores:
            ldi_score = session.query(LDIScore).filter(
                LDIScore.device_k_number == device.k_number
            ).order_by(LDIScore.calculation_date.desc()).first()
            
            if ldi_score:
                attrs.update({
                    'ldi_score': ldi_score.ldi_score,
                    'semantic_distance': ldi_score.semantic_distance,
                    'parameter_difference': ldi_score.parameter_difference,
                    'chain_length': ldi_score.chain_length,
                    'ldi_calculation_date': ldi_score.calculation_date.isoformat()
                })
            else:
                attrs.update({
                    'ldi_score': None,
                    'semantic_distance': None,
                    'parameter_difference': None,
                    'chain_length': None
                })
        
        # Add adverse event statistics
        adverse_events = session.query(AdverseEvent).filter(
            AdverseEvent.device_id == device.id
        ).all()
        
        attrs.update({
            'adverse_event_count': len(adverse_events),
            'serious_event_count': len([e for e in adverse_events if e.severity == 'serious']),
            'adverse_event_rate': len(adverse_events) / 1000  # Per 1000 uses (simplified)
        })
        
        return attrs
    
    def _calculate_network_metrics(self, graph: nx.DiGraph):
        """Calculate network-level metrics and add as node attributes."""
        # Centrality measures
        betweenness = nx.betweenness_centrality(graph)
        closeness = nx.closeness_centrality(graph)
        pagerank = nx.pagerank(graph)
        
        # In-degree and out-degree
        in_degree = dict(graph.in_degree())
        out_degree = dict(graph.out_degree())
        
        # Add metrics as node attributes
        for node in graph.nodes():
            graph.nodes[node].update({
                'betweenness_centrality': betweenness.get(node, 0),
                'closeness_centrality': closeness.get(node, 0),
                'pagerank': pagerank.get(node, 0),
                'in_degree': in_degree.get(node, 0),
                'out_degree': out_degree.get(node, 0),
                'total_degree': in_degree.get(node, 0) + out_degree.get(node, 0)
            })
        
        logger.info("Calculated network centrality metrics")
    
    def create_ldi_risk_heatmap(self, title: str = "LDI Risk Heatmap") -> go.Figure:
        """
        Create interactive LDI risk heatmap visualization.
        
        Args:
            title: Title for the visualization
            
        Returns:
            Plotly figure object
        """
        if not self.graph:
            raise ValueError("Network not built. Call build_predicate_network() first.")
        
        # Extract node data
        nodes_data = []
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get('ldi_score') is not None:
                nodes_data.append({
                    'k_number': node,
                    'device_name': attrs.get('device_name', 'Unknown'),
                    'approval_year': attrs.get('approval_year'),
                    'ldi_score': attrs.get('ldi_score'),
                    'adverse_event_rate': attrs.get('adverse_event_rate', 0),
                    'in_degree': attrs.get('in_degree', 0),
                    'out_degree': attrs.get('out_degree', 0),
                    'pagerank': attrs.get('pagerank', 0)
                })
        
        if not nodes_data:
            raise ValueError("No LDI scores available for visualization")
        
        df = pd.DataFrame(nodes_data)
        
        # Create scatter plot with LDI score vs adverse event rate
        fig = px.scatter(
            df,
            x='ldi_score',
            y='adverse_event_rate',
            size='pagerank',
            color='approval_year',
            hover_data=['k_number', 'device_name', 'in_degree', 'out_degree'],
            title=title,
            labels={
                'ldi_score': 'LDI Score',
                'adverse_event_rate': 'Adverse Event Rate',
                'approval_year': 'Approval Year',
                'pagerank': 'Network Importance (PageRank)'
            },
            color_continuous_scale='Viridis'
        )
        
        # Add risk threshold lines
        fig.add_hline(y=0.05, line_dash="dash", line_color="orange", 
                     annotation_text="Medium Risk Threshold")
        fig.add_hline(y=0.10, line_dash="dash", line_color="red", 
                     annotation_text="High Risk Threshold")
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray", 
                     annotation_text="Medium LDI Threshold")
        fig.add_vline(x=0.7, line_dash="dash", line_color="darkred", 
                     annotation_text="High LDI Threshold")
        
        # Update layout
        fig.update_layout(
            width=1000,
            height=700,
            showlegend=True,
            hovermode='closest'
        )
        
        return fig
    
    def create_network_graph_plotly(self, title: str = "Device Predicate Network") -> go.Figure:
        """
        Create interactive network graph using Plotly.
        
        Args:
            title: Title for the visualization
            
        Returns:
            Plotly figure object
        """
        if not self.graph:
            raise ValueError("Network not built. Call build_predicate_network() first.")
        
        # Use spring layout for node positioning
        pos = nx.spring_layout(self.graph, k=3, iterations=50, seed=42)
        
        # Extract edges
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in self.graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            confidence = edge[2].get('confidence', 1.0)
            predicate_type = edge[2].get('predicate_type', 'direct')
            edge_info.append(f"{edge[0]} → {edge[1]}<br>Type: {predicate_type}<br>Confidence: {confidence:.2f}")
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Extract nodes
        node_x = []
        node_y = []
        node_info = []
        node_colors = []
        node_sizes = []
        
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            attrs = self.graph.nodes[node]
            device_name = attrs.get('device_name', 'Unknown')
            ldi_score = attrs.get('ldi_score')
            adverse_events = attrs.get('adverse_event_count', 0)
            approval_year = attrs.get('approval_year', 'Unknown')
            
            info = f"K-Number: {node}<br>"
            info += f"Device: {device_name}<br>"
            info += f"Approval Year: {approval_year}<br>"
            info += f"Adverse Events: {adverse_events}<br>"
            
            if ldi_score is not None:
                info += f"LDI Score: {ldi_score:.3f}<br>"
                node_colors.append(ldi_score)
            else:
                info += "LDI Score: Not calculated<br>"
                node_colors.append(0)
            
            node_info.append(info)
            
            # Node size based on degree centrality
            degree = attrs.get('total_degree', 1)
            node_sizes.append(max(10, min(50, degree * 5)))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_info,
            marker=dict(
                showscale=True,
                colorscale='RdYlBu_r',
                reversescale=True,
                color=node_colors,
                size=node_sizes,
                colorbar=dict(
                    thickness=15,
                    len=0.5,
                    x=1.02,
                    title="LDI Score"
                ),
                line=dict(width=2)
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=title,
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Interactive Device Predicate Network<br>Node size = Degree Centrality, Color = LDI Score",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor="left", yanchor="bottom",
                               font=dict(size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           width=1200,
                           height=800
                       ))
        
        return fig
    
    def export_d3_json(self, filename: str = "network_data.json") -> str:
        """
        Export network data in D3.js compatible format.
        
        Args:
            filename: Name of the JSON file to create
            
        Returns:
            Path to the created JSON file
        """
        if not self.graph:
            raise ValueError("Network not built. Call build_predicate_network() first.")
        
        # Convert to D3.js format
        d3_data = {
            'nodes': [],
            'links': [],
            'metadata': {
                'creation_date': datetime.now().isoformat(),
                'node_count': self.graph.number_of_nodes(),
                'edge_count': self.graph.number_of_edges(),
                'is_directed': self.graph.is_directed()
            }
        }
        
        # Add nodes
        node_mapping = {node: i for i, node in enumerate(self.graph.nodes())}
        
        for node, attrs in self.graph.nodes(data=True):
            d3_node = {
                'id': node_mapping[node],
                'k_number': node,
                'name': attrs.get('device_name', 'Unknown'),
                **attrs
            }
            d3_data['nodes'].append(d3_node)
        
        # Add links/edges
        for source, target, attrs in self.graph.edges(data=True):
            d3_link = {
                'source': node_mapping[source],
                'target': node_mapping[target],
                **attrs
            }
            d3_data['links'].append(d3_link)
        
        # Save to file
        output_path = self.output_dir / "data" / filename
        with open(output_path, 'w') as f:
            json.dump(d3_data, f, indent=2, default=str)
        
        logger.info(f"Exported D3.js compatible network data to {output_path}")
        return str(output_path)
    
    def create_temporal_evolution_plot(self) -> go.Figure:
        """Create temporal evolution plot showing device approval over time."""
        if not self.graph:
            raise ValueError("Network not built. Call build_predicate_network() first.")
        
        # Extract temporal data
        temporal_data = []
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get('approval_year'):
                temporal_data.append({
                    'k_number': node,
                    'approval_year': attrs.get('approval_year'),
                    'ldi_score': attrs.get('ldi_score', 0),
                    'adverse_event_count': attrs.get('adverse_event_count', 0),
                    'in_degree': attrs.get('in_degree', 0),
                    'device_name': attrs.get('device_name', 'Unknown')
                })
        
        if not temporal_data:
            raise ValueError("No temporal data available")
        
        df = pd.DataFrame(temporal_data)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Device Approvals Over Time',
                'LDI Scores by Approval Year',
                'Network Complexity Evolution'
            ),
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": True}],
                   [{"secondary_y": False}]],
            vertical_spacing=0.1
        )
        
        # Plot 1: Device count by year
        year_counts = df.groupby('approval_year').size().reset_index(name='count')
        fig.add_trace(
            go.Bar(x=year_counts['approval_year'], y=year_counts['count'], 
                   name='Device Approvals', marker_color='lightblue'),
            row=1, col=1
        )
        
        # Plot 2: LDI scores over time (with adverse events)
        fig.add_trace(
            go.Scatter(x=df['approval_year'], y=df['ldi_score'], 
                      mode='markers', name='LDI Scores', 
                      marker=dict(color='red', size=6)),
            row=2, col=1
        )
        
        # Add trend line for LDI scores
        if len(df) > 1:
            z = np.polyfit(df['approval_year'], df['ldi_score'], 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(x=df['approval_year'], y=p(df['approval_year']),
                          mode='lines', name='LDI Trend', line=dict(color='darkred')),
                row=2, col=1
            )
        
        # Plot 3: Network complexity (in-degree distribution)
        complexity_data = df.groupby('approval_year')['in_degree'].mean().reset_index()
        fig.add_trace(
            go.Scatter(x=complexity_data['approval_year'], y=complexity_data['in_degree'],
                      mode='lines+markers', name='Avg. Predicate References',
                      marker=dict(color='green')),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            title_text="Temporal Evolution of Medical Device Network",
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Approval Year", row=3, col=1)
        fig.update_yaxes(title_text="Device Count", row=1, col=1)
        fig.update_yaxes(title_text="LDI Score", row=2, col=1)
        fig.update_yaxes(title_text="Average In-Degree", row=3, col=1)
        
        return fig
    
    def generate_network_report(self) -> Dict[str, Any]:
        """Generate comprehensive network analysis report."""
        if not self.graph:
            raise ValueError("Network not built. Call build_predicate_network() first.")
        
        # Basic network statistics
        basic_stats = {
            'node_count': self.graph.number_of_nodes(),
            'edge_count': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_connected(self.graph.to_undirected()),
            'number_of_components': nx.number_weakly_connected_components(self.graph)
        }
        
        # Centrality analysis
        centrality_stats = {
            'avg_betweenness': np.mean([attrs.get('betweenness_centrality', 0) 
                                       for _, attrs in self.graph.nodes(data=True)]),
            'avg_closeness': np.mean([attrs.get('closeness_centrality', 0) 
                                     for _, attrs in self.graph.nodes(data=True)]),
            'max_in_degree': max([attrs.get('in_degree', 0) 
                                 for _, attrs in self.graph.nodes(data=True)]),
            'max_out_degree': max([attrs.get('out_degree', 0) 
                                  for _, attrs in self.graph.nodes(data=True)])
        }
        
        # LDI analysis
        ldi_scores = [attrs.get('ldi_score') for _, attrs in self.graph.nodes(data=True) 
                     if attrs.get('ldi_score') is not None]
        
        if ldi_scores:
            ldi_stats = {
                'mean_ldi': np.mean(ldi_scores),
                'median_ldi': np.median(ldi_scores),
                'std_ldi': np.std(ldi_scores),
                'min_ldi': np.min(ldi_scores),
                'max_ldi': np.max(ldi_scores),
                'devices_with_ldi': len(ldi_scores)
            }
        else:
            ldi_stats = {'devices_with_ldi': 0}
        
        # Risk analysis
        adverse_events = [attrs.get('adverse_event_count', 0) 
                         for _, attrs in self.graph.nodes(data=True)]
        
        risk_stats = {
            'total_adverse_events': sum(adverse_events),
            'mean_adverse_events': np.mean(adverse_events),
            'devices_with_events': len([x for x in adverse_events if x > 0])
        }
        
        report = {
            'generation_date': datetime.now().isoformat(),
            'basic_statistics': basic_stats,
            'centrality_analysis': centrality_stats,
            'ldi_analysis': ldi_stats,
            'risk_analysis': risk_stats
        }
        
        return report
    
    def save_all_visualizations(self, session: Session, device_code: str = "KWA"):
        """Generate and save all visualizations."""
        logger.info(f"Generating all visualizations for device code: {device_code}")
        
        # Build network
        self.build_predicate_network(session, device_code)
        
        # Create and save LDI risk heatmap
        heatmap_fig = self.create_ldi_risk_heatmap(f"LDI Risk Analysis - {device_code}")
        heatmap_path = self.output_dir / "interactive" / f"ldi_risk_heatmap_{device_code.lower()}.html"
        heatmap_fig.write_html(str(heatmap_path))
        
        # Create and save network graph
        network_fig = self.create_network_graph_plotly(f"Device Predicate Network - {device_code}")
        network_path = self.output_dir / "interactive" / f"network_graph_{device_code.lower()}.html"
        network_fig.write_html(str(network_path))
        
        # Create and save temporal evolution
        temporal_fig = self.create_temporal_evolution_plot()
        temporal_path = self.output_dir / "interactive" / f"temporal_evolution_{device_code.lower()}.html"
        temporal_fig.write_html(str(temporal_path))
        
        # Export D3.js data
        d3_path = self.export_d3_json(f"network_data_{device_code.lower()}.json")
        
        # Generate report
        report = self.generate_network_report()
        report_path = self.output_dir / "data" / f"network_report_{device_code.lower()}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Saved all visualizations to {self.output_dir}")
        return {
            'heatmap': str(heatmap_path),
            'network_graph': str(network_path),
            'temporal_evolution': str(temporal_path),
            'd3_data': d3_path,
            'report': str(report_path)
        }


def main():
    """Test network visualization with synthetic data."""
    visualizer = NetworkVisualizer("test_network_output")
    
    # Create a test network
    test_graph = nx.DiGraph()
    
    # Add test nodes with LDI scores
    devices = [
        ("K123456", {"device_name": "Device A", "ldi_score": 0.3, "adverse_event_count": 5, "approval_year": 2015}),
        ("K123457", {"device_name": "Device B", "ldi_score": 0.5, "adverse_event_count": 12, "approval_year": 2018}),
        ("K123458", {"device_name": "Device C", "ldi_score": 0.8, "adverse_event_count": 25, "approval_year": 2020}),
        ("K123459", {"device_name": "Device D", "ldi_score": 0.2, "adverse_event_count": 3, "approval_year": 2012}),
    ]
    
    for k_number, attrs in devices:
        attrs['adverse_event_rate'] = attrs['adverse_event_count'] / 100
        test_graph.add_node(k_number, **attrs)
    
    # Add test edges
    test_graph.add_edge("K123456", "K123459", predicate_type="direct", confidence=0.9)
    test_graph.add_edge("K123457", "K123456", predicate_type="direct", confidence=0.8)
    test_graph.add_edge("K123458", "K123457", predicate_type="modified", confidence=0.7)
    
    visualizer.graph = test_graph
    visualizer._calculate_network_metrics(test_graph)
    
    # Create visualizations
    heatmap_fig = visualizer.create_ldi_risk_heatmap("Test LDI Risk Heatmap")
    heatmap_fig.show()
    
    network_fig = visualizer.create_network_graph_plotly("Test Network Graph")
    network_fig.show()
    
    # Export D3.js data
    d3_path = visualizer.export_d3_json("test_network.json")
    print(f"D3.js data exported to: {d3_path}")
    
    # Generate report
    report = visualizer.generate_network_report()
    print("Network Report:", json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()