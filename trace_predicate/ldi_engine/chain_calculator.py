"""
Chain Length Calculator for LDI Engine

This module implements predicate chain length calculation and analysis
to measure the generational distance from original device concepts.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
from sqlalchemy.orm import Session

from ..database.models import Device, PredicateRelationship
from ..database.operations import PredicateOperations

logger = logging.getLogger(__name__)


class ChainLengthCalculator:
    """Calculate predicate chain lengths and network metrics."""
    
    def __init__(self, max_depth: int = 15, decay_factor: float = 0.9):
        """
        Initialize chain length calculator.
        
        Args:
            max_depth: Maximum depth to traverse in predicate chains
            decay_factor: Decay factor for weighting distant predicates
        """
        self.max_depth = max_depth
        self.decay_factor = decay_factor
        self.predicate_graph = None
        self.generation_levels = {}
        
        logger.info(f"Initialized chain calculator with max_depth={max_depth}, decay={decay_factor}")
    
    def build_predicate_graph(self, session: Session, device_code: Optional[str] = None) -> nx.DiGraph:
        """
        Build a directed graph of predicate relationships.
        
        Args:
            session: Database session
            device_code: Optional filter for specific device code
            
        Returns:
            NetworkX directed graph of predicate relationships
        """
        logger.info("Building predicate relationship graph...")
        
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
                # Add nodes with device information
                graph.add_node(child_device.k_number, 
                              device_name=child_device.device_name,
                              approval_date=child_device.approval_date,
                              device_code=child_device.device_code,
                              device_id=child_device.id)
                
                graph.add_node(parent_device.k_number,
                              device_name=parent_device.device_name,
                              approval_date=parent_device.approval_date,
                              device_code=parent_device.device_code,
                              device_id=parent_device.id)
                
                # Add edge from child to parent (predicate relationship)
                graph.add_edge(child_device.k_number, parent_device.k_number,
                              predicate_type=rel.predicate_type,
                              confidence=rel.confidence_score or 1.0,
                              relationship_id=rel.id)
        
        self.predicate_graph = graph
        logger.info(f"Built graph with {graph.number_of_nodes()} devices and {graph.number_of_edges()} relationships")
        
        return graph
    
    def calculate_generation_levels(self) -> Dict[str, int]:
        """
        Calculate generation levels for all devices in the graph.
        Root devices (no predicates) are generation 0.
        
        Returns:
            Dictionary mapping K-numbers to generation levels
        """
        if not self.predicate_graph:
            logger.warning("Predicate graph not built, cannot calculate generation levels")
            return {}
        
        generation_levels = {}
        
        # Find root nodes (devices with no incoming edges - no predicates)
        root_nodes = [node for node in self.predicate_graph.nodes() 
                     if self.predicate_graph.in_degree(node) == 0]
        
        logger.info(f"Found {len(root_nodes)} root devices")
        
        # BFS to assign generation levels
        queue = [(node, 0) for node in root_nodes]
        visited = set()
        
        while queue:
            current_node, generation = queue.pop(0)
            
            if current_node in visited:
                continue
                
            visited.add(current_node)
            generation_levels[current_node] = generation
            
            # Add children (devices that use this as predicate)
            for successor in self.predicate_graph.predecessors(current_node):
                if successor not in visited:
                    queue.append((successor, generation + 1))
        
        # Handle disconnected components or cycles
        for node in self.predicate_graph.nodes():
            if node not in generation_levels:
                # For disconnected nodes, calculate shortest path to any root
                try:
                    min_distance = float('inf')
                    for root in root_nodes:
                        if nx.has_path(self.predicate_graph, node, root):
                            distance = nx.shortest_path_length(self.predicate_graph, node, root)
                            min_distance = min(min_distance, distance)
                    
                    if min_distance != float('inf'):
                        generation_levels[node] = min_distance
                    else:
                        generation_levels[node] = 0  # Treat as root if no path found
                        
                except nx.NetworkXNoPath:
                    generation_levels[node] = 0
        
        self.generation_levels = generation_levels
        logger.info(f"Calculated generation levels for {len(generation_levels)} devices")
        
        return generation_levels
    
    def get_predicate_chains(self, k_number: str) -> List[List[str]]:
        """
        Get all predicate chains for a specific device.
        
        Args:
            k_number: K-number of the device
            
        Returns:
            List of predicate chains (each chain is a list of K-numbers from device to root)
        """
        if not self.predicate_graph or k_number not in self.predicate_graph:
            return []
        
        chains = []
        
        def dfs_chains(current_node: str, current_chain: List[str], depth: int):
            """Depth-first search to find all chains to root nodes."""
            if depth > self.max_depth:
                return
            
            # Get predicates (outgoing edges in our graph structure)
            predicates = list(self.predicate_graph.successors(current_node))
            
            if not predicates:
                # Reached a root node
                chains.append(current_chain.copy())
                return
            
            # Continue with each predicate
            for predicate in predicates:
                new_chain = current_chain + [predicate]
                dfs_chains(predicate, new_chain, depth + 1)
        
        # Start DFS from the target device
        dfs_chains(k_number, [k_number], 0)
        
        logger.debug(f"Found {len(chains)} predicate chains for device {k_number}")
        return chains
    
    def calculate_chain_metrics(self, k_number: str) -> Tuple[float, Dict[str, any]]:
        """
        Calculate comprehensive chain metrics for a device.
        
        Args:
            k_number: K-number of the device
            
        Returns:
            Tuple of (weighted_chain_length, detailed_metrics)
        """
        if not self.predicate_graph:
            logger.warning("Predicate graph not available")
            return 0.0, {}
        
        metrics = {
            'direct_predicates': 0,
            'total_chains': 0,
            'max_chain_length': 0,
            'min_chain_length': 0,
            'avg_chain_length': 0.0,
            'weighted_chain_length': 0.0,
            'generation_level': 0,
            'chains': []
        }
        
        # Get all predicate chains
        chains = self.get_predicate_chains(k_number)
        
        if not chains:
            return 0.0, metrics
        
        # Calculate basic metrics
        chain_lengths = [len(chain) - 1 for chain in chains]  # -1 because chain includes the device itself
        
        metrics['direct_predicates'] = len(list(self.predicate_graph.successors(k_number)))
        metrics['total_chains'] = len(chains)
        metrics['max_chain_length'] = max(chain_lengths) if chain_lengths else 0
        metrics['min_chain_length'] = min(chain_lengths) if chain_lengths else 0
        metrics['avg_chain_length'] = sum(chain_lengths) / len(chain_lengths) if chain_lengths else 0
        metrics['chains'] = chains
        
        # Get generation level
        if self.generation_levels:
            metrics['generation_level'] = self.generation_levels.get(k_number, 0)
        else:
            # Calculate on-the-fly
            generation_levels = self.calculate_generation_levels()
            metrics['generation_level'] = generation_levels.get(k_number, 0)
        
        # Calculate weighted chain length
        # Weight longer chains less due to decay factor
        weighted_sum = 0.0
        total_weight = 0.0
        
        for chain_length in chain_lengths:
            weight = self.decay_factor ** chain_length
            weighted_sum += chain_length * weight
            total_weight += weight
        
        metrics['weighted_chain_length'] = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        logger.debug(f"Chain metrics for {k_number}: length={metrics['weighted_chain_length']:.2f}, generation={metrics['generation_level']}")
        
        return metrics['weighted_chain_length'], metrics
    
    def analyze_chain_complexity(self, k_number: str) -> Dict[str, any]:
        """
        Analyze the complexity of predicate chains for a device.
        
        Args:
            k_number: K-number of the device
            
        Returns:
            Dictionary with chain complexity analysis
        """
        chains = self.get_predicate_chains(k_number)
        
        analysis = {
            'branching_factor': 0.0,  # Average number of predicates per device in chain
            'chain_diversity': 0.0,   # How diverse are the predicate chains
            'temporal_span': 0.0,     # Time span covered by the chain
            'unique_predicates': 0,   # Number of unique predicates
            'convergence_points': 0   # Devices that appear in multiple chains
        }
        
        if not chains:
            return analysis
        
        # Collect all unique predicates
        all_predicates = set()
        predicate_counts = {}
        
        for chain in chains:
            for predicate in chain[1:]:  # Skip the device itself
                all_predicates.add(predicate)
                predicate_counts[predicate] = predicate_counts.get(predicate, 0) + 1
        
        analysis['unique_predicates'] = len(all_predicates)
        analysis['convergence_points'] = sum(1 for count in predicate_counts.values() if count > 1)
        
        # Calculate branching factor
        if chains:
            total_branches = 0
            total_nodes = 0
            
            for chain in chains:
                for i, device in enumerate(chain[:-1]):  # Don't count the last node
                    if device in self.predicate_graph:
                        branches = len(list(self.predicate_graph.successors(device)))
                        total_branches += branches
                        total_nodes += 1
            
            analysis['branching_factor'] = total_branches / total_nodes if total_nodes > 0 else 0.0
        
        # Calculate chain diversity (using set operations)
        if len(chains) > 1:
            # Calculate Jaccard similarity between chains
            similarities = []
            for i in range(len(chains)):
                for j in range(i + 1, len(chains)):
                    set1 = set(chains[i])
                    set2 = set(chains[j])
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    similarity = intersection / union if union > 0 else 0
                    similarities.append(similarity)
            
            # Diversity is 1 - average similarity
            analysis['chain_diversity'] = 1.0 - (sum(similarities) / len(similarities) if similarities else 1.0)
        
        # Temporal analysis (if approval dates are available)
        if self.predicate_graph:
            dates = []
            for chain in chains:
                for device_k in chain:
                    if device_k in self.predicate_graph.nodes:
                        approval_date = self.predicate_graph.nodes[device_k].get('approval_date')
                        if approval_date:
                            dates.append(approval_date)
            
            if len(dates) > 1:
                date_range = max(dates) - min(dates)
                analysis['temporal_span'] = date_range.days / 365.25  # Convert to years
        
        return analysis
    
    def batch_calculate_chain_lengths(self, k_numbers: List[str]) -> List[Tuple[float, Dict[str, any]]]:
        """
        Calculate chain lengths for multiple devices.
        
        Args:
            k_numbers: List of K-numbers
            
        Returns:
            List of (chain_length, metrics) tuples
        """
        results = []
        
        for k_number in k_numbers:
            chain_length, metrics = self.calculate_chain_metrics(k_number)
            results.append((chain_length, metrics))
        
        logger.info(f"Calculated chain lengths for {len(k_numbers)} devices")
        return results
    
    def get_network_statistics(self) -> Dict[str, any]:
        """
        Get overall network statistics.
        
        Returns:
            Dictionary with network-level statistics
        """
        if not self.predicate_graph:
            return {}
        
        stats = {
            'total_devices': self.predicate_graph.number_of_nodes(),
            'total_relationships': self.predicate_graph.number_of_edges(),
            'connected_components': nx.number_weakly_connected_components(self.predicate_graph),
            'average_degree': 0.0,
            'density': nx.density(self.predicate_graph),
            'generation_distribution': {},
            'longest_chain': 0,
            'root_devices': 0
        }
        
        # Calculate average degree
        if self.predicate_graph.number_of_nodes() > 0:
            total_degree = sum(dict(self.predicate_graph.degree()).values())
            stats['average_degree'] = total_degree / self.predicate_graph.number_of_nodes()
        
        # Generation distribution
        if self.generation_levels:
            generation_counts = {}
            for generation in self.generation_levels.values():
                generation_counts[generation] = generation_counts.get(generation, 0) + 1
            stats['generation_distribution'] = generation_counts
            stats['longest_chain'] = max(self.generation_levels.values()) if self.generation_levels else 0
        
        # Count root devices
        stats['root_devices'] = len([node for node in self.predicate_graph.nodes() 
                                   if self.predicate_graph.in_degree(node) == 0])
        
        return stats


def main():
    """Test chain length calculation with mock data."""
    calculator = ChainLengthCalculator()
    
    # Create a simple test graph
    test_graph = nx.DiGraph()
    
    # Add test devices and relationships
    # Device A -> B -> C -> D (linear chain)
    # Device E -> B (convergence)
    devices = [
        ("K123456", "Device A"),
        ("K123457", "Device B"), 
        ("K123458", "Device C"),
        ("K123459", "Device D"),
        ("K123460", "Device E")
    ]
    
    for k_number, name in devices:
        test_graph.add_node(k_number, device_name=name)
    
    # Add predicate relationships (child -> parent)
    test_graph.add_edge("K123456", "K123457")  # A uses B as predicate
    test_graph.add_edge("K123457", "K123458")  # B uses C as predicate  
    test_graph.add_edge("K123458", "K123459")  # C uses D as predicate
    test_graph.add_edge("K123460", "K123457")  # E uses B as predicate
    
    calculator.predicate_graph = test_graph
    
    # Calculate generation levels
    generation_levels = calculator.calculate_generation_levels()
    print("Generation Levels:", generation_levels)
    
    # Calculate chain metrics for Device A
    chain_length, metrics = calculator.calculate_chain_metrics("K123456")
    print(f"\nDevice A Chain Length: {chain_length:.3f}")
    print("Metrics:", metrics)
    
    # Analyze chain complexity
    complexity = calculator.analyze_chain_complexity("K123456")
    print("\nChain Complexity Analysis:", complexity)
    
    # Network statistics
    network_stats = calculator.get_network_statistics()
    print("\nNetwork Statistics:", network_stats)


if __name__ == "__main__":
    main()