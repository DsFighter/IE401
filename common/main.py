"""
Multi-Network Metro Resilience Analysis System
Comprehensive tool for analyzing and comparing multiple metro networks
"""

import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Set aesthetic style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class MetroNetworkAnalyzer:
    """Main class for metro network analysis"""
    
    def __init__(self, geojson_path, cost_per_km=50000000, network_name="Network"):
        """Initialize with GeoJSON file path and cost per km
        
        Args:
            geojson_path: Path to GeoJSON file
            cost_per_km: Cost per kilometer for new connections (default: 50 million per km)
            network_name: Name identifier for this network
        """
        self.geojson_path = geojson_path
        self.cost_per_km = cost_per_km
        self.network_name = network_name
        self.G = None
        self.stations = {}
        self.line_colors = {}
        self.load_network()
        
    def load_network(self):
        """Load GeoJSON and build network graph"""
        with open(self.geojson_path, 'r') as f:
            data = json.load(f)
        
        self.G = nx.Graph()
        
        # Extract stations
        for feature in data['features']:
            if feature['geometry']['type'] == 'Point':
                coords = feature['geometry']['coordinates']
                name = feature['properties']['name']
                lines = feature['properties']['lines']
                
                # Handle duplicate station names by making unique identifiers
                station_id = f"{name}_{coords[0]:.6f}_{coords[1]:.6f}"
                
                self.stations[station_id] = {
                    'name': name,
                    'coords': coords,
                    'lines': lines,
                    'lat': coords[1],
                    'lon': coords[0]
                }
                
                self.G.add_node(station_id, 
                               name=name,
                               pos=(coords[0], coords[1]),
                               lines=lines)
        
        # Build edges based on line connectivity
        self._build_edges()
        self._assign_line_colors()
        
    def _assign_line_colors(self):
        """Assign unique colors to each metro line"""
        # Collect all unique lines
        all_lines = set()
        for station_data in self.stations.values():
            all_lines.update(station_data['lines'])
        
        # Define color palette
        color_palette = [
            '#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6',
            '#ec4899', '#06b6d4', '#f97316', '#14b8a6', '#a855f7',
            '#84cc16', '#6366f1', '#f43f5e', '#0ea5e9', '#22c55e',
        ]
        
        # Assign colors to lines
        for i, line in enumerate(sorted(all_lines)):
            if line == 'interchange':
                self.line_colors[line] = '#64748b'
            elif line.startswith('resilience'):
                self.line_colors[line] = '#10b981'
            else:
                self.line_colors[line] = color_palette[i % len(color_palette)]
        
    def _build_edges(self):
        """Build edges between consecutive stations on same line"""
        # Group stations by line
        line_stations = {}
        for station_id, data in self.stations.items():
            for line in data['lines']:
                if line not in line_stations:
                    line_stations[line] = []
                line_stations[line].append(station_id)
        
        # Connect stations on same line based on geographical proximity
        for line, stations in line_stations.items():
            if len(stations) < 2:
                continue
            
            coords = np.array([self.stations[s]['coords'] for s in stations])
            station_ids = stations
            
            sorted_indices = np.lexsort((coords[:, 1], coords[:, 0]))
            sorted_stations = [station_ids[i] for i in sorted_indices]
            
            for i in range(len(sorted_stations) - 1):
                s1, s2 = sorted_stations[i], sorted_stations[i + 1]
                dist = self._haversine_distance(
                    self.stations[s1]['coords'],
                    self.stations[s2]['coords']
                )
                self.G.add_edge(s1, s2, weight=dist, line=line)
        
        # Connect interchange stations
        station_names = {}
        for station_id, data in self.stations.items():
            name = data['name']
            if name not in station_names:
                station_names[name] = []
            station_names[name].append(station_id)
        
        for name, station_list in station_names.items():
            if len(station_list) > 1:
                for i in range(len(station_list)):
                    for j in range(i + 1, len(station_list)):
                        if not self.G.has_edge(station_list[i], station_list[j]):
                            self.G.add_edge(station_list[i], station_list[j], 
                                          weight=0.1, line='interchange')
    
    def _haversine_distance(self, coord1, coord2):
        """Calculate distance between two coordinates in km"""
        lon1, lat1 = coord1
        lon2, lat2 = coord2
        
        R = 6371  # Earth's radius in km
        
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        
        a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def calculate_centralities(self):
        """Calculate betweenness, closeness, and degree centrality"""
        betweenness = nx.betweenness_centrality(self.G, weight='weight')
        closeness = nx.closeness_centrality(self.G, distance='weight')
        degree = nx.degree_centrality(self.G)
        
        return betweenness, closeness, degree
    
    def calculate_criticality(self, alpha=0.4, beta=0.3, gamma=0.3):
        """Calculate node criticality using weighted mean of centralities"""
        betweenness, closeness, degree = self.calculate_centralities()
        
        criticality = {}
        for node in self.G.nodes():
            criticality[node] = (alpha * betweenness[node] + 
                                beta * closeness[node] + 
                                gamma * degree[node])
        
        return criticality
    
    def calculate_efficiency(self):
        """Calculate network efficiency"""
        n = self.G.number_of_nodes()
        if n <= 1:
            return 0
        
        total_efficiency = 0
        for i in self.G.nodes():
            for j in self.G.nodes():
                if i != j:
                    try:
                        length = nx.shortest_path_length(self.G, i, j, weight='weight')
                        if length > 0:
                            total_efficiency += 1 / length
                    except nx.NetworkXNoPath:
                        pass
        
        efficiency = total_efficiency / (n * (n - 1))
        return efficiency
    
    def calculate_robustness(self, path_set):
        """Calculate robustness indicator based on overlap coefficient"""
        if not path_set or len(path_set) == 0:
            return 0
        
        total_overlap = 0
        path_count = len(path_set)
        
        for path in path_set:
            edge_scores = []
            for u, v in zip(path[:-1], path[1:]):
                edge_usage = sum(1 for p in path_set 
                               if any((p[i] == u and p[i+1] == v) or 
                                     (p[i] == v and p[i+1] == u) 
                                     for i in range(len(p)-1)))
                edge_scores.append(1 / edge_usage if edge_usage > 0 else 0)
            
            if edge_scores:
                total_overlap += np.mean(edge_scores)
        
        robustness = total_overlap / path_count if path_count > 0 else 0
        return robustness
    
    def add_resilience_lines(self, k=2):
        """Add alternate paths connecting k-th critical nodes to most critical nodes"""
        criticality = self.calculate_criticality()
        
        line_nodes = {}
        for node in self.G.nodes():
            lines = self.stations[node]['lines']
            for line in lines:
                if line not in line_nodes:
                    line_nodes[line] = []
                line_nodes[line].append(node)
        
        new_edges = []
        cost_breakdown = {}
        total_cost = 0
        
        for line, nodes in line_nodes.items():
            if len(nodes) < k + 1:
                continue
            
            line_criticality = {n: criticality[n] for n in nodes}
            sorted_line_nodes = sorted(line_criticality.items(), 
                                      key=lambda x: x[1], reverse=True)
            
            if len(sorted_line_nodes) <= k:
                continue
            
            kth_critical_node = sorted_line_nodes[k-1][0]
            most_critical_node = sorted_line_nodes[0][0]
            
            most_critical_lines = self.stations[most_critical_node]['lines']
            
            for target_line in most_critical_lines:
                if target_line == line:
                    continue
                
                if target_line not in line_nodes:
                    continue
                
                min_dist = float('inf')
                nearest_node = None
                
                kth_coords = self.stations[kth_critical_node]['coords']
                
                for target_node in line_nodes[target_line]:
                    dist = self._haversine_distance(
                        kth_coords,
                        self.stations[target_node]['coords']
                    )
                    if dist < min_dist:
                        min_dist = dist
                        nearest_node = target_node
                
                if nearest_node and not self.G.has_edge(kth_critical_node, nearest_node):
                    self.G.add_edge(kth_critical_node, nearest_node,
                                   weight=min_dist, line=f'resilience_{k}')
                    new_edges.append((kth_critical_node, nearest_node))
                    
                    edge_cost = min_dist * self.cost_per_km
                    total_cost += edge_cost
                    
                    edge_name = f"{self.stations[kth_critical_node]['name']} → {self.stations[nearest_node]['name']}"
                    cost_breakdown[edge_name] = {
                        'distance_km': min_dist,
                        'cost': edge_cost
                    }
        
        self._assign_line_colors()
        
        return new_edges, total_cost, cost_breakdown
    
    def get_metrics(self):
        """Get all network metrics"""
        criticality = self.calculate_criticality()
        efficiency = self.calculate_efficiency()
        
        # Calculate robustness
        sample_nodes = list(self.G.nodes())[:min(10, len(self.G.nodes()))]
        paths = []
        for i in range(len(sample_nodes)):
            for j in range(i+1, len(sample_nodes)):
                try:
                    path_list = list(nx.all_shortest_paths(self.G, 
                                                           sample_nodes[i], 
                                                           sample_nodes[j], 
                                                           weight='weight'))
                    paths.extend(path_list[:3])
                except:
                    pass
        
        robustness = self.calculate_robustness(paths)
        
        return {
            'criticality': criticality,
            'efficiency': efficiency,
            'robustness': robustness,
            'avg_criticality': np.mean(list(criticality.values())),
            'max_criticality': max(criticality.values()),
            'nodes': self.G.number_of_nodes(),
            'edges': self.G.number_of_edges()
        }


class MultiNetworkComparator:
    """Class for comparing multiple metro networks"""
    
    def __init__(self):
        self.networks = []
        self.network_names = []
        
    def add_network(self, geojson_path, network_name, cost_per_km=50000000):
        """Add a network to comparison"""
        analyzer = MetroNetworkAnalyzer(geojson_path, cost_per_km, network_name)
        self.networks.append(analyzer)
        self.network_names.append(network_name)
        print(f"\n✓ Loaded {network_name}:")
        print(f"  Nodes: {analyzer.G.number_of_nodes()}")
        print(f"  Edges: {analyzer.G.number_of_edges()}")
        
    def compare_baseline_metrics(self):
        """Compare baseline metrics across all networks"""
        print("\n" + "="*80)
        print("BASELINE METRICS COMPARISON")
        print("="*80)
        
        results = []
        for analyzer in self.networks:
            metrics = analyzer.get_metrics()
            results.append({
                'Network': analyzer.network_name,
                'Nodes': metrics['nodes'],
                'Edges': metrics['edges'],
                'Efficiency': metrics['efficiency'],
                'Robustness': metrics['robustness'],
                'Avg Criticality': metrics['avg_criticality'],
                'Max Criticality': metrics['max_criticality']
            })
        
        df = pd.DataFrame(results)
        print("\n" + df.to_string(index=False))
        
        # Create comparison visualizations
        self._plot_baseline_comparison(df)
        
        return df
    
    def compare_resilience_strategies(self, k_values=[2, 3]):
        """Compare resilience strategies across all networks"""
        print("\n" + "="*80)
        print("RESILIENCE STRATEGY COMPARISON")
        print("="*80)
        
        all_results = []
        
        for analyzer in self.networks:
            print(f"\n{'-'*80}")
            print(f"Analyzing {analyzer.network_name}")
            print(f"{'-'*80}")
            
            # Get original metrics
            G_original = analyzer.G.copy()
            original_metrics = analyzer.get_metrics()
            
            network_results = {
                'Network': analyzer.network_name,
                'Original_Efficiency': original_metrics['efficiency'],
                'Original_Robustness': original_metrics['robustness'],
                'Original_Avg_Criticality': original_metrics['avg_criticality']
            }
            
            # Test each k value
            for k in k_values:
                analyzer.G = G_original.copy()
                new_edges, cost, breakdown = analyzer.add_resilience_lines(k=k)
                
                new_metrics = analyzer.get_metrics()
                
                eff_improvement = ((new_metrics['efficiency'] - original_metrics['efficiency']) / 
                                  original_metrics['efficiency'] * 100)
                rob_improvement = ((new_metrics['robustness'] - original_metrics['robustness']) / 
                                  original_metrics['robustness'] * 100) if original_metrics['robustness'] > 0 else 0
                crit_reduction = ((original_metrics['avg_criticality'] - new_metrics['avg_criticality']) / 
                                 original_metrics['avg_criticality'] * 100)
                
                network_results[f'k{k}_Edges_Added'] = len(new_edges)
                network_results[f'k{k}_Cost_M'] = cost / 1e6
                network_results[f'k{k}_Efficiency'] = new_metrics['efficiency']
                network_results[f'k{k}_Efficiency_Imp_%'] = eff_improvement
                network_results[f'k{k}_Robustness'] = new_metrics['robustness']
                network_results[f'k{k}_Robustness_Imp_%'] = rob_improvement
                network_results[f'k{k}_Avg_Criticality'] = new_metrics['avg_criticality']
                network_results[f'k{k}_Criticality_Red_%'] = crit_reduction
                
                if eff_improvement > 0:
                    network_results[f'k{k}_Cost_Per_Eff_Gain_M'] = (cost / 1e6) / eff_improvement
                else:
                    network_results[f'k{k}_Cost_Per_Eff_Gain_M'] = float('inf')
                
                print(f"\nk={k} Results:")
                print(f"  Edges Added: {len(new_edges)}")
                print(f"  Cost: ${cost/1e6:.2f}M")
                print(f"  Efficiency Improvement: {eff_improvement:+.2f}%")
                print(f"  Robustness Improvement: {rob_improvement:+.2f}%")
                print(f"  Criticality Reduction: {crit_reduction:+.2f}%")
            
            # Restore original
            analyzer.G = G_original
            all_results.append(network_results)
        
        df = pd.DataFrame(all_results)
        
        # Create comprehensive comparison visualizations
        self._plot_resilience_comparison(df, k_values)
        
        return df
    
    def _plot_baseline_comparison(self, df):
        """Create baseline comparison charts"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor='white')
        fig.suptitle('Baseline Network Metrics Comparison', 
                    fontsize=18, weight='bold', color='#1e293b', y=0.995)
        
        colors = ['#3b82f6', '#10b981', '#f59e0b']
        
        metrics = [
            ('Nodes', 'Number of Stations', axes[0, 0]),
            ('Edges', 'Number of Connections', axes[0, 1]),
            ('Efficiency', 'Network Efficiency', axes[0, 2]),
            ('Robustness', 'Network Robustness', axes[1, 0]),
            ('Avg Criticality', 'Average Criticality', axes[1, 1]),
            ('Max Criticality', 'Maximum Criticality', axes[1, 2])
        ]
        
        for metric, title, ax in metrics:
            ax.set_facecolor('white')
            bars = ax.bar(df['Network'], df[metric], 
                         color=colors[:len(df)],
                         edgecolor='#1e293b', linewidth=2, alpha=0.85)
            ax.set_ylabel(metric, fontsize=12, color='#1e293b', weight='bold')
            ax.set_title(title, fontsize=13, color='#1e293b', weight='bold', pad=10)
            ax.tick_params(colors='#1e293b', labelsize=10, rotation=15)
            ax.grid(True, alpha=0.3, color='#cbd5e1', linestyle='--', axis='y')
            
            for spine in ax.spines.values():
                spine.set_edgecolor('#475569')
                spine.set_linewidth(1.5)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}' if metric in ['Efficiency', 'Robustness', 'Avg Criticality', 'Max Criticality'] else f'{int(height)}',
                       ha='center', va='bottom',
                       color='#1e293b', fontsize=10, weight='bold')
        
        plt.tight_layout()
        plt.savefig('multi_network_baseline_comparison.png', dpi=300, 
                   facecolor='white', bbox_inches='tight')
        print("\n✓ Saved: multi_network_baseline_comparison.png")
        plt.show()
    
    def _plot_resilience_comparison(self, df, k_values):
        """Create comprehensive resilience comparison charts"""
        n_metrics = 4  # Efficiency, Robustness, Criticality, Cost
        fig, axes = plt.subplots(n_metrics, len(k_values), 
                                figsize=(10*len(k_values), 16), facecolor='white')
        fig.suptitle('Resilience Strategy Comparison Across Networks', 
                    fontsize=20, weight='bold', color='#1e293b', y=0.995)
        
        if len(k_values) == 1:
            axes = axes.reshape(-1, 1)
        
        colors = ['#3b82f6', '#10b981', '#f59e0b']
        
        for k_idx, k in enumerate(k_values):
            # Efficiency Improvement
            ax = axes[0, k_idx]
            ax.set_facecolor('white')
            bars = ax.bar(df['Network'], df[f'k{k}_Efficiency_Imp_%'],
                         color=colors[:len(df)],
                         edgecolor='#1e293b', linewidth=2, alpha=0.85)
            ax.set_ylabel('Efficiency Improvement (%)', fontsize=12, 
                         color='#1e293b', weight='bold')
            ax.set_title(f'k={k}: Efficiency Improvement', fontsize=14,
                        color='#1e293b', weight='bold', pad=10)
            ax.tick_params(colors='#1e293b', labelsize=10)
            ax.grid(True, alpha=0.3, color='#cbd5e1', linestyle='--', axis='y')
            ax.axhline(y=0, color='#64748b', linestyle='-', linewidth=1)
            
            for spine in ax.spines.values():
                spine.set_edgecolor('#475569')
                spine.set_linewidth(1.5)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:+.2f}%', ha='center', 
                       va='bottom' if height >= 0 else 'top',
                       color='#1e293b', fontsize=10, weight='bold')
            
            # Robustness Improvement
            ax = axes[1, k_idx]
            ax.set_facecolor('white')
            bars = ax.bar(df['Network'], df[f'k{k}_Robustness_Imp_%'],
                         color=colors[:len(df)],
                         edgecolor='#1e293b', linewidth=2, alpha=0.85)
            ax.set_ylabel('Robustness Improvement (%)', fontsize=12,
                         color='#1e293b', weight='bold')
            ax.set_title(f'k={k}: Robustness Improvement', fontsize=14,
                        color='#1e293b', weight='bold', pad=10)
            ax.tick_params(colors='#1e293b', labelsize=10)
            ax.grid(True, alpha=0.3, color='#cbd5e1', linestyle='--', axis='y')
            ax.axhline(y=0, color='#64748b', linestyle='-', linewidth=1)
            
            for spine in ax.spines.values():
                spine.set_edgecolor('#475569')
                spine.set_linewidth(1.5)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:+.2f}%', ha='center',
                       va='bottom' if height >= 0 else 'top',
                       color='#1e293b', fontsize=10, weight='bold')
            
            # Criticality Reduction
            ax = axes[2, k_idx]
            ax.set_facecolor('white')
            bars = ax.bar(df['Network'], df[f'k{k}_Criticality_Red_%'],
                         color=colors[:len(df)],
                         edgecolor='#1e293b', linewidth=2, alpha=0.85)
            ax.set_ylabel('Avg Criticality Reduction (%)', fontsize=12,
                         color='#1e293b', weight='bold')
            ax.set_title(f'k={k}: Criticality Reduction', fontsize=14,
                        color='#1e293b', weight='bold', pad=10)
            ax.tick_params(colors='#1e293b', labelsize=10)
            ax.grid(True, alpha=0.3, color='#cbd5e1', linestyle='--', axis='y')
            ax.axhline(y=0, color='#64748b', linestyle='-', linewidth=1)
            
            for spine in ax.spines.values():
                spine.set_edgecolor('#475569')
                spine.set_linewidth(1.5)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:+.2f}%', ha='center',
                       va='bottom' if height >= 0 else 'top',
                       color='#1e293b', fontsize=10, weight='bold')
            
            # Total Cost
            ax = axes[3, k_idx]
            ax.set_facecolor('white')
            bars = ax.bar(df['Network'], df[f'k{k}_Cost_M'],
                         color=colors[:len(df)],
                         edgecolor='#1e293b', linewidth=2, alpha=0.85)
            ax.set_ylabel('Total Cost (Million $)', fontsize=12,
                         color='#1e293b', weight='bold')
            ax.set_title(f'k={k}: Construction Cost', fontsize=14,
                        color='#1e293b', weight='bold', pad=10)
            ax.tick_params(colors='#1e293b', labelsize=10)
            ax.grid(True, alpha=0.3, color='#cbd5e1', linestyle='--', axis='y')
            
            for spine in ax.spines.values():
                spine.set_edgecolor('#475569')
                spine.set_linewidth(1.5)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'${height:.1f}M', ha='center', va='bottom',
                       color='#1e293b', fontsize=10, weight='bold')
        
        plt.tight_layout()
        plt.savefig('multi_network_resilience_comparison.png', dpi=300,
                   facecolor='white', bbox_inches='tight')
        print("✓ Saved: multi_network_resilience_comparison.png")
        plt.show()
        
        # Create cost-benefit analysis chart
        self._plot_cost_benefit_analysis(df, k_values)
    
    def _plot_cost_benefit_analysis(self, df, k_values):
        """Create cost-benefit analysis comparison"""
        fig, axes = plt.subplots(1, len(k_values), 
                                figsize=(10*len(k_values), 6), facecolor='white')
        fig.suptitle('Cost-Benefit Analysis: Cost per 1% Efficiency Gain', 
                    fontsize=18, weight='bold', color='#1e293b', y=1.02)
        
        if len(k_values) == 1:
            axes = [axes]
        
        colors = ['#3b82f6', '#10b981', '#f59e0b']
        
        for k_idx, k in enumerate(k_values):
            ax = axes[k_idx]
            ax.set_facecolor('white')
            
            # Filter out infinite values
            valid_data = df[df[f'k{k}_Cost_Per_Eff_Gain_M'] != float('inf')]
            
            if len(valid_data) > 0:
                bars = ax.bar(valid_data['Network'], 
                            valid_data[f'k{k}_Cost_Per_Eff_Gain_M'],
                            color=colors[:len(valid_data)],
                            edgecolor='#1e293b', linewidth=2, alpha=0.85)
                ax.set_ylabel('Cost per 1% Efficiency Gain (Million $)', 
                            fontsize=12, color='#1e293b', weight='bold')
                ax.set_title(f'k={k} Strategy', fontsize=14,
                           color='#1e293b', weight='bold', pad=10)
                ax.tick_params(colors='#1e293b', labelsize=10)
                ax.grid(True, alpha=0.3, color='#cbd5e1', linestyle='--', axis='y')
                
                for spine in ax.spines.values():
                    spine.set_edgecolor('#475569')
                    spine.set_linewidth(1.5)
                
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'${height:.2f}M', ha='center', va='bottom',
                           color='#1e293b', fontsize=10, weight='bold')
            else:
                ax.text(0.5, 0.5, 'No positive efficiency gains', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12, color='#64748b')
        
        plt.tight_layout()
        plt.savefig('multi_network_cost_benefit.png', dpi=300,
                   facecolor='white', bbox_inches='tight')
        print("✓ Saved: multi_network_cost_benefit.png")
        plt.show()


def main():
    """Main execution function for multi-network comparison"""
    
    print("="*80)
    print("MULTI-NETWORK METRO RESILIENCE ANALYSIS")
    print("="*80)
    
    # Initialize comparator
    comparator = MultiNetworkComparator()
    
    # Add networks (replace with your actual GeoJSON files)
    # Example: Three different metro networks
    network_configs = [
        {
            'geojson_path': 'delhi_metro_stations_cleaned.geojson',
            'network_name': 'Delhi Metro',
            'cost_per_km': 50000000  # $50M per km
        },
        {
            'geojson_path': 'hk_metro_station.geojson',
            'network_name': 'Hong Kong Metro',
            'cost_per_km': 50000000
        },
        {
            'geojson_path': 'nyc_subway_stations_cleaned.geojson',
            'network_name': 'New York City Metro',
            'cost_per_km': 50000000
        }
    ]
    
    print("\nLoading networks...")
    print("-"*80)
    
    for config in network_configs:
        try:
            comparator.add_network(
                geojson_path=config['geojson_path'],
                network_name=config['network_name'],
                cost_per_km=config['cost_per_km']
            )
        except FileNotFoundError:
            print(f"⚠ Warning: {config['geojson_path']} not found. Skipping...")
        except Exception as e:
            print(f"⚠ Error loading {config['network_name']}: {str(e)}")
    
    if len(comparator.networks) == 0:
        print("\n❌ No networks loaded. Please check your GeoJSON file paths.")
        return
    
    # Step 1: Compare baseline metrics
    print("\n" + "="*80)
    print("STEP 1: Baseline Metrics Comparison")
    print("="*80)
    baseline_df = comparator.compare_baseline_metrics()
    
    # Step 2: Compare resilience strategies
    print("\n" + "="*80)
    print("STEP 2: Resilience Strategy Analysis")
    print("="*80)
    resilience_df = comparator.compare_resilience_strategies(k_values=[2, 3])
    
    # Save results to CSV
    baseline_df.to_csv('baseline_comparison.csv', index=False)
    resilience_df.to_csv('resilience_comparison.csv', index=False)
    print("\n✓ Saved results to CSV files:")
    print("  - baseline_comparison.csv")
    print("  - resilience_comparison.csv")
    
    # Print summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nNetworks analyzed: {len(comparator.networks)}")
    for analyzer in comparator.networks:
        print(f"  • {analyzer.network_name}")
    
    print("\nGenerated visualizations:")
    print("  • multi_network_baseline_comparison.png")
    print("  • multi_network_resilience_comparison.png")
    print("  • multi_network_cost_benefit.png")
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


# Alternative: Single network analysis (original functionality)
def analyze_single_network(geojson_path, network_name="Metro Network", cost_per_km=50000000):
    """Analyze a single metro network with full details"""
    
    print("="*80)
    print(f"SINGLE NETWORK ANALYSIS: {network_name}")
    print("="*80)
    
    analyzer = MetroNetworkAnalyzer(geojson_path, cost_per_km, network_name)
    
    print(f"\nNetwork Statistics:")
    print(f"  Nodes: {analyzer.G.number_of_nodes()}")
    print(f"  Edges: {analyzer.G.number_of_edges()}")
    print(f"  Cost per km: ${cost_per_km:,.2f}")
    
    # Get baseline metrics
    print("\n" + "="*80)
    print("Baseline Metrics")
    print("="*80)
    
    metrics = analyzer.get_metrics()
    print(f"\nEfficiency: {metrics['efficiency']:.6f}")
    print(f"Robustness: {metrics['robustness']:.6f}")
    print(f"Average Criticality: {metrics['avg_criticality']:.6f}")
    print(f"Maximum Criticality: {metrics['max_criticality']:.6f}")
    
    # Show top critical nodes
    criticality = metrics['criticality']
    top_critical = sorted(criticality.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 Critical Nodes:")
    for i, (node, score) in enumerate(top_critical, 1):
        print(f"{i:2d}. {analyzer.stations[node]['name']:30s} - {score:.6f}")
    
    # Test resilience strategies
    print("\n" + "="*80)
    print("Testing Resilience Strategies")
    print("="*80)
    
    G_original = analyzer.G.copy()
    
    for k in [2, 3]:
        print(f"\n{'-'*80}")
        print(f"k={k} Resilience Strategy")
        print(f"{'-'*80}")
        
        analyzer.G = G_original.copy()
        new_edges, cost, breakdown = analyzer.add_resilience_lines(k=k)
        
        print(f"\nEdges added: {len(new_edges)}")
        print(f"Total cost: ${cost:,.2f} ({cost/1e6:.2f}M)")
        
        print("\nCost breakdown:")
        for edge_name, details in breakdown.items():
            print(f"  {edge_name}")
            print(f"    Distance: {details['distance_km']:.2f} km")
            print(f"    Cost: ${details['cost']:,.2f}")
        
        new_metrics = analyzer.get_metrics()
        eff_imp = ((new_metrics['efficiency'] - metrics['efficiency']) / 
                   metrics['efficiency'] * 100)
        rob_imp = ((new_metrics['robustness'] - metrics['robustness']) / 
                   metrics['robustness'] * 100) if metrics['robustness'] > 0 else 0
        
        print(f"\nResults:")
        print(f"  New Efficiency: {new_metrics['efficiency']:.6f} ({eff_imp:+.2f}%)")
        print(f"  New Robustness: {new_metrics['robustness']:.6f} ({rob_imp:+.2f}%)")
    
    analyzer.G = G_original
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    
    return analyzer


if __name__ == "__main__":
    # Choose analysis mode
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--single':
        # Single network analysis
        geojson_path = sys.argv[2] if len(sys.argv) > 2 else "delhi_metro_stations.geojson"
        network_name = sys.argv[3] if len(sys.argv) > 3 else "Metro Network"
        analyze_single_network(geojson_path, network_name)
    else:
        # Multi-network comparison (default)
        main()