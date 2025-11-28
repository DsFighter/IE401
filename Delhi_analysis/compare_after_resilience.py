"""
PLUG-AND-PLAY METRO NETWORK RESILIENCE ANALYSIS
================================================
Simply run this script with your GeoJSON file!

Usage:
    python script.py your_metro_network.geojson

Or edit the configuration below and run directly.
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
from collections import defaultdict
import warnings
import sys
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================================

# Default GeoJSON file (change this to your file)
DEFAULT_GEOJSON = "nyc_subway_stations_cleaned.geojson"

# Network name for display
NETWORK_NAME = "Metro Network"

# Cost per kilometer for new connections (in currency units)
COST_PER_KM = 50000000  # $50 million per km

# Analysis options
RUN_TARGETED_ATTACK = True
RUN_RADIUS_ATTACK = True
RUN_ATTACK_COMPARISON = True

# Visualization options
COLOR_CODE_LINES = True
FLIP_AXES = False
NODE_SIZE = 80

# ============================================================================
# MAIN ANALYSIS CLASS
# ============================================================================

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class MetroNetworkAnalyzer:
    """Main class for metro network analysis with enhanced line detection"""
    
    def __init__(self, geojson_path, cost_per_km=50000000, network_name="Network"):
        """Initialize with GeoJSON file path and cost per km"""
        self.geojson_path = geojson_path
        self.cost_per_km = cost_per_km
        self.network_name = network_name
        self.G = None
        self.stations = {}
        self.line_colors = {}
        self.lines_data = {}
        self.load_network()
        
    def load_network(self):
        """Load GeoJSON and build network graph with enhanced line detection"""
        with open(self.geojson_path, 'r') as f:
            data = json.load(f)
        
        print(f"\n{'='*80}")
        print(f"LOADING NETWORK: {self.network_name}")
        print(f"{'='*80}")
        
        self.G = nx.Graph()
        
        # Step 1: Extract all stations
        print("\nStep 1: Extracting stations...")
        self._extract_stations(data)
        
        # Step 2: Detect and separate lines
        print("\nStep 2: Detecting and separating metro lines...")
        self._detect_and_separate_lines()
        
        # Step 3: Build network edges based on line connectivity
        print("\nStep 3: Building network edges...")
        self._build_edges()
        
        # Step 4: Assign colors to lines
        print("\nStep 4: Assigning line colors...")
        self._assign_line_colors()
        
        print(f"\n✓ Network loaded successfully!")
        print(f"  Total Stations: {len(self.stations)}")
        print(f"  Total Lines: {len(self.lines_data)}")
        print(f"  Total Edges: {self.G.number_of_edges()}")
        
    def _extract_stations(self, data):
        """Extract stations from GeoJSON"""
        station_count = 0
        
        for feature in data['features']:
            if feature['geometry']['type'] == 'Point':
                coords = feature['geometry']['coordinates']
                name = feature['properties']['name']
                lines = feature['properties']['lines']
                
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
                
                station_count += 1
        
        print(f"  Extracted {station_count} stations")
    
    def _detect_and_separate_lines(self):
        """Detect and separate metro lines with proper ordering"""
        line_stations = defaultdict(list)
        
        for station_id, data in self.stations.items():
            for line in data['lines']:
                line_stations[line].append(station_id)
        
        print(f"  Found {len(line_stations)} unique lines:")
        
        for line_name, stations in line_stations.items():
            if len(stations) < 2:
                print(f"    ⚠ {line_name}: Only {len(stations)} station(s) - skipping")
                continue
            
            ordered_stations = self._order_stations_by_proximity(stations)
            ordered_stations, branches = self._detect_branches(line_name, ordered_stations)
            
            self.lines_data[line_name] = {
                'stations': ordered_stations,
                'branches': branches,
                'station_count': len(stations)
            }
            
            branch_info = f" ({len(branches)} branches)" if branches else ""
            print(f"    ✓ {line_name}: {len(stations)} stations{branch_info}")
    
    def _order_stations_by_proximity(self, station_ids):
        """Order stations using nearest-neighbor approach"""
        if len(station_ids) <= 2:
            return station_ids
        
        max_min_dist = -1
        start_station = station_ids[0]
        
        for station_id in station_ids:
            coord = self.stations[station_id]['coords']
            min_dist_to_others = float('inf')
            
            for other_id in station_ids:
                if station_id != other_id:
                    other_coord = self.stations[other_id]['coords']
                    dist = self._haversine_distance(coord, other_coord)
                    min_dist_to_others = min(min_dist_to_others, dist)
            
            if min_dist_to_others > max_min_dist:
                max_min_dist = min_dist_to_others
                start_station = station_id
        
        ordered = [start_station]
        remaining = [s for s in station_ids if s != start_station]
        
        while remaining:
            current_coord = self.stations[ordered[-1]]['coords']
            nearest_dist = float('inf')
            nearest_station = None
            
            for station_id in remaining:
                station_coord = self.stations[station_id]['coords']
                dist = self._haversine_distance(current_coord, station_coord)
                
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_station = station_id
            
            if nearest_station:
                ordered.append(nearest_station)
                remaining.remove(nearest_station)
            else:
                ordered.extend(remaining)
                break
        
        return ordered
    
    def _detect_branches(self, line_name, ordered_stations):
        """Detect if a line has branches"""
        if len(ordered_stations) < 4:
            return ordered_stations, {}
        
        branches = {}
        
        for i in range(1, len(ordered_stations) - 1):
            current_id = ordered_stations[i]
            current_coord = self.stations[current_id]['coords']
            
            distances_ahead = []
            for j in range(i + 1, len(ordered_stations)):
                ahead_coord = self.stations[ordered_stations[j]]['coords']
                dist = self._haversine_distance(current_coord, ahead_coord)
                distances_ahead.append((ordered_stations[j], dist))
            
            if len(distances_ahead) >= 2:
                distances_ahead.sort(key=lambda x: x[1])
                
                if distances_ahead[0][1] > 0:
                    ratio = distances_ahead[1][1] / distances_ahead[0][1]
                    if 0.8 <= ratio <= 1.2:
                        branches[f"branch_{len(branches)+1}"] = {
                            'junction': current_id,
                            'stations': [distances_ahead[1][0]]
                        }
        
        return ordered_stations, branches
    
    def _build_edges(self):
        """Build edges between consecutive stations on each line"""
        total_edges = 0
        
        for line_name, line_data in self.lines_data.items():
            stations = line_data['stations']
            line_edges = 0
            
            for i in range(len(stations) - 1):
                s1 = stations[i]
                s2 = stations[i + 1]
                
                if (line_name in self.stations[s1]['lines'] and 
                    line_name in self.stations[s2]['lines']):
                    
                    dist = self._haversine_distance(
                        self.stations[s1]['coords'],
                        self.stations[s2]['coords']
                    )
                    
                    if not self.G.has_edge(s1, s2):
                        self.G.add_edge(s1, s2, weight=dist, line=line_name)
                        line_edges += 1
                    else:
                        existing_line = self.G[s1][s2].get('line', '')
                        if line_name not in existing_line:
                            self.G[s1][s2]['line'] = f"{existing_line},{line_name}"
            
            for branch_name, branch_data in line_data['branches'].items():
                junction = branch_data['junction']
                for branch_station in branch_data['stations']:
                    if not self.G.has_edge(junction, branch_station):
                        dist = self._haversine_distance(
                            self.stations[junction]['coords'],
                            self.stations[branch_station]['coords']
                        )
                        self.G.add_edge(junction, branch_station, 
                                      weight=dist, line=line_name)
                        line_edges += 1
            
            total_edges += line_edges
            print(f"    {line_name}: {line_edges} edges added")
        
        interchange_edges = self._add_interchange_connections()
        total_edges += interchange_edges
        
        print(f"  Total edges created: {total_edges}")
    
    def _add_interchange_connections(self):
        """Add connections between interchange stations"""
        station_names = defaultdict(list)
        interchange_count = 0
        
        for station_id, data in self.stations.items():
            name = data['name']
            station_names[name].append(station_id)
        
        for name, station_list in station_names.items():
            if len(station_list) > 1:
                for i in range(len(station_list)):
                    for j in range(i + 1, len(station_list)):
                        s1, s2 = station_list[i], station_list[j]
                        
                        if not self.G.has_edge(s1, s2):
                            dist = self._haversine_distance(
                                self.stations[s1]['coords'],
                                self.stations[s2]['coords']
                            )
                            
                            if dist < 0.5:
                                self.G.add_edge(s1, s2, weight=0.01, line='interchange')
                                interchange_count += 1
        
        if interchange_count > 0:
            print(f"    interchange: {interchange_count} connections added")
        
        return interchange_count
        
    def _assign_line_colors(self):
        """Assign unique colors to each metro line"""
        all_lines = set(self.lines_data.keys())
        
        color_palette = [
            '#ef4444', '#3b82f6', '#10b981', '#f59e0b', '#8b5cf6',
            '#ec4899', '#06b6d4', '#f97316', '#14b8a6', '#a855f7',
            '#84cc16', '#6366f1', '#f43f5e', '#0ea5e9', '#22c55e',
        ]
        
        for i, line in enumerate(sorted(all_lines)):
            if line == 'interchange':
                self.line_colors[line] = '#64748b'
            elif line.startswith('resilience'):
                self.line_colors[line] = '#10b981'
            else:
                self.line_colors[line] = color_palette[i % len(color_palette)]
    
    def _haversine_distance(self, coord1, coord2):
        """Calculate distance between two coordinates in km"""
        lon1, lat1 = coord1
        lon2, lat2 = coord2
        
        R = 6371
        
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        
        a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def get_line_summary(self):
        """Get summary of all detected lines"""
        summary = []
        
        for line_name, line_data in self.lines_data.items():
            summary.append({
                'Line': line_name,
                'Stations': line_data['station_count'],
                'Branches': len(line_data['branches']),
                'Color': self.line_colors.get(line_name, 'N/A')
            })
        
        return pd.DataFrame(summary)
    
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
    
    def plot_network(self, criticality=None, title="Metro Network", 
                    removed_nodes=None, filename=None, figsize=(16, 12),
                    node_size=80, flip_axes=False, added_edges=None, 
                    color_code_lines=True):
        """Plot network with line visualization"""
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        ax.set_facecolor('white')
        
        pos = {}
        for station_id, station_data in self.stations.items():
            pos[station_id] = (station_data['coords'][0], station_data['coords'][1])
        
        if criticality is None:
            criticality = self.calculate_criticality()
        
        node_colors = [criticality.get(node, 0) for node in self.G.nodes()]
        
        # Draw edges by line
        edges_by_line = defaultdict(list)
        
        for u, v, data in self.G.edges(data=True):
            line = data.get('line', 'unknown')
            
            if ',' in line:
                lines = line.split(',')
                for single_line in lines:
                    if u in pos and v in pos:
                        edge_coord = self._get_edge_coords(pos, u, v, flip_axes)
                        edges_by_line[single_line].append(edge_coord)
            else:
                if u in pos and v in pos:
                    edge_coord = self._get_edge_coords(pos, u, v, flip_axes)
                    edges_by_line[line].append(edge_coord)
        
        legend_elements = []
        for line, edge_coords in edges_by_line.items():
            if not edge_coords:
                continue
            
            is_resilience = line.startswith('resilience')
            
            if is_resilience:
                color = '#10b981'
                width = 3.5
                alpha = 0.9
            elif color_code_lines:
                color = self.line_colors.get(line, '#64748b')
                width = 2.5
                alpha = 0.7
            else:
                color = '#64748b'
                width = 2.5
                alpha = 0.7
            
            lc = LineCollection(edge_coords, colors=color, 
                              linewidths=width, alpha=alpha,
                              zorder=2 if is_resilience else 1)
            ax.add_collection(lc)
            
            if color_code_lines and line != 'interchange' and not is_resilience:
                from matplotlib.lines import Line2D
                legend_elements.append(Line2D([0], [0], color=color, 
                                             linewidth=3, label=line))
        
        # Draw nodes
        active_nodes = list(self.G.nodes())
        if flip_axes:
            node_positions = np.array([[pos[node][1], pos[node][0]] for node in active_nodes])
        else:
            node_positions = np.array([pos[node] for node in active_nodes])
        
        scatter = ax.scatter(node_positions[:, 0], node_positions[:, 1],
                           c=node_colors, cmap='YlOrRd', s=node_size,
                           edgecolors='#1e293b', linewidths=1.5,
                           alpha=0.9, zorder=3)
        
        # Draw removed nodes
        if removed_nodes:
            removed_nodes_list = [node for node in removed_nodes if node in pos]
            if len(removed_nodes_list) > 0:
                if flip_axes:
                    removed_pos = np.array([[pos[node][1], pos[node][0]] 
                                           for node in removed_nodes_list])
                else:
                    removed_pos = np.array([pos[node] for node in removed_nodes_list])
                
                ax.scatter(removed_pos[:, 0], removed_pos[:, 1],
                         s=node_size, c='#10b981', 
                         edgecolors='#1e293b', linewidths=1.5,
                         alpha=0.9, zorder=6)
                
                from matplotlib.lines import Line2D
                legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                             markerfacecolor='#10b981', markersize=8,
                                             markeredgecolor='#1e293b', markeredgewidth=1.5,
                                             linestyle='None',
                                             label='Removed Nodes'))
        
        if added_edges:
            from matplotlib.lines import Line2D
            legend_elements.append(Line2D([0], [0], color='#10b981', linewidth=3,
                                         label='Resilience Links'))
        
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Criticality Score', fontsize=14, color='#1e293b', weight='bold')
        cbar.ax.tick_params(labelsize=11, colors='#1e293b')
        
        ax.set_title(title, fontsize=20, weight='bold', color='#1e293b', pad=20)
        
        if flip_axes:
            ax.set_xlabel('Latitude', fontsize=14, color='#1e293b', weight='bold')
            ax.set_ylabel('Longitude', fontsize=14, color='#1e293b', weight='bold')
        else:
            ax.set_xlabel('Longitude', fontsize=14, color='#1e293b', weight='bold')
            ax.set_ylabel('Latitude', fontsize=14, color='#1e293b', weight='bold')
        
        ax.tick_params(colors='#1e293b', labelsize=11)
        ax.grid(True, alpha=0.3, color='#cbd5e1', linestyle='--')
        
        for spine in ax.spines.values():
            spine.set_edgecolor('#475569')
            spine.set_linewidth(1.5)
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='lower left', 
                     fontsize=11, framealpha=0.95, edgecolor='#475569',
                     title='Metro Lines' if color_code_lines else 'Network Elements', 
                     title_fontsize=12)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, facecolor='white', 
                       edgecolor='none', bbox_inches='tight')
            print(f"✓ Saved plot: {filename}")
        
        plt.show()
    
    def _get_edge_coords(self, pos, u, v, flip_axes):
        """Get edge coordinates with optional axis flipping"""
        if flip_axes:
            return [(pos[u][1], pos[u][0]), (pos[v][1], pos[v][0])]
        else:
            return [pos[u], pos[v]]
    
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
        
        print(f"\nAdding k={k} resilience connections (line-based only)...")
        
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
                if target_line == line or target_line == 'interchange':
                    continue
                
                if target_line not in line_nodes:
                    continue
                
                min_dist = float('inf')
                nearest_node = None
                
                kth_coords = self.stations[kth_critical_node]['coords']
                
                for target_node in line_nodes[target_line]:
                    if target_line not in self.stations[target_node]['lines']:
                        continue
                    
                    if self.G.has_edge(kth_critical_node, target_node):
                        continue
                    
                    dist = self._haversine_distance(
                        kth_coords,
                        self.stations[target_node]['coords']
                    )
                    
                    if dist < min_dist:
                        min_dist = dist
                        nearest_node = target_node
                
                if nearest_node and not self.G.has_edge(kth_critical_node, nearest_node):
                    kth_lines = self.stations[kth_critical_node]['lines']
                    nearest_lines = self.stations[nearest_node]['lines']
                    
                    if line in kth_lines and target_line in nearest_lines:
                        self.G.add_edge(kth_critical_node, nearest_node,
                                       weight=min_dist, line=f'resilience_{k}')
                        new_edges.append((kth_critical_node, nearest_node))
                        
                        edge_cost = min_dist * self.cost_per_km
                        total_cost += edge_cost
                        
                        kth_node_info = f"{self.stations[kth_critical_node]['name']} ({line})"
                        nearest_node_info = f"{self.stations[nearest_node]['name']} ({target_line})"
                        edge_name = f"{kth_node_info} → {nearest_node_info}"
                        cost_breakdown[edge_name] = {
                            'distance_km': min_dist,
                            'cost': edge_cost,
                            'from_line': line,
                            'to_line': target_line
                        }
                        
                        print(f"  ✓ Added: {edge_name}")
                        print(f"    Distance: {min_dist:.2f} km, Cost: ${edge_cost/1e6:.2f}M")
        
        self._assign_line_colors()
        
        print(f"\nTotal resilience connections added: {len(new_edges)}")
        print(f"Total construction cost: ${total_cost:,.2f} ({total_cost/1e6:.2f}M)")
        
        return new_edges, total_cost, cost_breakdown
    
    def compare_networks(self, color_code_lines=False, run_attack_comparison=True):
        """Compare original and resilience-enhanced networks with attack simulations"""
        print(f"\n{'='*60}")
        print("NETWORK COMPARISON")
        print(f"{'='*60}")
        
        G_original = self.G.copy()
        criticality_original = self.calculate_criticality()
        efficiency_original = self.calculate_efficiency()
        
        sample_nodes = list(self.G.nodes())[:min(10, len(self.G.nodes()))]
        original_paths = []
        for i in range(len(sample_nodes)):
            for j in range(i+1, len(sample_nodes)):
                try:
                    paths = list(nx.all_shortest_paths(self.G, 
                                                      sample_nodes[i], 
                                                      sample_nodes[j], 
                                                      weight='weight'))
                    original_paths.extend(paths[:3])
                except:
                    pass
        
        robustness_original = self.calculate_robustness(original_paths)
        
        results = []
        attack_comparison_data = []
        
        # Run original network attacks if requested
        if run_attack_comparison:
            print("\n" + "="*60)
            print("BASELINE ATTACK SIMULATIONS (Original Network)")
            print("="*60)
            original_attack_results = self._run_attack_simulation("Original", color_code_lines, save_plots=False)
            attack_comparison_data.append(original_attack_results)
        
        # Test with k=2 resilience
        new_edges_k2, cost_k2, breakdown_k2 = self.add_resilience_lines(k=2)
        criticality_k2 = self.calculate_criticality()
        efficiency_k2 = self.calculate_efficiency()
        
        print("\n" + "="*60)
        print("K=2 RESILIENCE COST BREAKDOWN")
        print("="*60)
        for edge_name, details in breakdown_k2.items():
            print(f"{edge_name}:")
            print(f"  Distance: {details['distance_km']:.2f} km")
            print(f"  Cost: ${details['cost']:,.2f} ({details['cost']/1e6:.2f} million)")
        
        k2_paths = []
        for i in range(len(sample_nodes)):
            for j in range(i+1, len(sample_nodes)):
                try:
                    paths = list(nx.all_shortest_paths(self.G, 
                                                      sample_nodes[i], 
                                                      sample_nodes[j], 
                                                      weight='weight'))
                    k2_paths.extend(paths[:3])
                except:
                    pass
        
        robustness_k2 = self.calculate_robustness(k2_paths)
        
        self.plot_network(
            criticality=criticality_k2,
            title="Network with k=2 Resilience Connections",
            filename="network_resilience_k2.png",
            node_size=NODE_SIZE,
            flip_axes=FLIP_AXES,
            added_edges=new_edges_k2,
            color_code_lines=color_code_lines
        )
        
        results.append({
            'network': 'k=2 Resilience',
            'edges_added': len(new_edges_k2),
            'total_cost': cost_k2,
            'efficiency': efficiency_k2,
            'robustness': robustness_k2,
            'efficiency_improvement_%': ((efficiency_k2 - efficiency_original) / 
                                        efficiency_original * 100),
            'robustness_improvement_%': ((robustness_k2 - robustness_original) / 
                                        robustness_original * 100) if robustness_original > 0 else 0,
            'cost_per_efficiency_gain': cost_k2 / ((efficiency_k2 - efficiency_original) * 100) if efficiency_k2 > efficiency_original else float('inf')
        })
        
        # Run k=2 attack simulations if requested
        if run_attack_comparison:
            print("\n" + "="*60)
            print("ATTACK SIMULATIONS (k=2 Resilience Network)")
            print("="*60)
            k2_attack_results = self._run_attack_simulation("k=2 Resilience", color_code_lines, save_plots=False)
            attack_comparison_data.append(k2_attack_results)
        
        # Restore and test with k=3
        self.G = G_original.copy()
        new_edges_k3, cost_k3, breakdown_k3 = self.add_resilience_lines(k=3)
        criticality_k3 = self.calculate_criticality()
        efficiency_k3 = self.calculate_efficiency()
        
        print("\n" + "="*60)
        print("K=3 RESILIENCE COST BREAKDOWN")
        print("="*60)
        for edge_name, details in breakdown_k3.items():
            print(f"{edge_name}:")
            print(f"  Distance: {details['distance_km']:.2f} km")
            print(f"  Cost: ${details['cost']:,.2f} ({details['cost']/1e6:.2f} million)")
        
        k3_paths = []
        for i in range(len(sample_nodes)):
            for j in range(i+1, len(sample_nodes)):
                try:
                    paths = list(nx.all_shortest_paths(self.G, 
                                                      sample_nodes[i], 
                                                      sample_nodes[j], 
                                                      weight='weight'))
                    k3_paths.extend(paths[:3])
                except:
                    pass
        
        robustness_k3 = self.calculate_robustness(k3_paths)
        
        self.plot_network(
            criticality=criticality_k3,
            title="Network with k=3 Resilience Connections",
            filename="network_resilience_k3.png",
            node_size=NODE_SIZE,
            flip_axes=FLIP_AXES,
            added_edges=new_edges_k3,
            color_code_lines=color_code_lines
        )
        
        results.append({
            'network': 'k=3 Resilience',
            'edges_added': len(new_edges_k3),
            'total_cost': cost_k3,
            'efficiency': efficiency_k3,
            'robustness': robustness_k3,
            'efficiency_improvement_%': ((efficiency_k3 - efficiency_original) / 
                                        efficiency_original * 100),
            'robustness_improvement_%': ((robustness_k3 - robustness_original) / 
                                        robustness_original * 100) if robustness_original > 0 else 0,
            'cost_per_efficiency_gain': cost_k3 / ((efficiency_k3 - efficiency_original) * 100) if efficiency_k3 > efficiency_original else float('inf')
        })
        
        # Run k=3 attack simulations if requested
        if run_attack_comparison:
            print("\n" + "="*60)
            print("ATTACK SIMULATIONS (k=3 Resilience Network)")
            print("="*60)
            k3_attack_results = self._run_attack_simulation("k=3 Resilience", color_code_lines, save_plots=False)
            attack_comparison_data.append(k3_attack_results)
        
        # Print comparison
        print(f"\n{'='*60}")
        print("NETWORK COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"\nOriginal Network:")
        print(f"  Efficiency: {efficiency_original:.6f}")
        print(f"  Robustness: {robustness_original:.6f}")
        print(f"  Total Cost: $0 (baseline)")
        
        for result in results:
            print(f"\n{result['network']}:")
            print(f"  Edges Added: {result['edges_added']}")
            print(f"  Total Cost: ${result['total_cost']:,.2f} ({result['total_cost']/1e6:.2f} million)")
            print(f"  Efficiency: {result['efficiency']:.6f} "
                  f"({result['efficiency_improvement_%']:+.2f}%)")
            print(f"  Robustness: {result['robustness']:.6f} "
                  f"({result['robustness_improvement_%']:+.2f}%)")
            if result['cost_per_efficiency_gain'] != float('inf'):
                print(f"  Cost per 1% efficiency gain: ${result['cost_per_efficiency_gain']:,.2f}")
        
        # Create comparison plots
        self._plot_comparison_charts(efficiency_original, robustness_original, results)
        
        # Create attack comparison plots if data available
        if run_attack_comparison and attack_comparison_data:
            self._plot_attack_comparison(attack_comparison_data)
        
        # Restore original
        self.G = G_original
        
        return results
    
    def _run_attack_simulation(self, network_name, color_code_lines, save_plots=False):
        """Run attack simulation and return results without plotting"""
        criticality = self.calculate_criticality()
        sorted_nodes = sorted(criticality.items(), key=lambda x: x[1], reverse=True)
        
        G_copy = self.G.copy()
        original_efficiency = self.calculate_efficiency()
        
        # Targeted attack (top 3 nodes)
        targeted_results = []
        removed_nodes = []
        
        for i in range(min(3, len(sorted_nodes))):
            node, crit_score = sorted_nodes[i]
            removed_nodes.append(node)
            self.G.remove_node(node)
            
            new_efficiency = self.calculate_efficiency()
            efficiency_drop = ((original_efficiency - new_efficiency) / 
                             original_efficiency * 100)
            
            targeted_results.append({
                'nodes_removed': i + 1,
                'efficiency': new_efficiency,
                'efficiency_drop_%': efficiency_drop
            })
        
        # Restore graph
        self.G = G_copy.copy()
        
        # Radius attack
        most_critical = max(criticality.items(), key=lambda x: x[1])
        most_critical_node = most_critical[0]
        center_coords = self.stations[most_critical_node]['coords']
        
        radius_results = []
        for radius_km in [1.0, 2.0, 3.0]:
            nodes_to_remove = []
            for node in self.G.nodes():
                if node == most_critical_node:
                    nodes_to_remove.append(node)
                    continue
                dist = self._haversine_distance(center_coords, 
                                               self.stations[node]['coords'])
                if dist <= radius_km:
                    nodes_to_remove.append(node)
            
            for node in nodes_to_remove:
                self.G.remove_node(node)
            
            new_efficiency = self.calculate_efficiency()
            efficiency_drop = ((original_efficiency - new_efficiency) / 
                             original_efficiency * 100)
            
            radius_results.append({
                'radius_km': radius_km,
                'nodes_removed': len(nodes_to_remove),
                'efficiency': new_efficiency,
                'efficiency_drop_%': efficiency_drop
            })
            
            # Restore for next iteration
            self.G = G_copy.copy()
        
        # Restore original
        self.G = G_copy
        
        return {
            'network_name': network_name,
            'original_efficiency': original_efficiency,
            'targeted_results': targeted_results,
            'radius_results': radius_results
        }
    
    def _plot_attack_comparison(self, attack_data):
        """Create comprehensive attack comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), facecolor='white')
        fig.suptitle('Attack Resilience Comparison: Original vs Enhanced Networks', 
                    fontsize=18, weight='bold', color='#1e293b', y=0.995)
        
        colors = ['#3b82f6', '#10b981', '#f59e0b']
        
        # Plot 1: Targeted Attack - Efficiency after removing nodes
        ax1 = axes[0, 0]
        ax1.set_facecolor('white')
        
        for i, data in enumerate(attack_data):
            nodes_removed = [r['nodes_removed'] for r in data['targeted_results']]
            efficiencies = [r['efficiency'] for r in data['targeted_results']]
            ax1.plot(nodes_removed, efficiencies, marker='o', linewidth=3, 
                    markersize=10, label=data['network_name'], color=colors[i], alpha=0.8)
        
        ax1.set_xlabel('Number of Critical Nodes Removed', fontsize=12, 
                      color='#1e293b', weight='bold')
        ax1.set_ylabel('Network Efficiency', fontsize=12, color='#1e293b', weight='bold')
        ax1.set_title('Targeted Attack: Efficiency Degradation', fontsize=14,
                     color='#1e293b', weight='bold', pad=10)
        ax1.legend(fontsize=11, framealpha=0.95)
        ax1.grid(True, alpha=0.3, color='#cbd5e1', linestyle='--')
        ax1.tick_params(colors='#1e293b', labelsize=10)
        
        for spine in ax1.spines.values():
            spine.set_edgecolor('#475569')
            spine.set_linewidth(1.5)
        
        # Plot 2: Targeted Attack - Efficiency drop percentage
        ax2 = axes[0, 1]
        ax2.set_facecolor('white')
        
        for i, data in enumerate(attack_data):
            nodes_removed = [r['nodes_removed'] for r in data['targeted_results']]
            efficiency_drops = [r['efficiency_drop_%'] for r in data['targeted_results']]
            ax2.plot(nodes_removed, efficiency_drops, marker='s', linewidth=3,
                    markersize=10, label=data['network_name'], color=colors[i], alpha=0.8)
        
        ax2.set_xlabel('Number of Critical Nodes Removed', fontsize=12,
                      color='#1e293b', weight='bold')
        ax2.set_ylabel('Efficiency Drop (%)', fontsize=12, color='#1e293b', weight='bold')
        ax2.set_title('Targeted Attack: Percentage Impact', fontsize=14,
                     color='#1e293b', weight='bold', pad=10)
        ax2.legend(fontsize=11, framealpha=0.95)
        ax2.grid(True, alpha=0.3, color='#cbd5e1', linestyle='--')
        ax2.tick_params(colors='#1e293b', labelsize=10)
        
        for spine in ax2.spines.values():
            spine.set_edgecolor('#475569')
            spine.set_linewidth(1.5)
        
        # Plot 3: Radius Attack - Efficiency vs radius
        ax3 = axes[1, 0]
        ax3.set_facecolor('white')
        
        for i, data in enumerate(attack_data):
            radii = [r['radius_km'] for r in data['radius_results']]
            efficiencies = [r['efficiency'] for r in data['radius_results']]
            ax3.plot(radii, efficiencies, marker='D', linewidth=3,
                    markersize=10, label=data['network_name'], color=colors[i], alpha=0.8)
        
        ax3.set_xlabel('Attack Radius (km)', fontsize=12, color='#1e293b', weight='bold')
        ax3.set_ylabel('Network Efficiency', fontsize=12, color='#1e293b', weight='bold')
        ax3.set_title('Radius Attack: Efficiency Degradation', fontsize=14,
                     color='#1e293b', weight='bold', pad=10)
        ax3.legend(fontsize=11, framealpha=0.95)
        ax3.grid(True, alpha=0.3, color='#cbd5e1', linestyle='--')
        ax3.tick_params(colors='#1e293b', labelsize=10)
        
        for spine in ax3.spines.values():
            spine.set_edgecolor('#475569')
            spine.set_linewidth(1.5)
        
        # Plot 4: Radius Attack - Efficiency drop percentage
        ax4 = axes[1, 1]
        ax4.set_facecolor('white')
        
        for i, data in enumerate(attack_data):
            radii = [r['radius_km'] for r in data['radius_results']]
            efficiency_drops = [r['efficiency_drop_%'] for r in data['radius_results']]
            ax4.plot(radii, efficiency_drops, marker='^', linewidth=3,
                    markersize=10, label=data['network_name'], color=colors[i], alpha=0.8)
        
        ax4.set_xlabel('Attack Radius (km)', fontsize=12, color='#1e293b', weight='bold')
        ax4.set_ylabel('Efficiency Drop (%)', fontsize=12, color='#1e293b', weight='bold')
        ax4.set_title('Radius Attack: Percentage Impact', fontsize=14,
                     color='#1e293b', weight='bold', pad=10)
        ax4.legend(fontsize=11, framealpha=0.95)
        ax4.grid(True, alpha=0.3, color='#cbd5e1', linestyle='--')
        ax4.tick_params(colors='#1e293b', labelsize=10)
        
        for spine in ax4.spines.values():
            spine.set_edgecolor('#475569')
            spine.set_linewidth(1.5)
        
        plt.tight_layout()
        plt.savefig('attack_resilience_comparison.png', dpi=300, facecolor='white',
                   edgecolor='none', bbox_inches='tight')
        print("\n✓ Saved attack comparison plot: attack_resilience_comparison.png")
        plt.show()
        
        # Create summary bar chart
        self._plot_attack_summary_bars(attack_data)
    
    def _plot_attack_summary_bars(self, attack_data):
        """Create bar chart comparing final attack impacts"""
        fig, axes = plt.subplots(1, 2, figsize=(18, 7), facecolor='white')
        fig.suptitle('Attack Impact Summary: Efficiency Drop Comparison', 
                    fontsize=18, weight='bold', color='#1e293b', y=0.98)
        
        colors = ['#3b82f6', '#10b981', '#f59e0b']
        
        # Targeted attack final impact
        ax1 = axes[0]
        ax1.set_facecolor('white')
        
        networks = [data['network_name'] for data in attack_data]
        targeted_drops = [data['targeted_results'][-1]['efficiency_drop_%'] 
                         for data in attack_data]
        
        bars1 = ax1.bar(networks, targeted_drops, color=colors[:len(networks)],
                       edgecolor='#1e293b', linewidth=2, alpha=0.85, width=0.6)
        
        ax1.set_ylabel('Efficiency Drop (%) After 3 Node Removal', 
                      fontsize=12, color='#1e293b', weight='bold')
        ax1.set_title('Targeted Attack: Final Impact', fontsize=14,
                     color='#1e293b', weight='bold', pad=10)
        ax1.tick_params(colors='#1e293b', labelsize=11)
        ax1.grid(True, alpha=0.3, color='#cbd5e1', linestyle='--', axis='y')
        
        for spine in ax1.spines.values():
            spine.set_edgecolor('#475569')
            spine.set_linewidth(1.5)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%', ha='center', va='bottom',
                    color='#1e293b', fontsize=12, weight='bold')
        
        # Radius attack final impact
        ax2 = axes[1]
        ax2.set_facecolor('white')
        
        radius_drops = [data['radius_results'][-1]['efficiency_drop_%'] 
                       for data in attack_data]
        
        bars2 = ax2.bar(networks, radius_drops, color=colors[:len(networks)],
                       edgecolor='#1e293b', linewidth=2, alpha=0.85, width=0.6)
        
        ax2.set_ylabel('Efficiency Drop (%) at 3km Radius', 
                      fontsize=12, color='#1e293b', weight='bold')
        ax2.set_title('Radius Attack: Final Impact', fontsize=14,
                     color='#1e293b', weight='bold', pad=10)
        ax2.tick_params(colors='#1e293b', labelsize=11)
        ax2.grid(True, alpha=0.3, color='#cbd5e1', linestyle='--', axis='y')
        
        for spine in ax2.spines.values():
            spine.set_edgecolor('#475569')
            spine.set_linewidth(1.5)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%', ha='center', va='bottom',
                    color='#1e293b', fontsize=12, weight='bold')
        
        plt.tight_layout()
        plt.savefig('attack_impact_summary.png', dpi=300, facecolor='white',
                   edgecolor='none', bbox_inches='tight')
        print("✓ Saved attack summary plot: attack_impact_summary.png")
        plt.show()
        
        # Print numerical comparison
        print("\n" + "="*60)
        print("ATTACK RESILIENCE IMPROVEMENT SUMMARY")
        print("="*60)
        
        original_targeted = attack_data[0]['targeted_results'][-1]['efficiency_drop_%']
        original_radius = attack_data[0]['radius_results'][-1]['efficiency_drop_%']
        
        print(f"\nOriginal Network:")
        print(f"  Targeted Attack (3 nodes): {original_targeted:.2f}% efficiency drop")
        print(f"  Radius Attack (3km): {original_radius:.2f}% efficiency drop")
        
        for i in range(1, len(attack_data)):
            data = attack_data[i]
            targeted_drop = data['targeted_results'][-1]['efficiency_drop_%']
            radius_drop = data['radius_results'][-1]['efficiency_drop_%']
            
            targeted_improvement = original_targeted - targeted_drop
            radius_improvement = original_radius - radius_drop
            
            print(f"\n{data['network_name']}:")
            print(f"  Targeted Attack (3 nodes): {targeted_drop:.2f}% efficiency drop")
            print(f"    → Improvement: {targeted_improvement:+.2f} percentage points")
            print(f"    → Resilience gain: {(targeted_improvement/original_targeted*100):+.2f}%")
            print(f"  Radius Attack (3km): {radius_drop:.2f}% efficiency drop")
            print(f"    → Improvement: {radius_improvement:+.2f} percentage points")
            print(f"    → Resilience gain: {(radius_improvement/original_radius*100):+.2f}%")
    
    def _plot_comparison_charts(self, eff_orig, rob_orig, results):
        """Create aesthetic comparison charts including cost analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 14), facecolor='white')
        
        networks = ['Original'] + [r['network'] for r in results]
        efficiencies = [eff_orig] + [r['efficiency'] for r in results]
        robustness = [rob_orig] + [r['robustness'] for r in results]
        costs = [0] + [r['total_cost']/1e6 for r in results]
        
        colors = ['#3b82f6', '#10b981', '#f59e0b']
        
        # Efficiency comparison
        ax1 = axes[0, 0]
        ax1.set_facecolor('white')
        bars1 = ax1.bar(networks, efficiencies, color=colors, 
                       edgecolor='#1e293b', linewidth=2, alpha=0.85)
        ax1.set_ylabel('Efficiency', fontsize=14, color='#1e293b', weight='bold')
        ax1.set_title('Network Efficiency Comparison', fontsize=16, 
                     color='#1e293b', weight='bold', pad=15)
        ax1.tick_params(colors='#1e293b', labelsize=11)
        ax1.grid(True, alpha=0.3, color='#cbd5e1', linestyle='--', axis='y')
        
        for spine in ax1.spines.values():
            spine.set_edgecolor('#475569')
            spine.set_linewidth(1.5)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom',
                    color='#1e293b', fontsize=11, weight='bold')
        
        # Robustness comparison
        ax2 = axes[0, 1]
        ax2.set_facecolor('white')
        bars2 = ax2.bar(networks, robustness, color=colors,
                       edgecolor='#1e293b', linewidth=2, alpha=0.85)
        ax2.set_ylabel('Robustness', fontsize=14, color='#1e293b', weight='bold')
        ax2.set_title('Network Robustness Comparison', fontsize=16,
                     color='#1e293b', weight='bold', pad=15)
        ax2.tick_params(colors='#1e293b', labelsize=11)
        ax2.grid(True, alpha=0.3, color='#cbd5e1', linestyle='--', axis='y')
        
        for spine in ax2.spines.values():
            spine.set_edgecolor('#475569')
            spine.set_linewidth(1.5)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom',
                    color='#1e293b', fontsize=11, weight='bold')
        
        # Cost comparison
        ax3 = axes[1, 0]
        ax3.set_facecolor('white')
        bars3 = ax3.bar(networks, costs, color=colors,
                       edgecolor='#1e293b', linewidth=2, alpha=0.85)
        ax3.set_ylabel('Total Cost (Million $)', fontsize=14, color='#1e293b', weight='bold')
        ax3.set_title('Construction Cost Comparison', fontsize=16,
                     color='#1e293b', weight='bold', pad=15)
        ax3.tick_params(colors='#1e293b', labelsize=11)
        ax3.grid(True, alpha=0.3, color='#cbd5e1', linestyle='--', axis='y')
        
        for spine in ax3.spines.values():
            spine.set_edgecolor('#475569')
            spine.set_linewidth(1.5)
        
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.1f}M', ha='center', va='bottom',
                    color='#1e293b', fontsize=11, weight='bold')
        
        # Cost-Benefit Analysis
        ax4 = axes[1, 1]
        ax4.set_facecolor('white')
        
        cost_benefit = []
        cb_networks = []
        for i, result in enumerate(results):
            if result['cost_per_efficiency_gain'] != float('inf'):
                cost_benefit.append(result['cost_per_efficiency_gain']/1e6)
                cb_networks.append(result['network'])
        
        if cost_benefit:
            bars4 = ax4.bar(cb_networks, cost_benefit, 
                           color=[colors[i+1] for i in range(len(cb_networks))],
                           edgecolor='#1e293b', linewidth=2, alpha=0.85)
            ax4.set_ylabel('Cost per 1% Efficiency Gain (Million $)', 
                          fontsize=14, color='#1e293b', weight='bold')
            ax4.set_title('Cost-Benefit Analysis', fontsize=16,
                         color='#1e293b', weight='bold', pad=15)
            ax4.tick_params(colors='#1e293b', labelsize=11)
            ax4.grid(True, alpha=0.3, color='#cbd5e1', linestyle='--', axis='y')
            
            for spine in ax4.spines.values():
                spine.set_edgecolor('#475569')
                spine.set_linewidth(1.5)
            
            for bar in bars4:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'${height:.2f}M', ha='center', va='bottom',
                        color='#1e293b', fontsize=11, weight='bold')
        else:
            ax4.text(0.5, 0.5, 'No efficiency gains to analyze', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=14, color='#64748b')
            ax4.set_xticks([])
            ax4.set_yticks([])
        
        plt.tight_layout()
        plt.savefig('network_comparison.png', dpi=300, facecolor='white',
                   edgecolor='none', bbox_inches='tight')
        print("\n✓ Saved comparison plot: network_comparison.png")
        plt.show()
    
    def get_metrics(self):
        """Get all network metrics"""
        criticality = self.calculate_criticality()
        efficiency = self.calculate_efficiency()
        
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


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function - Plug and Play!"""
    
    print("="*80)
    print("METRO NETWORK RESILIENCE ANALYSIS SYSTEM")
    print("Plug-and-Play Version")
    print("="*80)
    
    # Get GeoJSON file from command line or use default
    if len(sys.argv) > 1:
        geojson_file = sys.argv[1]
    else:
        geojson_file = DEFAULT_GEOJSON
    
    print(f"\nConfiguration:")
    print(f"  GeoJSON File: {geojson_file}")
    print(f"  Network Name: {NETWORK_NAME}")
    print(f"  Cost per km: ${COST_PER_KM:,}")
    print(f"  Color-coded lines: {COLOR_CODE_LINES}")
    
    try:
        # Load and analyze network
        analyzer = MetroNetworkAnalyzer(geojson_file, COST_PER_KM, NETWORK_NAME)
        
        # Display line summary
        print("\n" + "="*80)
        print("LINE SUMMARY")
        print("="*80)
        line_summary = analyzer.get_line_summary()
        print(line_summary.to_string(index=False))
        
        # Get baseline metrics
        print("\n" + "="*80)
        print("BASELINE METRICS")
        print("="*80)
        
        metrics = analyzer.get_metrics()
        print(f"\nNetwork Statistics:")
        print(f"  Nodes: {metrics['nodes']}")
        print(f"  Edges: {metrics['edges']}")
        print(f"  Efficiency: {metrics['efficiency']:.6f}")
        print(f"  Robustness: {metrics['robustness']:.6f}")
        print(f"  Average Criticality: {metrics['avg_criticality']:.6f}")
        print(f"  Maximum Criticality: {metrics['max_criticality']:.6f}")
        
        # Show top critical nodes
        criticality = metrics['criticality']
        top_critical = sorted(criticality.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop 10 Critical Nodes:")
        for i, (node, score) in enumerate(top_critical, 1):
            print(f"{i:2d}. {analyzer.stations[node]['name']:30s} - {score:.6f}")
        
        # Plot original network
        print("\n" + "="*80)
        print("VISUALIZING ORIGINAL NETWORK")
        print("="*80)
        analyzer.plot_network(
            criticality, 
            title=f"{NETWORK_NAME} - Criticality Heatmap",
            filename="original_network.png",
            node_size=NODE_SIZE,
            flip_axes=FLIP_AXES,
            color_code_lines=COLOR_CODE_LINES
        )
        
        # Test resilience strategies with attack comparison
        print("\n" + "="*80)
        print("TESTING RESILIENCE STRATEGIES WITH ATTACK COMPARISON")
        print("="*80)
        
        results = analyzer.compare_networks(
            color_code_lines=COLOR_CODE_LINES, 
            run_attack_comparison=RUN_ATTACK_COMPARISON
        )
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("\nGenerated Files:")
        print("  📊 original_network.png - Original network visualization")
        print("  📊 network_resilience_k2.png - k=2 resilience strategy")
        print("  📊 network_resilience_k3.png - k=3 resilience strategy")
        print("  📊 network_comparison.png - Comparison charts")
        
        if RUN_ATTACK_COMPARISON:
            print("  📊 attack_resilience_comparison.png - Attack degradation comparison")
            print("  📊 attack_impact_summary.png - Attack impact summary bars")
        
        print("\n" + "="*80)
        print("SUCCESS! All analyses completed.")
        print("="*80)
        
        return analyzer
        
    except FileNotFoundError:
        print(f"\n❌ Error: File '{geojson_file}' not found!")
        print("Please ensure the GeoJSON file exists in the current directory.")
        print("\nUsage:")
        print(f"  python {sys.argv[0]} your_network.geojson")
        return None
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    analyzer = main()
    
    # Optional: Keep the analyzer object available for further analysis
    if analyzer:
        print("\n" + "="*80)
        print("INTERACTIVE MODE")
        print("="*80)
        print("\nThe 'analyzer' object is available for further analysis.")
        print("\nExample commands you can run:")
        print("  • analyzer.get_line_summary() - View line details")
        print("  • analyzer.get_metrics() - Get current metrics")
        print("  • analyzer.attack_critical_nodes(top_k=3) - Run targeted attack")
        print("  • analyzer.attack_radius(radius_km=2.0) - Run radius attack")
        print("\nYou can also modify the configuration at the top of the script:")
        print("  • DEFAULT_GEOJSON - Change input file")
        print("  • NETWORK_NAME - Change display name")
        print("  • COST_PER_KM - Adjust construction cost")
        print("  • COLOR_CODE_LINES - Toggle line colors")
        print("  • RUN_ATTACK_COMPARISON - Enable/disable attack simulations")