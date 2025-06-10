import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Get the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

import gymnasium as gym
import sumo_rl
from sumo_rl.models.util import *
from sumo_rl.agents.pg_multi_agent_dcrnn import PGMultiAgent
from sumo_rl.models.dcrnn_model import *
import torch
import torch.optim as optim

class TrafficSignalFeatureEngineer:
    """
    Feature engineering and analysis for traffic signal control systems.
    Extracts features, analyzes correlations, and identifies important parameters.
    """
    
    def __init__(self, project_root):
        self.project_root = project_root
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.features_data = []
        self.target_data = []
        self.feature_names = []
        
    def setup_environment(self):
        """Setup the SUMO environment for feature extraction."""
        net_file = os.path.join(self.project_root, 'sumo_rl', 'nets', 'RESCO', 'grid4x4', 'grid4x4.net.xml')
        route_file = os.path.join(self.project_root, 'sumo_rl', 'nets', 'RESCO', 'grid4x4', 'grid4x4_1.rou.xml')
        output_csv = os.path.join(self.project_root, 'results', 'feature_analysis')
        
        self.env = sumo_rl.parallel_env(
            net_file=net_file,
            route_file=route_file,
            out_csv_name=output_csv,
            use_gui=False,
            num_seconds=1000,  # Shorter for feature analysis
            begin_time=100,
            fixed_ts=True,
            reward_fn="weighted_wait_queue"
        )
        
        # Get traffic signal information
        self.traffic_signals = [ts for _, ts in self.env.aec_env.env.env.env.traffic_signals.items()]
        self.max_lanes = max(len(ts.lanes) for ts in self.traffic_signals)
        self.max_green_phases = max(ts.num_green_phases for ts in self.traffic_signals)
        self.ts_phases = [ts.num_green_phases for ts in self.traffic_signals]
        
        # Graph representation
        self.ts_indx, self.num_nodes, self.lanes_index, self.adj_list, _, _ = construct_graph_representation(self.traffic_signals)
        
        return self.env
    
    def extract_features_from_episode(self, episode_data):
        """
        Extract comprehensive features from a single episode.
        
        Args:
            episode_data: Dictionary containing observations, actions, rewards, and metrics
        """
        features = {}
        
        # System-level features
        if 'system_metrics' in episode_data:
            metrics = episode_data['system_metrics']
            features.update({
                'avg_system_waiting_time': np.mean(metrics.get('system_total_waiting_time', [0])),
                'max_system_waiting_time': np.max(metrics.get('system_total_waiting_time', [0])),
                'avg_system_stopped_vehicles': np.mean(metrics.get('system_total_stopped', [0])),
                'avg_system_speed': np.mean(metrics.get('system_mean_speed', [0])),
                'system_speed_variance': np.var(metrics.get('system_mean_speed', [0])),
            })
        
        # Traffic signal features
        if 'observations' in episode_data:
            obs_data = episode_data['observations']
            for ts_id, ts_obs_list in obs_data.items():
                # Extract density and queue features
                densities = [obs['density'] for obs in ts_obs_list if 'density' in obs]
                queues = [obs['queue'] for obs in ts_obs_list if 'queue' in obs]
                
                if densities and queues:
                    features.update({
                        f'{ts_id}_avg_density': np.mean([np.mean(d) for d in densities]),
                        f'{ts_id}_max_density': np.max([np.max(d) for d in densities]),
                        f'{ts_id}_density_variance': np.var([np.mean(d) for d in densities]),
                        f'{ts_id}_avg_queue': np.mean([np.mean(q) for q in queues]),
                        f'{ts_id}_max_queue': np.max([np.max(q) for q in queues]),
                        f'{ts_id}_queue_variance': np.var([np.mean(q) for q in queues]),
                        f'{ts_id}_num_lanes': len(densities[0]) if densities else 0,
                    })
        
        # Action features
        if 'actions' in episode_data:
            actions_data = episode_data['actions']
            for ts_id, actions_list in actions_data.items():
                if actions_list:
                    features.update({
                        f'{ts_id}_action_frequency': len(set(actions_list)) / len(actions_list) if actions_list else 0,
                        f'{ts_id}_action_changes': sum(1 for i in range(1, len(actions_list)) 
                                                     if actions_list[i] != actions_list[i-1]),
                        f'{ts_id}_most_frequent_action': max(set(actions_list), key=actions_list.count) if actions_list else 0,
                    })
        
        # Network topology features
        features.update({
            'num_traffic_signals': len(self.traffic_signals),
            'max_lanes_per_signal': self.max_lanes,
            'max_green_phases': self.max_green_phases,
            'total_edges': len(self.adj_list),
            'network_density': len(self.adj_list) / (self.num_nodes * (self.num_nodes - 1)) if self.num_nodes > 1 else 0,
        })
        
        # Performance target (total reward)
        target = sum(episode_data.get('rewards', {}).values()) if 'rewards' in episode_data else 0
        
        return features, target
    
    def collect_data_from_simulation(self, num_episodes=10):
        """
        Run simulation episodes and collect feature data.
        
        Args:
            num_episodes: Number of episodes to run for data collection
        """
        print("Collecting data from simulation...")
        
        for episode in range(num_episodes):
            print(f"Episode {episode + 1}/{num_episodes}")
            
            # Run episode
            obs, _ = self.env.reset()
            episode_data = {
                'observations': {agent: [] for agent in self.env.agents},
                'actions': {agent: [] for agent in self.env.agents},
                'rewards': {agent: 0 for agent in self.env.agents},
                'system_metrics': {
                    'system_total_waiting_time': [],
                    'system_total_stopped': [],
                    'system_mean_speed': [],
                }
            }
            
            done = False
            step_count = 0
            
            while not done and step_count < 200:  # Limit steps for feature analysis
                # Store observations
                for agent in self.env.agents:
                    if agent in obs:
                        episode_data['observations'][agent].append(obs[agent])
                
                # Take random actions for feature analysis
                actions = {agent: self.env.action_space(agent).sample() for agent in self.env.agents}
                
                # Store actions
                for agent, action in actions.items():
                    episode_data['actions'][agent].append(action)
                
                # Step environment
                obs, rewards, terminations, truncations, infos = self.env.step(actions)
                
                # Store rewards
                for agent, reward in rewards.items():
                    episode_data['rewards'][agent] += reward
                
                # Store system metrics
                if self.env.agents and len(infos) > 0:
                    first_agent = list(infos.keys())[0]
                    if 'system_total_waiting_time' in infos[first_agent]:
                        episode_data['system_metrics']['system_total_waiting_time'].append(
                            infos[first_agent]['system_total_waiting_time'])
                    if 'system_total_stopped' in infos[first_agent]:
                        episode_data['system_metrics']['system_total_stopped'].append(
                            infos[first_agent]['system_total_stopped'])
                    if 'system_mean_speed' in infos[first_agent]:
                        episode_data['system_metrics']['system_mean_speed'].append(
                            infos[first_agent]['system_mean_speed'])
                
                done = all(terminations.values()) or all(truncations.values())
                step_count += 1
            
            # Extract features from episode
            features, target = self.extract_features_from_episode(episode_data)
            self.features_data.append(features)
            self.target_data.append(target)
        
        # Create feature names list
        if self.features_data:
            self.feature_names = list(self.features_data[0].keys())
    
    def create_feature_dataframe(self):
        """Convert collected feature data to pandas DataFrame."""
        if not self.features_data:
            raise ValueError("No feature data collected. Run collect_data_from_simulation first.")
        
        # Convert to DataFrame
        df_features = pd.DataFrame(self.features_data)
        df_features['target_reward'] = self.target_data
        
        # Handle missing values
        df_features = df_features.fillna(0)
        
        return df_features
    
    def analyze_correlations(self, df_features, save_path=None):
        """
        Analyze correlations between features and create correlation matrix.
        
        Args:
            df_features: DataFrame with features
            save_path: Optional path to save the correlation plot
        """
        print("Analyzing feature correlations...")
        
        # Calculate correlation matrix
        correlation_matrix = df_features.corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(20, 16))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        # Draw the heatmap
        sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, center=0,
                   square=True, linewidths=.5, cbar_kws={"shrink": .5}, 
                   annot=False, fmt='.2f')
        
        plt.title('Feature Correlation Matrix', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = abs(correlation_matrix.iloc[i, j])
                if corr_val > 0.8:  # High correlation threshold
                    high_corr_pairs.append((
                        correlation_matrix.columns[i], 
                        correlation_matrix.columns[j], 
                        corr_val
                    ))
        
        print(f"\nHighly correlated feature pairs (|r| > 0.8):")
        for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: x[2], reverse=True):
            print(f"{feat1} <-> {feat2}: {corr:.3f}")
        
        return correlation_matrix
    
    def feature_importance_analysis(self, df_features, save_path=None):
        """
        Analyze feature importance using multiple methods.
        
        Args:
            df_features: DataFrame with features
            save_path: Optional path to save the importance plots
        """
        print("Analyzing feature importance...")
        
        # Prepare data
        X = df_features.drop('target_reward', axis=1)
        y = df_features['target_reward']
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
        
        # Method 1: Random Forest Feature Importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_importance = rf.feature_importances_
        
        # Method 2: Mutual Information
        mi_scores = mutual_info_regression(X_train, y_train, random_state=42)
        
        # Method 3: F-statistics
        f_scores, _ = f_regression(X_train, y_train)
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'random_forest': rf_importance,
            'mutual_info': mi_scores,
            'f_statistic': f_scores
        })
        
        # Normalize scores for comparison
        for col in ['random_forest', 'mutual_info', 'f_statistic']:
            importance_df[f'{col}_normalized'] = importance_df[col] / importance_df[col].max()
        
        # Calculate average importance
        importance_df['avg_importance'] = importance_df[['random_forest_normalized', 
                                                        'mutual_info_normalized', 
                                                        'f_statistic_normalized']].mean(axis=1)
        
        # Sort by average importance
        importance_df = importance_df.sort_values('avg_importance', ascending=False)
        
        # Plot feature importance
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # Random Forest Importance
        top_rf = importance_df.nlargest(15, 'random_forest')
        axes[0, 0].barh(range(len(top_rf)), top_rf['random_forest'])
        axes[0, 0].set_yticks(range(len(top_rf)))
        axes[0, 0].set_yticklabels(top_rf['feature'], fontsize=10)
        axes[0, 0].set_title('Random Forest Feature Importance')
        axes[0, 0].set_xlabel('Importance Score')
        
        # Mutual Information
        top_mi = importance_df.nlargest(15, 'mutual_info')
        axes[0, 1].barh(range(len(top_mi)), top_mi['mutual_info'])
        axes[0, 1].set_yticks(range(len(top_mi)))
        axes[0, 1].set_yticklabels(top_mi['feature'], fontsize=10)
        axes[0, 1].set_title('Mutual Information Scores')
        axes[0, 1].set_xlabel('MI Score')
        
        # F-statistic
        top_f = importance_df.nlargest(15, 'f_statistic')
        axes[1, 0].barh(range(len(top_f)), top_f['f_statistic'])
        axes[1, 0].set_yticks(range(len(top_f)))
        axes[1, 0].set_yticklabels(top_f['feature'], fontsize=10)
        axes[1, 0].set_title('F-statistic Scores')
        axes[1, 0].set_xlabel('F Score')
        
        # Average Importance
        top_avg = importance_df.nlargest(15, 'avg_importance')
        axes[1, 1].barh(range(len(top_avg)), top_avg['avg_importance'])
        axes[1, 1].set_yticks(range(len(top_avg)))
        axes[1, 1].set_yticklabels(top_avg['feature'], fontsize=10)
        axes[1, 1].set_title('Average Normalized Importance')
        axes[1, 1].set_xlabel('Average Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print top features
        print(f"\nTop 10 Most Important Features:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            print(f"{i+1:2d}. {row['feature']:40s} (avg: {row['avg_importance']:.3f})")
        
        return importance_df
    
    def analyze_feature_distributions(self, df_features, save_path=None):
        """
        Analyze feature distributions and relationships with target.
        
        Args:
            df_features: DataFrame with features
            save_path: Optional path to save the distribution plots
        """
        print("Analyzing feature distributions...")
        
        # Select numerical features for distribution analysis
        numerical_features = df_features.select_dtypes(include=[np.number]).columns.tolist()
        if 'target_reward' in numerical_features:
            numerical_features.remove('target_reward')
        
        # Plot distributions for top features
        n_features = min(12, len(numerical_features))
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.ravel()
        
        for i, feature in enumerate(numerical_features[:n_features]):
            # Histogram
            axes[i].hist(df_features[feature], bins=20, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{feature}')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Feature Distributions', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_feature_summary_report(self, df_features, correlation_matrix, importance_df, save_path=None):
        """
        Generate a comprehensive feature analysis report.
        
        Args:
            df_features: DataFrame with features
            correlation_matrix: Correlation matrix
            importance_df: Feature importance DataFrame
            save_path: Optional path to save the report
        """
        print("Generating feature summary report...")
        
        # Basic statistics
        feature_stats = df_features.describe()
        
        # Missing values
        missing_values = df_features.isnull().sum()
        missing_pct = (missing_values / len(df_features)) * 100
        
        # Target correlations
        target_corr = correlation_matrix['target_reward'].abs().sort_values(ascending=False)
        
        # Create report
        report = []
        report.append("=" * 80)
        report.append("TRAFFIC SIGNAL FEATURE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Dataset Shape: {df_features.shape}")
        report.append(f"Number of Episodes: {len(df_features)}")
        report.append(f"Number of Features: {len(df_features.columns) - 1}")
        report.append("")
        
        # Top correlated features with target
        report.append("TOP 10 FEATURES CORRELATED WITH TARGET REWARD:")
        report.append("-" * 50)
        for i, (feature, corr) in enumerate(target_corr.head(11).items()):
            if feature != 'target_reward':
                report.append(f"{i:2d}. {feature:40s} {corr:6.3f}")
        report.append("")
        
        # Top important features
        report.append("TOP 10 MOST IMPORTANT FEATURES:")
        report.append("-" * 50)
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            report.append(f"{i+1:2d}. {row['feature']:40s} {row['avg_importance']:6.3f}")
        report.append("")
        
        # Missing values
        if missing_values.sum() > 0:
            report.append("FEATURES WITH MISSING VALUES:")
            report.append("-" * 50)
            for feature, count in missing_values[missing_values > 0].items():
                report.append(f"{feature:40s} {count:6d} ({missing_pct[feature]:5.1f}%)")
        else:
            report.append("NO MISSING VALUES FOUND")
        report.append("")
        
        # Feature categories
        system_features = [f for f in df_features.columns if f.startswith('avg_system') or f.startswith('system')]
        ts_features = [f for f in df_features.columns if any(ts_id in f for ts_id in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])]
        network_features = [f for f in df_features.columns if f.startswith('num_') or f.startswith('max_') or f.startswith('network')]
        
        report.append("FEATURE CATEGORIES:")
        report.append("-" * 50)
        report.append(f"System-level features:     {len(system_features):3d}")
        report.append(f"Traffic signal features:   {len(ts_features):3d}")
        report.append(f"Network topology features: {len(network_features):3d}")
        report.append("")
        
        # Summary statistics for top features
        top_features = importance_df.head(5)['feature'].tolist()
        report.append("SUMMARY STATISTICS FOR TOP 5 FEATURES:")
        report.append("-" * 50)
        for feature in top_features:
            stats = feature_stats[feature]
            report.append(f"{feature}:")
            report.append(f"  Mean: {stats['mean']:8.3f}  Std: {stats['std']:8.3f}")
            report.append(f"  Min:  {stats['min']:8.3f}  Max: {stats['max']:8.3f}")
            report.append("")
        
        # Join report and print
        report_text = "\n".join(report)
        print(report_text)
        
        # Save report if path provided
        if save_path:
            with open(os.path.join(save_path, 'feature_analysis_report.txt'), 'w') as f:
                f.write(report_text)
        
        return report_text

def main():
    """Main function to run the complete feature engineering analysis."""
    # Initialize feature engineer
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    fe = TrafficSignalFeatureEngineer(PROJECT_ROOT)
    
    # Create results directory
    results_dir = os.path.join(PROJECT_ROOT, 'results', 'feature_analysis')
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        print("Setting up environment...")
        env = fe.setup_environment()
        
        print("Collecting feature data...")
        fe.collect_data_from_simulation(num_episodes=20)  # Adjust number of episodes as needed
        
        print("Creating feature DataFrame...")
        df_features = fe.create_feature_dataframe()
        
        print("Performing correlation analysis...")
        correlation_matrix = fe.analyze_correlations(df_features, results_dir)
        
        print("Performing feature importance analysis...")
        importance_df = fe.feature_importance_analysis(df_features, results_dir)
        
        print("Analyzing feature distributions...")
        fe.analyze_feature_distributions(df_features, results_dir)
        
        print("Generating summary report...")
        fe.generate_feature_summary_report(df_features, correlation_matrix, importance_df, results_dir)
        
        # Save feature data
        df_features.to_csv(os.path.join(results_dir, 'feature_data.csv'), index=False)
        importance_df.to_csv(os.path.join(results_dir, 'feature_importance.csv'), index=False)
        
        print(f"\nAnalysis complete! Results saved to: {results_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        if hasattr(fe, 'env'):
            fe.env.close()

if __name__ == "__main__":
    main()