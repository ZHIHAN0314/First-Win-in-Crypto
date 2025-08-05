# This is the full, modified content of crypto_strategy_engine.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch  # <-- 新增的导入
import seaborn as sns
import os
from typing import Dict, List, Tuple, Optional
import warnings
from itertools import cycle
warnings.filterwarnings('ignore')

class CryptoStrategyEngine:
    """
    Crypto Strategy Engine - Enhanced with Custom Risk, Sentiment, and Drawdown Analysis
    """
    
    def __init__(self, data_path: str, sentiment_data_path: str):
        """
        Initialize strategy engine
        
        Args:
            data_path: CSV data file path
            sentiment_data_path: CSV sentiment data file path
        """
        self.data_path = data_path
        self.sentiment_data_path = sentiment_data_path

        self.data = None
        self.factor_scores = None
        self.portfolio_weights = None
        self.multi_factor_returns = None
        self.strategies = {}
        self.backtest_results = {}
        self.performance_metrics = {}
        
        # NEW: Weighting scheme configurations
        self.weighting_schemes = {
            'equal': 'Equal Weight',
            'factor_weighted': 'Factor Value Weighted', 
            'inverse_volatility': 'Inverse Volatility Weighted',
            'risk_parity': 'Risk Parity',
            'optimized': 'Mean-Variance Optimized'
        }

        # NEW: Storage for risk-based strategies
        self.risk_strategies = {}
        self.risk_backtest_results = {}
        self.risk_performance_metrics = {}
        
        # NEW: Store BTC cumulative returns once it's calculated correctly
        self.btc_cumulative_returns = None
        
        # Create result folder structure
        self.setup_result_folders()
        
        # Set plotting style
        plt.style.use('dark_background')
        sns.set_palette("husl")

    def setup_result_folders(self):
        """Create result folder structure in current working directory"""
        current_dir = os.getcwd()
        
        self.base_folder = os.path.join(current_dir, "Project_result")
        self.csv_folder = os.path.join(self.base_folder, "csv_data")
        self.charts_folder = os.path.join(self.base_folder, "charts")
        # NEW: Weekly allocations folder
        self.weekly_allocations_folder = os.path.join(self.base_folder, "weekly_allocations")
        
        for folder in [self.base_folder, self.csv_folder, self.charts_folder, self.weekly_allocations_folder]:
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"Created folder: {folder}")
            else:
                print(f"Folder exists: {folder}")
        
        print(f"\nFolder structure created at: {current_dir}")
        print(f"  - Main folder: {self.base_folder}")
        print(f"  - CSV data: {self.csv_folder}")
        print(f"  - Charts: {self.charts_folder}")
        print(f"  - Weekly allocations: {self.weekly_allocations_folder}")
        
    def load_data(self) -> pd.DataFrame:
        """Load data and create future return column for backtesting."""
        print("Loading data...")
        self.data = pd.read_csv(self.data_path)
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        # VERY IMPORTANT: Sort by symbol and then date to ensure correct shifting
        self.data = self.data.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Create the 'future_return' column for backtesting
        # This calculates the return for the *next* period for each symbol
        self.data['future_return'] = self.data.groupby('symbol')['return'].shift(-1)
        
        # Drop rows where future_return is NaN (typically the last entry for each symbol)
        self.data.dropna(subset=['future_return'], inplace=True)

        print(f"Data loaded successfully, {len(self.data)} rows, {len(self.data.columns)} columns")
        print(f"Time range: {self.data['date'].min()} to {self.data['date'].max()}")
        print(f"Number of currencies: {self.data['symbol'].nunique()}")
        
        return self.data
    
    def identify_factor_columns(self) -> Dict[str, List[str]]:
        """Identify four core factor columns"""
        factor_groups = {
            'momentum': [],      # Momentum factors
            'volatility': [],    # Volatility factors
            'volume_impact': [], # Volume impact factors (deduplicated)
            'reversal': []       # Short-term reversal factors
        }
        
        # Check available columns in data
        available_columns = set(self.data.columns)
        
        # 1. Momentum factors - momentum_xx format 
        momentum_factors = ['momentum_14', 'momentum_21', 'momentum_28', 'momentum_42', 'momentum_90']
        for factor in momentum_factors:
            if factor in available_columns:
                factor_groups['momentum'].append(factor)
        
        # 2. Volatility factors - volatility_xx format
        volatility_factors = ['volatility_14', 'volatility_21', 'volatility_28', 'volatility_42', 'volatility_90']
        for factor in volatility_factors:
            if factor in available_columns:
                factor_groups['volatility'].append(factor)
        
        # 3. Volume impact factors (deduplicated: prioritize usd_volume, fallback to btc_volume)
        if 'usd_volume' in available_columns:
            factor_groups['volume_impact'].append('usd_volume')
        elif 'btc_volume' in available_columns:
            factor_groups['volume_impact'].append('btc_volume')
        
        # 4. Short-term reversal factors
        reversal_factors = ['strev_weekly']
        for factor in reversal_factors:
            if factor in available_columns:
                factor_groups['reversal'].append(factor)
        
        # Remove empty groups
        factor_groups = {k: v for k, v in factor_groups.items() if v}
        
        return factor_groups
    
    def calculate_asset_weights(self, assets_data: pd.DataFrame, weighting_scheme: str, 
                          factor_values: Optional[pd.Series] = None) -> pd.Series:
        """Calculate asset weights based on different weighting schemes"""
        n_assets = len(assets_data)
        if n_assets == 0:
            return pd.Series(dtype=float)

        if weighting_scheme == 'equal':
            return pd.Series(1.0 / n_assets, index=assets_data.index)
        
        elif weighting_scheme == 'factor_weighted' and factor_values is not None:
            # Weight by factor values (normalized)
            abs_factor_values = factor_values.abs()
            # Handle cases where all factor values are zero
            if abs_factor_values.sum() > 1e-9:
                weights = abs_factor_values / abs_factor_values.sum()
            else:
                weights = pd.Series(1.0 / n_assets, index=assets_data.index)
            return weights
        
        elif weighting_scheme == 'inverse_volatility':
            # Weight by inverse volatility (if volatility data available)
            volatility_cols = [col for col in assets_data.columns if 'volatility' in col.lower()]
            if volatility_cols and volatility_cols[0] in assets_data.columns:
                # Use the first available volatility column
                vol_col = volatility_cols[0]
                vol_data = assets_data[vol_col].fillna(assets_data[vol_col].mean())
                # Avoid division by zero
                vol_data[vol_data < 1e-9] = assets_data[vol_col].mean()
                if vol_data.isnull().all() or (vol_data < 1e-9).all():
                     return pd.Series(1.0 / n_assets, index=assets_data.index)
                inv_vol = 1.0 / vol_data
                weights = inv_vol / inv_vol.sum()
            else:
                weights = pd.Series(1.0 / n_assets, index=assets_data.index)
            return weights
        
        elif weighting_scheme == 'risk_parity':
            # Simple risk parity approximation using volatility
            volatility_cols = [col for col in assets_data.columns if 'volatility' in col.lower()]
            if volatility_cols and volatility_cols[0] in assets_data.columns:
                vol_col = volatility_cols[0]
                vol_data = assets_data[vol_col].fillna(assets_data[vol_col].mean())
                vol_data[vol_data < 1e-9] = assets_data[vol_col].mean()
                if vol_data.isnull().all() or (vol_data < 1e-9).all():
                     return pd.Series(1.0 / n_assets, index=assets_data.index)
                # Risk parity: weight inversely proportional to volatility
                risk_contrib = 1.0 / vol_data
                weights = risk_contrib / risk_contrib.sum()
            else:
                weights = pd.Series(1.0 / n_assets, index=assets_data.index)
            return weights
        
        elif weighting_scheme == 'optimized':
            # Simple mean-variance optimization approximation
            return_cols = [col for col in assets_data.columns if 'return' in col.lower()]
            volatility_cols = [col for col in assets_data.columns if 'volatility' in col.lower()]
            
            if return_cols and volatility_cols:
                returns = assets_data[return_cols[0]].fillna(0)
                volatilities = assets_data[volatility_cols[0]].fillna(assets_data[volatility_cols[0]].mean())
                volatilities[volatilities < 1e-9] = volatilities.mean()
                
                # Simple optimization: maximize return/risk ratio
                sharpe_proxy = returns / volatilities
                sharpe_proxy = sharpe_proxy.fillna(0)
                
                if sharpe_proxy.sum() > 1e-9:
                    weights = np.maximum(sharpe_proxy, 0)
                    if weights.sum() > 1e-9:
                        weights = weights / weights.sum()
                    else:
                        weights = pd.Series(1.0 / n_assets, index=assets_data.index)
                else:
                    weights = pd.Series(1.0 / n_assets, index=assets_data.index)
            else:
                weights = pd.Series(1.0 / n_assets, index=assets_data.index)
            return weights
        
        else:
            # Default to equal weighting
            return pd.Series(1.0 / n_assets, index=assets_data.index)

    def optimize_factor_weights(self, factor_groups: Dict, historical_data: pd.DataFrame, 
                          lookback_periods: int = 52) -> Dict[str, float]:
        """Dynamically optimize factor weights based on recent performance"""
        if len(historical_data) < lookback_periods:
            # Not enough data, return equal weights
            n_factors = len(factor_groups)
            return {factor: 1.0/n_factors for factor in factor_groups.keys()}
        
        # Get recent data
        recent_data = historical_data.tail(lookback_periods)
        factor_performance = {}
        
        for factor_type, factors in factor_groups.items():
            if not factors:
                continue
                
            # Calculate simple factor performance
            factor_returns = []
            for factor in factors:
                if factor in recent_data.columns:
                    factor_data = recent_data[factor].fillna(0)
                    # Simple performance metric: correlation with future returns
                    if 'return' in recent_data.columns:
                        correlation = factor_data.corr(recent_data['return'].shift(-1))
                        if not np.isnan(correlation):
                            factor_returns.append(abs(correlation))
            
            if factor_returns:
                factor_performance[factor_type] = np.mean(factor_returns)
            else:
                factor_performance[factor_type] = 0.25  # Default weight
        
        # Normalize weights
        total_performance = sum(factor_performance.values())
        if total_performance > 0:
            optimized_weights = {k: v/total_performance for k, v in factor_performance.items()}
        else:
            n_factors = len(factor_performance)
            optimized_weights = {k: 1.0/n_factors for k in factor_performance.keys()}
        
        return optimized_weights
    
    # ==================== NEW: Custom Risk-Based Strategy Functions ====================
    
    def create_custom_risk_strategy(self, risk_tolerance: int, market_sentiment: str = 'neutral') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create custom strategy based on risk tolerance (0-10 scale).
        The asset weighting scheme and factor weights are now dynamically adjusted based on risk tolerance AND market sentiment.
        """
        if not 0 <= risk_tolerance <= 10:
            raise ValueError("Risk tolerance must be between 0 and 10")
        
        if market_sentiment not in ['positive', 'neutral', 'negative']:
            raise ValueError("Market sentiment must be 'positive', 'neutral', or 'negative'")

        print(f"\nCreating Custom Multi-Factor Strategy (Risk: {risk_tolerance}/10, Sentiment: {market_sentiment})...")
        
        # Base factor weights based on risk tolerance
        momentum_base_weight = 0.2 + (risk_tolerance / 10) * 0.4  # 0.2 to 0.6
        volatility_base_weight = 0.4 - (risk_tolerance / 10) * 0.25  # 0.4 to 0.15
        volume_base_weight = 0.15 + (risk_tolerance / 10) * 0.25  # 0.15 to 0.4
        reversal_base_weight = 0.25 - (risk_tolerance / 10) * 0.1  # 0.25 to 0.15

        # Adjust weights based on market sentiment
        if market_sentiment == 'positive':
            # In a positive market, prioritize momentum and volume
            momentum_base_weight += 0.15
            volume_base_weight += 0.10
            volatility_base_weight -= 0.15
            reversal_base_weight -= 0.10
        elif market_sentiment == 'negative':
            # In a negative market, prioritize volatility and reversal
            volatility_base_weight += 0.15
            reversal_base_weight += 0.10
            momentum_base_weight -= 0.15
            volume_base_weight -= 0.10
        
        # Ensure all weights are non-negative
        momentum_base_weight = max(0, momentum_base_weight)
        volatility_base_weight = max(0, volatility_base_weight)
        volume_base_weight = max(0, volume_base_weight)
        reversal_base_weight = max(0, reversal_base_weight)

        # Normalize weights
        total_weight = momentum_base_weight + volatility_base_weight + volume_base_weight + reversal_base_weight
        if total_weight == 0:
            total_weight = 1 # Avoid division by zero
        factor_weights = {
            'momentum': momentum_base_weight / total_weight,
            'volatility': volatility_base_weight / total_weight,
            'volume_impact': volume_base_weight / total_weight,
            'reversal': reversal_base_weight / total_weight
        }
        
        # Adjust portfolio size based on risk tolerance
        top_n_assets = max(3, min(10, 8 - risk_tolerance // 2))
        
        # Determine weighting scheme based on risk tolerance
        if risk_tolerance <= 3:
            weighting_scheme = 'inverse_volatility'
        elif risk_tolerance <= 6:
            weighting_scheme = 'risk_parity'
        else:
            weighting_scheme = 'factor_weighted'

        print(f"Dynamic factor weights: {factor_weights}")
        print(f"Portfolio size: {top_n_assets} assets")
        print(f"Weighting scheme based on risk: {weighting_scheme}")
        
        factor_groups = self.identify_factor_columns()
        strategy_data = []
        
        for date, group in self.data.groupby('date'):
            valid_data = group.dropna()
            if len(valid_data) < top_n_assets:
                continue
            
            composite_scores = pd.Series(0.0, index=valid_data.index)
            
            for factor_type, weight in factor_weights.items():
                if factor_type in factor_groups:
                    factors = factor_groups[factor_type]
                    
                    for factor in factors:
                        if factor in valid_data.columns:
                            factor_data = valid_data[factor].fillna(0)
                        
                            if factor_data.std() > 0:
                                normalized_scores = (factor_data - factor_data.mean()) / factor_data.std()
                            
                                if 'volatility' in factor.lower():
                                    normalized_scores = -normalized_scores
                                elif factor == 'strev_weekly':
                                    normalized_scores = -normalized_scores
                                
                                if len(factors) > 0:
                                    composite_scores += weight * normalized_scores / len(factors)
        
            if len(composite_scores) > 0:
                top_assets = valid_data.loc[composite_scores.nlargest(top_n_assets).index]

                if top_assets.empty:
                    continue

                asset_weights = self.calculate_asset_weights(
                    top_assets,
                    weighting_scheme,
                    composite_scores.loc[top_assets.index]
                )
                
                for idx, (_, asset) in enumerate(top_assets.iterrows()):
                    strategy_data.append({
                        'date': date,
                        'symbol': asset['symbol'],
                        'weight': asset_weights.iloc[idx],
                        'composite_score': composite_scores[asset.name],
                        'return': asset.get('future_return', 0),
                        'risk_tolerance': risk_tolerance,
                        'market_sentiment': market_sentiment,
                        'weighting_scheme': weighting_scheme
                    })
    
        strategy_df = pd.DataFrame(strategy_data)
        returns = self.calculate_portfolio_returns(strategy_df)
        
        print(f"Debug: Strategy generated {len(strategy_df)} holdings across {strategy_df['date'].nunique()} dates")
        print(f"Debug: Returns generated {len(returns)} periods")
        
        return strategy_df, returns

    def generate_weekly_allocation_csv(self, strategy_df: pd.DataFrame, strategy_name: str):
        """Generate weekly allocation CSV file"""
        if strategy_df.empty:
            print(f"Warning: Cannot generate weekly allocation for '{strategy_name}' as strategy data is empty.")
            return

        filename = f"{strategy_name.lower().replace(' ', '_').replace('/', '_')}_weekly_allocations.csv"
        filepath = os.path.join(self.weekly_allocations_folder, filename)
        
        weekly_allocations = []
        
        for date, group in strategy_df.groupby('date'):
            week_data = {
                'date': date.strftime('%Y-%m-%d'),
                'strategy': strategy_name,
                'total_assets': len(group)
            }
            
            for i, (_, asset) in enumerate(group.iterrows(), 1):
                week_data[f'asset_{i}_symbol'] = asset['symbol']
                week_data[f'asset_{i}_weight'] = asset['weight']
                week_data[f'asset_{i}_score'] = asset.get('composite_score', 0)
            
            weekly_allocations.append(week_data)
        
        allocation_df = pd.DataFrame(weekly_allocations)
        allocation_df.to_csv(filepath, index=False)
        
        print(f"Weekly allocation CSV saved: {filename}")
        return filepath

    # This function is now removed from the main execution and replaced by the new function below
    def create_risk_strategy_comparison_chart(self, strategies_data: Dict):
        """Create comprehensive risk strategy comparison chart with drawdown chart"""
        if not strategies_data:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='black')
        fig.suptitle('Custom Risk-Based Strategy Performance Comparison', fontsize=20, color='white', y=0.95)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98FB98', '#FFDAB9', '#ADD8E6']
        
        # Plot 1: Cumulative Returns
        ax1 = axes[0, 0]
        for i, (name, data) in enumerate(strategies_data.items()):
            returns = data['returns']
            if not returns.empty:
                cumulative = (1 + returns).cumprod()
                ax1.plot(cumulative.index, cumulative.values, 
                        label=name, linewidth=2, color=colors[i % len(colors)])
        
        # NEW: Use the stored, correct BTC cumulative returns if available
        if self.btc_cumulative_returns is not None and not self.btc_cumulative_returns.empty:
            btc_cumulative = self.btc_cumulative_returns
            ax1.plot(btc_cumulative.index, btc_cumulative.values, 
                   label='BTC Benchmark', linewidth=2, color='orange', linestyle='--')
        
        ax1.set_title('Cumulative Returns', fontsize=14, color='white')
        ax1.set_ylabel('Cumulative Return', color='white')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(colors='white')
        
        # Plot 2: Performance Metrics Bar Chart
        ax2 = axes[0, 1]
        strategy_names = list(strategies_data.keys())
        annual_returns = [data['metrics'].get('Annual Return', 0) for data in strategies_data.values()]
        sharpe_ratios = [data['metrics'].get('Sharpe Ratio', 0) for data in strategies_data.values()]
        
        x = np.arange(len(strategy_names))
        width = 0.35
        
        ax2.bar(x - width/2, annual_returns, width, label='Annual Return', alpha=0.8, color='skyblue')
        ax2_twin = ax2.twinx()
        ax2_twin.bar(x + width/2, sharpe_ratios, width, label='Sharpe Ratio', alpha=0.8, color='lightcoral')
        
        ax2.set_title('Performance Metrics', fontsize=14, color='white')
        ax2.set_xticks(x)
        ax2.set_xticklabels(strategy_names, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Annual Return', color='white')
        ax2_twin.set_ylabel('Sharpe Ratio', color='white')
        ax2.tick_params(colors='white')
        ax2_twin.tick_params(colors='white')
        
        # Plot 3: Rolling Sharpe Ratio
        ax3 = axes[1, 0]
        for i, (name, data) in enumerate(strategies_data.items()):
            returns = data['returns']
            if len(returns) >= 26:
                rolling_sharpe = returns.rolling(26).mean() / returns.rolling(26).std() * np.sqrt(52)
                ax3.plot(rolling_sharpe.index, rolling_sharpe.values, 
                        label=name, linewidth=2, color=colors[i % len(colors)])
        
        ax3.set_title('Rolling Sharpe Ratio (26W)', fontsize=14, color='white')
        ax3.set_ylabel('Sharpe Ratio', color='white')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(colors='white')
        
        # Plot 4: Maximum Drawdown
        ax4 = axes[1, 1]
        # NEW: Use the stored BTC cumulative returns for drawdown if available
        for i, (name, data) in enumerate(strategies_data.items()):
            returns = data['returns']
            if not returns.empty:
                cumulative = (1 + returns).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                ax4.plot(drawdown.index, drawdown.values, 
                         label=name, linewidth=2, color=colors[i % len(colors)])

        # NEW: Also plot BTC drawdown if data is available
        if self.btc_cumulative_returns is not None and not self.btc_cumulative_returns.empty:
            btc_cumulative = self.btc_cumulative_returns
            btc_rolling_max = btc_cumulative.expanding().max()
            btc_drawdown = (btc_cumulative - btc_rolling_max) / btc_rolling_max
            ax4.plot(btc_drawdown.index, btc_drawdown.values, 
                     label='BTC Benchmark', linewidth=2, color='orange', linestyle='--')
        
        ax4.set_title('Drawdown Analysis', fontsize=14, color='white')
        ax4.set_xlabel('Date', color='white')
        ax4.set_ylabel('Drawdown', color='white')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(colors='white')

        plt.tight_layout()
        chart_path = os.path.join(self.charts_folder, "risk_strategy_comparison.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='black')
        
        print(f"Risk strategy comparison chart saved: {chart_path}")

    def create_comparison_for_risk_level(self, risk_level: int, strategies_data: Dict):
        """
        Create a comparison chart for strategies related to a specific risk level,
        excluding the non-sentiment-based custom risk strategy.
        """
        
        if not strategies_data:
            return

        # Filter strategies for the specific risk level, excluding the neutral Custom_Risk_x/10 one
        filtered_strategies = {
            name: data for name, data in strategies_data.items()
            if f'Custom_Risk_{risk_level}' in name and 'neutral' in name
        }

        # Also get the other sentiment-based strategies for this risk level
        for name, data in strategies_data.items():
            if f'Custom_Risk_{risk_level}_positive' in name or f'Custom_Risk_{risk_level}_negative' in name:
                filtered_strategies[name] = data

        if not filtered_strategies:
            print(f"No sentiment-based strategies found for risk level {risk_level}.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='black')
        fig.suptitle(f'Risk Level {risk_level}/10 Strategy Performance Comparison (Sentiment-Based)', fontsize=20, color='white', y=0.95)

        # Define the new color palette from the user request
        from itertools import cycle
        custom_colors = ['yellow', '#FF7F50', '#6495ED'] # Yellow, Coral Red, Cornflower Blue
        color_cycle = cycle(custom_colors)

        # Define a color map for strategies
        color_map = {name: next(color_cycle) for name in filtered_strategies.keys()}

        # Plot 1: Cumulative Returns
        ax1 = axes[0, 0]
        for name, data in filtered_strategies.items():
            returns = data['returns']
            if not returns.empty:
                cumulative = (1 + returns).cumprod()
                ax1.plot(cumulative.index, cumulative.values,
                         label=name, linewidth=2, color=color_map[name])
        
        if self.btc_cumulative_returns is not None and not self.btc_cumulative_returns.empty:
            btc_cumulative = self.btc_cumulative_returns
            ax1.plot(btc_cumulative.index, btc_cumulative.values,
                   label='BTC Benchmark', linewidth=2, color='orange', linestyle='--')
        
        ax1.set_title('Cumulative Returns', fontsize=14, color='white')
        ax1.set_ylabel('Cumulative Return (Log Scale)', color='white')
        ax1.set_yscale('log')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(colors='white')

        # Plot 2: Performance Metrics Bar Chart
        ax2 = axes[0, 1]
        strategy_names = list(filtered_strategies.keys())
        annual_returns = [data['metrics'].get('Annual Return', 0) for data in filtered_strategies.values()]
        sharpe_ratios = [data['metrics'].get('Sharpe Ratio', 0) for data in filtered_strategies.values()]
        
        # Reset color cycle for the bar chart
        color_cycle = cycle(custom_colors)
        bar_colors = [next(color_cycle) for _ in strategy_names]

        x = np.arange(len(strategy_names))
        width = 0.35

        ax2.bar(x - width / 2, annual_returns, width, label='Annual Return', alpha=0.8, color=bar_colors)
        ax2_twin = ax2.twinx()
        ax2_twin.bar(x + width / 2, sharpe_ratios, width, label='Sharpe Ratio', alpha=0.8, color=bar_colors)

        ax2.set_title('Performance Metrics', fontsize=14, color='white')
        ax2.set_xticks(x)
        ax2.set_xticklabels(strategy_names, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Annual Return', color='white')
        ax2_twin.set_ylabel('Sharpe Ratio', color='white')
        ax2.tick_params(colors='white')
        ax2_twin.tick_params(colors='white')

        # Plot 3: Rolling Sharpe Ratio
        ax3 = axes[1, 0]
        # Reset color cycle for the new plot
        color_cycle = cycle(custom_colors)
        for name, data in filtered_strategies.items():
            returns = data['returns']
            if len(returns) >= 26:
                rolling_sharpe = returns.rolling(26).mean() / returns.rolling(26).std() * np.sqrt(52)
                ax3.plot(rolling_sharpe.index, rolling_sharpe.values,
                         label=name, linewidth=2, color=next(color_cycle))

        ax3.set_title('Rolling Sharpe Ratio (26W)', fontsize=14, color='white')
        ax3.set_ylabel('Sharpe Ratio', color='white')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(colors='white')

        # Plot 4: Maximum Drawdown
        ax4 = axes[1, 1]
        # Reset color cycle for the new plot
        color_cycle = cycle(custom_colors)
        for name, data in filtered_strategies.items():
            returns = data['returns']
            if not returns.empty:
                cumulative = (1 + returns).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                ax4.plot(drawdown.index, drawdown.values,
                         label=name, linewidth=2, color=next(color_cycle))

        # NEW: Also plot BTC drawdown if data is available
        if self.btc_cumulative_returns is not None and not self.btc_cumulative_returns.empty:
            btc_cumulative = self.btc_cumulative_returns
            btc_rolling_max = btc_cumulative.expanding().max()
            btc_drawdown = (btc_cumulative - btc_rolling_max) / btc_rolling_max
            ax4.plot(btc_drawdown.index, btc_drawdown.values,
                     label='BTC Benchmark', linewidth=2, color='orange', linestyle='--')
        
        ax4.set_title('Drawdown Analysis', fontsize=14, color='white')
        ax4.set_xlabel('Date', color='white')
        ax4.set_ylabel('Drawdown', color='white')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(colors='white')

        plt.tight_layout()
        chart_path = os.path.join(self.charts_folder, f"risk_strategy_comparison_{risk_level}.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"Risk strategy comparison chart for risk level {risk_level} saved: {chart_path}")

    def save_risk_strategy_summary(self, strategies_data: Dict):
        """Save risk strategy summary to CSV"""
        summary_data = []
        
        for name, data in strategies_data.items():
            metrics = data['metrics']
            summary_data.append({
                'Strategy': name,
                'Annual_Return': metrics.get('Annual Return', 0),
                'Volatility': metrics.get('Volatility', 0),
                'Sharpe_Ratio': metrics.get('Sharpe Ratio', 0),
                'Max_Drawdown': metrics.get('Max Drawdown', 0),
                'Win_Rate': metrics.get('Win Rate', 0),
                'Total_Return': metrics.get('Total Return', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(self.csv_folder, "risk_strategy_performance_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print(f"Risk strategy summary saved: {summary_path}")

    # ==================== Original Multi-factor Portfolio Functions ====================
    
    def generate_multi_factor_portfolio_from_strategies(self, 
                                                        strategy_names: List[str], 
                                                        weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """
        Generate a multi-factor portfolio by combining the returns of specified single-factor strategies.
        """
        if not strategy_names:
            print("No strategies provided for multi-factor portfolio generation.")
            self.multi_factor_returns = pd.Series(dtype=float)
            return self.multi_factor_returns

        print(f"\nGenerating multi-factor portfolio from strategies: {strategy_names}...")
        
        selected_returns = {}
        for name in strategy_names:
            if name in self.backtest_results:
                selected_returns[name] = self.backtest_results[name]
            else:
                print(f"Warning: Strategy '{name}' not found in backtest results. Skipping.")
        
        if not selected_returns:
            print("No valid single-factor strategy returns found to build multi-factor portfolio.")
            self.multi_factor_returns = pd.Series(dtype=float)
            return self.multi_factor_returns

        returns_df = pd.DataFrame(selected_returns)
        
        if weights:
            valid_weights = {s: w for s, w in weights.items() if s in returns_df.columns}
            if not valid_weights:
                print("Warning: Provided weights do not match any valid strategies. Using equal weighting.")
                strategy_weights = pd.Series(1.0 / len(returns_df.columns), index=returns_df.columns)
            else:
                total_weight = sum(valid_weights.values())
                if total_weight == 0:
                    print("Warning: Provided weights sum to zero. Using equal weighting.")
                    strategy_weights = pd.Series(1.0 / len(returns_df.columns), index=returns_df.columns)
                else:
                    strategy_weights = pd.Series(valid_weights) / total_weight
                    strategy_weights = strategy_weights.reindex(returns_df.columns, fill_value=0)
        else:
            strategy_weights = pd.Series(1.0 / len(returns_df.columns), index=returns_df.columns)
            print("No specific weights provided for multi-factor portfolio. Using equal weighting.")

        returns_df = returns_df.fillna(0)
        
        self.multi_factor_returns = (returns_df * strategy_weights).sum(axis=1)
        self.multi_factor_returns = self.multi_factor_returns.sort_index()
        
        print(f"Multi-factor portfolio returns calculated, {len(self.multi_factor_returns)} periods.")
        return self.multi_factor_returns

    # ==================== Original Single Factor Strategy Functions ====================
    
    def create_single_factor_strategy(self, factor_col: str, top_n: int = 5, 
                                    weighting_scheme: str = 'equal') -> pd.DataFrame:
        """Create single factor strategy with configurable weighting schemes"""
        strategy_data = []
        
        for date, group in self.data.groupby('date'):
            if factor_col not in group.columns:
                continue
                
            valid_data = group.dropna(subset=[factor_col])
            if len(valid_data) == 0:
                continue
            
            if 'volatility' in factor_col.lower():
                top_assets = valid_data.nsmallest(top_n, factor_col)
            elif factor_col == 'strev_weekly':
                top_assets = valid_data.nsmallest(top_n, factor_col)
            else:
                top_assets = valid_data.nlargest(top_n, factor_col)
        
            if top_assets.empty:
                continue

            factor_values = top_assets[factor_col]
            asset_weights = self.calculate_asset_weights(top_assets, weighting_scheme, factor_values)
            
            for idx, (_, asset) in enumerate(top_assets.iterrows()):
                strategy_data.append({
                    'date': date,
                    'symbol': asset['symbol'],
                    'weight': asset_weights.iloc[idx],
                    'factor_value': asset[factor_col],
                    'return': asset.get('future_return', 0),
                    'weighting_scheme': weighting_scheme
                })
        
        return pd.DataFrame(strategy_data)
    
    def calculate_portfolio_returns(self, strategy_df: pd.DataFrame) -> pd.Series:
        """Calculate portfolio returns"""
        if strategy_df.empty:
            return pd.Series(dtype=float)

        portfolio_returns = []
        
        for date, group in strategy_df.groupby('date'):
            weighted_return = (group['weight'] * group['return']).sum()
            portfolio_returns.append({
                'date': date,
                'return': weighted_return
            })
        
        if not portfolio_returns:
            return pd.Series(dtype=float)

        returns_df = pd.DataFrame(portfolio_returns)
        returns_df = returns_df.set_index('date')['return']
        return returns_df.sort_index()
    
    def calculate_performance_metrics(self, returns: pd.Series) -> Dict:
        """Calculate strategy performance metrics"""
        if len(returns) < 2:
            return {
                'Total Return': 0, 'Annual Return': 0, 'Volatility': 0,
                'Sharpe Ratio': 0, 'Max Drawdown': 0, 'Win Rate': 0, 'Avg Return': 0
            }
        
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + returns.mean()) ** 52 - 1
        volatility = returns.std() * np.sqrt(52)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        win_rate = (returns > 0).mean()
        avg_return = returns.mean()
        
        return {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': abs(max_drawdown),
            'Win Rate': win_rate,
            'Avg Return': avg_return
        }
    
    def run_all_single_factor_strategies(self, top_n_assets: int = 5, 
                                   weighting_scheme: str = 'factor_weighted') -> Dict:
        """Run all single factor strategies with improved weighting schemes"""
        print(f"\nStarting four core factor single factor strategies with {weighting_scheme} weighting...")
        
        factor_groups = self.identify_factor_columns()
        all_factors = []
        for factors in factor_groups.values():
            all_factors.extend(factors)
        
        print(f"Creating single factor strategies for the following {len(all_factors)} factors:")
        for i, factor in enumerate(all_factors, 1):
            print(f"  {i}. {factor}")
        
        successful_strategies = 0
        
        for i, factor in enumerate(all_factors):
            print(f"\nProcessing factor {i+1}/{len(all_factors)}: {factor}")
            
            try:
                strategy_df = self.create_single_factor_strategy(factor, top_n_assets, weighting_scheme)
            
                if len(strategy_df) == 0:
                    print(f"  Warning: Factor {factor} generated no valid strategy")
                    continue
                
                returns = self.calculate_portfolio_returns(strategy_df)
                
                if len(returns) == 0:
                    print(f"  Warning: Factor {factor} has no return data")
                    continue
                
                metrics = self.calculate_performance_metrics(returns)
                
                self.strategies[factor] = strategy_df
                self.backtest_results[factor] = returns
                self.performance_metrics[factor] = metrics
                
                successful_strategies += 1
                
                print(f"  Success: Annual Return: {metrics.get('Annual Return', 0):>8.2%}, "
                      f"Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):>6.3f}, "
                      f"Max Drawdown: {metrics.get('Max Drawdown', 0):>8.2%}")
                
            except Exception as e:
                print(f"  Error processing factor {factor}: {str(e)}")
                continue
    
        print(f"\nSuccessfully created {successful_strategies} single factor strategies using {weighting_scheme} weighting")
        return self.performance_metrics
    
    def get_top_strategies(self, n: int = 5, metric: str = 'Sharpe Ratio') -> List[str]:
        """Get top N performing strategies"""
        if not self.performance_metrics:
            return []
        
        sorted_strategies = sorted(
            self.performance_metrics.items(),
            key=lambda x: x[1].get(metric, -999),
            reverse=True
        )
        
        top_strategies = [name for name, _ in sorted_strategies[:n]]
        
        print(f"\nTop {n} strategies (sorted by {metric}):")
        print("-" * 70)
        print(f"{'Rank':<4} {'Factor Name':<20} {metric:<15} {'Annual Return':<12} {'Max Drawdown':<12}")
        print("-" * 70)
        
        for i, (name, metrics) in enumerate(sorted_strategies[:n], 1):
            print(f"{i:<4} {name:<20} {metrics.get(metric, 0):>13.4f} "
                  f"{metrics.get('Annual Return', 0):>10.2%} "
                  f"{metrics.get('Max Drawdown', 0):>10.2%}")
        
        return top_strategies
    
    # ==================== Visualization Functions ====================
    
    def create_performance_radar_chart(self, top_strategies: List[str], save_path: str = "performance_radar.png"):
        """Create performance metrics radar chart"""
        if not top_strategies:
            return
        
        save_path = os.path.join(self.charts_folder, save_path)
        metrics_to_plot = ['Annual Return', 'Sharpe Ratio', 'Win Rate', 'Max Drawdown', 'Volatility']
        
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
        angles += angles[:1]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_strategies)))
        
        for i, strategy in enumerate(top_strategies):
            if strategy not in self.performance_metrics:
                continue
                
            metrics = self.performance_metrics[strategy]
            
            values = []
            for metric in metrics_to_plot:
                value = metrics.get(metric, 0)
                if metric == 'Max Drawdown':
                    value = max(0, 1 - abs(value))
                elif metric == 'Volatility':
                    value = max(0, 1 - min(value, 1))
                else:
                    value = min(max(value, 0), 1)
                values.append(value)
            
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=strategy, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_to_plot, color='white', fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], color='white')
        ax.grid(True, color='gray', alpha=0.3)
        
        plt.title('Performance Metrics Radar Chart - Top Single Factor Strategies', 
                 size=16, color='white', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"Radar chart saved to: {save_path}")
    
    def create_rolling_sharpe_dashboard(self, top_strategies: List[str], window: int = 52, 
                                      save_path: str = "rolling_sharpe_dashboard.png"):
        """Create rolling Sharpe ratio dashboard"""
        if not top_strategies:
            return
        
        save_path = os.path.join(self.charts_folder, save_path)
        fig = plt.figure(figsize=(16, 12))
        
        ax1 = plt.subplot(2, 2, 1)
        
        for strategy in top_strategies:
            if strategy not in self.backtest_results:
                continue
            returns = self.backtest_results[strategy]
            if len(returns) >= window:
                rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(52)
                ax1.plot(rolling_sharpe.index, rolling_sharpe.values, label=strategy, linewidth=2)
        
        if self.multi_factor_returns is not None and len(self.multi_factor_returns) >= window:
            mf_rolling_sharpe = self.multi_factor_returns.rolling(window).mean() / self.multi_factor_returns.rolling(window).std() * np.sqrt(52)
            ax1.plot(mf_rolling_sharpe.index, mf_rolling_sharpe.values, label='Multi-Factor', color='red', linestyle='-.', linewidth=2)

        ax1.set_title(f'Rolling Sharpe Ratio ({window}W Window) - Top Single Factor Strategies vs Multi-Factor', 
                     color='white', fontsize=14)
        ax1.set_ylabel('Sharpe Ratio', color='white')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(colors='white')
        
        ax2 = plt.subplot(2, 2, 2)
        
        for strategy in top_strategies:
            if strategy not in self.backtest_results:
                continue
            returns = self.backtest_results[strategy]
            if len(returns) >= window:
                rolling_vol = returns.rolling(window).std() * np.sqrt(52)
                ax2.plot(rolling_vol.index, rolling_vol.values, label=strategy, linewidth=2)

        if self.multi_factor_returns is not None and len(self.multi_factor_returns) >= window:
            mf_rolling_vol = self.multi_factor_returns.rolling(window).std() * np.sqrt(52)
            ax2.plot(mf_rolling_vol.index, mf_rolling_vol.values, label='Multi-Factor', color='red', linestyle='-.', linewidth=2)
        
        ax2.set_title('Rolling Volatility - Top Single Factor Strategies vs Multi-Factor', color='white', fontsize=14)
        ax2.set_ylabel('Volatility', color='white')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(colors='white')
        
        ax3 = plt.subplot(2, 2, 3)
        all_returns = []
        for strategy in top_strategies:
            if strategy in self.backtest_results:
                all_returns.extend(self.backtest_results[strategy].values)
        
        if self.multi_factor_returns is not None and len(self.multi_factor_returns) > 0:
            all_returns.extend(self.multi_factor_returns.values)

        if all_returns:
            ax3.hist(all_returns, bins=50, alpha=0.7, color='skyblue', edgecolor='white')
        ax3.set_title('Return Distribution', color='white', fontsize=14)
        ax3.set_xlabel('Returns', color='white')
        ax3.set_ylabel('Frequency', color='white')
        ax3.tick_params(colors='white')
        
        ax4 = plt.subplot(2, 2, 4)
        corr_data = {}
        for strategy in top_strategies:
            if strategy in self.backtest_results:
                corr_data[strategy] = self.backtest_results[strategy]
        
        if corr_data:
            corr_df = pd.DataFrame(corr_data).corr()
            im = ax4.imshow(corr_df.values, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)
            ax4.set_xticks(range(len(corr_df.columns)))
            ax4.set_yticks(range(len(corr_df.columns)))
            ax4.set_xticklabels(corr_df.columns, rotation=45, ha='right', color='white')
            ax4.set_yticklabels(corr_df.columns, color='white')
            
            for i in range(len(corr_df.columns)):
                for j in range(len(corr_df.columns)):
                    ax4.text(j, i, f'{corr_df.iloc[i, j]:.2f}', 
                            ha='center', va='center', color='white', fontsize=10)
            
            plt.colorbar(im, ax=ax4)
        
        ax4.set_title('Correlation Matrix', color='white', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"Rolling Sharpe ratio dashboard saved to: {save_path}")
    
    def create_performance_bar_charts(self, top_strategies: List[str]):
        """Create performance metrics bar charts"""
        if not top_strategies:
            return
        
        metrics_to_plot = [
            ('Annual Return', 'Mean % p.a.', 'mean_bars.png'),
            ('Volatility', 'Vol % p.a.', 'vol_bars.png'),
            ('Sharpe Ratio', 'Sharpe', 'sharpe_bars.png')
        ]
        
        strategies = []
        for strategy in top_strategies:
            if strategy in self.performance_metrics:
                strategies.append(strategy)

        # Add benchmarks to the list
        btc_metrics = self.calculate_btc_benchmark()
        if btc_metrics:
            strategies.append('BTC')
        if self.multi_factor_returns is not None and len(self.multi_factor_returns) > 0:
            mf_metrics = self.calculate_performance_metrics(self.multi_factor_returns)
            if mf_metrics:
                strategies.append('Multi-Factor')

        # Define colors for all strategies and benchmarks
        custom_colors = ['yellow', '#FF7F50', '#6495ED'] # Yellow, Coral Red, Cornflower Blue
        color_cycle = cycle(custom_colors)
        
        color_map = {}
        for name in top_strategies:
            color_map[name] = next(color_cycle)
        if 'BTC' in strategies:
            color_map['BTC'] = 'orange'
        if 'Multi-Factor' in strategies:
            color_map['Multi-Factor'] = 'red'

        for metric, title, filename in metrics_to_plot:
            filename = os.path.join(self.charts_folder, filename)
            fig, ax = plt.subplots(figsize=(12, 8))
            
            values = []
            display_names = []
            plot_colors = []
            
            for strategy in strategies:
                display_name = strategy.replace('momentum_', 'Mom_').replace('volatility_', 'Vol_').replace('usd_volume', 'Volume').replace('strev_weekly', 'Reversal')
                display_names.append(display_name)
                
                if strategy == 'BTC':
                    value = btc_metrics.get(metric, 0)
                elif strategy == 'Multi-Factor':
                    value = mf_metrics.get(metric, 0)
                else:
                    value = self.performance_metrics[strategy].get(metric, 0)

                if metric in ['Annual Return', 'Volatility']:
                    value *= 100
                values.append(value)
                plot_colors.append(color_map[strategy])
            
            bars = ax.bar(display_names, values, color=plot_colors)
            
            ax.set_title(f'{title} – Top 5 Single Factor Strategies vs Benchmarks', 
                        fontsize=16, color='white', pad=20)
            ax.set_ylabel(title, fontsize=12, color='white')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3, color='gray')
            
            plt.xticks(rotation=45, ha='right', color='white')
            plt.yticks(color='white')
            
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
            print(f"{title} bar chart saved to: {filename}")
            
    def create_cumulative_returns_chart(self, top_strategies: List[str], 
                                      save_path: str = "cumulative_returns.png"):
        """Create cumulative returns chart"""
        if not top_strategies:
            return
        
        save_path = os.path.join(self.charts_folder, save_path)
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for strategy in top_strategies:
            if strategy not in self.backtest_results:
                continue
            
            returns = self.backtest_results[strategy]
            if not returns.empty:
                cumulative = (1 + returns).cumprod()
                display_name = strategy.replace('momentum_', 'Mom_').replace('volatility_', 'Vol_').replace('usd_volume', 'Volume').replace('strev_weekly', 'Reversal')
                ax.plot(cumulative.index, cumulative.values, label=display_name, linewidth=2)
        
        btc_returns = self.get_btc_returns()
        if btc_returns is not None and len(btc_returns) > 0:
            btc_cumulative = (1 + btc_returns).cumprod()
            ax.plot(btc_cumulative.index, btc_cumulative.values, 
                   label='BTC', linewidth=3, color='orange', linestyle='--')
            # NEW: Store the correct cumulative returns for future use
            self.btc_cumulative_returns = btc_cumulative
        
        if self.multi_factor_returns is not None and len(self.multi_factor_returns) > 0:
            mf_cumulative = (1 + self.multi_factor_returns).cumprod()
            ax.plot(mf_cumulative.index, mf_cumulative.values, 
                   label='Multi-Factor', linewidth=3, color='red', linestyle='-.')
        
        ax.set_title('Cumulative Returns – Top 5 Single Factor Strategies vs Benchmarks', 
                    fontsize=16, color='white')
        ax.set_xlabel('Date', fontsize=12, color='white')
        ax.set_ylabel('Cumulative Return', fontsize=12, color='white')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"Cumulative returns chart saved to: {save_path}")
    
    def create_monthly_returns_candles(self, top_strategies: List[str], 
                                     save_path: str = "monthly_returns_candles.png"):
        """Create monthly returns candle chart"""
        if not top_strategies:
            return
        
        save_path = os.path.join(self.charts_folder, save_path)
        fig, ax = plt.subplots(figsize=(20, 10))
        
        monthly_data = {}
        for strategy in top_strategies:
            if strategy not in self.backtest_results:
                continue
            
            returns = self.backtest_results[strategy]
            if not returns.empty:
                monthly_returns = returns.resample('M').sum()
                display_name = strategy.replace('momentum_', 'Mom_').replace('volatility_', 'Vol_').replace('usd_volume', 'Volume').replace('strev_weekly', 'Reversal')
                monthly_data[display_name] = monthly_returns
        
        if not monthly_data:
            return
        
        all_months = set()
        for returns in monthly_data.values():
            all_months.update(returns.index)
        all_months = sorted(all_months)
        
        width = 0.15
        x_positions = np.arange(len(all_months))
        colors = plt.cm.Set3(np.linspace(0, 1, len(monthly_data)))
        
        for i, (strategy, returns) in enumerate(monthly_data.items()):
            strategy_returns = []
            for month in all_months:
                if month in returns.index:
                    strategy_returns.append(returns[month])
                else:
                    strategy_returns.append(0)
            
            # Here we only set the label for the bar plot, but don't rely on its color for the legend
            bars = ax.bar(x_positions + i * width, strategy_returns, width, 
                         color=colors[i], alpha=0.8) # label removed from here
            
            for j, (bar, ret) in enumerate(zip(bars, strategy_returns)):
                if ret < 0:
                    bar.set_color('red')
                    bar.set_alpha(0.6)
        
        # --- START MODIFICATION ---
        # Create custom legend handles
        legend_handles = []
        for i, strategy_name in enumerate(monthly_data.keys()):
            legend_handles.append(Patch(color=colors[i], label=strategy_name))
        
        ax.legend(handles=legend_handles, fontsize=10)
        # --- END MODIFICATION ---
        
        ax.set_title('Monthly Return "Candles" - Top 5 Single Factor Strategies', 
                    fontsize=16, color='white', pad=20)
        ax.set_xlabel('Month', fontsize=12, color='white')
        ax.set_ylabel('Returns', fontsize=12, color='white')
        ax.grid(True, alpha=0.3)
        ax.tick_params(colors='white')
        
        ax.set_xticks(x_positions + width * (len(monthly_data) - 1) / 2)
        ax.set_xticklabels([month.strftime('%Y-%m') for month in all_months], 
                          rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"Monthly returns candle chart saved to: {save_path}")
    
    def create_drawdown_analysis(self, top_strategies: List[str], 
                               save_path: str = "drawdown_analysis.png"):
        """Create drawdown analysis chart"""
        if not top_strategies:
            return
        
        save_path = os.path.join(self.charts_folder, save_path)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_strategies)))
        
        for i, strategy in enumerate(top_strategies):
            if strategy not in self.backtest_results:
                continue
            
            returns = self.backtest_results[strategy]
            if not returns.empty:
                cumulative = (1 + returns).cumprod()
                display_name = strategy.replace('momentum_', 'Mom_').replace('volatility_', 'Vol_').replace('usd_volume', 'Volume').replace('strev_weekly', 'Reversal')
                ax1.plot(cumulative.index, cumulative.values, 
                        label=display_name, linewidth=2, color=colors[i])
                
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                major_drawdowns = drawdown < -0.05
                
                if major_drawdowns.any():
                    ax1.fill_between(cumulative.index, cumulative.values, 
                                   rolling_max.values, where=major_drawdowns,
                                   alpha=0.3, color=colors[i])
        
        if self.btc_cumulative_returns is not None and not self.btc_cumulative_returns.empty:
            btc_cumulative = self.btc_cumulative_returns
            ax1.plot(btc_cumulative.index, btc_cumulative.values, 
                   label='BTC Benchmark', linewidth=3, color='orange', linestyle='--')
            
            btc_rolling_max = btc_cumulative.expanding().max()
            btc_drawdown = (btc_cumulative - btc_rolling_max) / btc_rolling_max
            btc_major_drawdowns = btc_drawdown < -0.05
            if btc_major_drawdowns.any():
                ax1.fill_between(btc_cumulative.index, btc_cumulative.values, 
                                 btc_rolling_max.values, where=btc_major_drawdowns,
                                 alpha=0.3, color='orange')

        ax1.set_title('Cumulative Returns with Major Drawdowns (>5%) - Single Factor Strategies vs Benchmarks', 
                     fontsize=14, color='white')
        ax1.set_ylabel('Cumulative Return', color='white')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(colors='white')
        
        for i, strategy in enumerate(top_strategies):
            if strategy not in self.backtest_results:
                continue
            
            returns = self.backtest_results[strategy]
            if not returns.empty:
                cumulative = (1 + returns).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                
                display_name = strategy.replace('momentum_', 'Mom_').replace('volatility_', 'Vol_').replace('usd_volume', 'Volume').replace('strev_weekly', 'Reversal')
                ax2.fill_between(drawdown.index, 0, drawdown.values, 
                               alpha=0.7, color=colors[i], label=display_name)
        
        # Add BTC drawdown
        if self.btc_cumulative_returns is not None and not self.btc_cumulative_returns.empty:
            btc_cumulative = self.btc_cumulative_returns
            btc_rolling_max = btc_cumulative.expanding().max()
            btc_drawdown = (btc_cumulative - btc_rolling_max) / btc_rolling_max
            ax2.fill_between(btc_drawdown.index, 0, btc_drawdown.values, 
                             alpha=0.7, color='orange', label='BTC Benchmark')
        
        if self.multi_factor_returns is not None and len(self.multi_factor_returns) > 0:
            mf_cumulative = (1 + self.multi_factor_returns).cumprod()
            mf_rolling_max = mf_cumulative.expanding().max()
            mf_drawdown = (mf_cumulative - mf_rolling_max) / mf_rolling_max
            ax2.fill_between(mf_drawdown.index, 0, mf_drawdown.values, 
                             alpha=0.7, color='red', label='Multi-Factor')

        ax2.set_title('Drawdown Analysis - Single Factor Strategies vs Benchmarks', fontsize=14, color='white')
        ax2.set_xlabel('Date', color='white')
        ax2.set_ylabel('Drawdown', color='white')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(colors='white')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"Drawdown analysis chart saved to: {save_path}")
    
    def generate_all_charts(self, top_strategies: List[str]):
        """Generate all charts"""
        print("\nStarting to generate all visualization charts...")
        
        self.create_performance_radar_chart(top_strategies)
        self.create_rolling_sharpe_dashboard(top_strategies)
        self.create_performance_bar_charts(top_strategies)
        self.create_cumulative_returns_chart(top_strategies)
        self.create_monthly_returns_candles(top_strategies)
        self.create_drawdown_analysis(top_strategies)
        
        print("\nAll charts generated successfully!")
    
    def plot_all_portfolios_comparison(self, save_path: str = "all_portfolios_comparison.png"):
        """
        Plots a comprehensive comparison of all generated multi-factor portfolios.
        """
        print("\nGenerating comprehensive portfolio comparison chart...")

        has_risk_results = self.risk_backtest_results and any(not r.empty for r in self.risk_backtest_results.values())
        has_mf_results = self.multi_factor_returns is not None and not self.multi_factor_returns.empty

        if not has_risk_results and not has_mf_results:
            print("No portfolio returns data available for comprehensive plotting.")
            return

        save_path = os.path.join(self.charts_folder, save_path)
        fig, ax = plt.subplots(figsize=(16, 9), facecolor='black')
        fig.suptitle('Comprehensive Portfolio Performance Comparison', fontsize=20, color='white', y=0.95)

        num_colors = len(self.risk_backtest_results) + 2
        colors = plt.cm.viridis(np.linspace(0, 1, num_colors))
        color_idx = 0
        
        all_returns_data = {}

        for name, returns in self.risk_backtest_results.items():
            if not returns.empty:
                all_returns_data[name.replace('_', ' ').title()] = returns

        if has_mf_results:
            all_returns_data['Multi-Factor (Top Single Factors)'] = self.multi_factor_returns

        btc_returns = self.get_btc_returns()
        if btc_returns is not None and not btc_returns.empty:
            all_returns_data['BTC Benchmark'] = btc_returns
            
        common_start_date = max([ret.index.min() for ret in all_returns_data.values() if not ret.empty])

        for name, returns in all_returns_data.items():
            aligned_returns = returns[returns.index >= common_start_date]
            if not aligned_returns.empty:
                cumulative_returns = (1 + aligned_returns).cumprod()
                
                linestyle = '--' if 'BTC' in name else '-'
                linewidth = 2.5 if 'BTC' in name or 'Multi-Factor' in name else 2
                color = 'orange' if 'BTC' in name else ('cyan' if 'Multi-Factor' in name else colors[color_idx])
                
                ax.plot(cumulative_returns.index, cumulative_returns.values, 
                        label=name, color=color, linestyle=linestyle, linewidth=linewidth)
                
                if 'BTC' not in name and 'Multi-Factor' not in name:
                    color_idx += 1

        ax.set_title('Cumulative Returns', fontsize=16, color='white')
        ax.set_xlabel('Date', fontsize=12, color='white')
        ax.set_ylabel('Cumulative Return (Log Scale)', fontsize=12, color='white')
        ax.set_yscale('log')
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, which="both", ls="--", alpha=0.3, color='gray')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='black')
        print(f"Comprehensive portfolio comparison chart saved to: {save_path}")
    
    # ==================== Result Saving and Analysis Functions ====================
    
    def save_results(self, output_path: str = "portfolio_results.csv"):
        """Save multi-factor portfolio results"""
        if self.multi_factor_returns is None or len(self.multi_factor_returns) == 0:
            print("No multi-factor portfolio returns to save.")
            return
        
        output_path = os.path.join(self.csv_folder, output_path)
        self.multi_factor_returns.to_csv(output_path, header=['multi_factor_return'])
        print(f"Multi-factor portfolio returns saved to: {output_path}")
    
    def save_strategy_results(self, top_strategies: List[str]):
        """Save strategy analysis results"""
        if not self.performance_metrics:
            print("No strategy analysis results to save")
            return
        
        metrics_df = pd.DataFrame(self.performance_metrics).T
        
        if self.multi_factor_returns is not None and len(self.multi_factor_returns) > 0:
            mf_metrics = self.calculate_performance_metrics(self.multi_factor_returns)
            if mf_metrics:
                metrics_df.loc['Multi_Factor_Portfolio'] = mf_metrics
        
        metrics_path = os.path.join(self.csv_folder, "single_factor_strategy_performance_metrics.csv")
        metrics_df.to_csv(metrics_path)
        print(f"Single factor strategy performance metrics saved to: {metrics_path}")
        
        for i, strategy in enumerate(top_strategies, 1):
            if strategy in self.strategies:
                strategy_df = self.strategies[strategy]
                filename = f"top_{i}_single_factor_strategy_{strategy.replace('/', '_')}_holdings.csv"
                filepath = os.path.join(self.csv_folder, filename)
                strategy_df.to_csv(filepath, index=False)
                print(f"Top {i} single factor strategy holdings saved to: {filepath}")
    
        if self.backtest_results:
            returns_df = pd.DataFrame(self.backtest_results)
            
            if self.multi_factor_returns is not None and len(self.multi_factor_returns) > 0:
                returns_df['Multi_Factor_Portfolio'] = self.multi_factor_returns
            
            returns_path = os.path.join(self.csv_folder, "single_factor_strategy_returns_timeseries.csv")
            returns_df.to_csv(returns_path)
            print(f"Single factor strategy returns time series saved to: {returns_path}")
    
        comparison_metrics = ['Annual Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
        if self.performance_metrics:
            metrics_df = pd.DataFrame(self.performance_metrics).T
            top_5_comparison = metrics_df.loc[top_strategies, comparison_metrics]
            comparison_path = os.path.join(self.csv_folder, "top_5_single_factor_strategies_comparison.csv")
            top_5_comparison.to_csv(comparison_path)
            print(f"Top 5 single factor strategies comparison table saved to: {comparison_path}")
        
        self.save_detailed_weekly_holdings(top_strategies)
        self.create_weekly_summary_report(top_strategies)
    
    def save_detailed_weekly_holdings(self, top_strategies: List[str]):
        """Save detailed weekly holdings data for each strategy"""
        if not self.strategies:
            print("No strategy data to save")
            return
        
        detailed_folder = os.path.join(self.base_folder, "strategy_weekly_holdings")
        if not os.path.exists(detailed_folder):
            os.makedirs(detailed_folder)
            print(f"Created strategy detailed holdings folder: {detailed_folder}")
        
        print(f"\nStarting to save weekly holdings details for each strategy...")
        
        for i, strategy in enumerate(top_strategies, 1):
            if strategy not in self.strategies:
                continue
            
            strategy_df = self.strategies[strategy].copy()
            
            if len(strategy_df) == 0:
                print(f"  Warning: Strategy {strategy} has no holdings data")
                continue
            
            strategy_df['strategy_name'] = strategy
            strategy_df['strategy_rank'] = i
            
            if strategy in self.performance_metrics:
                metrics = self.performance_metrics[strategy]
                strategy_df['annual_return'] = metrics.get('Annual Return', 0)
                strategy_df['sharpe_ratio'] = metrics.get('Sharpe Ratio', 0)
                strategy_df['max_drawdown'] = metrics.get('Max Drawdown', 0)
                strategy_df['win_rate'] = metrics.get('Win Rate', 0)
                strategy_df['volatility'] = metrics.get('Volatility', 0)
            
            strategy_df = strategy_df.sort_values(['date', 'weight'], ascending=[True, False])
            
            weekly_stats = []
            for date, group in strategy_df.groupby('date'):
                weekly_stats.append({
                    'date': date,
                    'total_assets': len(group),
                    'top_asset': group.iloc[0]['symbol'],
                    'top_weight': group.iloc[0]['weight'],
                    'weight_concentration': (group['weight'] ** 2).sum(),
                    'avg_factor_value': group['factor_value'].mean(),
                    'weekly_return': (group['weight'] * group['return']).sum() if 'return' in group.columns else 0
                })
            
            weekly_stats_df = pd.DataFrame(weekly_stats)
            
            if not weekly_stats_df.empty:
                strategy_df_enhanced = strategy_df.merge(
                    weekly_stats_df.add_suffix('_weekly'), 
                    left_on='date', 
                    right_on='date_weekly', 
                    how='left'
                )
            else:
                strategy_df_enhanced = strategy_df.copy()

            column_order = [
                'strategy_name', 'strategy_rank', 'date', 'symbol', 'weight', 'factor_value', 'return',
                'total_assets_weekly', 'top_asset_weekly', 'top_weight_weekly', 
                'weight_concentration_weekly', 'avg_factor_value_weekly', 'weekly_return_weekly',
                'annual_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'volatility'
            ]
            
            available_columns = [col for col in column_order if col in strategy_df_enhanced.columns]
            strategy_df_final = strategy_df_enhanced[available_columns]
            
            clean_strategy_name = strategy.replace('/', '_').replace('\\', '_').replace(':', '_')
            filename = f"rank_{i:02d}_{clean_strategy_name}_weekly_holdings.csv"
            filepath = os.path.join(detailed_folder, filename)
            
            strategy_df_final.to_csv(filepath, index=False)
            
            total_weeks = strategy_df_final['date'].nunique()
            unique_assets = strategy_df_final['symbol'].nunique()
            avg_weekly_assets = strategy_df_final.groupby('date').size().mean()
            
            print(f"  Success: Rank {i} strategy: {strategy}")
            print(f"     File: {filename}")
            print(f"     Stats: {total_weeks} weeks, {unique_assets} assets, avg {avg_weekly_assets:.1f} holdings per week")
        
        self.create_strategies_overview(top_strategies, detailed_folder)
        self.create_weekly_summary_report(top_strategies)
    
    def create_strategies_overview(self, top_strategies: List[str], detailed_folder: str):
        """Create strategy overview file"""
        overview_data = []
        
        for i, strategy in enumerate(top_strategies, 1):
            if strategy not in self.strategies or strategy not in self.performance_metrics:
                continue
            
            strategy_df = self.strategies[strategy]
            metrics = self.performance_metrics[strategy]
            
            total_weeks = strategy_df['date'].nunique()
            unique_assets = strategy_df['symbol'].nunique()
            avg_weekly_assets = strategy_df.groupby('date').size().mean()
            
            asset_freq = strategy_df['symbol'].value_counts()
            most_frequent_asset = asset_freq.index[0] if len(asset_freq) > 0 else 'N/A'
            most_frequent_count = asset_freq.iloc[0] if len(asset_freq) > 0 else 0
            
            avg_weight = strategy_df['weight'].mean()
            max_weight = strategy_df['weight'].max()
            min_weight = strategy_df['weight'].min()
            
            overview_data.append({
                'rank': i,
                'strategy_name': strategy,
                'factor_type': self.get_factor_type(strategy),
                'annual_return': metrics.get('Annual Return', 0),
                'sharpe_ratio': metrics.get('Sharpe Ratio', 0),
                'max_drawdown': metrics.get('Max Drawdown', 0),
                'win_rate': metrics.get('Win Rate', 0),
                'volatility': metrics.get('Volatility', 0),
                'total_weeks': total_weeks,
                'unique_assets': unique_assets,
                'avg_weekly_assets': avg_weekly_assets,
                'most_frequent_asset': most_frequent_asset,
                'most_frequent_count': most_frequent_count,
                'most_frequent_percentage': (most_frequent_count / total_weeks * 100) if total_weeks > 0 else 0,
                'avg_weight': avg_weight,
                'max_weight': max_weight,
                'min_weight': min_weight,
                'filename': f"rank_{i:02d}_{strategy.replace('/', '_').replace('\\', '_').replace(':', '_')}_weekly_holdings.csv"
            })
        
        overview_df = pd.DataFrame(overview_data)
        overview_path = os.path.join(detailed_folder, "00_strategies_overview.csv")
        overview_df.to_csv(overview_path, index=False)
        
        print(f"  Strategy overview file saved: 00_strategies_overview.csv")

    def get_factor_type(self, strategy_name: str) -> str:
        """Determine factor type based on strategy name"""
        if 'momentum' in strategy_name.lower():
            return 'Momentum Factor'
        elif 'volatility' in strategy_name.lower():
            return 'Volatility Factor'
        elif 'volume' in strategy_name.lower():
            return 'Volume Impact Factor'
        elif 'strev' in strategy_name.lower():
            return 'Short-term Reversal Factor'
        else:
            return 'Other Factor'

    def create_weekly_summary_report(self, top_strategies: List[str]):
        """Create weekly summary report"""
        if not self.strategies:
            return
        
        detailed_folder = os.path.join(self.base_folder, "strategy_weekly_holdings")
        all_weekly_data = []
        
        for strategy in top_strategies:
            if strategy not in self.strategies:
                continue
            
            strategy_df = self.strategies[strategy]
            
            for date, group in strategy_df.groupby('date'):
                weekly_summary = {
                    'date': date,
                    'strategy': strategy,
                    'factor_type': self.get_factor_type(strategy),
                    'num_holdings': len(group),
                    'top_holding': group.loc[group['weight'].idxmax(), 'symbol'],
                    'top_weight': group['weight'].max(),
                    'weight_std': group['weight'].std(),
                    'avg_factor_value': group['factor_value'].mean(),
                    'portfolio_return': (group['weight'] * group['return']).sum() if 'return' in group.columns else 0
                }
                all_weekly_data.append(weekly_summary)
    
        if all_weekly_data:
            weekly_summary_df = pd.DataFrame(all_weekly_data)
            weekly_summary_df = weekly_summary_df.sort_values(['date', 'strategy'])
            
            summary_path = os.path.join(detailed_folder, "weekly_summary_all_strategies.csv")
            weekly_summary_df.to_csv(summary_path, index=False)
            
            print(f"  Weekly summary report saved: weekly_summary_all_strategies.csv")
    
    def compare_strategies(self, top_strategies: List[str]):
        """Strategy comparison analysis"""
        if not top_strategies or not self.performance_metrics:
            return
        
        print(f"\nSingle factor strategy comparison analysis:")
        print("-" * 90)
        print(f"{'Strategy Name':<20} {'Annual Return':<12} {'Volatility':<12} {'Sharpe Ratio':<12} {'Max Drawdown':<12} {'Win Rate':<12}")
        print("-" * 90)
        
        for strategy in top_strategies:
            if strategy in self.performance_metrics:
                display_name = strategy.replace('momentum_', 'Mom_').replace('volatility_', 'Vol_').replace('usd_volume', 'Volume').replace('strev_weekly', 'Reversal')
                metrics = self.performance_metrics[strategy]
                print(f"{display_name:<20} "
                      f"{metrics.get('Annual Return', 0):>10.2%} "
                      f"{metrics.get('Volatility', 0):>10.2%} "
                      f"{metrics.get('Sharpe Ratio', 0):>10.3f} "
                      f"{metrics.get('Max Drawdown', 0):>10.2%} "
                      f"{metrics.get('Win Rate', 0):>10.2%}")
        
        if self.multi_factor_returns is not None and len(self.multi_factor_returns) > 0:
            mf_metrics = self.calculate_performance_metrics(self.multi_factor_returns)
            if mf_metrics:
                print(f"{'Multi-Factor':<20} "
                      f"{mf_metrics.get('Annual Return', 0):>10.2%} "
                      f"{mf_metrics.get('Volatility', 0):>10.2%} "
                      f"{mf_metrics.get('Sharpe Ratio', 0):>10.3f} "
                      f"{mf_metrics.get('Max Drawdown', 0):>10.2%} "
                      f"{mf_metrics.get('Win Rate', 0):>10.2%}")
        
        if len(top_strategies) >= 2:
            corr_data = {}
            for strategy in top_strategies:
                if strategy in self.backtest_results:
                    display_name = strategy.replace('momentum_', 'Mom_').replace('volatility_', 'Vol_').replace('usd_volume', 'Volume').replace('strev_weekly', 'Reversal')
                    corr_data[display_name] = self.backtest_results[strategy]
            
            if self.multi_factor_returns is not None and len(self.multi_factor_returns) > 0:
                corr_data['Multi-Factor'] = self.multi_factor_returns
            
            if len(corr_data) >= 2:
                correlation_matrix = pd.DataFrame(corr_data).corr()
                plt.figure(figsize=(10, 8))
                ax = sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0,
                           square=True, fmt='.3f', cbar_kws={'label': 'Correlation'})
                
                for text in ax.texts:
                    text.set_color('white')

                plt.title('Top Single Factor Strategies Correlation Matrix', fontsize=14, pad=20, color='white')
                plt.xticks(color='white')
                plt.yticks(color='white')
                plt.xlabel('', color='white')
                plt.ylabel('', color='white')
                
                plt.tight_layout()
                chart_path = os.path.join(self.charts_folder, "single_factor_strategies_correlation_matrix.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='black')
                print(f"Single factor strategies correlation matrix saved to: {chart_path}")

    def calculate_btc_benchmark(self) -> Dict:
        """Calculate BTC benchmark strategy performance metrics"""
        btc_returns = self.get_btc_returns()
        if btc_returns is None or len(btc_returns) == 0:
            return {}
        
        return self.calculate_performance_metrics(btc_returns)

    def get_btc_returns(self) -> pd.Series:
        """Get BTC return data"""
        if self.data is None:
            return None
        
        btc_data = self.data[self.data['symbol'].str.upper() == 'BTC'].copy()
        if len(btc_data) == 0:
            btc_symbols = ['BTCUSDT', 'BTCUSD', 'BITCOIN']
            for symbol in btc_symbols:
                btc_data = self.data[self.data['symbol'].str.upper() == symbol].copy()
                if len(btc_data) > 0:
                    break
        
        if len(btc_data) == 0:
            print("Warning: No BTC data found")
            return None
        
        btc_data = btc_data.sort_values('date')
        
        if 'return' in btc_data.columns:
            btc_returns = btc_data.set_index('date')['return']
        else:
            btc_data = btc_data.set_index('date')
            btc_returns = btc_data['close'].pct_change().dropna()
        
        # NEW: Cap extreme return values to prevent explosion
        max_return_cap = 1.5 # 150% return per week
        btc_returns = btc_returns.clip(lower=None, upper=max_return_cap)
        
        return btc_returns.sort_index()

    def get_top_strategies_by_total_return(self, n: int = 5) -> List[str]:
        """Get top N strategies by total return"""
        if not self.performance_metrics:
            return []
        
        sorted_strategies = sorted(
            self.performance_metrics.items(),
            key=lambda x: x[1].get('Total Return', -999),
            reverse=True
        )
        
        top_strategies = [name for name, _ in sorted_strategies[:n]]
        
        print(f"\nTop {n} strategies by total return:")
        print("-" * 70)
        print(f"{'Rank':<4} {'Factor Name':<20} {'Total Return':<15} {'Annual Return':<12} {'Sharpe Ratio':<12}")
        print("-" * 70)
        
        for i, (name, metrics) in enumerate(sorted_strategies[:n], 1):
            print(f"{i:<4} {name:<20} {metrics.get('Total Return', 0):>13.2%} "
                  f"{metrics.get('Annual Return', 0):>10.2%} "
                  f"{metrics.get('Sharpe Ratio', 0):>10.3f}")
        
        return top_strategies
