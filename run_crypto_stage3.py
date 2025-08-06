from crypto_strategy_engine import CryptoStrategyEngine
import pandas as pd
import numpy as np
import os

def main():
    """
    Enhanced Crypto Four Core Factor Strategy Analysis System with Custom Risk and Sentiment
    """
    
    # ==================== Parameter Settings ====================
    
    # Data file path
    DATA_PATH = "/Users/zhangzhihan/Downloads/UNSW/Course 2025 T2/FINS5545/Project/Stage3&4/crypto_news_weekly_top.csv"
    
    # Sentiment data path
    SENTIMENT_DATA_PATH = "/Users/zhangzhihan/Downloads/UNSW/Course 2025 T2/FINS5545/Project/Stage3&4/aggregated_weekly.csv"

    # NEW: Weighting scheme configurations
    WEIGHTING_SCHEMES = {
        'single_factor_weighting': 'factor_weighted',    # 'equal', 'factor_weighted', 'inverse_volatility', 'risk_parity', 'optimized'
        'use_dynamic_factor_weights': True,              # Whether to use dynamic factor weight optimization
        'factor_weight_lookback': 52,                    # Lookback periods for factor weight optimization
    }

    # Custom risk strategy selection
    RISK_STRATEGY_SELECTION = {
        'run_custom_risk': True,     # Custom risk strategies
        'use_dynamic_weights': True, # Use dynamic factor weight optimization
    }
    
    # NEW: Market sentiment to test
    MARKET_SENTIMENTS = ['positive', 'neutral', 'negative']

    # NEW: Custom risk tolerance levels to test (0-10 scale)
    CUSTOM_RISK_LEVELS = [0, 5, 8]  # Low, Medium, High examples

    # Enhanced multi-factor portfolio parameters
    MULTI_FACTOR_PARAMS = {
        'top_n_strategies_for_mf': 5,           # Select top N single factor strategies for multi-factor portfolio
        'strategy_weights': {                   # Custom weights for strategies (if empty, will use optimization)
            # Example: 'usd_volume': 0.4, 'momentum_90': 0.3, 'momentum_14': 0.2, 'momentum_21': 0.1
            # Leave empty for automatic optimization
        },
        'use_optimized_weights': True,          # Whether to use optimization-based strategy weights
        'rebalance_frequency': 4,               # Rebalance every N weeks
        'save_output': True,                    # Whether to save results
        'output_path': "multi_factor_portfolio_from_strategies.csv"  # Output filename
    }

    # Enhanced single factor strategy parameters
    SINGLE_FACTOR_PARAMS = {
        'top_n_assets': 5,                      # Number of assets selected per strategy
        'top_n_strategies': 5,                  # Number of top performing strategies to select
        'ranking_metric': 'Sharpe Ratio',       # Ranking metric ('Sharpe Ratio', 'Annual Return', 'Win Rate')
        'weighting_scheme': 'factor_weighted',   # Asset weighting scheme
    }
    
    # Visualization parameters
    VISUALIZATION_PARAMS = {
        'generate_all_charts': True,            # Whether to generate all charts
        'rolling_window': 52,                   # Rolling window size (weeks)
    }
    
    # Run mode selection
    RUN_MODES = {
        'multi_factor_portfolio': True,         # Whether to run multi-factor portfolio
        'single_factor_strategies': True,       # Whether to run single factor strategy backtests
        'generate_visualizations': True,        # Whether to generate visualization charts
        'save_detailed_results': True,          # Whether to save detailed results
        'risk_based_strategies': True,          # NEW: Whether to run risk-based strategies
    }
    
    # ==================== Parameter Settings End ====================
    
    print("Enhanced Crypto Four Core Factor Strategy Analysis System")
    print("=" * 80)
    print("Core Factor Categories:")
    print("   1. Momentum factors (momentum_14/21/28/42/90)")
    print("   2. Volatility factors (volatility_14/21/28/42/90)")
    print("   3. Volume impact factors (usd_volume, deduplicated)")
    print("   4. Short-term reversal factors (strev_weekly)")
    print("\nNEW: Custom Risk Strategy Selection:")
    print("   • Custom Risk Strategy - User-defined risk tolerance (0-10 scale)")
    print("   Benchmark strategy: BTC buy and hold")
    print("=" * 80)
    print(f"Data file: {DATA_PATH}")
    print(f"Run modes: {[k for k, v in RUN_MODES.items() if v]}")
    print("=" * 80)
    
    try:
        # 1. Initialize enhanced strategy engine
        engine = CryptoStrategyEngine(DATA_PATH, SENTIMENT_DATA_PATH)
        
        print(f"\nFolder structure:")
        print(f"  - Main folder: {engine.base_folder}")
        print(f"  - CSV data: {engine.csv_folder}")
        print(f"  - Charts: {engine.charts_folder}")
        print(f"  - Weekly allocations: {engine.weekly_allocations_folder}")
        
        # 2. Load data
        print("\nStep 1: Load data")
        data = engine.load_data()
        if data is None:
            print("Data loading failed")
            return
        
        # NEW: 3. Run risk-based strategies
        all_risk_strategies_data = {}
        
        if RUN_MODES['risk_based_strategies']:
            print(f"\n" + "="*80)
            print("STEP 2: CUSTOM RISK-BASED STRATEGY ANALYSIS")
            print("="*80)
            
            if RISK_STRATEGY_SELECTION['run_custom_risk'] and CUSTOM_RISK_LEVELS:
                print(f"\n" + "-"*60)
                print("CUSTOM RISK STRATEGIES")
                print("-"*60)
                
                for risk_level in CUSTOM_RISK_LEVELS:
                    print(f"\n--- CUSTOM RISK {risk_level}/10 STRATEGY ---")
                    try:
                        strategy_df, returns = engine.create_custom_risk_strategy(risk_level)
                        metrics = engine.calculate_performance_metrics(returns)
                        
                        strategy_name = f'Custom_Risk_{risk_level}/10'
                        all_risk_strategies_data[strategy_name] = {
                            'returns': returns,
                            'metrics': metrics,
                            'holdings': strategy_df
                        }
                        
                        engine.generate_weekly_allocation_csv(strategy_df, f'Custom_Risk_{risk_level}')
                        
                        print("Results:")
                        print(f"  Annual Return: {metrics.get('Annual Return', 0):.2%}")
                        print(f"  Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.3f}")
                        print(f"  Max Drawdown: {metrics.get('Max Drawdown', 0):.2%}")
                        print(f"  Win Rate: {metrics.get('Win Rate', 0):.2%}")
                        
                    except Exception as e:
                        print(f"Error running custom risk {risk_level} strategy: {e}")
            
            # Run custom risk strategies for different market sentiments
            if RISK_STRATEGY_SELECTION['run_custom_risk'] and CUSTOM_RISK_LEVELS and MARKET_SENTIMENTS:
                print(f"\n" + "-"*60)
                print("SENTIMENT-DRIVEN CUSTOM RISK STRATEGIES")
                print("-"*60)
                
                for sentiment in MARKET_SENTIMENTS:
                    for risk_level in CUSTOM_RISK_LEVELS:
                        print(f"\n--- CUSTOM RISK {risk_level}/10 STRATEGY ({sentiment.upper()}) ---")
                        try:
                            strategy_df, returns = engine.create_custom_risk_strategy(risk_level, market_sentiment=sentiment)
                            metrics = engine.calculate_performance_metrics(returns)
                            
                            strategy_name = f'Custom_Risk_{risk_level}_{sentiment}/10'
                            all_risk_strategies_data[strategy_name] = {
                                'returns': returns,
                                'metrics': metrics,
                                'holdings': strategy_df
                            }
                            
                            engine.generate_weekly_allocation_csv(strategy_df, f'Custom_Risk_{risk_level}_{sentiment}')
                            
                            print("Results:")
                            print(f"  Annual Return: {metrics.get('Annual Return', 0):.2%}")
                            print(f"  Sharpe Ratio: {metrics.get('Sharpe Ratio', 0):.3f}")
                            print(f"  Max Drawdown: {metrics.get('Max Drawdown', 0):.2%}")
                            print(f"  Win Rate: {metrics.get('Win Rate', 0):.2%}")
                        
                        except Exception as e:
                            print(f"Error running custom risk {risk_level} with sentiment {sentiment}: {e}")
            
            # Generate risk strategy comparison
            if all_risk_strategies_data:
                print(f"\n" + "-"*60)
                print("RISK STRATEGY COMPARISON")
                print("-"*60)
                
                # REPLACED THE SINGLE FUNCTION CALL WITH A LOOP FOR EACH RISK LEVEL
                for risk_level in CUSTOM_RISK_LEVELS:
                    engine.create_comparison_for_risk_level(risk_level, all_risk_strategies_data)

                engine.save_risk_strategy_summary(all_risk_strategies_data)
                
                # Print comparison table
                print("\nRisk Strategy Performance Comparison:")
                print("-" * 100)
                print(f"{'Strategy':<25} {'Annual Return':<12} {'Sharpe Ratio':<12} {'Max Drawdown':<12} {'Volatility':<12} {'Win Rate':<10}")
                print("-" * 100)
                
                for name, data in all_risk_strategies_data.items():
                    metrics = data['metrics']
                    print(f"{name:<25} "
                          f"{metrics.get('Annual Return', 0):>10.2%} "
                          f"{metrics.get('Sharpe Ratio', 0):>10.3f} "
                          f"{metrics.get('Max Drawdown', 0):>10.2%} "
                          f"{metrics.get('Volatility', 0):>10.2%} "
                          f"{metrics.get('Win Rate', 0):>8.2%}")
                
                print("-" * 100)
                
                # Strategy recommendations
                print(f"\nStrategy Recommendations:")
                best_sharpe = max(all_risk_strategies_data.items(), key=lambda x: x[1]['metrics'].get('Sharpe Ratio', 0))
                best_return = max(all_risk_strategies_data.items(), key=lambda x: x[1]['metrics'].get('Annual Return', 0))
                lowest_drawdown = min(all_risk_strategies_data.items(), key=lambda x: x[1]['metrics'].get('Max Drawdown', 1))
                
                print(f"  • Best Risk-Adjusted Return: {best_sharpe[0]}")
                print(f"  • Highest Annual Return: {best_return[0]}")
                print(f"  • Lowest Max Drawdown: {lowest_drawdown[0]}")
                
                print(f"\nInvestment Guide:")
                print(f"  • Conservative Investors: Choose Custom Risk 1-3")
                print(f"  • Moderate Investors: Choose Custom Risk 4-6")
                print(f"  • Aggressive Investors: Choose Custom Risk 7-10")
                print(f"  • Custom risk allows fine-tuning based on personal risk tolerance")
                
                print(f"\nSentiment-Based Guide:")
                print(f"  • Positive Market Sentiment: Consider Custom Risk strategies with 'positive' suffix for higher growth potential")
                print(f"  • Negative Market Sentiment: Consider Custom Risk strategies with 'negative' suffix for better risk control")
        
        # 4. Run single factor strategy backtests (original functionality)
        if RUN_MODES['single_factor_strategies']:
            print(f"\n" + "="*80)
            print("STEP 3: FOUR CORE FACTOR SINGLE FACTOR STRATEGY BACKTESTS")
            print("="*80)
            print(f"Parameter configuration:")
            for key, value in SINGLE_FACTOR_PARAMS.items():
                print(f"  - {key}: {value}")
            
            # Run all single factor strategies with enhanced weighting
            performance_metrics = engine.run_all_single_factor_strategies(
                SINGLE_FACTOR_PARAMS['top_n_assets'],
                WEIGHTING_SCHEMES['single_factor_weighting']
            )
            
            if performance_metrics:
                # Select top N strategies based on ranking metric
                top_strategies_for_analysis = engine.get_top_strategies(
                    SINGLE_FACTOR_PARAMS['top_n_strategies'], 
                    SINGLE_FACTOR_PARAMS['ranking_metric']
                )
                
                if top_strategies_for_analysis:
                    # Strategy comparison analysis
                    engine.compare_strategies(top_strategies_for_analysis)
                    
                    # Best strategy summary
                    print(f"\nBest single factor strategy analysis:")
                    best_strategy = top_strategies_for_analysis[0]
                    best_metrics = engine.performance_metrics[best_strategy]
                    
                    print(f"Best strategy: {best_strategy}")
                    print(f"   Annual return: {best_metrics.get('Annual Return', 0):.2%}")
                    print(f"   Sharpe ratio: {best_metrics.get('Sharpe Ratio', 0):.3f}")
                    print(f"   Max drawdown: {best_metrics.get('Max Drawdown', 0):.2%}")
                    print(f"   Win rate: {best_metrics.get('Win Rate', 0):.2%}")
                    
                    # Factor category performance analysis
                    print(f"\nFactor category performance analysis:")
                    factor_performance = {}
                    for strategy in top_strategies_for_analysis:
                        factor_type = engine.get_factor_type(strategy)
                        if factor_type not in factor_performance:
                            factor_performance[factor_type] = []
                        factor_performance[factor_type].append(engine.performance_metrics[strategy]['Sharpe Ratio'])
                    
                    for factor_type, sharpe_ratios in factor_performance.items():
                        avg_sharpe = np.mean(sharpe_ratios)
                        print(f"   {factor_type}: Average Sharpe ratio {avg_sharpe:.3f} ({len(sharpe_ratios)} strategies)")
                else:
                    print("No valid strategies found for single factor analysis.")
            else:
                print("Single factor strategy backtest failed or no metrics generated.")
        else:
            print("\nSkipping single factor strategy backtests as disabled.")
            top_strategies_for_analysis = []

        # 5. Run multi-factor portfolio (if enabled)
        if RUN_MODES['multi_factor_portfolio'] and top_strategies_for_analysis:
            print(f"\n" + "="*80)
            print("STEP 4: GENERATE MULTI-FACTOR PORTFOLIO FROM TOP SINGLE STRATEGIES")
            print("="*80)
            print(f"Parameter configuration:")
            print(f"  - top_n_strategies_for_mf: {MULTI_FACTOR_PARAMS['top_n_strategies_for_mf']}")
            print(f"  - strategy_weights: {MULTI_FACTOR_PARAMS['strategy_weights']}")

            # Get the actual top N strategies based on the ranking metric
            top_strategies_for_mf_portfolio = engine.get_top_strategies(
                MULTI_FACTOR_PARAMS['top_n_strategies_for_mf'], 
                SINGLE_FACTOR_PARAMS['ranking_metric']
            )

            if top_strategies_for_mf_portfolio:
                # Generate multi-factor portfolio returns from these top strategies
                mf_returns = engine.generate_multi_factor_portfolio_from_strategies(
                    strategy_names=top_strategies_for_mf_portfolio,
                    weights=MULTI_FACTOR_PARAMS['strategy_weights']
                )
                
                if len(mf_returns) > 0:
                    mf_metrics = engine.calculate_performance_metrics(mf_returns)
                    print(f"\nMulti-factor portfolio performance:")
                    print(f"  - Annual Return: {mf_metrics.get('Annual Return', 0):.2%}")
                    print(f"  - Sharpe Ratio: {mf_metrics.get('Sharpe Ratio', 0):.3f}")
                    print(f"  - Max Drawdown: {mf_metrics.get('Max Drawdown', 0):.2%}")
                    print(f"  - Win Rate: {mf_metrics.get('Win Rate', 0):.2%}")
                else:
                    print("Multi-factor portfolio returns could not be calculated.")
            else:
                print("Could not determine top strategies for multi-factor portfolio construction.")
            
            # Save results
            if MULTI_FACTOR_PARAMS['save_output']:
                engine.save_results(MULTI_FACTOR_PARAMS['output_path'])
        elif not RUN_MODES['multi_factor_portfolio']:
            print("\nSkipping multi-factor portfolio generation as disabled.")
        else:
            print("\nCannot generate multi-factor portfolio: No single factor strategies available or found.")
        
        # 6. Generate visualization charts (if enabled)
        if RUN_MODES['generate_visualizations'] and VISUALIZATION_PARAMS['generate_all_charts']:
            print(f"\n" + "="*80)
            print("STEP 5: GENERATE VISUALIZATION CHARTS FOR SINGLE-FACTOR STRATEGIES")
            print("="*80)
            
            if top_strategies_for_analysis:
                engine.generate_all_charts(top_strategies_for_analysis)
            else:
                print("No single factor strategies available to generate comparison charts.")
        
        # 7. Save detailed results (if enabled)
        if RUN_MODES['save_detailed_results']:
            print(f"\n" + "="*80)
            print("STEP 6: SAVE DETAILED RESULTS")
            print("="*80)
            
            if top_strategies_for_analysis:
                engine.save_strategy_results(top_strategies_for_analysis)
            else:
                print("No single factor strategies available to save detailed results.")
            
            print(f"\nDetailed file structure:")
            print(f"   CSV data folder: {engine.csv_folder}")
            print(f"   Charts folder: {engine.charts_folder}")
            print(f"   Weekly allocations folder: {engine.weekly_allocations_folder}")
            print(f"   Strategy weekly holdings folder: {os.path.join(engine.base_folder, 'strategy_weekly_holdings')}")
            print(f"   ├── 00_strategies_overview.csv (Strategy overview)")
            print(f"   ├── weekly_summary_all_strategies.csv (Weekly summary)")
            print(f"   └── rank_XX_[strategy_name]_weekly_holdings.csv (Detailed holdings for each strategy)")
        
        # 8. Investment recommendations
        print(f"\n" + "="*80)
        print("INVESTMENT RECOMMENDATIONS")
        print("="*80)
        print(f"Based on four core factors:")
        print(f"   1. Momentum factors suitable for trend following, perform well in bull markets")
        print(f"   2. Volatility factors help with risk control, suitable for volatile markets")
        print(f"   3. Volume impact factors reflect market liquidity, suitable for short-term trading")
        print(f"   4. Short-term reversal factors capture oversold rebounds, suitable for contrarian investing")
        print(f"   5. Recommend dynamically adjusting factor weights based on market conditions")
        print(f"   6. Regularly rebalance portfolio and control risk exposure")
        
        print(f"\nNEW: Risk-based strategy selection:")
        print(f"   • For conservative investors: Choose Custom Risk 1-3")
        print(f"   • For moderate investors: Choose Custom Risk 4-6")
        print(f"   • For aggressive Investors: Choose Custom Risk 7-10")
        print(f"   • Custom risk allows fine-tuning based on personal risk tolerance")
        print(f"   • Consider using sentiment-driven strategies for dynamic market conditions")
        
        # NEW: 9. Generate FINAL Comprehensive Comparison Chart
        print(f"\n" + "="*80)
        print("STEP 7: GENERATE FINAL COMPREHENSIVE COMPARISON CHART")
        print("="*80)
        engine.plot_all_portfolios_comparison()

        # 10. System summary
        print(f"\n" + "="*80)
        print("SYSTEM RUN SUMMARY")
        print("="*80)
        
        # Risk-based strategies summary
        if all_risk_strategies_data:
            print(f"Custom Risk-based strategies: {len(all_risk_strategies_data)} strategies")
            for name, data in all_risk_strategies_data.items():
                metrics = data['metrics']
                print(f"   {name}: Annual Return {metrics.get('Annual Return', 0):.2%}, Sharpe {metrics.get('Sharpe Ratio', 0):.3f}")
        
        # Multi-factor portfolio summary
        if RUN_MODES['multi_factor_portfolio'] and engine.multi_factor_returns is not None and len(engine.multi_factor_returns) > 0:
            mf_metrics = engine.calculate_performance_metrics(engine.multi_factor_returns)
            print(f"Multi-factor portfolio (from top strategies): Annual Return {mf_metrics.get('Annual Return', 0):.2%}, Sharpe {mf_metrics.get('Sharpe Ratio', 0):.3f}")
        else:
            print("Multi-factor portfolio: Not generated or no valid returns.")

        # Single factor strategies summary
        if RUN_MODES['single_factor_strategies'] and engine.performance_metrics:
            print(f"Single factor strategies: {len(engine.performance_metrics)} strategies")
            
            # Count each factor type
            momentum_count = sum(1 for k in engine.performance_metrics.keys() if 'momentum' in k)
            volatility_count = sum(1 for k in engine.performance_metrics.keys() if 'volatility' in k)
            volume_count = sum(1 for k in engine.performance_metrics.keys() if 'volume' in k)
            reversal_count = sum(1 for k in engine.performance_metrics.keys() if 'strev' in k)
            
            print(f"     - Momentum factor strategies: {momentum_count}")
            print(f"     - Volatility factor strategies: {volatility_count}")
            print(f"     - Volume impact factor strategies: {volume_count} (deduplicated)")
            print(f"     - Short-term reversal factor strategies: {reversal_count}")
            
        if RUN_MODES['generate_visualizations']:
            print(f"Visualization charts: 7+ types of professional charts (including comprehensive comparison)")
        if RUN_MODES['save_detailed_results']:
            print(f"Results saved: CSV files and charts saved")

        print(f"\nAll results saved to:")
        print(f"   CSV data: {engine.csv_folder}")
        print(f"   Charts: {engine.charts_folder}")
        print(f"   Weekly allocations: {engine.weekly_allocations_folder}")

        print(f"\n" + "="*80)
        print("ENHANCED FOUR CORE FACTOR STRATEGY ANALYSIS SYSTEM COMPLETED!")
        print("="*80)
        
    except FileNotFoundError as e:
        if 'sentiment_data' in str(e):
             print(f"Error: Sentiment data file '{SENTIMENT_DATA_PATH}' not found")
        else:
             print(f"Error: Data file '{DATA_PATH}' not found")
        print("Please ensure the CSV file is in the correct path")
    except Exception as e:
        print(f"Runtime error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()