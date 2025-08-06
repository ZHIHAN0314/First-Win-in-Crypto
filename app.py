# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Wedge, Circle
import time
import random
from streamlit_option_menu import option_menu
import base64
import plotly.express as px  # 导入plotly.express


# Attempt to import modules, providing a clear error message on failure
try:
    from crypto_strategy_engine import CryptoStrategyEngine
except ImportError:
    st.error("Error: Could not import 'crypto_strategy_engine'. Please ensure 'crypto_strategy_engine.py' and other dependency files are in the same folder as 'app.py'.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="First Win in Crypto",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS for a Professional UI ---
def load_css():
    """Loads custom CSS for styling the Streamlit app."""
    st.markdown(f"""
    <style>
        /* --- Main App Styling --- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        html, body, [class*="st-"] {{
            font-family: 'Inter', sans-serif;
            color: #E0E0E0;
        }}

        .stApp {{
            background-color: #121212;
        }}
        .block-container {{
            padding-top: 1rem !important;
            padding-bottom: 2rem !important;
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            color: #FFFFFF;
        }}
        
        .stMarkdown, .stText, .stAlert {{
            color: #E0E0E0;
        }}

        /* --- Header Styling --- */
        [data-testid="stHeader"] {{ display: none !important; }}
        
        /* --- Custom Sticky Header and Buttons --- */
        .sticky-header {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 9999;
            background-color: #121212;
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #333333;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}
        .header-title-container {{
            display: flex;
            align-items: center;
            gap: 1.5rem;
        }}
        .header-title {{
            font-size: 2.5rem;
            font-weight: 700;
        }}
        .header-title .title-word-orange {{ color: #FF9900; }}
        .header-title .title-word-purple {{ color: #BB86FC; }}
        
        /* FIX: Welcome message and button alignment */
        .welcome-message-container {{
            display: flex;
            align-items: center;
            justify-content: flex-end;
            height: 40px; /* Match button height */
        }}
        .welcome-message {{
            font-size: 1rem;
            color: #E0E0E0;
            white-space: nowrap;
            margin: 0;
            padding-right: 15px;
        }}
        
        /* Adjust main content area to prevent being hidden by sticky header */
        .main-content-padding {{
            padding-top: 8rem;
        }}
        
        /* ================================================================== */
        /* ==               FINAL CSS FIXES FOR WIDGETS                  == */
        /* ================================================================== */
        
        /* Fix for Dropdown Menu OPTIONS LIST */
        div[data-baseweb="popover"] ul li,
        [data-testid="stVirtualDropdown"] li,
        div[data-baseweb="popover"] ul li div,
        [data-testid="stVirtualDropdown"] li div,
        div[data-baseweb="popover"] ul li span,
        [data-testid="stVirtualDropdown"] li span {{
            color: #BB86FC !important; /* 紫色 */
            background-color: white !important;
        }}
        
        /* Fix for Hover/Focus state in Dropdown */
        div[data-baseweb="popover"] ul li:hover,
        [data-testid="stVirtualDropdown"] li:hover {{
            background-color: #f0f2f6 !important;
        }}
        
        /* Fix for the SELECTED ITEM display in Selectbox */
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
            background-color: transparent; 
        }}
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div > div {{
             color: #BB86FC !important; 
        }}

        /* Style for the main Selectbox container to match buttons */
        [data-testid="stSelectbox"] {{
            border-radius: 8px !important;
            border: 2px solid #BB86FC !important;
            background-color: transparent !important;
        }}

        /* Style for the main MultiSelect container (新增的代码) */
        [data-testid="stMultiSelect"] {{
            border-radius: 8px !important;
            border: 2px solid #BB86FC !important;
            background-color: transparent !important;
            padding: 0.1em 0.6em;
        }}
        
        /* Fix for Link Button */
        .stLinkButton a {{
            border-radius: 8px !important;
            border: 2px solid #BB86FC !important;
            background-color: transparent !important;
            color: #BB86FC !important;
            transition: all 0.3s !important;
            font-weight: 600 !important;
            width: 100% !important;
            padding: 0.75em 1em !important;
        }}
        .stLinkButton a:hover {{
            background-color: #BB86FC !important;
            color: #121212 !important;
        }}
        /* ================================================================== */

        /* --- Metric Cards --- */
        .metric-card {{
            background-color: #1E1E1E;
            border-radius: 10px; 
            padding: 20px;
            text-align: center; 
            border: 1px solid #333333;
            transition: all 0.3s ease-in-out;
            height: 100%;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 20px rgba(187, 134, 252, 0.3);
            border-color: #BB86FC;
        }}
        .metric-card-title {{ font-size: 1rem; color: #BBBBBB; margin-bottom: 10px; font-weight: 600; }}
        .metric-card-value {{ font-size: 2rem; font-weight: 700; color: #FFFFFF; }}
        
        /* --- Custom Buttons and Widgets --- */
        .stButton>button {{
            border-radius: 8px; 
            border: 2px solid #BB86FC;
            background-color: transparent; 
            color: #BB86FC; 
            transition: all 0.3s;
            font-weight: 600;
            width: 100%;
            padding: 0.75em 1em;
        }}
        .stButton>button:hover {{
            background-color: #BB86FC; 
            color: #121212; 
            border-color: #BB86FC;
        }}
        .stButton>button:focus {{
            box-shadow: 0 0 0 2px #BB86FC !important;
        }}
        
        /* --- Auth Page Specific Styles --- */
        .auth-container {{
            max-width: 450px;
            margin: 5rem auto;
            padding: 2rem;
            background-color: #1E1E1E;
            border-radius: 10px;
            border: 1px solid #333333;
        }}
        
        /* --- Main Navigation Styling --- */
        div[data-testid="stHorizontalBlock"] .st-emotion-cache-1f19z1c {{
            background-color: transparent !important;
            border: 2px solid #FF9900;
            border-radius: 8px;
            padding: 5px;
        }}
        div[data-testid="stHorizontalBlock"] .st-emotion-cache-1f19z1c a {{
            background-color: transparent !important;
            color: #FFFFFF !important;
        }}
        div[data-testid="stHorizontalBlock"] .st-emotion-cache-1f19z1c a[aria-selected="true"] {{
            background-color: #BB86FC !important;
            border-radius: 6px;
        }}
        
        /* Dataframe styling */
        .stDataFrame table {{ background-color: #1E1E1E !important; color: #E0E0E0 !important; border-color: #333333 !important; }}
        .stDataFrame thead tr {{ background-color: #121212 !important; color: #FFFFFF !important; }}
        .stDataFrame tbody tr:hover {{ background-color: #262730 !important; }}
        .stDataFrame th, .stDataFrame td {{ border-color: #333333 !important; }}
        
        /* Footer disclaimer styling */
        .footer-disclaimer {{ font-size: 0.85rem; color: #AAAAAA; text-align: center; padding-top: 2rem; border-top: 1px solid #333333; margin-top: 2rem; }}
    </style>
    """, unsafe_allow_html=True)

# --- UI Helper Functions ---
def create_sentiment_gauge(score, title):
    normalized_score = (score + 1) * 50
    cmap = mcolors.LinearSegmentedColormap.from_list("sentiment_cmap", ["#B71C1C", "#FFEB3B", "#4CAF50"])
    
    fig, ax = plt.subplots(figsize=(3, 2))
    fig.patch.set_facecolor('#121212') 
    ax.set_facecolor('#121212')

    for i in range(100):
        color = cmap(i / 100.0)
        ax.add_patch(Wedge((0.5, 0.35), 0.4, i*1.8, (i+1)*1.8, facecolor=color, width=0.15))

    ax.add_patch(Circle((0.5, 0.35), 0.25, facecolor='#1E1E1E', zorder=10))

    angle = 180 - normalized_score * 1.8
    rad = np.deg2rad(angle)
    
    arrow_end_x = 0.5 + 0.38 * np.cos(rad)
    arrow_end_y = 0.35 + 0.38 * np.sin(rad)
    ax.arrow(0.5, 0.35, arrow_end_x - 0.5, arrow_end_y - 0.35,
              width=0.01, head_width=0.0, head_length=0.0,
              fc='#FFFFFF', ec='#FFFFFF', zorder=11)
    ax.add_patch(Circle((0.5, 0.35), 0.03, facecolor='#FFFFFF', zorder=12))

    ax.text(0.5, 0.45, f"{normalized_score:.1f}", ha='center', va='center', fontsize=18, color='#FFFFFF', weight='bold', zorder=11)
    ax.text(0.5, 0.2, title, ha='center', va='center', fontsize=9, color='#BBBBBB', zorder=11)
    ax.text(0.1, 0.1, "Fear", ha='center', va='center', fontsize=8, color='#E0E0E0', weight='semibold')
    ax.text(0.9, 0.1, "Greed", ha='center', va='center', fontsize=8, color='#E0E0E0', weight='semibold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.8)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    
    return fig

def metric_card(title, value, value_class=""):
    return f"""
    <div class="metric-card">
        <div class="metric-card-title">{title}</div>
        <div class="metric-card-value {value_class}">{value}</div>
    </div>
    """

# --- Constants & File Paths ---
FINAL_DATA_PATH = "crypto_news_weekly_top.csv"
SENTIMENT_DATA_PATH = "aggregated_weekly.csv"
CHARTS_PATH = Path("Project_result") / "charts"

# --- Caching & Data Loading ---
@st.cache_data(ttl=86400)
def run_full_data_pipeline():
    final_path = Path(FINAL_DATA_PATH)
    sentiment_path = Path(SENTIMENT_DATA_PATH)

    if not final_path.exists():
        st.error(f"Data file not found at expected path: '{FINAL_DATA_PATH}'.")
        st.stop()
    master_df = pd.read_csv(final_path, parse_dates=['date'])
    
    if not sentiment_path.exists():
        sentiment_df = pd.DataFrame()
    else:
        sentiment_df = pd.read_csv(sentiment_path, parse_dates=['period'])
    
    return master_df, sentiment_df

# --- Authentication and Page Functions ---

def page_home():
    """Renders the Home page."""
    st.markdown("""
        <h1 style='text-align: center; padding-top: 2rem;'>
            Welcome to your 
            <span style='color: #FF9900;'>First</span> 
            <span style='color: #BB86FC;'>Win</span> 
            <span style='color: #FF9900;'>in Crypto</span>
        </h1>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #AAAAAA;'>Your Multi-Factor Driven Investment Co-Pilot</p>", unsafe_allow_html=True)
    
    st.markdown("<p style='text-align: center; font-size: 0.9rem; color: #888888;'><i>Switch to dark mode for a better viewing experience.</i></p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="text-align: center; font-size: 1.1rem; padding: 1rem 3rem 2rem 3rem;">
        This platform leverages sophisticated <strong>multi-factor models</strong> to analyze market trends and provide data-driven investment suggestions in the volatile world of cryptocurrency.
        <br><br>
        Navigate using the menu above to explore backtested strategies, view current market sentiment, and discover portfolio recommendations tailored to your risk profile.
    </div>
    """, unsafe_allow_html=True)
    
def page_login():
    """Renders the Login page."""
    with st.container():
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.header("Login")
        with st.form("login_form"):
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if username in st.session_state.user_db and st.session_state.user_db[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.current_page = "Dashboard" # Redirect to dashboard
                    st.success("Logged in successfully!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
        
        if st.button("Back to Home", key="login_back"):
            st.session_state.current_page = "Home"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

def page_register():
    """Renders the Registration page."""
    with st.container():
        st.markdown('<div class="auth-container">', unsafe_allow_html=True)
        st.header("Register")
        with st.form("register_form"):
            new_username = st.text_input("Choose a Username", key="reg_user")
            new_password = st.text_input("Choose a Password", type="password", key="reg_pass")
            submitted = st.form_submit_button("Register")

            if submitted:
                if not new_username or not new_password:
                    st.warning("Username and password cannot be empty.")
                elif new_username in st.session_state.user_db:
                    st.error("Username already exists. Please choose another one.")
                else:
                    st.session_state.user_db[new_username] = new_password
                    st.success("Registration successful! Please log in.")
                    st.session_state.current_page = "Login"
                    time.sleep(2)
                    st.rerun()

        if st.button("Back to Home", key="register_back"):
            st.session_state.current_page = "Home"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- CORE App Page Rendering Functions ---
def page_dashboard(engine, sentiment_df):
    st.header("Market Overview & Sentiment Analysis")
    all_symbols = sorted(engine.data['symbol'].unique())
    selected_symbols = st.multiselect(
        "Select cryptocurrencies to compare:",
        options=all_symbols,
        default=['ETH', 'BTC'] if all(s in all_symbols for s in ['ETH', 'BTC']) else ([all_symbols[0]] if all_symbols else [])
    )

    if selected_symbols:
        st.subheader("Price History (Normalized) & Market Sentiment")
        chart_col, sentiment_col = st.columns([2, 1])

        with chart_col:
            # --- FINAL DATA LOGIC FIX STARTS HERE ---
            comparison_df = engine.data[engine.data['symbol'].isin(selected_symbols)]
            price_pivot = comparison_df.pivot(index='date', columns='symbol', values='close')

            # Create an empty DataFrame to store normalized prices
            normalized_prices = pd.DataFrame(index=price_pivot.index)

            # Iterate through each selected symbol to normalize it individually
            for symbol in price_pivot.columns:
                # Find the first valid (non-NaN) price for this specific symbol
                first_valid_index = price_pivot[symbol].first_valid_index()
                if first_valid_index is not None:
                    first_price = price_pivot.loc[first_valid_index, symbol]
                    # Normalize this symbol's series by its own first price
                    normalized_prices[symbol] = price_pivot[symbol] / first_price * 100

            # Reset index to make 'date' a column for Plotly
            df_to_plot = normalized_prices.reset_index()

            # Melt the DataFrame from wide to long format for Plotly
            df_melted = df_to_plot.melt(
                id_vars='date',
                value_vars=normalized_prices.columns,
                var_name='Symbol',
                value_name='Normalized Price'
            )

            # Create the interactive plot
            fig = px.line(
                df_melted,
                x='date',
                y='Normalized Price',
                color='Symbol',
                title="Price History (Normalized)",
                labels={'Normalized Price': 'Normalized Price (Basis 100)', 'date': 'Date'}
            )

            # Update layout for our dark theme
            fig.update_layout(
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#121212',
                font_color='white',
                xaxis=dict(gridcolor='#444444'),
                yaxis=dict(gridcolor='#444444'),
                legend_title_text='Symbols'
            )

            # Enable x-axis range slider
            fig.update_xaxes(rangeslider_visible=True)

            st.plotly_chart(fig, use_container_width=True)
            # --- FINAL DATA LOGIC FIX ENDS HERE ---

        with sentiment_col:
            st.markdown("<br/>", unsafe_allow_html=True)
            if not sentiment_df.empty:
                latest_overall_sentiment = sentiment_df.sort_values('period', ascending=False).iloc[0].get('market_avg_compound', 0)
                gauge_fig = create_sentiment_gauge(latest_overall_sentiment, "Overall Market")
                st.pyplot(gauge_fig)
            else:
                st.warning("Overall market sentiment data is not available.")

        st.markdown("---")
        st.subheader("Individual Crypto Sentiment Gauges")
        # ... (rest of the function remains the same)
        gauge_count = len(selected_symbols)
        gauges_per_row = 4
        for i in range(gauge_count):
            if i % gauges_per_row == 0:
                cols = st.columns(gauges_per_row)
            symbol = selected_symbols[i]
            with cols[i % gauges_per_row]:
                st.markdown(f"#### {symbol} Sentiment")
                symbol_df = engine.data[engine.data['symbol'] == symbol].copy().sort_values('date', ascending=False)
                if not symbol_df.empty and 'avg_compound' in symbol_df.columns:
                    latest_sentiment_score = symbol_df.iloc[0].get('avg_compound', 0)
                    gauge_fig = create_sentiment_gauge(latest_sentiment_score, f"{symbol} Sentiment")
                    st.pyplot(gauge_fig)
                else:
                    st.warning(f"No sentiment data available for {symbol}.")
    else:
        st.info("Please select at least one currency to display charts.")

def page_strategy_studio():
    st.header("Strategy Backtest Studio Report")
    st.markdown("This page displays pre-generated strategy analysis charts. Run the backend script to generate the latest charts.")

    charts_to_display = {
        "Top Single Factor Strategies vs Benchmarks: Cumulative Returns": "https://res.cloudinary.com/doiniwmm5/image/upload/v1754314887/cumulative_returns_yfxbse.png",
        "Top Single Factor vs Multi-Factor: Drawdown Analysis": "https://res.cloudinary.com/doiniwmm5/image/upload/v1754314892/drawdown_analysis_ixo8gi.png",
        "Risk-Based Strategy Performance Comparison": "https://res.cloudinary.com/doiniwmm5/image/upload/v1754314890/rolling_sharpe_dashboard_chzutj.png",
        "Top Single Factor Strategies: Performance Metrics Radar Chart": "https://res.cloudinary.com/doiniwmm5/image/upload/v1754314888/performance_radar_m1jirp.png",
        "Top Single Factor vs Multi-Factor: Rolling Sharpe & Volatility": "https://res.cloudinary.com/doiniwmm5/image/upload/v1754314890/rolling_sharpe_dashboard_chzutj.png",
        "Top Single Factor Strategies: Monthly Return Candles": "https://res.cloudinary.com/doiniwmm5/image/upload/v1754316227/monthly_returns_candles_c3gaha.png",
    }

    charts_found = 0
    for title, chart_path in charts_to_display.items():
        st.subheader(title)
        st.image(chart_path, use_container_width=True)
        charts_found += 1
            
    if charts_found == 0:
        st.warning("No charts found. Please check the URLs or generate local charts.")

def page_current_portfolio(engine):
    st.header("This Week's Recommended Portfolio")
    st.markdown("Select your desired risk level and market sentiment to see a dynamically generated portfolio recommendation.")
    
    st.link_button("Start Deploying Investment", "https://www.binance.com/en/trade/BTC_USDT", use_container_width=True, type="secondary")
    st.markdown("---") 

    col_risk, col_sentiment = st.columns(2)
    with col_risk:
        risk_tolerance = st.slider("Your Risk Tolerance (0=Conservative, 10=Aggressive)", 0, 10, 5, key="risk_slider")
    with col_sentiment:
        sentiment_options = {"positive": "Optimistic", "neutral": "Neutral", "negative": "Pessimistic"}
        market_sentiment = st.selectbox("Market Sentiment", options=list(sentiment_options.keys()), format_func=lambda x: sentiment_options[x], key="sentiment_selectbox")
        market_sentiment_label = sentiment_options[market_sentiment]

    with st.spinner(f"Backtesting and calculating portfolio..."):
        try:
            strategy_df, returns = engine.create_custom_risk_strategy(risk_tolerance=risk_tolerance, market_sentiment=market_sentiment)
            metrics = engine.calculate_performance_metrics(returns)
        except Exception as e:
            st.error(f"Could not generate portfolio. Error: {e}")
            return
            
    st.subheader(f"Portfolio Profile: Custom Risk {risk_tolerance}/10 ({market_sentiment_label} Market)")
    m_col1, m_col2, m_col3 = st.columns(3)
    sharpe_val = metrics.get('Sharpe Ratio', 0)
    sharpe_display = f"{sharpe_val:.3f}" if sharpe_val < 10 else "> 10.0"

    with m_col1: st.markdown(metric_card("Historical Weekly Return", f"{metrics.get('Avg Return', 0):.2%}"), unsafe_allow_html=True)
    with m_col2: st.markdown(metric_card("Historical Volatility", f"{metrics.get('Volatility', 0):.2%}"), unsafe_allow_html=True)
    with m_col3: st.markdown(metric_card("Historical Sharpe Ratio", sharpe_display), unsafe_allow_html=True)

    st.subheader("Recommended Holdings for This Week")
    if not strategy_df.empty:
        latest_date = strategy_df['date'].max()
        portfolio_df = strategy_df[strategy_df['date'] == latest_date][['symbol', 'weight']].sort_values('weight', ascending=False).reset_index(drop=True)
        st.dataframe(portfolio_df.style.format({'weight': '{:.2%}'}).background_gradient(cmap='viridis', subset=['weight']), use_container_width=True)
    else:
        st.warning("Could not determine this week's recommended holdings.")
    
    st.markdown("---")
    st.subheader("Backtest Performance")
    if not returns.empty:
        cumulative_returns = (1 + returns).cumprod()
        fig_cum, ax_cum = plt.subplots(figsize=(12, 6))
        fig_cum.patch.set_facecolor('#121212'); ax_cum.set_facecolor('#121212')
        ax_cum.plot(cumulative_returns.index, cumulative_returns.values, label="Portfolio", color='#BB86FC')
        if engine.btc_cumulative_returns is not None:
            ax_cum.plot(engine.btc_cumulative_returns.index, engine.btc_cumulative_returns.values, label='BTC Benchmark', color='orange', linestyle='--')
        ax_cum.set_title("Cumulative Returns", color='white'); ax_cum.set_xlabel("Date", color='white'); ax_cum.set_ylabel("Cumulative Return", color='white')
        ax_cum.legend(); ax_cum.grid(True, alpha=0.3); ax_cum.tick_params(colors='white')
        st.pyplot(fig_cum)
        
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        fig_dd, ax_dd = plt.subplots(figsize=(12, 6))
        fig_dd.patch.set_facecolor('#121212'); ax_dd.set_facecolor('#121212')
        ax_dd.fill_between(drawdown.index, drawdown.values, color='#BB86FC', alpha=0.7)
        ax_dd.set_title("Max Drawdown", color='white'); ax_dd.set_xlabel("Date", color='white'); ax_dd.set_ylabel("Drawdown", color='white')
        ax_dd.grid(True, alpha=0.3); ax_dd.tick_params(colors='white')
        st.pyplot(fig_dd)

    st.markdown("---")
    st.link_button("Start Deploying Investment", "https://www.binance.com/en/trade/BTC_USDT", use_container_width=True, type="secondary")

# --- Main Application View Controller ---
def main_app_view(engine, sentiment_df):
    """Renders the main application view based on login state and page selection."""

    # --- STICKY HEADER ---
    st.markdown('<div class="sticky-header">', unsafe_allow_html=True)
    title_col, button_col = st.columns([0.6, 0.4])
    with title_col:
        st.markdown(
            f"""
            <div class="header-title-container">
                <div class="header-title">
                    <span class="title-word-orange">First</span> 
                    <span class="title-word-purple">Win</span> 
                    <span class="title-word-orange">in Crypto</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with button_col:
        if not st.session_state.logged_in:
            l_col, r_col = st.columns(2)
            with l_col:
                if st.button("Login", key="header_login", use_container_width=True):
                    st.session_state.current_page = "Login"
                    st.rerun()
            with r_col:
                if st.button("Register", key="header_register", use_container_width=True):
                    st.session_state.current_page = "Register"
                    st.rerun()
        else:
            # FIX: Use st.columns and custom CSS to correctly align welcome message and logout button
            welcome_col, logout_col = st.columns([0.7, 0.3])
            with welcome_col:
                st.markdown(f'<div class="welcome-message-container"><p class="welcome-message">Welcome, {st.session_state.username}</p></div>', unsafe_allow_html=True)
            with logout_col:
                if st.button("Logout", key="logout_button", use_container_width=True):
                    st.session_state.logged_in = False
                    st.session_state.username = None
                    st.session_state.current_page = "Home" # Return to home on logout
                    st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="main-content-padding"></div>', unsafe_allow_html=True)

    # --- PAGE CONTENT ROUTER  ---
    # Handle special pages (Login/Register) first
    if st.session_state.current_page in ["Login", "Register"]:
        if st.session_state.current_page == "Login":
            page_login()
        elif st.session_state.current_page == "Register":
            page_register()
    else:
        options = ["Home", "Dashboard", "Strategy Studio", "This Week's Portfolio"]
        
        try:
            default_idx = options.index(st.session_state.current_page)
        except ValueError:
            default_idx = 0
            st.session_state.current_page = "Home"

        selected_page = option_menu(
            menu_title=None, 
            options=options, 
            icons=['house', 'bar-chart-line', 'tools', 'briefcase'], # 您可以添加一些图标
            default_index=default_idx,
            orientation="horizontal"
        )

        if st.session_state.current_page != selected_page:
            st.session_state.current_page = selected_page
            st.rerun()

     
        if st.session_state.current_page == "Home":
            page_home()
        elif st.session_state.current_page == "Dashboard":
            page_dashboard(engine, sentiment_df)
        elif st.session_state.current_page == "Strategy Studio":
            page_strategy_studio()
        elif st.session_state.current_page == "This Week's Portfolio":
            page_current_portfolio(engine)
        
        # Footer disclaimer
        st.markdown("---")
        st.markdown(
            """
            <div class="footer-disclaimer">
                <strong>Disclaimer:</strong> The information provided is for informational and educational purposes only and does not constitute financial advice.
            </div>
            """,
            unsafe_allow_html=True
        )

# --- Main Application Logic ---
def main():
    """Main function to run the Streamlit app."""
    load_css()
    
    # --- Initialize Session State ---
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"
    if "user_db" not in st.session_state:
        st.session_state.user_db = {"admin": "password123"}
    if "username" not in st.session_state:
        st.session_state.username = None

    try:
        @st.cache_resource
        def get_engine():
            engine = CryptoStrategyEngine(
                data_path=str(FINAL_DATA_PATH), 
                sentiment_data_path=str(SENTIMENT_DATA_PATH)
            )
            engine.load_data()
            return engine

        engine = get_engine()
        _, sentiment_df = run_full_data_pipeline()
        
        main_app_view(engine, sentiment_df)

    except FileNotFoundError as e:
        st.error(f"Error: A required data file was not found. Please check your file paths. Details: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()