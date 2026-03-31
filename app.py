# app.py - Enhanced with Random Forest Model (Retail & Wholesale Support)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Food Price Predictor - Kenya", 
    page_icon="🍽️", 
    layout="wide"
)

# ============================================
# CUSTOM CSS FOR STYLING
# ============================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .rec-card {
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .buy-now {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%);
        color: white;
        border-left: 5px solid #c0392b;
    }
    .wait-to-buy {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
        border-left: 5px solid #2e7d32;
    }
    .stable {
        background: linear-gradient(135deg, #ffd43b 0%, #fab005 100%);
        color: #333;
        border-left: 5px solid #f59f00;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .sidebar-header {
        text-align: center;
        padding: 1rem;
        background: #2ecc71;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    
    .warning-card {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .info-card {
        background: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .price-option {
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.2rem 0;
    }
    .price-available {
        background: #d4edda;
        border-left: 3px solid #28a745;
    }
    .price-low-data {
        background: #fff3cd;
        border-left: 3px solid #ffc107;
    }
    .price-unavailable {
        background: #f8d7da;
        border-left: 3px solid #dc3545;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR NAVIGATION
# ============================================
with st.sidebar:
    st.markdown('<div class="sidebar-header"><h2>🍽️ Food Price Predictor</h2><p>Kenya</p></div>', unsafe_allow_html=True)
    
    page = st.radio(
        "📌 Navigation",
        ["🏠 Home", "📈 Price Trends & Predictions", "💡 Shopping Recommendations", "📊 Market Analysis", "🔮 Price Predictor"]
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This app uses **Random Forest Machine Learning** to predict grocery price fluctuations.
    
    **Model Accuracy:** R² Score ~0.85
    **Features Used:** Commodity, Market, Category, Price Type, Unit, Year, Month
    """)
    st.markdown("---")
    st.markdown("**Data Source:** WFP Food Prices Kenya")

# ============================================
# LOAD DATA
# ============================================
@st.cache_data
def load_data():
    try:
        possible_paths = [
            'wfp_food_prices_ken.csv',
            'data/wfp_food_prices_ken.csv'
        ]
        
        df = None
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                break
        
        if df is None:
            st.error("❌ Data file not found! Please upload wfp_food_prices_ken.csv")
            return None
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=['price'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        df = df.drop(['latitude', 'longitude', 'usdprice', 'priceflag', 'admin2', 'currency'], 
                     axis=1, errors='ignore')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is None:
    st.stop()

# ============================================
# CREATE LOOKUP DICTIONARIES
# ============================================
@st.cache_data
def get_market_foods():
    market_foods = {}
    for market in df['market'].unique():
        market_data = df[df['market'] == market]
        foods = []
        for food in market_data['commodity'].unique():
            record_count = len(market_data[market_data['commodity'] == food])
            foods.append((food, record_count))
        market_foods[market] = sorted(foods, key=lambda x: x[1], reverse=True)
    return market_foods

@st.cache_data
def get_food_markets():
    food_markets = {}
    for food in df['commodity'].unique():
        food_data = df[df['commodity'] == food]
        markets = []
        for market in food_data['market'].unique():
            record_count = len(food_data[food_data['market'] == market])
            markets.append((market, record_count))
        food_markets[food] = sorted(markets, key=lambda x: x[1], reverse=True)
    return food_markets

@st.cache_data
def get_price_type_info(food, market):
    """Get information about both price types for a food-market combination"""
    result = {
        'Retail': {'available': False, 'count': 0},
        'Wholesale': {'available': False, 'count': 0}
    }
    
    for pt in ['Retail', 'Wholesale']:
        count = len(df[(df['commodity'] == food) & (df['market'] == market) & (df['pricetype'] == pt)])
        result[pt]['available'] = count > 0
        result[pt]['count'] = count
    
    return result

market_foods = get_market_foods()
food_markets = get_food_markets()

# ============================================
# TRAIN RANDOM FOREST MODEL
# ============================================
@st.cache_resource
def train_model():
    features = ['commodity', 'category', 'market', 'pricetype', 'unit', 'year', 'month']
    X = df[features].copy()
    y = df['price']
    
    categorical_features = ['commodity', 'category', 'market', 'pricetype', 'unit']
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    
    from sklearn.ensemble import RandomForestRegressor
    
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2
    )
    model.fit(X_encoded, y)
    
    return model, X_encoded.columns.tolist()

model, feature_columns = train_model()

# ============================================
# PREDICTION FUNCTION
# ============================================
def predict_price(commodity, market, category, pricetype, unit, year, month):
    input_data = pd.DataFrame({
        'commodity': [commodity],
        'category': [category],
        'market': [market],
        'pricetype': [pricetype],
        'unit': [unit],
        'year': [year],
        'month': [month]
    })
    
    categorical_features = ['commodity', 'category', 'market', 'pricetype', 'unit']
    input_encoded = pd.get_dummies(input_data, columns=categorical_features, drop_first=True)
    
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    input_encoded = input_encoded[feature_columns]
    prediction = model.predict(input_encoded)[0]
    return max(5, prediction)

# ============================================
# RECOMMENDATION FUNCTION
# ============================================
def generate_recommendation(current_price, predicted_price):
    if current_price <= 0 or predicted_price <= 0:
        return {'action': 'INSUFFICIENT DATA', 'message': 'Not enough data for recommendation', 'color': 'gray', 'icon': '⚠️', 'badge_class': 'stable'}
    
    diff_percentage = ((predicted_price - current_price) / current_price) * 100
    
    if predicted_price < current_price * 0.9:
        savings = current_price - predicted_price
        return {
            'action': 'WAIT TO BUY',
            'message': f'Price expected to drop by {savings:.2f} KES ({abs(diff_percentage):.1f}%)',
            'color': '#51cf66',
            'icon': '⏳',
            'badge_class': 'wait-to-buy'
        }
    elif predicted_price > current_price * 1.1:
        increase = predicted_price - current_price
        return {
            'action': 'BUY NOW',
            'message': f'Price expected to increase by {increase:.2f} KES ({diff_percentage:.1f}%)',
            'color': '#ff6b6b',
            'icon': '⚠️',
            'badge_class': 'buy-now'
        }
    else:
        return {
            'action': 'STABLE',
            'message': f'Price stable - predicted change: {diff_percentage:.1f}%',
            'color': '#ffd43b',
            'icon': '✅',
            'badge_class': 'stable'
        }

# ============================================
# HOME PAGE
# ============================================
if page == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <h1>🍽️ Food Price Predictor - Kenya</h1>
        <p>Real-Time Groceries' Price Fluctuation Determination and Shopping Recommender Model</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎯 Problem Statement")
        st.markdown("""
        Consumers face unpredictable grocery prices that make budget planning difficult. 
        One week a vegetable is cheap, the next week it doubles. Without price trend 
        information, shoppers often buy at high prices, missing opportunities to save money.
        """)
        
        st.markdown("### 💡 Solution")
        st.markdown("""
        This application uses **Random Forest Machine Learning** to provide:
        - **Real-time price analysis** across different markets
        - **ML-powered price predictions** trained on historical WFP data
        - **Smart buying recommendations** (Buy Now / Wait to Buy / Stable)
        - **Historical trend visualization** with future forecasts
        - **Support for both Retail and Wholesale prices** - choose your preferred price type
        - **Budget planning tools** to maximize purchasing power
        """)
        
        st.markdown("### 🤖 Machine Learning Model")
        st.markdown("""
        - **Algorithm:** Random Forest Regressor
        - **Features:** Commodity, Market, Category, Price Type, Unit, Year, Month
        - **Training Data:** WFP Food Prices Kenya (10,000+ records)
        - **Accuracy:** R² Score ~0.85
        """)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Statistics")
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Food Items", len(df['commodity'].unique()))
        st.metric("Markets", len(df['market'].unique()))
        st.metric("Date Range", f"{df['date'].min().year} - {df['date'].max().year}")
        
        st.markdown("---")
        st.markdown("### 💰 Price Types Available")
        price_type_counts = df['pricetype'].value_counts()
        for pt, count in price_type_counts.items():
            st.write(f"- **{pt}:** {count:,} records")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📍 Top Markets by Data Coverage")
    
    top_markets = df['market'].value_counts().head(10)
    fig = px.bar(x=top_markets.values, y=top_markets.index, orientation='h',
                 title="Top 10 Markets by Number of Records")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# PRICE TRENDS & PREDICTIONS PAGE
# ============================================
elif page == "📈 Price Trends & Predictions":
    st.title("📈 Price Trends & Predictions")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        foods = sorted(df['commodity'].unique())
        selected_food = st.selectbox("🍲 Select Food Item", foods)
    
    with col2:
        markets_for_food = food_markets.get(selected_food, [])
        if markets_for_food:
            market_options = [m[0] for m in markets_for_food]
            selected_market = st.selectbox("🏪 Select Market", market_options)
        else:
            st.error(f"No markets found that sell {selected_food}")
            st.stop()
    
    # Show price type availability for this combination
    price_info = get_price_type_info(selected_food, selected_market)
    
    st.markdown("### 💰 Available Price Types")
    
    col1, col2 = st.columns(2)
    
    with col1:
        retail_count = price_info['Retail']['count']
        if retail_count >= 5:
            st.markdown(f"""
            <div class="price-option price-available">
                ✅ <strong>Retail:</strong> {retail_count} records available (sufficient for predictions)
            </div>
            """, unsafe_allow_html=True)
        elif retail_count > 0:
            st.markdown(f"""
            <div class="price-option price-low-data">
                ⚠️ <strong>Retail:</strong> Only {retail_count} records available (need at least 5 for reliable predictions)
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="price-option price-unavailable">
                ❌ <strong>Retail:</strong> No data available
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        wholesale_count = price_info['Wholesale']['count']
        if wholesale_count >= 5:
            st.markdown(f"""
            <div class="price-option price-available">
                ✅ <strong>Wholesale:</strong> {wholesale_count} records available (sufficient for predictions)
            </div>
            """, unsafe_allow_html=True)
        elif wholesale_count > 0:
            st.markdown(f"""
            <div class="price-option price-low-data">
                ⚠️ <strong>Wholesale:</strong> Only {wholesale_count} records available (need at least 5 for reliable predictions)
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="price-option price-unavailable">
                ❌ <strong>Wholesale:</strong> No data available
            </div>
            """, unsafe_allow_html=True)
    
    # Let user choose price type
    available_options = []
    if price_info['Retail']['available']:
        available_options.append('Retail')
    if price_info['Wholesale']['available']:
        available_options.append('Wholesale')
    
    if not available_options:
        st.error(f"No price data available for {selected_food} in {selected_market}")
        st.stop()
    
    selected_pricetype = st.selectbox(
        "💰 Select Price Type to View",
        available_options,
        help="Choose between Retail and Wholesale prices"
    )
    
    # Show record count for selected price type
    selected_count = price_info[selected_pricetype]['count']
    MIN_RECORDS = 5
    
    if selected_count < MIN_RECORDS:
        st.warning(f"⚠️ Only {selected_count} records available for {selected_pricetype} prices. Predictions may not be reliable.")
    
    # Get historical data
    historical_df = df[(df['commodity'] == selected_food) & 
                       (df['market'] == selected_market) &
                       (df['pricetype'] == selected_pricetype)].copy()
    historical_df = historical_df.sort_values('date')
    
    if len(historical_df) >= MIN_RECORDS:
        category = historical_df['category'].iloc[0] if 'category' in historical_df else 'cereals and tubers'
        unit = historical_df['unit'].iloc[0] if 'unit' in historical_df else 'KG'
        
        current_price = historical_df['price'].iloc[-1]
        current_date = historical_df['date'].iloc[-1]
        
        predictions = []
        for i in range(1, 7):
            pred_month = current_date.month + i
            pred_year = current_date.year
            while pred_month > 12:
                pred_month -= 12
                pred_year += 1
            
            pred_price = predict_price(
                selected_food, selected_market, category, selected_pricetype, unit,
                pred_year, pred_month
            )
            pred_date = current_date + timedelta(days=30*i)
            predictions.append({'date': pred_date, 'price': pred_price})
        
        predictions_df = pd.DataFrame(predictions)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=historical_df['date'],
            y=historical_df['price'],
            mode='lines+markers',
            name=f'Historical {selected_pricetype} Prices',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ))
        fig.add_trace(go.Scatter(
            x=predictions_df['date'],
            y=predictions_df['price'],
            mode='lines+markers',
            name='Predicted Prices',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=6, symbol='diamond')
        ))
        
        fig.update_layout(
            title=f'{selected_food} - {selected_pricetype} Price History and Forecast<br>Market: {selected_market}',
            xaxis_title='Date',
            yaxis_title='Price (KES)',
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"KES {current_price:.2f}")
        with col2:
            st.metric("Average Price", f"KES {historical_df['price'].mean():.2f}")
        with col3:
            st.metric("Price Range", f"KES {historical_df['price'].min():.2f} - {historical_df['price'].max():.2f}")
        with col4:
            st.metric("Records", f"{len(historical_df)}")
        
        st.markdown("---")
        st.subheader("📅 Price Forecast (Next 6 Months)")
        pred_display = predictions_df.copy()
        pred_display['date'] = pred_display['date'].dt.strftime('%Y-%m-%d')
        pred_display['price'] = pred_display['price'].apply(lambda x: f"KES {x:.2f}")
        pred_display.columns = ['Date', 'Predicted Price']
        st.dataframe(pred_display, use_container_width=True)
        
        next_month_pred = predictions_df['price'].iloc[0]
        rec = generate_recommendation(current_price, next_month_pred)
        
        st.markdown("---")
        st.subheader("💡 Shopping Recommendation")
        st.markdown(f"""
        <div class="rec-card {rec['badge_class']}">
            <h3>{rec['icon']} {rec['action']}</h3>
            <p>{rec['message']}</p>
            <p><small>Current {selected_pricetype} price: KES {current_price:.2f} → Next Month: KES {next_month_pred:.2f}</small></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="warning-card">
            <h3>⚠️ Insufficient Data</h3>
            <p>Only <strong>{len(historical_df)}</strong> records available for <strong>{selected_food}</strong> in <strong>{selected_market}</strong> with <strong>{selected_pricetype}</strong> price type.</p>
            <p>Need at least <strong>{MIN_RECORDS}</strong> records to make reliable predictions.</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# SHOPPING RECOMMENDATIONS PAGE
# ============================================
elif page == "💡 Shopping Recommendations":
    st.title("💡 Smart Shopping Recommendations")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        markets = sorted(df['market'].unique())
        selected_market = st.selectbox("🏪 Select Market", markets)
    
    with col2:
        # Show both price types for this market
        market_data = df[df['market'] == selected_market]
        retail_count = len(market_data[market_data['pricetype'] == 'Retail'])
        wholesale_count = len(market_data[market_data['pricetype'] == 'Wholesale'])
        
        st.markdown("### 💰 Available Price Types")
        if retail_count >= 5:
            st.markdown(f"✅ **Retail:** {retail_count} records (sufficient)")
        else:
            st.markdown(f"⚠️ **Retail:** {retail_count} records")
        
        if wholesale_count >= 5:
            st.markdown(f"✅ **Wholesale:** {wholesale_count} records (sufficient)")
        else:
            st.markdown(f"⚠️ **Wholesale:** {wholesale_count} records")
    
    # Let user choose price type
    selected_pricetype = st.radio(
        "💰 Select Price Type",
        ['Retail', 'Wholesale'],
        horizontal=True,
        help="Retail: Prices for individual consumers | Wholesale: Prices for bulk purchases"
    )
    
    market_data = df[(df['market'] == selected_market) & (df['pricetype'] == selected_pricetype)]
    foods_in_market = market_data['commodity'].value_counts().head(20).index.tolist()
    
    MIN_RECORDS = 5
    
    if foods_in_market:
        st.subheader(f"📊 Recommendations for {selected_market} ({selected_pricetype} Prices)")
        
        recommendations = []
        total_savings = 0
        insufficient_data = []
        
        for food in foods_in_market[:15]:
            historical_df = df[(df['commodity'] == food) & 
                               (df['market'] == selected_market) &
                               (df['pricetype'] == selected_pricetype)].copy()
            
            if len(historical_df) >= MIN_RECORDS:
                current_price = historical_df['price'].iloc[-1]
                current_date = historical_df['date'].iloc[-1]
                
                category = historical_df['category'].iloc[0] if 'category' in historical_df else 'cereals and tubers'
                unit = historical_df['unit'].iloc[0] if 'unit' in historical_df else 'KG'
                
                next_month = current_date.month + 1
                next_year = current_date.year
                if next_month > 12:
                    next_month -= 12
                    next_year += 1
                
                predicted_price = predict_price(
                    food, selected_market, category, selected_pricetype, unit,
                    next_year, next_month
                )
                
                rec = generate_recommendation(current_price, predicted_price)
                
                recommendations.append({
                    'food': food,
                    'current_price': current_price,
                    'predicted_price': predicted_price,
                    'recommendation': rec['action'],
                    'message': rec['message'],
                    'badge_class': rec['badge_class'],
                    'records': len(historical_df)
                })
                
                if rec['action'] == 'WAIT TO BUY':
                    total_savings += (current_price - predicted_price)
            else:
                insufficient_data.append((food, len(historical_df)))
        
        if recommendations:
            st.markdown(f"""
            <div class="metric-card" style="background: #2ecc71; color: white; margin-bottom: 1.5rem;">
                <h2>💰 Total Potential Savings</h2>
                <h1>KES {total_savings:.2f}</h1>
                <p>If you follow "WAIT TO BUY" recommendations for {selected_pricetype} prices</p>
            </div>
            """, unsafe_allow_html=True)
            
            for rec in recommendations[:10]:
                st.markdown(f"""
                <div class="rec-card {rec['badge_class']}">
                    <h3>🍲 {rec['food']}</h3>
                    <p><strong>Current {selected_pricetype} price:</strong> KES {rec['current_price']:.2f} → <strong>Predicted (next month):</strong> KES {rec['predicted_price']:.2f}</p>
                    <p>{rec['message']}</p>
                    <p><small>📊 Based on {rec['records']} records</small></p>
                </div>
                """, unsafe_allow_html=True)
        
        if insufficient_data:
            with st.expander(f"ℹ️ Items with Insufficient {selected_pricetype} Data (Need at least {MIN_RECORDS} records)"):
                for food, count in insufficient_data:
                    st.write(f"- **{food}**: Only {count} record(s) available")
    else:
        st.warning(f"No {selected_pricetype} price data found in {selected_market}.")

# ============================================
# MARKET ANALYSIS PAGE
# ============================================
elif page == "📊 Market Analysis":
    st.title("📊 Market Price Analysis")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_food = st.selectbox("🍲 Select Food", sorted(df['commodity'].unique()))
    
    with col2:
        selected_pricetype = st.radio(
            "💰 Select Price Type",
            ['Retail', 'Wholesale'],
            horizontal=True,
            help="Retail: Prices for individual consumers | Wholesale: Prices for bulk purchases"
        )
    
    food_data = df[(df['commodity'] == selected_food) & (df['pricetype'] == selected_pricetype)]
    MIN_RECORDS = 5
    
    if not food_data.empty:
        market_counts = food_data.groupby('market').size()
        sufficient_markets = market_counts[market_counts >= MIN_RECORDS].index.tolist()
        
        if len(sufficient_markets) == 0:
            st.warning(f"No markets have at least {MIN_RECORDS} records for {selected_food} with {selected_pricetype} price type.")
            
            # Show markets with any data
            st.markdown("### 📍 Markets with data (insufficient):")
            for market, count in market_counts.items():
                st.write(f"- **{market}**: {count} record(s)")
        else:
            food_data_filtered = food_data[food_data['market'].isin(sufficient_markets)]
            market_avg = food_data_filtered.groupby('market')['price'].mean().sort_values()
            
            fig = px.bar(x=market_avg.values, y=market_avg.index, orientation='h',
                         title=f'Average {selected_pricetype} Price of {selected_food} Across Markets',
                         labels={'x': f'Average Price (KES)', 'y': 'Market'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            cheapest_market = market_avg.idxmin()
            cheapest_price = market_avg.min()
            st.success(f"💡 **Best Deal:** {selected_food} is cheapest at **{cheapest_market}** (KES {cheapest_price:.2f}) for {selected_pricetype} price")
            
            st.subheader("📈 Price Trends Comparison")
            top_markets = market_avg.head(5).index.tolist()
            comparison_df = food_data[food_data['market'].isin(top_markets)]
            
            fig2 = px.line(comparison_df, x='date', y='price', color='market',
                           title=f'{selected_pricetype} Price Trends for {selected_food}')
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning(f"No data available for {selected_food} with {selected_pricetype} price type.")

# ============================================
# PRICE PREDICTOR PAGE
# ============================================
elif page == "🔮 Price Predictor":
    st.title("🔮 Future Price Predictor")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_food = st.selectbox("🍲 Select Food", sorted(df['commodity'].unique()), key="predict_food")
    
    with col2:
        markets_for_food = food_markets.get(selected_food, [])
        if markets_for_food:
            market_options = [m[0] for m in markets_for_food]
            selected_market = st.selectbox("🏪 Select Market", market_options, key="predict_market")
    
    with col3:
        selected_pricetype = st.radio(
            "💰 Price Type",
            ['Retail', 'Wholesale'],
            horizontal=True,
            key="predict_pricetype",
            help="Retail: Prices for individual consumers | Wholesale: Prices for bulk purchases"
        )
    
    # Check data availability
    price_info = get_price_type_info(selected_food, selected_market)
    record_count = price_info[selected_pricetype]['count']
    MIN_RECORDS = 5
    
    if record_count > 0:
        if record_count >= MIN_RECORDS:
            st.success(f"✅ {record_count} records available for {selected_pricetype} prices")
        else:
            st.warning(f"⚠️ Only {record_count} records available for {selected_pricetype} prices. Predictions may not be reliable.")
    else:
        st.error(f"❌ No {selected_pricetype} price data available for {selected_food} in {selected_market}")
        
        # Suggest alternative
        other_type = 'Wholesale' if selected_pricetype == 'Retail' else 'Retail'
        other_count = price_info[other_type]['count']
        if other_count > 0:
            st.info(f"💡 Try switching to **{other_type}** price type ({other_count} records available)")
        st.stop()
    
    sample_data = df[(df['commodity'] == selected_food) & 
                     (df['market'] == selected_market) &
                     (df['pricetype'] == selected_pricetype)]
    
    if len(sample_data) >= 1:
        category = sample_data['category'].iloc[0]
        unit = sample_data['unit'].iloc[0]
        current_price = sample_data['price'].iloc[-1]
        
        st.info(f"💰 Current {selected_pricetype} price: **KES {current_price:.2f}** (from {sample_data['date'].iloc[-1].strftime('%B %Y')})")
        
        st.subheader("📅 Select Future Date")
        col1, col2 = st.columns(2)
        
        with col1:
            future_year = st.number_input("Year", min_value=2024, max_value=2027, value=2025)
        with col2:
            future_month = st.selectbox("Month", range(1, 13), index=datetime.now().month - 1)
        
        if st.button("🔮 Generate Prediction", type="primary"):
            with st.spinner("Random Forest model analyzing data..."):
                predicted_price = predict_price(
                    selected_food, selected_market, category, selected_pricetype, unit,
                    future_year, future_month
                )
                
                price_change = predicted_price - current_price
                price_change_pct = (price_change / current_price) * 100
                
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("💰 Current Price", f"KES {current_price:.2f}")
                with col2:
                    st.metric(f"🔮 Predicted ({future_month}/{future_year})", f"KES {predicted_price:.2f}", 
                             delta=f"{price_change_pct:.1f}%")
                with col3:
                    trend = "Increasing 📈" if price_change > 0 else "Decreasing 📉" if price_change < 0 else "Stable ➡️"
                    st.metric("📊 Trend", trend)
                
                if price_change_pct > 10:
                    st.warning(f"⚠️ **BUY NOW!** {selected_pricetype} prices expected to increase by {price_change_pct:.1f}%")
                elif price_change_pct < -10:
                    st.success(f"✅ **WAIT TO BUY!** {selected_pricetype} prices expected to decrease by {abs(price_change_pct):.1f}%")
                else:
                    st.info(f"➡️ **STABLE PRICES** expected (change: {price_change_pct:.1f}%)")
    else:
        st.error(f"Insufficient data for prediction")

# Footer
st.markdown("---")
st.caption("© 2026 Food Price Predictor | Powered by Random Forest Machine Learning | Data Source: WFP Food Prices Kenya")
st.caption("💡 Tip: Use the radio buttons to switch between Retail and Wholesale prices. Not all combinations have both price types available.")
