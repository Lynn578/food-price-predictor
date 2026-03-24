# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os

st.set_page_config(page_title="Food Price Predictor - Kenya", page_icon="🍽️", layout="wide")

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🍽️ Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Home", "📊 See Food Prices", "📈 Price Trends", "🔮 Predict Future Price", "💡 Buying Suggestions"]
    )
    st.markdown("---")
    st.info("**Data Source:** WFP Food Prices Kenya")

# Load data
@st.cache_data
def load_data():
    try:
        if os.path.exists('wfp_food_prices_ken.csv'):
            df = pd.read_csv('wfp_food_prices_ken.csv')
        elif os.path.exists('data/wfp_food_prices_ken.csv'):
            df = pd.read_csv('data/wfp_food_prices_ken.csv')
        else:
            st.error("❌ Data file not found!")
            return None
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=['price'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is None:
    st.stop()

# Create lookup dictionaries
@st.cache_data
def get_market_foods():
    """Get foods available in each market with record counts"""
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
    """Get markets that sell each food with record counts"""
    food_markets = {}
    for food in df['commodity'].unique():
        food_data = df[df['commodity'] == food]
        markets = []
        for market in food_data['market'].unique():
            record_count = len(food_data[food_data['market'] == market])
            markets.append((market, record_count))
        food_markets[food] = sorted(markets, key=lambda x: x[1], reverse=True)
    return food_markets

# Get the data
market_foods = get_market_foods()
food_markets = get_food_markets()

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.title("🍽️ Food Price Predictor - Kenya")
    st.markdown("### Real-Time Groceries' Price Fluctuation Determination and Shopping Recommender Model")
    st.markdown("---")
    
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
        This application provides:
        - **Real-time price analysis** across different markets
        - **Price trend visualization** with historical patterns
        - **Future price predictions** using historical trends
        - **Smart buying recommendations** based on predicted changes
        - **Budget planning tools** to maximize purchasing power
        """)
    
    with col2:
        st.markdown("### 📊 Statistics")
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Food Items", len(df['commodity'].unique()))
        st.metric("Markets", len(df['market'].unique()))
        st.metric("Date Range", f"{df['date'].min().year} - {df['date'].max().year}")
    
    # Show recommended markets with good data
    st.markdown("---")
    st.markdown("### 📍 Recommended Markets (Most Data)")
    
    # Get top markets by record count
    top_markets = df['market'].value_counts().head(10)
    
    col1, col2 = st.columns(2)
    with col1:
        for market, count in list(top_markets.items())[:5]:
            foods_count = len(market_foods[market])
            st.write(f"**{market}**: {count} records, {foods_count} food items")
    
    with col2:
        for market, count in list(top_markets.items())[5:10]:
            foods_count = len(market_foods[market])
            st.write(f"**{market}**: {count} records, {foods_count} food items")
    
    # Show top foods
    st.markdown("---")
    st.markdown("### 🍲 Most Commonly Tracked Foods")
    top_foods = df['commodity'].value_counts().head(10)
    fig = px.bar(x=top_foods.values, y=top_foods.index, orientation='h')
    fig.update_layout(height=400, title="Top 10 Foods by Data Records")
    st.plotly_chart(fig, use_container_width=True)

# ==================== SEE FOOD PRICES PAGE ====================
elif page == "📊 See Food Prices":
    st.title("📊 See Food Prices")
    st.markdown("---")
    
    # Show markets with good data first
    market_counts = df['market'].value_counts()
    markets = market_counts.index.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_market = st.selectbox("🏪 Choose Market", markets)
        
        # Show what's available in this market
        foods_in_market = market_foods.get(selected_market, [])
        if foods_in_market:
            st.caption(f"✅ {len(foods_in_market)} foods available in {selected_market}")
            # Show top 5 foods
            top_foods = foods_in_market[:5]
            st.write("**Most common foods here:**")
            for food, count in top_foods:
                st.write(f"- {food} ({count} records)")
    
    with col2:
        # Only show foods that exist in selected market
        foods_in_market = market_foods.get(selected_market, [])
        if foods_in_market:
            food_options = [f[0] for f in foods_in_market]
            selected_food = st.selectbox("🍲 Choose Food", food_options)
            
            # Show record count
            record_count = len(df[(df['market'] == selected_market) & (df['commodity'] == selected_food)])
            st.caption(f"✅ {record_count} price records available")
        else:
            st.error(f"No foods found in {selected_market}")
            st.stop()
    
    # Filter data
    filtered_df = df[(df['market'] == selected_market) & (df['commodity'] == selected_food)]
    
    if not filtered_df.empty:
        latest_data = filtered_df.sort_values('date', ascending=False).iloc[0]
        latest_price = latest_data['price']
        latest_date = latest_data['date'].strftime('%Y-%m-%d')
        avg_price = filtered_df['price'].mean()
        
        if len(filtered_df) > 1:
            prev_price = filtered_df.sort_values('date', ascending=False).iloc[1]['price']
            price_change = ((latest_price - prev_price) / prev_price) * 100
        else:
            price_change = 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💰 Latest Price", f"{latest_price:.2f} KES", 
                     delta=f"{price_change:.1f}%" if price_change != 0 else None)
        with col2:
            st.metric("📅 Latest Date", latest_date)
        with col3:
            st.metric("📊 Average Price", f"{avg_price:.2f} KES")
        
        st.markdown("---")
        st.subheader("🏷️ Price by Type")
        price_by_type = filtered_df.groupby('pricetype')['price'].mean()
        st.bar_chart(price_by_type)
        
        st.markdown("---")
        st.subheader("📋 Price History")
        
        display_df = filtered_df[['date', 'price', 'pricetype']].sort_values('date', ascending=False)
        st.dataframe(display_df, use_container_width=True, height=400)
        
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"{selected_food}_{selected_market}_prices.csv",
            mime="text/csv"
        )
    else:
        st.warning(f"No data available for {selected_food} in {selected_market}")

# ==================== PRICE TRENDS PAGE ====================
elif page == "📈 Price Trends":
    st.title("📈 Price Trends")
    st.markdown("---")
    
    # Show foods with good data first
    food_counts = df['commodity'].value_counts()
    foods = food_counts.index.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_food = st.selectbox("🍲 Choose Food", foods)
        
        # Show markets that have this food
        markets_for_food = food_markets.get(selected_food, [])
        if markets_for_food:
            st.caption(f"✅ Available in {len(markets_for_food)} markets")
            # Show top markets
            top_markets = markets_for_food[:5]
            st.write("**Best markets for this food:**")
            for market, count in top_markets:
                st.write(f"- {market} ({count} records)")
    
    with col2:
        # Only show markets that have this food
        markets_for_food = food_markets.get(selected_food, [])
        if markets_for_food:
            market_options = [m[0] for m in markets_for_food]
            selected_market = st.selectbox("🏪 Choose Market", market_options)
            
            # Show record count
            record_count = len(df[(df['commodity'] == selected_food) & (df['market'] == selected_market)])
            st.caption(f"✅ {record_count} price records available")
        else:
            st.error(f"No markets found that sell {selected_food}")
            st.stop()
    
    trend_df = df[(df['commodity'] == selected_food) & (df['market'] == selected_market)]
    
    if not trend_df.empty:
        trend_df = trend_df.sort_values('date')
        
        fig = px.line(trend_df, x='date', y='price', 
                      title=f'Price Trends for {selected_food} in {selected_market}',
                      labels={'date': 'Date', 'price': 'Price (KES)'})
        fig.update_layout(height=500, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        st.markdown("---")
        st.subheader("📊 Price Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Minimum", f"{trend_df['price'].min():.2f} KES")
        with col2:
            st.metric("Maximum", f"{trend_df['price'].max():.2f} KES")
        with col3:
            st.metric("Average", f"{trend_df['price'].mean():.2f} KES")
        with col4:
            st.metric("Volatility", f"{trend_df['price'].std():.2f} KES")
        
        # Monthly pattern
        st.markdown("---")
        st.subheader("📅 Monthly Price Pattern")
        
        monthly_avg = trend_df.groupby(trend_df['date'].dt.month)['price'].mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig2 = px.bar(x=month_names[:len(monthly_avg)], y=monthly_avg.values,
                      title="Average Price by Month",
                      labels={'x': 'Month', 'y': 'Average Price (KES)'})
        st.plotly_chart(fig2, use_container_width=True)
        
        best_month = monthly_avg.idxmin()
        st.info(f"💡 **Best Time to Buy:** Historically, prices are lowest in **{month_names[best_month-1]}** (avg: {monthly_avg.min():.2f} KES)")
    else:
        st.error(f"No price data available")

# ==================== PREDICT FUTURE PRICE PAGE ====================
elif page == "🔮 Predict Future Price":
    st.title("🔮 Predict Future Price")
    st.markdown("---")
    
    # Show foods with good data first
    food_counts = df['commodity'].value_counts()
    foods = food_counts.index.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_food = st.selectbox("🍲 Choose Food", foods, key="predict_food")
    
    with col2:
        # Only show markets that have this food
        markets_for_food = food_markets.get(selected_food, [])
        if markets_for_food:
            market_options = [m[0] for m in markets_for_food]
            selected_market = st.selectbox("🏪 Choose Market", market_options, key="predict_market")
            
            # Show record count
            record_count = len(df[(df['commodity'] == selected_food) & (df['market'] == selected_market)])
            st.caption(f"✅ {record_count} historical records available")
        else:
            st.error(f"No data available for {selected_food} in any market")
            st.stop()
    
    pricetype = st.radio("📊 Price Type", ['Retail', 'Wholesale'], horizontal=True)
    
    st.subheader("📅 Select Future Date")
    col1, col2 = st.columns(2)
    
    with col1:
        future_year = st.number_input("Year", min_value=2024, max_value=2026, value=datetime.now().year)
    with col2:
        future_month = st.selectbox("Month", range(1, 13), index=datetime.now().month - 1)
    
    if st.button("🔮 Predict Price", type="primary"):
        with st.spinner("Analyzing data and predicting..."):
            historical_df = df[(df['commodity'] == selected_food) & (df['market'] == selected_market)]
            
            if not historical_df.empty:
                current_data = historical_df.sort_values('date', ascending=False).iloc[0]
                current_price = current_data['price']
                
                # Simple prediction using monthly average
                monthly_avg = historical_df.groupby(historical_df['date'].dt.month)['price'].mean()
                if future_month in monthly_avg.index:
                    predicted = monthly_avg[future_month]
                else:
                    predicted = historical_df['price'].mean()
                
                price_change = predicted - current_price
                price_change_pct = (price_change / current_price) * 100
                
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("💰 Current Price", f"{current_price:.2f} KES")
                with col2:
                    st.metric("🔮 Predicted Price", f"{predicted:.2f} KES", 
                             delta=f"{price_change_pct:.1f}%" if price_change != 0 else None)
                with col3:
                    trend = "Increasing 📈" if price_change > 0 else "Decreasing 📉" if price_change < 0 else "Stable ➡️"
                    st.metric("📊 Trend", trend)
                
                if price_change > 0:
                    st.success(f"✅ BUY NOW! Prices expected to increase by {price_change_pct:.1f}%")
                elif price_change < 0:
                    st.info(f"⏰ WAIT! Prices expected to decrease by {abs(price_change_pct):.1f}%")
                else:
                    st.warning("➡️ STABLE PRICES expected")
                
                st.caption(f"Based on {len(historical_df)} historical records from {historical_df['date'].min().year} to {historical_df['date'].max().year}")
            else:
                st.error(f"No historical data available")

# ==================== BUYING SUGGESTIONS PAGE ====================
elif page == "💡 Buying Suggestions":
    st.title("💡 Buying Suggestions")
    st.markdown("---")
    
    # Show foods with good data first
    food_counts = df['commodity'].value_counts()
    foods = food_counts.index.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_food = st.selectbox("🍲 Choose Food", foods, key="suggest_food")
    
    with col2:
        # Only show markets that have this food
        markets_for_food = food_markets.get(selected_food, [])
        if markets_for_food:
            market_options = [m[0] for m in markets_for_food]
            selected_market = st.selectbox("🏪 Choose Market", market_options, key="suggest_market")
            
            # Show record count
            record_count = len(df[(df['commodity'] == selected_food) & (df['market'] == selected_market)])
            st.caption(f"✅ {record_count} historical records available")
        else:
            st.error(f"No data available for {selected_food} in any market")
            st.stop()
    
    budget = st.number_input("💰 Your Budget (KES)", min_value=0, value=1000, step=100)
    
    if st.button("💡 Get Recommendation", type="primary"):
        historical_df = df[(df['commodity'] == selected_food) & (df['market'] == selected_market)]
        
        if not historical_df.empty:
            current_data = historical_df.sort_values('date', ascending=False).iloc[0]
            current_price = current_data['price']
            avg_price = historical_df['price'].mean()
            
            units_now = budget // current_price
            units_avg = budget // avg_price if avg_price > 0 else 0
            
            st.markdown("---")
            st.subheader("💰 Price Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💰 Current Price", f"{current_price:.2f} KES")
            with col2:
                st.metric("📊 Average Price", f"{avg_price:.2f} KES")
            with col3:
                st.metric("📦 Units Now", f"{units_now:.0f}")
            
            if current_price > avg_price * 1.1:
                st.warning(f"""
                ### ⚠️ PRICE IS HIGH!
                Current price is {((current_price/avg_price)-1)*100:.1f}% above average.
                
                **Recommendation:** WAIT if possible. Your budget would get you {units_avg} units at average prices.
                """)
            elif current_price < avg_price * 0.9:
                st.success(f"""
                ### ✅ GREAT DEAL!
                Current price is {((avg_price/current_price)-1)*100:.1f}% below average.
                
                **Recommendation:** BUY NOW! Your budget gets you {units_now} units, which is {units_now - units_avg} more than average!
                """)
            else:
                st.info(f"""
                ### 📊 FAIR PRICE
                Current price is close to the average of {avg_price:.2f} KES.
                
                **Recommendation:** You can buy now at a reasonable price.
                """)
            
            # Best month
            monthly_prices = historical_df.groupby(historical_df['date'].dt.month)['price'].mean()
            if not monthly_prices.empty:
                best_month = monthly_prices.idxmin()
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                st.info(f"💡 **Best Time to Buy:** Historically, prices are lowest in **{month_names[best_month-1]}**.")
            
            st.caption(f"Based on {len(historical_df)} historical records")
        else:
            st.error(f"No data available")

# Footer
st.markdown("---")
st.caption("© 2024 Food Price Predictor | Data Source: WFP Food Prices Kenya")