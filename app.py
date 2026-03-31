# ============================================
# PRICE TRENDS & PREDICTIONS PAGE (SIMPLIFIED)
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
    
    # Let user choose price type directly
    selected_pricetype = st.radio(
        "💰 Select Price Type",
        ['Retail', 'Wholesale'],
        horizontal=True,
        help="Retail: Prices for individual consumers | Wholesale: Prices for bulk purchases"
    )
    
    # Check data availability for selected combination
    price_info = get_price_type_info(selected_food, selected_market)
    record_count = price_info[selected_pricetype]['count']
    MIN_RECORDS = 5
    
    if record_count == 0:
        st.error(f"❌ No {selected_pricetype} price data available for {selected_food} in {selected_market}")
        st.stop()
    
    if record_count < MIN_RECORDS:
        st.warning(f"⚠️ Only {record_count} records available for {selected_pricetype} prices. Predictions may not be reliable.")
    else:
        st.info(f"✅ {record_count} records available for {selected_pricetype} prices")
    
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
