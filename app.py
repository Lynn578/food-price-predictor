# ============================================
# PRICE TRENDS & PREDICTIONS PAGE
# ============================================
elif page == "📈 Price Trends & Predictions":
    st.title("📈 Price Trends & Predictions")
    st.markdown("---")
    
    # Item selection
    foods = sorted(df['commodity'].unique())
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_food = st.selectbox("🍲 Select Food Item", foods)
    
    with col2:
        markets_for_food = food_markets.get(selected_food, [])
        if markets_for_food:
            market_options = [m[0] for m in markets_for_food]
            selected_market = st.selectbox("🏪 Select Market", market_options)
        else:
            st.error(f"No markets found that sell {selected_food}")
            st.stop()
    
    with col3:
        # Let user choose Retail or Wholesale
        selected_pricetype = st.radio(
            "💰 Price Type",
            ['Retail', 'Wholesale'],
            horizontal=True
        )
    
    # Get historical data with selected price type
    historical_df = df[(df['commodity'] == selected_food) & 
                       (df['market'] == selected_market) &
                       (df['pricetype'] == selected_pricetype)].copy()
    historical_df = historical_df.sort_values('date')
    
    if len(historical_df) >= 5:
        # Get category and unit from data
        category = historical_df['category'].iloc[0] if 'category' in historical_df else 'cereals and tubers'
        unit = historical_df['unit'].iloc[0] if 'unit' in historical_df else 'KG'
        
        # Generate predictions for next months
        current_price = historical_df['price'].iloc[-1]
        current_date = historical_df['date'].iloc[-1]
        
        # Show record count
        st.info(f"📊 {len(historical_df)} records available for {selected_pricetype} prices")
        
        # Predict for next 6 months
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
        
        # Create chart with historical + predictions
        fig = go.Figure()
        
        # Historical data (solid line)
        fig.add_trace(go.Scatter(
            x=historical_df['date'],
            y=historical_df['price'],
            mode='lines+markers',
            name=f'Historical {selected_pricetype} Prices',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4, color='#1f77b4')
        ))
        
        # Predictions (dashed line)
        fig.add_trace(go.Scatter(
            x=predictions_df['date'],
            y=predictions_df['price'],
            mode='lines+markers',
            name='Predicted Prices',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            marker=dict(size=6, symbol='diamond', color='#ff7f0e')
        ))
        
        fig.update_layout(
            title=f'{selected_food} - {selected_pricetype} Price History and Forecast<br>Market: {selected_market}',
            xaxis_title='Date',
            yaxis_title='Price (KES)',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"KES {current_price:.2f}")
        with col2:
            st.metric("Average Price", f"KES {historical_df['price'].mean():.2f}")
        with col3:
            st.metric("Price Range", f"KES {historical_df['price'].min():.2f} - {historical_df['price'].max():.2f}")
        with col4:
            st.metric("Records", f"{len(historical_df)}")
        
        # Show predictions table
        st.markdown("---")
        st.subheader("📅 Price Forecast (Next 6 Months)")
        
        pred_display = predictions_df.copy()
        pred_display['date'] = pred_display['date'].dt.strftime('%Y-%m-%d')
        pred_display['price'] = pred_display['price'].apply(lambda x: f"KES {x:.2f}")
        pred_display.columns = ['Date', 'Predicted Price']
        st.dataframe(pred_display, use_container_width=True)
        
        # Recommendation
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
        st.warning(f"Insufficient data for {selected_food} in {selected_market} with {selected_pricetype} price type. Need at least 5 records.")
        
        # Show what's available
        retail_count = len(df[(df['commodity'] == selected_food) & (df['market'] == selected_market) & (df['pricetype'] == 'Retail')])
        wholesale_count = len(df[(df['commodity'] == selected_food) & (df['market'] == selected_market) & (df['pricetype'] == 'Wholesale')])
        
        st.info(f"📊 Data available: Retail: {retail_count} records | Wholesale: {wholesale_count} records")
