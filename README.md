Food Price Predictor - Kenya 🍽️

Streamlit web app that tracks Kenyan food prices across markets, shows trends, predicts future prices, and gives smart shopping recommendations. Perfect for dealing with those crazy grocery price swings.

🔴 Live Demo: https://food-price-predictor-3e5qhnh2vqxjszdf8ta76a.streamlit.app/

✨ What You'll Get
📊 Current Prices - Latest prices by market/food + full history

📈 Price Trends - Interactive charts + monthly patterns

🔮 Price Prediction - Future price forecasts (seasonal trends)

💡 Buying Tips - Buy now/wait advice based on your budget

📥 CSV Export - Download any food/market data

📱 Mobile-friendly - Works everywhere

🏗️ Folder Structure
text
food-price-predictor-kenya/
├── app.py                    # Main app (this file)
├── wfp_food_prices_ken.csv   # REQUIRED data file
│   └── (or put in data/ folder)
├── requirements.txt          # Dependencies
└── README.md                # You're reading it
🚀 Run It (5 mins)
1. Install Python stuff
bash
pip install streamlit pandas plotly
2. Get the data
Download WFP Kenya food prices CSV → name it wfp_food_prices_ken.csv → put in same folder as app.py

3. Launch
bash
streamlit run app.py
Boom! Opens at http://localhost:8501

📦 Full Requirements
text
streamlit>=1.28.0
pandas>=2.0.0  
plotly>=5.15.0
Save as requirements.txt or just run the pip command above.

📊 The Data
Source: World Food Programme (WFP) Food Prices Kenya

Needs these CSV columns:

text
date, market, commodity, price, pricetype
2023-01-15, Nairobi, Maize, 65.5, Retail
App auto-finds CSV in root folder or data/ subfolder.

🧭 5 Main Pages
Emoji	Page	Shows
🏠	Home	Stats, top markets/foods, quick overview
📊	See Prices	Latest price, avg, history table + CSV download
📈	Trends	Line charts, monthly patterns, min/max/volatility
🔮	Predict	Pick future date → get predicted price + buy/wait advice
💡	Suggestions	Enter budget → "buy now" or "wait" recommendation
🎯 How Predictions Work
Super simple but smart:

Grab all historical prices for that food/market

Calculate average price by month (Jan, Feb, etc.)

Predict = that month's historical average

Compare to current price → "buy now" or "wait"

Code snippet (line ~450):

python
monthly_avg = historical_df.groupby(historical_df['date'].dt.month)['price'].mean()
predicted = monthly_avg[future_month] if future_month in monthly_avg.index else historical_df['price'].mean()
💰 Buying Recommendations
Situation	Rule	Advice
Great Deal	Current < 90% of avg	✅ BUY NOW!
Fair Price	90-110% of avg	📊 OK to buy
Too High	Current > 110% of avg	⏳ WAIT!
Example: Maize 75 KES (avg 85 KES) = 12% below → GREAT DEAL!

🛠️ Customize It
Add new markets/foods
Just drop updated CSV → app auto-detects everything.

Fancier predictions
Edit the prediction section (~line 450). Could add:

Moving averages

Linear regression

External factors (inflation, weather)

New features
Add email alerts for price drops

WhatsApp price notifications

Compare multiple foods/markets

📈 Real Examples
Maize in Nairobi:

📊 Prices → See latest 75 KES (was 72 last week)

📈 Trends → Lowest prices in June (68 KES avg)

🔮 Predict → July 2024 = 82 KES predicted (+9%)

💡 Budget 2000 KES → Buy now! Gets 26 bags

🚨 Troubleshooting
Problem	Fix
"Data file not found"	Put wfp_food_prices_ken.csv in right spot
Blank charts	Need 5+ price records for that food/market
Slow loading	First run caches data, gets fast after
Prediction weird	Needs multi-year data for good monthly avgs
🤝 Contributing
text
1. Fork it
2. git checkout -b my-cool-feature
3. git commit -am "Add cool feature"
4. git push origin my-cool-feature
5. Open PR
📄 License
MIT - do whatever, just don't sue me.

🙌 Thanks
WFP - Amazing free Kenya price data

Streamlit - Made this stupid easy to build

You - Hope this saves you money at the market! 💰

Built for Kenyan shoppers tired of price surprises 🇰🇪🍲
