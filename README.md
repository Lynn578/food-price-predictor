# Food Price Predictor - Kenya

A Streamlit web application that helps users analyze food prices, view trends, predict future prices, and get buying recommendations for grocery items in Kenya.

## Features

- 📊 Current Prices - View latest prices by market and food item

📈 Price Trends - Interactive charts showing historical patterns

🔮 Price Prediction - Forecast future prices based on seasonal trends

💡 Buying Advice - Smart recommendations based on price analysis

📥 Data Export - Download historical data as CSV

📱 Responsive Design - Works on desktop and mobile

## Live Demo

[https://food-price-predictor-kenya.streamlit.app](https://food-price-predictor-3e5qhnh2vqxjszdf8ta76a.streamlit.app/)

🏗️ App Structure
text
Food Price Predictor Kenya/
├── app.py                 # Main Streamlit application
├── wfp_food_prices_ken.csv  # Required data file (or data/ folder)
└── README.md             # This file
🚀 Quick Start
Prerequisites
Python 3.8+

Required packages listed in requirements.txt

Installation
bash
# Clone or download the repository
git clone <your-repo-url>
cd food-price-predictor-kenya

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
Download Data
Place the WFP Food Prices Kenya dataset as wfp_food_prices_ken.csv in the root or data/ folder.

Run the App
bash
streamlit run app.py
Open http://localhost:8501 in your browser.

📦 Requirements
text
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.15.0
Install with: pip install -r requirements.txt

📊 Data Source
WFP Food Prices Kenya - World Food Programme's food price monitoring data for Kenyan markets. The app requires the CSV file with columns: date, market, commodity, price, pricetype.

🧭 Navigation Pages
Page	Description
🏠 Home	App overview, statistics, top markets & foods
📊 See Food Prices	Latest prices, averages, price history
📈 Price Trends	Line charts, monthly patterns, statistics
🔮 Predict Future Price	Seasonal price forecasts for any date
💡 Buying Suggestions	Budget-based buy/wait recommendations
🎯 How It Works
Data Loading - Automatically detects CSV in root or data/ folder

Smart Filtering - Only shows markets/foods with sufficient data

Trend Analysis - Uses historical patterns for predictions

Recommendations - Compares current vs average prices

🔧 Key Features Explained
Price Prediction
Uses monthly historical averages to predict future prices. Example logic:

text
If target month has historical data → Use monthly average
Else → Use overall average
Buying Recommendations
GREAT DEAL (<90% of avg): Buy now!

FAIR PRICE (90-110% of avg): Reasonable

PRICE HIGH (>110% of avg): Wait if possible

🛠️ Customization
Add New Markets/Foods
Ensure your CSV has the required columns

App auto-detects new markets and foods

Caching ensures fast performance

Modify Predictions
Edit the prediction logic in the "🔮 Predict Future Price" section:

python
# Simple seasonal prediction (line ~450)
monthly_avg = historical_df.groupby(historical_df['date'].dt.month)['price'].mean()
predicted = monthly_avg[future_month] if future_month in monthly_avg.index else historical_df['price'].mean()
📈 Example Usage
Check maize prices: Go to "📊 See Food Prices" → Nairobi → Maize

See trends: "📈 Price Trends" → Maize → Select market

Predict next month: "🔮 Predict Future Price" → Future date

Budget shopping: "💡 Buying Suggestions" → Enter budget

🤝 Contributing
Fork the repository

Create feature branch (git checkout -b feature/amazing-feature)

Commit changes (git commit -m 'Add amazing feature')

Push to branch (git push origin feature/amazing-feature)

Open Pull Request

📄 License
MIT License - See LICENSE file for details.

🙏 Acknowledgments
World Food Programme (WFP) for Kenya food price data

Streamlit Team for the amazing framework

Kenyan consumers facing food price volatility

Made with ❤️ for Kenyan shoppers 🍽️🇰🇪


