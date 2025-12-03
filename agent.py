import os
import time
import smtplib
import json
import yfinance as yf
import google.generativeai as genai
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- CONFIGURATION ---
# The Watchlist: Blue Chips, ETFs (VOO, QQQ), Gold (GLD), Bitcoin (BTC-USD)
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "VOO", "QQQ", "GLD", "BTC-USD"]
TARGET_MA = 60
THRESHOLD = 0.02 # 2% buffer (Cryptos are volatile, so 2% is safer than 1%)

# --- SECRETS ---
GEMINI_KEY = os.environ["GEMINI_API_KEY"]
GMAIL_USER = os.environ["GMAIL_USER"]
GMAIL_PASS = os.environ["GMAIL_PASS"]
TARGET_EMAIL = os.environ["TARGET_EMAIL"]

# --- SETUP AI ---
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def send_email(subject, body_html):
    try:
        # Use MIMEMultipart to support HTML (links/formatting)
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = GMAIL_USER
        msg['To'] = TARGET_EMAIL
        
        msg.attach(MIMEText(body_html, 'html'))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(GMAIL_USER, GMAIL_PASS)
            server.send_message(msg)
        print(f"✅ Email sent: {subject}")
    except Exception as e:
        print(f"❌ Email failed: {e}")

def get_stock_news(stock_obj):
    """Fetches top 3 recent news headlines using yfinance"""
    try:
        news_items = stock_obj.news
        headlines = []
        # Get top 3 stories
        for item in news_items[:3]:
            title = item.get('title', 'No Title')
            link = item.get('link', '#')
            headlines.append(f"- <a href='{link}'>{title}</a>")
        return "\n".join(headlines)
    except:
        return "No recent news found."

def analyze_market():
    print(f"🚀 Starting analysis for: {TICKERS}")
    
    for ticker in TICKERS:
        try:
            print(f"\n--- Checking {ticker} ---")
            stock = yf.Ticker(ticker)
            
            # Fetch 1 Year of data
            hist = stock.history(period="1y")
            if hist.empty:
                print(f"Skipping {ticker}: No data found.")
                continue

            current_price = hist['Close'].iloc[-1]
            ma_60 = hist['Close'].rolling(window=TARGET_MA).mean().iloc[-1]
            
            # Calculate proximity to MA
            diff = abs(current_price - ma_60)
            percent_diff = diff / ma_60
            
            print(f"Price: ${current_price:.2f} | MA60: ${ma_60:.2f} | Diff: {percent_diff:.2%}")

            # --- DECISION LOGIC ---
            if percent_diff <= THRESHOLD:
                print(f"⚠ TRIGGER! {ticker} is near MA60. Gathering intel...")
                
                # 1. Get News
                news_summary = get_stock_news(stock)
                
                # 2. Ask Gemini for Analysis
                prompt = f"""
                Act as a Senior Hedge Fund Analyst.
                
                Asset: {ticker}
                Current Price: ${current_price:.2f}
                60-Day Moving Average: ${ma_60:.2f} (Technical Support/Resistance Level)
                
                Here are the latest news headlines for {ticker}:
                {news_summary}
                
                TASK:
                1. Analyze if the news is Bullish or Bearish.
                2. Given the price is touching the critical MA60 trendline, provide a recommendation: BUY, SELL, or WAIT.
                3. Write a short explanation (2-3 sentences).
                
                Return JSON ONLY:
                {{
                    "subject": "🚨 Trade Alert: {ticker} at Key Level ({ticker} Recommendation)",
                    "body": "<h3>Technical Analysis</h3><p>The price is testing the 60-day MA.</p><h3>News Analysis</h3><p>...</p><h3>Verdict: [BUY/SELL]</h3>"
                }}
                """
                
                response = model.generate_content(prompt)
                clean_text = response.text.replace("```json", "").replace("```", "")
                
                data = json.loads(clean_text)
                
                # Append the actual news links to the email body for the user
                full_body = data['body'] + f"<br><hr><strong>Related News:</strong><br>{news_summary}"
                
                send_email(data['subject'], full_body)
                
                # IMPORTANT: Pause to respect Gemini Free Tier limits (15 requests/min)
                time.sleep(4) 
                
            else:
                print(f"Status: Safe (Gap > {THRESHOLD*100}%)")
                
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            continue

if __name__ == "__main__":
    analyze_market()