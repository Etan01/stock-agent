import os
import time
import smtplib
import json
import yfinance as yf
import google.generativeai as genai
import pandas as pd # Explicitly importing pandas for checking NaN/Empty data
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from dotenv import load_dotenv
load_dotenv()  # This loads the .env file locally

# --- CONFIGURATION ---
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "NFLX", "VOO", "QQQ", "GLD", "BTC-USD"]
TARGET_MA = 60      # The "Dip" level
TREND_MA = 200      # The "Trend" filter (Must be above this)
THRESHOLD = 0.02    # 2% buffer

# --- SECRETS ---
GEMINI_KEY = os.environ["GEMINI_API_KEY"]
GMAIL_USER = os.environ["GMAIL_USER"]
GMAIL_PASS = os.environ["GMAIL_PASS"]
TARGET_EMAIL = os.environ["TARGET_EMAIL"]

# --- SETUP AI ---
genai.configure(api_key=GEMINI_KEY)
# Using the stable version to avoid 404 errors
model = genai.GenerativeModel(
    model_name="models/gemini-flash-latest",
    generation_config={"temperature": 0.3}
)

print("USING FUNCTION:", model.generate_content)

try:
    test_response = model.generate_content("Hello Gemini! Just testing.")
    print("Gemini Test Output:", test_response.text)
except Exception as e:
    print("Gemini Test Error:", e)

def send_email(subject, body_html):
    try:
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
    try:
        news_items = stock_obj.news
        headlines = []

        for item in news_items[:3]:
            content = item.get('content', {})
            title = content.get('title')
            link = content.get('previewUrl') or content.get('clickThroughUrl', {}).get('url')
            if title and link:
                headlines.append(f"<li><a href='{link}'>{title}</a></li>")

        if not headlines:
            return "<p>No recent news found.</p>"

        return "<ul>" + "".join(headlines) + "</ul>"

    except Exception as e:
        print("News fetch error:", e)
        return "<p>No recent news found.</p>"

def analyze_market():
    print(f"🚀 Starting Advanced Strategy (Dip in Uptrend) for: {TICKERS}")
    
    for ticker in TICKERS:
        try:
            print(f"\n--- Analyzing {ticker} ---")
            stock = yf.Ticker(ticker)
            
            # Fetch 2 Years of data (Need enough history for accurate MA200)
            hist = stock.history(period="2y")
            
            if len(hist) < 200:
                print(f"Skipping {ticker}: Not enough data for MA200.")
                continue

            # Get Latest Data Points
            current_price = hist['Close'].iloc[-1]
            ma_60 = hist['Close'].rolling(window=TARGET_MA).mean().iloc[-1]
            ma_200 = hist['Close'].rolling(window=TREND_MA).mean().iloc[-1]
            
            # --- STRATEGY CHECKS ---
            
            # 1. Trend Check: Is the stock in a long-term Uptrend?
            is_uptrend = current_price > ma_200
            
            # 2. Trigger Check: Is price pulling back to MA60?
            diff = abs(current_price - ma_60)
            percent_diff = diff / ma_60
            is_near_ma60 = percent_diff <= THRESHOLD

            print(f"Price: ${current_price:.2f}")
            print(f"MA60:  ${ma_60:.2f} (Diff: {percent_diff:.2%})")
            print(f"MA200: ${ma_200:.2f} (Trend: {'UP' if is_uptrend else 'DOWN'})")

            # --- DECISION LOGIC ---
            if is_near_ma60 and is_uptrend:
                print(f"🔥 SIGNAL FOUND! {ticker} is in an Uptrend and touching MA60.")
                
                news_summary = get_stock_news(stock)
                print("NEWS:", news_summary)
                
                prompt = f"""
                You are a Quantitative Hedge Fund Manager.
                
                STRATEGY CONTEXT:
                We look for 'Mean Reversion in an Uptrend'.
                1. The Asset ({ticker}) is in a confirmed Bull Market (Price > 200-Day MA).
                2. The Price is currently pulling back to the 60-Day MA (Support Level).
                
                DATA:
                - Price: ${current_price:.2f}
                - MA60: ${ma_60:.2f} (Immediate Support)
                - MA200: ${ma_200:.2f} (Major Trend Support)
                
                NEWS HEADLINES:
                {news_summary}
                
                TASK:
                Analyze the news to ensure there is no fundamental reason to panic sell. 
                If the news is neutral or positive, this is a strong BUY signal.
                
                Return JSON ONLY:
                {{
                    "subject": "🔥 BUY THE DIP: {ticker} at Support",
                    "body": "<h3>Strategy: Trend Pullback</h3><p><strong>{ticker}</strong> is in a verified uptrend (Above MA200) and has pulled back to the MA60 support line.</p><h3>Analyst Verdict</h3><p>[Insert your detailed analysis here based on the news]</p><h3>Recommendation: [STRONG BUY / CAUTIOUS HOLD]</h3>"
                }}
                """
                
                response = model.generate_content(prompt)
                clean_text = response.text.replace("```json", "").replace("```", "")
                
                try:
                    data = json.loads(clean_text)
                    full_body = data['body'] + f"<br><hr><strong>Sources:</strong><br>{news_summary}"
                    send_email(data['subject'], full_body)
                except json.JSONDecodeError:
                    # Fallback if Gemini hallucinates formatting
                    send_email(f"Buy Signal: {ticker}", f"Price ${current_price} is at MA60 support in an uptrend.")
                
                time.sleep(4) # Respect rate limits
                
            elif is_near_ma60 and not is_uptrend:
                print(f"⛔ Filtered: {ticker} hit MA60, but is in a DOWNTREND (Below MA200). Ignoring.")
            else:
                print(f"Status: No signal.")
                
        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            continue

if __name__ == "__main__":
    analyze_market()