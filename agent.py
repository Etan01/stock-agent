import os
import time
import smtplib
import json
import yfinance as yf
import google.generativeai as genai
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "INTC", "AMD", "GOOGL", "META", "NFLX", "VOO", "VTI", "QQQ", "QQQM", "GLD", "GLDM", "BTC-USD"]
TARGET_MA = 60      
TREND_MA = 200      
THRESHOLD = 0.025   # 2.5% buffer
RSI_PERIOD = 14     

# --- SECRETS ---
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_PASS = os.environ.get("GMAIL_PASS")
TARGET_EMAIL = os.environ.get("TARGET_EMAIL")

# --- SETUP AI ---
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel("gemini-flash-latest")

# --- UTILITY FUNCTIONS ---

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_historical_volatility(data, window=252):
    """Calculates annualized historical volatility"""
    log_returns = np.log(data['Close'] / data['Close'].shift(1))
    return log_returns.rolling(window=window).std().iloc[-1] * np.sqrt(252)

def check_macro_environment():
    """Checks VIX and 10Y Yields to determine market regime"""
    try:
        vix = yf.Ticker("^VIX").history(period="5d")['Close'].iloc[-1]
        tnx = yf.Ticker("^TNX").history(period="5d")['Close'].iloc[-1]
        
        status = "NEUTRAL"
        if vix > 30: status = "EXTREME FEAR (High Risk)"
        elif vix > 20: status = "FEAR (Caution)"
        elif vix < 15: status = "GREED (Bullish)"
        
        return {"vix": vix, "tnx": tnx, "status": status}
    except:
        return {"vix": 0, "tnx": 0, "status": "Error Fetching Macro"}

def get_option_idea(stock, current_price):
    """Finds a Long Call option approx 30-45 days out"""
    try:
        expirations = stock.options
        if not expirations:
            return None
            
        # Find expiration 30-60 days out
        target_date = None
        min_days = 30
        max_days = 60
        today = datetime.now()
        
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            days_out = (exp_date - today).days
            if min_days <= days_out <= max_days:
                target_date = exp_str
                break
        
        if not target_date:
            target_date = expirations[0] # Fallback to nearest
            
        # Get Chain
        opt_chain = stock.option_chain(target_date)
        calls = opt_chain.calls
        
        # Find Strike slightly OTM (At The Money + 1 strike up)
        # We want strike closest to price * 1.02
        target_strike = current_price * 1.02
        calls['abs_diff'] = abs(calls['strike'] - target_strike)
        best_call = calls.loc[calls['abs_diff'].idxmin()]
        
        return {
            "expiration": target_date,
            "strike": best_call['strike'],
            "lastPrice": best_call['lastPrice'],
            "impliedVolatility": best_call['impliedVolatility'],
            "volume": best_call['volume']
        }
    except Exception as e:
        print(f"Option error: {e}")
        return None

def generate_chart(ticker, data, ma60, ma200):
    plt.switch_backend('Agg') 
    plt.figure(figsize=(10, 5))
    
    recent_data = data.iloc[-180:]
    
    plt.plot(recent_data.index, recent_data['Close'], label='Price', color='black', linewidth=1.5)
    plt.plot(recent_data.index, recent_data['MA60'], label='MA60 (Support)', color='green', linestyle='--')
    plt.plot(recent_data.index, recent_data['MA200'], label='MA200 (Trend)', color='red', linestyle='-')
    
    plt.title(f"{ticker} - Technical Setup")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def run_backtest(data):
    df = data.copy()
    df['Signal'] = (df['Low'] <= df['MA60'] * (1 + THRESHOLD)) & \
                   (df['Close'] > df['MA200']) & \
                   (df['Close'] > df['MA60'])
                   
    signals = df[df['Signal']].index
    
    if len(signals) < 1: return "No similar setups."

    trades = []
    for date in signals:
        idx = df.index.get_loc(date)
        if idx + 10 < len(df):
            buy_price = df.iloc[idx]['Close']
            sell_price = df.iloc[idx + 10]['Close']
            trades.append((sell_price - buy_price) / buy_price)

    if not trades: return "Insufficient data."

    win_rate = len([t for t in trades if t > 0]) / len(trades) * 100
    avg_return = (sum(trades) / len(trades)) * 100
    
    return f"Win Rate: {win_rate:.1f}% | Avg Return (10-day): {avg_return:.1f}% ({len(trades)} trades)"

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

def send_email(subject, body_html, image_buffer=None):
    try:
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = GMAIL_USER
        msg['To'] = TARGET_EMAIL
        
        msg.attach(MIMEText(body_html, 'html'))
        
        if image_buffer:
            img = MIMEImage(image_buffer.read())
            img.add_header('Content-ID', '<chart_image>')
            img.add_header('Content-Disposition', 'inline', filename='chart.png')
            msg.attach(img)

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(GMAIL_USER, GMAIL_PASS)
            server.send_message(msg)
        print(f"✅ Email sent: {subject}")
    except Exception as e:
        print(f"❌ Email failed: {e}")

# --- MAIN LOGIC ---

def analyze_market():
    # 1. Check Macro Context First
    macro = check_macro_environment()
    print(f"🌍 MACRO CONTEXT: VIX={macro['vix']:.2f} ({macro['status']})")
    
    # If Panic, strict filtering could be applied here, but we pass info to AI instead.
    
    print(f"🚀 Starting Scan for: {TICKERS}")
    
    for ticker in TICKERS:
        try:
            print(f"\n--- Analyzing {ticker} ---")
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2y")
            
            if len(hist) < 200: continue

            # Indicators
            hist['MA60'] = hist['Close'].rolling(window=TARGET_MA).mean()
            hist['MA200'] = hist['Close'].rolling(window=TREND_MA).mean()
            hist['RSI'] = calculate_rsi(hist)
            
            current = hist.iloc[-1]
            price = current['Close']
            ma60 = current['MA60']
            ma200 = current['MA200']
            rsi = current['RSI']
            
            # Volatility Check
            hist_vol = calculate_historical_volatility(hist) # e.g. 0.25 for 25%
            
            # Strategy Logic
            is_uptrend = price > ma200
            diff = abs(price - ma60)
            is_near_ma60 = (diff / ma60) <= THRESHOLD

            print(f"Price: ${price:.2f} | MA60: ${ma60:.2f} | Trend: {'UP' if is_uptrend else 'DOWN'}")

            if is_near_ma60 and is_uptrend:
                print(f"🔥 SIGNAL: {ticker}")
                
                # Get Extras
                backtest_result = run_backtest(hist)
                chart_buf = generate_chart(ticker, hist, ma60, ma200)
                news_html = get_stock_news(stock)
                
                # Get Option Idea (Long Call)
                option_data = get_option_idea(stock, price)
                opt_text = "No options available."
                if option_data:
                    opt_text = (f"Call Exp: {option_data['expiration']}, Strike: ${option_data['strike']}, "
                                f"Price: ${option_data['lastPrice']:.2f}, IV: {option_data['impliedVolatility']:.2%}")

                # AI Analysis
                prompt = f"""
                You are a Senior Trader.
                Signal: {ticker} is in an Uptrend (>MA200) and touched Support (MA60).
                
                MACRO CONTEXT:
                - VIX: {macro['vix']:.2f} ({macro['status']})
                
                ASSET DATA:
                - Price: ${price:.2f}
                - RSI: {rsi:.2f} (Oversold < 30)
                - Historical Volatility (HV): {hist_vol:.2%}
                
                OPTION IDEA (Long Call):
                {opt_text}
                (Note: If Implied Volatility (IV) >> Historical Volatility (HV), options are expensive. Prefer shares.)
                
                BACKTEST (2Y):
                {backtest_result}
                
                NEWS:
                {news_html}
                
                TASK:
                1. Score the setup (0-10) based on Technicals + Macro + News.
                2. Evaluate the Option Play: Is IV cheap enough to buy calls? Or stick to shares?
                
                Return JSON: {{
                    "subject": "🎯 Buy Signal: {ticker} (Score: [SCORE]/10)",
                    "body": "..."
                }}
                """
                
                try:
                    response = model.generate_content(prompt)
                    clean_text = response.text.replace("```json", "").replace("```", "")
                    data = json.loads(clean_text)
                    
                    final_body = f"""
                    <h2>Strategy Signal: Dip in Uptrend</h2>
                    <p><strong>{ticker}</strong> has pulled back to the 60-Day Moving Average.</p>
                    
                    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
                        <strong>🌍 Market Environment:</strong> VIX {macro['vix']:.2f} ({macro['status']})
                    </div>
                    <br>
                    
                    <table border="1" style="border-collapse: collapse; padding: 5px;">
                        <tr><td><strong>Current Price</strong></td><td>${price:.2f}</td></tr>
                        <tr><td><strong>Support (MA60)</strong></td><td>${ma60:.2f}</td></tr>
                        <tr><td><strong>Trend (MA200)</strong></td><td>${ma200:.2f}</td></tr>
                        <tr><td><strong>Option IV / HV</strong></td><td>IV: {option_data['impliedVolatility']:.1%} / HV: {hist_vol:.1%}</td></tr>
                        <tr><td><strong>Backtest</strong></td><td>{backtest_result}</td></tr>
                    </table>
                    
                    <h3>AI Analysis & Option Verdict</h3>
                    {data['body']}
                    
                    <h3>Suggested Option</h3>
                    <p>{opt_text}</p>
                    
                    <h3>Chart Snapshot</h3>
                    <img src="cid:chart_image" alt="Chart" style="width:100%; max-width:600px;">
                    
                    <h3>Recent News</h3>
                    {news_html}
                    """
                    
                    send_email(data['subject'], final_body, chart_buf)
                    time.sleep(4)
                    
                except Exception as e:
                    print(f"AI/Email Error: {e}")
            else:
                print("Status: No signal.")

        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")
            continue

if __name__ == "__main__":
    analyze_market()