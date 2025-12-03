import os
import smtplib
import yfinance as yf
import google.generativeai as genai
from email.mime.text import MIMEText

# --- CONFIGURATION FROM SECRETS ---
# We read these from GitHub's secure environment variables
GEMINI_KEY = os.environ["GEMINI_API_KEY"]
GMAIL_USER = os.environ["GMAIL_USER"]
GMAIL_PASS = os.environ["GMAIL_PASS"]
TARGET_EMAIL = os.environ["TARGET_EMAIL"]

TICKER = "AAPL"
TARGET_MA = 120
THRESHOLD = 0.015 # 1.5% buffer

# --- SETUP GEMINI ---
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def send_email(subject, body):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = GMAIL_USER
    msg['To'] = TARGET_EMAIL

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(GMAIL_USER, GMAIL_PASS)
        server.send_message(msg)
    print("Email sent!")

def analyze_market():
    print(f"Fetching data for {TICKER}...")
    stock = yf.Ticker(TICKER)
    # Get enough data for 120 days
    hist = stock.history(period="1y") 
    
    current_price = hist['Close'].iloc[-1]
    ma_120 = hist['Close'].rolling(window=TARGET_MA).mean().iloc[-1]
    
    # Calculate distance
    diff = abs(current_price - ma_120)
    percent_diff = diff / ma_120
    
    print(f"Price: ${current_price:.2f} | MA120: ${ma_120:.2f} | Diff: {percent_diff:.4f}")

    # --- THE AGENTIC DECISION ---
    # We ask Gemini to decide if this is worth an email. 
    # This prevents "dumb" alerts if the data is weird or barely touching.
    
    if percent_diff <= THRESHOLD:
        prompt = f"""
        You are a financial analyst. 
        The stock {TICKER} is at ${current_price:.2f}.
        The 120-Day Moving Average is ${ma_120:.2f}.
        The difference is only {percent_diff:.2%}.
        
        Write a short, urgent email subject and body alerting the user that the price is testing the trend line.
        Return ONLY valid JSON like: {{"subject": "...", "body": "..."}}
        """
        
        response = model.generate_content(prompt)
        # Clean up response to get pure JSON (sometimes models add markdown)
        text = response.text.replace("```json", "").replace("```", "")
        
        # Simple parsing (or use json.loads for robustness)
        import json
        try:
            data = json.loads(text)
            send_email(data['subject'], data['body'])
        except:
            # Fallback if AI fails
            send_email(f"Stock Alert: {TICKER}", f"Price ${current_price} is near MA120 ${ma_120}")
            
    else:
        print("Price is far from MA120. No action taken.")

if __name__ == "__main__":
    analyze_market()