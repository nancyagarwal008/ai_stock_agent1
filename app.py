import streamlit as st
import yfinance as yf
from google import genai
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from fpdf import FPDF
import requests

# --- 1. SETUP & AUTHENTICATION ---
st.set_page_config(page_title="AI Stock Agent 2026", layout="wide", page_icon="📈")

# Persistent Client
if 'client' not in st.session_state:
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
        st.session_state.client = genai.Client(api_key=api_key)
    except Exception:
        st.error("Missing GOOGLE_API_KEY in Streamlit Secrets.")
        st.stop()

# Initialize Session State for Data Holding
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'analysis_text' not in st.session_state:
    st.session_state.analysis_text = None
if 'comp_info' not in st.session_state:
    st.session_state.comp_info = {}

client = st.session_state.client

# --- 2. HELPER FUNCTIONS ---

def get_ticker_and_logo(query):
    try:
        search_url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(search_url, headers=headers).json()
        ticker = response['quotes'][0]['symbol']
        stock_info = yf.Ticker(ticker).info
        website = stock_info.get('website', '').replace('http://', '').replace('https://', '').split('/')[0]
        name = stock_info.get('longName', ticker)
        return ticker, name, website
    except:
        return None, None, None

def generate_pdf(ticker, name, analysis):
    clean_analysis = analysis.replace('–', '-').replace('—', '-').replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 15, f"AI Research: {name} ({ticker})", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 8, clean_analysis.encode('latin-1', 'replace').decode('latin-1'))
    return pdf.output()

# --- 3. DASHBOARD UI ---
tab1, tab2 = st.tabs(["🚀 Live Analysis", "📊 Model Accuracy"])

with tab1:
    st.title("Autonomous AI Stock Intelligence")
    
    st.sidebar.header("Agent Parameters")
    user_query = st.sidebar.text_input("Enter Company or Ticker", value="NVIDIA")
    time_period = st.sidebar.selectbox("History", ["1mo", "3mo", "6mo", "1y"])

    # SEARCH TRIGGER
    if st.sidebar.button("Run Live Analysis"):
        ticker, name, domain = get_ticker_and_logo(user_query)
        
        if ticker:
            with st.spinner("Processing Market Data..."):
                hist = yf.Ticker(ticker).history(period=time_period)
                
                # Store in Session State
                st.session_state.stock_data = hist
                st.session_state.comp_info = {'ticker': ticker, 'name': name, 'domain': domain}
                
                # Trigger AI
                prompt = f"Analyze {name} ({ticker}). Latest data:\n{hist.tail(5).to_string()}\nProvide a BUY/SELL/HOLD signal."
                try:
                    response = client.models.generate_content(model="gemini-2.0-flash", contents=[prompt])
                    st.session_state.analysis_text = response.text
                except Exception as e:
                    st.error(f"AI Error: {e}")
        else:
            st.error("Ticker not found.")

    # PERSISTENT DISPLAY LOGIC
    if st.session_state.stock_data is not None:
        info = st.session_state.comp_info
        hist = st.session_state.stock_data
        
        # Header
        col_l, col_t = st.columns([1, 8])
        if info['domain']:
            with col_l: st.image(f"https://logo.clearbit.com/{info['domain']}", width=60)
        with col_t: st.subheader(f"{info['name']} ({info['ticker']})")

        # Chart
        fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # AI Output
        if st.session_state.analysis_text:
            st.markdown("---")
            st.write("### 🧠 AI Strategic Analysis")
            st.info(st.session_state.analysis_text)
            
            # PDF Download
            pdf_data = generate_pdf(info['ticker'], info['name'], st.session_state.analysis_text)
            st.download_button("📥 Download Research PDF", data=bytes(pdf_data), file_name=f"{info['ticker']}_Report.pdf")

with tab2:
    st.header("Historical Model Accuracy")
    # This tab stays independent or can use the same session_state data
    eval_ticker = st.text_input("Ticker to Evaluate", value=st.session_state.comp_info.get('ticker', 'AAPL'))
    
    if st.button("Calculate Confidence Score"):
        data = yf.Ticker(eval_ticker).history(period="1y")
        if not data.empty:
            # Simple RSI Accuracy Logic
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            data['RSI'] = 100 - (100 / (1 + (gain / loss)))
            data['5D_Return'] = data['Close'].shift(-5) / data['Close'] - 1
            buys = data[data['RSI'] < 35].dropna()
            
            if not buys.empty:
                acc = (buys['5D_Return'] > 0).mean() * 100
                st.metric("Strategy Accuracy", f"{acc:.1f}%")
                st.dataframe(buys[['Close', 'RSI', '5D_Return']].tail(10))
