# ✈️ AI Travel Agent (LangGraph + OpenAI)

An autonomous tool-using AI travel assistant built with **OpenAI GPT-4o**, **LangGraph**, and external APIs.

This project demonstrates how to build a **stateful, multi-step LLM agent** that can:

- Understand natural language travel queries
- Decide which tools to call (Flights / Hotels)
- Execute tools via external APIs
- Aggregate structured travel results
- Generate formatted travel recommendations
- Optionally send results via email (Gmail SMTP)
- Provide a web interface via Gradio

---

## 🚀 Demo Features

### 🔎 Flight Search
- Uses Google Flights via SerpAPI
- Supports round trip / one-way queries
- Returns price, airline, links (if available)

### 🏨 Hotel Search
- Uses Google Hotels via SerpAPI
- Supports date range, guests, room count
- Returns top-rated hotels with pricing

### 🧠 Autonomous Tool Reasoning
- LLM decides which tool(s) to call
- Supports multi-step tool invocation
- Reinserts tool results into conversation state

### 📧 Email Integration
- Sends generated travel info via Gmail SMTP

### 🌐 Web Interface
- Built with Gradio
- Clean interactive UI

---

## 🏗️ Agent Architecture

The agent is implemented using **LangGraph state machine orchestration**:
Environment Variables

Create a .env file in the root directory:

OPENAI_API_KEY=your_openai_key
SERPAPI_API_KEY=your_serpapi_key
FROM_EMAIL=your_email
TO_EMAIL=receiver_email
GMAIL_APP_PASSWORD=your_gmail_app_password
EMAIL_SUBJECT=Travel Information
