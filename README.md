# 🛍️ Revalio — AI-Powered Product Review Analysis and Shopping Assistant
An intelligent web-based shopping assistant built for Malaysian university students to make
smarter purchasing decisions on electronics. Automatically analyzes hundreds of YouTube reviews,
compares prices across Shopee and Lazada, and lets you search products by photo.

✨ Technologies
* `Python`
* `FastAPI`
* `React`
* `BERT & GPT-4`
* `Celery`
* `PostgreSQL`
* `Netlify`

🚀 Features
* Analyzes YouTube review videos using a GPT + BERT hybrid pipeline to generate objective aspect-level product ratings
* Automated price comparison across Shopee and Lazada in real time
* Visual product search powered by GPT-4 Vision — upload a photo to find the product
* GPT-4 chatbot assistant that understands natural language and recommends products by budget and needs
* Auto-selects the best 20 products from 48 candidates based on data richness across 4 categories

📍 The Process
Buying electronics is genuinely painful. You spend hours watching YouTube reviews,
jumping between Shopee and Lazada tabs, and still end up unsure if you're making the right call.
I wanted to build something that did all of that work for you. The hardest part was the review
pipeline — collecting transcripts from hundreds of videos, filtering out the noise, and turning
messy reviewer opinions into a clean, reliable rating. Ended up combining BERT for aspect-level
sentiment with GPT for preprocessing, which made the results a lot more trustworthy than either
model alone. Threw in price comparison, visual search, and a chatbot on top to make it a proper
end-to-end shopping assistant. Still ironing out a few mobile issues, but the core system works.

🚦 Running the Project
1. Clone the repository
2. Install backend dependencies: `pip install -r requirements.txt`
3. Start the FastAPI server: `uvicorn main:app --reload`
4. Start the Celery worker: `celery -A tasks.celery_app worker --loglevel=info --pool=solo`
5. Run ngrok to expose the backend: `python ngrok.py`
6. Copy the ngrok URL into `config.js`, then deploy the frontend folder to Netlify
7. Open the Netlify URL in your browser
