# Apify Configuration Guide

This guide explains how to set up the data collection backend using Apify.

## 1. Create Account
1. Go to [apify.com](https://apify.com).
2. Sign up for an account. The free tier includes $5/month of credit, which is sufficient for testing. For production (continuous monitoring), you will likely need a paid plan (~$49/mo).
3. Verify your email address.

## 2. Find the Scraper
1. Log in to the Apify Console.
2. Go to the **Store**.
3. Search for **"Reddit Scraper"**.
4. Select the **Reddit Scraper** by **epctex** (High rating, reliable maintenence).
   - [Direct Link to Actor](https://apify.com/epctex/reddit-scraper)

## 3. Configure the Scraper
You can configure a saved "Task" in Apify for easier management, or let our Python script handle the configuration dynamically.
**Our Python script (`ingest.py`) sends the configuration JSON directly**, so you **do not** need to manually configure the input fields on the website unless you are testing manually.

However, ensure your account has access to **Residential Proxies** if possible (often requires paid plan) to avoid Reddit blocking. If on a free tier, standard proxies might work but are less reliable.

**Key Settings (Managed by Script):**
- **Subreddits**: `Residency`, `medicalschool`, `medicine`
- **Search Queries**: (See `config.py` for full list)
- **Time**: All time (for backfill)
- **Include Comments**: YES (Critical for sentiment)

## 4. Get Your API Token
1. In Apify Console, go to **Settings** (bottom left) -> **Integrations**.
2. Look for **Personal API Token**.
3. Click **Copy**.

## 5. Secure Your Token
1. Open your project folder.
2. Duplicate `.env.example` and rename it to `.env`.
3. Paste your token:
   ```bash
   APIFY_TOKEN=your_token_starts_with_apify_api_...
   ```
4. **Never** commit this file to GitHub.

## 6. Verify Connection
Run the pipeline in dry-run or incremental mode to test the connection (requires valid token).
```bash
python main.py --incremental
```

## 7. Cost Estimates
| Activity | Compute Units | Estimated Cost |
|----------|---------------|----------------|
| **24-Month Backfill** | ~50-100 CUs | ~$25 - $50 (One-time) |
| **Daily Updates** | ~2-5 CUs | ~$1-2 / day |
| **Monthly Maintenence** | -- | ~$49/mo Plan |

*Note: Costs vary by data volume. Monitor your Apify dashboard closely during the first run.*
