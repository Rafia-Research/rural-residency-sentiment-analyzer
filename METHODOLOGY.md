# Methodology: How "Why They Leave" Works

This document explains how the Rural Residency Sentiment Analyzer turns messy internet comments into actionable workforce data.

## 1. The Data
**Where it comes from**: We collect public discussions from Reddit, specifically looking at communities where medical professionals gather: r/Residency, r/medicalschool, and r/medicine.

**Why Reddit?**: Reddit provides informal peer-to-peer discussions that can reveal themes worth testing in formal workforce research. Posters' identities and professional roles cannot be verified, so these discussions are treated as exploratory signals rather than ground truth.

**What we search for**: The system listens for 6 specific conversation categories:
1.  **Recruitment**: Choosing programs, matching, ranking rural tracks.
2.  **Retention**: Burnout, leaving practice, quitting.
3.  **Partner/Family**: Spouse employment, schools, "two-body problem".
4.  **Oregon/OHSU**: Specific mentions of our region.
5.  **Compensation**: Salary, loans, cost of living.
6.  **Career**: Scope of practice, mentorship, autonomy.

**Volume**: Backfills are restricted to a configurable 24-month window. Incremental runs can be scheduled as often as the research workflow requires.

## 2. How We Measure Sentiment
We use a "Sentiment Analysis" model to grade every post.
- **Tools**: We use **RoBERTa** from Hugging Face (`cardiffnlp/twitter-roberta-base-sentiment-latest`). It was trained on social-media text, but its performance must still be validated on healthcare-workforce discussions.
- **The Score**: Every post gets a score from **-1.0 (Very Negative)** to **+1.0 (Very Positive)**.
    - *Example*: "I love the autonomy of rural practice!" → **+0.9** (Positive)
    - *Example*: "The call schedule is brutal and I miss the city." → **-0.8** (Negative)
    - *Example*: "Rural residency is 3 years long." → **0.1** (Neutral/Factual)

## 3. How We Find Topics (v2 Feature)
Imagine dumping 5,000 letters onto a table. "Topic Modeling" is like having a robot automatically sort them into piles based on what they are about, without being told the categories beforehand.
- It finds hidden themes we didn't know to look for.
- *Example*: It might create a pile for "Housing Shortages" even if we never programmed it to search for "housing".

## 4. How We Protect Privacy
Even though Reddit is public, we apply privacy-minimizing controls before analytics data reaches disk or an optional external model.
- **PII Detection**: We use Microsoft Presidio to scan every post for:
    - Names (e.g., "Dr. Smith")
    - Locations (e.g., "Main Street Clinic")
    - Emails & Phone Numbers
- **Redaction**: These are automatically replaced with `[REDACTED]`.
- **Pseudonymization**: Source IDs are converted to stable hashes; usernames, source URLs, raw text, and exact source timestamps are excluded from analytics outputs.
- **Audit**: We record where redactions happened without retaining the detected values in analytics outputs.

This is a privacy-conscious engineering design, not a claim of HIPAA certification or formal HIPAA de-identification.

## 5. What This Data CAN Tell You
- **Trends**: "Are complaints about rural partner employment increasing this year compared to last?"
- **Relative Pain Points**: "Do residents complain more about *salary* or *isolation*?"
- **Regional Sentiment**: "Is sentiment about Oregon residencies better or worse than the national average?"

## 6. What This Data CANNOT Tell You
- **NOT Representative**: This is only the opinion of people who post on Reddit (typically younger, tech-savvy). It is not a random sample of all physicians.
- **NOT Verification**: We cannot verify if a poster is actually a doctor.
- **NOT Prevalence**: Targeted search terms intentionally oversample particular workforce concerns, so category and sentiment percentages do not estimate their prevalence among physicians.
- **NOT Independent Responses**: Comments within the same discussion are clustered and should not be treated as independent survey participants.
- **Correlation ≠ Causation**: Just because attrition mentions go up, it doesn't prove *why* without further investigation.

## 7. How To Use The Outputs
Use the Power BI-ready outputs to **generate hypotheses**, not to make final decisions.
- *Wrong way*: "Reddit says retention is bad, so we must raise salaries."
- *Right way*: "Reddit signals that partner employment is a rising concern. Let's add specific questions about spousal support to our next official survey to validate this."

## 8. Technical Architecture
```mermaid
graph LR
    Reddit[Reddit Data] --> Ingest[Python Pipeline]
    Ingest --> Analysis[AI Sentiment & Privacy Engine]
    Analysis --> Data[Clean CSV Data]
    Data --> Dashboard[Power BI-ready Analysis]
```
