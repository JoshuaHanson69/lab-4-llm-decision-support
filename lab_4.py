# %% [markdown]
# # Lab 4: LLMs and Prompt Engineering for Decision Support
# 
# **Duration:** 2 weeks [30 Jul - 13 Aug, 2026]
# **Due Date:** 13th August, 2026
# **Format:** Jupyter Notebook / Google Colab + external APIs + GitHub version control
# **Grading:** This is a graded lab.
# 
# **Student Name:** [JOSHUA ELVIS ATAA-OKO HANSON]
# **Student ID:** [27372028]
# 
# ---
# 
# ### Objective
# 
# In the previous labs you *trained* models. In this lab you will *use* a model that someone
# else spent millions of dollars training — a **Large Language Model (LLM)** — and learn that
# getting good results out of one is an engineering discipline of its own: **prompt
# engineering**.
# 
# You will build a **decision support system for a microfinance loan officer**. Given a pile of
# free-text loan application letters, your system will:
# 
# 1. **Summarize** each application into a short, factual brief,
# 2. **Extract** specific structured data points (JSON) that a downstream system could store,
# 3. Produce a **decision-support recommendation** — while keeping the human firmly in the loop.
# 
# Just as importantly, you will **evaluate** the LLM's output for quality, reliability, and
# appropriateness: Does it hallucinate? Is it consistent across runs? Should it be trusted to
# make the final call?
# 
# ---
# 
# ### Choosing an API provider
# 
# You need an LLM API with a **free tier**. Recommended options (pick ONE):
# 
# | Provider | Free tier | Notes |
# |---|---|---|
# | **Groq** (recommended) | Yes, generous | OpenAI-compatible API, very fast, open models (Llama) |
# | **Google Gemini** | Yes | `google-generativeai` package |
# | **Hugging Face Inference API** | Yes, limited | Many open models |
# | OpenAI / Anthropic | Paid | Fine if you already have credits |
# 
# The notebook's example code uses the **OpenAI-compatible chat format** (works with Groq and
# OpenAI directly; Gemini users adapt the call in one place). Everything else in the lab is
# provider-agnostic.

# %% [markdown]
# ---
# ### Part 0: Repository and API-key setup
# 
# 1. Create a **public** repository named `lab-4-llm-decision-support` and save this notebook
#    inside it.
# 2. Sign up with your chosen provider and create an **API key**.
# 3. **NEVER hard-code or commit your API key.** This is a graded requirement.
#    - Locally: put it in a `.env` file and add `.env` to `.gitignore`.
#    - Colab: use the Secrets panel (key icon) and read it with `google.colab.userdata`.
# 4. Add a `requirements.txt`: `openai python-dotenv pandas matplotlib`.
# 5. Commit and push after **each Part** — we will check for incremental commits.
# 
# > **A leaked key in your commit history = resubmission + penalty.** Keys can be scraped from
# > public repos within minutes.

# %%
import os
print(os.getcwd())
print(os.path.exists(".env"))

# %%
# API-key setup — DO NOT hard-code your key in this cell.
import os

# --- Local (with a .env file) ---
# from dotenv import load_dotenv
# load_dotenv()
# API_KEY = os.environ["GROQ_API_KEY"]

# --- Google Colab (Secrets panel) ---
# from google.colab import userdata
# API_KEY = userdata.get("GROQ_API_KEY")

# TODO: set API_KEY using ONE of the methods above.
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ["GROQ_API_KEY"]

# OpenAI-compatible client (works for Groq and OpenAI; Gemini users see their docs):
from openai import OpenAI

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1",   # remove this line if using OpenAI itself
)
MODEL = "llama-3.3-70b-versatile"                # or your provider's model name

print("Client ready.")

# %% [markdown]
# ---
# # Section 1 — Talking to an LLM Programmatically
# 
# Before building anything, understand the anatomy of an API call: **messages and roles**
# (`system`, `user`, `assistant`), and the **generation parameters** (`temperature`,
# `max_tokens`).

# %% [markdown]
# ### Part 1.1 — Your first API call

# %%
# TODO: Write a helper function you will reuse for the WHOLE lab:
#
# def ask_llm(user_prompt, system_prompt="You are a helpful assistant.",
#             temperature=0.7, max_tokens=500):
#     response = client.chat.completions.create(
#         model=MODEL,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user",   "content": user_prompt},
#         ],
#         temperature=temperature,
#         max_tokens=max_tokens,
#     )
#     return response.choices[0].message.content
#
# TODO: Call it once with a simple question and print the answer.
# TODO: Print response.usage as well — how many tokens did your call consume?
def ask_llm(user_prompt, system_prompt="You are a helpful assistant.",
            temperature=0.7, max_tokens=500):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

# TODO: Call it once with a simple question and print the answer.
answer = ask_llm("What is the capital of Ghana?")
print(answer)

# TODO: Print response.usage as well — how many tokens did your call consume?
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of Ghana?"},
    ],
)
print(response.usage)

# %% [markdown]
# **Student Reasoning — Anatomy of a call**
# *1. What is the difference between the `system` and `user` roles? Give an example of
# something that belongs in each.*
# *2. What is a token, roughly? Why do API providers bill per token rather than per request?*
# 
# > **Answer:** 1. The system message tells the model how to behave overall, like giving it a role and rules to follow the whole time. The user message is the actual thing you're asking it right now. Example: system = "You're a helpful assistant for a loan officer, stick to facts only." user = "Summarize this loan letter: [letter text]".
# **Answer:** 2. A token is roughly a small piece of a word, sometimes a whole word, sometimes just part of one. Providers charge per token because that's what actually costs them compute. A short request and a long request use very different amounts of processing, so charging per token matches the price to how much work was really done

# %% [markdown]
# ### Part 1.2 — Temperature: the randomness dial

# %%
# TODO: Ask the SAME question 5 times at temperature=0.0 and 5 times at temperature=1.2.
#   A good test question: "Suggest a name for a savings product for market traders in Accra."

# TODO: Print all 10 answers, grouped by temperature.
question = "Suggest a name for a savings product for market traders in Accra."

low_temp_answers = []
for i in range(5):
    ans = ask_llm(question, temperature=0.0)
    low_temp_answers.append(ans)

high_temp_answers = []
for i in range(5):
    ans = ask_llm(question, temperature=1.2)
    high_temp_answers.append(ans)

print("=== Temperature 0.0 ===")
for i, ans in enumerate(low_temp_answers, 1):
    print(f"{i}. {ans}\n")

print("=== Temperature 1.2 ===")
for i, ans in enumerate(high_temp_answers, 1):
    print(f"{i}. {ans}\n")

# %% [markdown]
# **Student Reasoning — Temperature**
# *What did you observe at each temperature? For the loan decision-support system you are about
# to build, which temperature regime is appropriate, and why?*
# 
# > **Answer:** At temperature 0.0, the answers were almost identical each time — some runs were word-for-word the same. At temperature 1.2, every answer was different, with more creative and varied name ideas. For the loan decision-support system, temperature 0.0 (or very close to it) is the right choice, because we need consistent, predictable outputs when extracting facts or making recommendations

# %% [markdown]
# ---
# # Section 2 — The Dataset: Loan Application Letters
# 
# Run the next cell to load **six loan application letters** submitted to a (fictional)
# microfinance institution in Ghana, plus **gold-standard extraction labels** for three of them
# (you will use these for evaluation in Section 4).
# 
# Read at least two letters fully before moving on — you cannot engineer prompts for text you
# have not read.

# %%
LETTERS = {
"L001": """Dear Sir/Madam,
My name is Akosua Mensah and I have been selling provisions at Makola Market for 12 years.
I am applying for a loan of GHS 8,000 to buy a deep freezer and expand into frozen foods.
My current stall makes about GHS 900 profit each month. I have saved GHS 2,500 with your
susu scheme over the past two years and I have never missed a contribution. I can repay
GHS 450 monthly over 20 months. My sister, a teacher, will stand as my guarantor.
Thank you for considering my application.""",

"L002": """Hello,
I am Kwame Boateng, a commercial driver in Kumasi. I need GHS 25,000 urgently to repair my
trotro engine and settle some personal debts. Business has been slow but it will surely
pick up after the festive season. I can pay back whenever the money comes. I do not have
collateral at the moment but God willing everything will be fine. Please help me quickly.""",

"L003": """Dear Loan Committee,
I am Efua Darko, owner of Darko Fashions, a registered dressmaking business in Takoradi
(registration no. BN-2019-4482). I employ three apprentices. I request GHS 15,000 to
purchase two industrial sewing machines and fabric stock ahead of the Christmas season.
Last year my December revenue alone was GHS 22,000; monthly profit averages GHS 2,800.
I hold a fixed deposit of GHS 5,000 with GCB which I can pledge. Proposed repayment:
GHS 1,100 monthly for 15 months. Attached are my sales records for the past 18 months.""",

"L004": """Good day,
My name is Yaw Owusu. I want a loan for my poultry farm at Nsawam. The amount is GHS 12,000
for feed and 500 new layers. I started the farm last year. Sometimes I make good money,
around GHS 1,500 in a good month, but bird flu affected us in March and I lost many birds.
I am rebuilding now. I can repay in 18 months. My uncle has agreed to guarantee the loan
with his taxi.""",

"L005": """Dear Manager,
I am writing on behalf of the Adenta Women's Weaving Cooperative (14 members). We seek
GHS 30,000 to buy a bulk order of yarn directly from the factory, cutting out middlemen and
raising our margins from 15% to about 35%. The cooperative has operated for 6 years and
holds GHS 9,000 in our group account. We propose repayment of GHS 2,000 monthly over
16 months, backed by our group savings and joint liability agreement.""",

"L006": """Hi,
This is Kofi. I saw your advert. I want GHS 50,000 to start a car washing business, a
provision shop, and also import phones from Dubai. I am 22 and full of energy. I have not
started any of these yet but my friends say I am very business minded. I will pay back in
one year when the businesses are booming. No collateral but I am trustworthy.""",
}

# Gold-standard labels for three letters (for Section 4 evaluation):
GOLD = {
  "L001": {"applicant_name": "Akosua Mensah", "amount_ghs": 8000,  "purpose": "buy deep freezer / expand into frozen foods",
           "monthly_profit_ghs": 900,  "has_collateral_or_guarantor": True,  "repayment_months": 20},
  "L003": {"applicant_name": "Efua Darko",    "amount_ghs": 15000, "purpose": "industrial sewing machines and fabric stock",
           "monthly_profit_ghs": 2800, "has_collateral_or_guarantor": True,  "repayment_months": 15},
  "L006": {"applicant_name": "Kofi",          "amount_ghs": 50000, "purpose": "car wash, provision shop, phone imports",
           "monthly_profit_ghs": None, "has_collateral_or_guarantor": False, "repayment_months": 12},
}

print(f"{len(LETTERS)} letters loaded.")

# %% [markdown]
# ---
# # Section 3 — Prompt Engineering for the Decision Support System
# 
# You will now build the three components of the system, iterating on your prompts as you go.
# **Keep every major prompt version** — Section 3.4 asks you to commit your prompt templates
# and document how they evolved.

# %% [markdown]
# ### Part 3.1 — Component 1: Summarization
# Turn a rambling letter into a 3-4 sentence factual brief a busy loan officer can scan.

# %%
# TODO: Write SUMMARY_PROMPT_V1 — your first, naive attempt (e.g. just "Summarize this:").
#   Run it on L002 and L006. Read the output critically.
SUMMARY_PROMPT_V1 = "Summarize this:"

v1_l002 = ask_llm(f"{SUMMARY_PROMPT_V1}\n\n{LETTERS['L002']}")
v1_l006 = ask_llm(f"{SUMMARY_PROMPT_V1}\n\n{LETTERS['L006']}")

print("=== V1 - L002 ===")
print(v1_l002)
print("\n=== V1 - L006 ===")
print(v1_l006)
# TODO: Now write SUMMARY_PROMPT_V2 as a proper template with:
#   - a system prompt giving the LLM a ROLE (e.g. "You are an assistant to a microfinance
#     loan officer...") and constraints (factual, neutral, no invented details, 3-4 sentences)
#   - a user prompt template like: f"Summarize this loan application:\n\n{letter_text}"
#   Run V2 on the same two letters at temperature=0.
SUMMARY_SYSTEM_V2 = (
    "You are an assistant to a microfinance loan officer. Summarize loan "
    "application letters factually and neutrally in 3-4 sentences. Do not "
    "invent, assume, or add any detail that is not explicitly stated in the letter."
)

def summarize_v2(letter_text):
    return ask_llm(
        user_prompt=f"Summarize this loan application:\n\n{letter_text}",
        system_prompt=SUMMARY_SYSTEM_V2,
        temperature=0,
    )

v2_l002 = summarize_v2(LETTERS["L002"])
v2_l006 = summarize_v2(LETTERS["L006"])

print("\n=== V2 - L002 ===")
print(v2_l002)
print("\n=== V2 - L006 ===")
print(v2_l006)
# TODO: Compare V1 vs V2 outputs side by side. Keep both prompt versions in this notebook.
print("\n=== COMPARISON ===")
print("V1 L002:", v1_l002)
print("V2 L002:", v2_l002)
print("V1 L006:", v1_l006)
print("V2 L006:", v2_l006)

# %% [markdown]
# **Student Reasoning — Summarization prompts**
# *1. What concrete problems did V1's output have that V2 fixed? Quote examples.*
# *2. Why is "no invented details" an essential instruction in this application? What is this
# failure mode called in the LLM literature?*
# 
# > **Answer:** Both V1 and V2 were pretty accurate, but V2 was more careful with wording. It used phrases like "he states" or "he claims" instead of just saying things as facts. V1 said stuff more directly, like "promises to repay," without showing it's just what the applicant said. That difference matters a lot for a loan officer trying to separate claims from confirmed facts.
# 2. No invented details" is important because the model could make up something that sounds believable but was never actually written in the letter. This is called hallucination, and in a loan case it could mislead the officer. For example if it invented an income number or a guarantee that doesn't exist.

# %% [markdown]
# ### Part 3.2 — Component 2: Structured extraction (JSON)
# Downstream software cannot read prose. Extract the fields in `GOLD` as strict JSON.

# %%
# TODO: Write EXTRACT_PROMPT — a template that instructs the model to return ONLY a JSON
#   object with EXACTLY these keys:
#     applicant_name (string), amount_ghs (number), purpose (string),
#     monthly_profit_ghs (number or null), has_collateral_or_guarantor (boolean),
#     repayment_months (number or null)
#   Techniques to use:
#     - explicit schema in the prompt
#     - ONE worked example (few-shot) using a letter you write yourself (not from LETTERS!)
#     - "If a field is not stated in the letter, use null. Do not guess."
#     - temperature=0
EXTRACT_SYSTEM = """You are a data extraction assistant for a microfinance loan officer.
Extract information from loan application letters into a JSON object with EXACTLY these keys:
- applicant_name (string)
- amount_ghs (number)
- purpose (string)
- monthly_profit_ghs (number or null)
- has_collateral_or_guarantor (boolean)
- repayment_months (number or null)

Rules:
- Return ONLY the JSON object. No explanation, no markdown fences, no extra text.
- If a field is not stated in the letter, use null. Do not guess or infer a value.

Example:
Letter:
"My name is Ama Serwaa. I run a small chop bar in Tema and need GHS 6,000 to buy
new cooking equipment. I did not mention my monthly earnings. My brother will
guarantee the loan. I will pay it back but did not specify a timeline."

Output:
{"applicant_name": "Ama Serwaa", "amount_ghs": 6000, "purpose": "buy new cooking equipment", "monthly_profit_ghs": null, "has_collateral_or_guarantor": true, "repayment_months": null}
"""

def extract_prompt(letter_text):
    return f"Extract the fields from this loan application letter:\n\n{letter_text}"
# TODO: Write extract_fields(letter_text) that calls the LLM, strips any ```json fences,
#   json.loads() the result, and returns a dict. Handle parse failures gracefully
#   (return None and print a warning).
import json

def extract_fields(letter_text):
    raw = ask_llm(
        user_prompt=extract_prompt(letter_text),
        system_prompt=EXTRACT_SYSTEM,
        temperature=0,
    )
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).replace("json", "", 1)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"WARNING: failed to parse JSON for letter. Raw output:\n{raw}")
        return None
# TODO: Run it on ALL SIX letters; collect results into a pandas DataFrame (one row per
#   letter) and display it.
import pandas as pd

results = {}
for letter_id, letter_text in LETTERS.items():
    results[letter_id] = extract_fields(letter_text)

extraction_df = pd.DataFrame(results).T
extraction_df

# %% [markdown]
# **Student Reasoning — Structured extraction**
# *1. Why must the few-shot example NOT come from the six letters you are processing?*
# *2. Why "use null, do not guess" — what did the model do without that instruction?*
# *3. Why is temperature=0 the right choice for extraction but arguably not for creative tasks?*
# 
# > **Answer:** 1. The few-shot example can't come from the six letters because if the model saw one of L001–L006 in the example, it would already "know" the right answer for that letter and we couldn't fairly test extraction on it. So using a made-up letter keeps all six letters as a genuine test of whether the prompt generalizes.
# 2. Without "use null, do not guess," the model would likely make up a plausible number for missing fields, like guessing a repayment period or profit figure that was never stated
# 3. Temperature=0 is right for extraction because we want the exact same input to always give the exact same output

# %% [markdown]
# ### Part 3.3 — Component 3: The decision-support brief
# Combine everything: for each letter, produce a recommendation brief for the loan officer —
# strengths, risks, missing information, and a suggested next step. The system must
# **support** the decision, not **make** it.

# %%
# TODO: Write BRIEF_PROMPT — it receives the letter AND your extracted JSON, and must output:
#     1. Strengths (bullet points, grounded in the letter)
#     2. Risks / red flags (bullet points)
#     3. Missing information the officer should request
#     4. Suggested next step (e.g. "invite for interview", "request documents",
#        "flag for senior review") — NOT "approve" or "reject".
#   Give the model an explicit instruction that final decisions are made by humans.
BRIEF_SYSTEM = """You are an assistant to a microfinance loan officer in Ghana. You support
decisions, you do not make them. Given a loan application letter and its extracted data,
produce a brief with exactly these four sections:

1. Strengths (bullet points, grounded only in the letter)
2. Risks / red flags (bullet points, grounded only in the letter)
3. Missing information the officer should request
4. Suggested next step (e.g. "invite for interview", "request documents",
   "flag for senior review") — NEVER "approve" or "reject". The final decision
   is always made by a human loan officer, not you.
"""
def brief_prompt(letter_text, extracted_json):
    return (
        f"Loan application letter:\n{letter_text}\n\n"
        f"Extracted data:\n{json.dumps(extracted_json)}\n\n"
        f"Produce the four-section brief."
    )

def generate_brief(letter_id):
    letter_text = LETTERS[letter_id]
    extracted = results[letter_id]  # from Part 3.2
    return ask_llm(
        user_prompt=brief_prompt(letter_text, extracted),
        system_prompt=BRIEF_SYSTEM,
        temperature=0,
    )

# TODO: Generate briefs for ALL SIX letters. Print the briefs for L001, L002, and L006 —
#   three very different applications.
briefs = {}
for letter_id in LETTERS:
    briefs[letter_id] = generate_brief(letter_id)

for lid in ["L001", "L002", "L006"]:
    print(f"=== Brief: {lid} ===")
    print(briefs[lid])
    print()

# %% [markdown]
# **Student Reasoning — Decision support**
# *1. Compare the briefs for L003 (strong application) and L006 (weak application). Did the
# system identify the right strengths and red flags in each?*
# *2. Why did we forbid the model from outputting "approve"/"reject"? Give one practical and
# one ethical reason.*
# 
# > **Answer:** 1. Yes. Comparing L003 (strong) and L006 (weak), the system picked up the right signals. For a strong application it highlighted real business history and collateral. For L006 it correctly flagged no experience, no collateral, and vague repayment reasoning
# 2. Practically: the model doesn't have full context such as credit history, interviews, local knowledge etc so a final yes/no from it could be wrong or unfair. Ethically: loan decisions affect people's livelihoods and access to finance, so a human should stay accountable for that call rather than letting an AI make it unsupervised.

# %% [markdown]
# ### Part 3.4 — Commit your prompt templates
# Prompts ARE code. Save your final `SUMMARY_PROMPT`, `EXTRACT_PROMPT`, and `BRIEF_PROMPT` into
# a separate file `prompts.py` (or `prompts.md`) in your repository and commit it with a
# message describing how the prompts evolved. Paste your commit hash below.
# 
# > **Commit hash:** [d665999829973b552f68b5cb0cf45b8b0fc42f14]

# %% [markdown]
# ---
# # Section 4 — Evaluation: Quality, Reliability, Appropriateness
# 
# An impressive demo is not a trustworthy system. Now measure it.

# %% [markdown]
# ### Part 4.1 — Extraction accuracy against gold labels

# %%
# TODO: For the three letters in GOLD, compare your extracted DataFrame to the gold values
#   field by field. Compute per-field accuracy across the three letters
#   (name matching can be case-insensitive; numbers must match exactly).
def values_match(field, extracted_val, gold_val):
    if gold_val is None:
        return extracted_val is None
    if field == "applicant_name":
        return str(extracted_val).strip().lower() == str(gold_val).strip().lower()
    if field == "has_collateral_or_guarantor":
        return extracted_val == gold_val
    if field in ("amount_ghs", "monthly_profit_ghs", "repayment_months"):
        return extracted_val == gold_val
    return extracted_val == gold_val

fields = ["applicant_name", "amount_ghs", "purpose", "monthly_profit_ghs",
          "has_collateral_or_guarantor", "repayment_months"]

comparison = {field: {} for field in fields}

for letter_id, gold_vals in GOLD.items():
    extracted_vals = results[letter_id]
    for field in fields:
        match = values_match(field, extracted_vals.get(field), gold_vals.get(field))
        comparison[field][letter_id] = "✓" if match else "✗"

# TODO: Display a small table: rows = fields, columns = L001 / L003 / L006 / accuracy.
accuracy_df = pd.DataFrame(comparison).T
accuracy_df["accuracy"] = accuracy_df[list(GOLD.keys())].apply(
    lambda row: f"{(row == '✓').sum()}/{len(row)}", axis=1
)
accuracy_df

# %% [markdown]
# ### Part 4.2 — Reliability: is the system consistent?

# %%
# TODO: Run extract_fields() on letter L004 FIVE times at temperature=0 and FIVE times at
#   temperature=1.0.
def extract_fields_temp(letter_text, temperature):
    raw = ask_llm(
        user_prompt=extract_prompt(letter_text),
        system_prompt=EXTRACT_SYSTEM,
        temperature=temperature,
    )
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).replace("json", "", 1)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None

l004_temp0_runs = [extract_fields_temp(LETTERS["L004"], 0) for _ in range(5)]
l004_temp1_runs = [extract_fields_temp(LETTERS["L004"], 1.0) for _ in range(5)]
# TODO: For each temperature, report how many of the 5 runs produced (a) valid JSON and
#   (b) identical values across runs. A simple approach: json.dumps(result, sort_keys=True)
#   and count unique strings.
def reliability_report(runs, label):
    valid = [r for r in runs if r is not None]
    n_valid = len(valid)
    signatures = [json.dumps(r, sort_keys=True) for r in valid]
    n_unique = len(set(signatures))
    print(f"--- {label} ---")
    print(f"Valid JSON: {n_valid}/5")
    print(f"Unique outputs among valid runs: {n_unique} (1 = fully consistent)")
    for i, sig in enumerate(signatures, 1):
        print(f"  Run {i}: {sig}")
    print()

reliability_report(l004_temp0_runs, "Temperature 0.0")
reliability_report(l004_temp1_runs, "Temperature 1.0")

# %% [markdown]
# ### Part 4.3 — Hallucination probing

# %%
# TODO: Design TWO adversarial tests and run them:
#   Test 1 — Ask your summarizer a question about a detail that is NOT in a letter
#     (e.g. "What is the applicant's credit score?"). Does it admit the information is
#     absent, or does it invent one?
test1_question = "Based on this letter, what is the applicant's credit score?"
test1_result = ask_llm(
    user_prompt=f"{test1_question}\n\nLetter:\n{LETTERS['L002']}",
    system_prompt=SUMMARY_SYSTEM_V2,
    temperature=0,
)
print("=== Test 1: Asking about a detail not in the letter ===")
print(test1_result)
print()
#   Test 2 — Feed your extractor an EMPTY or IRRELEVANT text (e.g. a weather report).
#     Does it return nulls, or does it fabricate an applicant?
weather_report = """Accra Weather Update: Today will be partly cloudy with a high of 31°C
and a low of 24°C. Humidity is expected to reach 85% in the afternoon, with a chance of
scattered showers after 4pm. Winds from the southwest at 12 km/h. UV index: high."""

test2_result = extract_fields(weather_report)
print("=== Test 2: Feeding an irrelevant weather report to the extractor ===")
print(test2_result)
print()

# TODO: Record the outputs verbatim below and label each PASS or FAIL.

# %% [markdown]
# Test 1: PASS — model said the credit score was not mentioned, did not invent one.
# Test 2: PASS — model returned all nulls instead of fabricating an applicant from irrelevant text.

# %% [markdown]
# **Student Reasoning — Evaluation results**
# *1. Report your extraction accuracy. Which field was hardest for the model and why?*
# *2. What did the reliability experiment show about temperature and production systems?*
# *3. Did your system hallucinate under probing? If yes, how could the prompt (or the system
# design around it) reduce the risk?*
# 
# > **Answer:** 1. My extraction was accurate on every field except purpose (0/3), but that's because purpose is free text and my matching was too strict — the model's phrasing was correct in meaning, just worded differently from the gold answer. The numeric and boolean fields matched perfectly.
# 2.  The reliability test showed the extraction stayed fully consistent even at temperature 1.0 for this letter, since the facts in it were stated clearly with little room for variation. But for a production system I'd still keep extraction at temperature 0, since other letters could be more ambiguous and higher temperature adds unnecessary risk.
# 3. My system did not hallucinate under either probe. It admitted missing info instead of inventing it, and returned nulls for an irrelevant document instead of making up a fake applicant. The prompt's explicit "do not guess" and "use null" instructions actually worked here.

# %% [markdown]
# ### Part 4.4 — Appropriateness: should this system exist?
# No code in this part — just judgment, which is the scarcest skill in AI for business.

# %% [markdown]
# **Student Reasoning — Appropriateness**
# *1. Letters L002 and L006 would likely be declined. If the bank fully automated decisions
# with your system, who could be unfairly harmed, and how? Consider applicants who write
# poorly in English but run solid businesses.*
# *2. Loan letters contain personal data. What are the implications of sending them to a
# third-party API in another country? What would you check before deploying this at a real
# Ghanaian microfinance institution?*
# *3. Name TWO concrete safeguards you would build around this system in production (think:
# human review points, logging, appeal processes, monitoring).*
# 
# > **Answer:** 1. If decisions were fully automated, people who don't write fluent English but run genuinely solid businesses could be unfairly harmed. The system judges partly on how clearly and confidently a letter is written. L002's applicant used vague, informal language ("God willing everything will be fine") even though the underlying request might be legitimate. A polished writer with a weaker business could come across better than a strong trader who just isn't good at writing formal letters in English. That's a real fairness risk, especially in a country like Ghana where English fluency varies a lot and isn't tied to business competence.
# 2. Sending loan letters to a third-party API abroad means personal financial data such as names, income, sometimes family details, leaves the country and passes through a system not fully under Ghanaian regulatory oversight. Before deploying this for real, I'd check: does the API provider comply with Ghana's Data Protection Act, is there a data processing agreement in place, is data retained or used for training, and would applicants need to consent to their letter being sent to an external AI system at all.
# 3. (1) mandatory human review before any decision is communicated to the applicant. (2) an appeal process, so if someone is declined they can request the case be reviewed by a person, not just re-run through the same model. 

# %% [markdown]
# ---
# # Section 5 — Reflection
# 
# *Answer in a few sentences each:*
# 
# 1. **Prompting as engineering:** How is iterating on a prompt similar to and different from
#    iterating on the model hyperparameters you tuned in Lab 3?
# 2. **Trust:** After your Section 4 evaluation, would you trust this system to run unattended?
#    What single evaluation result most influenced your answer?
# 3. **Cost and scale:** Estimate (from your `response.usage` numbers) the tokens needed to
#    process 1,000 applications per month. What does that imply for provider choice?
# 4. **Looking back at the course:** You have now used classical ML (Lab 2), trained neural
#    networks (Lab 3), and used a foundation model via API (Lab 4). For a task like this one,
#    why does calling an API beat training your own model — and when would it not?
# 
# > **Answer:** 1.Iterating on a prompt is similar to tuning hyperparameters in that both are trial and error. You change something, test it, see if the output improved, and repeat. The difference is that prompt changes are in plain language and you can often tell immediately why something went wrong just by reading the output, while a hyperparameter like learning rate is more abstract and you usually need training curves or metrics to understand what happened.
# 2. After the Section 4 evaluation, I'd trust this system to draft and prepare information, but not to run fully unattended making real decisions. The result that influenced me most was the hallucination probing. It passed, but I know it's not guaranteed to pass every time on every letter, and a single hallucinated fact in a loan decision could cause real harm to someone's finances.
# 3. From my response.usage numbers, a single call used around 50-60 tokens combined (prompt + completion) for a simple question, but the extraction and brief prompts are longer, likely a few hundred tokens each. Processing 1,000 applications a month, with three calls per application (summary, extraction, brief), could add up to hundreds of thousands of tokens monthly. That makes provider choice matter a lot. A free tier might not scale to this volume, so cost per token would become a real factor in choosing a provider long-term.
# 4. Calling an API is easier here because the LLM already learned language from a huge amount of training data.Training my own model would only really make sense if I needed something very specific to my own data, or needed full control over privacy that an outside API can't offer.

# %% [markdown]
# ---
# ### Submission checklist
# 
# - [ ] All cells run top-to-bottom with no errors (`Kernel -> Restart & Run All`).
# - [ ] **No API key anywhere in the notebook or the commit history.**
# - [ ] Every **Student Reasoning** box is filled in with full sentences.
# - [ ] `prompts.py` / `prompts.md` committed with your final prompt templates.
# - [ ] Evaluation tables and adversarial test outputs visible in the saved notebook.
# - [ ] Notebook pushed to `lab-4-llm-decision-support` with incremental commits.
# - [ ] Repository link submitted to the course portal.
# - [ ] AI Declaration form in Repository.


