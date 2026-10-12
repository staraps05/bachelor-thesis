# %% fingpt_sentiment_pilot.py
# Purpose: Archived Colab pilot of the FinGPT scorer (LoRA adapter on Llama-2-7B, p_pos/p_neu/p_neg per headline); not part of the pipeline, superseded by pipeline/colab/17b_score_fingpt.ipynb.
# Inputs:  an uploaded CSV with a `headline` column (Colab files.upload), or the built-in fallback list
# Outputs: pilot_scores.csv in the Colab working directory
# Notes:
# - Archived; not part of the pipeline and not used by any thesis number.
# - Runs only in Colab on a CUDA GPU (T4): bitsandbytes 4-bit is NVIDIA-only.
# - Model: NousResearch/Llama-2-7b-hf + FinGPT/fingpt-mt_llama2-7b_lora, 4-bit nf4.
!pip -q install -U transformers peft accelerate bitsandbytes "pandas<2.3"

# %% [cell 2] Load base + FinGPT LoRA adapter in 4-bit
import torch, pandas as pd, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

assert torch.cuda.is_available(), "No CUDA GPU — switch Colab runtime to a T4 GPU."

BASE    = "NousResearch/Llama-2-7b-hf"          # ungated Llama-2-7B mirror (no Meta gating)
ADAPTER = "FinGPT/fingpt-mt_llama2-7b_lora"      # 7B multi-task LoRA (includes sentiment); fits a T4
# The adapter's label must be text sentiment (Financial PhraseBank style), not a
# price-reaction label; otherwise p_pos - p_neg does not mean sentiment.

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                         bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(model, ADAPTER)   # attach FinGPT sentiment LoRA
model.eval()

# %% [cell 3] Headlines: CSV upload with a hardcoded fallback so it runs immediately
FALLBACK = [
    "Apple beats quarterly earnings expectations, raises guidance",
    "Tesla recalls two million vehicles over autopilot safety concern",
    "Fed holds interest rates steady amid cooling inflation",
    "Amazon announces 18,000 layoffs across retail divisions",
    "Nvidia unveils next-gen AI chip, orders sell out instantly",
    "Boeing shares slide after new 737 MAX production halt",
    "Microsoft closes record cloud deal with government agency",
    "Oil prices tumble as OPEC signals higher output",
    "Startup files for bankruptcy after failed funding round",
    "Retail sales flat in December, matching analyst forecasts",
    "Pfizer wins FDA approval for new cancer drug",
    "Regional bank collapses, triggering sector-wide sell-off",
    "Gold hits all-time high as investors seek safe havens",
    "Company reports steady revenue, no change to outlook",
    "Airline profits soar on strong summer travel demand",
]
try:
    from google.colab import files
    print("Upload a CSV with a 'headline' column (or Cancel to use the built-in samples).")
    up = files.upload()
    headlines = pd.read_csv(next(iter(up)))["headline"].dropna().astype(str).tolist() if up else FALLBACK
except Exception as e:
    print("No upload / not on Colab -> using fallback headlines:", e)
    headlines = FALLBACK

# %% [cell 4] CORE: softmax over the three class-token logits at the answer position
# FinGPT's documented prompt template; the trailing space after "Answer:" matters
# (it decides whether the next token is the bare class word or a space-prefixed one).
PROMPT = ("Instruction: What is the sentiment of this news? Please choose an answer "
          "from {{negative/neutral/positive}}\nInput: {}\nAnswer: ")

# Softmax over the three class-token logits at the answer position gives
# (p_pos, p_neu, p_neg) summing to 1.
# Because PROMPT ends with a space the next token is the bare word, so first_id
# tokenises the bare word (use " " + word if the trailing space is dropped).
def first_id(word):
    return tok(word, add_special_tokens=False).input_ids[0]
POS_ID, NEU_ID, NEG_ID = first_id("positive"), first_id("neutral"), first_id("negative")
print("class token ids -> positive:", POS_ID, " neutral:", NEU_ID, " negative:", NEG_ID)  # sanity check

rows = []
for h in headlines:
    ids = tok(PROMPT.format(h), return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**ids).logits[0, -1]                 # logits at the ANSWER position
    cls = logits[[POS_ID, NEU_ID, NEG_ID]].float()          # pull the 3 class-token logits
    p_pos, p_neu, p_neg = F.softmax(cls, dim=-1).tolist()   # softmax over ONLY those 3
    label = ["positive", "neutral", "negative"][int(torch.tensor([p_pos, p_neu, p_neg]).argmax())]
    rows.append(dict(headline=h, p_pos=p_pos, p_neu=p_neu, p_neg=p_neg,
                     label=label, score=p_pos - p_neg))      # score s = p_pos - p_neg in [-1, 1]

df = pd.DataFrame(rows)

# %% [cell 5] Sanity checks, table (sorted worst -> best), save + download
assert ((df[["p_pos", "p_neu", "p_neg"]] >= 0) & (df[["p_pos", "p_neu", "p_neg"]] <= 1)).all().all()
assert (df[["p_pos", "p_neu", "p_neg"]].sum(axis=1).sub(1).abs() < 1e-4).all()   # each row sums to ~1
assert df["score"].between(-1, 1).all()

pd.set_option("display.width", 160, "display.max_colwidth", 55)
print(df.sort_values("score").round(3).to_string(index=False))
df.to_csv("pilot_scores.csv", index=False)
try:
    from google.colab import files
    files.download("pilot_scores.csv")
except Exception:
    print("Saved pilot_scores.csv (download only works on Colab).")