from src.text_metrics import classify,is_degenerate,is_refusal

def run(x):
    if set(x) != {"text"} or not isinstance(x["text"],str) or len(x["text"])>4096: raise ValueError("Use response text up to 4096 characters")
    return {"classification":classify(x["text"]),"degenerate":is_degenerate(x["text"]),"refusal_marker":is_refusal(x["text"]),"meaning":"ANSWERED is a heuristic label, not a quality judgment."}
