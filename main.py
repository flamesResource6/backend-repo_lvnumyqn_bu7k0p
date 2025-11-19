import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Config ---------
MODEL_NAME = os.getenv("MODEL_NAME", "t5-small")  # default small seq2seq for offline fine-tuning
DEVICE = "cpu"  # set lazily when torch is available

class TrainRequest(BaseModel):
    train_path_a: str
    train_path_b: str
    output_dir: str = "./model_out"
    max_length: int = 128
    batch_size: int = 8
    epochs: int = 1
    lr: float = 5e-5

class GenerateRequest(BaseModel):
    context: List[str]
    last_message_from_b: str
    max_new_tokens: int = 64
    num_beams: int = 4

# Lazy globals
_tokenizer = None
_model = None
_model_type = "seq2seq"  # or "causal"


def _import_torch_device() -> str:
    try:
        import torch  # type: ignore
        return "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_inference_model(model_dir: Optional[str] = None):
    """Load tokenizer and model for inference from a directory or default name.
    Imported lazily to avoid hard dependency at startup.
    """
    global _tokenizer, _model, _model_type, DEVICE
    from importlib import import_module

    transformers = import_module("transformers")
    AutoTokenizer = getattr(transformers, "AutoTokenizer")
    AutoModelForCausalLM = getattr(transformers, "AutoModelForCausalLM")
    AutoModelForSeq2SeqLM = getattr(transformers, "AutoModelForSeq2SeqLM")

    DEVICE = _import_torch_device()

    name = model_dir or MODEL_NAME
    lower = name.lower()
    if any(k in lower for k in ["t5", "flan", "bart"]):
        _model_type = "seq2seq"
        _tokenizer = AutoTokenizer.from_pretrained(name)
        _model = AutoModelForSeq2SeqLM.from_pretrained(name)
    else:
        _model_type = "causal"
        _tokenizer = AutoTokenizer.from_pretrained(name)
        if getattr(_tokenizer, "pad_token", None) is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _model = AutoModelForCausalLM.from_pretrained(name)

    # move to device if torch exists
    try:
        import torch  # type: ignore
        _model.to(DEVICE)  # type: ignore
    except Exception:
        pass


def build_prompt(context: List[str], last_b: str) -> str:
    history = "\n".join(context[-6:])  # cap context turns
    prompt = f"context:\n{history}\nB: {last_b}\nA:"
    return prompt


@app.post("/api/train")
def train_model(req: TrainRequest):
    """Offline fine-tuning endpoint: expects two JSONL files with fields: conv_id, turn, speaker, text.
    """
    # Lazy imports to avoid hard dependency at server start
    import json
    from importlib import import_module

    datasets = import_module("datasets")
    transformers = import_module("transformers")
    DataCollatorForSeq2Seq = getattr(transformers, "DataCollatorForSeq2Seq")
    AdamW = getattr(transformers, "AdamW")
    get_linear_schedule_with_warmup = getattr(transformers, "get_linear_schedule_with_warmup")

    try:
        import torch  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PyTorch not available: {e}")

    from torch.utils.data import DataLoader  # type: ignore
    from tqdm import tqdm  # type: ignore

    global _model, _tokenizer, _model_type, DEVICE

    # Load raw lines (expect JSONL)
    def load_jsonl(path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except Exception:
                    pass
        return data

    data_a = load_jsonl(req.train_path_a)
    data_b = load_jsonl(req.train_path_b)
    if not data_a or not data_b:
        raise HTTPException(status_code=400, detail="Empty training files or invalid format")

    # Merge by conv_id + turn
    all_events = data_a + data_b
    from collections import defaultdict
    convs = defaultdict(list)
    for ev in all_events:
        convs[ev.get("conv_id", 0)].append(ev)
    for cid in convs:
        convs[cid] = sorted(convs[cid], key=lambda x: x.get("turn", 0))

    # Build supervised examples
    examples = []
    for _, turns in convs.items():
        context = []
        for i, ev in enumerate(turns):
            spk = ev.get("speaker")
            text = ev.get("text", "").strip()
            if not text:
                continue
            prefix = f"A: {text}" if spk == "A" else f"B: {text}"
            context.append(prefix)
            if spk == "B" and i + 1 < len(turns) and turns[i + 1].get("speaker") == "A":
                last_b = text
                a_next = turns[i + 1].get("text", "")
                input_text = build_prompt(context[:-1], last_b)
                target_text = a_next
                examples.append({"input": input_text, "target": target_text})

    if len(examples) == 0:
        raise HTTPException(status_code=400, detail="No A-after-B pairs found in data")

    # Init model/tokenizer
    load_inference_model(MODEL_NAME)

    # Tokenize
    tokenizer = _tokenizer

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["input"],
            max_length=req.max_length,
            padding=False,
            truncation=True,
        )
        # Target tokenization for seq2seq
        if hasattr(tokenizer, "as_target_tokenizer"):
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    batch["target"],
                    max_length=req.max_length,
                    padding=False,
                    truncation=True,
                )
        else:
            labels = tokenizer(
                batch["target"],
                max_length=req.max_length,
                padding=False,
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    Dataset = getattr(datasets, "Dataset")
    ds = Dataset.from_list(examples)
    ds = ds.shuffle(seed=42)
    tokenized = ds.map(preprocess, batched=True, remove_columns=["input", "target"])  # type: ignore

    data_collator = DataCollatorForSeq2Seq(tokenizer=_tokenizer, model=_model)

    loader = DataLoader(tokenized, batch_size=req.batch_size, shuffle=True, collate_fn=data_collator)

    optimizer = AdamW(_model.parameters(), lr=req.lr)  # type: ignore
    num_training_steps = req.epochs * max(1, len(loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.06 * num_training_steps), num_training_steps=num_training_steps
    )

    _model.train()
    DEVICE = _import_torch_device()
    pbar = tqdm(range(num_training_steps), desc="training")
    step = 0
    for _ in range(req.epochs):
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = _model(**batch)  # type: ignore
            loss = outputs.loss
            loss.backward()
            import torch as _torch  # type: ignore
            _torch.nn.utils.clip_grad_norm_(_model.parameters(), 1.0)  # type: ignore
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            pbar.update(1)
    pbar.close()

    # Save
    os.makedirs(req.output_dir, exist_ok=True)
    _model.save_pretrained(req.output_dir)  # type: ignore
    _tokenizer.save_pretrained(req.output_dir)

    return {"status": "ok", "trained_steps": step, "examples": len(examples), "saved_to": req.output_dir}


@app.post("/api/generate")
def generate_reply(req: GenerateRequest):
    global _model, _tokenizer, _model_type, DEVICE
    # Lazy import of torch only when generating
    try:
        import torch as _torch  # type: ignore
    except Exception:
        _torch = None  # type: ignore

    if _model is None or _tokenizer is None:
        load_inference_model(os.getenv("INFERENCE_MODEL_DIR"))
    if _model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    prompt = build_prompt(req.context, req.last_message_from_b)
    inputs = _tokenizer(prompt, return_tensors="pt")

    if _torch is not None:
        DEVICE = _import_torch_device()
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}  # type: ignore

    if _model_type == "seq2seq":
        outputs = _model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            num_beams=req.num_beams,
            early_stopping=True,
        )
        text = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        pad = getattr(_tokenizer, "pad_token_id", None)
        eos = getattr(_tokenizer, "eos_token_id", None)
        outputs = _model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            num_beams=req.num_beams,
            pad_token_id=pad,
            eos_token_id=eos,
        )
        text = _tokenizer.decode(outputs[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
    if text.lower().startswith("a:"):
        text = text[2:].strip()

    return {"reply": text}


class EvalRequest(BaseModel):
    references: List[str]
    predictions: List[str]


@app.post("/api/evaluate")
def evaluate_scores(req: EvalRequest):
    # Lazy import evaluation to avoid startup dependency
    from importlib import import_module
    hf_eval = import_module("evaluate")

    if len(req.references) != len(req.predictions):
        raise HTTPException(status_code=400, detail="references and predictions must be same length")

    bleu = hf_eval.load("sacrebleu")
    rouge = hf_eval.load("rouge")

    bleu_res = bleu.compute(predictions=req.predictions, references=[[r] for r in req.references])
    rouge_res = rouge.compute(predictions=req.predictions, references=req.references)

    return {"bleu": bleu_res, "rouge": rouge_res, "perplexity": None}


@app.get("/")
def read_root():
    return {"message": "Chat reply recommender backend ready"}


@app.get("/test")
def test_database():
    cuda = False
    try:
        import torch  # type: ignore
        cuda = torch.cuda.is_available()
        device = "cuda" if cuda else "cpu"
    except Exception:
        device = "cpu"
    return {"backend": "✅ Running", "cuda": cuda, "device": device, "model_name": MODEL_NAME}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
