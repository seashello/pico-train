"""Pretrain a small Llama-style language model from scratch on tokenized BabyLM Strict-Small
with the Hugging Face Trainer.

Same data, same sweep knobs (lr, gradient accumulation, schedule) and same BabyLM
checkpoint schedule as the pico-train runs. Only the training loop and model class change.
"""

import argparse
import csv
import json
import math
import os

# BabyLM Strict-Small schedule: every 1M words up to 10M, then every 10M words up to 100M
MARKS = list(range(1_000_000, 10_000_001, 1_000_000)) + list(
    range(20_000_000, 100_000_001, 10_000_000)
)


def crossed_marks(prev_words, words, marks=MARKS):
    """Marks m with prev_words < m <= words (several can be crossed in one big step)."""
    return [m for m in marks if prev_words < m <= words]


def ffn_dim(d_model):
    """SwiGLU hidden size: about 8/3 * d_model, rounded up to a multiple of 64."""
    return ((int(8 * d_model / 3) + 63) // 64) * 64


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", required=True)
    p.add_argument(
        "--dataset",
        required=True,
        help="HF dataset with train/validation splits and input_ids",
    )
    p.add_argument("--tokenizer", default="allenai/OLMo-7B-0724-hf")
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--accum", type=int, required=True)
    p.add_argument("--sched", choices=["linear", "cosine"], required=True)
    p.add_argument("--micro_batch", type=int, default=16)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=0, help="0 = d_model // 64")
    p.add_argument("--vocab_size", type=int, default=50304)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--epochs", type=float, default=10)
    p.add_argument("--max_steps", type=int, default=-1, help="smoke tests only")
    p.add_argument(
        "--eval_every_steps", type=int, default=0, help="0 = BabyLM word-count schedule"
    )
    p.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="runs_hf")
    p.add_argument("--stats", default="data_stats.json")
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        LlamaConfig,
        LlamaForCausalLM,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, "val_ppl.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)

    # ---------- data ----------
    ds = load_dataset(args.dataset)
    train_ds, val_ds = ds["train"], ds["validation"]
    stats = json.load(open(args.stats)) if os.path.exists(args.stats) else {}
    words_per_epoch = stats.get("words_per_epoch", 9_499_705)
    words_per_sample = words_per_epoch / len(train_ds)
    eff_batch = args.micro_batch * args.accum
    print(
        f"train={len(train_ds)} val={len(val_ds)} words/sample={words_per_sample:.1f} effective_batch={eff_batch}"
    )

    def collate(features):
        ids = torch.tensor(
            [f["input_ids"][: args.seq_len] for f in features], dtype=torch.long
        )
        return {
            "input_ids": ids,
            "labels": ids.clone(),
        }  # the model shifts labels internally

    # ---------- model: random initialisation, nothing pretrained ----------
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    heads = args.heads or args.d_model // 64
    assert args.d_model % heads == 0, "d_model must be divisible by the number of heads"
    config = LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.d_model,
        intermediate_size=ffn_dim(args.d_model),
        num_hidden_layers=args.layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        max_position_embeddings=args.seq_len,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=True,  # input and output embeddings share weights
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    model = LlamaForCausalLM(config)
    n_params = sum(p.numel() for p in model.parameters())
    n_embed = model.get_input_embeddings().weight.numel()
    print(
        f"parameters: {n_params:,} total, {n_embed:,} in the (tied) embedding "
        f"({100 * n_embed / n_params:.0f}%), {n_params - n_embed:,} in the transformer layers"
    )

    # ---------- BabyLM checkpoint / validation schedule ----------
    class BabyLMSchedule(TrainerCallback):
        def __init__(self):
            self.prev_words, self.last_marks, self.last_eval_step = 0.0, [], -1

        def on_step_end(self, a, state, control, **kw):
            words = state.global_step * eff_batch * words_per_sample
            if args.eval_every_steps > 0:
                hit = state.global_step % args.eval_every_steps == 0
                self.last_marks = [state.global_step]
            else:
                self.last_marks = crossed_marks(self.prev_words, words)
                hit = bool(self.last_marks)
            self.prev_words = words
            if hit:
                control.should_evaluate = True
                control.should_save = True
            return control

        def on_save(self, a, state, control, **kw):
            # name checkpoints chck_<N>M (millions of words), the convention the BabyLM eval scripts use
            src = os.path.abspath(
                os.path.join(a.output_dir, f"checkpoint-{state.global_step}")
            )
            for m in self.last_marks:
                label = (
                    f"chck_{m // 1_000_000}M"
                    if args.eval_every_steps == 0
                    else f"chck_step{m}"
                )
                link = os.path.join(a.output_dir, label)
                try:
                    if os.path.lexists(link):
                        os.remove(link)
                    os.symlink(src, link)
                except OSError as e:
                    print("could not create", link, e)
            return control

    schedule = BabyLMSchedule()

    class PplTrainer(Trainer):
        def evaluate(self, *a, **kw):
            metrics = super().evaluate(*a, **kw)
            loss = metrics["eval_loss"]
            ppl = math.exp(loss)
            step = self.state.global_step
            self.log({"eval_perplexity": ppl})  # shows up in W&B as eval/perplexity
            schedule.last_eval_step = step
            new = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(
                        [
                            "run",
                            "step",
                            "samples_seen",
                            "words_seen",
                            "val_loss",
                            "val_ppl",
                            "lr",
                            "grad_accum",
                            "schedule",
                        ]
                    )
                w.writerow(
                    [
                        args.run_name,
                        step,
                        step * eff_batch,
                        round(step * eff_batch * words_per_sample),
                        f"{loss:.6f}",
                        f"{ppl:.4f}",
                        f"{args.lr:.3e}",
                        args.accum,
                        args.sched,
                    ]
                )
            return metrics

    use_wandb = bool(os.environ.get("WANDB_API_KEY")) and bool(
        os.environ.get("WANDB_ENTITY")
    )
    os.environ.setdefault("WANDB_PROJECT", "babylm-sweep-hf")

    targs = TrainingArguments(
        output_dir=run_dir,
        run_name=args.run_name,
        per_device_train_batch_size=args.micro_batch,
        per_device_eval_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type=args.sched,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        fp16=args.precision == "fp16",
        bf16=args.precision == "bf16",
        eval_strategy="no",
        save_strategy="no",  # our callback decides when
        save_only_model=True,
        prediction_loss_only=True,
        logging_steps=50,
        report_to="wandb" if use_wandb else "none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
    )
    trainer = PplTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
        processing_class=tok,
        callbacks=[schedule],
    )
    trainer.train()

    if (
        schedule.last_eval_step != trainer.state.global_step
    ):  # make sure the last step is evaluated
        trainer.evaluate()
    trainer.save_model(os.path.join(run_dir, "final"))
    tok.save_pretrained(os.path.join(run_dir, "final"))
    print("done:", run_dir)


if __name__ == "__main__":
    main()
