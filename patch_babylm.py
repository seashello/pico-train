def patch(path, edits):
    s = open(path).read()
    for old, new in edits:
        assert (
            s.count(old) == 1
        ), f"{path}: expected 1 match, found {s.count(old)} for:\n{old}"
        s = s.replace(old, new)
    open(path, "w").write(s)


COSINE = """    elif training_config.optimization.lr_scheduler == "cosine_with_warmup":
        import math

        def _cosine_lambda(curr_step, num_warmup_steps, max_steps):
            if curr_step < num_warmup_steps:
                return float(curr_step) / float(max(1, num_warmup_steps))
            progress = float(curr_step - num_warmup_steps) / float(
                max(1, max_steps - num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: _cosine_lambda(
                step,
                training_config.optimization.lr_warmup_steps,
                training_config.max_steps,
            ),
        )
"""

VAL_METHODS = """    @property
    def save_steps(self):
        steps = self.configs["checkpointing"].save_steps
        if not steps:
            raise ValueError("Set checkpointing.save_steps in the config")
        return set(int(s) for s in steps)

    @torch.no_grad()
    def _run_validation(self, batch_step):
        import csv
        import math

        if getattr(self, "_val_ids", None) is None:
            self._val_ids = load_dataset(
                self.configs["data"].dataset.name, split="validation"
            )["input_ids"]

        self.model.eval()
        total_nll = torch.tensor(0.0, device=self.fabric.device)
        total_tokens = 0
        bs = 16
        for i in range(0, len(self._val_ids), bs):
            batch = torch.tensor(self._val_ids[i : i + bs], device=self.fabric.device)
            logits, _ = self.model(batch[:, :-1])
            labels = batch[:, 1:]
            total_nll += F.cross_entropy(
                logits.float().transpose(1, 2), labels, reduction="sum"
            )
            total_tokens += labels.numel()
        self.model.train()

        val_loss = (total_nll / total_tokens).item()
        val_ppl = math.exp(val_loss)

        ckpt = self.configs["checkpointing"]
        run_dir = os.path.join(ckpt.runs_dir, ckpt.run_name)
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, "val_ppl.csv")
        is_new = not os.path.exists(path)
        eff_bs = self.configs["data"].dataloader.batch_size
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if is_new:
                w.writerow(["step", "samples_seen", "val_loss", "val_ppl"])
            w.writerow([batch_step, batch_step * eff_bs, f"{val_loss:.6f}", f"{val_ppl:.4f}"])
        self.fabric.log("val/perplexity", val_ppl, step=batch_step)
        self.fabric.log("val/loss", val_loss, step=batch_step)
        self.log(f"Step {batch_step} -- validation perplexity {val_ppl:.2f}")

"""

patch(
    "src/config/checkpointing_config.py",
    [
        (
            "    save_every_n_steps: int = 1000\n",
            "    save_every_n_steps: int = 1000\n\n"
            "    # Explicit steps at which to checkpoint and validate (BabyLM schedule)\n"
            "    save_steps: Optional[List[int]] = None\n",
        )
    ],
)

init = "src/training/utils/initialization.py"
if "cosine_with_warmup" not in open(init).read():
    anchor = '    else:\n        raise ValueError(\n            f"Invalid learning rate scheduler'
    patch(init, [(anchor, COSINE + anchor)])

patch(
    "src/training/trainer.py",
    [
        (
            'if batch_step % self.configs["checkpointing"].save_every_n_steps == 0:\n                self.log(f"Step {batch_step} -- ',
            'if batch_step in self.save_steps:\n                self._run_validation(batch_step)\n                self.log(f"Step {batch_step} -- ',
        ),
        (
            'if final_step % self.configs["checkpointing"].save_every_n_steps != 0:',
            "if final_step not in self.save_steps:",
        ),
        (
            "    def _log_evaluation_results(",
            VAL_METHODS + "    def _log_evaluation_results(",
        ),
    ],
)
print("patched")
