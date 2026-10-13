"""Check the text-only checkpoint answers exactly like d1-omni-600M (run with `uv run --with transformers`)."""
import sys

import torch
from transformers import AutoModel

from decision_searchbar.model import load

omni_path, text_path = sys.argv[1], sys.argv[2]
q = {"type": "choice", "instructions": "Which team should handle this?",
     "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults",
                  "fraud": "Suspected unauthorised use"}}
n = {"type": "noul", "instructions": "Is the customer asking for a refund?"}
state = "I was charged twice this month, please refund one of them."
omni = AutoModel.from_pretrained(omni_path, trust_remote_code=True, dtype=torch.float32)
text = load(text_path, device="cpu")
a, b = omni.probabilities(state, [q, n]), text.probabilities(state, [q, n])
print("omni", a); print("text", b)
assert all(abs(x - y) < 1e-5 for pa, pb in zip(a, b) for x, y in zip(pa, pb))
print("parity ok")
