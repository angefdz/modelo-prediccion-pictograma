import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class BilingualPictogramPredictor(nn.Module):
    """Encoder multilingüe compartido y una cabeza independiente por idioma."""

    def __init__(self, base_model: str, number_of_labels: int, local_config: str | None = None):
        super().__init__()
        self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(local_config)) if local_config else AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.15)
        self.heads = nn.ModuleDict({
            "es": nn.Linear(hidden, number_of_labels),
            "en": nn.Linear(hidden, number_of_labels),
        })

    def forward(self, input_ids, attention_mask, language):
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(encoded.last_hidden_state[:, 0])
        return self.heads[language](pooled)
