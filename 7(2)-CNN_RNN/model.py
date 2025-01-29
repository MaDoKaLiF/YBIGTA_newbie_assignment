from torch import nn, Tensor, LongTensor
from gru import GRU


class MyGRULanguageModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        num_classes: int,
        embeddings: int | Tensor
    ) -> None:
        super().__init__()
        if isinstance(embeddings, int):
            self.embeddings = nn.Embedding(embeddings, d_model)
        elif isinstance(embeddings, Tensor):
            self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True)

        self.gru = GRU(d_model, hidden_size)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, input_ids: LongTensor) -> Tensor:
        inputk = self.embeddings(input_ids)
        inputk = inputk.transpose(0, 1)

        last_hidden_state = self.gru(inputk)
        last_hidden_state=last_hidden_state[-1]
        logits = self.head(last_hidden_state)
        return logits