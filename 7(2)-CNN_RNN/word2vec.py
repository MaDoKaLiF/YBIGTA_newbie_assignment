import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam

from transformers import PreTrainedTokenizer

from typing import Literal

# 구현하세요!


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method
        # 구현하세요!
        pass

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        # 구현하세요!
        for epoch in range(num_epochs):
            total_loss = 0.0

            for sentence in corpus:
                tokens = tokenizer.encode(sentence, add_special_tokens=False,  truncation=True)
                print(f"[sentence {sentence}/{len(corpus)}]")
                if self.method == "skipgram":
                    loss = self._train_skipgram(tokens, optimizer, criterion)
                else:  # cbow
                    loss = self._train_cbow(tokens, optimizer, criterion)

                total_loss += loss
            
            print(f"[Epoch {epoch+1}/{num_epochs}] Loss = {total_loss:.4f}")

    def _train_skipgram(
        self,
        tokens: list[int],
        optimizer: Adam,
        criterion: nn.CrossEntropyLoss
    ) -> float:
        """
        Skip-Gram:
        center -> context를 예측하도록 학습
        """
        loss_val = 0.0
        
        for i in range(len(tokens)):
            center = tokens[i]
            # window_size 주변 단어들을 context로 삼음
            for w in range(-self.window_size, self.window_size + 1):
                print(f"[Epoch {i}/{w}]")
                if w == 0:
                    continue
                context_idx = i + w
                if context_idx < 0 or context_idx >= len(tokens):
                    continue
                
                context = tokens[context_idx]

                # 1) 순전파
                optimizer.zero_grad()
                center_emb = self.embeddings(torch.tensor([center]))   # shape: [1, d_model]
                logits = self.weight(center_emb)                       # shape: [1, vocab_size]

                # 2) 로스 계산
                # Skip-Gram에서는 (center -> context)을 CrossEntropy 로 학습
                # logits.shape=[1, vocab_size], target.shape=[1]
                loss = criterion(logits, torch.tensor([context]))
                
                # 3) 역전파 & 업데이트
                loss.backward()
                optimizer.step()

                loss_val += loss.item()
        return loss_val

    def _train_cbow(
        self,
        tokens: list[int],
        optimizer: Adam,
        criterion: nn.CrossEntropyLoss
    ) -> float:
        """
        CBOW(Continuous Bag-of-Words):
        주변 단어들의 임베딩을 합(또는 평균) -> center 단어를 예측
        """
        loss_val = 0.0

        for i in range(len(tokens)):
            center = tokens[i]
            
            # 주변 단어들의 임베딩 합을 구함
            context_emb_sum = torch.zeros((1, self.embeddings.embedding_dim))
            count_context = 0

            for w in range(-self.window_size, self.window_size + 1):
                if w == 0:
                    continue
                context_idx = i + w
                if context_idx < 0 or context_idx >= len(tokens):
                    continue

                context = tokens[context_idx]
                context_emb_sum += self.embeddings(torch.tensor([context]))
                count_context += 1

            # 실제 context가 없는 경우(문장 길이가 매우 짧은 경우) skip
            if count_context == 0:
                continue

            # 1) 순전파
            optimizer.zero_grad()
            # 보통은 평균을 취하거나, sum을 그대로 사용할 수 있음
            context_emb_avg = context_emb_sum / count_context  # shape: [1, d_model]
            logits = self.weight(context_emb_avg)              # shape: [1, vocab_size]

            # 2) 로스 계산
            # CBOW에서는 (context -> center)을 CrossEntropy 로 학습
            loss = criterion(logits, torch.tensor([center]))

            # 3) 역전파 & 업데이트
            loss.backward()
            optimizer.step()

            loss_val += loss.item()

        return loss_val