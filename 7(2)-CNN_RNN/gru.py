import torch
from torch import nn, Tensor


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        """
        GRUCell의 파라미터를 정의합니다.
        여기서는 x(t)와 h(t-1)을 concat해서 한 번에 처리하는 방식을 사용합니다.
        
        Args:
            input_size (int): 입력 차원
            hidden_size (int): 은닉 상태(hidden state) 차원
        """
        super().__init__()
        self.hidden_size = hidden_size

        # Update gate, Reset gate, Candidate hidden state를 위한 레이어
        # (x_t, h_{t-1}) => z_t, r_t, h_t~ 변환
        # concat(x, h)에 대해 선형 변환을 각각 적용
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)
    
    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        """
        GRUCell 한 스텝 수행:
          z_t = sigmoid(W_z [x_t, h_{t-1}])
          r_t = sigmoid(W_r [x_t, h_{t-1}])
          h_t~ = tanh(W_h [x_t, r_t * h_{t-1}])
          h_t = (1 - z_t) * h_{t-1} + z_t * h_t~
        
        Args:
            x (Tensor):  [batch_size, input_size]
            h (Tensor):  [batch_size, hidden_size] (이전 시점 은닉 상태)

        Returns:
            h_next (Tensor): [batch_size, hidden_size] (현재 시점 은닉 상태)
        """
        # [x, h] concat
        combined = torch.cat([x, h], dim=-1)

        # Update gate (z_t)
        z_t = torch.sigmoid(self.W_z(combined))

        # Reset gate (r_t)
        r_t = torch.sigmoid(self.W_r(combined))

        # Candidate hidden (h_t~)
        combined_r = torch.cat([x, r_t * h], dim=-1)
        h_tilde = torch.tanh(self.W_h(combined_r))

        # 최종 h(t)
        h_next = (1 - z_t) * h + z_t * h_tilde
        return h_next


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        """
        단일 레이어 GRU를 구현한 모듈입니다.
        Args:
            input_size (int): 입력 차원
            hidden_size (int): 은닉 상태(hidden state) 차원
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)

    def forward(self, inputs: Tensor) -> Tensor:
        """
        전체 시퀀스에 대한 GRU 순전파:
        입력과 은닉 상태를 순회하며 매 타임스텝의 hidden state를 계산.

        Args:
            inputs (Tensor): [seq_len, batch_size, input_size] 형태의 입력 시퀀스

        Returns:
            outputs (Tensor): [seq_len, batch_size, hidden_size] 모든 타임스텝의 hidden state
        """
        seq_len, batch_size, _ = inputs.size()

        # 초기 hidden state: 0으로 초기화 (필요 시 전달 인자로 받아서 초기화 가능)
        h_t = torch.zeros(batch_size, self.hidden_size, device=inputs.device)

        outputs = []
        for t in range(seq_len):
            x_t = inputs[t]             # shape: [batch_size, input_size]
            h_t = self.cell(x_t, h_t)  # GRUCell 한 스텝
            outputs.append(h_t.unsqueeze(0))

        # 모든 타임스텝의 hidden state를 [seq_len, batch_size, hidden_size]로 cat
        outputs = torch.cat(outputs, dim=0)
        return outputs