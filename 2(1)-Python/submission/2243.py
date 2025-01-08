from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Callable, List


"""
TODO:
- SegmentTree 구현하기
"""


T = TypeVar("T")
U = TypeVar("U")

class SegmentTree(Generic[T, U]):
    """
    - data: List[T]             -> 원본 배열(크기 n)
    - default: U                -> 세그먼트 트리에 사용될 기본값
    - convert: Callable[[T], U] -> 원본배열 원소(T)를 트리 노드타입(U)으로 변환하는 함수
    - combine: Callable[[U, U], U] -> 두 노드를 병합할 때 사용할 함수
    """

    def __init__(
        self,
        data: List[T],
        default: U,
        convert: Callable[[T], U],
        combine: Callable[[U, U], U]
    ) -> None:
        """
        세그먼트 트리 초기화
          - n: 원본 배열 크기
          - size: 2의 거듭제곱 (n 이상)
          - tree: 1-based 인덱스 (길이=2*size)
            * leaves: [size, size + n - 1]
            * 내부 노드: [1 .. size-1]
        """
        self.n = len(data)
        self.default = default
        self.convert = convert
        self.combine = combine

        self.size = 1
        while self.size < self.n:
            self.size <<= 1  

        self.tree = [default] * (2 * self.size)

        for i in range(self.n):
            self.tree[self.size + i] = self.convert(data[i])

        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self.combine(self.tree[i << 1], self.tree[i << 1 | 1])

    def update(self, idx: int, value: T) -> None:
        """
        단일 위치 idx(0-based 아님)에 대해
        '새로운 값'으로 세팅 (기존 값에 더하거나 빼는 것이 아님!)
        """
        idx += self.size
        self.tree[idx] = self.convert(value)

        idx >>= 1
        while idx >= 1:
            self.tree[idx] = self.combine(self.tree[idx << 1], self.tree[idx << 1 | 1])
            idx >>= 1

    def query(self, left: int, right: int) -> U:
        """
        [left, right) 구간의 병합 결과(U)를 반환
        """
        res = self.default
        left += self.size
        right += self.size

        while left < right:
            if left & 1:
                res = self.combine(res, self.tree[left])
                left += 1
            if right & 1:
                right -= 1
                res = self.combine(res, self.tree[right])
            left >>= 1
            right >>= 1
        return res

    def find_kth(self, k: int) -> int:
        """
        'k번째 사탕'의 맛(인덱스)을 찾기 위한 함수
        - 트리에 저장된 값은 '해당 구간의 사탕 개수 합'
        - 루트(1)부터 시작
          * 왼쪽 자식의 합 >= k 이면 왼쪽으로 이동
          * 아니면 k에서 왼쪽 자식 합을 빼고 오른쪽 자식으로 이동
        - 리프에 도달하면, 그 리프 인덱스 - (size - 1)이 '맛' (1-based)
        """
        idx = 1
        while idx < self.size:
            left = idx << 1
            right = left | 1
            left_sum = self.tree[left]
            if left_sum >= k:  # type: ignore
                idx = left
            else:
                k -= left_sum  # type: ignore
                idx = right
        return idx - (self.size - 1)

    def __repr__(self) -> str:
        return f"SegmentTree(size={self.size}, n={self.n}, tree={self.tree})"



import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""


def main() -> None:
    input = sys.stdin.readline

    n = int(input())  #
    MAX_TASTE = 1_000_000

    data = [0] * (MAX_TASTE + 1)

    st = SegmentTree(
        data=data,             
        default=0,              
        convert=lambda x: x,    
        combine=lambda a, b: a + b  
    )

    for _ in range(n):
        line = list(map(int, input().split()))
        if line[0] == 1:
            _, B = line
            taste = st.find_kth(B)  
            print(taste)
            curr_count = st.query(taste, taste + 1)
            st.update(taste, curr_count - 1)

        else:
            _, B, C = line
            curr_count = st.query(B, B + 1)
            st.update(B, curr_count + C)

if __name__ == "__main__":
    main()