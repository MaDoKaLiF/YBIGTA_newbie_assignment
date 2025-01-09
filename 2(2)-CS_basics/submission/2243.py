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

    (1-based 인덱스 사용 예시)
    - update(1, 10) : '맛=1'인 리프 노드에 값을 세팅/변경
    - query(1, 3)   : [1,3) 구간 병합 결과
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
            * 내부 노드: [1..size-1]
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
        단일 위치 idx(1-based)에 대해
        '새로운 값'으로 세팅 (기존 값에 더하거나 빼는 것이 아님!)
        예) update(5, 10) -> 맛=5인 리프에 새 값=10으로 설정
        """
        pos = self.size + (idx - 1)
        self.tree[pos] = self.convert(value)

        pos >>= 1
        while pos >= 1:
            self.tree[pos] = self.combine(self.tree[pos << 1], self.tree[pos << 1 | 1])
            pos >>= 1

    def query(self, left: int, right: int) -> U:
        """
        [left, right) 구간의 병합 결과(U)를 반환 (1-based)
        
        예) query(1, 2) -> [1,2) 즉 "맛=1"만 포함
            query(2, 5) -> [2,5) => 맛2, 맛3, 맛4
        """
        res = self.default
        left = self.size + (left - 1)
        right = self.size + (right - 1)

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
        'k번째 사탕'의 맛(1-based)을 찾는 함수
        - 트리에 저장된 값은 구간 합(사탕 개수)
        - 루트(인덱스 1)부터 내려가며
          left_sum = self.tree[left_child]
          if left_sum >= k:  왼쪽 자식으로
          else: k -= left_sum 후 오른쪽 자식으로
        - 리프 도달 시, 맛 번호 = (idx - (size - 1))
        """
        idx = 1
        while idx < self.size:
            left = idx << 1
            left_sum = self.tree[left]
            if left_sum >= k: # type: ignore
                idx = left
            else:
                k -= left_sum # type: ignore
                idx = left + 1
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

    n = int(input()) 
    MAX_TASTE = 1_000_000

    data = [0]*(MAX_TASTE + 1)

    st = SegmentTree(
        data=data,
        default=0,
        convert=lambda x: x,
        combine=lambda a,b: a+b
    )

    for _ in range(n):
        cmd = list(map(int, input().split()))
        if cmd[0] == 1:
            _, B = cmd
            taste = st.find_kth(B)   
            print(taste)
            curr_cnt = st.query(taste, taste+1)  
            st.update(taste, curr_cnt - 1)       

        else:
            _, B, C = cmd
            curr_cnt = st.query(B, B+1)
            st.update(B, curr_cnt + C)

if __name__ == "__main__":
    main()
