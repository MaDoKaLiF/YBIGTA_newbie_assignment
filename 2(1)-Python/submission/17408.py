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
        self.n = len(data) -1
        self.default = default
        self.convert = convert
        self.combine = combine

        self.size = 1
        while self.size < self.n+1:
            self.size <<= 1

        self.tree = [default] * (2 * self.size)

        
        for i in range(self.n+1):
            self.tree[self.size + i] = self.convert(data[i])

        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self.combine(self.tree[i << 1], self.tree[i << 1 | 1])

    def update(self, idx: int, value: T) -> None:
        """
        단일 위치 idx(1-based)에 대해
        '새로운 값'으로 세팅 (기존 값에 더하거나 빼는 것이 아님!)
        예) update(5, 10) -> 맛=5인 리프에 새 값=10으로 설정
        """
        pos = self.size + (idx)
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
            if left_sum >= k:
                idx = left
            else:
                k -= left_sum
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


class Pair(tuple[int, int]):
    """
    힌트: 2243, 3653에서 int에 대한 세그먼트 트리를 만들었다면 여기서는 Pair에 대한 세그먼트 트리를 만들 수 있을지도...?
    """
    def __new__(cls, a: int, b: int) -> 'Pair':
        return super().__new__(cls, (a, b))

    @staticmethod
    def default() -> 'Pair':
        """
        기본값
        이게 왜 필요할까...?
        """
        return Pair(0, 0)

    @staticmethod
    def f_conv(w: int) -> 'Pair':
        """
        원본 수열의 값을 대응되는 Pair 값으로 변환하는 연산
        이게 왜 필요할까...?
        """
        return Pair(w, 0)

    @staticmethod
    def f_merge(a: Pair, b: Pair) -> 'Pair':
        """
        두 Pair를 하나의 Pair로 합치는 연산
        이게 왜 필요할까...?
        """
        return Pair(*sorted([*a, *b], reverse=True)[:2])

    def sum(self) -> int:
        return self[0] + self[1]

def main() -> None:
    input = sys.stdin.readline

    N = int(input().strip())             
    arr = list(map(int, input().split())) 
    M = int(input().strip())            

    '''
    세그먼트 트리 생성
    - 1-based 인덱스로 사용하기 위해, data 길이를 N+1로 해서
    data[1..N] 구간에 수열을 매핑 (data[0]는 dummy)
    '''
    data = [0]*(N+1)
    for i in range(1, N+1):
        data[i] = arr[i-1]
    '''
    # SegmentTree에 들어갈 배열은 Pair 변환
       default = Pair(0,0)
      convert = Pair.f_conv
       combine = Pair.f_merge
    '''
    st = SegmentTree(
        data=data,                
        default=Pair.default(),
        convert=Pair.f_conv,
        combine=Pair.f_merge
    )

    '''
    쿼리 처리
    1 i v -> Ai = v -> 세그먼트 트리에서 (i) 위치를 (v)로 교체
    2 l r -> [l, r] 구간에서 Ai + Aj 최대값
        ->st.query(l, r+1) => [l, r+1) = [l, r]
        ->병합 결과가 Pair(a, b) -> a+b가 최대 두 수의 합
    '''
    output = []
    for _ in range(M):
        line = list(map(int, input().split()))
        if line[0] == 1:
            _, i, v = line
            st.update(i, v)  
        else:
            _, l, r = line
            ans_pair = st.query(l, r+1)
            output.append(str(ans_pair.sum()))

    print("\n".join(output))


if __name__ == "__main__":
    main()
