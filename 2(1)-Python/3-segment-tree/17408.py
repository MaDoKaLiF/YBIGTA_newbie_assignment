from lib import SegmentTree
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
