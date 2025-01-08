from lib import SegmentTree
import sys


"""
TODO:
- 일단 SegmentTree부터 구현하기
- main 구현하기
"""

def main() -> None:
    input = sys.stdin.readline
    T = int(input()) 

    for _ in range(T):
        n, m = map(int, input().split())
        to_watch = list(map(int, input().split()))
        '''
        size : 세그먼트 트리 용량: n + m (DVD 최대 개수 + 시청 횟수)
        각 DVD의 현재 위치 pos[dvd], 1-based 인덱스
        큰 인덱스일수록 물리적으로 더 '위'로 가정.
        예) n=3 -> DVD1=pos[1]=3, DVD2=pos[2]=2, DVD3=pos[3]=1
        DVD i -> 초기 위치 (n-i+1)
        '''
        size = n + m  
        pos = [0]*(n+1)
        for dvd in range(1, n+1):
            pos[dvd] = n - dvd + 1  
        '''
        세그먼트 트리 초기 배열: 모두 0 (크기 size+1, 1-based 사용)
        초기 DVD 배치: pos[dvd] 위치에 1
        '''
        data = [0]*(size+1)
        for dvd in range(1, n+1):
            data[pos[dvd]] = 1

        st = SegmentTree(
            data = data,
            default = 0,
            convert = lambda x: x,
            combine = lambda a,b: a+b
        )

        top = n  

        '''
        x 위에 놓인 DVD 개수 = 구간 합 [pos[x]+1, size+1)
        1) x DVD 제거
        2) top 한 칸 올리고, 그 위치에 x DVD 올림
        3) x의 위치 갱신
        '''
        result: list[str] = []

        for x in to_watch:
            above = st.query(pos[x]+1, size+1)
            if len(result)==0: 
            	above-=1	
            result.append(str(above))
            st.update(pos[x], 0)
            top += 1
            st.update(top, 1)
            pos[x] = top
        print(" ".join(result))


if __name__ == "__main__":
    main()
