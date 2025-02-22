from lib import SegmentTree
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
