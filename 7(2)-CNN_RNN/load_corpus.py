# 구현하세요!

def load_corpus() -> list[str]:
    """
    IMDb 영화 리뷰 데이터셋(train split)에서 앞쪽 N개(예: 100개) 리뷰 텍스트만
    list[str] 형태로 읽어와 반환합니다.
    """
    from datasets import load_dataset

    # IMDb의 train split 불러오기
    dataset = load_dataset("imdb", split="train")

    # 전체 25,000개 중, 앞쪽 N개만 선택 (예: 100개)
    N = 10
    dataset = dataset.select(range(N))

    # "text" 필드를 추출하여 list[str] 형태로 변환
    corpus = [sample["text"] for sample in dataset]

    return corpus
