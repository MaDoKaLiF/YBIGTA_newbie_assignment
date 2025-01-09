#!/bin/bash

# miniconda가 존재하지 않을 경우 설치
## TODO
if ! command -v conda &> /dev/null; then
    echo "Miniconda가 설치되어 있지 않습니다. 설치를 진행합니다."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    eval "$($HOME/miniconda/bin/conda shell.bash hook)"
    echo "Miniconda 설치 완료"
else
    echo "Miniconda가 이미 설치되어 있습니다."
fi

# Conda 환경 생성 및 활성화
## TODO
if ! conda info --envs | grep -q "myenv"; then
    echo "Conda 환경 'myenv'를 생성합니다."
    conda create -y -n myenv python=3.8
fi
conda activate myenv


## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "가상환경 활성화: 성공"
else
    echo "가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
## TODO
pip install mypy 

# Submission 폴더 파일 실행
cd submission || { echo "submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    input_file="../${file%.py}_input"
    output_file="../output/${file%.py}_output"
    
    if [[ -f "$input_file" ]]; then
        echo "실행 중: $file (input: $input_file)"
        python "$file" < "$input_file" > "$output_file"
        echo "결과 저장: $output_file"
    else
        echo "Input 파일이 없어 실행할 수 없습니다: $file"
    fi
done

# mypy 테스트 실행행
## TODO
echo "mypy 테스트를 실행합니다."
mypy *.py || echo "mypy 테스트에서 일부 오류가 발생했습니다."

# 가상환경 비활성화
## TODO
conda deactivate
echo "가상환경 비활성화 완료"