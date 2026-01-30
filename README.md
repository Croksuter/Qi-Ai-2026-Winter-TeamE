# KKBox Churn Prediction Project

KKBox 음악 스트리밍 서비스의 **고객 이탈(Churn) 예측** 프로젝트입니다.

## 프로젝트 개요

### 배경
KKBox는 아시아 최대의 음악 스트리밍 서비스입니다. 이 프로젝트는 사용자의 구독 행동 데이터를 분석하여 **다음 달에 구독을 해지할 가능성이 있는 사용자**를 예측하는 것을 목표로 합니다.

### 비즈니스 가치
- **이탈 예측**: 이탈 가능성이 높은 고객을 사전에 파악
- **타겟 마케팅**: 예측된 이탈 고객에게 맞춤형 프로모션 제공
- **수익 보호**: 고객 이탈로 인한 매출 손실 방지

### 데이터 출처
[Kaggle - KKBox Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge)

---

## 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd KKBox

# Python 버전 확인 (3.13 이상 필요)
python --version

# 의존성 설치 (uv 사용)
uv sync
```

### 2. 데이터 준비

Kaggle에서 데이터를 다운로드하여 `data/csv/` 폴더에 배치합니다:

```
data/csv/
├── raw_train_v1.csv
├── raw_train_v2.csv
├── raw_transactions_v1.csv
├── raw_transactions_v2.csv
├── raw_user_logs_v1.csv
├── raw_user_logs_v2.csv
└── raw_members_v3.csv
```

### 3. 데이터 파이프라인 실행

```bash
# 전처리 파이프라인 실행
uv run python src/main.py

# 피처 엔지니어링 실행
uv run python src/feature-engineer.py
```

---

## 프로젝트 구조

```
KKBox/
├── README.md                 # 프로젝트 설명서 (현재 파일)
├── pyproject.toml           # 프로젝트 설정 및 의존성
├── src/
│   ├── main.py              # 데이터 전처리 파이프라인
│   ├── feature-engineer.py  # 피처 엔지니어링 파이프라인
│   └── code-guide.md        # 코드 스타일 가이드
├── data/
│   ├── csv/                 # 원본 CSV 파일 (gitignore)
│   ├── parquet/             # 처리된 Parquet 파일 (gitignore)
│   ├── analysis/            # 분석 결과 (히트맵, 차트 등)
│   └── data.duckdb          # DuckDB 데이터베이스 (gitignore)
└── .venv/                   # Python 가상환경 (gitignore)
```

---

## 데이터 파이프라인 상세

### 전체 흐름도

```
[CSV 파일들]
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  1. CSV → DuckDB 로드                                        │
│     • 대용량 CSV를 청크 단위로 읽어 DuckDB 테이블로 저장      │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  2. 테이블명 정규화                                          │
│     • train → train_v1 (버전 명시)                          │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  3. v1/v2 병합                                               │
│     • train_v1 + train_v2 → train_merge                     │
│     • transactions, user_logs, members 동일하게 처리        │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  4. user_id 매핑 생성                                        │
│     • msno(문자열 해시) → user_id(정수) 변환                 │
│     • 메모리 효율 및 조인 성능 향상                          │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  5. 데이터 필터링                                            │
│     • 5-1. 공통 user_id 교집합 필터링                        │
│     • 5-2. 기준 테이블 기반 필터링                           │
│     • 5-3. Churn 전이 기반 필터링 (v1=0 → v2=0/1)           │
│     • 5-4. 중복 트랜잭션 유저 제외                           │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  6~8. 데이터 변환                                            │
│     • gender: male/female → 0/1 정수 변환                   │
│     • 날짜: YYYYMMDD 정수 → DATE 타입 변환                  │
│     • 컬럼명: msno → user_id 통일                           │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  9. Parquet 내보내기                                         │
│     • 압축된 Parquet 포맷으로 저장 (zstd)                    │
│     • ML 모델 학습에 바로 사용 가능                          │
└─────────────────────────────────────────────────────────────┘
```

### 주요 테이블 설명

| 테이블명 | 설명 | 주요 컬럼 |
|---------|------|----------|
| `train_merge` | 학습 레이블 (이탈 여부) | user_id, is_churn |
| `transactions_merge` | 결제 트랜잭션 기록 | user_id, payment_method_id, plan_list_price, transaction_date |
| `user_logs_merge` | 일별 사용 로그 | user_id, date, num_25, num_50, num_75, num_100, total_secs |
| `members_merge` | 회원 정보 | user_id, city, bd (나이), gender, registered_via |

### 분석 함수

| 함수명 | 설명 | 출력 |
|--------|------|------|
| `analyze_churn_transition()` | v1→v2 이탈 상태 전이 분석 | 3x3 히트맵 |
| `analyze_duplicate_transactions()` | 중복 트랜잭션 유저 분석 | 파이 차트 |
| `analyze_feature_correlation()` | 피처 간 상관관계 분석 | 상관관계 히트맵 |

---

## 핵심 개념 설명 (비전공자용)

### Churn (이탈)이란?
- 구독 서비스에서 **고객이 구독을 해지**하는 것
- `is_churn = 1`: 이탈함 (구독 해지)
- `is_churn = 0`: 유지함 (구독 지속)

### 왜 v1, v2가 있나요?
- **v1**: 2017년 2월 기준 데이터
- **v2**: 2017년 3월 기준 데이터
- 2월(v1)에 유지했던 고객이 3월(v2)에 이탈하는 패턴을 예측

### user_logs의 num_25, num_50 등은 무엇인가요?
- `num_25`: 곡의 25% 미만 재생 후 스킵한 횟수
- `num_50`: 곡의 25~50% 재생 후 스킵한 횟수
- `num_75`: 곡의 50~75% 재생 후 스킵한 횟수
- `num_985`: 곡의 75~98.5% 재생 후 스킵한 횟수
- `num_100`: 곡을 끝까지 재생한 횟수
- `num_unq`: 재생한 고유 곡 수
- `total_secs`: 총 재생 시간 (초)

### DuckDB란?
- **파일 기반 분석용 데이터베이스**
- SQLite처럼 가볍지만 **대용량 분석에 최적화**
- 별도 서버 설치 없이 파일 하나로 동작

---

## 기술 스택

| 구분 | 기술 | 버전 |
|------|------|------|
| Language | Python | 3.13+ |
| Package Manager | uv | - |
| Database | DuckDB | 1.4+ |
| Data Processing | Pandas | 2.3+ |
| Visualization | Matplotlib, Seaborn | - |
| Progress Bar | tqdm | - |

---

## 개발 가이드 (팀원용)

### 코드 스타일
프로젝트의 코드 스타일은 `src/code-guide.md`를 참고하세요.

주요 규칙:
- **타입 힌트** 필수
- **Google 스타일 Docstring** 사용
- **로깅**: 작업 시작/완료 시 `=== 작업명 ===` 형식
- **숫자 포맷**: `{value:,}` (천 단위 콤마)

### 파이프라인 단계별 실행

`src/main.py`의 `__main__` 블록에서 주석을 해제하여 특정 단계만 실행할 수 있습니다:

```python
if __name__ == "__main__":
    DB_PATH = "data/data.duckdb"

    # 1단계만 실행하려면:
    load_csv_to_duckdb(...)

    # 나머지는 주석 처리
    # rename_tables_add_v1_suffix(...)
    # create_merge_tables(...)
```

### 새 함수 추가 시

1. 적절한 섹션 번호 부여 (예: 5-8, 6 등)
2. 구분선 추가:
   ```python
   # ============================================================================
   # 섹션번호. 함수 설명
   # ============================================================================
   ```
3. 타입 힌트와 Docstring 작성
4. 작업 전/후 로그 출력

### 분석 결과 저장 위치
- 이미지: `data/analysis/`
- CSV: `data/analysis/`

---

## 알려진 데이터 이슈

### 1. total_secs 오버플로우
- **문제**: 일부 행에서 `total_secs`가 INT64 최대값 (9.22e+15)
- **원인**: 원본 데이터 오류 또는 오버플로우
- **해결**: `nullify_out_of_range()` 함수로 0~86400 범위 밖 값을 NULL 처리

### 2. bd (나이) 컬럼의 잘못된 해석
- **주의**: `bd`는 생년(birth year)이 아니라 **나이(age)**
- `bd = 0`: 나이 정보 없음 (결측)
- `bd = 27`: 27세

### 3. gender 결측 표시
- `gender = -1`: 성별 정보 없음
- `gender = 0`: 여성
- `gender = 1`: 남성

---

## 자주 묻는 질문 (FAQ)

### Q: 데이터 처리에 얼마나 걸리나요?
전체 파이프라인 실행 시 약 **30분~1시간** 소요 (하드웨어에 따라 다름)

### Q: DuckDB 파일이 너무 큰데요?
처리된 `data.duckdb` 파일은 약 **35GB**입니다. SSD 사용을 권장합니다.

### Q: 특정 단계만 다시 실행하고 싶어요
`src/main.py`에서 해당 함수의 주석만 해제하고 실행하세요.

### Q: 메모리가 부족해요
- `chunksize` 파라미터를 줄이세요 (기본 1,000,000 → 500,000)
- `sample_size` 옵션으로 샘플링하여 분석

---

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 사용됩니다.
데이터는 [Kaggle Competition Rules](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/rules)를 따릅니다.

---

## 기여자

- 2026 QI AI Winter 팀
