# KKBox Churn Prediction - Claude Reference Document

이 문서는 KKBox 프로젝트의 데이터 구조, 전처리 파이프라인, 테이블 스키마를 종합 정리한 참조 문서입니다.

---

## 1. 프로젝트 개요

### 1.1 목표
- KKBox 음악 스트리밍 서비스의 **사용자 이탈(Churn) 예측**
- 2017년 3월 기준 구독 해지 여부 예측

### 1.2 핵심 수치
| 항목 | 값 |
|------|-----|
| 총 유저 수 | 730,694명 |
| 이탈 유저 (is_churn=1) | 32,601명 (4.46%) |
| 유지 유저 (is_churn=0) | 698,093명 (95.54%) |
| 데이터 기간 | 2015-01-01 ~ 2017-03-31 |

### 1.3 데이터 출처
[Kaggle - KKBox Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge)

---

## 2. 테이블 스키마 상세

### 2.1 train_merge (타겟 레이블)

| 컬럼 | 타입 | 설명 | 비고 |
|------|------|------|------|
| `user_id` | BIGINT | 사용자 ID (PK) | 1 ~ 730,694 |
| `is_churn` | BIGINT | 이탈 여부 | 0=유지, 1=이탈 |

**생성 과정:**
- `train_v1` + `train_v2` 병합 (v2가 v1 덮어씀)
- Churn 전이 필터링 적용 (v1=0인 유저만)

---

### 2.2 members_merge (사용자 정보)

| 컬럼 | 타입 | 설명 | 값 범위/특징 |
|------|------|------|--------------|
| `user_id` | BIGINT | 사용자 ID (PK) | 1 ~ 730,694 |
| `gender` | INTEGER | 성별 | -1=미상, 0=여성, 1=남성 |
| `city` | BIGINT | 도시 코드 | 1~21 |
| `bd` | BIGINT | 나이 | 이상치 다수 (0, 음수, 1000+ 등) |
| `registered_via` | BIGINT | 가입 경로 | 1~13 |
| `registration_init_time` | DATE | 가입일 | 2004-03-26 ~ 2017-03-30 |
| `last_expire` | DATE | 마지막 만료일 | 원본 members_v3에서 유래 |
| `membership_seq_group_id` | BIGINT | 멤버십 시퀀스 그룹 ID | (선택적 추가) |
| `membership_seq_id` | BIGINT | 그룹 내 순번 | (선택적 추가) |
| `previous_membership_duration` | BIGINT | 직전 구독 기간 (일) | (선택적 추가) |
| `previous_membership_seq_duration` | BIGINT | 시퀀스 누적 기간 (일) | (선택적 추가) |
| `last_expire_date` | DATE | 2017년 3월 effective 마지막 만료일 | 14.3에서 추가 |
| `last_seq_id` | BIGINT | 해당 트랜잭션의 sequence_group_id | 14.3에서 추가 |
| `p_tx_id` | BIGINT | 해당 트랜잭션의 sequence_id | 14.3에서 추가 |
| `pp_tx_id` | BIGINT | 직전 트랜잭션의 sequence_id | NULL=직전 없음 |
| `is_churn` | BIGINT | 이탈 여부 | 14.4에서 train_merge에서 복사 |

**주의사항:**
- `gender`: NULL 값 약 414,061명 존재
- `bd`: 이상치 정제 필요 (음수, 0, 100+ 등)
- `last_expire_date` 계산 로직:
  1. 시퀀스별 3월 내 최대 membership_expire_date 찾기
  2. 해당 트랜잭션 이후 cancel이 있으면 cancel의 expire_date 사용
  3. 유저별 가장 마지막 effective expire_date 선택

---

### 2.3 transactions_merge (결제 트랜잭션)

| 컬럼 | 타입 | 설명 | 값 범위/특징 |
|------|------|------|--------------|
| `user_id` | BIGINT | 사용자 ID | - |
| `payment_method_id` | BIGINT | 결제 수단 ID | - |
| `payment_plan_days` | BIGINT | 결제 플랜 기간 (일) | 주로 30, 90, 180, 365 |
| `plan_list_price` | BIGINT | 정가 | - |
| `actual_amount_paid` | BIGINT | 실결제액 | - |
| `is_auto_renew` | BIGINT | 자동 갱신 여부 | 0/1 |
| `transaction_date` | DATE | 거래일 | 2015-01-01 ~ 2017-03-31 |
| `membership_expire_date` | DATE | 멤버십 만료일 | - |
| `is_cancel` | BIGINT | 취소 여부 | 0=정상, 1=취소 |

**통계:**
- 총 트랜잭션: ~13M건 (필터링 후)
- 유저당 평균: ~18건

---

### 2.4 transactions_seq (트랜잭션 시퀀스)

`transactions_merge`의 모든 컬럼 + 추가 시퀀스 정보

| 추가 컬럼 | 타입 | 설명 | 계산 방식 |
|-----------|------|------|-----------|
| `sequence_group_id` | BIGINT | 연속 구독 그룹 ID | 이탈(30일 gap) 시 증가 |
| `sequence_id` | BIGINT | 그룹 내 순번 | 0부터 시작 |
| `before_transaction_term` | BIGINT | 직전 거래와의 간격 (일) | transaction_date - prev_transaction_date |
| `before_membership_expire_term` | BIGINT | 직전 만료일과의 간격 (일) | transaction_date - prev_membership_expire_date |
| `is_churn` | BIGINT | 트랜잭션 레벨 이탈 | before_membership_expire_term > 30이면 1 |
| `actual_plan_days` | BIGINT | 실제 플랜 일수 | membership_expire_date - prev_membership_expire_date |
| `plan_days_diff` | BIGINT | 플랜 일수 차이 | actual_plan_days - payment_plan_days |

**시퀀스 개념:**
```
User A 예시:
┌─────────────────────────────────────────────────────┐
│ Group 0: [seq 0] → [seq 1] → [seq 2]  (연속 구독)   │
│          만료 전 갱신, gap ≤ 30일                   │
├─────────────────────────────────────────────────────┤
│ (이탈: gap > 30일)                                  │
├─────────────────────────────────────────────────────┤
│ Group 1: [seq 0] → [seq 1]  (재구독)                │
└─────────────────────────────────────────────────────┘
```

**actual_plan_days 해석:**
- `is_cancel=0`: 양수 (정상 연장)
- `is_cancel=1`: 음수 또는 0 (취소로 인한 만료일 감소)

---

### 2.5 user_logs_merge (사용 로그)

| 컬럼 | 타입 | 설명 | 값 범위/특징 |
|------|------|------|--------------|
| `user_id` | BIGINT | 사용자 ID | - |
| `date` | DATE | 로그 날짜 | 2015-01-01 ~ 2017-03-31 |
| `num_25` | BIGINT | 25% 미만 재생 곡 수 | 스킵 |
| `num_50` | BIGINT | 25~50% 재생 곡 수 | 스킵 |
| `num_75` | BIGINT | 50~75% 재생 곡 수 | 스킵 |
| `num_985` | BIGINT | 75~98.5% 재생 곡 수 | 거의 완청 |
| `num_100` | BIGINT | 98.5%+ 재생 곡 수 | 완전 재생 |
| `num_unq` | BIGINT | 고유 곡 수 | - |
| `total_secs` | DOUBLE | 총 재생 시간 (초) | 이상치 존재 |
| `total_hours` | DOUBLE | 총 재생 시간 (시간) | total_secs/3600, [0,24] 클리핑 |

**통계:**
- 총 로그: ~219M건
- 유저당 평균: ~300일

**주의사항:**
- `total_secs`: 음수, 86400초(24시간) 초과 이상치 존재
- Feature Engineering 시 `total_hours` 사용 권장

---

### 2.6 user_id_map (ID 매핑)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `msno` | VARCHAR | 원본 해시 ID |
| `user_id` | BIGINT | 정수 ID (1부터 시작) |

---

## 3. 전처리 파이프라인 (src/main.py)

### 3.1 데이터 로드 및 병합 (1~4단계)

| 단계 | 함수 | 설명 |
|------|------|------|
| 1 | `load_csv_to_duckdb()` | CSV → DuckDB 로드 (청크 단위) |
| 2 | `rename_tables_add_v1_suffix()` | train, transactions, user_logs에 _v1 suffix 추가 |
| 3 | `create_merge_tables()` | v1/v2 병합 → _merge 테이블 생성 |
| 4 | `create_user_id_mapping()` | msno → user_id 정수 매핑 생성 |

**병합 전략:**
- `train`: v2가 v1 덮어씀 (같은 user_id면 v2 값 사용)
- `members`: members_v3 단독 사용
- 나머지: UNION ALL

---

### 3.2 데이터 필터링 (5~9단계)

| 단계 | 함수 | 제외 조건 |
|------|------|-----------|
| 5 | `filter_common_msno()` | 모든 테이블에 없는 유저 |
| 6 | `convert_gender_to_int()` | (변환만, 제외 없음) |
| 7 | `convert_date_fields()` | (변환만, 제외 없음) |
| 8 | `rename_msno_to_user_id()` | (컬럼명 변경만) |
| 9.1 | `filter_by_churn_transition()` | v1에서 is_churn≠0인 유저 |
| 9.2 | `exclude_duplicate_transaction_users()` | 같은 날 비취소 트랜잭션 2+개 유저 |
| 9.3 | `exclude_null_total_secs_users()` | total_secs가 NULL인 로그 있는 유저 |
| 9.5 | `exclude_out_of_range_expire_date_users()` | expire_date가 2015~2017 범위 밖 유저 |

---

### 3.3 시퀀스 생성 및 분석 (10~14단계)

| 단계 | 함수 | 설명 |
|------|------|------|
| 10 | `create_transactions_seq()` | 시퀀스 테이블 생성 (gap_days=30) |
| 11.1 | `add_actual_plan_days()` | actual_plan_days 컬럼 추가 |
| 11.4 | `exclude_out_of_range_actual_plan_days_users()` | 비정상 actual_plan_days 유저 제외 |
| 12.1 | `add_plan_days_diff()` | plan_days_diff 컬럼 추가 |
| 13.2 | `exclude_cancel_expire_before_prev_transaction_users()` | cancel인데 expire < prev_tx_date 유저 제외 |
| 13.3 | `exclude_cancel_expire_before_prev_prev_expire_users()` | cancel인데 expire < prev_prev_expire 유저 제외 |
| 14.2 | `exclude_consecutive_cancel_users()` | 연속 cancel 트랜잭션 유저 제외 |
| 14.3 | `add_march_2017_last_txn_info()` | members에 3월 마지막 트랜잭션 정보 추가 |
| 14.4 | `add_is_churn_to_members()` | members에 is_churn 컬럼 추가 |

---

## 4. 핵심 비즈니스 로직

### 4.1 Churn 전이 필터링

```
v1(2월) → v2(3월) 전이 중 유효한 케이스:
✓ v1=0 → v2=0 (유지 → 유지)
✓ v1=0 → v2=1 (유지 → 이탈)
✗ v1=1 → v2=0 (이탈 → 유지) - 재가입 케이스, 제외
✗ v1=1 → v2=1 (이탈 → 이탈) - 이미 이탈, 제외
```

### 4.2 시퀀스 그룹 결정 로직

```python
# gap_days = 30 (기본값)
if before_membership_expire_term > gap_days:
    # 이탈로 간주, 새 시퀀스 그룹 시작
    sequence_group_id += 1
    sequence_id = 0
else:
    # 연속 구독
    sequence_id += 1
```

### 4.3 actual_plan_days 계산

```sql
actual_plan_days = membership_expire_date - LAG(membership_expire_date)
                   OVER (PARTITION BY user_id ORDER BY transaction_date, sequence_id)

-- 첫 트랜잭션: membership_expire_date - transaction_date
```

**유효 범위:**
- `is_cancel=0`: 1 ~ 410일
- `is_cancel=1`: -410 ~ 0일

### 4.4 2017년 3월 마지막 트랜잭션 결정 (14.3)

```
1. 각 시퀀스에서 membership_expire_date가 3월인 트랜잭션 중 최대값 찾기
2. 해당 트랜잭션 이후에 cancel 트랜잭션이 있는지 확인
3. cancel이 있으면 → cancel의 expire_date를 effective로 사용
   예: 03-25까지 연장 → cancel로 03-13 → effective = 03-13
4. 유저별로 가장 마지막 effective expire_date를 가진 시퀀스 선택
```

---

## 5. 데이터 품질 이슈 및 처리

### 5.1 알려진 이상치

| 테이블 | 컬럼 | 이슈 | 처리 |
|--------|------|------|------|
| user_logs_merge | total_secs | 음수, 86400초 초과 | NULL 변환 또는 유저 제외 |
| members_merge | bd | 음수, 0, 1000+ | 클리핑 또는 NULL 처리 필요 |
| members_merge | gender | NULL 다수 | -1로 표시됨 |
| transactions | actual_plan_days | 범위 밖 | 유저 제외 (11.4) |

### 5.2 제외된 유저 유형

1. **Churn 전이 비정상**: v1에서 이미 이탈한 유저
2. **중복 트랜잭션**: 같은 날 비취소 트랜잭션 2개 이상
3. **이상 expire_date**: 2015~2017 범위 밖
4. **이상 actual_plan_days**: 비정상 플랜 일수
5. **비정상 cancel**: expire < prev_transaction_date
6. **연속 cancel**: 2개 이상 연속 취소 트랜잭션

---

## 6. 파일 구조

```
KKBox/
├── CLAUDE.md                    # 이 문서
├── README.md                    # 프로젝트 개요
├── codex.md                     # 코드베이스 노트
├── pyproject.toml               # 의존성 (uv)
├── src/
│   ├── main.py                  # 전처리 파이프라인 (핵심)
│   ├── utils.py                 # 범용 유틸리티
│   ├── feature-engineer-for-ml.py   # ML 피처 생성
│   ├── feature-engineer-for-daily.py # 일별 피처 생성
│   ├── code-guide.md            # 코드 스타일 가이드
│   ├── structure-guide.md       # 구조 가이드
│   ├── feature-engineer-plan-for-ml.md # 피처 설계 문서
│   └── feature-formulas.md      # 피처 계산식
└── data/
    ├── data.duckdb              # DuckDB 데이터베이스 (~35GB)
    ├── csv/                     # 원본 CSV
    ├── parquet/                 # Parquet 출력
    └── analysis/                # 분석 결과 그래프
```

---

## 7. 실행 방법

```bash
# 환경 설정
uv sync

# 전처리 실행
uv run python src/main.py

# 피처 엔지니어링
uv run python src/feature-engineer-for-ml.py
```

---

## 8. 코드 작성 규칙

### 8.1 네이밍
- 함수: `snake_case`, 동사로 시작 (`create_`, `filter_`, `add_`)
- 변수: `snake_case`, 타입 힌트 접미사 (`_count`, `_table`, `_col`)
- 상수: `UPPER_SNAKE_CASE`

### 8.2 로깅
```python
logger.info(f"=== 작업명 시작 ===")
logger.info(f"  세부 정보: {value:,}")  # 숫자는 천 단위 콤마
logger.info(f"=== 작업명 완료 ===\n")
```

### 8.3 DuckDB 패턴
```python
con = duckdb.connect(db_path)
con.execute("PRAGMA threads=8;")

# 임시 테이블 패턴
con.execute(f"CREATE OR REPLACE TABLE {table}_temp AS ...")
con.execute(f"DROP TABLE {table}")
con.execute(f"ALTER TABLE {table}_temp RENAME TO {table}")

con.close()
```

### 8.4 섹션 구분
```python
# ============================================================================
# 섹션번호. 함수 설명
# ============================================================================
def function_name(...) -> None:
    """Google 스타일 docstring"""
```

---

## 9. 주요 쿼리 패턴

### 9.1 유저별 최신 트랜잭션
```sql
WITH ranked AS (
    SELECT *, ROW_NUMBER() OVER (
        PARTITION BY user_id
        ORDER BY transaction_date DESC, sequence_id DESC
    ) AS rn
    FROM transactions_seq
)
SELECT * FROM ranked WHERE rn = 1
```

### 9.2 시퀀스 내 집계
```sql
SELECT
    user_id,
    sequence_group_id,
    COUNT(*) AS txn_count,
    MIN(transaction_date) AS seq_start,
    MAX(membership_expire_date) AS seq_end
FROM transactions_seq
GROUP BY user_id, sequence_group_id
```

### 9.3 LAG 함수 활용
```sql
SELECT
    *,
    LAG(membership_expire_date) OVER (
        PARTITION BY user_id
        ORDER BY transaction_date, sequence_id
    ) AS prev_expire_date
FROM transactions_seq
```

---

## 10. 구현 지침 상세

### 10.1 `__main__` 블록 구조

```python
# ============================================================================
# 메인 실행
# ============================================================================
if __name__ == "__main__":
    # ========================================================================
    # 상수 설정
    # ========================================================================
    DB_PATH = "data/data.duckdb"
    CSV_DIR = "data/csv"
    PARQUET_DIR = "data/parquet"

    # ========================================================================
    # 파이프라인 실행 (각 줄을 주석처리하여 특정 단계 건너뛰기 가능)
    # ========================================================================

    # 1. 단계 설명
    # function_name(
    #     db_path=DB_PATH,
    #     param1=value1,
    # )

    # 2. 활성화된 단계
    another_function(
        db_path=DB_PATH,
        param1=value1,
        num_samples=5,
        dry_run=False,
    )
```

**핵심 원칙:**
- 모든 함수 호출은 **명시적 키워드 인자** 사용 (`param_name=value`)
- 파라미터마다 **줄바꿈**, 마지막에 **trailing comma**
- 주석 처리로 **단계별 선택 실행** 가능
- 상수는 `__main__` 블록 최상단에 정의

---

### 10.2 utils.py 분리 기준

**utils.py에 포함 (범용 함수):**
```python
# Database/Table Operations
drop_tables()                    # 테이블 삭제
copy_table()                     # 테이블 복사
rename_column()                  # 컬럼명 변경
add_column()                     # 컬럼 추가
table_exists()                   # 테이블 존재 여부
get_row_count()                  # 행 수 조회
show_database_info()             # DB 전체 정보
show_table_info()                # 테이블 상세 정보

# Data I/O
load_csv_to_duckdb()             # CSV → DuckDB
export_to_parquet()              # DuckDB → Parquet
export_to_csv()                  # DuckDB → CSV

# Data Transformation
nullify_out_of_range()           # 범위 밖 값 NULL 변환
apply_conditional_transform()    # 조건부 변환
add_converted_column()           # 단위 변환 컬럼 추가
fill_null_values()               # NULL 채우기

# Data Analysis
analyze_clipping_distribution()  # 클리핑 분포 분석
analyze_feature_correlation()    # 상관관계 분석
get_column_stats()               # 컬럼 통계
```

**main.py에 포함 (도메인 특화 함수):**
```python
# KKBox 비즈니스 로직
filter_by_churn_transition()          # Churn 전이 필터링
create_transactions_seq()             # 트랜잭션 시퀀스 생성
add_membership_seq_info()             # 멤버십 시퀀스 정보
exclude_consecutive_cancel_users()    # 연속 취소 유저 제외
add_march_2017_last_txn_info()        # 3월 마지막 트랜잭션 정보
```

**분리 기준:**
| 구분 | utils.py | main.py |
|------|----------|---------|
| 재사용성 | 다른 프로젝트에서도 사용 가능 | KKBox 전용 |
| 도메인 지식 | 불필요 | 필요 (Churn, 시퀀스 등) |
| 테이블 의존성 | 파라미터로 받음 | 특정 테이블 구조 가정 |

---

### 10.3 num_samples 패턴

**목적:** 처리 결과를 샘플로 확인하여 검증

```python
def some_function(
    db_path: str,
    table_name: str,
    num_samples: int | None = None,  # None이면 샘플 출력 안 함
) -> None:
    # ... 처리 로직 ...

    # 샘플 출력 (선택적)
    if num_samples is not None and num_samples > 0:
        sample = con.execute(f"""
            SELECT *
            FROM {table_name}
            WHERE some_condition
            ORDER BY some_column
            LIMIT {num_samples}
        """).fetchdf()
        logger.info(f"샘플 데이터 (상위 {num_samples}개):\n{sample.to_string()}")
```

**사용 예시:**
```python
# 샘플 없이 실행
process_data(db_path=DB_PATH, num_samples=None)

# 5개 샘플 확인
process_data(db_path=DB_PATH, num_samples=5)

# 디버깅 시 많은 샘플
process_data(db_path=DB_PATH, num_samples=20)
```

---

### 10.4 dry_run 패턴

**목적:** 실제 변경 없이 영향 범위 미리 확인

```python
def exclude_some_users(
    db_path: str,
    target_tables: list[str],
    num_samples: int | None = None,
    dry_run: bool = False,  # True면 실제 변경 없음
) -> None:
    logger.info(f"=== 작업 시작 ===")
    if dry_run:
        logger.info("*** DRY RUN 모드 - 실제 변경 없음 ***")

    con = duckdb.connect(db_path)

    # 영향받을 유저 찾기
    affected_users = con.execute("""
        SELECT user_id FROM ... WHERE ...
    """).fetchall()

    logger.info(f"제외 대상 유저: {len(affected_users):,}명")

    # dry_run이면 여기서 종료
    if dry_run:
        if num_samples:
            sample = con.execute(f"SELECT ... LIMIT {num_samples}").fetchdf()
            logger.info(f"샘플 (dry_run):\n{sample.to_string()}")
        con.close()
        logger.info(f"=== 작업 완료 (dry_run) ===\n")
        return

    # 실제 삭제 수행
    for table in target_tables:
        before = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        con.execute(f"DELETE FROM {table} WHERE user_id IN (...)")
        after = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        logger.info(f"  {table}: {before:,} → {after:,} ({before - after:,} 삭제)")

    con.close()
    logger.info(f"=== 작업 완료 ===\n")
```

**사용 패턴:**
```python
# 1단계: dry_run으로 영향 확인
exclude_some_users(
    db_path=DB_PATH,
    target_tables=["train_merge", "members_merge"],
    num_samples=10,
    dry_run=True,   # 먼저 확인
)

# 2단계: 확인 후 실제 실행
exclude_some_users(
    db_path=DB_PATH,
    target_tables=["train_merge", "members_merge"],
    num_samples=5,
    dry_run=False,  # 실제 실행
)
```

---

### 10.5 재사용 가능한 함수 설계

**파라미터 설계 원칙:**

```python
def flexible_function(
    # 1. 필수 파라미터 (기본값 없음)
    db_path: str,

    # 2. 대상 지정 파라미터 (기본값 있음)
    table_name: str = "default_table",
    target_tables: list[str] | None = None,

    # 3. 동작 조절 파라미터
    threshold: int = 30,
    min_value: int = 0,
    max_value: int = 100,

    # 4. 출력/디버깅 파라미터 (항상 마지막)
    num_samples: int | None = None,
    dry_run: bool = False,
) -> None:
```

**함수 시그니처 예시:**

```python
# 필터링 함수
def exclude_out_of_range_users(
    db_path: str,
    source_table: str,                    # 조건 검사할 테이블
    target_tables: list[str],             # 실제 필터링할 테이블들
    column_name: str,                     # 검사할 컬럼
    min_value: int | float | None = None, # 최소값 (None이면 체크 안 함)
    max_value: int | float | None = None, # 최대값 (None이면 체크 안 함)
    num_samples: int | None = None,
    dry_run: bool = False,
) -> None:

# 컬럼 추가 함수
def add_computed_column(
    db_path: str,
    table_name: str,
    source_col: str,                      # 원본 컬럼
    target_col: str,                      # 새 컬럼명
    expression: str | None = None,        # SQL 표현식 (None이면 단순 복사)
    divisor: float | None = None,         # 나누기 연산
    clip_min: float | None = None,        # 클리핑 최소
    clip_max: float | None = None,        # 클리핑 최대
    force_overwrite: bool = True,         # 기존 컬럼 덮어쓰기
) -> None:

# 분석 함수
def analyze_distribution(
    db_path: str,
    table_name: str,
    column_name: str,
    output_dir: str = "data/analysis",    # 출력 디렉토리
    sample_size: int | None = None,       # 샘플링 (대용량 테이블용)
    reference_table: str | None = None,   # 기준 테이블 (특정 유저만 분석)
) -> None:
```

---

### 10.6 함수 호출 구조 예시

**분석 → 필터링 → 검증 패턴:**

```python
if __name__ == "__main__":
    DB_PATH = "data/data.duckdb"

    # ========================================================================
    # Phase 1: 분석 (선택적, 주석 처리 가능)
    # ========================================================================

    # # 9.2.1. 중복 트랜잭션 분석
    # analyze_duplicate_transactions(
    #     db_path=DB_PATH,
    #     transactions_table="transactions_merge",
    #     output_dir="data/analysis",
    #     reference_table="members_merge",
    # )

    # ========================================================================
    # Phase 2: 필터링 (dry_run으로 먼저 확인 권장)
    # ========================================================================

    # 9.2.2. 중복 트랜잭션 유저 제외
    exclude_duplicate_transaction_users(
        db_path=DB_PATH,
        transactions_table="transactions_merge",
        target_tables=[
            "train_merge",
            "transactions_merge",
            "user_logs_merge",
            "members_merge",
        ],
        min_txn_count=2,
        num_samples=5,
        dry_run=False,  # True로 먼저 테스트
    )

    # ========================================================================
    # Phase 3: 검증
    # ========================================================================

    # 최종 데이터베이스 정보 출력
    show_database_info(db_path=DB_PATH)
```

---

### 10.7 에러 처리 패턴

```python
def safe_function(
    db_path: str,
    table_name: str,
) -> None:
    logger.info(f"=== 작업 시작 ===")

    con = duckdb.connect(db_path)

    # 1. 테이블 존재 확인
    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if table_name not in existing_tables:
        logger.error(f"테이블 {table_name}이 존재하지 않습니다.")
        con.close()
        return

    # 2. 컬럼 존재 확인
    cols = [row[0] for row in con.execute(f"DESCRIBE {table_name}").fetchall()]
    if "required_column" not in cols:
        logger.warning(f"  {table_name}: required_column 없음, 건너뜀")
        con.close()
        return

    # 3. 실제 작업 수행
    # ...

    con.close()
    logger.info(f"=== 작업 완료 ===\n")
```

---

### 10.8 target_tables 패턴

**여러 테이블에 동일 작업 적용:**

```python
def apply_to_multiple_tables(
    db_path: str,
    target_tables: list[str],
    # ...
) -> None:
    con = duckdb.connect(db_path)
    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    for table in target_tables:
        # 존재 확인
        if table not in existing_tables:
            logger.warning(f"  {table}: 존재하지 않음, 건너뜀")
            continue

        # 작업 전 상태
        before = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        # 작업 수행
        con.execute(f"DELETE FROM {table} WHERE ...")

        # 작업 후 상태
        after = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        # 결과 로깅
        logger.info(f"  {table}: {before:,} → {after:,} ({before - after:,} 변경)")

    con.close()
```

**표준 target_tables 목록:**

```python
# 모든 _merge 테이블 (일반적인 필터링)
target_tables=[
    "train_merge",
    "transactions_merge",
    "user_logs_merge",
    "members_merge",
]

# transactions_seq 포함 (시퀀스 관련 필터링)
target_tables=[
    "train_merge",
    "transactions_merge",
    "user_logs_merge",
    "members_merge",
    "transactions_seq",
]
```

---

*문서 최종 수정: 2026-02-04*
