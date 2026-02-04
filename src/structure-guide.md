# KKBox 프로젝트 구조 가이드

이 문서는 KKBox 이탈 예측 프로젝트의 데이터 구조, 코드 구조, 전처리 워크플로우를 정리합니다.
Feature Engineering 작업 시 참고용으로 사용합니다.

---

## 1. 프로젝트 개요

### 1.1 목표
KKBox 음악 스트리밍 서비스의 **사용자 이탈(Churn) 예측** 모델 개발

### 1.2 타겟 변수
- **is_churn**: 2017년 3월 기준 구독 갱신 여부 (0: 유지, 1: 이탈)
- 이탈률: **4.46%** (730,694명 중 32,601명)

### 1.3 분석 기간
- 데이터 범위: **2015-01-01 ~ 2017-03-31** (821일)
- 예측 기준점: 2017년 3월

---

## 2. 환경 설정

### 2.1 패키지 관리: uv
```bash
# 의존성 설치
uv sync

# 스크립트 실행
uv run python src/main.py
uv run python src/feature-engineer.py
```

### 2.2 pyproject.toml
```toml
[project]
name = "kkbox"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = [
    "duckdb>=1.4.3",
    "matplotlib>=3.10.8",
    "pandas>=2.3.3",
    "rich>=14.2.0",
    "seaborn>=0.13.2",
    "tqdm>=4.67.1",
]
```

### 2.3 DuckDB 연결
```python
import duckdb

# 읽기/쓰기
con = duckdb.connect("data/data.duckdb")
con.execute("PRAGMA threads=8;")  # 멀티스레드 활성화

# 읽기 전용 (분석용)
con = duckdb.connect("data/data.duckdb", read_only=True)
```

---

## 3. 디렉토리 구조

```
KKBox/
├── pyproject.toml          # 프로젝트 의존성 (uv)
├── uv.lock                  # 의존성 락 파일
├── src/
│   ├── main.py             # 데이터 전처리 파이프라인
│   ├── utils.py            # 범용 유틸리티 함수
│   ├── feature-engineer.py # Feature Engineering (작성 예정)
│   ├── code-guide.md       # 코드 스타일 가이드
│   └── structure-guide.md  # 이 문서
├── data/
│   ├── data.duckdb         # DuckDB 데이터베이스 (~35GB)
│   ├── csv/                # 원본 CSV 파일
│   ├── parquet/            # Parquet 내보내기 결과
│   └── analysis/           # 분석 결과 그래프
```

---

## 4. 데이터베이스 테이블 구조

### 4.1 테이블 목록

| 테이블 | 행 수 | 설명 |
|--------|-------|------|
| **train_merge** | 730,694 | 타겟 레이블 (user_id, is_churn) |
| **members_merge** | 730,694 | 사용자 기본 정보 |
| **transactions_merge** | 13,193,683 | 결제 트랜잭션 기록 |
| **transactions_seq** | 13,193,683 | 트랜잭션 + 시퀀스 정보 |
| **user_logs_merge** | 219,271,395 | 일별 사용 로그 |
| user_id_map | 730,694 | msno → user_id 매핑 |

### 4.2 공통 사항
- **user_id**: 모든 _merge 테이블의 공통 키 (BIGINT, 1~730,694)
- **날짜 컬럼**: DATE 타입으로 변환됨
- **모든 테이블 동일 유저**: 730,694명 (교집합 필터링 완료)

---

## 5. 테이블 상세 스키마

### 5.1 train_merge (타겟 레이블)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| **user_id** | BIGINT | 사용자 ID (PK) |
| **is_churn** | BIGINT | 이탈 여부 (0/1) |

**통계:**
- 총 유저: 730,694명
- 이탈(is_churn=1): 32,601명 (4.46%)
- 유지(is_churn=0): 698,093명 (95.54%)

---

### 5.2 members_merge (사용자 정보)

| 컬럼 | 타입 | 설명 | 값 범위/통계 |
|------|------|------|--------------|
| **user_id** | BIGINT | 사용자 ID (PK) | 1 ~ 730,694 |
| **gender** | INTEGER | 성별 | 0=여성(166,962), 1=남성(149,671), NULL=미상 |
| **city** | BIGINT | 도시 코드 | 1~21 (21개 도시) |
| **bd** | BIGINT | 나이 (birth date 기반) | 평균 13.2세 (이상치 포함) |
| **registered_via** | BIGINT | 가입 경로 | 1~5 (5개 채널) |
| **registration_init_time** | DATE | 가입일 | 2004-03-26 ~ 2017-03-30 |
| **last_expire** | DATE | 마지막 만료일 | - |
| **membership_seq_group_id** | BIGINT | 멤버십 시퀀스 그룹 | 연속 구독 그룹 ID |
| **membership_seq_id** | BIGINT | 멤버십 시퀀스 내 순번 | 그룹 내 몇 번째 구독인지 |
| **previous_membership_duration** | BIGINT | 직전 구독 기간 (일) | NULL=첫 구독 |
| **previous_membership_seq_duration** | BIGINT | 시퀀스 누적 기간 (일) | NULL=첫 구독 |

**유의사항:**
- `bd`(나이): 이상치 다수 존재, 정제 필요
- `gender`: NULL 값 존재 (약 414,061명)

---

### 5.3 transactions_merge (결제 트랜잭션)

| 컬럼 | 타입 | 설명 | 값 범위/통계 |
|------|------|------|--------------|
| **user_id** | BIGINT | 사용자 ID | - |
| **payment_method_id** | BIGINT | 결제 수단 ID | - |
| **payment_plan_days** | BIGINT | 결제 플랜 기간 (일) | 주로 30, 90, 180, 365 |
| **plan_list_price** | BIGINT | 정가 | - |
| **actual_amount_paid** | BIGINT | 실결제액 | 평균 136.03 |
| **is_auto_renew** | BIGINT | 자동 갱신 여부 | 0/1 (94.4%가 자동갱신) |
| **transaction_date** | DATE | 거래일 | 2015-01-01 ~ 2017-03-31 |
| **membership_expire_date** | DATE | 멤버십 만료일 | - |
| **is_cancel** | BIGINT | 취소 여부 | 0/1 (219,404건 취소) |

**통계:**
- 총 트랜잭션: 13,193,683건
- 고유 사용자: 730,694명
- 유저당 평균 트랜잭션: ~18건

---

### 5.4 transactions_seq (트랜잭션 시퀀스)

`transactions_merge`에 시퀀스 정보가 추가된 테이블

| 컬럼 | 타입 | 설명 |
|------|------|------|
| (transactions_merge 컬럼 전체) | - | - |
| **sequence_group_id** | BIGINT | 연속 구독 그룹 ID (0~8) |
| **sequence_id** | BIGINT | 그룹 내 순번 (0~53) |
| **before_transaction_term** | BIGINT | 직전 거래와의 간격 (일) |
| **before_membership_expire_term** | BIGINT | 직전 만료일과의 간격 (일) |
| **is_churn** | BIGINT | 트랜잭션 레벨 이탈 여부 |

**시퀀스 개념:**
```
User A의 트랜잭션 예시:
┌──────────────────────────────────────────────────────────┐
│ Group 0: [seq 0] → [seq 1] → [seq 2]  (연속 구독)        │
│          30일 간격   만료일 기준 0일 차이                 │
├──────────────────────────────────────────────────────────┤
│ (이탈 기간: 90일 gap)                                    │
├──────────────────────────────────────────────────────────┤
│ Group 1: [seq 0] → [seq 1]  (재구독 후 연속)             │
└──────────────────────────────────────────────────────────┘
```

**is_churn 정의 (트랜잭션 레벨):**
- `before_membership_expire_term > 30`: 만료 후 30일 초과 미갱신 시 이탈로 간주
- 총 이탈 트랜잭션: 242,108건

---

### 5.5 user_logs_merge (사용 로그)

| 컬럼 | 타입 | 설명 | 값 범위/통계 |
|------|------|------|--------------|
| **user_id** | BIGINT | 사용자 ID | - |
| **date** | DATE | 로그 날짜 | 2015-01-01 ~ 2017-03-31 |
| **num_25** | BIGINT | 25% 미만 재생 곡 수 | - |
| **num_50** | BIGINT | 25~50% 재생 곡 수 | - |
| **num_75** | BIGINT | 50~75% 재생 곡 수 | - |
| **num_985** | BIGINT | 75~98.5% 재생 곡 수 | - |
| **num_100** | BIGINT | 98.5% 이상 재생 곡 수 | - |
| **num_unq** | BIGINT | 고유 곡 수 | - |
| **total_secs** | DOUBLE | 총 재생 시간 (초) | 원본 값 |
| **total_hours** | DOUBLE | 총 재생 시간 (시간) | 0~24 클리핑됨 |

**통계:**
- 총 로그: 219,271,395건
- 고유 사용자: 730,694명
- 평균 total_hours: 2.24시간/일
- 유저당 평균 로그: ~300일

**전처리 적용:**
- `total_secs`: 이상치 존재 (음수, 86400초 초과)
- `total_hours`: `total_secs / 3600`, [0, 24] 범위로 클리핑

---

## 6. 코드 구조

### 6.1 src/utils.py (범용 유틸리티)

총 **21개 함수**, 4개 카테고리로 구성:

#### Database/Table Operations (10개)
```python
drop_tables(db_path, tables)                    # 테이블 삭제
copy_table(db_path, source, target)             # 단일 테이블 복사
copy_tables(db_path, table_mapping)             # 여러 테이블 복사
rename_column(db_path, table, old, new)         # 컬럼명 변경
add_column(db_path, table, col, type, default)  # 컬럼 추가
show_database_info(db_path)                     # DB 전체 정보 출력
show_table_info(db_path, table)                 # 테이블 상세 정보
get_table_columns(db_path, table)               # 컬럼 목록 반환
table_exists(db_path, table)                    # 테이블 존재 여부
get_row_count(db_path, table)                   # 행 수 반환
```

#### Data I/O (3개)
```python
load_csv_to_duckdb(db_path, csv_dir, files)     # CSV → DuckDB
export_to_parquet(db_path, output_dir, tables)  # DuckDB → Parquet
export_to_csv(db_path, output_dir, tables)      # DuckDB → CSV
```

#### Data Transformation (4개)
```python
nullify_out_of_range(db_path, table, col, min, max)  # 범위 외 값 NULL 변환
apply_conditional_transform(db_path, table, rules)   # 조건부 변환/삭제
add_converted_column(db_path, table, src, tgt, ...)  # 단위 변환 컬럼 추가
fill_null_values(db_path, table, col, method)        # NULL 채우기
```

#### Data Analysis (4개)
```python
analyze_clipping_distribution(db_path, table, col, ...)  # 클리핑 분포 분석
analyze_feature_correlation(db_path, table, ...)         # 상관관계 분석
get_column_stats(db_path, table, col)                    # 컬럼 통계 반환
```

---

### 6.2 src/main.py (KKBox 도메인 함수)

총 **16개 함수**, 전처리 파이프라인 순서대로 구성:

```python
# 1. 테이블 준비
rename_tables_add_v1_suffix()      # _v1 suffix 추가
create_merge_tables()              # v1/v2 병합 → _merge 테이블

# 2. ID 매핑
create_user_id_mapping()           # msno → user_id 매핑 테이블 생성

# 3. 데이터 필터링
filter_common_msno()               # 공통 msno만 남기기 (교집합)
filter_by_reference_table()        # 기준 테이블 기반 필터링
filter_by_churn_transition()       # Churn 전이 기반 필터링 (v1=0 필수)
exclude_duplicate_transaction_users()  # 중복 트랜잭션 유저 제외

# 4. 분석
analyze_churn_transition()         # Churn 전이행렬 분석
analyze_duplicate_transactions()   # 중복 트랜잭션 분석

# 5. 데이터 변환
convert_gender_to_int()            # gender → 정수 (0/1)
convert_date_fields()              # 날짜 컬럼 → DATE 타입
rename_msno_to_user_id()           # msno → user_id 컬럼명 변경

# 6. 시퀀스/멤버십 정보 추가
create_transactions_seq()          # 트랜잭션 시퀀스 테이블 생성
add_membership_seq_info()          # members에 시퀀스 정보 추가
add_membership_duration_info()     # members에 기간 정보 추가
```

---

## 7. 전처리 워크플로우

### 7.1 파이프라인 개요

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: CSV 로드                                                │
│   raw_*.csv → DuckDB 테이블                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: 테이블 정규화                                           │
│   train → train_v1                                              │
│   train_v1 + train_v2 → train_merge                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 3: 데이터 필터링                                           │
│   - 공통 msno 교집합 필터링                                     │
│   - Churn 전이 필터링 (v1=0인 유저만)                           │
│   - 중복 트랜잭션 유저 제외                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 4: ID 매핑 및 변환                                         │
│   msno (해시) → user_id (정수)                                  │
│   gender → 정수, 날짜 → DATE                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 5: 시퀀스 정보 추가                                        │
│   transactions_seq: 구독 시퀀스 정보                            │
│   members_merge: 멤버십 시퀀스/기간 정보                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 6: Feature Engineering (예정)                              │
│   user_logs → 일별/주별/월별 집계                               │
│   transactions → 결제 패턴 피처                                 │
│   members → 유저 프로필 피처                                    │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 필터링 조건 상세

#### Churn 전이 필터링
```
v1(2월 기준) → v2(3월 기준) 전이 중 유효한 케이스만 선택:
- v1=0, v2=0: 유지 → 유지 (유효)
- v1=0, v2=1: 유지 → 이탈 (유효)
- v1=1, v2=0: 이탈 → 유지 (제외) - 재가입 케이스
- v1=1, v2=1: 이탈 → 이탈 (제외) - 이미 이탈한 케이스
```

#### 중복 트랜잭션 유저 제외
```sql
-- 같은 날 is_cancel=0인 트랜잭션이 2개 이상인 유저 제외
SELECT user_id
FROM transactions_merge
WHERE is_cancel = 0
GROUP BY user_id, transaction_date
HAVING COUNT(*) >= 2
```

---

## 8. Feature Engineering 참고사항

### 8.1 유용한 피처 후보

#### user_logs_merge 기반
```python
# 기간별 집계
- 최근 7일/14일/30일/90일 총 재생시간
- 최근 N일 활동 일수 (로그 존재 일수)
- 일평균 재생시간 추이 (감소/증가)
- num_100 비율 (완전 재생 비율)
- 활동 공백 (마지막 로그로부터 경과일)

# 패턴
- 주중/주말 사용 비율
- 시간대별 사용 패턴 (시간 데이터 없으면 불가)
```

#### transactions_merge/transactions_seq 기반
```python
# 결제 정보
- 평균 결제 금액
- 결제 빈도 (월평균 트랜잭션 수)
- 최근 결제일로부터 경과일
- 자동 갱신 비율

# 시퀀스 정보
- 총 sequence_group_id 수 (재가입 횟수)
- 현재 시퀀스 길이 (연속 구독 횟수)
- 평균 before_transaction_term (결제 주기)
- 과거 이탈 경험 (is_churn=1 기록 유무)
```

#### members_merge 기반
```python
# 기본 정보
- 가입 기간 (registration_init_time ~ 기준일)
- 성별, 도시, 가입 경로

# 멤버십 정보
- previous_membership_duration (직전 구독 기간)
- previous_membership_seq_duration (누적 구독 기간)
- membership_seq_id (현재 몇 번째 연속 구독인지)
```

### 8.2 DuckDB 집계 쿼리 예시

```python
# 최근 30일 사용 통계
con.execute("""
    SELECT
        user_id,
        COUNT(*) AS active_days_30d,
        SUM(total_hours) AS total_hours_30d,
        AVG(total_hours) AS avg_hours_30d,
        MAX(date) AS last_active_date
    FROM user_logs_merge
    WHERE date >= DATE '2017-03-01' - INTERVAL 30 DAY
    GROUP BY user_id
""")

# 트랜잭션 집계
con.execute("""
    SELECT
        user_id,
        COUNT(*) AS txn_count,
        AVG(actual_amount_paid) AS avg_paid,
        SUM(CASE WHEN is_cancel = 1 THEN 1 ELSE 0 END) AS cancel_count,
        MAX(transaction_date) AS last_txn_date
    FROM transactions_merge
    GROUP BY user_id
""")
```

### 8.3 주의사항

1. **클래스 불균형**: 이탈률 4.46%로 심각한 불균형
   - Stratified sampling, SMOTE, class weight 조정 필요

2. **시간 누수(Time Leakage) 방지**:
   - 예측 기준일(2017-03-01) 이후 데이터 사용 금지
   - 피처 계산 시 기준일 명시 필요

3. **대용량 데이터 처리**:
   - user_logs_merge: 2.2억 건 → 샘플링 또는 청크 처리
   - DuckDB의 USING SAMPLE 또는 윈도우 함수 활용

---

## 9. 자주 사용하는 쿼리 패턴

### 9.1 테이블 정보 확인
```python
# 테이블 목록
con.execute("SHOW TABLES").fetchall()

# 컬럼 정보
con.execute("DESCRIBE table_name").fetchall()

# 행 수
con.execute("SELECT COUNT(*) FROM table_name").fetchone()[0]
```

### 9.2 통계 확인
```python
# 기본 통계
con.execute("""
    SELECT
        COUNT(*) AS total,
        COUNT(column_name) AS non_null,
        AVG(column_name) AS mean,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY column_name) AS median,
        MIN(column_name) AS min,
        MAX(column_name) AS max
    FROM table_name
""").fetchdf()
```

### 9.3 JOIN 패턴
```python
# 모든 _merge 테이블 JOIN
con.execute("""
    SELECT
        t.user_id,
        t.is_churn,
        m.gender,
        m.city,
        ...
    FROM train_merge t
    LEFT JOIN members_merge m USING (user_id)
    LEFT JOIN (
        SELECT user_id, SUM(total_hours) AS total_hours
        FROM user_logs_merge
        GROUP BY user_id
    ) ul USING (user_id)
""")
```

---

## 10. 참고 파일

| 파일 | 설명 |
|------|------|
| `src/code-guide.md` | 코드 스타일 가이드 |
| `data/analysis/` | 분석 결과 그래프 |
| `data/parquet/` | Parquet 내보내기 결과 |

---

*문서 최종 수정: 2026-02-02*
