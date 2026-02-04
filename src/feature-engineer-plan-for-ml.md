# Feature Engineering Plan for ML

이 문서는 KKBox 이탈 예측을 위한 Feature Engineering 계획을 정리합니다.

> **Note**: 청취시간 관련 피처는 모두 `total_hours` (시간 단위, 0~24 클리핑됨)를 사용합니다.
> 이는 스케일 정규화를 위한 것으로, `total_secs`(초 단위, 0~86400) 대신 사용합니다.

---

## 1. 핵심 개념 정의

### 1.1 직전 멤버십 갱신 기간 (Last Membership Period)

**정의**: 유저의 마지막 트랜잭션에서 `transaction_date` ~ `membership_expire_date` 사이의 기간

```
예시 (user_id=11):
transaction_date: 2017-03-15
membership_expire_date: 2017-04-15
→ 직전 멤버십 갱신 기간 = 31일
```

**계산 기준**:
- `transactions_seq` 테이블에서 유저별 `MAX(sequence_group_id)` → `MAX(sequence_id)`인 행
- 해당 행의 `transaction_date`와 `membership_expire_date` 사용

### 1.2 이전 멤버십 유지 기간 (Previous Membership Duration)

**정의**: 현재 시퀀스 그룹 내에서 마지막 트랜잭션 이전까지의 총 기간

```
예시:
시퀀스: [txn_0] → [txn_1] → [txn_2] → [txn_3(마지막)]
이전 멤버십 유지 기간 = txn_0.transaction_date ~ txn_2.membership_expire_date
```

### 1.3 기준점

- **예측 기준일**: 2017-03-31 (user_logs, transactions 데이터의 마지막 날)
- **피처 계산 기준**: 유저별 마지막 트랜잭션의 `transaction_date` ~ `membership_expire_date`

---

## 2. 사전 계산 테이블

Feature 계산 효율화를 위해 다음 중간 테이블들을 먼저 생성합니다.

### 2.1 user_last_txn (유저별 마지막 트랜잭션)

```sql
CREATE TABLE user_last_txn AS
WITH ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY user_id
            ORDER BY sequence_group_id DESC, sequence_id DESC
        ) AS rn
    FROM transactions_seq
)
SELECT
    user_id,
    transaction_date AS last_txn_date,
    membership_expire_date AS last_expire_date,
    payment_plan_days AS plan_days,
    is_cancel AS last_is_cancel,
    payment_method_id,
    sequence_group_id AS last_seq_group,
    sequence_id AS last_seq_id,
    -- 직전 멤버십 갱신 기간
    membership_expire_date - transaction_date AS membership_period,
    -- user_logs 조회 범위
    transaction_date AS log_start_date,
    membership_expire_date AS log_end_date
FROM ranked
WHERE rn = 1;
```

**컬럼 설명**:
| 컬럼 | 설명 |
|------|------|
| last_txn_date | 마지막 트랜잭션 날짜 |
| last_expire_date | 마지막 멤버십 만료일 |
| plan_days | 마지막 결제 플랜 기간 |
| last_is_cancel | 마지막 트랜잭션 취소 여부 |
| membership_period | 직전 멤버십 갱신 기간 (일) |
| log_start_date | user_logs 조회 시작일 |
| log_end_date | user_logs 조회 종료일 |

### 2.2 user_membership_history (멤버십 이력 요약)

```sql
CREATE TABLE user_membership_history AS
WITH last_group AS (
    SELECT user_id, MAX(sequence_group_id) AS last_seq_group
    FROM transactions_seq
    GROUP BY user_id
),
seq_stats AS (
    SELECT
        t.user_id,
        t.sequence_group_id,
        -- 시퀀스 내 트랜잭션 수
        COUNT(*) AS txn_count,
        -- 시퀀스 내 취소 존재 여부
        MAX(CASE WHEN t.is_cancel = 1 THEN 1 ELSE 0 END) AS has_cancel,
        -- 시퀀스 시작~종료 기간
        MIN(t.transaction_date) AS seq_start_date,
        MAX(t.membership_expire_date) AS seq_end_date
    FROM transactions_seq t
    GROUP BY t.user_id, t.sequence_group_id
)
SELECT
    s.user_id,
    lg.last_seq_group,

    -- 현재(마지막) 시퀀스의 정보
    curr.txn_count AS tx_seq_length,
    curr.has_cancel AS cancel_exist,
    curr.seq_end_date - curr.seq_start_date AS membership_duration,

    -- 이전 churn 여부 (sequence_group_id > 0이면 이전에 churn한 적 있음)
    CASE WHEN lg.last_seq_group > 0 THEN 1 ELSE 0 END AS had_churn

FROM last_group lg
JOIN seq_stats curr
    ON lg.user_id = curr.user_id
    AND lg.last_seq_group = curr.sequence_group_id
JOIN (SELECT DISTINCT user_id FROM transactions_seq) s
    ON lg.user_id = s.user_id;
```

### 2.3 user_logs_daily (일별 집계 - 멤버십 기간 필터링)

```sql
-- 멤버십 기간 내 user_logs만 필터링하여 저장
-- NOTE: 청취시간은 total_hours (시간 단위, 0~24 클리핑됨) 사용
CREATE TABLE user_logs_filtered AS
SELECT
    ul.user_id,
    ul.date,
    ul.num_25,
    ul.num_50,
    ul.num_75,
    ul.num_985,
    ul.num_100,
    ul.num_unq,
    ul.total_hours,  -- 시간 단위 (정규화됨)
    -- 멤버십 만료일 기준 역산 일수
    ult.last_expire_date - ul.date AS days_before_expire
FROM user_logs_merge ul
JOIN user_last_txn ult ON ul.user_id = ult.user_id
WHERE ul.date >= ult.log_start_date
  AND ul.date < ult.log_end_date;  -- 만료일 당일은 제외
```

---

## 3. Feature 계산 상세 계획

### 3.1 user_logs 기반 Features

#### 3.1.1 기본 평균 Features (7개)

| Feature | 설명 | 계산식 |
|---------|------|--------|
| num_25_avg | 25% 미만 재생 평균 | SUM(num_25) / membership_period |
| num_50_avg | 25~50% 재생 평균 | SUM(num_50) / membership_period |
| num_75_avg | 50~75% 재생 평균 | SUM(num_75) / membership_period |
| num_985_avg | 75~98.5% 재생 평균 | SUM(num_985) / membership_period |
| num_100_avg | 완전 재생 평균 | SUM(num_100) / membership_period |
| num_unq_avg | 고유곡 평균 | SUM(num_unq) / membership_period |
| total_hours_avg | 총 재생시간 평균 (시간) | SUM(total_hours) / membership_period |

```sql
SELECT
    ulf.user_id,
    SUM(ulf.num_25) * 1.0 / ult.membership_period AS num_25_avg,
    SUM(ulf.num_50) * 1.0 / ult.membership_period AS num_50_avg,
    SUM(ulf.num_75) * 1.0 / ult.membership_period AS num_75_avg,
    SUM(ulf.num_985) * 1.0 / ult.membership_period AS num_985_avg,
    SUM(ulf.num_100) * 1.0 / ult.membership_period AS num_100_avg,
    SUM(ulf.num_unq) * 1.0 / ult.membership_period AS num_unq_avg,
    SUM(ulf.total_hours) * 1.0 / ult.membership_period AS total_hours_avg
FROM user_logs_filtered ulf
JOIN user_last_txn ult ON ulf.user_id = ult.user_id
GROUP BY ulf.user_id, ult.membership_period;
```

#### 3.1.2 접속 텀 Features (4개)

| Feature | 설명 | 계산식 |
|---------|------|--------|
| log_term_min | 접속 간격 최소 | MIN(date - LAG(date)) |
| log_term_max | 접속 간격 최대 | MAX(date - LAG(date)) |
| log_term_avg | 접속 간격 평균 | AVG(date - LAG(date)) |
| log_term_median | 접속 간격 중앙값 | MEDIAN(date - LAG(date)) |

```sql
WITH log_gaps AS (
    SELECT
        user_id,
        date,
        date - LAG(date) OVER (PARTITION BY user_id ORDER BY date) AS gap_days
    FROM user_logs_filtered
)
SELECT
    user_id,
    MIN(gap_days) AS log_term_min,
    MAX(gap_days) AS log_term_max,
    AVG(gap_days) AS log_term_avg,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_days) AS log_term_median
FROM log_gaps
WHERE gap_days IS NOT NULL  -- 첫 로그 제외
GROUP BY user_id;
```

#### 3.1.3 접속 비율 Feature (1개)

| Feature | 설명 | 계산식 |
|---------|------|--------|
| log_days_ratio | 활동 일수 비율 | COUNT(DISTINCT date) / membership_period |

```sql
SELECT
    ulf.user_id,
    COUNT(DISTINCT ulf.date) * 1.0 / ult.membership_period AS log_days_ratio
FROM user_logs_filtered ulf
JOIN user_last_txn ult ON ulf.user_id = ult.user_id
GROUP BY ulf.user_id, ult.membership_period;
```

#### 3.1.4 주중/주말 비율 Feature (1개)

| Feature | 설명 | 계산식 |
|---------|------|--------|
| week_day_ratio | 주일 청취 비율 평균 | 주별 (월~금 total_hours / 전체 total_hours)의 평균 |

```sql
WITH weekly_stats AS (
    SELECT
        user_id,
        DATE_TRUNC('week', date) AS week_start,
        SUM(CASE WHEN DAYOFWEEK(date) BETWEEN 2 AND 6 THEN total_hours ELSE 0 END) AS weekday_hours,
        SUM(total_hours) AS total_hours
    FROM user_logs_filtered
    GROUP BY user_id, DATE_TRUNC('week', date)
)
SELECT
    user_id,
    AVG(CASE WHEN total_hours > 0 THEN weekday_hours / total_hours ELSE 0 END) AS week_day_ratio
FROM weekly_stats
GROUP BY user_id;
```

#### 3.1.5 청취 가속도 Feature (1개)

| Feature | 설명 | 계산식 |
|---------|------|--------|
| log_acc | 첫/마지막 1주 비율 | last_week_hours / (first_week_hours + last_week_hours) |

```sql
WITH first_last_week AS (
    SELECT
        ulf.user_id,
        SUM(CASE WHEN ulf.days_before_expire >= ult.membership_period - 7
                 THEN ulf.total_hours ELSE 0 END) AS first_week_hours,
        SUM(CASE WHEN ulf.days_before_expire < 7
                 THEN ulf.total_hours ELSE 0 END) AS last_week_hours
    FROM user_logs_filtered ulf
    JOIN user_last_txn ult ON ulf.user_id = ult.user_id
    GROUP BY ulf.user_id
)
SELECT
    user_id,
    CASE
        WHEN first_week_hours + last_week_hours > 0
        THEN last_week_hours / (first_week_hours + last_week_hours)
        ELSE 0.5  -- 둘 다 0이면 중립값
    END AS log_acc
FROM first_last_week;
```

#### 3.1.6 최대 대비 최근 비율 Feature (1개)

| Feature | 설명 | 계산식 |
|---------|------|--------|
| max_ratio | 최대 주 대비 최근 주 비율 | last_week_hours / max_week_hours |

```sql
WITH weekly_hours AS (
    SELECT
        user_id,
        DATE_TRUNC('week', date) AS week_start,
        SUM(total_hours) AS week_hours
    FROM user_logs_filtered
    GROUP BY user_id, DATE_TRUNC('week', date)
),
user_weekly_stats AS (
    SELECT
        wh.user_id,
        MAX(wh.week_hours) AS max_week_hours,
        -- 마지막 주 (만료일 기준 직전 7일)
        MAX(CASE
            WHEN wh.week_start >= ult.last_expire_date - INTERVAL 7 DAY
            THEN wh.week_hours
            ELSE 0
        END) AS last_week_hours
    FROM weekly_hours wh
    JOIN user_last_txn ult ON wh.user_id = ult.user_id
    GROUP BY wh.user_id
)
SELECT
    user_id,
    CASE
        WHEN max_week_hours > 0 THEN last_week_hours / max_week_hours
        ELSE 0
    END AS max_ratio
FROM user_weekly_stats;
```

#### 3.1.7 주간 청취 표준편차 Feature (1개)

| Feature | 설명 | 계산식 |
|---------|------|--------|
| log_std | 주간 청취량 표준편차 (시간) | STDDEV(weekly_total_hours) |

```sql
WITH weekly_hours AS (
    SELECT
        user_id,
        DATE_TRUNC('week', date) AS week_start,
        SUM(total_hours) AS week_hours
    FROM user_logs_filtered
    GROUP BY user_id, DATE_TRUNC('week', date)
)
SELECT
    user_id,
    COALESCE(STDDEV(week_hours), 0) AS log_std
FROM weekly_hours
GROUP BY user_id;
```

#### 3.1.8 만료 전 주별 청취량 Features (4개)

| Feature | 설명 | 계산식 |
|---------|------|--------|
| week_1 | 만료 직전 1주 청취량 (시간) | days_before_expire 0~6일 합계 |
| week_2 | 만료 2주전 청취량 (시간) | days_before_expire 7~13일 합계 |
| week_3 | 만료 3주전 청취량 (시간) | days_before_expire 14~20일 합계 |
| week_4 | 만료 4주전 청취량 (시간) | days_before_expire 21~27일 합계 |

```sql
SELECT
    user_id,
    SUM(CASE WHEN days_before_expire BETWEEN 0 AND 6 THEN total_hours ELSE 0 END) AS week_1,
    SUM(CASE WHEN days_before_expire BETWEEN 7 AND 13 THEN total_hours ELSE 0 END) AS week_2,
    SUM(CASE WHEN days_before_expire BETWEEN 14 AND 20 THEN total_hours ELSE 0 END) AS week_3,
    SUM(CASE WHEN days_before_expire BETWEEN 21 AND 27 THEN total_hours ELSE 0 END) AS week_4
FROM user_logs_filtered
GROUP BY user_id;
```

#### 3.1.9 청취 비율 Features (2개)

| Feature | 설명 | 계산식 |
|---------|------|--------|
| num_25_ratio | 25% 미만 재생 비율 | SUM(num_25) / SUM(total_songs) |
| num_100_ratio | 완전 재생 비율 | SUM(num_100) / SUM(total_songs) |

```sql
SELECT
    user_id,
    SUM(num_25) * 1.0 / NULLIF(SUM(num_25 + num_50 + num_75 + num_985 + num_100), 0) AS num_25_ratio,
    SUM(num_100) * 1.0 / NULLIF(SUM(num_25 + num_50 + num_75 + num_985 + num_100), 0) AS num_100_ratio
FROM user_logs_filtered
GROUP BY user_id;
```

---

### 3.2 transactions 기반 Features

| Feature | 설명 | 소스 테이블/계산 |
|---------|------|------------------|
| plan_days | 직전 멤버십 갱신 기간 | user_last_txn.plan_days |
| last_is_cancel | 마지막 취소 여부 | user_last_txn.last_is_cancel |
| payment_method_id | 마지막 결제 수단 | user_last_txn.payment_method_id |
| membership_duration | 이전 멤버십 유지 기간 | user_membership_history.membership_duration |
| tx_seq_length | 시퀀스 내 트랜잭션 수 | user_membership_history.tx_seq_length |
| cancel_exist | 시퀀스 내 취소 존재 여부 | user_membership_history.cancel_exist |
| had_churn | 이전 churn 경험 여부 | user_membership_history.had_churn |

```sql
SELECT
    ult.user_id,
    ult.plan_days,
    ult.last_is_cancel,
    ult.payment_method_id,
    umh.membership_duration,
    umh.tx_seq_length,
    umh.cancel_exist,
    umh.had_churn
FROM user_last_txn ult
JOIN user_membership_history umh ON ult.user_id = umh.user_id;
```

---

### 3.3 members 기반 Features

| Feature | 설명 | 계산식 |
|---------|------|--------|
| registration_init | 클리핑된 가입일 (일수) | MAX(registration_init_time, '2015-01-01') - '2015-01-01' |

```sql
SELECT
    user_id,
    GREATEST(registration_init_time, DATE '2015-01-01') - DATE '2015-01-01' AS registration_init
FROM members_merge;
```

**클리핑 이유**: 2015-01-01 이전 가입자는 모두 동일하게 취급 (데이터 시작점 이전이므로)

---

## 4. 최종 Feature 테이블 생성

### 4.1 테이블 구조

```sql
CREATE TABLE ml_features AS
SELECT
    t.user_id,
    t.is_churn,  -- 타겟 레이블

    -- user_logs 기반 (21개) - 청취시간은 total_hours (시간 단위) 사용
    ul.num_25_avg,
    ul.num_50_avg,
    ul.num_75_avg,
    ul.num_985_avg,
    ul.num_100_avg,
    ul.num_unq_avg,
    ul.total_hours_avg,
    ul.log_term_min,
    ul.log_term_max,
    ul.log_term_avg,
    ul.log_term_median,
    ul.log_days_ratio,
    ul.week_day_ratio,
    ul.log_acc,
    ul.max_ratio,
    ul.log_std,
    ul.week_1,
    ul.week_2,
    ul.week_3,
    ul.week_4,
    ul.num_25_ratio,
    ul.num_100_ratio,

    -- transactions 기반 (7개)
    tx.plan_days,
    tx.last_is_cancel,
    tx.payment_method_id,
    tx.membership_duration,
    tx.tx_seq_length,
    tx.cancel_exist,
    tx.had_churn,

    -- members 기반 (1개)
    m.registration_init

FROM train_merge t
LEFT JOIN user_logs_features ul ON t.user_id = ul.user_id
LEFT JOIN transactions_features tx ON t.user_id = tx.user_id
LEFT JOIN members_features m ON t.user_id = m.user_id;
```

### 4.2 Feature 요약

| 카테고리 | Feature 수 | 설명 |
|----------|------------|------|
| user_logs | 21개 | 청취 패턴, 활동량, 추세 |
| transactions | 7개 | 결제 정보, 멤버십 이력 |
| members | 1개 | 가입 정보 |
| **총계** | **29개** | - |

---

## 5. 구현 순서

### Phase 1: 사전 테이블 생성
```
1. user_last_txn 생성
2. user_membership_history 생성
3. user_logs_filtered 생성 (시간 소요 예상: 가장 큼)
```

### Phase 2: Feature 계산
```
4. user_logs_features 생성 (21개 피처)
   4.1 기본 평균 (7개)
   4.2 접속 텀 (4개)
   4.3 접속 비율 (1개)
   4.4 주중/주말 비율 (1개)
   4.5 청취 가속도 (1개)
   4.6 최대 대비 비율 (1개)
   4.7 주간 표준편차 (1개)
   4.8 만료 전 주별 (4개)
   4.9 청취 비율 (2개) - 총 22개인데 위에 21개라고 했는데 재확인 필요

5. transactions_features 생성 (7개 피처)

6. members_features 생성 (1개 피처)
```

### Phase 3: 통합
```
7. ml_features 테이블 생성 (JOIN)
8. NULL 처리 및 검증
9. Parquet 내보내기
```

---

## 6. NULL/NaN 처리 전략

### 6.1 파라미터

`create_user_logs_features()` 함수의 파라미터:

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `fill_nan` | bool | `True` | True면 NULL 유지, False면 default_value로 채움 |
| `default_value` | float | `-1` | fill_nan=False일 때 사용할 기본값 |

### 6.2 동작 방식

| fill_nan | 동작 | 용도 |
|----------|------|------|
| `True` (기본) | NULL 유지 | XGBoost, LightGBM 등 NaN 처리 가능 모델 |
| `False` | default_value로 채움 | NaN 처리 어려운 모델 (기본값 -1로 구분) |

### 6.3 NULL 발생 조건

| Feature | NULL 발생 조건 |
|---------|----------------|
| `num_*_avg`, `total_hours_avg` | membership_period = 0 또는 로그 없음 |
| `log_term_*` | 로그가 1개 이하 |
| `log_days_ratio` | membership_period = 0 |
| `week_day_ratio` | 해당 주 total_hours = 0 |
| `log_acc` | first + last = 0 |
| `max_ratio` | max_week = 0 |
| `num_*_ratio` | 총 재생 수 = 0 |

### 6.4 코드 예시

```python
# fill_nan=True (기본): NULL 유지
create_user_logs_features(
    db_path=DB_PATH,
    fill_nan=True,
)

# fill_nan=False: -1로 채움 (기본)
create_user_logs_features(
    db_path=DB_PATH,
    fill_nan=False,
    default_value=-1,
)

# fill_nan=False: 0으로 채움
create_user_logs_features(
    db_path=DB_PATH,
    fill_nan=False,
    default_value=0,
)
```

### 6.5 히스토그램에서 NaN 비율 확인

`plot_feature_statistics()` 함수는 각 히스토그램에 NaN 비율을 표시:
- 그래프 제목에 `| NaN: X.XX%` 형태로 표시
- 우측 상단에 NaN 개수와 비율 박스 표시

---

## 7. 성능 최적화 고려사항

### 7.1 user_logs_filtered 생성 시
- **원본 크기**: 219,271,395건
- **예상 필터링 후**: ~20-30M건 (멤버십 기간 평균 30일 기준)
- **최적화**:
  - 인덱스 생성 불필요 (DuckDB는 자동 최적화)
  - PRAGMA threads=8 설정

### 7.2 주간 집계 시
- DATE_TRUNC 함수 사용
- GROUP BY 최소화

### 7.3 메모리 관리
```python
# 대용량 쿼리 실행 시
con.execute("PRAGMA memory_limit='8GB';")
con.execute("PRAGMA threads=8;")
```

---

## 8. 검증 체크리스트

- [ ] 모든 user_id (730,694개)가 ml_features에 존재하는지
- [ ] NULL/NaN 비율이 예상 범위 내인지 (히스토그램으로 확인)
- [ ] Feature 값 범위가 합리적인지 (음수 체크, 이상치 체크)
- [ ] 타겟 레이블(is_churn) 분포가 원본과 동일한지
- [ ] Feature 간 상관관계 분석 (다중공선성 체크)
- [ ] fill_nan 설정이 사용할 ML 모델에 적합한지

```sql
-- 검증 쿼리 예시
SELECT
    COUNT(*) AS total_users,
    COUNT(num_25_avg) AS non_null_count,
    AVG(num_25_avg) AS avg_value,
    MIN(num_25_avg) AS min_value,
    MAX(num_25_avg) AS max_value
FROM ml_features;
```

---

## 9. 파일 구조

```
src/
├── feature-engineer.py          # 메인 실행 스크립트
├── feature-engineer-plan-for-ml.md  # 이 문서
└── utils.py                     # 공통 유틸리티

data/
├── data.duckdb                  # 원본 + 피처 테이블
└── parquet/
    └── ml_features.parquet      # ML용 최종 데이터셋
```

---

## 10. 예상 실행 시간

| 단계 | 예상 시간 | 비고 |
|------|-----------|------|
| user_last_txn | ~30초 | 13M rows 처리 |
| user_membership_history | ~30초 | 집계 연산 |
| user_logs_filtered | ~5-10분 | 220M rows 필터링 |
| user_logs_features | ~3-5분 | 복잡한 집계 |
| transactions_features | ~10초 | 이미 계산된 값 활용 |
| members_features | ~5초 | 단순 변환 |
| ml_features (JOIN) | ~30초 | 4개 테이블 JOIN |
| **총계** | **~10-20분** | - |

---

*문서 작성: 2026-02-02*
