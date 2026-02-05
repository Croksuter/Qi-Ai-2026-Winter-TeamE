# Feature Engineering Plan for ML

이 문서는 KKBox 이탈 예측을 위한 Feature Engineering 계획을 정리합니다.

> **Note**: 모든 시간 피처는 `hours` 단위로, 모든 기간 피처는 `months` 단위로 정규화합니다.
> - secs → hours: ÷ 3600
> - days → months: ÷ 30

---

## 1. 핵심 개념 정의

### 1.1 직전 멤버십 갱신 기간 (Last Membership Period)

**정의**: 유저의 2017년 3월 마지막 트랜잭션의 `transaction_date` ~ `membership_expire_date` 기간

```
예시:
transaction_date: 2017-03-15
membership_expire_date (last_expire_date): 2017-04-15
→ 직전 멤버십 갱신 기간 = 31일
```

**계산 기준**:
- `members_merge.last_expire_date`: 2017년 3월 effective 마지막 만료일
- `members_merge.last_seq_id`: 해당 트랜잭션의 sequence_group_id
- `members_merge.p_tx_id`: 해당 트랜잭션의 sequence_id
- 해당 트랜잭션의 `transaction_date`와 `last_expire_date` 차이 = membership_period

### 1.2 이전 멤버십 유지 기간 (Previous Membership Duration)

**정의**: 현재 시퀀스 그룹 내에서 마지막 트랜잭션(p_tx_id) 이전까지의 총 기간

```
예시:
시퀀스: [txn_0] → [txn_1] → [txn_2] → [txn_3(p_tx_id)]
이전 멤버십 유지 기간 = txn_0.transaction_date ~ txn_2.membership_expire_date
```

### 1.3 기준점

- **예측 기준일**: 2017-03-31
- **피처 계산 기준**: `members_merge.last_expire_date` 기반

---

## 2. 사전 계산 테이블

### 2.1 user_last_txn (유저별 마지막 트랜잭션 정보)

`members_merge`에 이미 14.3, 14.4에서 추가된 컬럼들을 활용:

```sql
CREATE TABLE user_last_txn AS
SELECT
    m.user_id,
    m.last_expire_date,
    m.last_seq_id,
    m.p_tx_id,
    m.pp_tx_id,
    m.is_churn,
    t.transaction_date AS last_txn_date,
    t.payment_plan_days,
    t.is_cancel AS last_is_cancel,
    t.payment_method_id,
    -- 직전 멤버십 갱신 기간
    m.last_expire_date - t.transaction_date AS membership_period,
    -- user_logs 조회 범위
    t.transaction_date AS log_start_date,
    m.last_expire_date AS log_end_date
FROM members_merge m
JOIN transactions_seq t
    ON m.user_id = t.user_id
    AND m.last_seq_id = t.sequence_group_id
    AND m.p_tx_id = t.sequence_id
WHERE m.last_expire_date IS NOT NULL;
```

**컬럼 설명**:
| 컬럼 | 설명 |
|------|------|
| last_txn_date | 마지막 트랜잭션 날짜 |
| last_expire_date | 2017년 3월 effective 마지막 만료일 |
| payment_plan_days | 결제 플랜 기간 (30, 90, 180, 365 등) |
| membership_period | 직전 멤버십 갱신 기간 (last_expire_date - transaction_date) |
| log_start_date | user_logs 조회 시작일 |
| log_end_date | user_logs 조회 종료일 |

### 2.2 user_membership_history (멤버십 이력 요약)

```sql
CREATE TABLE user_membership_history AS
WITH seq_stats AS (
    SELECT
        t.user_id,
        t.sequence_group_id,
        COUNT(*) AS txn_count,
        MAX(CASE WHEN t.is_cancel = 1 THEN 1 ELSE 0 END) AS has_cancel,
        MIN(t.transaction_date) AS seq_start_date,
        MAX(t.membership_expire_date) AS seq_end_date
    FROM transactions_seq t
    GROUP BY t.user_id, t.sequence_group_id
)
SELECT
    m.user_id,
    m.last_seq_id,

    -- 현재(마지막) 시퀀스의 정보 (p_tx_id 이전까지)
    -- tx_seq_length: p_tx_id 이전 트랜잭션 수 (p_tx_id 자체는 제외)
    (SELECT COUNT(*)
     FROM transactions_seq t2
     WHERE t2.user_id = m.user_id
       AND t2.sequence_group_id = m.last_seq_id
       AND t2.sequence_id < m.p_tx_id) AS tx_seq_length,

    -- cancel_exist: p_tx_id 이전 트랜잭션들 중 cancel 존재 여부
    (SELECT MAX(CASE WHEN t2.is_cancel = 1 THEN 1 ELSE 0 END)
     FROM transactions_seq t2
     WHERE t2.user_id = m.user_id
       AND t2.sequence_group_id = m.last_seq_id
       AND t2.sequence_id < m.p_tx_id) AS cancel_exist,

    -- membership_duration: 시퀀스 시작 ~ p_tx_id 직전 트랜잭션의 expire까지
    (SELECT MAX(t2.membership_expire_date) - MIN(t2.transaction_date)
     FROM transactions_seq t2
     WHERE t2.user_id = m.user_id
       AND t2.sequence_group_id = m.last_seq_id
       AND t2.sequence_id < m.p_tx_id) AS membership_duration,

    -- had_churn: 이전에 churn한 적 있는지 (sequence_group_id > 0)
    CASE WHEN m.last_seq_id > 0 THEN 1 ELSE 0 END AS had_churn

FROM members_merge m
WHERE m.last_expire_date IS NOT NULL;
```

### 2.3 user_logs_filtered (멤버십 기간 내 로그)

```sql
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
    ul.total_hours,
    -- 멤버십 만료일 기준 역산 일수
    ult.log_end_date - ul.date AS days_before_expire,
    ult.membership_period
FROM user_logs_merge ul
JOIN user_last_txn ult ON ul.user_id = ult.user_id
WHERE ul.date >= ult.log_start_date
  AND ul.date < ult.log_end_date;  -- 만료일 당일은 제외
```

---

## 3. Feature 계산 상세

### 3.1 user_logs 기반 Features (21개)

#### 3.1.1 기본 평균 (7개)

| Feature | 설명 | 계산식 |
|---------|------|--------|
| num_25_avg | 25% 미만 재생 평균 | SUM(num_25) / membership_period |
| num_50_avg | 25~50% 재생 평균 | SUM(num_50) / membership_period |
| num_75_avg | 50~75% 재생 평균 | SUM(num_75) / membership_period |
| num_985_avg | 75~98.5% 재생 평균 | SUM(num_985) / membership_period |
| num_100_avg | 완전 재생 평균 | SUM(num_100) / membership_period |
| num_unq_avg | 고유곡 평균 | SUM(num_unq) / membership_period |
| total_hours_avg | 총 재생시간 평균 (hours) | SUM(total_secs / 3600) / membership_period |

```sql
SELECT
    user_id,
    SUM(num_25) * 1.0 / membership_period AS num_25_avg,
    SUM(num_50) * 1.0 / membership_period AS num_50_avg,
    SUM(num_75) * 1.0 / membership_period AS num_75_avg,
    SUM(num_985) * 1.0 / membership_period AS num_985_avg,
    SUM(num_100) * 1.0 / membership_period AS num_100_avg,
    SUM(num_unq) * 1.0 / membership_period AS num_unq_avg,
    SUM(total_secs) / 3600.0 / membership_period AS total_hours_avg
FROM user_logs_filtered
GROUP BY user_id, membership_period;
```

#### 3.1.2 접속 텀 (4개)

| Feature | 설명 | 계산식 |
|---------|------|--------|
| log_term_min | 접속 간격 최소 (months) | MIN(gap_days) / 30 |
| log_term_max | 접속 간격 최대 (months) | MAX(gap_days) / 30 |
| log_term_avg | 접속 간격 평균 (months) | AVG(gap_days) / 30 |
| log_term_median | 접속 간격 중앙값 (months) | MEDIAN(gap_days) / 30 |

```sql
WITH log_gaps AS (
    SELECT
        user_id,
        date - LAG(date) OVER (PARTITION BY user_id ORDER BY date) AS gap_days
    FROM user_logs_filtered
)
SELECT
    user_id,
    MIN(gap_days) / 30.0 AS log_term_min,
    MAX(gap_days) / 30.0 AS log_term_max,
    AVG(gap_days) / 30.0 AS log_term_avg,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_days) / 30.0 AS log_term_median
FROM log_gaps
WHERE gap_days IS NOT NULL
GROUP BY user_id;
```

#### 3.1.3 접속 비율 (1개)

| Feature | 설명 | 계산식 |
|---------|------|--------|
| log_days_ratio | 활동 일수 비율 | COUNT(DISTINCT date) / membership_period |

```sql
SELECT
    user_id,
    COUNT(DISTINCT date) * 1.0 / membership_period AS log_days_ratio
FROM user_logs_filtered
GROUP BY user_id, membership_period;
```

#### 3.1.4 주중/주말 비율 (1개)

| Feature | 설명 | 계산식 |
|---------|------|--------|
| week_day_ratio | 주별 주일 청취 비율의 평균 | AVG(weekday_hours / total_hours) per week |

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
    AVG(CASE WHEN total_hours > 0 THEN weekday_hours / total_hours ELSE NULL END) AS week_day_ratio
FROM weekly_stats
GROUP BY user_id;
```

#### 3.1.5 청취 가속도 (1개)

| Feature | 설명 | 계산식 |
|---------|------|--------|
| log_acc | 첫/마지막 1주 비율 | last_week / (first_week + last_week) |

```sql
WITH first_last_week AS (
    SELECT
        user_id,
        SUM(CASE WHEN days_before_expire >= membership_period - 7
                 THEN total_hours ELSE 0 END) AS first_week_hours,
        SUM(CASE WHEN days_before_expire < 7
                 THEN total_hours ELSE 0 END) AS last_week_hours
    FROM user_logs_filtered
    GROUP BY user_id, membership_period
)
SELECT
    user_id,
    CASE
        WHEN first_week_hours + last_week_hours > 0
        THEN last_week_hours / (first_week_hours + last_week_hours)
        ELSE NULL
    END AS log_acc
FROM first_last_week;
```

#### 3.1.6 최대 대비 비율 (1개)

| Feature | 설명 | 계산식 |
|---------|------|--------|
| max_ratio | 최대 주 대비 최근 주 비율 | last_week / max_week |

```sql
WITH weekly_hours AS (
    SELECT
        user_id,
        DATE_TRUNC('week', date) AS week_start,
        SUM(total_hours) AS week_hours
    FROM user_logs_filtered
    GROUP BY user_id, DATE_TRUNC('week', date)
),
user_stats AS (
    SELECT
        wh.user_id,
        MAX(wh.week_hours) AS max_week_hours
    FROM weekly_hours wh
    GROUP BY wh.user_id
),
last_week AS (
    SELECT
        user_id,
        SUM(CASE WHEN days_before_expire < 7 THEN total_hours ELSE 0 END) AS last_week_hours
    FROM user_logs_filtered
    GROUP BY user_id
)
SELECT
    us.user_id,
    CASE
        WHEN us.max_week_hours > 0 THEN lw.last_week_hours / us.max_week_hours
        ELSE NULL
    END AS max_ratio
FROM user_stats us
JOIN last_week lw ON us.user_id = lw.user_id;
```

#### 3.1.7 주간 표준편차 (1개)

| Feature | 설명 | 계산식 |
|---------|------|--------|
| log_std | 주간 청취량 표준편차 | STDDEV(weekly_total_hours) |

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

#### 3.1.8 만료 전 주별 청취량 (4개)

| Feature | 설명 | 계산식 |
|---------|------|--------|
| week_1 | 만료 직전 1주 청취량 | days_before_expire 0~6일 |
| week_2 | 만료 2주전 청취량 | days_before_expire 7~13일 |
| week_3 | 만료 3주전 청취량 | days_before_expire 14~20일 |
| week_4 | 만료 4주전 청취량 | days_before_expire 21~27일 |

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

#### 3.1.9 청취 비율 (2개)

| Feature | 설명 | 계산식 |
|---------|------|--------|
| num_25_ratio | 스킵 비율 | SUM(num_25) / SUM(total_songs) |
| num_100_ratio | 완청 비율 | SUM(num_100) / SUM(total_songs) |

```sql
SELECT
    user_id,
    SUM(num_25) * 1.0 / NULLIF(SUM(num_25 + num_50 + num_75 + num_985 + num_100), 0) AS num_25_ratio,
    SUM(num_100) * 1.0 / NULLIF(SUM(num_25 + num_50 + num_75 + num_985 + num_100), 0) AS num_100_ratio
FROM user_logs_filtered
GROUP BY user_id;
```

---

### 3.2 transactions 기반 Features (7개)

| Feature | 설명 | 소스 |
|---------|------|------|
| payment_plan_months | 결제 플랜 기간 (months) | 가장 최근 non-cancel tx의 payment_plan_days / 30 |
| last_is_cancel | 취소로 인한 만료 여부 | user_last_txn.last_is_cancel |
| payment_method_id | 결제 수단 | user_last_txn.payment_method_id |
| membership_months | 이전 멤버십 유지 기간 (months) | user_membership_history.membership_duration / 30 |
| tx_seq_length | 이전 트랜잭션 수 | user_membership_history |
| cancel_exist | 이전 기간 내 취소 존재 | user_membership_history |
| had_churn | 이전 churn 경험 | user_membership_history |

```sql
SELECT
    ult.user_id,
    -- p_tx가 cancel이면 pp_tx의 payment_plan_days 사용, months로 정규화
    CASE WHEN ult.last_is_cancel = 1 AND pp.payment_plan_days IS NOT NULL
         THEN pp.payment_plan_days / 30.0
         ELSE ult.payment_plan_days / 30.0
    END AS payment_plan_months,
    ult.last_is_cancel,
    ult.payment_method_id,
    COALESCE(umh.membership_duration, 0) / 30.0 AS membership_months,
    COALESCE(umh.tx_seq_length, 0) AS tx_seq_length,
    COALESCE(umh.cancel_exist, 0) AS cancel_exist,
    umh.had_churn
FROM user_last_txn ult
LEFT JOIN user_membership_history umh ON ult.user_id = umh.user_id
LEFT JOIN transactions_seq pp ON ult.user_id = pp.user_id
    AND ult.pp_tx_id = pp.sequence_id
    AND ult.last_seq_id = pp.sequence_group_id;
```

---

### 3.3 members 기반 Features (2개)

| Feature | 설명 | 계산식 |
|---------|------|--------|
| registration_dur | 가입 기간 (months) | (last_expire_date - registration_init_time) / 30 |
| actual_plan_months | 실제 멤버십 기간 (months) | p_tx가 non-cancel → (expire - txn_date) / 30<br>p_tx가 cancel → (expire - pp_txn_date) / 30 |

```sql
SELECT
    m.user_id,
    -- registration_dur: last_expire_date로부터 가입일까지의 개월 수
    (m.last_expire_date - m.registration_init_time) / 30.0 AS registration_dur,
    -- actual_plan_months: p_tx cancel 여부에 따라 다르게 계산, months로 정규화
    CASE
        WHEN p.is_cancel = 0 THEN (p.membership_expire_date - p.transaction_date) / 30.0
        WHEN p.is_cancel = 1 AND pp.transaction_date IS NOT NULL
             THEN (p.membership_expire_date - pp.transaction_date) / 30.0
        ELSE (p.membership_expire_date - p.transaction_date) / 30.0  -- pp 없으면 p 기준
    END AS actual_plan_months
FROM members_merge m
JOIN transactions_seq p ON m.user_id = p.user_id
    AND m.last_seq_id = p.sequence_group_id
    AND m.p_tx_id = p.sequence_id
LEFT JOIN transactions_seq pp ON m.user_id = pp.user_id
    AND m.last_seq_id = pp.sequence_group_id
    AND m.pp_tx_id = pp.sequence_id;
```

---

## 4. 최종 Feature 테이블

### 4.1 ml_features 테이블 구조

```sql
CREATE TABLE ml_features AS
SELECT
    m.user_id,
    m.is_churn,  -- 타겟 레이블

    -- user_logs 기반 (21개)
    ul.num_25_avg,
    ul.num_50_avg,
    ul.num_75_avg,
    ul.num_985_avg,
    ul.num_100_avg,
    ul.num_unq_avg,
    ul.total_hours_avg,      -- hours 단위
    ul.log_term_min,         -- months 단위
    ul.log_term_max,         -- months 단위
    ul.log_term_avg,         -- months 단위
    ul.log_term_median,      -- months 단위
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
    tx.payment_plan_months,  -- months 단위
    tx.last_is_cancel,
    tx.payment_method_id,
    tx.membership_months,    -- months 단위
    tx.tx_seq_length,
    tx.cancel_exist,
    tx.had_churn,

    -- members 기반 (2개)
    mem.registration_dur,    -- months 단위
    mem.actual_plan_months   -- months 단위

FROM members_merge m
LEFT JOIN user_logs_features ul ON m.user_id = ul.user_id
LEFT JOIN transactions_features tx ON m.user_id = tx.user_id
LEFT JOIN members_features mem ON m.user_id = mem.user_id
WHERE m.last_expire_date IS NOT NULL;
```

### 4.2 Feature 요약

| 카테고리 | Feature 수 | 피처 목록 |
|----------|------------|-----------|
| user_logs | 21 | num_*_avg(7), log_term_*(4), log_days_ratio, week_day_ratio, log_acc, max_ratio, log_std, week_1~4(4), num_*_ratio(2) |
| transactions | 7 | payment_plan_months, last_is_cancel, payment_method_id, membership_months, tx_seq_length, cancel_exist, had_churn |
| members | 2 | registration_dur, actual_plan_months |
| **총계** | **30** | (타겟 is_churn 제외) |

### 4.3 정규화 단위

| 원본 단위 | 정규화 단위 | 변환 | 해당 피처 |
|-----------|-------------|------|-----------|
| secs | hours | ÷ 3600 | total_hours_avg |
| days | months | ÷ 30 | log_term_*, payment_plan_months, membership_months, registration_dur, actual_plan_months |

---

## 5. 구현 순서

### Phase 1: 사전 테이블 생성
```
1. user_last_txn 생성 (members_merge + transactions_seq JOIN)
2. user_membership_history 생성
3. user_logs_filtered 생성
```

### Phase 2: Feature 계산
```
4. user_logs_features 생성 (21개)
5. transactions_features 생성 (7개)
6. members_features 생성 (2개)
```

### Phase 3: 통합
```
7. ml_features 테이블 생성 (JOIN)
8. NULL 처리 및 검증
9. Parquet 내보내기
```

---

## 6. NULL 처리 전략

### 6.1 NULL 발생 조건

| Feature | NULL 발생 조건 |
|---------|----------------|
| num_*_avg, total_hours_avg | membership_period = 0 또는 로그 없음 |
| log_term_* | 로그가 1개 이하 |
| log_days_ratio | membership_period = 0 |
| week_day_ratio | 해당 주 total_hours = 0 |
| log_acc | first + last = 0 |
| max_ratio | max_week = 0 |
| num_*_ratio | 총 재생 수 = 0 |
| membership_months, tx_seq_length | p_tx_id = 0 (첫 트랜잭션) |
| actual_plan_months | p_tx가 cancel이고 pp_tx가 없는 경우 |

### 6.2 기본 전략

- XGBoost, LightGBM: NULL 유지 (자체 처리)
- 기타 모델: -1 또는 0으로 대체

---

## 7. 검증 체크리스트

- [ ] `last_expire_date IS NOT NULL`인 유저만 포함되는지
- [ ] 모든 Feature 값 범위가 합리적인지
- [ ] NULL 비율이 예상 범위 내인지
- [ ] is_churn 분포가 원본과 동일한지
- [ ] payment_plan_months와 actual_plan_months 차이 확인
- [ ] p_tx가 cancel인 경우 actual_plan_months 계산 검증
- [ ] registration_dur 값 범위 확인 (음수 없어야 함)
- [ ] 정규화 검증: months 피처는 일반적으로 0~12 범위, hours 피처는 0~24 범위

---

*문서 수정: 2026-02-04*
