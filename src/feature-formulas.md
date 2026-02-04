# Feature 계산식 정리

`feature-engineer-for-ml.py`에서 계산하는 모든 피처의 수식 정리

---

## 기본 정의

```
membership_period = membership_expire_date - transaction_date  (마지막 트랜잭션 기준)
days_before_expire = membership_expire_date - log_date
gap_days = date - LAG(date)  (연속 로그 날짜 간 간격)
```

---

## 1. User Logs Features (21개)

### 1.1 기본 평균 (7개)

| Feature | Formula |
|---------|---------|
| `num_25_avg` | ∑(num_25) / membership_period |
| `num_50_avg` | ∑(num_50) / membership_period |
| `num_75_avg` | ∑(num_75) / membership_period |
| `num_985_avg` | ∑(num_985) / membership_period |
| `num_100_avg` | ∑(num_100) / membership_period |
| `num_unq_avg` | ∑(num_unq) / membership_period |
| `total_hours_avg` | ∑(total_hours) / membership_period |

### 1.2 접속 텀 (4개)

| Feature | Formula |
|---------|---------|
| `log_term_min` | min(gap_days) |
| `log_term_max` | max(gap_days) |
| `log_term_avg` | mean(gap_days) |
| `log_term_median` | median(gap_days) |

### 1.3 접속 비율 (1개)

| Feature | Formula |
|---------|---------|
| `log_days_ratio` | COUNT(DISTINCT date) / membership_period |

### 1.4 주중/주말 비율 (1개)

| Feature | Formula |
|---------|---------|
| `week_day_ratio` | mean(weekday_hours_w / total_hours_w) |

- `weekday_hours_w` = 주 w의 월~금 청취시간
- `total_hours_w` = 주 w의 전체 청취시간

### 1.5 청취 가속도 (1개)

| Feature | Formula |
|---------|---------|
| `log_acc` | last_week_hours / (first_week_hours + last_week_hours) |

- `first_week_hours` = 멤버십 시작 첫 7일 청취시간 (days_before_expire >= membership_period - 7)
- `last_week_hours` = 만료 전 마지막 7일 청취시간 (days_before_expire < 7)
- **분모 = 0**: fill_nan=True → NULL, fill_nan=False → default_value

### 1.6 최대 대비 비율 (1개)

| Feature | Formula |
|---------|---------|
| `max_ratio` | last_week_hours / max(week_hours_w) |

- `max(week_hours_w)` = 전체 기간 중 주별 최대 청취시간
- `last_week_hours` = 만료 전 마지막 7일 청취시간

### 1.7 주간 표준편차 (1개)

| Feature | Formula |
|---------|---------|
| `log_std` | stddev(week_hours_w) |

- `week_hours_w` = 각 주별 총 청취시간

### 1.8 만료 전 주별 청취량 (4개)

| Feature | Formula |
|---------|---------|
| `week_1` | ∑(total_hours) where days_before_expire ∈ [0, 6] |
| `week_2` | ∑(total_hours) where days_before_expire ∈ [7, 13] |
| `week_3` | ∑(total_hours) where days_before_expire ∈ [14, 20] |
| `week_4` | ∑(total_hours) where days_before_expire ∈ [21, 27] |

### 1.9 청취 비율 (2개)

| Feature | Formula |
|---------|---------|
| `num_25_ratio` | ∑(num_25) / ∑(num_25 + num_50 + num_75 + num_985 + num_100) |
| `num_100_ratio` | ∑(num_100) / ∑(num_25 + num_50 + num_75 + num_985 + num_100) |

---

## 2. Transactions Features (7개)

| Feature | Formula |
|---------|---------|
| `plan_days` | 마지막 트랜잭션의 payment_plan_days |
| `last_is_cancel` | 마지막 트랜잭션의 is_cancel (0 또는 1) |
| `payment_method_id` | 마지막 트랜잭션의 결제 수단 ID |
| `membership_duration` | seq_end_date - seq_start_date (현재 시퀀스) |
| `tx_seq_length` | COUNT(*) per current sequence_group |
| `cancel_exist` | max(is_cancel) per current sequence_group |
| `had_churn` | 1 if sequence_group_id > 0 else 0 |

### 상세 설명

- **sequence_group**: 연속된 멤버십 기간을 하나의 그룹으로 묶은 것
- **had_churn**: sequence_group_id > 0이면 이전에 이탈 후 재가입한 경험이 있음

---

## 3. Members Features (1개)

| Feature | Formula |
|---------|---------|
| `registration_init` | max(registration_init_time, '2015-01-01') - '2015-01-01' |

- 가입일을 2015-01-01 기준으로 클리핑 후 일수로 변환
- 2015-01-01 이전 가입자는 모두 0으로 처리

---

## NULL/NaN 처리

### 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `fill_nan` | bool | `True` | True면 NULL 유지, False면 default_value로 채움 |
| `default_value` | float | `-1` | fill_nan=False일 때 사용할 기본값 |

### 동작 방식

| fill_nan | 동작 |
|----------|------|
| `True` (기본) | 분모가 0 또는 데이터 없음 → **NULL (NaN)** 유지 |
| `False` | 분모가 0 또는 데이터 없음 → **default_value** 로 채움 |

### 사용 예시

```python
# 기본: NULL 유지
create_user_logs_features(fill_nan=True)

# NULL을 -1로 채움 (기본값)
create_user_logs_features(fill_nan=False, default_value=-1)

# NULL을 0으로 채움
create_user_logs_features(fill_nan=False, default_value=0)
```

### NULL이 발생하는 경우

| 피처 | NULL 발생 조건 |
|------|----------------|
| `num_*_avg`, `total_hours_avg` | membership_period = 0 또는 로그 없음 |
| `log_term_*` | 로그가 1개 이하 |
| `log_days_ratio` | membership_period = 0 또는 로그 없음 |
| `week_day_ratio` | 해당 주 total_hours = 0 |
| `log_acc` | first_week + last_week = 0 |
| `max_ratio` | max_week_hours = 0 |
| `num_25_ratio`, `num_100_ratio` | 총 재생 수 = 0 |
| `week_1` ~ `week_4` | 해당 기간 로그 없음 |

---

## 피처 요약

| 카테고리 | 피처 수 | 피처 목록 |
|----------|---------|-----------|
| User Logs - 기본 평균 | 7 | num_25_avg, num_50_avg, num_75_avg, num_985_avg, num_100_avg, num_unq_avg, total_hours_avg |
| User Logs - 접속 텀 | 4 | log_term_min, log_term_max, log_term_avg, log_term_median |
| User Logs - 접속 비율 | 1 | log_days_ratio |
| User Logs - 주중/주말 | 1 | week_day_ratio |
| User Logs - 가속도 | 1 | log_acc |
| User Logs - 최대 대비 | 1 | max_ratio |
| User Logs - 표준편차 | 1 | log_std |
| User Logs - 주별 청취 | 4 | week_1, week_2, week_3, week_4 |
| User Logs - 청취 비율 | 2 | num_25_ratio, num_100_ratio |
| Transactions | 7 | plan_days, last_is_cancel, payment_method_id, membership_duration, tx_seq_length, cancel_exist, had_churn |
| Members | 1 | registration_init |
| **총계** | **30** | (타겟 is_churn 제외) |
