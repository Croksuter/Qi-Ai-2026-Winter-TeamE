"""
KKBox ML Feature Engineering 파이프라인

이 스크립트는 전처리된 데이터로부터 ML 학습용 피처를 생성합니다.

Phase 1: 사전 테이블 생성
    1. user_last_txn - 유저별 마지막 트랜잭션 정보
    2. user_membership_history - 멤버십 이력 요약
    3. user_logs_filtered - 멤버십 기간 내 로그

Phase 2: Feature 계산
    4. user_logs_features (21개)
    5. transactions_features (7개)
    6. members_features (2개)

Phase 3: 통합 및 내보내기
    7. ml_features 테이블 생성
    8. Parquet 내보내기

정규화 단위:
    - secs -> hours: / 3600
    - days -> months: / 30
"""

import os
import logging
from typing import Optional

import duckdb
import matplotlib.pyplot as plt

from utils import (
    show_database_info,
    show_table_info,
    export_to_parquet,
)


# ============================================================================
# 헬퍼 함수
# ============================================================================
def _table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    """기존 연결을 사용하여 테이블 존재 여부를 확인합니다."""
    tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    return table_name in tables


def _get_row_count(con: duckdb.DuckDBPyConnection, table_name: str) -> int:
    """기존 연결을 사용하여 테이블 행 수를 반환합니다."""
    return con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]

# ============================================================================
# 로깅 설정
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Phase 1: 사전 테이블 생성
# ============================================================================

# ============================================================================
# 1.1 user_last_txn 생성
# ============================================================================
def create_user_last_txn(
    db_path: str,
    members_table: str = "members_merge",
    transactions_table: str = "transactions_seq",
    target_table: str = "user_last_txn",
    dry_run: bool = False,
) -> None:
    """
    유저별 마지막 트랜잭션 정보 테이블을 생성합니다.

    members_merge의 last_expire_date, last_seq_id, p_tx_id를 기반으로
    transactions_seq와 조인하여 마지막 트랜잭션 상세 정보를 추출합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        members_table: 멤버 테이블 이름
        transactions_table: 트랜잭션 테이블 이름
        target_table: 생성할 테이블 이름
        dry_run: True면 실제 변경 없이 로그만 출력
    """
    logger.info(f"=== {target_table} 생성 시작 ===")
    if dry_run:
        logger.info("*** DRY RUN 모드 - 실제 변경 없음 ***")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    # 테이블 존재 확인
    for tbl in [members_table, transactions_table]:
        if not _table_exists(con, tbl):
            logger.error(f"테이블 {tbl}이 존재하지 않습니다.")
            con.close()
            return

    query = f"""
    CREATE OR REPLACE TABLE {target_table} AS
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
        -- 직전 멤버십 갱신 기간 (days)
        m.last_expire_date - t.transaction_date AS membership_period,
        -- user_logs 조회 범위
        t.transaction_date AS log_start_date,
        m.last_expire_date AS log_end_date
    FROM {members_table} m
    JOIN {transactions_table} t
        ON m.user_id = t.user_id
        AND m.last_seq_id = t.sequence_group_id
        AND m.p_tx_id = t.sequence_id
    WHERE m.last_expire_date IS NOT NULL
    """

    if dry_run:
        # 건수만 확인
        count_query = f"""
        SELECT COUNT(*) FROM {members_table} m
        JOIN {transactions_table} t
            ON m.user_id = t.user_id
            AND m.last_seq_id = t.sequence_group_id
            AND m.p_tx_id = t.sequence_id
        WHERE m.last_expire_date IS NOT NULL
        """
        count = con.execute(count_query).fetchone()[0]
        logger.info(f"  예상 행 수: {count:,}")
    else:
        con.execute(query)
        row_count = _get_row_count(con, target_table)
        logger.info(f"  생성 완료: {row_count:,} 행")

        # 샘플 출력
        sample = con.execute(f"SELECT * FROM {target_table} LIMIT 3").fetchdf()
        logger.info(f"  샘플:\n{sample.to_string()}")

    con.close()
    logger.info(f"=== {target_table} 생성 완료 ===\n")


# ============================================================================
# 1.2 user_membership_history 생성
# ============================================================================
def create_user_membership_history(
    db_path: str,
    members_table: str = "members_merge",
    transactions_table: str = "transactions_seq",
    target_table: str = "user_membership_history",
    dry_run: bool = False,
) -> None:
    """
    유저별 멤버십 이력 요약 테이블을 생성합니다.

    p_tx_id 이전의 트랜잭션들에 대한 통계를 계산합니다:
    - tx_seq_length: 이전 트랜잭션 수
    - cancel_exist: 이전 기간 내 취소 존재 여부
    - membership_duration: 이전 멤버십 유지 기간 (days)
    - had_churn: 이전 churn 경험 여부

    Args:
        db_path: DuckDB 데이터베이스 경로
        members_table: 멤버 테이블 이름
        transactions_table: 트랜잭션 테이블 이름
        target_table: 생성할 테이블 이름
        dry_run: True면 실제 변경 없이 로그만 출력
    """
    logger.info(f"=== {target_table} 생성 시작 ===")
    if dry_run:
        logger.info("*** DRY RUN 모드 - 실제 변경 없음 ***")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    query = f"""
    CREATE OR REPLACE TABLE {target_table} AS
    SELECT
        m.user_id,
        m.last_seq_id,

        -- tx_seq_length: p_tx_id 이전 트랜잭션 수
        (SELECT COUNT(*)
         FROM {transactions_table} t2
         WHERE t2.user_id = m.user_id
           AND t2.sequence_group_id = m.last_seq_id
           AND t2.sequence_id < m.p_tx_id) AS tx_seq_length,

        -- cancel_exist: p_tx_id 이전 트랜잭션들 중 cancel 존재 여부
        (SELECT COALESCE(MAX(CASE WHEN t2.is_cancel = 1 THEN 1 ELSE 0 END), 0)
         FROM {transactions_table} t2
         WHERE t2.user_id = m.user_id
           AND t2.sequence_group_id = m.last_seq_id
           AND t2.sequence_id < m.p_tx_id) AS cancel_exist,

        -- membership_duration: 시퀀스 시작 ~ p_tx_id 직전 트랜잭션의 expire까지 (days)
        (SELECT MAX(t2.membership_expire_date) - MIN(t2.transaction_date)
         FROM {transactions_table} t2
         WHERE t2.user_id = m.user_id
           AND t2.sequence_group_id = m.last_seq_id
           AND t2.sequence_id < m.p_tx_id) AS membership_duration,

        -- had_churn: 이전에 churn한 적 있는지
        CASE WHEN m.last_seq_id > 0 THEN 1 ELSE 0 END AS had_churn

    FROM {members_table} m
    WHERE m.last_expire_date IS NOT NULL
    """

    if dry_run:
        count = con.execute(f"""
            SELECT COUNT(*) FROM {members_table}
            WHERE last_expire_date IS NOT NULL
        """).fetchone()[0]
        logger.info(f"  예상 행 수: {count:,}")
    else:
        con.execute(query)
        row_count = _get_row_count(con, target_table)
        logger.info(f"  생성 완료: {row_count:,} 행")

        # 샘플 출력
        sample = con.execute(f"SELECT * FROM {target_table} LIMIT 3").fetchdf()
        logger.info(f"  샘플:\n{sample.to_string()}")

    con.close()
    logger.info(f"=== {target_table} 생성 완료 ===\n")


# ============================================================================
# 1.3 user_logs_filtered 생성
# ============================================================================
def create_user_logs_filtered(
    db_path: str,
    logs_table: str = "user_logs_merge",
    last_txn_table: str = "user_last_txn",
    target_table: str = "user_logs_filtered",
    dry_run: bool = False,
) -> None:
    """
    멤버십 기간 내 로그만 필터링한 테이블을 생성합니다.

    log_start_date <= date < log_end_date 범위의 로그만 포함하고,
    days_before_expire (만료일 기준 역산 일수)를 계산합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        logs_table: 로그 테이블 이름
        last_txn_table: 마지막 트랜잭션 테이블 이름
        target_table: 생성할 테이블 이름
        dry_run: True면 실제 변경 없이 로그만 출력
    """
    logger.info(f"=== {target_table} 생성 시작 ===")
    if dry_run:
        logger.info("*** DRY RUN 모드 - 실제 변경 없음 ***")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    query = f"""
    CREATE OR REPLACE TABLE {target_table} AS
    SELECT
        ul.user_id,
        ul.date,
        ul.num_25,
        ul.num_50,
        ul.num_75,
        ul.num_985,
        ul.num_100,
        ul.num_unq,
        ul.total_secs,
        -- total_hours 계산 (secs -> hours)
        ul.total_secs / 3600.0 AS total_hours,
        -- 멤버십 만료일 기준 역산 일수
        ult.log_end_date - ul.date AS days_before_expire,
        ult.membership_period
    FROM {logs_table} ul
    JOIN {last_txn_table} ult ON ul.user_id = ult.user_id
    WHERE ul.date >= ult.log_start_date
      AND ul.date < ult.log_end_date
    """

    if dry_run:
        count_query = f"""
        SELECT COUNT(*) FROM {logs_table} ul
        JOIN {last_txn_table} ult ON ul.user_id = ult.user_id
        WHERE ul.date >= ult.log_start_date
          AND ul.date < ult.log_end_date
        """
        count = con.execute(count_query).fetchone()[0]
        logger.info(f"  예상 행 수: {count:,}")
    else:
        con.execute(query)
        row_count = _get_row_count(con, target_table)
        logger.info(f"  생성 완료: {row_count:,} 행")

        # 유저 수 확인
        user_count = con.execute(f"SELECT COUNT(DISTINCT user_id) FROM {target_table}").fetchone()[0]
        logger.info(f"  유저 수: {user_count:,}")

    con.close()
    logger.info(f"=== {target_table} 생성 완료 ===\n")


# ============================================================================
# Phase 2: Feature 계산
# ============================================================================

# ============================================================================
# 2.1 user_logs_features 생성 (21개)
# ============================================================================
def create_user_logs_features(
    db_path: str,
    last_txn_table: str = "user_last_txn",
    filtered_logs_table: str = "user_logs_filtered",
    target_table: str = "user_logs_features",
    dry_run: bool = False,
) -> None:
    """
    user_logs 기반 피처 21개를 생성합니다.

    모든 유저(user_last_txn)를 base로 사용하여, 로그가 없는 유저는 0으로 채웁니다.

    log_term 계산 시 양쪽 경계 텀도 포함합니다:
    - log_start_date → 첫 로그 날짜
    - 마지막 로그 날짜 → log_end_date
    - 로그가 없으면 membership_period 전체를 하나의 텀으로 사용

    log_acc 분모에 1e-7을 더해 수치 안정성을 확보합니다.

    피처 목록:
    - 기본 평균 (7개): num_25_avg ~ total_hours_avg
    - 접속 텀 (4개): log_term_min ~ log_term_median (months)
    - 접속 비율 (1개): log_days_ratio
    - 주중/주말 비율 (1개): week_day_ratio
    - 청취 가속도 (1개): log_acc
    - 최대 대비 비율 (1개): max_ratio
    - 주간 표준편차 (1개): log_std
    - 만료 전 주별 청취량 (4개): week_1 ~ week_4
    - 청취 비율 (2개): num_25_ratio, num_100_ratio

    Args:
        db_path: DuckDB 데이터베이스 경로
        last_txn_table: 마지막 트랜잭션 테이블 이름
        filtered_logs_table: 필터링된 로그 테이블 이름
        target_table: 생성할 테이블 이름
        dry_run: True면 실제 변경 없이 로그만 출력
    """
    logger.info(f"=== {target_table} 생성 시작 (21개 피처) ===")
    if dry_run:
        logger.info("*** DRY RUN 모드 - 실제 변경 없음 ***")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    query = f"""
    CREATE OR REPLACE TABLE {target_table} AS
    WITH
    -- 기본 평균 피처
    basic_avg AS (
        SELECT
            user_id,
            membership_period,
            SUM(num_25) * 1.0 / NULLIF(membership_period, 0) AS num_25_avg,
            SUM(num_50) * 1.0 / NULLIF(membership_period, 0) AS num_50_avg,
            SUM(num_75) * 1.0 / NULLIF(membership_period, 0) AS num_75_avg,
            SUM(num_985) * 1.0 / NULLIF(membership_period, 0) AS num_985_avg,
            SUM(num_100) * 1.0 / NULLIF(membership_period, 0) AS num_100_avg,
            SUM(num_unq) * 1.0 / NULLIF(membership_period, 0) AS num_unq_avg,
            SUM(total_secs) / 3600.0 / NULLIF(membership_period, 0) AS total_hours_avg
        FROM {filtered_logs_table}
        GROUP BY user_id, membership_period
    ),

    -- 접속 텀: 양쪽 경계 텀 포함
    -- log_start→첫로그, 로그간gap, 마지막로그→log_end 모두 포함
    -- 로그 없으면 membership_period 전체가 하나의 텀
    user_log_bounds AS (
        SELECT
            user_id,
            MIN(date) AS first_log_date,
            MAX(date) AS last_log_date
        FROM {filtered_logs_table}
        GROUP BY user_id
    ),
    log_gaps_inner AS (
        -- 로그 간 gap (기존)
        SELECT
            user_id,
            date - LAG(date) OVER (PARTITION BY user_id ORDER BY date) AS gap_days
        FROM {filtered_logs_table}
    ),
    log_gaps_all AS (
        -- 내부 gap
        SELECT user_id, gap_days
        FROM log_gaps_inner
        WHERE gap_days IS NOT NULL
        UNION ALL
        -- 경계 gap: log_start_date → 첫 로그
        SELECT
            b.user_id,
            b.first_log_date - ult.log_start_date AS gap_days
        FROM user_log_bounds b
        JOIN {last_txn_table} ult ON b.user_id = ult.user_id
        UNION ALL
        -- 경계 gap: 마지막 로그 → log_end_date
        SELECT
            b.user_id,
            ult.log_end_date - b.last_log_date AS gap_days
        FROM user_log_bounds b
        JOIN {last_txn_table} ult ON b.user_id = ult.user_id
    ),
    log_term AS (
        SELECT
            user_id,
            MIN(gap_days) / 30.0 AS log_term_min,
            MAX(gap_days) / 30.0 AS log_term_max,
            AVG(gap_days) / 30.0 AS log_term_avg,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_days) / 30.0 AS log_term_median
        FROM log_gaps_all
        GROUP BY user_id
    ),

    -- 로그가 없는 유저의 log_term: membership_period 전체
    no_log_term AS (
        SELECT
            ult.user_id,
            ult.membership_period / 30.0 AS log_term_min,
            ult.membership_period / 30.0 AS log_term_max,
            ult.membership_period / 30.0 AS log_term_avg,
            ult.membership_period / 30.0 AS log_term_median
        FROM {last_txn_table} ult
        WHERE ult.user_id NOT IN (SELECT DISTINCT user_id FROM {filtered_logs_table})
    ),

    -- 접속 비율
    log_ratio AS (
        SELECT
            user_id,
            COUNT(DISTINCT date) * 1.0 / NULLIF(membership_period, 0) AS log_days_ratio
        FROM {filtered_logs_table}
        GROUP BY user_id, membership_period
    ),

    -- 주중/주말 비율
    weekly_stats AS (
        SELECT
            user_id,
            DATE_TRUNC('week', date) AS week_start,
            SUM(CASE WHEN DAYOFWEEK(date) BETWEEN 2 AND 6 THEN total_hours ELSE 0 END) AS weekday_hours,
            SUM(total_hours) AS week_total_hours
        FROM {filtered_logs_table}
        GROUP BY user_id, DATE_TRUNC('week', date)
    ),
    week_day AS (
        SELECT
            user_id,
            AVG(CASE WHEN week_total_hours > 0 THEN weekday_hours / week_total_hours ELSE NULL END) AS week_day_ratio
        FROM weekly_stats
        GROUP BY user_id
    ),

    -- 청취 가속도 (분모에 1e-7 추가)
    first_last AS (
        SELECT
            user_id,
            SUM(CASE WHEN days_before_expire >= membership_period - 7
                     THEN total_hours ELSE 0 END) AS first_week_hours,
            SUM(CASE WHEN days_before_expire < 7
                     THEN total_hours ELSE 0 END) AS last_week_hours
        FROM {filtered_logs_table}
        GROUP BY user_id, membership_period
    ),
    log_acceleration AS (
        SELECT
            user_id,
            last_week_hours / (first_week_hours + last_week_hours + 1e-7) AS log_acc
        FROM first_last
    ),

    -- 최대 대비 비율
    weekly_hours AS (
        SELECT
            user_id,
            DATE_TRUNC('week', date) AS week_start,
            SUM(total_hours) AS week_hours
        FROM {filtered_logs_table}
        GROUP BY user_id, DATE_TRUNC('week', date)
    ),
    max_week AS (
        SELECT
            user_id,
            MAX(week_hours) AS max_week_hours
        FROM weekly_hours
        GROUP BY user_id
    ),
    last_week_hours_cte AS (
        SELECT
            user_id,
            SUM(CASE WHEN days_before_expire < 7 THEN total_hours ELSE 0 END) AS last_week_hours
        FROM {filtered_logs_table}
        GROUP BY user_id
    ),
    max_ratio_calc AS (
        SELECT
            mw.user_id,
            CASE
                WHEN mw.max_week_hours > 0 THEN lw.last_week_hours / mw.max_week_hours
                ELSE 0
            END AS max_ratio
        FROM max_week mw
        JOIN last_week_hours_cte lw ON mw.user_id = lw.user_id
    ),

    -- 주간 표준편차
    log_std_calc AS (
        SELECT
            user_id,
            COALESCE(STDDEV_SAMP(week_hours), 0) AS log_std
        FROM weekly_hours
        GROUP BY user_id
    ),

    -- 만료 전 주별 청취량
    weekly_buckets AS (
        SELECT
            user_id,
            SUM(CASE WHEN days_before_expire BETWEEN 0 AND 6 THEN total_hours ELSE 0 END) AS week_1,
            SUM(CASE WHEN days_before_expire BETWEEN 7 AND 13 THEN total_hours ELSE 0 END) AS week_2,
            SUM(CASE WHEN days_before_expire BETWEEN 14 AND 20 THEN total_hours ELSE 0 END) AS week_3,
            SUM(CASE WHEN days_before_expire BETWEEN 21 AND 27 THEN total_hours ELSE 0 END) AS week_4
        FROM {filtered_logs_table}
        GROUP BY user_id
    ),

    -- 청취 비율
    play_ratio AS (
        SELECT
            user_id,
            SUM(num_25) * 1.0 / NULLIF(SUM(num_25 + num_50 + num_75 + num_985 + num_100), 0) AS num_25_ratio,
            SUM(num_100) * 1.0 / NULLIF(SUM(num_25 + num_50 + num_75 + num_985 + num_100), 0) AS num_100_ratio
        FROM {filtered_logs_table}
        GROUP BY user_id
    )

    -- 최종: user_last_txn을 base로 모든 유저 포함, 로그 없으면 0
    SELECT
        ult.user_id,
        -- 기본 평균 (7개) — 로그 없으면 0
        COALESCE(ba.num_25_avg, 0) AS num_25_avg,
        COALESCE(ba.num_50_avg, 0) AS num_50_avg,
        COALESCE(ba.num_75_avg, 0) AS num_75_avg,
        COALESCE(ba.num_985_avg, 0) AS num_985_avg,
        COALESCE(ba.num_100_avg, 0) AS num_100_avg,
        COALESCE(ba.num_unq_avg, 0) AS num_unq_avg,
        COALESCE(ba.total_hours_avg, 0) AS total_hours_avg,
        -- 접속 텀 (4개) — 로그 있으면 경계포함 텀, 없으면 갱신기간 전체
        COALESCE(lt.log_term_min, nlt.log_term_min) AS log_term_min,
        COALESCE(lt.log_term_max, nlt.log_term_max) AS log_term_max,
        COALESCE(lt.log_term_avg, nlt.log_term_avg) AS log_term_avg,
        COALESCE(lt.log_term_median, nlt.log_term_median) AS log_term_median,
        -- 접속 비율 (1개) — 로그 없으면 0
        COALESCE(lr.log_days_ratio, 0) AS log_days_ratio,
        -- 주중/주말 비율 (1개) — 로그 없으면 0
        COALESCE(wd.week_day_ratio, 0) AS week_day_ratio,
        -- 청취 가속도 (1개) — 로그 없으면 0 (분모에 1e-7 있으므로 값은 ~0)
        COALESCE(la.log_acc, 0) AS log_acc,
        -- 최대 대비 비율 (1개) — 로그 없으면 0
        COALESCE(mr.max_ratio, 0) AS max_ratio,
        -- 주간 표준편차 (1개) — 로그 없으면 0
        COALESCE(ls.log_std, 0) AS log_std,
        -- 만료 전 주별 청취량 (4개) — 로그 없으면 0
        COALESCE(wb.week_1, 0) AS week_1,
        COALESCE(wb.week_2, 0) AS week_2,
        COALESCE(wb.week_3, 0) AS week_3,
        COALESCE(wb.week_4, 0) AS week_4,
        -- 청취 비율 (2개) — 로그 없으면 0
        COALESCE(pr.num_25_ratio, 0) AS num_25_ratio,
        COALESCE(pr.num_100_ratio, 0) AS num_100_ratio
    FROM {last_txn_table} ult
    LEFT JOIN basic_avg ba ON ult.user_id = ba.user_id
    LEFT JOIN log_term lt ON ult.user_id = lt.user_id
    LEFT JOIN no_log_term nlt ON ult.user_id = nlt.user_id
    LEFT JOIN log_ratio lr ON ult.user_id = lr.user_id
    LEFT JOIN week_day wd ON ult.user_id = wd.user_id
    LEFT JOIN log_acceleration la ON ult.user_id = la.user_id
    LEFT JOIN max_ratio_calc mr ON ult.user_id = mr.user_id
    LEFT JOIN log_std_calc ls ON ult.user_id = ls.user_id
    LEFT JOIN weekly_buckets wb ON ult.user_id = wb.user_id
    LEFT JOIN play_ratio pr ON ult.user_id = pr.user_id
    """

    if dry_run:
        count = con.execute(f"SELECT COUNT(*) FROM {last_txn_table}").fetchone()[0]
        logger.info(f"  예상 유저 수: {count:,}")
    else:
        con.execute(query)
        row_count = _get_row_count(con, target_table)
        col_count = len(con.execute(f"DESCRIBE {target_table}").fetchall())
        logger.info(f"  생성 완료: {row_count:,} 행, {col_count} 열")

        # 컬럼 목록 출력
        cols = [row[0] for row in con.execute(f"DESCRIBE {target_table}").fetchall()]
        logger.info(f"  컬럼: {cols}")

    con.close()
    logger.info(f"=== {target_table} 생성 완료 ===\n")


# ============================================================================
# 2.2 transactions_features 생성 (7개)
# ============================================================================
def create_transactions_features(
    db_path: str,
    last_txn_table: str = "user_last_txn",
    history_table: str = "user_membership_history",
    transactions_table: str = "transactions_seq",
    target_table: str = "transactions_features",
    dry_run: bool = False,
) -> None:
    """
    transactions 기반 피처 7개를 생성합니다.

    피처 목록:
    - payment_plan_months: 결제 플랜 기간 (months)
    - last_is_cancel: 취소로 인한 만료 여부
    - payment_method_id: 결제 수단
    - membership_months: 이전 멤버십 유지 기간 (months)
    - tx_seq_length: 이전 트랜잭션 수
    - cancel_exist: 이전 기간 내 취소 존재
    - had_churn: 이전 churn 경험

    Args:
        db_path: DuckDB 데이터베이스 경로
        last_txn_table: 마지막 트랜잭션 테이블 이름
        history_table: 멤버십 이력 테이블 이름
        transactions_table: 트랜잭션 테이블 이름
        target_table: 생성할 테이블 이름
        dry_run: True면 실제 변경 없이 로그만 출력
    """
    logger.info(f"=== {target_table} 생성 시작 (7개 피처) ===")
    if dry_run:
        logger.info("*** DRY RUN 모드 - 실제 변경 없음 ***")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    query = f"""
    CREATE OR REPLACE TABLE {target_table} AS
    SELECT
        ult.user_id,
        -- payment_plan_months: p_tx가 cancel이면 pp_tx의 payment_plan_days 사용
        CASE
            WHEN ult.last_is_cancel = 1 AND pp.payment_plan_days IS NOT NULL
            THEN pp.payment_plan_days / 30.0
            ELSE ult.payment_plan_days / 30.0
        END AS payment_plan_months,
        ult.last_is_cancel,
        ult.payment_method_id,
        -- membership_months: days -> months
        COALESCE(umh.membership_duration, 0) / 30.0 AS membership_months,
        COALESCE(umh.tx_seq_length, 0) AS tx_seq_length,
        COALESCE(umh.cancel_exist, 0) AS cancel_exist,
        umh.had_churn
    FROM {last_txn_table} ult
    LEFT JOIN {history_table} umh ON ult.user_id = umh.user_id
    LEFT JOIN {transactions_table} pp
        ON ult.user_id = pp.user_id
        AND ult.last_seq_id = pp.sequence_group_id
        AND ult.pp_tx_id = pp.sequence_id
    """

    if dry_run:
        count = con.execute(f"SELECT COUNT(*) FROM {last_txn_table}").fetchone()[0]
        logger.info(f"  예상 행 수: {count:,}")
    else:
        con.execute(query)
        row_count = _get_row_count(con, target_table)
        col_count = len(con.execute(f"DESCRIBE {target_table}").fetchall())
        logger.info(f"  생성 완료: {row_count:,} 행, {col_count} 열")

        # 컬럼 목록 출력
        cols = [row[0] for row in con.execute(f"DESCRIBE {target_table}").fetchall()]
        logger.info(f"  컬럼: {cols}")

    con.close()
    logger.info(f"=== {target_table} 생성 완료 ===\n")


# ============================================================================
# 2.3 members_features 생성 (2개)
# ============================================================================
def create_members_features(
    db_path: str,
    members_table: str = "members_merge",
    transactions_table: str = "transactions_seq",
    target_table: str = "members_features",
    dry_run: bool = False,
) -> None:
    """
    members 기반 피처 2개를 생성합니다.

    피처 목록:
    - registration_dur: 가입 기간 (months)
    - actual_plan_months: 실제 멤버십 기간 (months)
      - p_tx가 non-cancel: (expire - txn_date) / 30
      - p_tx가 cancel: (expire - pp_txn_date) / 30

    Args:
        db_path: DuckDB 데이터베이스 경로
        members_table: 멤버 테이블 이름
        transactions_table: 트랜잭션 테이블 이름
        target_table: 생성할 테이블 이름
        dry_run: True면 실제 변경 없이 로그만 출력
    """
    logger.info(f"=== {target_table} 생성 시작 (2개 피처) ===")
    if dry_run:
        logger.info("*** DRY RUN 모드 - 실제 변경 없음 ***")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    query = f"""
    CREATE OR REPLACE TABLE {target_table} AS
    SELECT
        m.user_id,
        -- registration_dur: last_expire_date로부터 가입일까지의 개월 수
        (m.last_expire_date - m.registration_init_time) / 30.0 AS registration_dur,
        -- actual_plan_months: p_tx cancel 여부에 따라 다르게 계산
        CASE
            WHEN p.is_cancel = 0
                THEN (p.membership_expire_date - p.transaction_date) / 30.0
            WHEN p.is_cancel = 1 AND pp.transaction_date IS NOT NULL
                THEN (p.membership_expire_date - pp.transaction_date) / 30.0
            ELSE (p.membership_expire_date - p.transaction_date) / 30.0
        END AS actual_plan_months
    FROM {members_table} m
    JOIN {transactions_table} p
        ON m.user_id = p.user_id
        AND m.last_seq_id = p.sequence_group_id
        AND m.p_tx_id = p.sequence_id
    LEFT JOIN {transactions_table} pp
        ON m.user_id = pp.user_id
        AND m.last_seq_id = pp.sequence_group_id
        AND m.pp_tx_id = pp.sequence_id
    WHERE m.last_expire_date IS NOT NULL
    """

    if dry_run:
        count = con.execute(f"""
            SELECT COUNT(*) FROM {members_table}
            WHERE last_expire_date IS NOT NULL
        """).fetchone()[0]
        logger.info(f"  예상 행 수: {count:,}")
    else:
        con.execute(query)
        row_count = _get_row_count(con, target_table)
        col_count = len(con.execute(f"DESCRIBE {target_table}").fetchall())
        logger.info(f"  생성 완료: {row_count:,} 행, {col_count} 열")

        # 컬럼 목록 출력
        cols = [row[0] for row in con.execute(f"DESCRIBE {target_table}").fetchall()]
        logger.info(f"  컬럼: {cols}")

    con.close()
    logger.info(f"=== {target_table} 생성 완료 ===\n")


# ============================================================================
# Phase 3: 통합 및 내보내기
# ============================================================================

# ============================================================================
# 3.1 ml_features 테이블 생성
# ============================================================================
def create_ml_features(
    db_path: str,
    members_table: str = "members_merge",
    logs_features_table: str = "user_logs_features",
    txn_features_table: str = "transactions_features",
    mem_features_table: str = "members_features",
    target_table: str = "ml_features",
    dry_run: bool = False,
) -> None:
    """
    모든 피처를 통합한 ML 학습용 테이블을 생성합니다.

    총 30개 피처:
    - user_logs: 21개
    - transactions: 7개
    - members: 2개
    + 타겟: is_churn

    Args:
        db_path: DuckDB 데이터베이스 경로
        members_table: 멤버 테이블 이름
        logs_features_table: 로그 피처 테이블 이름
        txn_features_table: 트랜잭션 피처 테이블 이름
        mem_features_table: 멤버 피처 테이블 이름
        target_table: 생성할 테이블 이름
        dry_run: True면 실제 변경 없이 로그만 출력
    """
    logger.info(f"=== {target_table} 생성 시작 (30개 피처 + 타겟) ===")
    if dry_run:
        logger.info("*** DRY RUN 모드 - 실제 변경 없음 ***")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    query = f"""
    CREATE OR REPLACE TABLE {target_table} AS
    SELECT
        m.user_id,
        m.is_churn,

        -- user_logs 기반 (21개)
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
        tx.payment_plan_months,
        tx.last_is_cancel,
        tx.payment_method_id,
        tx.membership_months,
        tx.tx_seq_length,
        tx.cancel_exist,
        tx.had_churn,

        -- members 기반 (2개)
        mem.registration_dur,
        mem.actual_plan_months

    FROM {members_table} m
    LEFT JOIN {logs_features_table} ul ON m.user_id = ul.user_id
    LEFT JOIN {txn_features_table} tx ON m.user_id = tx.user_id
    LEFT JOIN {mem_features_table} mem ON m.user_id = mem.user_id
    WHERE m.last_expire_date IS NOT NULL
    """

    if dry_run:
        count = con.execute(f"""
            SELECT COUNT(*) FROM {members_table}
            WHERE last_expire_date IS NOT NULL
        """).fetchone()[0]
        logger.info(f"  예상 행 수: {count:,}")
    else:
        con.execute(query)
        row_count = _get_row_count(con, target_table)
        col_count = len(con.execute(f"DESCRIBE {target_table}").fetchall())
        logger.info(f"  생성 완료: {row_count:,} 행, {col_count} 열")

        # 컬럼 목록 출력
        cols = [row[0] for row in con.execute(f"DESCRIBE {target_table}").fetchall()]
        logger.info(f"  컬럼 ({len(cols)}개): {cols}")

        # is_churn 분포 확인
        churn_dist = con.execute(f"""
            SELECT is_churn, COUNT(*) as cnt
            FROM {target_table}
            GROUP BY is_churn
            ORDER BY is_churn
        """).fetchdf()
        logger.info(f"  is_churn 분포:\n{churn_dist.to_string()}")

        # NULL 비율 확인
        null_check = con.execute(f"""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN num_25_avg IS NULL THEN 1 ELSE 0 END) as num_25_avg_null,
                SUM(CASE WHEN log_term_min IS NULL THEN 1 ELSE 0 END) as log_term_null,
                SUM(CASE WHEN log_acc IS NULL THEN 1 ELSE 0 END) as log_acc_null,
                SUM(CASE WHEN registration_dur IS NULL THEN 1 ELSE 0 END) as reg_dur_null
            FROM {target_table}
        """).fetchdf()
        logger.info(f"  NULL 체크:\n{null_check.to_string()}")

    con.close()
    logger.info(f"=== {target_table} 생성 완료 ===\n")


# ============================================================================
# 3.2 Parquet 내보내기
# ============================================================================
def export_ml_features(
    db_path: str,
    table_name: str = "ml_features",
    output_dir: str = "data/parquet",
    dry_run: bool = False,
) -> None:
    """
    ml_features 테이블을 Parquet 파일로 내보냅니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 내보낼 테이블 이름
        output_dir: Parquet 파일 저장 디렉토리
        dry_run: True면 실제 변경 없이 로그만 출력
    """
    logger.info(f"=== {table_name} Parquet 내보내기 시작 ===")
    if dry_run:
        logger.info("*** DRY RUN 모드 - 실제 변경 없음 ***")
        return

    export_to_parquet(
        db_path=db_path,
        output_dir=output_dir,
        tables=[table_name],
    )

    logger.info(f"=== {table_name} Parquet 내보내기 완료 ===\n")


# ============================================================================
# 유틸리티: 피처 검증
# ============================================================================
def validate_ml_features(
    db_path: str,
    table_name: str = "ml_features",
) -> None:
    """
    ml_features 테이블의 피처를 검증합니다.

    검증 항목:
    - 행 수 확인
    - 컬럼 수 확인 (32개: user_id + is_churn + 30 features)
    - is_churn 분포
    - 각 피처의 NULL 비율
    - 값 범위 확인 (months, hours 정규화 검증)

    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 검증할 테이블 이름
    """
    logger.info(f"=== {table_name} 검증 시작 ===")

    con = duckdb.connect(db_path)

    # 기본 통계
    row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    cols = [row[0] for row in con.execute(f"DESCRIBE {table_name}").fetchall()]
    logger.info(f"  행 수: {row_count:,}")
    logger.info(f"  컬럼 수: {len(cols)} (예상: 32)")

    # is_churn 분포
    churn_dist = con.execute(f"""
        SELECT
            is_churn,
            COUNT(*) as cnt,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as pct
        FROM {table_name}
        GROUP BY is_churn
        ORDER BY is_churn
    """).fetchdf()
    logger.info(f"  is_churn 분포:\n{churn_dist.to_string()}")

    # NULL 비율
    null_query = """
    SELECT
        'num_25_avg' as col, SUM(CASE WHEN num_25_avg IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as null_pct
    FROM {t}
    UNION ALL SELECT 'log_term_min', SUM(CASE WHEN log_term_min IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) FROM {t}
    UNION ALL SELECT 'log_acc', SUM(CASE WHEN log_acc IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) FROM {t}
    UNION ALL SELECT 'max_ratio', SUM(CASE WHEN max_ratio IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) FROM {t}
    UNION ALL SELECT 'week_day_ratio', SUM(CASE WHEN week_day_ratio IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) FROM {t}
    UNION ALL SELECT 'num_25_ratio', SUM(CASE WHEN num_25_ratio IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) FROM {t}
    UNION ALL SELECT 'registration_dur', SUM(CASE WHEN registration_dur IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) FROM {t}
    UNION ALL SELECT 'actual_plan_months', SUM(CASE WHEN actual_plan_months IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) FROM {t}
    """.format(t=table_name)
    null_stats = con.execute(null_query).fetchdf()
    logger.info(f"  NULL 비율 (%):\n{null_stats.to_string()}")

    # 값 범위 확인 (months 피처)
    range_query = f"""
    SELECT
        MIN(log_term_min) as log_term_min_min, MAX(log_term_max) as log_term_max_max,
        MIN(payment_plan_months) as ppm_min, MAX(payment_plan_months) as ppm_max,
        MIN(membership_months) as mm_min, MAX(membership_months) as mm_max,
        MIN(registration_dur) as rd_min, MAX(registration_dur) as rd_max,
        MIN(actual_plan_months) as apm_min, MAX(actual_plan_months) as apm_max,
        MIN(total_hours_avg) as tha_min, MAX(total_hours_avg) as tha_max
    FROM {table_name}
    """
    range_stats = con.execute(range_query).fetchdf()
    logger.info(f"  값 범위:\n{range_stats.T.to_string()}")

    con.close()
    logger.info(f"=== {table_name} 검증 완료 ===\n")


# ============================================================================
# 유틸리티: NULL 분포 및 샘플 출력
# ============================================================================
def analyze_null_distribution(
    db_path: str,
    table_name: str = "ml_features",
    num_samples: int = 5,
) -> None:
    """
    각 컬럼별 NULL 분포를 출력하고 NULL이 있는 샘플을 보여줍니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 분석할 테이블 이름
        num_samples: NULL 샘플 출력 개수 (기본값: 5)
    """
    logger.info(f"=== {table_name} NULL 분포 분석 시작 ===")

    con = duckdb.connect(db_path)

    # 전체 행 수
    total_rows = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    logger.info(f"  전체 행 수: {total_rows:,}")

    # 컬럼 목록 (user_id, is_churn 제외)
    all_cols = [row[0] for row in con.execute(f"DESCRIBE {table_name}").fetchall()]
    feature_cols = [c for c in all_cols if c not in ("user_id", "is_churn")]

    # 각 컬럼별 NULL 수 계산
    null_counts = []
    for col in feature_cols:
        null_count = con.execute(f"""
            SELECT COUNT(*) FROM {table_name} WHERE {col} IS NULL
        """).fetchone()[0]
        null_counts.append({
            "column": col,
            "null_count": null_count,
            "null_pct": round(null_count * 100.0 / total_rows, 2),
        })

    # NULL이 있는 컬럼만 필터링하고 정렬
    null_cols = [x for x in null_counts if x["null_count"] > 0]
    null_cols.sort(key=lambda x: x["null_count"], reverse=True)

    if not null_cols:
        logger.info("  NULL이 있는 컬럼이 없습니다.")
        con.close()
        logger.info(f"=== {table_name} NULL 분포 분석 완료 ===\n")
        return

    # NULL 분포 테이블 출력
    logger.info(f"\n  [NULL 분포 요약] (NULL이 있는 컬럼: {len(null_cols)}개)")
    logger.info(f"  {'컬럼명':<25} {'NULL 수':>12} {'비율 (%)':>10}")
    logger.info(f"  {'-'*25} {'-'*12} {'-'*10}")
    for item in null_cols:
        logger.info(f"  {item['column']:<25} {item['null_count']:>12,} {item['null_pct']:>10.2f}")

    # 각 NULL 패턴별 샘플 출력
    logger.info(f"\n  [NULL 샘플] (각 컬럼별 최대 {num_samples}개)")

    # NULL 패턴을 그룹화 (유사한 NULL 패턴끼리 묶음)
    # 주요 NULL 패턴: user_logs 기반 피처들이 모두 NULL인 경우
    user_logs_cols = [
        "num_25_avg", "num_50_avg", "num_75_avg", "num_985_avg",
        "num_100_avg", "num_unq_avg", "total_hours_avg",
        "log_term_min", "log_term_max", "log_term_avg", "log_term_median",
        "log_days_ratio", "week_day_ratio", "log_acc", "max_ratio", "log_std",
        "week_1", "week_2", "week_3", "week_4",
        "num_25_ratio", "num_100_ratio"
    ]

    # 패턴 1: 모든 user_logs 피처가 NULL (로그 데이터 없음)
    all_logs_null_condition = " AND ".join([f"{c} IS NULL" for c in user_logs_cols])
    pattern1_count = con.execute(f"""
        SELECT COUNT(*) FROM {table_name} WHERE {all_logs_null_condition}
    """).fetchone()[0]

    if pattern1_count > 0:
        logger.info(f"\n  --- 패턴 1: 모든 user_logs 피처 NULL (로그 없음) ---")
        logger.info(f"  해당 유저 수: {pattern1_count:,} ({pattern1_count * 100.0 / total_rows:.2f}%)")
        sample = con.execute(f"""
            SELECT user_id, is_churn, payment_plan_months, membership_months,
                   registration_dur, actual_plan_months
            FROM {table_name}
            WHERE {all_logs_null_condition}
            LIMIT {num_samples}
        """).fetchdf()
        logger.info(f"  샘플:\n{sample.to_string()}")

    # 패턴 2: log_term_* 만 NULL (로그가 1개만 있음)
    log_term_null_but_avg_not = f"""
        log_term_min IS NULL AND num_25_avg IS NOT NULL
    """
    pattern2_count = con.execute(f"""
        SELECT COUNT(*) FROM {table_name} WHERE {log_term_null_but_avg_not}
    """).fetchone()[0]

    if pattern2_count > 0:
        logger.info(f"\n  --- 패턴 2: log_term_* NULL (로그 1개) ---")
        logger.info(f"  해당 유저 수: {pattern2_count:,} ({pattern2_count * 100.0 / total_rows:.2f}%)")
        sample = con.execute(f"""
            SELECT user_id, is_churn, num_25_avg, log_term_min, log_days_ratio,
                   payment_plan_months, registration_dur
            FROM {table_name}
            WHERE {log_term_null_but_avg_not}
            LIMIT {num_samples}
        """).fetchdf()
        logger.info(f"  샘플:\n{sample.to_string()}")

    # 패턴 3: log_acc만 NULL (first_week + last_week = 0)
    log_acc_null_only = f"""
        log_acc IS NULL AND num_25_avg IS NOT NULL AND log_term_min IS NOT NULL
    """
    pattern3_count = con.execute(f"""
        SELECT COUNT(*) FROM {table_name} WHERE {log_acc_null_only}
    """).fetchone()[0]

    if pattern3_count > 0:
        logger.info(f"\n  --- 패턴 3: log_acc NULL (first+last week = 0) ---")
        logger.info(f"  해당 유저 수: {pattern3_count:,} ({pattern3_count * 100.0 / total_rows:.2f}%)")
        sample = con.execute(f"""
            SELECT user_id, is_churn, total_hours_avg, week_1, week_4, log_acc,
                   payment_plan_months
            FROM {table_name}
            WHERE {log_acc_null_only}
            LIMIT {num_samples}
        """).fetchdf()
        logger.info(f"  샘플:\n{sample.to_string()}")

    # 개별 컬럼별 NULL 샘플 (위 패턴에 해당하지 않는 경우)
    for item in null_cols[:5]:  # 상위 5개 컬럼만
        col = item["column"]
        # 이미 위에서 다룬 패턴은 제외
        if col in user_logs_cols:
            continue

        logger.info(f"\n  --- {col} NULL 샘플 ---")
        sample = con.execute(f"""
            SELECT user_id, is_churn, {col},
                   payment_plan_months, registration_dur
            FROM {table_name}
            WHERE {col} IS NULL
            LIMIT {num_samples}
        """).fetchdf()
        logger.info(f"  샘플:\n{sample.to_string()}")

    con.close()
    logger.info(f"\n=== {table_name} NULL 분포 분석 완료 ===\n")


# ============================================================================
# 유틸리티: is_churn 분포 시각화
# ============================================================================
def analyze_churn_distribution(
    db_path: str,
    table_name: str = "ml_features",
    output_dir: str = "data/analysis",
) -> None:
    """
    ml_features 테이블의 is_churn 분포를 로그와 이미지로 출력합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 분석할 테이블 이름
        output_dir: 이미지 저장 디렉토리
    """
    logger.info(f"=== {table_name} is_churn 분포 분석 시작 ===")

    con = duckdb.connect(db_path, read_only=True)

    # 분포 조회
    dist = con.execute(f"""
        SELECT
            is_churn,
            COUNT(*) AS cnt
        FROM {table_name}
        GROUP BY is_churn
        ORDER BY is_churn
    """).fetchdf()

    total = dist["cnt"].sum()
    dist["pct"] = (dist["cnt"] / total * 100).round(2)

    # 로그 출력
    logger.info(f"  Total: {total:,}")
    for _, row in dist.iterrows():
        label = "Retained" if row["is_churn"] == 0 else "Churned"
        logger.info(
            f"  is_churn={int(row['is_churn'])} ({label}): "
            f"{int(row['cnt']):,} ({row['pct']:.2f}%)"
        )

    con.close()

    # 이미지 생성
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ml_features_churn_distribution.png")

    cnt_0 = int(dist.iloc[0]["cnt"])
    cnt_1 = int(dist.iloc[1]["cnt"])
    pct_0 = dist.iloc[0]["pct"]
    pct_1 = dist.iloc[1]["pct"]
    counts = [cnt_0, cnt_1]
    colors = ["#4C72B0", "#DD8452"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 바 차트
    bar_labels = ["Retained (0)", "Churned (1)"]
    bars = axes[0].bar(bar_labels, counts, color=colors, width=0.5)
    axes[0].set_title("is_churn Distribution (Count)", fontsize=13)
    axes[0].set_ylabel("Users")
    for bar, cnt in zip(bars, counts):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{cnt:,}", ha="center", va="bottom", fontsize=10,
        )

    # 파이 차트
    pie_labels = [
        f"Retained (0)\n{cnt_0:,}\n({pct_0}%)",
        f"Churned (1)\n{cnt_1:,}\n({pct_1}%)",
    ]
    axes[1].pie(
        counts, labels=pie_labels, colors=colors,
        autopct="", startangle=90, textprops={"fontsize": 11},
    )
    axes[1].set_title("is_churn Distribution (Ratio)", fontsize=13)

    fig.suptitle(
        f"ml_features is_churn Distribution (N={total:,})",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"  이미지 저장: {output_path}")
    logger.info(f"=== {table_name} is_churn 분포 분석 완료 ===\n")


# ============================================================================
# 메인 실행
# ============================================================================
if __name__ == "__main__":
    # ========================================================================
    # 상수 설정
    # ========================================================================
    DB_PATH = "data/data.duckdb"
    PARQUET_DIR = "data/parquet"
    ANALYSIS_DIR = "data/analysis"
    DRY_RUN = False  # True로 설정하면 실제 변경 없이 로그만 출력

    # ========================================================================
    # Phase 1: 사전 테이블 생성
    # ========================================================================

    # 1.1 user_last_txn 생성
    create_user_last_txn(
        db_path=DB_PATH,
        dry_run=DRY_RUN,
    )

    # 1.2 user_membership_history 생성
    create_user_membership_history(
        db_path=DB_PATH,
        dry_run=DRY_RUN,
    )

    # 1.3 user_logs_filtered 생성
    create_user_logs_filtered(
        db_path=DB_PATH,
        dry_run=DRY_RUN,
    )

    # ========================================================================
    # Phase 2: Feature 계산
    # ========================================================================

    # 2.1 user_logs_features 생성 (21개)
    create_user_logs_features(
        db_path=DB_PATH,
        dry_run=DRY_RUN,
    )

    # 2.2 transactions_features 생성 (7개)
    create_transactions_features(
        db_path=DB_PATH,
        dry_run=DRY_RUN,
    )

    # 2.3 members_features 생성 (2개)
    create_members_features(
        db_path=DB_PATH,
        dry_run=DRY_RUN,
    )

    # ========================================================================
    # Phase 3: 통합 및 내보내기
    # ========================================================================

    # 3.1 ml_features 테이블 생성
    create_ml_features(
        db_path=DB_PATH,
        dry_run=DRY_RUN,
    )

    # 3.2 검증
    if not DRY_RUN:
        validate_ml_features(
            db_path=DB_PATH,
        )

    # 3.3 NULL 분포 분석
    if not DRY_RUN:
        analyze_null_distribution(
            db_path=DB_PATH,
            num_samples=5,
        )

    # 3.4 is_churn 분포 시각화
    if not DRY_RUN:
        analyze_churn_distribution(
            db_path=DB_PATH,
            output_dir=ANALYSIS_DIR,
        )

    # 3.5 Parquet 내보내기
    export_ml_features(
        db_path=DB_PATH,
        output_dir=PARQUET_DIR,
        dry_run=DRY_RUN,
    )

    # ========================================================================
    # 완료
    # ========================================================================
    logger.info("=== Feature Engineering 파이프라인 완료 ===")
