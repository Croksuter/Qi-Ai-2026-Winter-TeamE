"""
KKBox ML용 Feature Engineering 파이프라인

이 스크립트는 전처리된 데이터로부터 ML 학습용 피처를 생성합니다:

Phase 1: 사전 테이블 생성
  1. ml_user_last_txn - 유저별 마지막 트랜잭션 정보
  2. ml_user_membership_history - 멤버십 이력 요약
  3. ml_user_logs_filtered - 멤버십 기간 내 로그 필터링

Phase 2: Feature 계산
  4. ml_user_logs_features - user_logs 기반 피처 (21개)
  5. ml_transactions_features - transactions 기반 피처 (7개)
  6. ml_members_features - members 기반 피처 (1개)

Phase 3: 통합
  7. ml_features - 최종 피처 테이블 (JOIN)
  8. Parquet 내보내기 (ml_features.parquet)

참고: 청취시간은 total_hours (시간 단위, 0~24 클리핑됨) 사용
"""

import os
import logging

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 범용 유틸리티 함수 import
from utils import (
    show_database_info,
    show_table_info,
    drop_tables,
    export_to_parquet,
)


# ============================================================================
# 헬퍼 함수 (연결 재사용)
# ============================================================================
def _table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    """기존 연결을 사용하여 테이블 존재 여부 확인"""
    tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    return table_name in tables


def _get_row_count(con: duckdb.DuckDBPyConnection, table_name: str) -> int:
    """기존 연결을 사용하여 행 수 반환"""
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
# 1.1 user_last_txn (유저별 마지막 트랜잭션)
# ============================================================================
def create_user_last_txn(
    db_path: str,
    source_table: str = "transactions_seq",
    target_table: str = "user_last_txn",
    force_overwrite: bool = True,
) -> None:
    """
    유저별 마지막 트랜잭션 정보 테이블을 생성합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        source_table: 원본 트랜잭션 시퀀스 테이블
        target_table: 생성할 테이블명
        force_overwrite: True면 기존 테이블 덮어씀
    """
    logger.info(f"=== {target_table} 테이블 생성 시작 ===")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    if _table_exists(con, target_table) and not force_overwrite:
        logger.info(f"{target_table} 이미 존재, 건너뜀")
        con.close()
        return

    logger.info(f"원본 테이블: {source_table}")

    con.execute(f"""
        CREATE OR REPLACE TABLE {target_table} AS
        WITH ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id
                    ORDER BY sequence_group_id DESC, sequence_id DESC
                ) AS rn
            FROM {source_table}
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
    """)

    row_count = _get_row_count(con, target_table)
    logger.info(f"생성 완료: {row_count:,} 행")

    # 샘플 출력
    sample = con.execute(f"""
        SELECT * FROM {target_table}
        ORDER BY user_id
        LIMIT 5
    """).fetchdf()
    logger.info(f"샘플 데이터:\n{sample.to_string()}")

    # 통계
    stats = con.execute(f"""
        SELECT
            AVG(membership_period) AS avg_period,
            MIN(membership_period) AS min_period,
            MAX(membership_period) AS max_period,
            AVG(plan_days) AS avg_plan_days
        FROM {target_table}
    """).fetchone()
    logger.info(f"멤버십 기간: 평균={stats[0]:.1f}일, 범위=[{stats[1]}, {stats[2]}]")
    logger.info(f"plan_days 평균: {stats[3]:.1f}일")

    con.close()
    logger.info(f"=== user_last_txn 테이블 생성 완료 ===\n")


# ============================================================================
# 1.2 user_membership_history (멤버십 이력 요약)
# ============================================================================
def create_user_membership_history(
    db_path: str,
    source_table: str = "transactions_seq",
    target_table: str = "user_membership_history",
    force_overwrite: bool = True,
) -> None:
    """
    유저별 멤버십 이력 요약 테이블을 생성합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        source_table: 원본 트랜잭션 시퀀스 테이블
        target_table: 생성할 테이블명
        force_overwrite: True면 기존 테이블 덮어씀
    """
    logger.info(f"=== {target_table} 테이블 생성 시작 ===")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    if _table_exists(con, target_table) and not force_overwrite:
        logger.info(f"{target_table} 이미 존재, 건너뜀")
        con.close()
        return

    logger.info(f"원본 테이블: {source_table}")

    con.execute(f"""
        CREATE OR REPLACE TABLE {target_table} AS
        WITH last_group AS (
            SELECT user_id, MAX(sequence_group_id) AS last_seq_group
            FROM {source_table}
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
            FROM {source_table} t
            GROUP BY t.user_id, t.sequence_group_id
        )
        SELECT
            lg.user_id,
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
            AND lg.last_seq_group = curr.sequence_group_id;
    """)

    row_count = _get_row_count(con, target_table)
    logger.info(f"생성 완료: {row_count:,} 행")

    # 통계
    stats = con.execute(f"""
        SELECT
            AVG(tx_seq_length) AS avg_seq_len,
            SUM(cancel_exist) AS cancel_users,
            SUM(had_churn) AS had_churn_users,
            AVG(membership_duration) AS avg_duration
        FROM {target_table}
    """).fetchone()
    logger.info(f"평균 시퀀스 길이: {stats[0]:.2f}")
    logger.info(f"취소 경험 유저: {stats[1]:,}")
    logger.info(f"이전 churn 경험 유저: {stats[2]:,}")
    logger.info(f"평균 멤버십 기간: {stats[3]:.1f}일")

    con.close()
    logger.info(f"=== {target_table} 테이블 생성 완료 ===\n")


# ============================================================================
# 1.3 user_logs_filtered (멤버십 기간 내 로그)
# ============================================================================
def create_user_logs_filtered(
    db_path: str,
    logs_table: str = "user_logs_merge",
    txn_table: str = "user_last_txn",
    target_table: str = "user_logs_filtered",
    force_overwrite: bool = True,
) -> None:
    """
    멤버십 기간 내 user_logs만 필터링하여 저장합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        logs_table: 원본 user_logs 테이블
        txn_table: user_last_txn 테이블
        target_table: 생성할 테이블명
        force_overwrite: True면 기존 테이블 덮어씀
    """
    logger.info(f"=== {target_table} 테이블 생성 시작 ===")
    logger.info(f"원본: {logs_table} (대용량, 시간 소요 예상)")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='8GB';")

    if _table_exists(con, target_table) and not force_overwrite:
        logger.info(f"{target_table} 이미 존재, 건너뜀")
        con.close()
        return

    # 원본 행 수 확인
    original_count = _get_row_count(con, logs_table)
    logger.info(f"원본 행 수: {original_count:,}")

    logger.info("필터링 중... (대용량 데이터, 수 분 소요)")

    con.execute(f"""
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
            ul.total_hours,
            -- 멤버십 만료일 기준 역산 일수
            ult.last_expire_date - ul.date AS days_before_expire
        FROM {logs_table} ul
        JOIN {txn_table} ult ON ul.user_id = ult.user_id
        WHERE ul.date >= ult.log_start_date
          AND ul.date < ult.log_end_date;
    """)

    filtered_count = _get_row_count(con, target_table)
    logger.info(f"필터링 완료: {filtered_count:,} 행 ({100*filtered_count/original_count:.2f}%)")

    # 유저당 평균 로그 수
    avg_logs = con.execute(f"""
        SELECT AVG(cnt) FROM (
            SELECT user_id, COUNT(*) AS cnt
            FROM {target_table}
            GROUP BY user_id
        )
    """).fetchone()[0]
    logger.info(f"유저당 평균 로그 수: {avg_logs:.1f}")

    con.close()
    logger.info(f"=== {target_table} 테이블 생성 완료 ===\n")


# ============================================================================
# Phase 2: Feature 계산
# ============================================================================

# ============================================================================
# 2.1 user_logs_features (user_logs 기반 피처)
# ============================================================================
def create_user_logs_features(
    db_path: str,
    logs_table: str = "user_logs_filtered",
    txn_table: str = "user_last_txn",
    target_table: str = "user_logs_features",
    force_overwrite: bool = True,
    fill_nan: bool = True,
    default_value: float = -1,
) -> None:
    """
    user_logs 기반 피처를 계산합니다. (21개)

    피처 목록:
    - 기본 평균 (7개): num_25_avg ~ total_hours_avg
    - 접속 텀 (4개): log_term_min/max/avg/median
    - 접속 비율 (1개): log_days_ratio
    - 주중/주말 비율 (1개): week_day_ratio
    - 청취 가속도 (1개): log_acc
    - 최대 대비 비율 (1개): max_ratio
    - 주간 표준편차 (1개): log_std
    - 만료 전 주별 (4개): week_1 ~ week_4
    - 청취 비율 (2개): num_25_ratio, num_100_ratio

    Args:
        db_path: DuckDB 데이터베이스 경로
        logs_table: 필터링된 user_logs 테이블
        txn_table: user_last_txn 테이블
        target_table: 생성할 테이블명
        force_overwrite: True면 기존 테이블 덮어씀
        fill_nan: True면 분모가 0일 때 NULL 유지, False면 default_value로 채움
        default_value: fill_nan=False일 때 사용할 기본값 (기본: -1)
    """
    logger.info(f"=== {target_table} 테이블 생성 시작 ===")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='8GB';")

    if _table_exists(con, target_table) and not force_overwrite:
        logger.info(f"{target_table} 이미 존재, 건너뜀")
        con.close()
        return

    logger.info("피처 계산 중... (복잡한 집계, 수 분 소요)")
    if fill_nan:
        logger.info("NULL 유지 모드: ON (분모 0 -> NULL)")
    else:
        logger.info(f"NULL 유지 모드: OFF (분모 0 -> {default_value})")

    # fill_nan에 따라 COALESCE 적용 여부 결정
    def col(expr: str) -> str:
        return expr if fill_nan else f"COALESCE({expr}, {default_value})"

    con.execute(f"""
        CREATE OR REPLACE TABLE {target_table} AS
        WITH
        -- =====================================================================
        -- 1. 기본 평균 피처 (7개)
        -- =====================================================================
        basic_avg AS (
            SELECT
                ulf.user_id,
                SUM(ulf.num_25) * 1.0 / ult.membership_period AS num_25_avg,
                SUM(ulf.num_50) * 1.0 / ult.membership_period AS num_50_avg,
                SUM(ulf.num_75) * 1.0 / ult.membership_period AS num_75_avg,
                SUM(ulf.num_985) * 1.0 / ult.membership_period AS num_985_avg,
                SUM(ulf.num_100) * 1.0 / ult.membership_period AS num_100_avg,
                SUM(ulf.num_unq) * 1.0 / ult.membership_period AS num_unq_avg,
                SUM(ulf.total_hours) * 1.0 / ult.membership_period AS total_hours_avg
            FROM {logs_table} ulf
            JOIN {txn_table} ult ON ulf.user_id = ult.user_id
            GROUP BY ulf.user_id, ult.membership_period
        ),

        -- =====================================================================
        -- 2. 접속 텀 피처 (4개)
        -- =====================================================================
        log_gaps AS (
            SELECT
                user_id,
                date,
                date - LAG(date) OVER (PARTITION BY user_id ORDER BY date) AS gap_days
            FROM {logs_table}
        ),
        log_term AS (
            SELECT
                user_id,
                MIN(gap_days) AS log_term_min,
                MAX(gap_days) AS log_term_max,
                AVG(gap_days) AS log_term_avg,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY gap_days) AS log_term_median
            FROM log_gaps
            WHERE gap_days IS NOT NULL
            GROUP BY user_id
        ),

        -- =====================================================================
        -- 3. 접속 비율 피처 (1개)
        -- =====================================================================
        log_ratio AS (
            SELECT
                ulf.user_id,
                COUNT(DISTINCT ulf.date) * 1.0 / ult.membership_period AS log_days_ratio
            FROM {logs_table} ulf
            JOIN {txn_table} ult ON ulf.user_id = ult.user_id
            GROUP BY ulf.user_id, ult.membership_period
        ),

        -- =====================================================================
        -- 4. 주중/주말 비율 피처 (1개)
        -- =====================================================================
        weekly_weekday AS (
            SELECT
                user_id,
                DATE_TRUNC('week', date) AS week_start,
                SUM(CASE WHEN DAYOFWEEK(date) BETWEEN 2 AND 6 THEN total_hours ELSE 0 END) AS weekday_hours,
                SUM(total_hours) AS total_hours
            FROM {logs_table}
            GROUP BY user_id, DATE_TRUNC('week', date)
        ),
        week_day AS (
            SELECT
                user_id,
                AVG(CASE WHEN total_hours > 0 THEN weekday_hours / total_hours ELSE 0 END) AS week_day_ratio
            FROM weekly_weekday
            GROUP BY user_id
        ),

        -- =====================================================================
        -- 5. 청취 가속도 피처 (1개)
        -- =====================================================================
        first_last AS (
            SELECT
                ulf.user_id,
                SUM(CASE WHEN ulf.days_before_expire >= ult.membership_period - 7
                         THEN ulf.total_hours ELSE 0 END) AS first_week_hours,
                SUM(CASE WHEN ulf.days_before_expire < 7
                         THEN ulf.total_hours ELSE 0 END) AS last_week_hours
            FROM {logs_table} ulf
            JOIN {txn_table} ult ON ulf.user_id = ult.user_id
            GROUP BY ulf.user_id
        ),
        log_acceleration AS (
            SELECT
                user_id,
                CASE
                    WHEN first_week_hours + last_week_hours > 0
                    THEN last_week_hours / (first_week_hours + last_week_hours)
                    ELSE 0.5
                END AS log_acc
            FROM first_last
        ),

        -- =====================================================================
        -- 6. 최대 대비 비율 피처 (1개)
        -- =====================================================================
        weekly_hours AS (
            SELECT
                user_id,
                DATE_TRUNC('week', date) AS week_start,
                SUM(total_hours) AS week_hours
            FROM {logs_table}
            GROUP BY user_id, DATE_TRUNC('week', date)
        ),
        max_week AS (
            SELECT
                wh.user_id,
                MAX(wh.week_hours) AS max_week_hours
            FROM weekly_hours wh
            GROUP BY wh.user_id
        ),
        last_week_hours AS (
            SELECT
                ulf.user_id,
                SUM(CASE WHEN ulf.days_before_expire < 7 THEN ulf.total_hours ELSE 0 END) AS last_week_hours
            FROM {logs_table} ulf
            GROUP BY ulf.user_id
        ),
        max_ratio_calc AS (
            SELECT
                mw.user_id,
                CASE
                    WHEN mw.max_week_hours > 0 THEN lwh.last_week_hours / mw.max_week_hours
                    ELSE 0
                END AS max_ratio
            FROM max_week mw
            JOIN last_week_hours lwh ON mw.user_id = lwh.user_id
        ),

        -- =====================================================================
        -- 7. 주간 표준편차 피처 (1개)
        -- =====================================================================
        log_stddev AS (
            SELECT
                user_id,
                COALESCE(STDDEV(week_hours), 0) AS log_std
            FROM weekly_hours
            GROUP BY user_id
        ),

        -- =====================================================================
        -- 8. 만료 전 주별 청취량 피처 (4개)
        -- =====================================================================
        weekly_before_expire AS (
            SELECT
                user_id,
                SUM(CASE WHEN days_before_expire BETWEEN 0 AND 6 THEN total_hours ELSE 0 END) AS week_1,
                SUM(CASE WHEN days_before_expire BETWEEN 7 AND 13 THEN total_hours ELSE 0 END) AS week_2,
                SUM(CASE WHEN days_before_expire BETWEEN 14 AND 20 THEN total_hours ELSE 0 END) AS week_3,
                SUM(CASE WHEN days_before_expire BETWEEN 21 AND 27 THEN total_hours ELSE 0 END) AS week_4
            FROM {logs_table}
            GROUP BY user_id
        ),

        -- =====================================================================
        -- 9. 청취 비율 피처 (2개)
        -- =====================================================================
        listen_ratio AS (
            SELECT
                user_id,
                SUM(num_25) * 1.0 / NULLIF(SUM(num_25 + num_50 + num_75 + num_985 + num_100), 0) AS num_25_ratio,
                SUM(num_100) * 1.0 / NULLIF(SUM(num_25 + num_50 + num_75 + num_985 + num_100), 0) AS num_100_ratio
            FROM {logs_table}
            GROUP BY user_id
        ),

        -- =====================================================================
        -- 모든 유저 목록 (로그 없는 유저 포함)
        -- =====================================================================
        all_users AS (
            SELECT DISTINCT user_id FROM {txn_table}
        )

        -- =====================================================================
        -- 최종 JOIN
        -- =====================================================================
        SELECT
            u.user_id,

            -- 기본 평균 (7개)
            {col('ba.num_25_avg')} AS num_25_avg,
            {col('ba.num_50_avg')} AS num_50_avg,
            {col('ba.num_75_avg')} AS num_75_avg,
            {col('ba.num_985_avg')} AS num_985_avg,
            {col('ba.num_100_avg')} AS num_100_avg,
            {col('ba.num_unq_avg')} AS num_unq_avg,
            {col('ba.total_hours_avg')} AS total_hours_avg,

            -- 접속 텀 (4개)
            {col('lt.log_term_min')} AS log_term_min,
            {col('lt.log_term_max')} AS log_term_max,
            {col('lt.log_term_avg')} AS log_term_avg,
            {col('lt.log_term_median')} AS log_term_median,

            -- 접속 비율 (1개)
            {col('lr.log_days_ratio')} AS log_days_ratio,

            -- 주중/주말 비율 (1개)
            {col('wd.week_day_ratio')} AS week_day_ratio,

            -- 청취 가속도 (1개)
            {col('la.log_acc')} AS log_acc,

            -- 최대 대비 비율 (1개)
            {col('mr.max_ratio')} AS max_ratio,

            -- 주간 표준편차 (1개)
            {col('ls.log_std')} AS log_std,

            -- 만료 전 주별 (4개)
            {col('wbe.week_1')} AS week_1,
            {col('wbe.week_2')} AS week_2,
            {col('wbe.week_3')} AS week_3,
            {col('wbe.week_4')} AS week_4,

            -- 청취 비율 (2개)
            {col('lratio.num_25_ratio')} AS num_25_ratio,
            {col('lratio.num_100_ratio')} AS num_100_ratio

        FROM all_users u
        LEFT JOIN basic_avg ba ON u.user_id = ba.user_id
        LEFT JOIN log_term lt ON u.user_id = lt.user_id
        LEFT JOIN log_ratio lr ON u.user_id = lr.user_id
        LEFT JOIN week_day wd ON u.user_id = wd.user_id
        LEFT JOIN log_acceleration la ON u.user_id = la.user_id
        LEFT JOIN max_ratio_calc mr ON u.user_id = mr.user_id
        LEFT JOIN log_stddev ls ON u.user_id = ls.user_id
        LEFT JOIN weekly_before_expire wbe ON u.user_id = wbe.user_id
        LEFT JOIN listen_ratio lratio ON u.user_id = lratio.user_id;
    """)

    row_count = _get_row_count(con, target_table)
    logger.info(f"생성 완료: {row_count:,} 행, 21개 피처")

    # 피처 통계
    stats = con.execute(f"""
        SELECT
            AVG(total_hours_avg) AS avg_hours,
            AVG(log_days_ratio) AS avg_ratio,
            AVG(log_acc) AS avg_acc,
            AVG(num_100_ratio) AS avg_100_ratio
        FROM {target_table}
    """).fetchone()
    logger.info(f"피처 통계:")
    logger.info(f"  total_hours_avg 평균: {stats[0]:.4f}")
    logger.info(f"  log_days_ratio 평균: {stats[1]:.4f}")
    logger.info(f"  log_acc 평균: {stats[2]:.4f}")
    logger.info(f"  num_100_ratio 평균: {stats[3]:.4f}")

    con.close()
    logger.info(f"=== {target_table} 테이블 생성 완료 ===\n")


# ============================================================================
# 2.2 transactions_features (transactions 기반 피처)
# ============================================================================
def create_transactions_features(
    db_path: str,
    txn_table: str = "user_last_txn",
    history_table: str = "user_membership_history",
    target_table: str = "transactions_features",
    force_overwrite: bool = True,
) -> None:
    """
    transactions 기반 피처를 계산합니다. (7개)

    피처 목록:
    - plan_days: 직전 멤버십 갱신 기간
    - last_is_cancel: 마지막 취소 여부
    - payment_method_id: 결제 수단
    - membership_duration: 이전 멤버십 유지 기간
    - tx_seq_length: 시퀀스 내 트랜잭션 수
    - cancel_exist: 시퀀스 내 취소 존재 여부
    - had_churn: 이전 churn 경험 여부

    Args:
        db_path: DuckDB 데이터베이스 경로
        txn_table: user_last_txn 테이블
        history_table: user_membership_history 테이블
        target_table: 생성할 테이블명
        force_overwrite: True면 기존 테이블 덮어씀
    """
    logger.info(f"=== {target_table} 테이블 생성 시작 ===")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    if _table_exists(con, target_table) and not force_overwrite:
        logger.info(f"{target_table} 이미 존재, 건너뜀")
        con.close()
        return

    con.execute(f"""
        CREATE OR REPLACE TABLE {target_table} AS
        SELECT
            ult.user_id,
            ult.plan_days,
            ult.last_is_cancel,
            ult.payment_method_id,
            umh.membership_duration,
            umh.tx_seq_length,
            umh.cancel_exist,
            umh.had_churn
        FROM {txn_table} ult
        JOIN {history_table} umh ON ult.user_id = umh.user_id;
    """)

    row_count = _get_row_count(con, target_table)
    logger.info(f"생성 완료: {row_count:,} 행, 7개 피처")

    # 피처 통계
    stats = con.execute(f"""
        SELECT
            AVG(plan_days) AS avg_plan,
            SUM(last_is_cancel) AS cancel_cnt,
            SUM(had_churn) AS churn_cnt,
            AVG(tx_seq_length) AS avg_seq_len
        FROM {target_table}
    """).fetchone()
    logger.info(f"피처 통계:")
    logger.info(f"  plan_days 평균: {stats[0]:.1f}")
    logger.info(f"  last_is_cancel 수: {stats[1]:,}")
    logger.info(f"  had_churn 수: {stats[2]:,}")
    logger.info(f"  tx_seq_length 평균: {stats[3]:.2f}")

    con.close()
    logger.info(f"=== {target_table} 테이블 생성 완료 ===\n")


# ============================================================================
# 2.3 members_features (members 기반 피처)
# ============================================================================
def create_members_features(
    db_path: str,
    source_table: str = "members_merge",
    target_table: str = "members_features",
    clip_date: str = "2015-01-01",
    force_overwrite: bool = True,
) -> None:
    """
    members 기반 피처를 계산합니다. (1개)

    피처 목록:
    - registration_init: 클리핑된 가입일 (일수)

    Args:
        db_path: DuckDB 데이터베이스 경로
        source_table: 원본 members 테이블
        target_table: 생성할 테이블명
        clip_date: 가입일 클리핑 기준일
        force_overwrite: True면 기존 테이블 덮어씀
    """
    logger.info(f"=== {target_table} 테이블 생성 시작 ===")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    if _table_exists(con, target_table) and not force_overwrite:
        logger.info(f"{target_table} 이미 존재, 건너뜀")
        con.close()
        return

    logger.info(f"가입일 클리핑 기준: {clip_date}")

    con.execute(f"""
        CREATE OR REPLACE TABLE {target_table} AS
        SELECT
            user_id,
            GREATEST(registration_init_time, DATE '{clip_date}') - DATE '{clip_date}' AS registration_init
        FROM {source_table};
    """)

    row_count = _get_row_count(con, target_table)
    logger.info(f"생성 완료: {row_count:,} 행, 1개 피처")

    # 통계
    stats = con.execute(f"""
        SELECT
            AVG(registration_init) AS avg_days,
            MIN(registration_init) AS min_days,
            MAX(registration_init) AS max_days
        FROM {target_table}
    """).fetchone()
    logger.info(f"registration_init: 평균={stats[0]:.1f}일, 범위=[{stats[1]}, {stats[2]}]")

    con.close()
    logger.info(f"=== {target_table} 테이블 생성 완료 ===\n")


# ============================================================================
# Phase 3: 통합
# ============================================================================

# ============================================================================
# 3.1 ml_features (최종 피처 테이블)
# ============================================================================
def create_ml_features(
    db_path: str,
    train_table: str = "train_merge",
    logs_features: str = "user_logs_features",
    txn_features: str = "transactions_features",
    members_features: str = "members_features",
    target_table: str = "ml_features",
    force_overwrite: bool = True,
) -> None:
    """
    모든 피처를 JOIN하여 최종 ML 피처 테이블을 생성합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        train_table: 타겟 레이블 테이블
        logs_features: user_logs 피처 테이블
        txn_features: transactions 피처 테이블
        members_features: members 피처 테이블
        target_table: 생성할 테이블명
        force_overwrite: True면 기존 테이블 덮어씀
    """
    logger.info(f"=== {target_table} 테이블 생성 시작 ===")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    if _table_exists(con, target_table) and not force_overwrite:
        logger.info(f"{target_table} 이미 존재, 건너뜀")
        con.close()
        return

    con.execute(f"""
        CREATE OR REPLACE TABLE {target_table} AS
        SELECT
            t.user_id,
            t.is_churn,

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
            tx.plan_days,
            tx.last_is_cancel,
            tx.payment_method_id,
            tx.membership_duration,
            tx.tx_seq_length,
            tx.cancel_exist,
            tx.had_churn,

            -- members 기반 (1개)
            m.registration_init

        FROM {train_table} t
        LEFT JOIN {logs_features} ul ON t.user_id = ul.user_id
        LEFT JOIN {txn_features} tx ON t.user_id = tx.user_id
        LEFT JOIN {members_features} m ON t.user_id = m.user_id;
    """)

    row_count = _get_row_count(con, target_table)
    col_count = len(con.execute(f"DESCRIBE {target_table}").fetchall())
    logger.info(f"생성 완료: {row_count:,} 행, {col_count} 컬럼")

    # 컬럼 목록
    cols = [row[0] for row in con.execute(f"DESCRIBE {target_table}").fetchall()]
    logger.info(f"컬럼 목록: {cols}")

    # NULL 체크
    null_check = con.execute(f"""
        SELECT
            SUM(CASE WHEN num_25_avg IS NULL THEN 1 ELSE 0 END) AS null_logs,
            SUM(CASE WHEN plan_days IS NULL THEN 1 ELSE 0 END) AS null_txn,
            SUM(CASE WHEN registration_init IS NULL THEN 1 ELSE 0 END) AS null_members
        FROM {target_table}
    """).fetchone()
    logger.info(f"NULL 체크: logs={null_check[0]}, txn={null_check[1]}, members={null_check[2]}")

    # 타겟 분포
    target_dist = con.execute(f"""
        SELECT
            SUM(is_churn) AS churn_count,
            COUNT(*) AS total,
            AVG(is_churn) AS churn_rate
        FROM {target_table}
    """).fetchone()
    logger.info(f"타겟 분포: churn={target_dist[0]:,} ({target_dist[2]*100:.2f}%)")

    con.close()
    logger.info(f"=== {target_table} 테이블 생성 완료 ===\n")


# ============================================================================
# 3.2 검증 및 통계
# ============================================================================
def validate_ml_features(
    db_path: str,
    table_name: str = "ml_features",
) -> None:
    """
    ml_features 테이블의 유효성을 검증합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 검증할 테이블명
    """
    logger.info(f"=== {table_name} 검증 시작 ===")

    con = duckdb.connect(db_path, read_only=True)

    # 기본 정보
    row_count = _get_row_count(con, table_name)
    cols_info = con.execute(f"DESCRIBE {table_name}").fetchall()
    logger.info(f"테이블: {table_name}")
    logger.info(f"행 수: {row_count:,}")
    logger.info(f"컬럼 수: {len(cols_info)}")

    # 각 피처별 통계
    logger.info("\n--- 피처별 통계 ---")

    # 수치형 컬럼만
    numeric_cols = [col[0] for col in cols_info if col[0] not in ['user_id']]

    for col in numeric_cols:
        stats = con.execute(f"""
            SELECT
                COUNT({col}) AS non_null,
                AVG({col}) AS mean,
                MIN({col}) AS min_val,
                MAX({col}) AS max_val
            FROM {table_name}
        """).fetchone()
        null_count = row_count - stats[0]
        logger.info(f"  {col}: mean={stats[1]:.4f}, range=[{stats[2]:.4f}, {stats[3]:.4f}], null={null_count}")

    con.close()
    logger.info(f"=== {table_name} 검증 완료 ===\n")


# ============================================================================
# 3.3 피처별 상세 통계 출력
# ============================================================================
def print_feature_statistics(
    db_path: str,
    table_name: str = "ml_features",
) -> None:
    """
    ml_features 테이블의 각 피처별 분포와 통계값을 출력합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 분석할 테이블명
    """
    logger.info(f"=== {table_name} 피처별 상세 통계 ===\n")

    con = duckdb.connect(db_path, read_only=True)

    # 컬럼 정보 조회
    cols_info = con.execute(f"DESCRIBE {table_name}").fetchall()
    row_count = _get_row_count(con, table_name)

    logger.info(f"테이블: {table_name}")
    logger.info(f"총 행 수: {row_count:,}")
    logger.info(f"총 피처 수: {len(cols_info) - 2} (user_id, is_churn 제외)\n")

    # 타겟 변수 분포
    logger.info("=" * 70)
    logger.info("[ 타겟 변수: is_churn ]")
    logger.info("=" * 70)
    target_stats = con.execute(f"""
        SELECT
            SUM(CASE WHEN is_churn = 0 THEN 1 ELSE 0 END) AS class_0,
            SUM(CASE WHEN is_churn = 1 THEN 1 ELSE 0 END) AS class_1,
            AVG(is_churn) AS churn_rate
        FROM {table_name}
    """).fetchone()
    logger.info(f"  Class 0 (유지): {target_stats[0]:,} ({100*(1-target_stats[2]):.2f}%)")
    logger.info(f"  Class 1 (이탈): {target_stats[1]:,} ({100*target_stats[2]:.2f}%)")
    logger.info(f"  불균형 비율: 1:{target_stats[0]/target_stats[1]:.1f}\n")

    # 피처 카테고리 정의
    feature_categories = {
        "user_logs 기반 (평균)": [
            "num_25_avg", "num_50_avg", "num_75_avg", "num_985_avg",
            "num_100_avg", "num_unq_avg", "total_hours_avg"
        ],
        "user_logs 기반 (접속 텀)": [
            "log_term_min", "log_term_max", "log_term_avg", "log_term_median"
        ],
        "user_logs 기반 (활동 패턴)": [
            "log_days_ratio", "week_day_ratio", "log_acc", "max_ratio", "log_std"
        ],
        "user_logs 기반 (주별 청취량)": [
            "week_1", "week_2", "week_3", "week_4"
        ],
        "user_logs 기반 (청취 비율)": [
            "num_25_ratio", "num_100_ratio"
        ],
        "transactions 기반": [
            "plan_days", "last_is_cancel", "payment_method_id",
            "membership_duration", "tx_seq_length", "cancel_exist", "had_churn"
        ],
        "members 기반": [
            "registration_init"
        ]
    }

    for category, features in feature_categories.items():
        logger.info("=" * 70)
        logger.info(f"[ {category} ]")
        logger.info("=" * 70)

        for feature in features:
            # 해당 피처가 테이블에 존재하는지 확인
            col_names = [col[0] for col in cols_info]
            if feature not in col_names:
                logger.warning(f"  {feature}: 컬럼 없음")
                continue

            stats = con.execute(f"""
                SELECT
                    COUNT({feature}) AS non_null,
                    SUM(CASE WHEN {feature} IS NULL THEN 1 ELSE 0 END) AS null_cnt,
                    AVG({feature}) AS mean,
                    STDDEV({feature}) AS std,
                    MIN({feature}) AS min_val,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {feature}) AS q1,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY {feature}) AS median,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {feature}) AS q3,
                    MAX({feature}) AS max_val,
                    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY {feature}) AS p1,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY {feature}) AS p99
                FROM {table_name}
            """).fetchone()

            # 0 값 비율 (희소성)
            zero_stats = con.execute(f"""
                SELECT
                    SUM(CASE WHEN {feature} = 0 THEN 1 ELSE 0 END) AS zero_cnt
                FROM {table_name}
            """).fetchone()

            logger.info(f"\n  [{feature}]")
            logger.info(f"    NULL: {stats[1]:,} ({100*stats[1]/row_count:.2f}%)")
            logger.info(f"    Zero: {zero_stats[0]:,} ({100*zero_stats[0]/row_count:.2f}%)")
            logger.info(f"    Mean: {stats[2]:.4f}")
            logger.info(f"    Std:  {stats[3]:.4f}" if stats[3] else "    Std:  N/A")
            logger.info(f"    Min:  {stats[4]:.4f}")
            logger.info(f"    Q1:   {stats[5]:.4f}")
            logger.info(f"    Med:  {stats[6]:.4f}")
            logger.info(f"    Q3:   {stats[7]:.4f}")
            logger.info(f"    Max:  {stats[8]:.4f}")
            logger.info(f"    P1-P99: [{stats[9]:.4f}, {stats[10]:.4f}]")

            # Churn 그룹별 평균 비교
            churn_comparison = con.execute(f"""
                SELECT
                    AVG(CASE WHEN is_churn = 0 THEN {feature} END) AS mean_retained,
                    AVG(CASE WHEN is_churn = 1 THEN {feature} END) AS mean_churned
                FROM {table_name}
            """).fetchone()

            if churn_comparison[0] is not None and churn_comparison[1] is not None:
                diff_pct = 0
                if churn_comparison[0] != 0:
                    diff_pct = (churn_comparison[1] - churn_comparison[0]) / abs(churn_comparison[0]) * 100
                logger.info(f"    Churn=0 평균: {churn_comparison[0]:.4f}")
                logger.info(f"    Churn=1 평균: {churn_comparison[1]:.4f} ({diff_pct:+.1f}%)")

        logger.info("")

    # 피처 간 상관관계 요약 (타겟과의 상관관계)
    logger.info("=" * 70)
    logger.info("[ 타겟(is_churn)과의 상관관계 Top 10 ]")
    logger.info("=" * 70)

    correlations = []
    for col_name, col_type, *_ in cols_info:
        if col_name in ["user_id", "is_churn"]:
            continue
        try:
            corr = con.execute(f"""
                SELECT CORR(is_churn, {col_name}) FROM {table_name}
            """).fetchone()[0]
            if corr is not None:
                correlations.append((col_name, corr))
        except Exception:
            pass

    # 절대값 기준 정렬
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for i, (col_name, corr) in enumerate(correlations[:10], 1):
        sign = "+" if corr > 0 else ""
        logger.info(f"  {i:2d}. {col_name:25s}: {sign}{corr:.4f}")

    logger.info("")

    con.close()
    logger.info(f"=== {table_name} 피처별 상세 통계 완료 ===\n")


# ============================================================================
# 3.4 피처별 그래프 생성
# ============================================================================
def plot_feature_statistics(
    db_path: str,
    table_name: str = "ml_features",
    output_dir: str = "data/analysis",
    log_scale: bool = False,
    y_from_zero: bool = True,
) -> None:
    """
    ml_features 테이블의 각 피처별 분포 그래프를 생성하고 저장합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 분석할 테이블명
        output_dir: 그래프 저장 디렉토리
        log_scale: True면 Y축에 로그 스케일 적용
        y_from_zero: True면 Y축 0부터 시작, False면 데이터 범위에 맞춤
    """
    logger.info(f"=== {table_name} 피처별 그래프 생성 ===\n")

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"출력 디렉토리: {output_dir}")

    # 데이터 로드
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute(f"SELECT * FROM {table_name}").fetchdf()
    con.close()

    logger.info(f"데이터 로드 완료: {len(df):,} rows")

    # 스타일 설정
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10

    # 피처 카테고리 정의
    feature_categories = {
        "user_logs_avg": [
            "num_25_avg", "num_50_avg", "num_75_avg", "num_985_avg",
            "num_100_avg", "num_unq_avg", "total_hours_avg"
        ],
        "user_logs_term": [
            "log_term_min", "log_term_max", "log_term_avg", "log_term_median"
        ],
        "user_logs_activity": [
            "log_days_ratio", "week_day_ratio", "log_acc", "max_ratio", "log_std"
        ],
        "user_logs_weekly": [
            "week_1", "week_2", "week_3", "week_4"
        ],
        "user_logs_ratio": [
            "num_25_ratio", "num_100_ratio"
        ],
        "transactions": [
            "plan_days", "last_is_cancel", "payment_method_id",
            "membership_duration", "tx_seq_length", "cancel_exist", "had_churn"
        ],
        "members": [
            "registration_init"
        ]
    }

    # 모든 피처 리스트
    all_features = []
    for features in feature_categories.values():
        all_features.extend(features)

    # 실제 존재하는 피처만 필터링
    available_features = [f for f in all_features if f in df.columns]
    logger.info(f"분석 대상 피처: {len(available_features)}개\n")

    # 옵션 로깅
    logger.info(f"Y축 로그스케일: {'ON' if log_scale else 'OFF'}")
    logger.info(f"Y축 0부터 시작: {'ON' if y_from_zero else 'OFF (데이터 범위)'}")

    # ========================================================================
    # 개별 피처 히스토그램 생성
    # ========================================================================
    logger.info("피처별 히스토그램 생성 중...")

    for feature in tqdm(available_features, desc="히스토그램 생성"):
        fig, ax = plt.subplots(figsize=(10, 6))

        # NaN 비율 계산
        total_count = len(df)
        nan_count = df[feature].isna().sum()
        nan_ratio = nan_count / total_count * 100
        data = df[feature].dropna()

        ax.hist(data, bins=50, color='#3498db', edgecolor='white', alpha=0.8)

        if log_scale:
            ax.set_yscale('log')
            scale_label = " (y-log)"
        else:
            scale_label = ""

        # Y축 범위 설정
        if y_from_zero:
            y_max = ax.get_ylim()[1]
            if log_scale:
                ax.set_ylim(bottom=0.8, top=y_max * 1.1)
            else:
                ax.set_ylim(bottom=0, top=y_max * 1.05)

        # 통계 정보 표시
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
        ax.axvline(median_val, color='#2ecc71', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')

        ax.set_xlabel(feature + scale_label)
        ax.set_ylabel('Count')

        # 제목에 NaN 비율 표시
        nan_text = f" | NaN: {nan_ratio:.2f}%" if nan_ratio > 0 else ""
        ax.set_title(f'{feature} Distribution{scale_label}{nan_text}')

        # NaN 비율이 높으면 그래프에 별도 표시
        if nan_ratio > 0:
            ax.text(0.98, 0.95, f'NaN: {nan_count:,} ({nan_ratio:.2f}%)',
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.8))

        ax.legend(loc='upper left')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'hist_{feature}_{"log" if log_scale else "linear"}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    logger.info(f"  -> {len(available_features)}개 히스토그램 저장 완료")

    # ========================================================================
    # 피처 통계 요약 테이블 (CSV)
    # ========================================================================
    logger.info("\n피처 통계 요약 테이블 생성 중...")

    stats_data = []
    for feature in available_features:
        stats_data.append({
            'feature': feature,
            'mean': df[feature].mean(),
            'std': df[feature].std(),
            'min': df[feature].min(),
            'q1': df[feature].quantile(0.25),
            'median': df[feature].median(),
            'q3': df[feature].quantile(0.75),
            'max': df[feature].max(),
            'null_pct': df[feature].isnull().mean() * 100,
            'zero_pct': (df[feature] == 0).mean() * 100,
        })

    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(os.path.join(output_dir, 'feature_statistics.csv'), index=False)

    logger.info("  -> feature_statistics.csv 저장 완료")

    # ========================================================================
    # 완료 메시지
    # ========================================================================
    logger.info(f"\n=== {table_name} 그래프 생성 완료 ===")
    logger.info(f"총 {len(available_features)}개 히스토그램 + 1개 CSV가 {output_dir}에 저장됨\n")


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

    # ========================================================================
    # 파이프라인 실행 (각 줄을 주석처리하여 특정 단계 건너뛰기 가능)
    # ========================================================================

    # Phase 1: 사전 테이블 생성
    # ------------------------------------------------------------------------

    # 1.1 ml_user_last_txn (유저별 마지막 트랜잭션)
    create_user_last_txn(
        db_path=DB_PATH,
        source_table="transactions_seq",
        target_table="ml_user_last_txn",
        force_overwrite=True,
    )

    # 1.2 ml_user_membership_history (멤버십 이력 요약)
    create_user_membership_history(
        db_path=DB_PATH,
        source_table="transactions_seq",
        target_table="ml_user_membership_history",
        force_overwrite=True,
    )

    # 1.3 ml_user_logs_filtered (멤버십 기간 내 로그) - 대용량, 시간 소요
    create_user_logs_filtered(
        db_path=DB_PATH,
        logs_table="user_logs_merge",
        txn_table="ml_user_last_txn",
        target_table="ml_user_logs_filtered",
        force_overwrite=True,
    )

    # Phase 2: Feature 계산
    # ------------------------------------------------------------------------

    # 2.1 ml_user_logs_features (21개 피처) - 복잡한 집계, 시간 소요
    create_user_logs_features(
        db_path=DB_PATH,
        logs_table="ml_user_logs_filtered",
        txn_table="ml_user_last_txn",
        target_table="ml_user_logs_features",
        force_overwrite=True,
        fill_nan=False,      # True: 분모 0 -> NULL, False: default_value로 채움
        default_value=-1,   # fill_nan=False일 때 사용할 기본값
    )

    # 2.2 ml_transactions_features (7개 피처)
    create_transactions_features(
        db_path=DB_PATH,
        txn_table="ml_user_last_txn",
        history_table="ml_user_membership_history",
        target_table="ml_transactions_features",
        force_overwrite=True,
    )

    # 2.3 ml_members_features (1개 피처)
    create_members_features(
        db_path=DB_PATH,
        source_table="members_merge",
        target_table="ml_members_features",
        clip_date="2015-01-01",
        force_overwrite=True,
    )

    # Phase 3: 통합
    # ------------------------------------------------------------------------

    # 3.1 ml_features (최종 피처 테이블)
    create_ml_features(
        db_path=DB_PATH,
        train_table="train_merge",
        logs_features="ml_user_logs_features",
        txn_features="ml_transactions_features",
        members_features="ml_members_features",
        target_table="ml_features",
        force_overwrite=True,
    )

    # 3.2 검증
    validate_ml_features(
        db_path=DB_PATH,
        table_name="ml_features",
    )

    # 3.3 피처별 상세 통계 출력
    print_feature_statistics(
        db_path=DB_PATH,
        table_name="ml_features",
    )

    # 3.4 피처별 그래프 생성
    plot_feature_statistics(
        db_path=DB_PATH,
        table_name="ml_features",
        output_dir=ANALYSIS_DIR,
        log_scale=True,   # Y축 로그스케일
        y_from_zero=True,  # Y축 0부터 시작 (False면 데이터 범위)
    )

    # 3.5 Parquet 내보내기
    export_to_parquet(
        db_path=DB_PATH,
        output_dir=PARQUET_DIR,
        tables=["ml_features"],
        compression="zstd",
    )

    # 최종 데이터베이스 정보 출력
    show_database_info(db_path=DB_PATH)
