"""
KKBox 피처 엔지니어링 파이프라인

이 스크립트는 user_logs 데이터에 대해 다음 피처 엔지니어링을 수행합니다:
1. 모든 유니크 유저에 대해 누락된 날짜의 행을 0으로 채워 생성
2. date_idx 피처 컬럼 추가 (2015.01.01부터 0~의 정수)
3. user_id, date_idx 기준 오름차순 정렬
"""

import os
import sys
import logging
from typing import Optional
from datetime import date

import duckdb
import pandas as pd
from tqdm import tqdm

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
# 1. 누락된 날짜 행 생성 (0으로 채움)
# ============================================================================
def fill_missing_dates(
    db_path: str,
    source_table: str,
    target_table: str,
    user_id_col: str = "user_id",
    date_col: str = "date",
    start_date: str = "2015-01-01",
    end_date: str = "2017-03-31",
) -> None:
    """
    모든 유니크 유저에 대해 지정된 기간 내 누락된 날짜의 행을 0으로 채워 생성합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        source_table: 원본 테이블 이름
        target_table: 결과를 저장할 테이블 이름
        user_id_col: 유저 ID 컬럼명
        date_col: 날짜 컬럼명
        start_date: 시작 날짜 (YYYY-MM-DD)
        end_date: 종료 날짜 (YYYY-MM-DD)
    """
    logger.info(f"=== 누락된 날짜 행 생성 시작 ===")
    logger.info(f"원본 테이블: {source_table}")
    logger.info(f"대상 테이블: {target_table}")
    logger.info(f"기간: {start_date} ~ {end_date}")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    # 테이블 존재 확인
    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if source_table not in existing_tables:
        logger.error(f"원본 테이블 {source_table}이 존재하지 않습니다.")
        con.close()
        return

    # 원본 테이블 정보
    original_rows = con.execute(f"SELECT COUNT(*) FROM {source_table}").fetchone()[0]
    unique_users = con.execute(f"SELECT COUNT(DISTINCT {user_id_col}) FROM {source_table}").fetchone()[0]
    logger.info(f"원본 행 수: {original_rows:,}")
    logger.info(f"유니크 유저 수: {unique_users:,}")

    # 컬럼 정보 조회 (user_id, date 제외한 피처 컬럼들)
    cols_info = con.execute(f"DESCRIBE {source_table}").fetchall()
    all_cols = [row[0] for row in cols_info]
    feature_cols = [col for col in all_cols if col not in [user_id_col, date_col]]
    logger.info(f"피처 컬럼: {feature_cols}")

    # 날짜 범위의 일수 계산
    total_days = con.execute(f"""
        SELECT DATE '{end_date}' - DATE '{start_date}' + 1
    """).fetchone()[0]
    logger.info(f"총 날짜 수: {total_days:,}일")

    expected_rows = unique_users * total_days
    logger.info(f"예상 결과 행 수: {expected_rows:,} ({unique_users:,} 유저 x {total_days:,} 일)")

    # 0으로 채울 피처 컬럼들의 SELECT 구문 생성
    feature_zero_selects = ", ".join([f"COALESCE(s.{col}, 0) AS {col}" for col in feature_cols])

    # 전체 날짜 범위와 유저의 CROSS JOIN 후 LEFT JOIN으로 누락된 날짜 채우기
    logger.info("누락된 날짜 행 생성 중... (시간이 걸릴 수 있습니다)")

    con.execute(f"""
        CREATE OR REPLACE TABLE {target_table} AS
        WITH date_range AS (
            SELECT UNNEST(generate_series(DATE '{start_date}', DATE '{end_date}', INTERVAL 1 DAY))::DATE AS {date_col}
        ),
        all_users AS (
            SELECT DISTINCT {user_id_col} FROM {source_table}
        ),
        user_date_grid AS (
            SELECT
                u.{user_id_col},
                d.{date_col}
            FROM all_users u
            CROSS JOIN date_range d
        )
        SELECT
            g.{user_id_col},
            g.{date_col},
            {feature_zero_selects}
        FROM user_date_grid g
        LEFT JOIN {source_table} s
            ON g.{user_id_col} = s.{user_id_col}
            AND g.{date_col} = s.{date_col};
    """)

    # 결과 확인
    result_rows = con.execute(f"SELECT COUNT(*) FROM {target_table}").fetchone()[0]
    result_users = con.execute(f"SELECT COUNT(DISTINCT {user_id_col}) FROM {target_table}").fetchone()[0]

    added_rows = result_rows - original_rows
    logger.info(f"결과 행 수: {result_rows:,} (추가된 행: {added_rows:,})")
    logger.info(f"결과 유니크 유저 수: {result_users:,}")

    # 샘플 확인
    sample = con.execute(f"""
        SELECT * FROM {target_table}
        ORDER BY {user_id_col}, {date_col}
        LIMIT 5
    """).fetchdf()
    logger.info(f"샘플 데이터:\n{sample.to_string()}")

    con.close()
    logger.info(f"=== 누락된 날짜 행 생성 완료 ===\n")


# ============================================================================
# 2. date_idx 피처 컬럼 추가
# ============================================================================
def add_date_idx(
    db_path: str,
    table_name: str,
    date_col: str = "date",
    base_date: str = "2015-01-01",
    date_idx_col: str = "date_idx",
) -> None:
    """
    날짜를 기준 날짜부터의 일수(0~)로 변환한 date_idx 컬럼을 추가합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 대상 테이블 이름
        date_col: 날짜 컬럼명
        base_date: 기준 날짜 (이 날짜가 0)
        date_idx_col: 추가할 date_idx 컬럼명
    """
    logger.info(f"=== date_idx 피처 추가 시작 ===")
    logger.info(f"대상 테이블: {table_name}")
    logger.info(f"기준 날짜: {base_date} (이 날짜 = 0)")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    # 테이블 존재 확인
    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if table_name not in existing_tables:
        logger.error(f"테이블 {table_name}이 존재하지 않습니다.")
        con.close()
        return

    # 이미 date_idx 컬럼이 있는지 확인
    cols = [row[0] for row in con.execute(f"DESCRIBE {table_name}").fetchall()]
    if date_idx_col in cols:
        logger.warning(f"{date_idx_col} 컬럼이 이미 존재합니다. 재계산합니다.")

    # date_idx 컬럼 추가
    logger.info("date_idx 계산 중...")

    # 기존 컬럼 목록 (date_idx 제외)
    other_cols = [col for col in cols if col != date_idx_col]
    other_cols_str = ", ".join(other_cols)

    con.execute(f"""
        CREATE OR REPLACE TABLE {table_name}_temp AS
        SELECT
            {other_cols_str},
            ({date_col} - DATE '{base_date}')::INTEGER AS {date_idx_col}
        FROM {table_name};
    """)

    con.execute(f"DROP TABLE {table_name};")
    con.execute(f"ALTER TABLE {table_name}_temp RENAME TO {table_name};")

    # 결과 확인
    min_idx = con.execute(f"SELECT MIN({date_idx_col}) FROM {table_name}").fetchone()[0]
    max_idx = con.execute(f"SELECT MAX({date_idx_col}) FROM {table_name}").fetchone()[0]
    unique_dates = con.execute(f"SELECT COUNT(DISTINCT {date_idx_col}) FROM {table_name}").fetchone()[0]

    logger.info(f"date_idx 범위: {min_idx} ~ {max_idx}")
    logger.info(f"유니크 date_idx 수: {unique_dates:,}")

    # 샘플 확인
    sample = con.execute(f"""
        SELECT {date_col}, {date_idx_col}, COUNT(*) as cnt
        FROM {table_name}
        GROUP BY {date_col}, {date_idx_col}
        ORDER BY {date_idx_col}
        LIMIT 5
    """).fetchdf()
    logger.info(f"샘플 (날짜별 행 수):\n{sample.to_string()}")

    con.close()
    logger.info(f"=== date_idx 피처 추가 완료 ===\n")


# ============================================================================
# 3. 테이블 정렬
# ============================================================================
def sort_table(
    db_path: str,
    table_name: str,
    sort_columns: list[str],
    ascending: list[bool] = None,
) -> None:
    """
    테이블을 지정된 컬럼 기준으로 정렬합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 대상 테이블 이름
        sort_columns: 정렬 기준 컬럼 리스트
        ascending: 각 컬럼별 오름차순 여부 (기본값: 모두 True)
    """
    logger.info(f"=== 테이블 정렬 시작 ===")
    logger.info(f"대상 테이블: {table_name}")
    logger.info(f"정렬 기준: {sort_columns}")

    if ascending is None:
        ascending = [True] * len(sort_columns)

    logger.info(f"오름차순 여부: {ascending}")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    # 테이블 존재 확인
    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if table_name not in existing_tables:
        logger.error(f"테이블 {table_name}이 존재하지 않습니다.")
        con.close()
        return

    # ORDER BY 구문 생성
    order_parts = []
    for col, asc in zip(sort_columns, ascending):
        order_parts.append(f"{col} {'ASC' if asc else 'DESC'}")
    order_clause = ", ".join(order_parts)

    # 정렬된 테이블 생성
    logger.info("정렬 중... (시간이 걸릴 수 있습니다)")

    row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    logger.info(f"정렬할 행 수: {row_count:,}")

    con.execute(f"""
        CREATE OR REPLACE TABLE {table_name}_sorted AS
        SELECT * FROM {table_name}
        ORDER BY {order_clause};
    """)

    con.execute(f"DROP TABLE {table_name};")
    con.execute(f"ALTER TABLE {table_name}_sorted RENAME TO {table_name};")

    # 결과 확인
    sample = con.execute(f"""
        SELECT * FROM {table_name}
        LIMIT 10
    """).fetchdf()
    logger.info(f"정렬 후 상위 10행:\n{sample.to_string()}")

    con.close()
    logger.info(f"=== 테이블 정렬 완료 ===\n")


# ============================================================================
# 유틸리티: 테이블 덮어쓰기
# ============================================================================
def replace_table(
    db_path: str,
    source_table: str,
    target_table: str,
    drop_source: bool = True,
) -> None:
    """
    source_table의 데이터로 target_table을 덮어씁니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        source_table: 원본 테이블 이름 (이 데이터로 덮어씀)
        target_table: 덮어쓸 대상 테이블 이름
        drop_source: True면 덮어쓴 후 source_table 삭제 (기본값: True)
    """
    logger.info(f"=== 테이블 덮어쓰기 시작 ===")
    logger.info(f"{source_table} -> {target_table} 덮어쓰기")

    con = duckdb.connect(db_path)

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    # 원본 테이블 존재 확인
    if source_table not in existing_tables:
        logger.error(f"원본 테이블 {source_table}이 존재하지 않습니다.")
        con.close()
        return

    # source 테이블 정보
    source_rows = con.execute(f"SELECT COUNT(*) FROM {source_table}").fetchone()[0]
    source_cols = len(con.execute(f"DESCRIBE {source_table}").fetchall())

    # target 테이블이 존재하면 삭제
    if target_table in existing_tables:
        target_rows = con.execute(f"SELECT COUNT(*) FROM {target_table}").fetchone()[0]
        logger.info(f"기존 {target_table}: {target_rows:,} 행 -> 삭제")
        con.execute(f"DROP TABLE {target_table}")

    # source를 target으로 이름 변경
    con.execute(f"ALTER TABLE {source_table} RENAME TO {target_table}")

    logger.info(f"덮어쓰기 완료: {source_rows:,} 행, {source_cols} 열")

    con.close()
    logger.info(f"=== 테이블 덮어쓰기 완료 ===\n")


# ============================================================================
# 4. user_logs와 transactions 조인
# ============================================================================
def join_user_logs_with_transactions(
    db_path: str,
    user_logs_table: str,
    transactions_table: str,
    target_table: str,
    user_id_col: str = "user_id",
    user_logs_date_col: str = "date",
    transactions_date_col: str = "transaction_date",
    default_int_value: Optional[int] = -1,
    default_date_value: Optional[str] = "2000-01-01",
) -> None:
    """
    user_logs와 transactions를 user_id와 날짜 기준으로 LEFT JOIN합니다.
    트랜잭션이 없는 경우 transactions의 피처들을 기본값으로 채웁니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        user_logs_table: user_logs 테이블 이름
        transactions_table: transactions 테이블 이름
        target_table: 결과를 저장할 테이블 이름
        user_id_col: 유저 ID 컬럼명
        user_logs_date_col: user_logs의 날짜 컬럼명
        transactions_date_col: transactions의 날짜 컬럼명
        default_int_value: 트랜잭션 없을 때 정수/실수 피처 기본값 (기본: -1, None이면 NULL 유지)
        default_date_value: 트랜잭션 없을 때 날짜 피처 기본값 (기본: 2000-01-01, None이면 NULL 유지)
    """
    logger.info(f"=== user_logs + transactions 조인 시작 ===")
    logger.info(f"user_logs: {user_logs_table}")
    logger.info(f"transactions: {transactions_table}")
    logger.info(f"결과 테이블: {target_table}")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    # 테이블 존재 확인
    if user_logs_table not in existing_tables:
        logger.error(f"{user_logs_table} 테이블이 존재하지 않습니다.")
        con.close()
        return

    if transactions_table not in existing_tables:
        logger.error(f"{transactions_table} 테이블이 존재하지 않습니다.")
        con.close()
        return

    # user_logs 컬럼 정보
    ul_cols_info = con.execute(f"DESCRIBE {user_logs_table}").fetchall()
    ul_cols = [(row[0], row[1]) for row in ul_cols_info]
    logger.info(f"user_logs 컬럼: {[c[0] for c in ul_cols]}")

    # transactions 컬럼 정보
    tx_cols_info = con.execute(f"DESCRIBE {transactions_table}").fetchall()
    tx_cols = [(row[0], row[1]) for row in tx_cols_info]
    logger.info(f"transactions 컬럼: {[c[0] for c in tx_cols]}")

    # transactions에서 조인 키 제외한 피처 컬럼들
    tx_feature_cols = [(name, dtype) for name, dtype in tx_cols
                       if name not in [user_id_col, transactions_date_col]]
    logger.info(f"transactions 피처 컬럼: {[c[0] for c in tx_feature_cols]}")

    # 원본 테이블 통계
    ul_rows = con.execute(f"SELECT COUNT(*) FROM {user_logs_table}").fetchone()[0]
    tx_rows = con.execute(f"SELECT COUNT(*) FROM {transactions_table}").fetchone()[0]
    logger.info(f"user_logs 행 수: {ul_rows:,}")
    logger.info(f"transactions 행 수: {tx_rows:,}")

    # user_logs 컬럼 SELECT 구문
    ul_select_parts = [f"ul.{col[0]}" for col in ul_cols]

    # transactions 피처 컬럼 SELECT 구문 (NULL일 경우 기본값으로 대체)
    # default_int_value가 None이면 NULL 유지, 값이 있으면 COALESCE로 대체
    tx_select_parts = []
    for col_name, col_type in tx_feature_cols:
        col_type_upper = col_type.upper()
        if "DATE" in col_type_upper or "TIMESTAMP" in col_type_upper:
            if default_date_value is None:
                tx_select_parts.append(f"tx.{col_name} AS tx_{col_name}")
            else:
                tx_select_parts.append(
                    f"COALESCE(tx.{col_name}, DATE '{default_date_value}') AS tx_{col_name}"
                )
        else:
            if default_int_value is None:
                tx_select_parts.append(f"tx.{col_name} AS tx_{col_name}")
            else:
                tx_select_parts.append(
                    f"COALESCE(tx.{col_name}, {default_int_value}) AS tx_{col_name}"
                )

    # 전체 SELECT 구문
    all_select = ", ".join(ul_select_parts + tx_select_parts)

    # LEFT JOIN 실행
    logger.info("조인 중... (시간이 걸릴 수 있습니다)")

    con.execute(f"""
        CREATE OR REPLACE TABLE {target_table} AS
        SELECT
            {all_select}
        FROM {user_logs_table} ul
        LEFT JOIN {transactions_table} tx
            ON ul.{user_id_col} = tx.{user_id_col}
            AND ul.{user_logs_date_col} = tx.{transactions_date_col};
    """)

    # 결과 확인
    result_rows = con.execute(f"SELECT COUNT(*) FROM {target_table}").fetchone()[0]
    result_cols = len(con.execute(f"DESCRIBE {target_table}").fetchall())

    logger.info(f"결과 행 수: {result_rows:,}")
    logger.info(f"결과 컬럼 수: {result_cols}")

    # 트랜잭션 매칭 통계
    if tx_feature_cols:
        first_tx_col = f"tx_{tx_feature_cols[0][0]}"
        if default_int_value is None:
            # NULL로 채운 경우: NULL이 아닌 행이 매칭된 행
            matched_rows = con.execute(f"""
                SELECT COUNT(*) FROM {target_table}
                WHERE {first_tx_col} IS NOT NULL
            """).fetchone()[0]
        else:
            # 기본값으로 채운 경우: 기본값이 아닌 행이 매칭된 행
            matched_rows = con.execute(f"""
                SELECT COUNT(*) FROM {target_table}
                WHERE {first_tx_col} != {default_int_value}
            """).fetchone()[0]
        unmatched_rows = result_rows - matched_rows
        logger.info(f"트랜잭션 매칭된 행: {matched_rows:,} ({matched_rows/result_rows*100:.2f}%)")
        logger.info(f"트랜잭션 없는 행: {unmatched_rows:,} ({unmatched_rows/result_rows*100:.2f}%)")

    # 샘플 확인
    sample = con.execute(f"""
        SELECT * FROM {target_table}
        LIMIT 5
    """).fetchdf()
    logger.info(f"샘플 데이터:\n{sample.to_string()}")

    con.close()
    logger.info(f"=== user_logs + transactions 조인 완료 ===\n")


# ============================================================================
# 5. Forward Fill (이전 값으로 채우기)
# ============================================================================
def forward_fill_column(
    db_path: str,
    table_name: str,
    column_name: str,
    partition_col: str = "user_id",
    order_col: str = "date_idx",
    null_indicator_value: str = None,
    batch_size: int = 10000,
    expire_check_date_col: str = None,
) -> None:
    """
    특정 컬럼의 기본값/NULL을 이전 행의 값으로 채웁니다 (Forward Fill).
    각 유저별로 시간순 정렬 후 이전 유효값을 전파합니다.
    메모리 효율을 위해 유저 배치 단위로 처리합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 대상 테이블 이름
        column_name: forward fill을 적용할 컬럼명
        partition_col: 파티션 기준 컬럼 (기본: user_id)
        order_col: 정렬 기준 컬럼 (기본: date_idx)
        null_indicator_value: NULL로 취급할 값 (예: "DATE '2000-01-01'" 또는 "-1")
                              None이면 실제 NULL만 처리
        batch_size: 한 번에 처리할 유저 수 (기본: 10000)
        expire_check_date_col: 만료 체크용 날짜 컬럼 (예: "date")
                               설정 시, forward fill된 값이 이 컬럼보다 작으면 NULL로 설정
                               (즉, 만료된 경우 NULL 처리)
    """
    logger.info(f"=== Forward Fill 시작 ===")
    logger.info(f"테이블: {table_name}")
    logger.info(f"대상 컬럼: {column_name}")
    logger.info(f"파티션: {partition_col}, 정렬: {order_col}")
    logger.info(f"배치 크기: {batch_size:,} 유저")
    if expire_check_date_col:
        logger.info(f"만료 체크: {column_name} < {expire_check_date_col} 이면 NULL 처리")

    con = duckdb.connect(db_path)
    con.execute("SET threads=4;")
    con.execute("SET preserve_insertion_order=false;")

    # 테이블 존재 확인
    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if table_name not in existing_tables:
        logger.error(f"테이블 {table_name}이 존재하지 않습니다.")
        con.close()
        return

    # 컬럼 존재 확인
    cols = [row[0] for row in con.execute(f"DESCRIBE {table_name}").fetchall()]
    if column_name not in cols:
        logger.error(f"컬럼 {column_name}이 존재하지 않습니다.")
        con.close()
        return

    # Forward Fill 전 통계
    if null_indicator_value:
        before_null_count = con.execute(f"""
            SELECT COUNT(*) FROM {table_name}
            WHERE {column_name} = {null_indicator_value} OR {column_name} IS NULL
        """).fetchone()[0]
    else:
        before_null_count = con.execute(f"""
            SELECT COUNT(*) FROM {table_name}
            WHERE {column_name} IS NULL
        """).fetchone()[0]

    total_rows = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    logger.info(f"Forward Fill 전 NULL/기본값 행: {before_null_count:,} ({before_null_count/total_rows*100:.2f}%)")

    # 다른 컬럼들
    other_cols = [col for col in cols if col != column_name]
    other_cols_str = ", ".join(other_cols)

    # 유니크 유저 목록 조회
    unique_users = con.execute(f"""
        SELECT DISTINCT {partition_col} FROM {table_name} ORDER BY {partition_col}
    """).fetchall()
    unique_users = [row[0] for row in unique_users]
    total_users = len(unique_users)
    logger.info(f"총 유저 수: {total_users:,}")

    # 결과 테이블 생성 (첫 배치에서 CREATE, 이후 INSERT)
    result_table = f"{table_name}_ffill"
    con.execute(f"DROP TABLE IF EXISTS {result_table}")

    # 배치 단위로 처리
    num_batches = (total_users + batch_size - 1) // batch_size
    logger.info(f"총 배치 수: {num_batches}")

    for batch_idx in tqdm(range(num_batches), desc="Forward Fill 배치 처리"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_users)
        batch_users = unique_users[start_idx:end_idx]

        # 유저 ID 리스트를 SQL IN 절로 변환
        user_list = ", ".join([str(u) for u in batch_users])

        # Forward Fill 로직 (배치 단위)
        # 먼저 forward fill을 수행하고, expire_check_date_col이 있으면 만료 체크 적용
        if null_indicator_value:
            ffill_expr = f"""
                LAST_VALUE(
                    CASE WHEN {column_name} = {null_indicator_value} THEN NULL ELSE {column_name} END
                    IGNORE NULLS
                ) OVER (
                    PARTITION BY {partition_col}
                    ORDER BY {order_col}
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                )
            """
        else:
            ffill_expr = f"""
                LAST_VALUE({column_name} IGNORE NULLS) OVER (
                    PARTITION BY {partition_col}
                    ORDER BY {order_col}
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                )
            """

        # 만료 체크: 서브쿼리로 forward fill 먼저 계산 후, 외부에서 만료 체크 적용
        if expire_check_date_col:
            query = f"""
                SELECT
                    {other_cols_str},
                    CASE
                        WHEN _ffilled_val < {expire_check_date_col} THEN NULL
                        ELSE _ffilled_val
                    END AS {column_name}
                FROM (
                    SELECT
                        {other_cols_str},
                        {ffill_expr} AS _ffilled_val
                    FROM {table_name}
                    WHERE {partition_col} IN ({user_list})
                ) subq
            """
        else:
            query = f"""
                SELECT
                    {other_cols_str},
                    {ffill_expr} AS {column_name}
                FROM {table_name}
                WHERE {partition_col} IN ({user_list})
            """

        if batch_idx == 0:
            con.execute(f"CREATE TABLE {result_table} AS {query}")
        else:
            con.execute(f"INSERT INTO {result_table} {query}")

    # 원본 테이블 교체
    con.execute(f"DROP TABLE {table_name};")
    con.execute(f"ALTER TABLE {table_name}_ffill RENAME TO {table_name};")

    # Forward Fill 후 통계
    after_null_count = con.execute(f"""
        SELECT COUNT(*) FROM {table_name}
        WHERE {column_name} IS NULL
    """).fetchone()[0]

    filled_count = before_null_count - after_null_count
    logger.info(f"Forward Fill 후 NULL 행: {after_null_count:,}")
    logger.info(f"채워진 행: {filled_count:,}")

    # 샘플 확인 (유저 1명의 시계열)
    sample_user = con.execute(f"SELECT {partition_col} FROM {table_name} LIMIT 1").fetchone()[0]
    sample = con.execute(f"""
        SELECT {partition_col}, {order_col}, {column_name}
        FROM {table_name}
        WHERE {partition_col} = {sample_user}
        ORDER BY {order_col}
        LIMIT 20
    """).fetchdf()
    logger.info(f"샘플 (유저 {sample_user}):\n{sample.to_string()}")

    con.close()
    logger.info(f"=== Forward Fill 완료 ===\n")


# ============================================================================
# 유틸리티: 테이블 정보 출력
# ============================================================================
def show_table_info(
    db_path: str,
    table_name: str,
) -> None:
    """테이블의 상세 정보를 출력합니다."""
    logger.info(f"=== 테이블 정보: {table_name} ===")

    con = duckdb.connect(db_path)

    # 테이블 존재 확인
    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if table_name not in existing_tables:
        logger.error(f"테이블 {table_name}이 존재하지 않습니다.")
        con.close()
        return

    # 기본 정보
    row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    cols_info = con.execute(f"DESCRIBE {table_name}").fetchall()

    logger.info(f"행 수: {row_count:,}")
    logger.info(f"컬럼 수: {len(cols_info)}")
    logger.info(f"컬럼 정보:")
    for col in cols_info:
        logger.info(f"  {col[0]}: {col[1]}")

    # 샘플 데이터
    sample = con.execute(f"SELECT * FROM {table_name} LIMIT 5").fetchdf()
    logger.info(f"샘플 데이터:\n{sample.to_string()}")

    # 결측치 정보
    logger.info("결측치 정보:")
    for col in cols_info:
        col_name = col[0]
        null_count = con.execute(f"""
            SELECT SUM(CASE WHEN {col_name} IS NULL THEN 1 ELSE 0 END)
            FROM {table_name}
        """).fetchone()[0]
        if null_count > 0:
            logger.info(f"  {col_name}: {null_count:,} ({null_count/row_count*100:.2f}%)")

    con.close()
    logger.info(f"================================\n")


# ============================================================================
# 7. 날짜 컬럼 하방 클리핑
# ============================================================================
def clip_date_lower(
    db_path: str,
    table_name: str,
    date_col: str,
    min_date: str,
) -> None:
    """
    날짜 컬럼의 값을 지정된 최소 날짜로 하방 클리핑합니다.
    min_date보다 이전인 날짜는 모두 min_date로 변경됩니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 대상 테이블 이름
        date_col: 클리핑할 날짜 컬럼명
        min_date: 최소 날짜 (YYYY-MM-DD), 이보다 이전 날짜는 이 값으로 변경
    """
    logger.info(f"=== 날짜 하방 클리핑 시작 ===")
    logger.info(f"대상 테이블: {table_name}")
    logger.info(f"대상 컬럼: {date_col}")
    logger.info(f"최소 날짜: {min_date}")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    # 테이블 존재 확인
    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if table_name not in existing_tables:
        logger.error(f"테이블 {table_name}이 존재하지 않습니다.")
        con.close()
        return

    # 컬럼 존재 확인
    cols = [row[0] for row in con.execute(f"DESCRIBE {table_name}").fetchall()]
    if date_col not in cols:
        logger.error(f"컬럼 {date_col}이 존재하지 않습니다.")
        con.close()
        return

    # 클리핑 전 통계
    total_rows = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    before_clip_count = con.execute(f"""
        SELECT COUNT(*) FROM {table_name}
        WHERE {date_col} < DATE '{min_date}'
    """).fetchone()[0]
    min_before = con.execute(f"SELECT MIN({date_col}) FROM {table_name}").fetchone()[0]
    max_before = con.execute(f"SELECT MAX({date_col}) FROM {table_name}").fetchone()[0]

    logger.info(f"클리핑 전 범위: {min_before} ~ {max_before}")
    logger.info(f"클리핑 대상 행 수: {before_clip_count:,} ({before_clip_count/total_rows*100:.2f}%)")

    # 클리핑 수행
    logger.info("클리핑 중...")

    other_cols = [col for col in cols if col != date_col]
    other_cols_str = ", ".join(other_cols)

    con.execute(f"""
        CREATE OR REPLACE TABLE {table_name}_clipped AS
        SELECT
            {other_cols_str},
            CASE
                WHEN {date_col} < DATE '{min_date}' THEN DATE '{min_date}'
                ELSE {date_col}
            END AS {date_col}
        FROM {table_name};
    """)

    con.execute(f"DROP TABLE {table_name};")
    con.execute(f"ALTER TABLE {table_name}_clipped RENAME TO {table_name};")

    # 클리핑 후 통계
    after_clip_count = con.execute(f"""
        SELECT COUNT(*) FROM {table_name}
        WHERE {date_col} < DATE '{min_date}'
    """).fetchone()[0]
    min_after = con.execute(f"SELECT MIN({date_col}) FROM {table_name}").fetchone()[0]
    max_after = con.execute(f"SELECT MAX({date_col}) FROM {table_name}").fetchone()[0]

    logger.info(f"클리핑 후 범위: {min_after} ~ {max_after}")
    logger.info(f"클리핑된 행 수: {before_clip_count:,}")
    logger.info(f"클리핑 후 최소 날짜 미만 행 수: {after_clip_count:,}")

    # 샘플 확인
    sample = con.execute(f"""
        SELECT {date_col}, COUNT(*) as cnt
        FROM {table_name}
        GROUP BY {date_col}
        ORDER BY {date_col}
        LIMIT 10
    """).fetchdf()
    logger.info(f"클리핑 후 날짜별 분포 (상위 10개):\n{sample.to_string()}")

    con.close()
    logger.info(f"=== 날짜 하방 클리핑 완료 ===\n")


# ============================================================================
# 메인 실행
# ============================================================================
if __name__ == "__main__":
    # ========================================================================
    # 상수 설정
    # ========================================================================
    DB_PATH = "data/data.duckdb"

    # 대상 테이블
    SOURCE_TABLE = "user_logs_merge"  # 원본 user_logs 테이블
    TARGET_TABLE = "user_logs_filled"  # 결과 테이블

    # 날짜 범위
    START_DATE = "2015-01-01"
    END_DATE = "2017-03-31"

    # ========================================================================
    # 파이프라인 실행 (각 줄을 주석처리하여 특정 단계 건너뛰기 가능)
    # ========================================================================

    # (선택) 원본 테이블 정보 확인
    show_table_info(db_path=DB_PATH, table_name=SOURCE_TABLE)

    # 1. 누락된 날짜 행 생성 (0으로 채움)
    # fill_missing_dates(
    #     db_path=DB_PATH,
    #     source_table=SOURCE_TABLE,
    #     target_table=TARGET_TABLE,
    #     user_id_col="user_id",
    #     date_col="date",
    #     start_date=START_DATE,
    #     end_date=END_DATE,
    # )

    # # 2. date_idx 피처 추가 (2015-01-01 = 0)
    # add_date_idx(
    #     db_path=DB_PATH,
    #     table_name=TARGET_TABLE,
    #     date_col="date",
    #     base_date=START_DATE,
    #     date_idx_col="date_idx",
    # )

    # # 3. user_id, date_idx 기준 오름차순 정렬
    # sort_table(
    #     db_path=DB_PATH,
    #     table_name=TARGET_TABLE,
    #     sort_columns=["user_id", "date_idx"],
    #     ascending=[True, True],
    # )

    # # 4. 결과 테이블로 원본 테이블 덮어쓰기 (user_logs_filled -> user_logs_merge)
    # replace_table(
    #     db_path=DB_PATH,
    #     source_table=TARGET_TABLE,
    #     target_table=SOURCE_TABLE,
    #     drop_source=True,
    # )

    # 5. user_logs와 transactions 조인
    # join_user_logs_with_transactions(
    #     db_path=DB_PATH,
    #     user_logs_table="user_logs_merge",
    #     transactions_table="transactions_merge",
    #     target_table="user_logs_with_transactions",
    #     user_id_col="user_id",
    #     user_logs_date_col="date",
    #     transactions_date_col="transaction_date",
    #     default_int_value=-1,
    #     default_date_value="2000-01-01",
    # )

    # 6. tx_membership_expire_date Forward Fill (이전 트랜잭션의 expire_date로 채움)
    #    - 만료 체크: 로그 날짜(date)가 expire_date보다 크면 (멤버십 만료) NULL 처리
    forward_fill_column(
        db_path=DB_PATH,
        table_name="user_logs_with_transactions",
        column_name="tx_membership_expire_date",
        partition_col="user_id",
        order_col="date_idx",
        null_indicator_value="DATE '2000-01-01'",  # 기본값 2000-01-01을 NULL로 취급
        batch_size=10000,  # 메모리 효율을 위한 유저 배치 크기
        expire_check_date_col="date",  # 로그 날짜가 expire_date보다 크면 NULL 처리
    )

    # (선택) 결과 테이블 정보 확인
    show_table_info(db_path=DB_PATH, table_name="user_logs_with_transactions")

    # 7. members_merge의 registration_init_time 하방 클리핑 (2015-01-01 이전 -> 2015-01-01)
    # clip_date_lower(
    #     db_path=DB_PATH,
    #     table_name="members_merge",
    #     date_col="registration_init_time",
    #     min_date=START_DATE,
    # )
