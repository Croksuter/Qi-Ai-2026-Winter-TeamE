"""
DuckDB 데이터베이스 유틸리티 함수 모음

범용적으로 사용 가능한 데이터베이스 조작 및 분석 함수들을 제공합니다.

카테고리:
1. Database/Table Operations - 테이블 삭제, 복사, 정보 조회
2. Data I/O - CSV 로드, Parquet 내보내기
3. Data Transformation - 데이터 정제, 조건부 변환, 컬럼 변환
4. Data Analysis - 분포 분석, 상관관계 분석
"""

import os
import sys
import logging
from typing import Optional

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
# 1. Database/Table Operations
# ============================================================================
def drop_tables(
    db_path: str,
    tables: list[str],
) -> None:
    """
    지정된 테이블들을 삭제합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        tables: 삭제할 테이블 이름 리스트
    """
    logger.info(f"=== 테이블 삭제 시작 ===")
    logger.info(f"삭제 대상: {tables}")

    con = duckdb.connect(db_path)

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    dropped_count = 0
    for table in tqdm(tables, desc="테이블 삭제"):
        if table not in existing_tables:
            logger.warning(f"  {table}: 존재하지 않음, 건너뜀")
            continue

        row_count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        con.execute(f"DROP TABLE {table}")
        logger.info(f"  {table}: 삭제됨 ({row_count:,} 행)")
        dropped_count += 1

    final_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    logger.info(f"삭제된 테이블 수: {dropped_count}")
    logger.info(f"남은 테이블 목록: {final_tables}")

    con.close()
    logger.info(f"=== 테이블 삭제 완료 ===\n")


def copy_table(
    db_path: str,
    source_table: str,
    target_table: str,
    force_overwrite: bool = True,
) -> None:
    """
    테이블을 다른 이름으로 복사합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        source_table: 원본 테이블 이름
        target_table: 복사할 새 테이블 이름
        force_overwrite: True면 대상 테이블이 존재할 경우 덮어씀 (기본값: True)
    """
    logger.info(f"=== 테이블 복사 시작 ===")
    logger.info(f"원본: {source_table} -> 대상: {target_table}")

    con = duckdb.connect(db_path)

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    if source_table not in existing_tables:
        logger.error(f"원본 테이블 {source_table}이 존재하지 않습니다.")
        con.close()
        return

    if target_table in existing_tables:
        if not force_overwrite:
            logger.warning(f"대상 테이블 {target_table}이 이미 존재, 건너뜀")
            con.close()
            return
        else:
            logger.info(f"대상 테이블 {target_table}이 이미 존재, 덮어쓰기")

    con.execute(f"CREATE OR REPLACE TABLE {target_table} AS SELECT * FROM {source_table}")

    row_count = con.execute(f"SELECT COUNT(*) FROM {target_table}").fetchone()[0]
    col_count = len(con.execute(f"DESCRIBE {target_table}").fetchall())
    logger.info(f"복사 완료: {row_count:,} 행, {col_count} 열")

    con.close()
    logger.info(f"=== 테이블 복사 완료 ===\n")


def copy_tables(
    db_path: str,
    table_mapping: dict[str, str],
    force_overwrite: bool = True,
) -> None:
    """
    여러 테이블을 다른 이름으로 복사합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        table_mapping: {원본 테이블명: 대상 테이블명} 딕셔너리
        force_overwrite: True면 대상 테이블이 존재할 경우 덮어씀 (기본값: True)
    """
    logger.info(f"=== 여러 테이블 복사 시작 ===")
    logger.info(f"복사 대상: {table_mapping}")

    con = duckdb.connect(db_path)

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    copied_count = 0
    for source, target in tqdm(table_mapping.items(), desc="테이블 복사"):
        if source not in existing_tables:
            logger.warning(f"  {source}: 존재하지 않음, 건너뜀")
            continue

        if target in existing_tables and not force_overwrite:
            logger.warning(f"  {target}: 이미 존재, 건너뜀")
            continue

        con.execute(f"CREATE OR REPLACE TABLE {target} AS SELECT * FROM {source}")
        row_count = con.execute(f"SELECT COUNT(*) FROM {target}").fetchone()[0]
        logger.info(f"  {source} -> {target}: {row_count:,} 행")
        copied_count += 1

    logger.info(f"복사된 테이블 수: {copied_count}")

    con.close()
    logger.info(f"=== 여러 테이블 복사 완료 ===\n")


def rename_column(
    db_path: str,
    table_name: str,
    old_name: str,
    new_name: str,
) -> bool:
    """
    테이블의 컬럼명을 변경합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 대상 테이블명
        old_name: 기존 컬럼명
        new_name: 새 컬럼명

    Returns:
        bool: 성공 여부
    """
    con = duckdb.connect(db_path)

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if table_name not in existing_tables:
        logger.warning(f"{table_name}: 존재하지 않음")
        con.close()
        return False

    cols = [row[0] for row in con.execute(f"DESCRIBE {table_name}").fetchall()]

    if old_name not in cols:
        logger.warning(f"{table_name}.{old_name}: 컬럼 없음")
        con.close()
        return False

    if new_name in cols:
        logger.warning(f"{table_name}.{new_name}: 이미 존재")
        con.close()
        return False

    con.execute(f"ALTER TABLE {table_name} RENAME COLUMN {old_name} TO {new_name};")
    logger.info(f"{table_name}: {old_name} -> {new_name} 변경 완료")

    con.close()
    return True


def add_column(
    db_path: str,
    table_name: str,
    column_name: str,
    column_type: str,
    default_value: Optional[str] = None,
    force_overwrite: bool = True,
) -> bool:
    """
    테이블에 새 컬럼을 추가합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 대상 테이블명
        column_name: 추가할 컬럼명
        column_type: 컬럼 타입 (예: "INTEGER", "VARCHAR", "DOUBLE")
        default_value: 기본값 SQL 표현식 (예: "0", "NULL", "'unknown'")
        force_overwrite: True면 기존 컬럼 삭제 후 재생성 (기본값: True)

    Returns:
        bool: 성공 여부
    """
    con = duckdb.connect(db_path)

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if table_name not in existing_tables:
        logger.error(f"테이블 {table_name}이 존재하지 않습니다.")
        con.close()
        return False

    cols = [row[0] for row in con.execute(f"DESCRIBE {table_name}").fetchall()]

    if column_name in cols:
        if not force_overwrite:
            logger.warning(f"컬럼 {column_name}이 이미 존재, 건너뜀")
            con.close()
            return False
        else:
            con.execute(f"ALTER TABLE {table_name} DROP COLUMN {column_name}")
            logger.info(f"기존 {column_name} 컬럼 삭제")

    con.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")

    if default_value is not None:
        con.execute(f"UPDATE {table_name} SET {column_name} = {default_value}")

    logger.info(f"{table_name}에 {column_name} ({column_type}) 컬럼 추가 완료")

    con.close()
    return True


def show_database_info(db_path: str) -> None:
    """데이터베이스의 전체 정보를 출력합니다."""
    logger.info(f"=== 데이터베이스 정보 ===")

    con = duckdb.connect(db_path, read_only=True)

    tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    logger.info(f"테이블 수: {len(tables)}")

    for table in tables:
        row_count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        cols = con.execute(f"DESCRIBE {table}").fetchall()
        logger.info(f"  {table}: {row_count:,} 행, {len(cols)} 열")

    con.close()
    logger.info(f"=========================\n")


def show_table_info(
    db_path: str,
    table_name: str,
    show_sample: bool = False,
    sample_size: int = 5,
) -> None:
    """
    테이블의 상세 정보를 출력합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 테이블명
        show_sample: 샘플 데이터 출력 여부 (기본값: False)
        sample_size: 샘플 크기 (기본값: 5)
    """
    con = duckdb.connect(db_path, read_only=True)

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if table_name not in existing_tables:
        logger.error(f"테이블 {table_name}이 존재하지 않습니다.")
        con.close()
        return

    row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    cols_info = con.execute(f"DESCRIBE {table_name}").fetchall()

    logger.info(f"=== 테이블 정보: {table_name} ===")
    logger.info(f"행 수: {row_count:,}")
    logger.info(f"컬럼 수: {len(cols_info)}")
    logger.info(f"컬럼 목록:")
    for col_name, col_type, *_ in cols_info:
        logger.info(f"  {col_name}: {col_type}")

    if show_sample:
        sample = con.execute(f"SELECT * FROM {table_name} LIMIT {sample_size}").fetchdf()
        logger.info(f"\n샘플 데이터:\n{sample.to_string()}")

    con.close()
    logger.info(f"==============================\n")


def get_table_columns(
    db_path: str,
    table_name: str,
) -> list[tuple[str, str]]:
    """
    테이블의 컬럼 목록을 반환합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 테이블명

    Returns:
        list[tuple[str, str]]: [(컬럼명, 타입), ...] 리스트
    """
    con = duckdb.connect(db_path, read_only=True)
    cols_info = con.execute(f"DESCRIBE {table_name}").fetchall()
    con.close()
    return [(row[0], row[1]) for row in cols_info]


def table_exists(db_path: str, table_name: str) -> bool:
    """테이블 존재 여부를 확인합니다."""
    con = duckdb.connect(db_path, read_only=True)
    tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    con.close()
    return table_name in tables


def get_row_count(db_path: str, table_name: str) -> int:
    """테이블의 행 수를 반환합니다."""
    con = duckdb.connect(db_path, read_only=True)
    count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    con.close()
    return count


# ============================================================================
# 2. Data I/O
# ============================================================================
def load_csv_to_duckdb(
    db_path: str,
    csv_dir: str,
    csv_files: list[str],
    chunksize: int = 100_000,
    force_overwrite: bool = True,
) -> None:
    """
    지정된 CSV 파일들을 DuckDB 테이블로 로드합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        csv_dir: CSV 파일들이 있는 디렉토리 경로
        csv_files: 로드할 CSV 파일명 리스트 (확장자 포함)
        chunksize: 한 번에 읽을 행 수
        force_overwrite: True면 이미 존재하는 테이블을 덮어씀 (기본값: True)
    """
    logger.info(f"=== CSV -> DuckDB 로드 시작 ===")
    logger.info(f"데이터베이스: {db_path}")
    logger.info(f"CSV 디렉토리: {csv_dir}")
    logger.info(f"대상 CSV 파일: {csv_files}")

    con = duckdb.connect(db_path)

    for csv_file in tqdm(csv_files, desc="CSV 파일 로드"):
        table_name = csv_file.replace(".csv", "")
        csv_path = os.path.join(csv_dir, csv_file)

        if not os.path.exists(csv_path):
            logger.warning(f"  {csv_file}: 파일이 존재하지 않음, 건너뜀")
            continue

        table_exists = False
        try:
            result = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
            if result is not None:
                table_exists = True
                if not force_overwrite:
                    logger.info(f"  {table_name}: 이미 존재 (행 수: {result[0]:,}), 건너뜀")
                    continue
                else:
                    logger.info(f"  {table_name}: 이미 존재 (행 수: {result[0]:,}), 덮어쓰기")
                    con.execute(f"DROP TABLE {table_name}")
        except Exception:
            pass

        logger.info(f"  {csv_file} -> {table_name} 로드 중...")

        total_rows = 0
        chunk_count = 0
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
            if i == 0:
                con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM chunk")
            else:
                con.execute(f"INSERT INTO {table_name} SELECT * FROM chunk")
            total_rows += len(chunk)
            chunk_count = i + 1
            sys.stdout.write(f"\r    청크 {chunk_count:,} 완료 | 누적 행: {total_rows:,}")
            sys.stdout.flush()

        sys.stdout.write("\n")
        sys.stdout.flush()

        col_count = len(con.execute(f"DESCRIBE {table_name}").fetchall())
        logger.info(f"    완료: {total_rows:,} 행, {col_count} 열 (총 {chunk_count:,} 청크)")

    con.close()
    logger.info(f"=== CSV -> DuckDB 로드 완료 ===\n")


def export_to_parquet(
    db_path: str,
    output_dir: str,
    tables: list[str],
    compression: str = "zstd",
) -> None:
    """
    지정된 테이블들을 Parquet 포맷으로 내보냅니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        output_dir: Parquet 파일 출력 디렉토리
        tables: 내보낼 테이블 이름 리스트
        compression: 압축 방식 (기본값: zstd)
    """
    logger.info(f"=== Parquet 내보내기 시작 ===")
    logger.info(f"출력 디렉토리: {output_dir}")
    logger.info(f"대상 테이블: {tables}")
    logger.info(f"압축: {compression}")

    os.makedirs(output_dir, exist_ok=True)

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    for table in tqdm(tables, desc="Parquet 내보내기"):
        if table not in existing_tables:
            logger.warning(f"  {table}: 존재하지 않음, 건너뜀")
            continue

        output_file = os.path.join(output_dir, f"{table}.parquet")

        row_count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        cols_info = con.execute(f"DESCRIBE {table}").fetchall()
        col_count = len(cols_info)

        logger.info(f"  {table}: {row_count:,} 행, {col_count} 열")
        logger.info(f"    컬럼: {[col[0] for col in cols_info]}")

        con.execute(f"""
            COPY {table}
            TO '{output_file}'
            (FORMAT parquet, COMPRESSION {compression});
        """)

        file_size = os.path.getsize(output_file) / (1024 * 1024)
        logger.info(f"    -> {output_file} ({file_size:.2f} MB)")

    con.close()
    logger.info(f"=== Parquet 내보내기 완료 ===\n")


def export_to_csv(
    db_path: str,
    output_dir: str,
    tables: list[str],
    include_header: bool = True,
) -> None:
    """
    지정된 테이블들을 CSV 포맷으로 내보냅니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        output_dir: CSV 파일 출력 디렉토리
        tables: 내보낼 테이블 이름 리스트
        include_header: 헤더 포함 여부 (기본값: True)
    """
    logger.info(f"=== CSV 내보내기 시작 ===")
    logger.info(f"출력 디렉토리: {output_dir}")
    logger.info(f"대상 테이블: {tables}")

    os.makedirs(output_dir, exist_ok=True)

    con = duckdb.connect(db_path)

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    for table in tqdm(tables, desc="CSV 내보내기"):
        if table not in existing_tables:
            logger.warning(f"  {table}: 존재하지 않음, 건너뜀")
            continue

        output_file = os.path.join(output_dir, f"{table}.csv")

        row_count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        logger.info(f"  {table}: {row_count:,} 행")

        header_option = "true" if include_header else "false"
        con.execute(f"""
            COPY {table}
            TO '{output_file}'
            (FORMAT csv, HEADER {header_option});
        """)

        file_size = os.path.getsize(output_file) / (1024 * 1024)
        logger.info(f"    -> {output_file} ({file_size:.2f} MB)")

    con.close()
    logger.info(f"=== CSV 내보내기 완료 ===\n")


# ============================================================================
# 3. Data Transformation
# ============================================================================
def nullify_out_of_range(
    db_path: str,
    table_name: str,
    column_name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> None:
    """
    테이블의 특정 컬럼에서 지정된 범위를 벗어나는 값을 NULL로 변환합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 대상 테이블 이름
        column_name: 대상 컬럼 이름
        min_val: 최소값 (이 값 미만은 NULL로 변환, None이면 하한 없음)
        max_val: 최대값 (이 값 초과는 NULL로 변환, None이면 상한 없음)
    """
    logger.info(f"=== 범위 외 값 NULL 변환 시작 ===")
    logger.info(f"테이블: {table_name}, 컬럼: {column_name}")
    logger.info(f"유효 범위: [{min_val}, {max_val}]")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if table_name not in existing_tables:
        logger.error(f"{table_name} 테이블이 존재하지 않습니다.")
        con.close()
        return

    cols = [row[0] for row in con.execute(f"DESCRIBE {table_name}").fetchall()]
    if column_name not in cols:
        logger.error(f"{column_name} 컬럼이 존재하지 않습니다.")
        con.close()
        return

    stats_before = con.execute(f"""
        SELECT
            COUNT(*) as total,
            COUNT({column_name}) as non_null,
            SUM(CASE WHEN {column_name} IS NULL THEN 1 ELSE 0 END) as null_cnt,
            MIN({column_name}) as min_val,
            MAX({column_name}) as max_val,
            AVG({column_name}) as avg_val
        FROM {table_name}
    """).fetchdf()
    logger.info(f"변환 전 통계:")
    logger.info(f"  전체 행: {stats_before['total'].iloc[0]:,}")
    logger.info(f"  NULL 수: {stats_before['null_cnt'].iloc[0]:,}")
    logger.info(f"  최소값: {stats_before['min_val'].iloc[0]}")
    logger.info(f"  최대값: {stats_before['max_val'].iloc[0]}")
    logger.info(f"  평균: {stats_before['avg_val'].iloc[0]:.4f}")

    conditions = []
    if min_val is not None:
        conditions.append(f"{column_name} < {min_val}")
    if max_val is not None:
        conditions.append(f"{column_name} > {max_val}")

    if not conditions:
        logger.warning("min_val과 max_val이 모두 None입니다. 변환할 내용이 없습니다.")
        con.close()
        return

    condition_str = " OR ".join(conditions)
    out_of_range_count = con.execute(f"""
        SELECT COUNT(*) FROM {table_name}
        WHERE {condition_str}
    """).fetchone()[0]

    logger.info(f"범위 외 값 개수: {out_of_range_count:,}")

    if out_of_range_count == 0:
        logger.info("범위 외 값이 없습니다. 변환을 건너뜁니다.")
        con.close()
        logger.info(f"=== 범위 외 값 NULL 변환 완료 ===\n")
        return

    case_conditions = []
    if min_val is not None:
        case_conditions.append(f"WHEN {column_name} < {min_val} THEN NULL")
    if max_val is not None:
        case_conditions.append(f"WHEN {column_name} > {max_val} THEN NULL")
    case_str = " ".join(case_conditions)

    con.execute(f"""
        CREATE OR REPLACE TABLE {table_name}_temp AS
        SELECT
            * EXCLUDE ({column_name}),
            CASE
                {case_str}
                ELSE {column_name}
            END AS {column_name}
        FROM {table_name};
    """)

    con.execute(f"DROP TABLE {table_name};")
    con.execute(f"ALTER TABLE {table_name}_temp RENAME TO {table_name};")

    stats_after = con.execute(f"""
        SELECT
            COUNT(*) as total,
            COUNT({column_name}) as non_null,
            SUM(CASE WHEN {column_name} IS NULL THEN 1 ELSE 0 END) as null_cnt,
            MIN({column_name}) as min_val,
            MAX({column_name}) as max_val,
            AVG({column_name}) as avg_val
        FROM {table_name}
    """).fetchdf()
    logger.info(f"변환 후 통계:")
    logger.info(f"  전체 행: {stats_after['total'].iloc[0]:,}")
    logger.info(f"  NULL 수: {stats_after['null_cnt'].iloc[0]:,} (+{out_of_range_count:,})")
    logger.info(f"  최소값: {stats_after['min_val'].iloc[0]}")
    logger.info(f"  최대값: {stats_after['max_val'].iloc[0]}")
    logger.info(f"  평균: {stats_after['avg_val'].iloc[0]:.4f}")

    con.close()
    logger.info(f"=== 범위 외 값 NULL 변환 완료 ===\n")


def apply_conditional_transform(
    db_path: str,
    table_name: str,
    rules: list[dict],
    dry_run: bool = False,
) -> None:
    """
    특정 컬럼에 조건을 적용하여 값을 변환하거나 행을 삭제합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 대상 테이블명
        rules: 변환 규칙 리스트. 각 규칙은 다음 키를 포함하는 딕셔너리:
            - column: 대상 컬럼명
            - operator: 비교 연산자
                - "==", "!=", "<", ">", "<=", ">=" (단일 값 비교)
                - "in", "not_in" (리스트 포함 여부)
                - "is_null", "is_not_null" (NULL 체크)
                - "between", "not_between" (범위, value는 [min, max] 리스트)
            - value: 비교할 값 (operator에 따라 단일값, 리스트, 또는 생략)
            - action: 수행할 동작
                - "set_null": 해당 컬럼 값을 NULL로 변경
                - "set_value": 해당 컬럼 값을 new_value로 변경
                - "delete_row": 조건을 만족하는 행 삭제
            - new_value: action이 "set_value"일 때 설정할 값
        dry_run: True면 실제 변경 없이 영향받는 행 수만 출력 (기본값: False)

    예시:
        # 음수 total_secs를 NULL로, 86400 초과는 86400으로 클리핑
        apply_conditional_transform(
            db_path="data.duckdb",
            table_name="user_logs",
            rules=[
                {"column": "total_secs", "operator": "<", "value": 0, "action": "set_null"},
                {"column": "total_secs", "operator": ">", "value": 86400, "action": "set_value", "new_value": 86400},
            ]
        )
    """
    logger.info(f"=== 조건부 데이터 변환 시작 ===")
    logger.info(f"테이블: {table_name}")
    logger.info(f"규칙 수: {len(rules)}")
    if dry_run:
        logger.info("*** DRY RUN 모드 - 실제 변경 없음 ***")

    con = duckdb.connect(db_path, read_only=dry_run)
    if not dry_run:
        con.execute("PRAGMA threads=8;")

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if table_name not in existing_tables:
        logger.error(f"테이블 {table_name}이 존재하지 않습니다.")
        con.close()
        return

    cols = [row[0] for row in con.execute(f"DESCRIBE {table_name}").fetchall()]

    before_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    logger.info(f"변환 전 행 수: {before_count:,}")

    def build_condition(rule: dict) -> str:
        """규칙으로부터 SQL 조건절 생성"""
        column = rule["column"]
        operator = rule["operator"]
        value = rule.get("value")

        if operator == "==":
            if isinstance(value, str):
                return f"{column} = '{value}'"
            return f"{column} = {value}"
        elif operator == "!=":
            if isinstance(value, str):
                return f"{column} != '{value}'"
            return f"{column} != {value}"
        elif operator == "<":
            return f"{column} < {value}"
        elif operator == ">":
            return f"{column} > {value}"
        elif operator == "<=":
            return f"{column} <= {value}"
        elif operator == ">=":
            return f"{column} >= {value}"
        elif operator == "in":
            if isinstance(value[0], str):
                values_str = ", ".join([f"'{v}'" for v in value])
            else:
                values_str = ", ".join([str(v) for v in value])
            return f"{column} IN ({values_str})"
        elif operator == "not_in":
            if isinstance(value[0], str):
                values_str = ", ".join([f"'{v}'" for v in value])
            else:
                values_str = ", ".join([str(v) for v in value])
            return f"{column} NOT IN ({values_str})"
        elif operator == "is_null":
            return f"{column} IS NULL"
        elif operator == "is_not_null":
            return f"{column} IS NOT NULL"
        elif operator == "between":
            return f"{column} BETWEEN {value[0]} AND {value[1]}"
        elif operator == "not_between":
            return f"{column} NOT BETWEEN {value[0]} AND {value[1]}"
        else:
            raise ValueError(f"지원하지 않는 연산자: {operator}")

    for i, rule in enumerate(rules):
        column = rule["column"]
        action = rule["action"]

        if column not in cols:
            logger.warning(f"  규칙 {i+1}: 컬럼 '{column}'이 존재하지 않음, 건너뜀")
            continue

        condition = build_condition(rule)

        affected_count = con.execute(f"""
            SELECT COUNT(*) FROM {table_name} WHERE {condition}
        """).fetchone()[0]

        rule_desc = f"규칙 {i+1}: {column} {rule['operator']} {rule.get('value', '')} -> {action}"
        if action == "set_value":
            rule_desc += f" ({rule.get('new_value')})"
        logger.info(f"  {rule_desc}")
        logger.info(f"    영향받는 행: {affected_count:,} ({100*affected_count/before_count:.4f}%)")

        if dry_run or affected_count == 0:
            continue

        if action == "set_null":
            con.execute(f"""
                UPDATE {table_name}
                SET {column} = NULL
                WHERE {condition}
            """)
            logger.info(f"    -> {affected_count:,}개 행의 {column}을 NULL로 변경")

        elif action == "set_value":
            new_value = rule["new_value"]
            if isinstance(new_value, str):
                new_value_sql = f"'{new_value}'"
            else:
                new_value_sql = str(new_value)
            con.execute(f"""
                UPDATE {table_name}
                SET {column} = {new_value_sql}
                WHERE {condition}
            """)
            logger.info(f"    -> {affected_count:,}개 행의 {column}을 {new_value}로 변경")

        elif action == "delete_row":
            con.execute(f"""
                DELETE FROM {table_name}
                WHERE {condition}
            """)
            logger.info(f"    -> {affected_count:,}개 행 삭제")

        else:
            logger.warning(f"    알 수 없는 action: {action}, 건너뜀")

    after_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    deleted_count = before_count - after_count

    logger.info(f"\n--- 변환 결과 ---")
    logger.info(f"변환 전: {before_count:,}")
    logger.info(f"변환 후: {after_count:,}")
    if deleted_count > 0:
        logger.info(f"삭제된 행: {deleted_count:,} ({100*deleted_count/before_count:.4f}%)")

    con.close()
    logger.info(f"=== 조건부 데이터 변환 완료 ===\n")


def add_converted_column(
    db_path: str,
    table_name: str,
    source_col: str,
    target_col: str,
    divisor: float = 1.0,
    multiplier: float = 1.0,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    force_overwrite: bool = True,
) -> None:
    """
    기존 컬럼을 변환하여 새 컬럼을 추가합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 대상 테이블명
        source_col: 원본 컬럼명
        target_col: 생성할 컬럼명
        divisor: 나눗셈 값 (기본값: 1.0, 예: 3600으로 초->시간 변환)
        multiplier: 곱셈 값 (기본값: 1.0, divisor와 함께 사용: source * multiplier / divisor)
        clip_min: 클리핑 최솟값 (None이면 클리핑 안함)
        clip_max: 클리핑 최댓값 (None이면 클리핑 안함)
        force_overwrite: True면 기존 컬럼 덮어씀 (기본값: True)

    예시:
        # total_secs를 total_hours로 변환 (0~24시간 클리핑)
        add_converted_column(
            db_path="data.duckdb",
            table_name="user_logs",
            source_col="total_secs",
            target_col="total_hours",
            divisor=3600,
            clip_min=0,
            clip_max=24,
        )
    """
    logger.info(f"=== 컬럼 단위 변환 시작 ===")
    logger.info(f"테이블: {table_name}")
    logger.info(f"변환: {source_col} -> {target_col} (×{multiplier} ÷{divisor})")
    if clip_min is not None or clip_max is not None:
        logger.info(f"클리핑: [{clip_min}, {clip_max}]")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if table_name not in existing_tables:
        logger.error(f"테이블 {table_name}이 존재하지 않습니다.")
        con.close()
        return

    cols = [row[0] for row in con.execute(f"DESCRIBE {table_name}").fetchall()]
    if source_col not in cols:
        logger.error(f"원본 컬럼 {source_col}이 존재하지 않습니다.")
        con.close()
        return

    if target_col in cols:
        if not force_overwrite:
            logger.warning(f"컬럼 {target_col}이 이미 존재, 건너뜀")
            con.close()
            return
        else:
            con.execute(f"ALTER TABLE {table_name} DROP COLUMN {target_col}")
            logger.info(f"기존 {target_col} 컬럼 삭제")

    before_stats = con.execute(f"""
        SELECT
            COUNT(*) AS total,
            COUNT({source_col}) AS non_null,
            AVG({source_col}) AS avg_val,
            MIN({source_col}) AS min_val,
            MAX({source_col}) AS max_val
        FROM {table_name}
    """).fetchone()
    logger.info(f"변환 전 {source_col}: 평균={before_stats[2]:.2f}, 범위=[{before_stats[3]}, {before_stats[4]}]")

    # 변환 표현식 생성
    base_expr = f"{source_col} * {multiplier} / {divisor}"

    if clip_min is not None and clip_max is not None:
        transform_expr = f"""
            CASE
                WHEN {source_col} IS NULL THEN NULL
                WHEN {base_expr} < {clip_min} THEN {clip_min}
                WHEN {base_expr} > {clip_max} THEN {clip_max}
                ELSE {base_expr}
            END
        """
    elif clip_min is not None:
        transform_expr = f"""
            CASE
                WHEN {source_col} IS NULL THEN NULL
                WHEN {base_expr} < {clip_min} THEN {clip_min}
                ELSE {base_expr}
            END
        """
    elif clip_max is not None:
        transform_expr = f"""
            CASE
                WHEN {source_col} IS NULL THEN NULL
                WHEN {base_expr} > {clip_max} THEN {clip_max}
                ELSE {base_expr}
            END
        """
    else:
        transform_expr = base_expr

    con.execute(f"ALTER TABLE {table_name} ADD COLUMN {target_col} DOUBLE")

    logger.info("변환 중...")
    con.execute(f"""
        UPDATE {table_name}
        SET {target_col} = {transform_expr}
    """)

    after_stats = con.execute(f"""
        SELECT
            COUNT({target_col}) AS non_null,
            AVG({target_col}) AS avg_val,
            MIN({target_col}) AS min_val,
            MAX({target_col}) AS max_val,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {target_col}) AS median
        FROM {table_name}
    """).fetchone()
    logger.info(f"변환 후 {target_col}: 평균={after_stats[1]:.4f}, 중앙값={after_stats[4]:.4f}, 범위=[{after_stats[2]}, {after_stats[3]}]")

    if clip_min is not None or clip_max is not None:
        clip_stats = con.execute(f"""
            SELECT
                SUM(CASE WHEN {target_col} = {clip_min if clip_min is not None else 'NULL'} THEN 1 ELSE 0 END) AS clipped_min,
                SUM(CASE WHEN {target_col} = {clip_max if clip_max is not None else 'NULL'} THEN 1 ELSE 0 END) AS clipped_max
            FROM {table_name}
        """).fetchone()
        logger.info(f"클리핑 영향: 최솟값={clip_stats[0]:,}건, 최댓값={clip_stats[1]:,}건")

    sample = con.execute(f"""
        SELECT {source_col}, {target_col}
        FROM {table_name}
        WHERE {source_col} IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 5
    """).fetchdf()
    logger.info(f"샘플 데이터:\n{sample.to_string()}")

    con.close()
    logger.info(f"=== 컬럼 단위 변환 완료 ===\n")


def fill_null_values(
    db_path: str,
    table_name: str,
    column_name: str,
    fill_value: Optional[str] = None,
    fill_method: str = "value",
) -> None:
    """
    NULL 값을 채웁니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 대상 테이블명
        column_name: 대상 컬럼명
        fill_value: 채울 값 (fill_method="value"일 때 사용)
        fill_method: 채우기 방법
            - "value": fill_value로 채움
            - "mean": 평균값으로 채움
            - "median": 중앙값으로 채움
            - "mode": 최빈값으로 채움
    """
    logger.info(f"=== NULL 값 채우기 시작 ===")
    logger.info(f"테이블: {table_name}, 컬럼: {column_name}")
    logger.info(f"방법: {fill_method}")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    null_count = con.execute(f"""
        SELECT COUNT(*) FROM {table_name} WHERE {column_name} IS NULL
    """).fetchone()[0]

    if null_count == 0:
        logger.info("NULL 값이 없습니다.")
        con.close()
        return

    logger.info(f"NULL 개수: {null_count:,}")

    if fill_method == "value":
        if isinstance(fill_value, str) and not fill_value.replace(".", "").replace("-", "").isdigit():
            actual_value = f"'{fill_value}'"
        else:
            actual_value = fill_value
    elif fill_method == "mean":
        actual_value = con.execute(f"SELECT AVG({column_name}) FROM {table_name}").fetchone()[0]
        logger.info(f"평균값: {actual_value}")
    elif fill_method == "median":
        actual_value = con.execute(f"""
            SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {column_name})
            FROM {table_name}
        """).fetchone()[0]
        logger.info(f"중앙값: {actual_value}")
    elif fill_method == "mode":
        actual_value = con.execute(f"""
            SELECT {column_name}
            FROM {table_name}
            WHERE {column_name} IS NOT NULL
            GROUP BY {column_name}
            ORDER BY COUNT(*) DESC
            LIMIT 1
        """).fetchone()[0]
        logger.info(f"최빈값: {actual_value}")
    else:
        logger.error(f"지원하지 않는 fill_method: {fill_method}")
        con.close()
        return

    con.execute(f"""
        UPDATE {table_name}
        SET {column_name} = {actual_value}
        WHERE {column_name} IS NULL
    """)

    logger.info(f"{null_count:,}개 NULL 값을 {actual_value}로 채움")

    con.close()
    logger.info(f"=== NULL 값 채우기 완료 ===\n")


# ============================================================================
# 4. Data Analysis
# ============================================================================
def analyze_clipping_distribution(
    db_path: str,
    table_name: str,
    column_name: str,
    output_dir: str,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    sample_size: Optional[int] = None,
    bins: int = 50,
    figsize: tuple = (14, 10),
) -> None:
    """
    특정 feature의 클리핑 구간에 대한 통계 분포를 분석하고 그래프를 저장합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 분석할 테이블 이름
        column_name: 분석할 컬럼 이름
        output_dir: 그래프 이미지 저장 디렉토리
        clip_min: 클리핑 최솟값 (None이면 분석 안함)
        clip_max: 클리핑 최댓값 (None이면 분석 안함)
        sample_size: 샘플 크기 (None이면 전체 데이터)
        bins: 히스토그램 bin 개수 (기본값: 50)
        figsize: 그래프 크기 (기본값: (14, 10))

    출력:
        - 로그: 구간별 통계 (개수, 비율, 평균, 중앙값 등)
        - 그래프: 전체 분포, 구간별 분포, 박스플롯
    """
    import matplotlib.pyplot as plt
    import numpy as np

    logger.info(f"=== 클리핑 구간 분석 시작 ===")
    logger.info(f"테이블: {table_name}, 컬럼: {column_name}")
    logger.info(f"클리핑 범위: [{clip_min}, {clip_max}]")
    if sample_size:
        logger.info(f"샘플 크기: {sample_size:,}")

    con = duckdb.connect(db_path, read_only=True)

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if table_name not in existing_tables:
        logger.error(f"테이블 {table_name}이 존재하지 않습니다.")
        con.close()
        return

    cols = [row[0] for row in con.execute(f"DESCRIBE {table_name}").fetchall()]
    if column_name not in cols:
        logger.error(f"컬럼 {column_name}이 존재하지 않습니다.")
        con.close()
        return

    # 데이터 로드
    logger.info("데이터 로드 중...")
    if sample_size:
        query = f"""
            SELECT {column_name}
            FROM {table_name}
            USING SAMPLE {sample_size}
        """
    else:
        query = f"SELECT {column_name} FROM {table_name}"

    data = con.execute(query).fetchdf()[column_name]

    total_count = len(data)
    valid_data = data.dropna()
    valid_count = len(valid_data)
    null_count = total_count - valid_count

    logger.info(f"전체: {total_count:,}, 유효: {valid_count:,}, NULL: {null_count:,}")

    if valid_count == 0:
        logger.warning("유효한 데이터가 없습니다.")
        con.close()
        return

    # 기본 통계
    logger.info(f"\n--- 전체 기본 통계 ---")
    logger.info(f"  평균: {valid_data.mean():.4f}")
    logger.info(f"  중앙값: {valid_data.median():.4f}")
    logger.info(f"  표준편차: {valid_data.std():.4f}")
    logger.info(f"  범위: [{valid_data.min():.4f}, {valid_data.max():.4f}]")
    percentiles = np.percentile(valid_data, [1, 5, 25, 75, 95, 99])
    logger.info(f"  백분위수: P1={percentiles[0]:.4f}, P5={percentiles[1]:.4f}, P25={percentiles[2]:.4f}, "
                f"P75={percentiles[3]:.4f}, P95={percentiles[4]:.4f}, P99={percentiles[5]:.4f}")

    # 구간별 통계
    logger.info(f"\n--- 구간별 통계 ---")

    below_min = valid_data[valid_data < clip_min] if clip_min is not None else pd.Series([], dtype=float)
    above_max = valid_data[valid_data > clip_max] if clip_max is not None else pd.Series([], dtype=float)

    if clip_min is not None and clip_max is not None:
        in_range_data = valid_data[(valid_data >= clip_min) & (valid_data <= clip_max)]
    elif clip_min is not None:
        in_range_data = valid_data[valid_data >= clip_min]
    elif clip_max is not None:
        in_range_data = valid_data[valid_data <= clip_max]
    else:
        in_range_data = valid_data

    if len(below_min) > 0 and clip_min is not None:
        logger.info(f"  [clip_min 미만] ({column_name} < {clip_min}):")
        logger.info(f"    개수: {len(below_min):,} ({100*len(below_min)/valid_count:.4f}%)")
        logger.info(f"    평균: {below_min.mean():.4f}, 중앙값: {below_min.median():.4f}")
        logger.info(f"    범위: [{below_min.min():.4f}, {below_min.max():.4f}]")

    if len(above_max) > 0 and clip_max is not None:
        logger.info(f"  [clip_max 초과] ({column_name} > {clip_max}):")
        logger.info(f"    개수: {len(above_max):,} ({100*len(above_max)/valid_count:.4f}%)")
        logger.info(f"    평균: {above_max.mean():.4f}, 중앙값: {above_max.median():.4f}")
        logger.info(f"    범위: [{above_max.min():.4f}, {above_max.max():.4f}]")

    logger.info(f"  [범위 내] ({clip_min} <= {column_name} <= {clip_max}):")
    logger.info(f"    개수: {len(in_range_data):,} ({100*len(in_range_data)/valid_count:.4f}%)")
    logger.info(f"    평균: {in_range_data.mean():.4f}, 중앙값: {in_range_data.median():.4f}")
    logger.info(f"    표준편차: {in_range_data.std():.4f}")
    logger.info(f"    범위: [{in_range_data.min():.4f}, {in_range_data.max():.4f}]")

    # 그래프 생성
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. 전체 분포 히스토그램
    ax1 = axes[0, 0]
    ax1.hist(valid_data, bins=bins, edgecolor='black', alpha=0.7)
    if clip_min is not None:
        ax1.axvline(x=clip_min, color='r', linestyle='--', label=f'clip_min={clip_min}')
    if clip_max is not None:
        ax1.axvline(x=clip_max, color='r', linestyle='--', label=f'clip_max={clip_max}')
    ax1.set_title(f'Full Distribution: {column_name}', fontsize=12)
    ax1.set_xlabel(column_name)
    ax1.set_ylabel('Count')
    ax1.legend()

    # 2. 범위 내 데이터 히스토그램
    ax2 = axes[0, 1]
    ax2.hist(in_range_data, bins=bins, edgecolor='black', alpha=0.7, color='green')
    ax2.set_title(f'In-Range Distribution ({clip_min} <= x <= {clip_max})', fontsize=12)
    ax2.set_xlabel(column_name)
    ax2.set_ylabel('Count')

    # 3. 박스플롯
    ax3 = axes[1, 0]
    box_data = [valid_data]
    box_labels = ['All']
    if len(in_range_data) > 0:
        box_data.append(in_range_data)
        box_labels.append('In-Range')
    ax3.boxplot(box_data, labels=box_labels)
    ax3.set_title('Box Plot Comparison', fontsize=12)
    ax3.set_ylabel(column_name)

    # 4. 구간별 비율 파이차트
    ax4 = axes[1, 1]
    sizes = []
    labels = []
    colors = []
    if len(below_min) > 0:
        sizes.append(len(below_min))
        labels.append(f'Below {clip_min}\n({len(below_min):,})')
        colors.append('#ff6b6b')
    sizes.append(len(in_range_data))
    labels.append(f'In Range\n({len(in_range_data):,})')
    colors.append('#4ecdc4')
    if len(above_max) > 0:
        sizes.append(len(above_max))
        labels.append(f'Above {clip_max}\n({len(above_max):,})')
        colors.append('#ffe66d')

    ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%', startangle=90)
    ax4.set_title('Distribution by Range', fontsize=12)

    plt.suptitle(f'Clipping Analysis: {table_name}.{column_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 저장
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"clipping_analysis_{table_name}_{column_name}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"\n그래프 저장: {output_file}")

    # 추가 통계
    logger.info("\n--- 범위 내 데이터 백분위수 ---")
    if len(in_range_data) > 0:
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(in_range_data, p)
            logger.info(f"  P{p}: {val:.4f}")

    con.close()
    logger.info(f"=== 클리핑 구간 분석 완료 ===\n")


def analyze_feature_correlation(
    db_path: str,
    table_name: str,
    output_dir: str,
    exclude_cols: list[str] = None,
    sample_size: int = None,
    corr_threshold: float = 0.5,
    null_handling: str = "dropna",
) -> None:
    """
    테이블의 수치형 컬럼 간 상관관계를 분석하고 히트맵을 저장합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 분석할 테이블 이름
        output_dir: 히트맵 이미지 저장 디렉토리
        exclude_cols: 분석에서 제외할 컬럼 리스트
        sample_size: 샘플 크기 (None이면 전체 데이터)
        corr_threshold: 로그에 출력할 상관관계 임계값 (기본값: 0.5)
        null_handling: NULL 값 처리 방법 (기본값: "dropna")
            - "dropna": NULL이 있는 행 제외
            - "fillzero": NULL을 0으로 채움
            - "fillmean": NULL을 해당 컬럼 평균으로 채움
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    logger.info(f"=== Feature Correlation 분석 시작 ===")
    logger.info(f"대상 테이블: {table_name}")

    con = duckdb.connect(db_path, read_only=True)

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    if table_name not in existing_tables:
        logger.error(f"{table_name} 테이블이 존재하지 않습니다.")
        con.close()
        return

    cols_info = con.execute(f"DESCRIBE {table_name}").fetchall()
    all_cols = {row[0]: row[1] for row in cols_info}

    logger.info(f"전체 컬럼 수: {len(all_cols)}")

    if exclude_cols is None:
        exclude_cols = ["user_id", "id", "date", "timestamp"]

    numeric_types = ["INTEGER", "BIGINT", "DOUBLE", "FLOAT", "DECIMAL", "REAL", "SMALLINT", "TINYINT"]
    numeric_cols = []
    for col_name, col_type in all_cols.items():
        if col_name.lower() in [c.lower() for c in exclude_cols]:
            continue
        if any(nt in col_type.upper() for nt in numeric_types):
            numeric_cols.append(col_name)

    logger.info(f"수치형 컬럼 수: {len(numeric_cols)}")
    logger.info(f"분석 대상 컬럼: {numeric_cols}")

    if len(numeric_cols) < 2:
        logger.warning("상관관계 분석에 필요한 수치형 컬럼이 2개 미만입니다.")
        con.close()
        return

    # 데이터 로드
    cols_str = ", ".join(numeric_cols)
    if sample_size:
        query = f"SELECT {cols_str} FROM {table_name} USING SAMPLE {sample_size}"
        logger.info(f"샘플 크기: {sample_size:,}")
    else:
        query = f"SELECT {cols_str} FROM {table_name}"

    df = con.execute(query).fetchdf()

    # NULL 처리
    if null_handling == "dropna":
        df = df.dropna()
    elif null_handling == "fillzero":
        df = df.fillna(0)
    elif null_handling == "fillmean":
        df = df.fillna(df.mean())

    logger.info(f"분석 데이터 행 수: {len(df):,}")

    if len(df) < 10:
        logger.warning("분석할 데이터가 너무 적습니다.")
        con.close()
        return

    # 상관관계 계산
    corr_matrix = df.corr()

    # 높은 상관관계 출력
    logger.info(f"\n상관관계 |r| >= {corr_threshold}인 쌍:")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= corr_threshold:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                high_corr_pairs.append((col1, col2, corr_val))
                logger.info(f"  {col1} <-> {col2}: {corr_val:.4f}")

    if not high_corr_pairs:
        logger.info(f"  (임계값 이상인 쌍 없음)")

    # 상관관계 통계
    corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
    logger.info(f"\n상관관계 요약 통계:")
    logger.info(f"  평균: {np.mean(corr_values):.4f}")
    logger.info(f"  표준편차: {np.std(corr_values):.4f}")
    logger.info(f"  최소: {np.min(corr_values):.4f}")
    logger.info(f"  최대: {np.max(corr_values):.4f}")

    # 히트맵 생성
    fig_size = max(10, len(numeric_cols) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    sns.heatmap(
        corr_matrix,
        annot=True if len(numeric_cols) <= 15 else False,
        fmt='.2f' if len(numeric_cols) <= 15 else '',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation', 'shrink': 0.8},
        ax=ax
    )

    ax.set_title(f'Feature Correlation Matrix: {table_name}\n({len(df):,} rows, {len(numeric_cols)} features)',
                 fontsize=14, fontweight='bold', pad=20)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # 저장
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"correlation_{table_name}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"\n히트맵 저장: {output_file}")

    # CSV 저장
    csv_file = os.path.join(output_dir, f"correlation_{table_name}.csv")
    corr_matrix.to_csv(csv_file)
    logger.info(f"상관관계 행렬 CSV 저장: {csv_file}")

    con.close()
    logger.info(f"=== Feature Correlation 분석 완료 ===\n")


def get_column_stats(
    db_path: str,
    table_name: str,
    column_name: str,
) -> dict:
    """
    컬럼의 기본 통계를 반환합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 테이블명
        column_name: 컬럼명

    Returns:
        dict: 통계 정보 딕셔너리
    """
    con = duckdb.connect(db_path, read_only=True)

    stats = con.execute(f"""
        SELECT
            COUNT(*) as total,
            COUNT({column_name}) as non_null,
            SUM(CASE WHEN {column_name} IS NULL THEN 1 ELSE 0 END) as null_count,
            MIN({column_name}) as min_val,
            MAX({column_name}) as max_val,
            AVG({column_name}) as avg_val,
            STDDEV({column_name}) as std_val,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {column_name}) as median
        FROM {table_name}
    """).fetchone()

    con.close()

    return {
        "total": stats[0],
        "non_null": stats[1],
        "null_count": stats[2],
        "null_ratio": stats[2] / stats[0] if stats[0] > 0 else 0,
        "min": stats[3],
        "max": stats[4],
        "mean": stats[5],
        "std": stats[6],
        "median": stats[7],
    }
