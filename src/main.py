"""
KKBox 데이터 전처리 파이프라인

이 스크립트는 KKBox 데이터셋을 다음 단계로 전처리합니다:
1. CSV 파일들을 DuckDB 테이블로 로드
2. 테이블 이름 정규화 (_v1 appendix 추가)
3. v1, v2 테이블 병합 (_merge 테이블 생성)
4. msno를 정수 user_id로 매핑
5. 모든 _merge 테이블에 공통으로 존재하는 msno만 필터링 (교집합)
5-2. 기준 테이블 기반 msno 필터링
5-3. Churn 전이 기반 필터링 (v1=0 & v2=0/1인 유저만 남김)
5-4. 중복 트랜잭션 유저 제외 (같은 날 비취소 트랜잭션 2개 이상인 유저 제거)
5-5. Churn 전이행렬 분석 및 히트맵 저장
5-6. 중복 트랜잭션 유저 분석 및 결과 저장
5-7. Feature Correlation 분석 및 히트맵 저장
6. gender 필드 정수 변환
7. 날짜 필드 datetime 변환
8. msno -> user_id 컬럼명 변경
9. Parquet 포맷으로 내보내기
"""

import os
import sys
import logging
from typing import Optional

import duckdb
import pandas as pd
from tqdm import tqdm

# 범용 유틸리티 함수 import
from utils import (
    # Database/Table Operations
    drop_tables,
    copy_table,
    copy_tables,
    show_database_info,
    show_table_info,
    rename_column,
    add_column,
    get_table_columns,
    table_exists,
    get_row_count,
    # Data I/O
    load_csv_to_duckdb,
    export_to_parquet,
    export_to_csv,
    # Data Transformation
    nullify_out_of_range,
    apply_conditional_transform,
    add_converted_column,
    fill_null_values,
    # Data Analysis
    analyze_clipping_distribution,
    analyze_feature_correlation,
    get_column_stats,
)

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
# KKBox 도메인 특화 함수
# ============================================================================

# ============================================================================
# 1. 테이블 이름 정규화 (_v1 appendix 추가)
# ============================================================================
def rename_tables_add_v1_suffix(
    db_path: str,
    tables: list[str],
) -> None:
    """
    지정된 테이블들에 _v1 suffix를 추가합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        tables: _v1 suffix를 추가할 테이블 이름 리스트
    """
    logger.info(f"=== 테이블 이름 정규화 (_v1 추가) 시작 ===")
    logger.info(f"대상 테이블: {tables}")

    con = duckdb.connect(db_path)

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    renamed_count = 0
    for table in tqdm(tables, desc="테이블 이름 정규화"):
        if table not in existing_tables:
            logger.warning(f"  {table}: 존재하지 않음, 건너뜀")
            continue

        new_name = f"{table}_v1"

        if new_name in existing_tables:
            logger.warning(f"  {new_name}: 이미 존재, 건너뜀")
            continue

        con.execute(f"ALTER TABLE {table} RENAME TO {new_name}")
        logger.info(f"  {table} -> {new_name}")
        renamed_count += 1

    # 최종 테이블 목록
    final_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    logger.info(f"변경된 테이블 수: {renamed_count}")
    logger.info(f"최종 테이블 목록: {final_tables}")

    con.close()
    logger.info(f"=== 테이블 이름 정규화 완료 ===\n")


# ============================================================================
# 3. v1, v2 병합 테이블 생성
# ============================================================================
def create_merge_tables(
    db_path: str,
    base_names: list[str],
    drop_source_tables: bool = False,
    id_col: str = "msno",
) -> None:
    """
    v1, v2 테이블을 병합하여 _merge 테이블을 생성합니다.
    - 일반 테이블: UNION ALL (concat)
    - train: v2 우선 업데이트 방식
    - members: members_v3를 그대로 members_merge로 복사
    - v1/v2 중 하나만 있으면 그것을 _merge로 복사

    Args:
        db_path: DuckDB 데이터베이스 경로
        base_names: 병합할 기본 테이블 이름 리스트 (예: ["train", "transactions", "user_logs", "members"])
        drop_source_tables: True면 병합 후 원본 테이블(v1, v2, v3) 삭제 (기본값: True)
    """
    logger.info(f"=== _merge 테이블 생성 시작 ===")
    logger.info(f"병합 대상: {base_names}")
    logger.info(f"원본 테이블 삭제: {drop_source_tables}")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    # 모든 테이블 조회
    tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    logger.info(f"현재 테이블 목록: {tables}")

    for base in tqdm(base_names, desc="병합 테이블 생성"):
        v1_name = f"{base}_v1"
        v2_name = f"{base}_v2"
        v3_name = f"{base}_v3"
        merge_name = f"{base}_merge"

        v1_exists = v1_name in tables
        v2_exists = v2_name in tables
        v3_exists = v3_name in tables

        logger.info(f"  {base}: v1={v1_exists}, v2={v2_exists}, v3={v3_exists}")

        # members 특수 처리: members_v3 -> members_merge
        if base == "members":
            if v3_exists:
                con.execute(f"""
                    CREATE OR REPLACE TABLE {merge_name} AS
                    SELECT * FROM {v3_name};
                """)
                row_count = con.execute(f"SELECT COUNT(*) FROM {merge_name}").fetchone()[0]
                logger.info(f"    {merge_name}: {v3_name}에서 복사, {row_count:,} 행")
            else:
                logger.warning(f"    {base}: members_v3 없음, 건너뜀")
            continue

        # train 특수 처리: v2 우선 업데이트
        if base == "train":
            if v1_exists and v2_exists:
                con.execute(f"""
                    CREATE OR REPLACE TABLE {merge_name} AS
                    SELECT *
                    EXCLUDE (rn, priority)
                    FROM (
                        SELECT
                            *,
                            row_number() OVER (PARTITION BY {id_col} ORDER BY priority) AS rn
                        FROM (
                            SELECT t2.*, 0 AS priority
                            FROM {v2_name} t2
                            UNION ALL
                            SELECT t1.*, 1 AS priority
                            FROM {v1_name} t1
                        ) u
                    ) ranked
                    WHERE rn = 1;
                """)
                row_count = con.execute(f"SELECT COUNT(*) FROM {merge_name}").fetchone()[0]
                distinct_msno = con.execute(f"SELECT COUNT(DISTINCT {id_col}) FROM {merge_name}").fetchone()[0]
                logger.info(f"    {merge_name}: v2 우선 병합, {row_count:,} 행, {distinct_msno:,} 고유 {id_col}")
            elif v1_exists:
                con.execute(f"CREATE OR REPLACE TABLE {merge_name} AS SELECT * FROM {v1_name};")
                row_count = con.execute(f"SELECT COUNT(*) FROM {merge_name}").fetchone()[0]
                logger.info(f"    {merge_name}: {v1_name}에서 복사, {row_count:,} 행")
            elif v2_exists:
                con.execute(f"CREATE OR REPLACE TABLE {merge_name} AS SELECT * FROM {v2_name};")
                row_count = con.execute(f"SELECT COUNT(*) FROM {merge_name}").fetchone()[0]
                logger.info(f"    {merge_name}: {v2_name}에서 복사, {row_count:,} 행")
            else:
                logger.warning(f"    {base}: v1, v2 모두 없음, 건너뜀")
            continue

        # 일반 테이블: UNION ALL (concat)
        if v1_exists and v2_exists:
            con.execute(f"""
                CREATE OR REPLACE TABLE {merge_name} AS
                SELECT * FROM {v1_name}
                UNION ALL
                SELECT * FROM {v2_name};
            """)
            row_count = con.execute(f"SELECT COUNT(*) FROM {merge_name}").fetchone()[0]
            logger.info(f"    {merge_name}: v1 + v2 concat, {row_count:,} 행")
        elif v1_exists:
            con.execute(f"CREATE OR REPLACE TABLE {merge_name} AS SELECT * FROM {v1_name};")
            row_count = con.execute(f"SELECT COUNT(*) FROM {merge_name}").fetchone()[0]
            logger.info(f"    {merge_name}: {v1_name}에서 복사, {row_count:,} 행")
        elif v2_exists:
            con.execute(f"CREATE OR REPLACE TABLE {merge_name} AS SELECT * FROM {v2_name};")
            row_count = con.execute(f"SELECT COUNT(*) FROM {merge_name}").fetchone()[0]
            logger.info(f"    {merge_name}: {v2_name}에서 복사, {row_count:,} 행")
        else:
            logger.warning(f"    {base}: v1, v2 모두 없음, 건너뜀")

    # 원본 테이블 삭제
    if drop_source_tables:
        logger.info("원본 테이블 삭제 중...")
        tables_to_drop = []
        for base in base_names:
            for suffix in ["_v1", "_v2", "_v3"]:
                table_name = f"{base}{suffix}"
                if table_name in tables:
                    tables_to_drop.append(table_name)

        for table in tables_to_drop:
            con.execute(f"DROP TABLE IF EXISTS {table}")
            logger.info(f"    {table}: 삭제됨")

        logger.info(f"  총 {len(tables_to_drop)}개 원본 테이블 삭제됨")

    con.close()
    logger.info(f"=== _merge 테이블 생성 완료 ===\n")


# ============================================================================
# 4. msno -> user_id 매핑 테이블 생성 및 변환
# ============================================================================
def create_user_id_mapping(
    db_path: str,
    target_tables: list[str],
    mapping_table_name: str = "user_id_map",
) -> None:
    """
    지정된 테이블들에서 유니크 msno를 수집하여 정수 user_id로 매핑하고,
    각 테이블의 msno를 변환합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        target_tables: 매핑을 적용할 테이블 이름 리스트
        mapping_table_name: 매핑 테이블 이름
    """
    logger.info(f"=== user_id 매핑 시작 ===")
    logger.info(f"대상 테이블: {target_tables}")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    # msno 컬럼이 있는 테이블만 필터링
    tables_with_msno = []
    for t in target_tables:
        if t not in existing_tables:
            logger.warning(f"  {t}: 존재하지 않음, 건너뜀")
            continue
        cols = [row[0] for row in con.execute(f"DESCRIBE {t}").fetchall()]
        if "msno" in cols:
            tables_with_msno.append(t)
        else:
            logger.warning(f"  {t}: msno 컬럼 없음, 건너뜀")

    logger.info(f"msno 컬럼이 있는 테이블: {tables_with_msno}")

    if not tables_with_msno:
        logger.warning("msno 컬럼이 있는 테이블이 없습니다.")
        con.close()
        return

    # 모든 테이블의 유니크 msno 합집합
    union_parts = [f"SELECT DISTINCT msno FROM {t}" for t in tables_with_msno]
    union_query = " UNION ".join(union_parts)

    con.execute(f"""
        CREATE OR REPLACE TABLE {mapping_table_name} AS
        SELECT
            msno,
            (row_number() OVER (ORDER BY msno)) - 1 AS user_id
        FROM ({union_query}) t
        WHERE msno IS NOT NULL AND msno <> '';
    """)

    # 매핑 테이블 정보
    total_users = con.execute(f"SELECT COUNT(*) FROM {mapping_table_name}").fetchone()[0]
    min_id = con.execute(f"SELECT MIN(user_id) FROM {mapping_table_name}").fetchone()[0]
    max_id = con.execute(f"SELECT MAX(user_id) FROM {mapping_table_name}").fetchone()[0]

    logger.info(f"매핑 테이블 생성: {mapping_table_name}")
    logger.info(f"  총 사용자 수: {total_users:,}")
    logger.info(f"  user_id 범위: {min_id} ~ {max_id}")

    # 중복 확인
    dup_count = con.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT msno FROM {mapping_table_name} GROUP BY msno HAVING COUNT(*) > 1
        )
    """).fetchone()[0]
    if dup_count > 0:
        logger.warning(f"  중복 msno 발견: {dup_count}개")
    else:
        logger.info(f"  중복 msno: 없음")

    # 각 테이블의 msno를 user_id로 변환
    for table in tqdm(tables_with_msno, desc="msno -> user_id 변환"):
        logger.info(f"  {table} 변환 중...")

        original_count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        con.execute(f"""
            CREATE OR REPLACE TABLE {table}_temp AS
            SELECT
                m.user_id AS msno,
                t.* EXCLUDE (msno)
            FROM {table} t
            JOIN {mapping_table_name} m USING (msno);
        """)

        new_count = con.execute(f"SELECT COUNT(*) FROM {table}_temp").fetchone()[0]

        # 원본 테이블 교체
        con.execute(f"DROP TABLE {table};")
        con.execute(f"ALTER TABLE {table}_temp RENAME TO {table};")

        # 매칭되지 않은 행 확인
        lost_rows = original_count - new_count
        if lost_rows > 0:
            logger.warning(f"    매칭되지 않은 행: {lost_rows:,}")
        else:
            logger.info(f"    완료: {new_count:,} 행 (손실 없음)")

    con.close()
    logger.info(f"=== user_id 매핑 완료 ===\n")


# ============================================================================
# 5. 공통 msno 교집합 필터링
# ============================================================================
def filter_common_msno(
    db_path: str,
    target_tables: list[str],
    id_col: str = "msno",
) -> None:
    """
    지정된 테이블들에 공통으로 존재하는 msno(user_id)만 남깁니다.
    각 테이블에서 교집합에 해당하지 않는 행은 제거됩니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        target_tables: 교집합 필터링을 적용할 테이블 이름 리스트
    """
    logger.info(f"=== 공통 {id_col} 교집합 필터링 시작 ===")
    logger.info(f"대상 테이블: {target_tables}")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    # msno/user_id 컬럼이 있는 테이블만 필터링
    tables_with_msno = []
    msno_col_name = {}  # 테이블별 msno 컬럼명 (msno 또는 user_id)
    for t in target_tables:
        if t not in existing_tables:
            logger.warning(f"  {t}: 존재하지 않음, 건너뜀")
            continue
        cols = [row[0] for row in con.execute(f"DESCRIBE {t}").fetchall()]
        if id_col in cols:
            tables_with_msno.append(t)
            msno_col_name[t] = id_col
        else:
            logger.warning(f"  {t}: {id_col} 컬럼 없음, 건너뜀")

    logger.info(f"실제 대상 테이블: {tables_with_msno}")

    if len(tables_with_msno) < 2:
        logger.warning("교집합을 구할 테이블이 2개 미만입니다. 건너뜀.")
        con.close()
        return

    # 각 테이블의 유니크 {id_col} 수 확인 (필터링 전)
    logger.info("필터링 전 각 테이블 통계:")
    before_stats = {}
    for t in tables_with_msno:
        col = msno_col_name[t]
        row_count = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        unique_msno = con.execute(f"SELECT COUNT(DISTINCT {col}) FROM {t}").fetchone()[0]
        before_stats[t] = {"rows": row_count, "unique_msno": unique_msno}
        logger.info(f"  {t}: {row_count:,} 행, {unique_msno:,} 고유 {id_col}")

    # 교집합 구하기: 모든 테이블에 공통으로 존재하는 msno
    intersect_parts = []
    for t in tables_with_msno:
        col = msno_col_name[t]
        intersect_parts.append(f"SELECT DISTINCT {col} AS {id_col} FROM {t}")

    intersect_query = " INTERSECT ".join(intersect_parts)

    con.execute(f"""
        CREATE OR REPLACE TABLE _common_msno_temp AS
        {intersect_query};
    """)

    common_count = con.execute("SELECT COUNT(*) FROM _common_msno_temp").fetchone()[0]
    logger.info(f"공통 {id_col} (교집합) 수: {common_count:,}")

    if common_count == 0:
        logger.error(f"공통 {id_col}가 없습니다! 필터링을 중단합니다.")
        con.execute("DROP TABLE IF EXISTS _common_msno_temp;")
        con.close()
        return

    # 각 테이블에서 교집합에 해당하는 행만 남기기
    for t in tqdm(tables_with_msno, desc="교집합 필터링"):
        col = msno_col_name[t]
        logger.info(f"  {t} 필터링 중...")

        con.execute(f"""
            CREATE OR REPLACE TABLE {t}_filtered AS
            SELECT t.*
            FROM {t} t
            WHERE t.{col} IN (SELECT {id_col} FROM _common_msno_temp);
        """)

        # 원본 테이블 교체
        con.execute(f"DROP TABLE {t};")
        con.execute(f"ALTER TABLE {t}_filtered RENAME TO {t};")

    # 임시 테이블 삭제
    con.execute("DROP TABLE IF EXISTS _common_msno_temp;")

    # 필터링 후 통계
    logger.info("필터링 후 각 테이블 통계:")
    for t in tables_with_msno:
        col = msno_col_name[t]
        row_count = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        unique_msno = con.execute(f"SELECT COUNT(DISTINCT {col}) FROM {t}").fetchone()[0]

        rows_removed = before_stats[t]["rows"] - row_count
        msno_removed = before_stats[t]["unique_msno"] - unique_msno

        logger.info(f"  {t}: {row_count:,} 행 ({rows_removed:,} 제거), "
                    f"{unique_msno:,} 고유 {id_col} ({msno_removed:,} 제거)")

    con.close()
    logger.info(f"=== 공통 {id_col} 교집합 필터링 완료 ===\n")


# ============================================================================
# 5-2. 기준 테이블 기반 msno 필터링
# ============================================================================
def filter_by_reference_table(
    db_path: str,
    reference_table: str,
    target_tables: list[str],
) -> None:
    """
    기준 테이블에 존재하는 msno(user_id)만 남기도록 대상 테이블들을 필터링합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        reference_table: 기준이 되는 테이블 이름 (이 테이블의 msno를 기준으로 필터링)
        target_tables: 필터링을 적용할 대상 테이블 이름 리스트
    """
    logger.info(f"=== 기준 테이블 기반 msno 필터링 시작 ===")
    logger.info(f"기준 테이블: {reference_table}")
    logger.info(f"대상 테이블: {target_tables}")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    # 기준 테이블 존재 확인
    if reference_table not in existing_tables:
        logger.error(f"기준 테이블 {reference_table}이 존재하지 않습니다.")
        con.close()
        return

    # 기준 테이블의 msno/user_id 컬럼 확인
    ref_cols = [row[0] for row in con.execute(f"DESCRIBE {reference_table}").fetchall()]
    if "msno" in ref_cols:
        ref_col = "msno"
    elif "user_id" in ref_cols:
        ref_col = "user_id"
    else:
        logger.error(f"기준 테이블 {reference_table}에 msno/user_id 컬럼이 없습니다.")
        con.close()
        return

    # 기준 테이블의 유니크 msno 수
    ref_unique_count = con.execute(f"SELECT COUNT(DISTINCT {ref_col}) FROM {reference_table}").fetchone()[0]
    logger.info(f"기준 테이블 유니크 msno 수: {ref_unique_count:,}")

    # 대상 테이블 필터링
    for t in tqdm(target_tables, desc="기준 테이블 기반 필터링"):
        if t not in existing_tables:
            logger.warning(f"  {t}: 존재하지 않음, 건너뜀")
            continue

        if t == reference_table:
            logger.info(f"  {t}: 기준 테이블과 동일, 건너뜀")
            continue

        # 대상 테이블의 msno/user_id 컬럼 확인
        target_cols = [row[0] for row in con.execute(f"DESCRIBE {t}").fetchall()]
        if "msno" in target_cols:
            target_col = "msno"
        elif "user_id" in target_cols:
            target_col = "user_id"
        else:
            logger.warning(f"  {t}: msno/user_id 컬럼 없음, 건너뜀")
            continue

        logger.info(f"  {t} 필터링 중...")

        # 필터링 전 통계
        before_rows = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        before_unique = con.execute(f"SELECT COUNT(DISTINCT {target_col}) FROM {t}").fetchone()[0]

        # 기준 테이블의 msno만 남기기
        con.execute(f"""
            CREATE OR REPLACE TABLE {t}_filtered AS
            SELECT t.*
            FROM {t} t
            WHERE t.{target_col} IN (SELECT DISTINCT {ref_col} FROM {reference_table});
        """)

        # 원본 테이블 교체
        con.execute(f"DROP TABLE {t};")
        con.execute(f"ALTER TABLE {t}_filtered RENAME TO {t};")

        # 필터링 후 통계
        after_rows = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        after_unique = con.execute(f"SELECT COUNT(DISTINCT {target_col}) FROM {t}").fetchone()[0]

        rows_removed = before_rows - after_rows
        msno_removed = before_unique - after_unique

        logger.info(f"    {before_rows:,} -> {after_rows:,} 행 ({rows_removed:,} 제거)")
        logger.info(f"    {before_unique:,} -> {after_unique:,} 고유 msno ({msno_removed:,} 제거)")

    con.close()
    logger.info(f"=== 기준 테이블 기반 msno 필터링 완료 ===\n")


# ============================================================================
# 5-3. Churn 전이 기반 필터링 (v1=0 & v2=0/1인 유저만 남김)
# ============================================================================
def filter_by_churn_transition(
    db_path: str,
    train_v1_table: str,
    train_v2_table: str,
    target_tables: list[str],
    v1_churn_value: int = 0,
    v2_churn_values: list[int] = [0, 1],
) -> None:
    """
    train_v1과 train_v2의 is_churn 전이 조건을 만족하는 유저만 남깁니다.
    기본값: train_v1에서 is_churn=0이고, train_v2에서 is_churn이 0 또는 1인 유저

    Args:
        db_path: DuckDB 데이터베이스 경로
        train_v1_table: train v1 테이블 이름
        train_v2_table: train v2 테이블 이름
        target_tables: 필터링을 적용할 대상 테이블 이름 리스트 (_merge 테이블들)
        v1_churn_value: train_v1에서 필터링할 is_churn 값 (기본값: 0)
        v2_churn_values: train_v2에서 허용할 is_churn 값 리스트 (기본값: [0, 1])
    """
    logger.info(f"=== Churn 전이 기반 필터링 시작 ===")
    logger.info(f"train_v1 테이블: {train_v1_table}")
    logger.info(f"train_v2 테이블: {train_v2_table}")
    logger.info(f"조건: v1.is_churn={v1_churn_value} AND v2.is_churn IN {v2_churn_values}")
    logger.info(f"대상 테이블: {target_tables}")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    # train 테이블 존재 확인
    if train_v1_table not in existing_tables:
        logger.error(f"{train_v1_table} 테이블이 존재하지 않습니다.")
        con.close()
        return

    if train_v2_table not in existing_tables:
        logger.error(f"{train_v2_table} 테이블이 존재하지 않습니다.")
        con.close()
        return

    # train_v1의 msno/user_id 컬럼 확인
    v1_cols = [row[0] for row in con.execute(f"DESCRIBE {train_v1_table}").fetchall()]
    if "msno" in v1_cols:
        id_col = "msno"
    elif "user_id" in v1_cols:
        id_col = "user_id"
    else:
        logger.error(f"{train_v1_table}에 msno/user_id 컬럼이 없습니다.")
        con.close()
        return

    # train_v1, train_v2 통계
    v1_total = con.execute(f"SELECT COUNT(*) FROM {train_v1_table}").fetchone()[0]
    v2_total = con.execute(f"SELECT COUNT(*) FROM {train_v2_table}").fetchone()[0]
    logger.info(f"{train_v1_table} 총 행 수: {v1_total:,}")
    logger.info(f"{train_v2_table} 총 행 수: {v2_total:,}")

    # 조건에 맞는 msno 추출
    v2_values_str = ", ".join([str(v) for v in v2_churn_values])

    con.execute(f"""
        CREATE OR REPLACE TABLE _churn_transition_msno_temp AS
        SELECT v1.{id_col}
        FROM {train_v1_table} v1
        INNER JOIN {train_v2_table} v2 ON v1.{id_col} = v2.{id_col}
        WHERE v1.is_churn = {v1_churn_value}
          AND v2.is_churn IN ({v2_values_str});
    """)

    filtered_count = con.execute("SELECT COUNT(*) FROM _churn_transition_msno_temp").fetchone()[0]
    logger.info(f"조건 만족 유저 수: {filtered_count:,}")

    # is_churn 전이 분포 확인
    logger.info("선택된 유저의 v2 is_churn 분포:")
    dist = con.execute(f"""
        SELECT v2.is_churn, COUNT(*) as cnt
        FROM _churn_transition_msno_temp t
        JOIN {train_v2_table} v2 ON t.{id_col} = v2.{id_col}
        GROUP BY v2.is_churn
        ORDER BY v2.is_churn
    """).fetchall()
    for row in dist:
        logger.info(f"  is_churn={row[0]}: {row[1]:,}")

    if filtered_count == 0:
        logger.error("조건을 만족하는 유저가 없습니다! 필터링을 중단합니다.")
        con.execute("DROP TABLE IF EXISTS _churn_transition_msno_temp;")
        con.close()
        return

    # 대상 테이블 필터링
    for t in tqdm(target_tables, desc="Churn 전이 기반 필터링"):
        if t not in existing_tables:
            logger.warning(f"  {t}: 존재하지 않음, 건너뜀")
            continue

        # 대상 테이블의 msno/user_id 컬럼 확인
        target_cols = [row[0] for row in con.execute(f"DESCRIBE {t}").fetchall()]
        if "msno" in target_cols:
            target_col = "msno"
        elif "user_id" in target_cols:
            target_col = "user_id"
        else:
            logger.warning(f"  {t}: msno/user_id 컬럼 없음, 건너뜀")
            continue

        logger.info(f"  {t} 필터링 중...")

        # 필터링 전 통계
        before_rows = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        before_unique = con.execute(f"SELECT COUNT(DISTINCT {target_col}) FROM {t}").fetchone()[0]

        # 조건에 맞는 msno만 남기기
        con.execute(f"""
            CREATE OR REPLACE TABLE {t}_filtered AS
            SELECT t.*
            FROM {t} t
            WHERE t.{target_col} IN (SELECT {id_col} FROM _churn_transition_msno_temp);
        """)

        # 원본 테이블 교체
        con.execute(f"DROP TABLE {t};")
        con.execute(f"ALTER TABLE {t}_filtered RENAME TO {t};")

        # 필터링 후 통계
        after_rows = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        after_unique = con.execute(f"SELECT COUNT(DISTINCT {target_col}) FROM {t}").fetchone()[0]

        rows_removed = before_rows - after_rows
        msno_removed = before_unique - after_unique

        logger.info(f"    {before_rows:,} -> {after_rows:,} 행 ({rows_removed:,} 제거)")
        logger.info(f"    {before_unique:,} -> {after_unique:,} 고유 유저 ({msno_removed:,} 제거)")

    # 임시 테이블 삭제
    con.execute("DROP TABLE IF EXISTS _churn_transition_msno_temp;")

    con.close()
    logger.info(f"=== Churn 전이 기반 필터링 완료 ===\n")


# ============================================================================
# 5-4. 중복 트랜잭션 유저 제외 (같은 날 비취소 트랜잭션 2개 이상인 유저 제거)
# ============================================================================
def exclude_duplicate_transaction_users(
    db_path: str,
    transactions_table: str,
    target_tables: list[str],
    date_col: str = "transaction_date",
    min_txn_count: int = 2,
) -> None:
    """
    같은 날 비취소(is_cancel=0) 트랜잭션이 지정된 개수 이상인 유저를 제외합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        transactions_table: 트랜잭션 테이블 이름
        target_tables: 필터링을 적용할 대상 테이블 이름 리스트
        date_col: 날짜 컬럼명 (기본값: transaction_date)
        min_txn_count: 제외 기준이 되는 최소 트랜잭션 수 (기본값: 2)
    """
    logger.info(f"=== 중복 트랜잭션 유저 제외 시작 ===")
    logger.info(f"트랜잭션 테이블: {transactions_table}")
    logger.info(f"제외 조건: 같은 날 비취소 트랜잭션 {min_txn_count}개 이상")
    logger.info(f"대상 테이블: {target_tables}")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    # 트랜잭션 테이블 존재 확인
    if transactions_table not in existing_tables:
        logger.error(f"{transactions_table} 테이블이 존재하지 않습니다.")
        con.close()
        return

    # 트랜잭션 테이블의 user_id 컬럼 확인
    tx_cols = [row[0] for row in con.execute(f"DESCRIBE {transactions_table}").fetchall()]
    if "user_id" in tx_cols:
        id_col = "user_id"
    elif "msno" in tx_cols:
        id_col = "msno"
    else:
        logger.error(f"{transactions_table}에 user_id/msno 컬럼이 없습니다.")
        con.close()
        return

    # 전체 유저 수
    total_users = con.execute(f"""
        SELECT COUNT(DISTINCT {id_col}) FROM {transactions_table}
    """).fetchone()[0]
    logger.info(f"트랜잭션 테이블 전체 유저 수: {total_users:,}")

    # 중복 트랜잭션이 있는 유저 추출
    con.execute(f"""
        CREATE OR REPLACE TABLE _duplicate_txn_users_temp AS
        WITH non_cancel_txns AS (
            SELECT * FROM {transactions_table} WHERE is_cancel = 0
        ),
        user_date_counts AS (
            SELECT {id_col}, {date_col}
            FROM non_cancel_txns
            GROUP BY {id_col}, {date_col}
            HAVING COUNT(*) >= {min_txn_count}
        )
        SELECT DISTINCT {id_col} FROM user_date_counts;
    """)

    dup_user_count = con.execute("SELECT COUNT(*) FROM _duplicate_txn_users_temp").fetchone()[0]
    logger.info(f"중복 트랜잭션 유저 수: {dup_user_count:,} ({dup_user_count/total_users*100:.2f}%)")

    if dup_user_count == 0:
        logger.info("제외할 유저가 없습니다.")
        con.execute("DROP TABLE IF EXISTS _duplicate_txn_users_temp;")
        con.close()
        return

    # 대상 테이블에서 중복 유저 제외
    for t in tqdm(target_tables, desc="중복 트랜잭션 유저 제외"):
        if t not in existing_tables:
            logger.warning(f"  {t}: 존재하지 않음, 건너뜀")
            continue

        # 대상 테이블의 user_id 컬럼 확인
        target_cols = [row[0] for row in con.execute(f"DESCRIBE {t}").fetchall()]
        if "user_id" in target_cols:
            target_col = "user_id"
        elif "msno" in target_cols:
            target_col = "msno"
        else:
            logger.warning(f"  {t}: user_id/msno 컬럼 없음, 건너뜀")
            continue

        logger.info(f"  {t} 필터링 중...")

        # 필터링 전 통계
        before_rows = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        before_unique = con.execute(f"SELECT COUNT(DISTINCT {target_col}) FROM {t}").fetchone()[0]

        # 중복 유저 제외 (NOT IN)
        con.execute(f"""
            CREATE OR REPLACE TABLE {t}_filtered AS
            SELECT *
            FROM {t}
            WHERE {target_col} NOT IN (SELECT {id_col} FROM _duplicate_txn_users_temp);
        """)

        # 원본 테이블 교체
        con.execute(f"DROP TABLE {t};")
        con.execute(f"ALTER TABLE {t}_filtered RENAME TO {t};")

        # 필터링 후 통계
        after_rows = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        after_unique = con.execute(f"SELECT COUNT(DISTINCT {target_col}) FROM {t}").fetchone()[0]

        rows_removed = before_rows - after_rows
        users_removed = before_unique - after_unique

        logger.info(f"    {before_rows:,} -> {after_rows:,} 행 ({rows_removed:,} 제거)")
        logger.info(f"    {before_unique:,} -> {after_unique:,} 고유 유저 ({users_removed:,} 제거)")

    # 임시 테이블 삭제
    con.execute("DROP TABLE IF EXISTS _duplicate_txn_users_temp;")

    con.close()
    logger.info(f"=== 중복 트랜잭션 유저 제외 완료 ===\n")


# ============================================================================
# 5-5. Churn 전이행렬 분석 및 히트맵 저장
# ============================================================================
def analyze_churn_transition(
    db_path: str,
    train_v1_table: str,
    train_v2_table: str,
    output_dir: str,
    reference_table: str = None,
) -> None:
    """
    train_v1과 train_v2의 is_churn 전이행렬을 분석하고 히트맵을 저장합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        train_v1_table: train v1 테이블 이름
        train_v2_table: train v2 테이블 이름
        output_dir: 히트맵 이미지 저장 디렉토리
        reference_table: 기준 테이블 (지정 시 해당 테이블에 있는 유저만 분석)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    logger.info(f"=== Churn 전이행렬 분석 시작 ===")
    logger.info(f"train_v1: {train_v1_table}")
    logger.info(f"train_v2: {train_v2_table}")
    if reference_table:
        logger.info(f"기준 테이블: {reference_table}")

    con = duckdb.connect(db_path, read_only=True)

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    # 테이블 존재 확인
    if train_v1_table not in existing_tables:
        logger.error(f"{train_v1_table} 테이블이 존재하지 않습니다.")
        con.close()
        return

    if train_v2_table not in existing_tables:
        logger.error(f"{train_v2_table} 테이블이 존재하지 않습니다.")
        con.close()
        return

    # user_id 컬럼 확인
    v1_cols = [row[0] for row in con.execute(f"DESCRIBE {train_v1_table}").fetchall()]
    id_col = "user_id" if "user_id" in v1_cols else "msno"

    # 기준 테이블 조건 (COALESCE로 v1=NULL인 경우도 포함)
    ref_condition = ""
    if reference_table and reference_table in existing_tables:
        ref_cols = [row[0] for row in con.execute(f"DESCRIBE {reference_table}").fetchall()]
        ref_id_col = "user_id" if "user_id" in ref_cols else "msno"
        ref_condition = f"AND COALESCE(v1.{id_col}, v2.{id_col}) IN (SELECT {ref_id_col} FROM {reference_table})"

    # 전이행렬 쿼리 (3x3: v1=0/1/NULL -> v2=0/1/NULL)
    transition_df = con.execute(f"""
        SELECT
            CASE
                WHEN v1.is_churn IS NULL THEN 'NULL'
                ELSE CAST(v1.is_churn AS VARCHAR)
            END as v1_churn,
            CASE
                WHEN v2.is_churn IS NULL THEN 'NULL'
                ELSE CAST(v2.is_churn AS VARCHAR)
            END as v2_churn,
            COUNT(*) as cnt
        FROM {train_v1_table} v1
        FULL OUTER JOIN {train_v2_table} v2 ON v1.{id_col} = v2.{id_col}
        WHERE 1=1 {ref_condition}
        GROUP BY 1, 2
        ORDER BY 1, 2
    """).fetchdf()

    logger.info(f"전이행렬 데이터 (3x3):")
    for _, row in transition_df.iterrows():
        logger.info(f"  v1={row['v1_churn']} -> v2={row['v2_churn']}: {row['cnt']:,}")

    # Pivot table 생성 (3x3 고정)
    pivot_df = transition_df.pivot(index='v1_churn', columns='v2_churn', values='cnt').fillna(0).astype(int)
    order = ['0', '1', 'NULL']
    # 모든 행/열이 존재하도록 reindex (없으면 0으로 채움)
    pivot_df = pivot_df.reindex(index=order, columns=order, fill_value=0)

    total = pivot_df.values.sum()
    pct_df = (pivot_df / total * 100).round(2)

    logger.info(f"총 유저 수: {total:,}")

    # 히트맵 생성
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 히트맵 1: Count
    ax1 = axes[0]
    sns.heatmap(pivot_df, annot=True, fmt=',d', cmap='Blues', ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title(f'Churn Transition: {train_v1_table} -> {train_v2_table} (Count)', fontsize=12)
    ax1.set_xlabel(f'{train_v2_table} is_churn', fontsize=10)
    ax1.set_ylabel(f'{train_v1_table} is_churn', fontsize=10)

    # 히트맵 2: Count + Percentage
    ax2 = axes[1]
    annot_labels = []
    for i in range(pivot_df.shape[0]):
        row_labels = []
        for j in range(pivot_df.shape[1]):
            count = pivot_df.iloc[i, j]
            pct = pct_df.iloc[i, j]
            row_labels.append(f'{count:,}\n({pct:.1f}%)')
        annot_labels.append(row_labels)
    annot_labels = np.array(annot_labels)

    sns.heatmap(pivot_df, annot=annot_labels, fmt='', cmap='YlOrRd', ax=ax2, cbar_kws={'label': 'Count'})
    title_suffix = f" (ref: {reference_table})" if reference_table else ""
    ax2.set_title(f'Churn Transition (Count & %){title_suffix}', fontsize=12)
    ax2.set_xlabel(f'{train_v2_table} is_churn', fontsize=10)
    ax2.set_ylabel(f'{train_v1_table} is_churn', fontsize=10)

    plt.tight_layout()

    # 저장
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "churn_transition_heatmap.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"히트맵 저장: {output_file}")

    con.close()
    logger.info(f"=== Churn 전이행렬 분석 완료 ===\n")


# ============================================================================
# 5-6. 중복 트랜잭션 유저 분석 및 결과 저장
# ============================================================================
def analyze_duplicate_transactions(
    db_path: str,
    transactions_table: str,
    output_dir: str,
    date_col: str = "transaction_date",
    min_txn_count: int = 2,
    reference_table: str = None,
) -> None:
    """
    같은 날 비취소 트랜잭션이 지정된 개수 이상인 유저를 분석하고 결과를 저장합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        transactions_table: 트랜잭션 테이블 이름
        output_dir: 결과 이미지 저장 디렉토리
        date_col: 날짜 컬럼명 (기본값: transaction_date)
        min_txn_count: 중복으로 간주할 최소 트랜잭션 수 (기본값: 2)
        reference_table: 기준 테이블 (지정 시 해당 테이블에 있는 유저만 분석)
    """
    import matplotlib.pyplot as plt

    logger.info(f"=== 중복 트랜잭션 유저 분석 시작 ===")
    logger.info(f"트랜잭션 테이블: {transactions_table}")
    logger.info(f"중복 기준: 같은 날 비취소 트랜잭션 {min_txn_count}개 이상")
    if reference_table:
        logger.info(f"기준 테이블: {reference_table}")

    con = duckdb.connect(db_path, read_only=True)

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    if transactions_table not in existing_tables:
        logger.error(f"{transactions_table} 테이블이 존재하지 않습니다.")
        con.close()
        return

    # user_id 컬럼 확인
    tx_cols = [row[0] for row in con.execute(f"DESCRIBE {transactions_table}").fetchall()]
    id_col = "user_id" if "user_id" in tx_cols else "msno"

    # 기준 테이블 조건
    ref_condition = ""
    if reference_table and reference_table in existing_tables:
        ref_cols = [row[0] for row in con.execute(f"DESCRIBE {reference_table}").fetchall()]
        ref_id_col = "user_id" if "user_id" in ref_cols else "msno"
        ref_condition = f"AND {id_col} IN (SELECT {ref_id_col} FROM {reference_table})"

    # 전체 유저 수
    total_users = con.execute(f"""
        SELECT COUNT(DISTINCT {id_col})
        FROM {transactions_table}
        WHERE 1=1 {ref_condition}
    """).fetchone()[0]

    logger.info(f"분석 대상 유저 수: {total_users:,}")

    # 중복 트랜잭션 유저 분석
    dup_analysis = con.execute(f"""
        WITH non_cancel_txns AS (
            SELECT * FROM {transactions_table}
            WHERE is_cancel = 0 {ref_condition}
        ),
        user_date_counts AS (
            SELECT {id_col}, {date_col}, COUNT(*) as txn_count
            FROM non_cancel_txns
            GROUP BY {id_col}, {date_col}
            HAVING COUNT(*) >= {min_txn_count}
        ),
        dup_users AS (
            SELECT DISTINCT {id_col} FROM user_date_counts
        )
        SELECT
            (SELECT COUNT(*) FROM dup_users) as dup_user_count,
            (SELECT COUNT(*) FROM user_date_counts) as dup_pair_count,
            (SELECT SUM(txn_count) FROM user_date_counts) as dup_txn_count
    """).fetchone()

    dup_user_count = dup_analysis[0]
    dup_pair_count = dup_analysis[1]
    dup_txn_count = dup_analysis[2] or 0

    clean_users = total_users - dup_user_count
    dup_pct = dup_user_count / total_users * 100 if total_users > 0 else 0

    logger.info(f"중복 트랜잭션 유저 수: {dup_user_count:,} ({dup_pct:.2f}%)")
    logger.info(f"중복 유저-날짜 쌍 수: {dup_pair_count:,}")
    logger.info(f"중복 트랜잭션 총 수: {dup_txn_count:,}")
    logger.info(f"정상 유저 수: {clean_users:,}")

    # 중복 트랜잭션 개수별 분포
    txn_count_dist = con.execute(f"""
        WITH non_cancel_txns AS (
            SELECT * FROM {transactions_table}
            WHERE is_cancel = 0 {ref_condition}
        ),
        user_date_counts AS (
            SELECT {id_col}, {date_col}, COUNT(*) as txn_count
            FROM non_cancel_txns
            GROUP BY {id_col}, {date_col}
            HAVING COUNT(*) >= {min_txn_count}
        )
        SELECT txn_count, COUNT(*) as pairs
        FROM user_date_counts
        GROUP BY txn_count
        ORDER BY txn_count
    """).fetchdf()

    if not txn_count_dist.empty:
        logger.info(f"중복 트랜잭션 개수별 분포:")
        for _, row in txn_count_dist.iterrows():
            logger.info(f"  {row['txn_count']}건/일: {row['pairs']:,} 쌍")

    # 파이차트 생성
    fig, ax = plt.subplots(figsize=(8, 8))

    labels = [
        f'Without Duplicate Txns\n({clean_users:,})',
        f'With Duplicate Txns\n({dup_user_count:,})'
    ]
    sizes = [clean_users, dup_user_count]
    colors = ['#4CAF50', '#FF5722']
    explode = (0, 0.05)

    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct='%1.2f%%',
        startangle=90,
        textprops={'fontsize': 12},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )

    for autotext in autotexts:
        autotext.set_fontsize(14)
        autotext.set_fontweight('bold')

    title_suffix = f"\n(ref: {reference_table})" if reference_table else ""
    ax.set_title(f'Users with Duplicate Non-Cancel Transactions on Same Day\n(Total: {total_users:,} users){title_suffix}',
                 fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    # 저장
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "duplicate_txn_users_pie.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"파이차트 저장: {output_file}")

    con.close()
    logger.info(f"=== 중복 트랜잭션 유저 분석 완료 ===\n")


# ============================================================================
# 6. gender 필드 정수 변환 (null->-1, male->0, female->1)
# ============================================================================
def convert_gender_to_int(
    db_path: str,
    table_name: str,
) -> None:
    """
    지정된 테이블의 gender 필드를 정수로 변환합니다.
    null -> -1, male -> 0, female -> 1

    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 대상 테이블 이름
    """
    logger.info(f"=== gender 필드 정수 변환 시작 ===")
    logger.info(f"대상 테이블: {table_name}")

    con = duckdb.connect(db_path)

    # 테이블 존재 확인
    tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if table_name not in tables:
        logger.warning(f"{table_name} 테이블이 존재하지 않습니다.")
        con.close()
        return

    # gender 컬럼 존재 확인
    cols = [row[0] for row in con.execute(f"DESCRIBE {table_name}").fetchall()]
    if "gender" not in cols:
        logger.warning(f"{table_name}에 gender 컬럼이 없습니다.")
        con.close()
        return

    # 변환 전 분포 확인
    logger.info(f"변환 전 gender 분포:")
    before_dist = con.execute(f"""
        SELECT gender, COUNT(*) as cnt
        FROM {table_name}
        GROUP BY gender
        ORDER BY gender
    """).fetchdf()
    logger.info(f"\n{before_dist.to_string()}")

    # gender 변환
    con.execute(f"""
        CREATE OR REPLACE TABLE {table_name}_temp AS
        SELECT
            CASE
                WHEN gender IS NULL THEN -1
                WHEN gender = 'male' THEN 0
                WHEN gender = 'female' THEN 1
                ELSE -1
            END AS gender,
            * EXCLUDE (gender)
        FROM {table_name};
    """)

    con.execute(f"DROP TABLE {table_name};")
    con.execute(f"ALTER TABLE {table_name}_temp RENAME TO {table_name};")

    # 변환 후 분포 확인
    logger.info(f"변환 후 gender 분포:")
    after_dist = con.execute(f"""
        SELECT gender, COUNT(*) as cnt
        FROM {table_name}
        GROUP BY gender
        ORDER BY gender
    """).fetchdf()
    logger.info(f"\n{after_dist.to_string()}")

    con.close()
    logger.info(f"=== gender 필드 정수 변환 완료 ===\n")


# ============================================================================
# 7. 날짜 필드 datetime 변환
# ============================================================================
def convert_date_fields(
    db_path: str,
    target_tables: list[str],
    date_field_patterns: list[str] = [
        "date",
        "transaction_date",
        "membership_expire_date",
        "registration_init_time",
        "expiration_date",
    ],
) -> None:
    """
    지정된 테이블들의 YYYYmmdd 형식 날짜 필드를 DATE 타입으로 변환합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        target_tables: 변환을 적용할 테이블 이름 리스트
        date_field_patterns: 날짜로 변환할 컬럼명 패턴 리스트
    """
    logger.info(f"=== 날짜 필드 변환 시작 ===")
    logger.info(f"대상 테이블: {target_tables}")
    logger.info(f"날짜 필드 패턴: {date_field_patterns}")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    for table in tqdm(target_tables, desc="날짜 필드 변환"):
        if table not in existing_tables:
            logger.warning(f"  {table}: 존재하지 않음, 건너뜀")
            continue

        cols_info = con.execute(f"DESCRIBE {table}").fetchall()
        cols = {row[0]: row[1] for row in cols_info}  # {col_name: col_type}

        converted_cols = []

        for col_name in cols:
            # 날짜 필드 패턴 매칭
            if col_name.lower() not in date_field_patterns:
                continue

            col_type = cols[col_name].upper()

            # 이미 DATE/TIMESTAMP 타입이면 건너뜀
            if "DATE" in col_type or "TIMESTAMP" in col_type:
                logger.info(f"  {table}.{col_name}: 이미 {col_type}, 건너뜀")
                continue

            # INTEGER 또는 BIGINT 타입 (YYYYmmdd 형식)
            if "INT" in col_type or "BIGINT" in col_type:
                try:
                    # 샘플 데이터 확인
                    sample = con.execute(f"""
                        SELECT {col_name} FROM {table}
                        WHERE {col_name} IS NOT NULL AND {col_name} != 0
                        LIMIT 1
                    """).fetchone()

                    if sample:
                        sample_val = str(sample[0])
                        # YYYYmmdd 형식 확인 (8자리 숫자)
                        if len(sample_val) == 8 and sample_val.isdigit():
                            con.execute(f"""
                                ALTER TABLE {table}
                                ALTER COLUMN {col_name}
                                SET DATA TYPE DATE
                                USING CASE
                                    WHEN {col_name} IS NULL OR {col_name} = 0 THEN NULL
                                    ELSE STRPTIME(CAST({col_name} AS VARCHAR), '%Y%m%d')::DATE
                                END;
                            """)
                            converted_cols.append(col_name)
                            logger.info(f"  {table}.{col_name}: INT -> DATE 변환 완료")
                except Exception as e:
                    logger.error(f"  {table}.{col_name}: 변환 실패 - {e}")

        if converted_cols:
            # 변환된 컬럼의 범위 확인
            for col in converted_cols:
                try:
                    stats = con.execute(f"""
                        SELECT
                            MIN({col}) as min_date,
                            MAX({col}) as max_date,
                            COUNT(*) as total,
                            SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END) as null_count
                        FROM {table}
                    """).fetchone()
                    logger.info(f"    {col}: 범위 {stats[0]} ~ {stats[1]}, NULL: {stats[3]:,}/{stats[2]:,}")
                except Exception as e:
                    logger.error(f"    {col} 통계 조회 실패: {e}")

    con.close()
    logger.info(f"=== 날짜 필드 변환 완료 ===\n")


# ============================================================================
# 8. msno 컬럼명 -> user_id로 변경
# ============================================================================
def rename_msno_to_user_id(
    db_path: str,
    target_tables: list[str],
) -> None:
    """
    지정된 테이블들의 msno 컬럼명을 user_id로 변경합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        target_tables: 컬럼명을 변경할 테이블 이름 리스트
    """
    logger.info(f"=== msno -> user_id 컬럼명 변경 시작 ===")
    logger.info(f"대상 테이블: {target_tables}")

    con = duckdb.connect(db_path)

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    for table in tqdm(target_tables, desc="컬럼명 변경"):
        if table not in existing_tables:
            logger.warning(f"  {table}: 존재하지 않음, 건너뜀")
            continue

        cols = [row[0] for row in con.execute(f"DESCRIBE {table}").fetchall()]

        if "msno" in cols:
            con.execute(f"ALTER TABLE {table} RENAME COLUMN msno TO user_id;")
            logger.info(f"  {table}: msno -> user_id 변경 완료")
        elif "user_id" in cols:
            logger.info(f"  {table}: 이미 user_id 존재")
        else:
            logger.warning(f"  {table}: msno/user_id 컬럼 없음")

    con.close()
    logger.info(f"=== msno -> user_id 컬럼명 변경 완료 ===\n")


# ============================================================================
# 8.5. 트랜잭션 시퀀스 테이블 생성
# ============================================================================
def create_transactions_seq(
    db_path: str,
    source_table: str = "transactions_merge",
    target_table: str = "transactions_seq",
    gap_days: int = 30,
    cutoff_date: str = "2017-03-31",
) -> None:
    """
    트랜잭션 시퀀스 테이블을 생성합니다.

    각 유저의 트랜잭션을 시간순으로 정렬하고, 이전 membership_expire_date로부터
    gap_days일 이내에 발생한 연속 트랜잭션을 동일 시퀀스 그룹으로 연결합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        source_table: 원본 트랜잭션 테이블명 (기본값: transactions_merge)
        target_table: 생성할 시퀀스 테이블명 (기본값: transactions_seq)
        gap_days: 시퀀스 연결 기준 일수 (기본값: 30)
        cutoff_date: 데이터 범위 마지막 날짜 (기본값: 2017-03-31)

    생성되는 컬럼:
        - sequence_group_id: 유저 내 시퀀스 그룹 ID (0, 1, ...)
        - sequence_id: 그룹 내 순서 (0, 1, ...)
        - before_transaction_term: 이전 트랜잭션으로부터의 일수
        - before_membership_expire_term: 이전 membership_expire_date로부터의 일수
        - is_churn: 이탈 여부 (0: 유지, 1: 이탈, -1: 판단 불가)
    """
    logger.info(f"=== 트랜잭션 시퀀스 테이블 생성 시작 ===")
    logger.info(f"원본 테이블: {source_table}")
    logger.info(f"대상 테이블: {target_table}")
    logger.info(f"시퀀스 연결 기준: {gap_days}일")
    logger.info(f"데이터 cutoff: {cutoff_date}")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    # 원본 테이블 존재 확인
    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if source_table not in existing_tables:
        logger.error(f"원본 테이블 {source_table}이 존재하지 않습니다.")
        con.close()
        return

    # 원본 테이블 정보
    source_count = con.execute(f"SELECT COUNT(*) FROM {source_table}").fetchone()[0]
    logger.info(f"원본 테이블 행 수: {source_count:,}")

    # SQL 기반으로 시퀀스 테이블 생성 (Window 함수 활용)
    logger.info(f"SQL 기반 시퀀스 계산 중...")

    con.execute(f"""
        CREATE OR REPLACE TABLE {target_table} AS
        WITH ordered_txn AS (
            -- Step 1: 정렬 및 이전 행 정보 가져오기
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id
                    ORDER BY transaction_date, membership_expire_date
                ) AS row_num,
                LAG(transaction_date) OVER (
                    PARTITION BY user_id
                    ORDER BY transaction_date, membership_expire_date
                ) AS prev_txn_date,
                LAG(membership_expire_date) OVER (
                    PARTITION BY user_id
                    ORDER BY transaction_date, membership_expire_date
                ) AS prev_expire_date
            FROM {source_table}
        ),
        with_gap AS (
            -- Step 2: 이전 expire_date로부터의 일수 계산 및 시퀀스 끊김 여부 판단
            SELECT
                *,
                CASE
                    WHEN row_num = 1 THEN 1  -- 유저의 첫 트랜잭션은 새 그룹 시작
                    WHEN transaction_date - prev_expire_date > {gap_days} THEN 1  -- gap 초과시 새 그룹
                    ELSE 0
                END AS is_new_group
            FROM ordered_txn
        ),
        with_group AS (
            -- Step 3: 시퀀스 그룹 ID 계산 (cumsum of is_new_group - 1)
            SELECT
                *,
                SUM(is_new_group) OVER (
                    PARTITION BY user_id
                    ORDER BY transaction_date, membership_expire_date
                    ROWS UNBOUNDED PRECEDING
                ) - 1 AS sequence_group_id
            FROM with_gap
        ),
        with_seq_id AS (
            -- Step 4: 그룹 내 시퀀스 ID 계산
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id, sequence_group_id
                    ORDER BY transaction_date, membership_expire_date
                ) - 1 AS sequence_id
            FROM with_group
        ),
        with_terms AS (
            -- Step 5: before_transaction_term, before_membership_expire_term 계산
            -- 시퀀스 그룹 내 첫번째가 아니면: 바로 직전 트랜잭션으로부터 계산
            -- 시퀀스 그룹 내 첫번째이고 이전 시퀀스가 있으면: 이전 시퀀스 마지막으로부터 계산
            -- 시퀀스 그룹 내 첫번째이고 이전 시퀀스가 없으면: -1
            SELECT
                *,
                CASE
                    WHEN sequence_id > 0 THEN transaction_date - prev_txn_date
                    WHEN sequence_group_id > 0 THEN transaction_date - prev_txn_date
                    ELSE -1
                END AS before_transaction_term,
                CASE
                    WHEN sequence_id > 0 THEN transaction_date - prev_expire_date
                    WHEN sequence_group_id > 0 THEN transaction_date - prev_expire_date
                    ELSE -1
                END AS before_membership_expire_term
            FROM with_seq_id
        ),
        with_group_info AS (
            -- Step 6: 각 그룹의 마지막 트랜잭션 여부 및 다음 그룹 존재 여부 확인
            SELECT
                *,
                MAX(sequence_id) OVER (
                    PARTITION BY user_id, sequence_group_id
                ) AS max_seq_in_group,
                MAX(sequence_group_id) OVER (
                    PARTITION BY user_id
                ) AS max_group_for_user,
                LEAD(transaction_date) OVER (
                    PARTITION BY user_id
                    ORDER BY transaction_date, membership_expire_date
                ) AS next_txn_date
            FROM with_terms
        ),
        final AS (
            -- Step 7: is_churn 계산
            -- 그룹의 마지막이 아니면: 0
            -- 그룹의 마지막이고:
            --   - 다음 그룹이 있으면 (next_txn가 있고 gap 초과):
            --       expire + 30 > cutoff면 -1, 아니면 1
            --   - 유저의 마지막 트랜잭션이면:
            --       expire + 30 > cutoff면 -1, 아니면 1
            SELECT
                user_id,
                payment_method_id,
                payment_plan_days,
                plan_list_price,
                actual_amount_paid,
                is_auto_renew,
                transaction_date,
                membership_expire_date,
                is_cancel,
                CAST(sequence_group_id AS BIGINT) AS sequence_group_id,
                CAST(sequence_id AS BIGINT) AS sequence_id,
                CAST(before_transaction_term AS BIGINT) AS before_transaction_term,
                CAST(before_membership_expire_term AS BIGINT) AS before_membership_expire_term,
                CAST(
                    CASE
                        WHEN sequence_id < max_seq_in_group THEN 0  -- 그룹 중간
                        -- 그룹의 마지막 트랜잭션
                        WHEN membership_expire_date + INTERVAL '{gap_days} days' > DATE '{cutoff_date}' THEN -1
                        ELSE 1
                    END AS BIGINT
                ) AS is_churn
            FROM with_group_info
        )
        SELECT * FROM final
        ORDER BY user_id, transaction_date, membership_expire_date;
    """)

    # 결과 통계
    result_count = con.execute(f"SELECT COUNT(*) FROM {target_table}").fetchone()[0]
    unique_users = con.execute(f"SELECT COUNT(DISTINCT user_id) FROM {target_table}").fetchone()[0]

    # is_churn 분포
    churn_stats = con.execute(f"""
        SELECT is_churn, COUNT(*) as cnt
        FROM {target_table}
        GROUP BY is_churn
        ORDER BY is_churn
    """).fetchall()

    logger.info(f"생성 완료: {result_count:,} 행, {unique_users:,} 유저")
    logger.info(f"is_churn 분포:")
    for churn_val, cnt in churn_stats:
        logger.info(f"  {churn_val}: {cnt:,}")

    # 샘플 데이터 출력
    sample = con.execute(f"""
        SELECT user_id, transaction_date, membership_expire_date,
               sequence_group_id, sequence_id,
               before_transaction_term, before_membership_expire_term, is_churn
        FROM {target_table}
        WHERE user_id = (
            SELECT user_id FROM {target_table}
            GROUP BY user_id
            HAVING COUNT(DISTINCT sequence_group_id) >= 2
            LIMIT 1
        )
        ORDER BY transaction_date
        LIMIT 15
    """).fetchdf()
    logger.info(f"샘플 데이터 (시퀀스 그룹 2개 이상 유저):\n{sample.to_string()}")

    con.close()
    logger.info(f"=== 트랜잭션 시퀀스 테이블 생성 완료 ===\n")


# ============================================================================
# 8.6. members_merge에 멤버십 시퀀스 정보 추가
# ============================================================================
def add_membership_seq_info(
    db_path: str,
    members_table: str = "members_merge",
    seq_table: str = "transactions_seq",
) -> None:
    """
    members_merge 테이블에 멤버십 시퀀스 정보를 추가합니다.

    각 유저의 last_expire에 해당하는 트랜잭션의 시퀀스 정보를 저장합니다.
    만약 last_expire이 cancel로 인한 expire 갱신일 경우,
    is_cancel이 아닌 가장 최근의 멤버십을 직전 멤버십으로 간주합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        members_table: 멤버 테이블명 (기본값: members_merge)
        seq_table: 시퀀스 테이블명 (기본값: transactions_seq)

    추가되는 컬럼:
        - membership_seq_group_id: last_expire 트랜잭션의 sequence_group_id
        - membership_seq_id: 해당 시퀀스의 sequence_id
    """
    logger.info(f"=== 멤버십 시퀀스 정보 추가 시작 ===")
    logger.info(f"대상 테이블: {members_table}")
    logger.info(f"시퀀스 테이블: {seq_table}")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    # 테이블 존재 확인
    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if members_table not in existing_tables:
        logger.error(f"테이블 {members_table}이 존재하지 않습니다.")
        con.close()
        return
    if seq_table not in existing_tables:
        logger.error(f"테이블 {seq_table}이 존재하지 않습니다.")
        con.close()
        return

    # 기존 컬럼 삭제 (있으면)
    cols = [row[0] for row in con.execute(f"DESCRIBE {members_table}").fetchall()]
    if "membership_seq_group_id" in cols:
        con.execute(f"ALTER TABLE {members_table} DROP COLUMN membership_seq_group_id")
        logger.info("기존 membership_seq_group_id 컬럼 삭제")
    if "membership_seq_id" in cols:
        con.execute(f"ALTER TABLE {members_table} DROP COLUMN membership_seq_id")
        logger.info("기존 membership_seq_id 컬럼 삭제")

    # 새 컬럼 추가
    con.execute(f"ALTER TABLE {members_table} ADD COLUMN membership_seq_group_id BIGINT")
    con.execute(f"ALTER TABLE {members_table} ADD COLUMN membership_seq_id BIGINT")

    logger.info("시퀀스 정보 매핑 중...")

    # last_expire에 해당하는 트랜잭션의 시퀀스 정보 찾기
    # 1. last_expire와 membership_expire_date가 일치하는 트랜잭션 찾기
    # 2. 해당 트랜잭션이 is_cancel=1이면, is_cancel=0인 가장 최근 트랜잭션 찾기
    con.execute(f"""
        WITH matched_txn AS (
            -- last_expire와 일치하는 모든 트랜잭션 찾기
            SELECT
                m.user_id,
                m.last_expire,
                t.transaction_date,
                t.membership_expire_date,
                t.sequence_group_id,
                t.sequence_id,
                t.is_cancel,
                ROW_NUMBER() OVER (
                    PARTITION BY m.user_id
                    ORDER BY t.transaction_date DESC, t.sequence_id DESC
                ) AS rn
            FROM {members_table} m
            JOIN {seq_table} t ON m.user_id = t.user_id
                AND m.last_expire = t.membership_expire_date
            WHERE m.last_expire IS NOT NULL
        ),
        last_expire_txn AS (
            -- 각 유저의 last_expire에 해당하는 가장 마지막 트랜잭션
            SELECT * FROM matched_txn WHERE rn = 1
        ),
        non_cancel_fallback AS (
            -- is_cancel=1인 유저의 경우, is_cancel=0인 가장 최근 트랜잭션 찾기
            SELECT
                l.user_id,
                t2.sequence_group_id AS fallback_seq_group_id,
                t2.sequence_id AS fallback_seq_id,
                ROW_NUMBER() OVER (
                    PARTITION BY l.user_id
                    ORDER BY t2.transaction_date DESC, t2.sequence_id DESC
                ) AS rn2
            FROM last_expire_txn l
            JOIN {seq_table} t2 ON l.user_id = t2.user_id
                AND t2.is_cancel = 0
                AND t2.transaction_date <= l.transaction_date
            WHERE l.is_cancel = 1
        ),
        final_mapping AS (
            -- 최종 매핑: is_cancel=1이면 fallback 사용, 아니면 원래 값 사용
            SELECT
                l.user_id,
                CASE
                    WHEN l.is_cancel = 1 AND nc.fallback_seq_group_id IS NOT NULL
                        THEN nc.fallback_seq_group_id
                    ELSE l.sequence_group_id
                END AS final_seq_group_id,
                CASE
                    WHEN l.is_cancel = 1 AND nc.fallback_seq_id IS NOT NULL
                        THEN nc.fallback_seq_id
                    ELSE l.sequence_id
                END AS final_seq_id
            FROM last_expire_txn l
            LEFT JOIN non_cancel_fallback nc ON l.user_id = nc.user_id AND nc.rn2 = 1
        )
        UPDATE {members_table} m
        SET
            membership_seq_group_id = fm.final_seq_group_id,
            membership_seq_id = fm.final_seq_id
        FROM final_mapping fm
        WHERE m.user_id = fm.user_id;
    """)

    # 결과 통계
    total_members = con.execute(f"SELECT COUNT(*) FROM {members_table}").fetchone()[0]
    with_seq = con.execute(f"""
        SELECT COUNT(*) FROM {members_table}
        WHERE membership_seq_group_id IS NOT NULL
    """).fetchone()[0]
    without_seq = total_members - with_seq

    logger.info(f"전체 멤버: {total_members:,}")
    logger.info(f"시퀀스 정보 매핑됨: {with_seq:,}")
    logger.info(f"시퀀스 정보 없음: {without_seq:,}")

    # 샘플 데이터 출력
    sample = con.execute(f"""
        SELECT m.user_id, m.last_expire, m.membership_seq_group_id, m.membership_seq_id,
               t.is_cancel, t.is_churn
        FROM {members_table} m
        LEFT JOIN {seq_table} t ON m.user_id = t.user_id
            AND m.membership_seq_group_id = t.sequence_group_id
            AND m.membership_seq_id = t.sequence_id
        WHERE m.membership_seq_group_id IS NOT NULL
        LIMIT 10
    """).fetchdf()
    logger.info(f"샘플 데이터:\n{sample.to_string()}")

    con.close()
    logger.info(f"=== 멤버십 시퀀스 정보 추가 완료 ===\n")


# ============================================================================
# 8.7. members_merge에 멤버십 기간 정보 추가
# ============================================================================
def add_membership_duration_info(
    db_path: str,
    members_table: str = "members_merge",
    seq_table: str = "transactions_seq",
) -> None:
    """
    members_merge 테이블에 멤버십 기간 정보를 추가합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        members_table: 멤버 테이블명 (기본값: members_merge)
        seq_table: 시퀀스 테이블명 (기본값: transactions_seq)

    추가되는 컬럼:
        - previous_membership_duration: 직전 멤버십 시작(transaction_date) ~ last_expire 기간 (days)
        - previous_membership_seq_duration: 시퀀스 그룹 시작 ~ last_expire 기간 (days)
    """
    logger.info(f"=== 멤버십 기간 정보 추가 시작 ===")
    logger.info(f"대상 테이블: {members_table}")
    logger.info(f"시퀀스 테이블: {seq_table}")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    # 테이블 존재 확인
    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if members_table not in existing_tables:
        logger.error(f"테이블 {members_table}이 존재하지 않습니다.")
        con.close()
        return
    if seq_table not in existing_tables:
        logger.error(f"테이블 {seq_table}이 존재하지 않습니다.")
        con.close()
        return

    # 기존 컬럼 삭제 (있으면)
    cols = [row[0] for row in con.execute(f"DESCRIBE {members_table}").fetchall()]
    if "previous_membership_duration" in cols:
        con.execute(f"ALTER TABLE {members_table} DROP COLUMN previous_membership_duration")
        logger.info("기존 previous_membership_duration 컬럼 삭제")
    if "previous_membership_seq_duration" in cols:
        con.execute(f"ALTER TABLE {members_table} DROP COLUMN previous_membership_seq_duration")
        logger.info("기존 previous_membership_seq_duration 컬럼 삭제")

    # 새 컬럼 추가
    con.execute(f"ALTER TABLE {members_table} ADD COLUMN previous_membership_duration BIGINT")
    con.execute(f"ALTER TABLE {members_table} ADD COLUMN previous_membership_seq_duration BIGINT")

    logger.info("멤버십 기간 계산 중...")

    # 기간 계산
    # 1. previous_membership_duration: membership_seq_id 트랜잭션의 transaction_date ~ last_expire
    # 2. previous_membership_seq_duration: 시퀀스 그룹 첫 트랜잭션(seq_id=0)의 transaction_date ~ last_expire
    con.execute(f"""
        WITH membership_txn AS (
            -- 각 유저의 membership_seq_id에 해당하는 트랜잭션 정보
            SELECT
                m.user_id,
                m.last_expire,
                t.transaction_date AS membership_start_date
            FROM {members_table} m
            JOIN {seq_table} t ON m.user_id = t.user_id
                AND m.membership_seq_group_id = t.sequence_group_id
                AND m.membership_seq_id = t.sequence_id
            WHERE m.membership_seq_group_id IS NOT NULL
        ),
        seq_start AS (
            -- 각 유저의 시퀀스 그룹 시작일 (sequence_id = 0인 트랜잭션)
            SELECT
                m.user_id,
                t.transaction_date AS seq_start_date
            FROM {members_table} m
            JOIN {seq_table} t ON m.user_id = t.user_id
                AND m.membership_seq_group_id = t.sequence_group_id
                AND t.sequence_id = 0
            WHERE m.membership_seq_group_id IS NOT NULL
        ),
        duration_calc AS (
            SELECT
                mt.user_id,
                mt.last_expire - mt.membership_start_date AS membership_duration,
                mt.last_expire - ss.seq_start_date AS seq_duration
            FROM membership_txn mt
            JOIN seq_start ss ON mt.user_id = ss.user_id
        )
        UPDATE {members_table} m
        SET
            previous_membership_duration = dc.membership_duration,
            previous_membership_seq_duration = dc.seq_duration
        FROM duration_calc dc
        WHERE m.user_id = dc.user_id;
    """)

    # 결과 통계
    stats = con.execute(f"""
        SELECT
            COUNT(*) AS total,
            COUNT(previous_membership_duration) AS with_duration,
            AVG(previous_membership_duration) AS avg_membership_duration,
            AVG(previous_membership_seq_duration) AS avg_seq_duration,
            MIN(previous_membership_duration) AS min_membership_duration,
            MAX(previous_membership_duration) AS max_membership_duration,
            MIN(previous_membership_seq_duration) AS min_seq_duration,
            MAX(previous_membership_seq_duration) AS max_seq_duration
        FROM {members_table}
    """).fetchone()

    logger.info(f"전체 멤버: {stats[0]:,}")
    logger.info(f"기간 정보 매핑됨: {stats[1]:,}")
    logger.info(f"previous_membership_duration - 평균: {stats[2]:.1f}일, 범위: {stats[4]}~{stats[5]}일")
    logger.info(f"previous_membership_seq_duration - 평균: {stats[3]:.1f}일, 범위: {stats[6]}~{stats[7]}일")

    # 샘플 데이터 출력
    sample = con.execute(f"""
        SELECT user_id, last_expire, membership_seq_group_id, membership_seq_id,
               previous_membership_duration, previous_membership_seq_duration
        FROM {members_table}
        WHERE previous_membership_duration IS NOT NULL
        ORDER BY previous_membership_seq_duration DESC
        LIMIT 10
    """).fetchdf()
    logger.info(f"샘플 데이터 (시퀀스 기간 긴 순):\n{sample.to_string()}")

    con.close()
    logger.info(f"=== 멤버십 기간 정보 추가 완료 ===\n")


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

    # 1. CSV -> DuckDB 로드
    # load_csv_to_duckdb(
    #     db_path=DB_PATH,
    #     csv_dir=CSV_DIR,
    #     csv_files=[
    #         "raw_train_v1.csv",
    #         "raw_train_v2.csv",
    #         "raw_transactions_v1.csv",
    #         "raw_transactions_v2.csv",
    #         "raw_user_logs_v1.csv",
    #         "raw_user_logs_v2.csv",
    #         "raw_members_v3.csv",
    #     ],
    #     chunksize=1_000_000,
    #     force_overwrite=True,
    # )

    # 2. 테이블 이름 정규화 (_v1 추가)
    # rename_tables_add_v1_suffix(
    #     db_path=DB_PATH,
    #     tables=[
    #         "train",
    #         "transactions",
    #         "user_logs",
    #     ]
    # )

    # # 3. v1/v2 병합 테이블 생성
    # create_merge_tables(
    #     db_path=DB_PATH,
    #     base_names=[
    #         "train",
    #         "transactions",
    #         "user_logs",
    #         "members",
    #     ],
    #     drop_source_tables=False,
    #     id_col="user_id",
    # )

    # 4. user_id 매핑 생성 및 변환
    # create_user_id_mapping(
    #     db_path=DB_PATH,
    #     target_tables=[
    #         "train_v1",
    #         "train_v2",
    #         "transactions_v1",
    #         "transactions_v2",
    #         "user_logs_v1",
    #         "user_logs_v2",
    #         "members_v3",

    #         "train_merge",
    #         "transactions_merge",
    #         "user_logs_merge",
    #         "members_merge",
    #     ],
    #     mapping_table_name="user_id_map",
    # )

    # # 5. 공통 msno 교집합 필터링 (모든 대상 테이블에 존재하는 msno만 남김)
    # filter_common_msno(
    #     db_path=DB_PATH,
    #     target_tables=[
    #         "train_merge",
    #         "transactions_merge",
    #         "user_logs_merge",
    #         "members_merge",
    #     ],
    #     id_col="user_id",
    # )

    # # 5.5. members_merge 테이블을 기준으로 필터링
    # filter_by_reference_table(
    #     db_path=DB_PATH,
    #     reference_table="members_merge",  # 기준 테이블
    #     target_tables=[                  # 필터링할 대상 테이블들
    #         "train_v1",
    #         "train_v2",
    #         "transactions_v1",
    #         "transactions_v2",
    #         "user_logs_v1",
    #         "user_logs_v2",
    #         "members_v3",
    #     ],
    # )

    # # 6. gender 필드 정수 변환 (null->-1, male->0, female->1)
    # convert_gender_to_int(db_path=DB_PATH, table_name="members_v3")
    # convert_gender_to_int(db_path=DB_PATH, table_name="members_merge")

    # # 7. 날짜 필드 변환
    # convert_date_fields(
    #     db_path=DB_PATH,
    #     target_tables=[
    #         "train_v1",
    #         "train_v2",
    #         "transactions_v1",
    #         "transactions_v2",
    #         "user_logs_v1",
    #         "user_logs_v2",
    #         "members_v3",

    #         "train_merge",
    #         "transactions_merge",
    #         "user_logs_merge",
    #         "members_merge",
    #     ],
    # )

    # # 8. msno -> user_id 컬럼명 변경
    # rename_msno_to_user_id(
    #     db_path=DB_PATH,
    #     target_tables=[
    #         "train_v1",
    #         "train_v2",
    #         "transactions_v1",
    #         "transactions_v2",
    #         "user_logs_v1",
    #         "user_logs_v2",
    #         "members_v3",

    #         "train_merge",
    #         "transactions_merge",
    #         "user_logs_merge",
    #         "members_merge",

    #         # "user_id_map",
    #     ]
    # )

    # # 8.1.1. 전이행렬 분석 히트맵 저장
    # analyze_churn_transition(
    #     db_path=DB_PATH,
    #     train_v1_table="train_v1",
    #     train_v2_table="train_v2",
    #     output_dir="data/analysis",
    #     reference_table="members_merge",  # 현재 기준 유저만 분석
    # )

    # # 8.1. Churn 전이 기반 필터링 (v1=0 & v2=0/1인 유저만 남김)
    # filter_by_churn_transition(
    #     db_path=DB_PATH,
    #     train_v1_table="train_v1",
    #     train_v2_table="train_v2",
    #     target_tables=[
    #         "train_merge",
    #         "transactions_merge",
    #         "user_logs_merge",
    #         "members_merge",
    #         "user_logs_with_transactions",
    #     ],
    #     v1_churn_value=0,           # train_v1에서 is_churn=0인 유저
    #     v2_churn_values=[0, 1],     # train_v2에서 is_churn이 0 또는 1인 유저
    # )

    # # 8.2.1. 중복 트랜잭션 비율 분석 표 출력 및 결과 저장
    # analyze_duplicate_transactions(
    #     db_path=DB_PATH,
    #     transactions_table="transactions_merge",
    #     output_dir="data/analysis",
    #     date_col="transaction_date",
    #     min_txn_count=2,
    #     reference_table="members_merge",  # 현재 기준 유저만 분석
    # )

    # # 8.2. 중복 트랜잭션 유저 제외 (같은 날 비취소 트랜잭션 2개 이상인 유저 제거)
    # exclude_duplicate_transaction_users(
    #     db_path=DB_PATH,
    #     transactions_table="transactions_merge",
    #     target_tables=[
    #         "train_merge",
    #         "transactions_merge",
    #         "user_logs_merge",
    #         "members_merge",
    #         "user_logs_with_transactions",
    #     ],
    #     date_col="transaction_date",
    #     min_txn_count=2,            # 같은 날 2개 이상이면 제외
    # )

    # # 8.3. total_secs 범위 밖 값을 NULL로 변환
    # nullify_out_of_range(
    #     db_path=DB_PATH,
    #     table_name="user_logs_merge",
    #     column_name="total_secs",
    #     min_val=0,
    #     max_val=86400,
    # )

    # analyze_feature_correlation(
    #     db_path=DB_PATH,
    #     table_name="user_logs_merge",
    #     output_dir="data/analysis",
    #     exclude_cols=["user_id"],
    #     # sample_size=100000,  # 대용량 테이블은 샘플링 권장
    #     corr_threshold=0.7,
    #     null_values={
    #         "total_secs": {"min": 0, "max": 86400},
    #     },
    # )
    # analyze_feature_correlation(
    #     db_path=DB_PATH,
    #     table_name="members_merge",
    #     output_dir="data/analysis",
    #     exclude_cols=["user_id"],
    #     # sample_size=100000,  # 대용량 테이블은 샘플링 권장
    #     corr_threshold=0.7,
    #     null_values={
    #         "gender": [-1],              # -1을 NULL로
    #         "bd": {"min": 0, "max": 100},                   # 0-100 범위 밖을 NULL로
    #     },
    # )

    # # 8.5. 트랜잭션 시퀀스 테이블 생성
    # create_transactions_seq(
    #     db_path=DB_PATH,
    #     source_table="transactions_merge",
    #     target_table="transactions_seq",
    #     gap_days=30,
    #     cutoff_date="2017-03-31",
    # )

    # # 8.6. members_merge에 멤버십 시퀀스 정보 추가
    # add_membership_seq_info(
    #     db_path=DB_PATH,
    #     members_table="members_merge",
    #     seq_table="transactions_seq",
    # )

    # # 8.7. members_merge에 멤버십 기간 정보 추가
    # add_membership_duration_info(
    #     db_path=DB_PATH,
    #     members_table="members_merge",
    #     seq_table="transactions_seq",
    # )

    # # 8.8. user_logs_merge에 total_hours 컬럼 추가
    # add_converted_column(
    #     db_path=DB_PATH,
    #     table_name="user_logs_merge",
    #     source_col="total_secs",
    #     target_col="total_hours",
    #     divisor=3600,
    #     # clip_min=0,
    #     # clip_max=24,
    #     force_overwrite=True,
    # )

    # # 8.9. Feature 클리핑 구간 분석
    # analyze_clipping_distribution(
    #     db_path=DB_PATH,
    #     table_name="user_logs_merge",
    #     column_name="total_secs",
    #     output_dir="data/analysis",
    #     clip_min=0,
    #     clip_max=86400,  # 24시간 = 86400초
    #     sample_size=1_000_000,  # 대용량 테이블은 샘플링 권장
    # )

    # 8.10. 조건부 데이터 변환 (이상치 처리 등)
    # apply_conditional_transform(
    #     db_path=DB_PATH,
    #     table_name="user_logs_merge",
    #     rules=[
    #         # 음수 total_secs 값을 NULL로 변환
    #         {
    #             "column": "total_secs",
    #             "operator": "<",
    #             "value": 0,
    #             "action": "delete_row",
    #         },
    #         # 24시간(86400초) 초과 값을 86400으로 클리핑
    #         {
    #             "column": "total_secs",
    #             "operator": ">",
    #             "value": 86400,
    #             "action": "set_value",
    #             "new_value": 86400,
    #         },
    #     ],
    #     dry_run=False,  # True로 설정하면 실제 변경 없이 영향 행 수만 출력
    # )

    # 9. Parquet 내보내기
    export_to_parquet(
        db_path=DB_PATH,
        output_dir=PARQUET_DIR,
        tables=[
            # "raw_train_v1",
            # "raw_train_v2",
            # "raw_transactions_v1",
            # "raw_transactions_v2",
            # "raw_user_logs_v1",
            # "raw_user_logs_v2",
            # "raw_members_v3",

            # "train_v1",
            # "train_v2",
            # "transactions_v1",
            # "transactions_v2",
            # "user_logs_v1",
            # "user_logs_v2",
            # "members_v3",

            "train_merge",
            "transactions_merge",
            "user_logs_merge",
            "members_merge",
            "transactions_seq",

            # "user_id_map",

            # "user_logs_with_transactions"
        ],
    )

    # (선택) 최종 데이터베이스 정보 출력
    show_database_info(db_path=DB_PATH)
