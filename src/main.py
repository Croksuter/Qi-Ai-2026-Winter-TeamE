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
import matplotlib.pyplot as plt
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
        merge_name = f"{base.replace('raw_', '')}_merge"

        v1_exists = v1_name in tables
        v2_exists = v2_name in tables
        v3_exists = v3_name in tables

        logger.info(f"  {base}: v1={v1_exists}, v2={v2_exists}, v3={v3_exists}")

        # members 특수 처리: members_v3 -> members_merge
        if base == "raw_members":
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
        if base == "raw_train":
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


def exclude_null_total_secs_users(
    db_path: str,
    logs_table: str = "user_logs_merge",
    target_tables: list[str] = None,
) -> None:
    """
    user_logs_merge에서 total_secs가 NULL인 유저를 모든 _merge 테이블에서 제외합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        logs_table: 로그 테이블 이름 (기본값: user_logs_merge)
        target_tables: 필터링을 적용할 대상 테이블 이름 리스트
                       기본값: ["train_merge", "transactions_merge", "user_logs_merge", "members_merge"]
    """
    if target_tables is None:
        target_tables = ["train_merge", "transactions_merge", "user_logs_merge", "members_merge"]

    logger.info(f"=== NULL total_secs 유저 제외 시작 ===")
    logger.info(f"로그 테이블: {logs_table}")
    logger.info(f"대상 테이블: {target_tables}")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    # 로그 테이블 존재 확인
    if logs_table not in existing_tables:
        logger.error(f"{logs_table} 테이블이 존재하지 않습니다.")
        con.close()
        return

    # 로그 테이블의 user_id 컬럼 확인
    logs_cols = [row[0] for row in con.execute(f"DESCRIBE {logs_table}").fetchall()]
    if "user_id" in logs_cols:
        id_col = "user_id"
    elif "msno" in logs_cols:
        id_col = "msno"
    else:
        logger.error(f"{logs_table}에 user_id/msno 컬럼이 없습니다.")
        con.close()
        return

    # 전체 유저 수
    total_users = con.execute(f"""
        SELECT COUNT(DISTINCT {id_col}) FROM {logs_table}
    """).fetchone()[0]
    logger.info(f"로그 테이블 전체 유저 수: {total_users:,}")

    # NULL total_secs가 있는 유저 추출
    con.execute(f"""
        CREATE OR REPLACE TABLE _null_total_secs_users_temp AS
        SELECT DISTINCT {id_col}
        FROM {logs_table}
        WHERE total_secs IS NULL;
    """)

    null_user_count = con.execute("SELECT COUNT(*) FROM _null_total_secs_users_temp").fetchone()[0]
    logger.info(f"NULL total_secs 유저 수: {null_user_count:,} ({null_user_count/total_users*100:.2f}%)")

    if null_user_count == 0:
        logger.info("제외할 유저가 없습니다.")
        con.execute("DROP TABLE IF EXISTS _null_total_secs_users_temp;")
        con.close()
        return

    # 대상 테이블에서 NULL 유저 제외
    for t in tqdm(target_tables, desc="NULL total_secs 유저 제외"):
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

        # NULL 유저 제외 (NOT IN)
        con.execute(f"""
            CREATE OR REPLACE TABLE {t}_filtered AS
            SELECT *
            FROM {t}
            WHERE {target_col} NOT IN (SELECT {id_col} FROM _null_total_secs_users_temp);
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
    con.execute("DROP TABLE IF EXISTS _null_total_secs_users_temp;")

    con.close()
    logger.info(f"=== NULL total_secs 유저 제외 완료 ===\n")


def analyze_plan_days_30_31_users(
    db_path: str,
    transactions_table: str = "transactions_merge",
) -> None:
    """
    plan_days가 30, 31인 트랜잭션만 가진 유저 수를 분석합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        transactions_table: 트랜잭션 테이블 이름 (기본값: transactions_merge)
    """
    logger.info(f"=== plan_days 30/31 유저 분석 시작 ===")
    logger.info(f"트랜잭션 테이블: {transactions_table}")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    if transactions_table not in existing_tables:
        logger.error(f"{transactions_table} 테이블이 존재하지 않습니다.")
        con.close()
        return

    # 전체 유저 수
    total_users = con.execute(f"""
        SELECT COUNT(DISTINCT user_id) FROM {transactions_table}
    """).fetchone()[0]
    logger.info(f"전체 유저 수: {total_users:,}")

    # plan_days가 30 또는 31인 트랜잭션만 가진 유저 수
    only_30_31_users = con.execute(f"""
        WITH user_plan_days AS (
            SELECT user_id, ARRAY_AGG(DISTINCT payment_plan_days) AS plan_days_list
            FROM {transactions_table}
            GROUP BY user_id
        )
        SELECT COUNT(*)
        FROM user_plan_days
        WHERE list_sort(plan_days_list) = [30]
           OR list_sort(plan_days_list) = [31]
           OR list_sort(plan_days_list) = [30, 31]
    """).fetchone()[0]

    logger.info(f"plan_days 30/31만 가진 유저 수: {only_30_31_users:,} ({only_30_31_users/total_users*100:.2f}%)")

    # 상세 분포
    plan_days_dist = con.execute(f"""
        WITH user_plan_days AS (
            SELECT user_id, ARRAY_AGG(DISTINCT payment_plan_days ORDER BY payment_plan_days) AS plan_days_list
            FROM {transactions_table}
            GROUP BY user_id
        )
        SELECT
            CASE
                WHEN plan_days_list = [30] THEN 'only_30'
                WHEN plan_days_list = [31] THEN 'only_31'
                WHEN plan_days_list = [30, 31] THEN 'both_30_31'
                ELSE 'other'
            END AS category,
            COUNT(*) AS user_count
        FROM user_plan_days
        GROUP BY category
        ORDER BY user_count DESC
    """).fetchall()

    logger.info(f"상세 분포:")
    for category, count in plan_days_dist:
        logger.info(f"  {category}: {count:,} ({count/total_users*100:.2f}%)")

    con.close()
    logger.info(f"=== plan_days 30/31 유저 분석 완료 ===\n")


def exclude_out_of_range_expire_date_users(
    db_path: str,
    transactions_table: str = "transactions_merge",
    target_tables: list[str] = None,
    min_year: int = 2015,
    max_year: int = 2017,
) -> None:
    """
    membership_expire_date가 지정된 연도 범위를 벗어나는 유저를 모든 _merge 테이블에서 제외합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        transactions_table: 트랜잭션 테이블 이름 (기본값: transactions_merge)
        target_tables: 필터링을 적용할 대상 테이블 이름 리스트
                       기본값: ["train_merge", "transactions_merge", "user_logs_merge", "members_merge"]
        min_year: 최소 연도 (포함, 기본값: 2015)
        max_year: 최대 연도 (포함, 기본값: 2017)
    """
    if target_tables is None:
        target_tables = ["train_merge", "transactions_merge", "user_logs_merge", "members_merge"]

    logger.info(f"=== membership_expire_date 범위 밖 유저 제외 시작 ===")
    logger.info(f"트랜잭션 테이블: {transactions_table}")
    logger.info(f"연도 범위: [{min_year}, {max_year}]")
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

    # 범위 밖 membership_expire_date를 가진 유저 추출
    con.execute(f"""
        CREATE OR REPLACE TABLE _out_of_range_expire_users_temp AS
        SELECT DISTINCT {id_col}
        FROM {transactions_table}
        WHERE YEAR(membership_expire_date) < {min_year}
           OR YEAR(membership_expire_date) > {max_year};
    """)

    out_of_range_count = con.execute("SELECT COUNT(*) FROM _out_of_range_expire_users_temp").fetchone()[0]
    logger.info(f"범위 밖 유저 수: {out_of_range_count:,} ({out_of_range_count/total_users*100:.2f}%)")

    if out_of_range_count == 0:
        logger.info("제외할 유저가 없습니다.")
        con.execute("DROP TABLE IF EXISTS _out_of_range_expire_users_temp;")
        con.close()
        return

    # 대상 테이블에서 범위 밖 유저 제외
    for t in tqdm(target_tables, desc="범위 밖 expire_date 유저 제외"):
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

        # 범위 밖 유저 제외 (NOT IN)
        con.execute(f"""
            CREATE OR REPLACE TABLE {t}_filtered AS
            SELECT *
            FROM {t}
            WHERE {target_col} NOT IN (SELECT {id_col} FROM _out_of_range_expire_users_temp);
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
    con.execute("DROP TABLE IF EXISTS _out_of_range_expire_users_temp;")

    con.close()
    logger.info(f"=== membership_expire_date 범위 밖 유저 제외 완료 ===\n")


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
# 8.5.1. transactions_seq에 actual_plan_days 추가
# ============================================================================
def add_actual_plan_days(
    db_path: str,
    seq_table: str = "transactions_seq",
) -> None:
    """
    transactions_seq 테이블에 actual_plan_days 컬럼을 추가합니다.

    payment_plan_days는 결제 상품의 기간이지만, 실제 멤버십 연장 기간과 다를 수 있습니다.
    actual_plan_days는 실제로 멤버십이 연장되는 기간을 계산합니다:

    - 직전 멤버십 만료 후 트랜잭션 (transaction_date > prev_expire_date):
      actual_plan_days = membership_expire_date - transaction_date

    - 직전 멤버십 만료 전 트랜잭션 (transaction_date <= prev_expire_date):
      actual_plan_days = membership_expire_date - prev_expire_date

    - 유저의 첫 트랜잭션 (prev_expire_date 없음):
      actual_plan_days = membership_expire_date - transaction_date

    Args:
        db_path: DuckDB 데이터베이스 경로
        seq_table: 트랜잭션 시퀀스 테이블명 (기본값: transactions_seq)
    """
    logger.info(f"=== {seq_table}에 actual_plan_days 추가 시작 ===")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    # 테이블 존재 확인
    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if seq_table not in existing_tables:
        logger.error(f"테이블 {seq_table}이 존재하지 않습니다.")
        con.close()
        return

    # 이미 컬럼이 존재하는지 확인
    columns = [row[0] for row in con.execute(f"DESCRIBE {seq_table}").fetchall()]
    if "actual_plan_days" in columns:
        logger.info("actual_plan_days 컬럼이 이미 존재합니다. 재계산합니다.")
        con.execute(f"ALTER TABLE {seq_table} DROP COLUMN actual_plan_days")

    # actual_plan_days 계산 및 추가
    logger.info("actual_plan_days 계산 중...")

    con.execute(f"""
        -- 임시 테이블에 actual_plan_days 계산
        CREATE OR REPLACE TABLE {seq_table}_temp AS
        WITH with_prev_expire AS (
            SELECT
                *,
                LAG(membership_expire_date) OVER (
                    PARTITION BY user_id
                    ORDER BY transaction_date, membership_expire_date
                ) AS prev_expire_date
            FROM {seq_table}
        )
        SELECT
            *,
            CAST(
                CASE
                    -- 유저의 첫 트랜잭션 (이전 만료일 없음)
                    WHEN prev_expire_date IS NULL THEN
                        membership_expire_date - transaction_date

                    -- 멤버십 만료 후 트랜잭션 (갱신 전 이탈 기간 있음)
                    WHEN transaction_date > prev_expire_date THEN
                        membership_expire_date - transaction_date

                    -- 멤버십 만료 전 트랜잭션 (연속 갱신)
                    ELSE
                        membership_expire_date - prev_expire_date
                END AS BIGINT
            ) AS actual_plan_days
        FROM with_prev_expire;
    """)

    # prev_expire_date 컬럼 제거하고 원본 테이블 교체
    con.execute(f"""
        CREATE OR REPLACE TABLE {seq_table} AS
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
            sequence_group_id,
            sequence_id,
            before_transaction_term,
            before_membership_expire_term,
            is_churn,
            actual_plan_days
        FROM {seq_table}_temp
        ORDER BY user_id, transaction_date, membership_expire_date;
    """)

    # 임시 테이블 삭제
    con.execute(f"DROP TABLE IF EXISTS {seq_table}_temp")

    # 결과 통계
    stats = con.execute(f"""
        SELECT
            COUNT(*) AS total_rows,
            AVG(actual_plan_days) AS avg_actual,
            AVG(payment_plan_days) AS avg_payment,
            SUM(CASE WHEN actual_plan_days != payment_plan_days THEN 1 ELSE 0 END) AS diff_count,
            MIN(actual_plan_days) AS min_actual,
            MAX(actual_plan_days) AS max_actual
        FROM {seq_table}
    """).fetchone()

    logger.info(f"추가 완료: {stats[0]:,} 행")
    logger.info(f"actual_plan_days 통계:")
    logger.info(f"  평균: {stats[1]:.1f}일 (payment_plan_days 평균: {stats[2]:.1f}일)")
    logger.info(f"  범위: [{stats[4]}, {stats[5]}]일")
    logger.info(f"  payment_plan_days와 다른 행: {stats[3]:,} ({100*stats[3]/stats[0]:.1f}%)")

    # 차이 분포 샘플
    diff_sample = con.execute(f"""
        SELECT
            payment_plan_days,
            actual_plan_days,
            actual_plan_days - payment_plan_days AS diff,
            COUNT(*) AS cnt
        FROM {seq_table}
        WHERE actual_plan_days != payment_plan_days
        GROUP BY payment_plan_days, actual_plan_days
        ORDER BY cnt DESC
        LIMIT 10
    """).fetchdf()
    if len(diff_sample) > 0:
        logger.info(f"차이 분포 (상위 10개):\n{diff_sample.to_string()}")

    con.close()
    logger.info(f"=== {seq_table}에 actual_plan_days 추가 완료 ===\n")


# ============================================================================
# 8.5.2. actual_plan_days 분석
# ============================================================================
def analyze_actual_plan_days(
    db_path: str,
    seq_table: str = "transactions_seq",
    output_dir: str = "data/analysis",
) -> None:
    """
    actual_plan_days의 음수/0/양수 분포 및 추가 통계를 분석합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        seq_table: 트랜잭션 시퀀스 테이블명 (기본값: transactions_seq)
        output_dir: 그래프 저장 디렉토리 (기본값: data/analysis)
    """
    import matplotlib.pyplot as plt

    logger.info(f"=== actual_plan_days 분석 시작 ===")
    logger.info(f"테이블: {seq_table}")
    logger.info(f"출력 디렉토리: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    # 테이블 존재 확인
    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if seq_table not in existing_tables:
        logger.error(f"테이블 {seq_table}이 존재하지 않습니다.")
        con.close()
        return

    # 전체 행/유저 수
    total_stats = con.execute(f"""
        SELECT COUNT(*) AS total_rows, COUNT(DISTINCT user_id) AS total_users
        FROM {seq_table}
    """).fetchone()
    total_rows, total_users = total_stats
    logger.info(f"전체: {total_rows:,} 행, {total_users:,} 유저")

    # 1. 음수/0/양수 분포 (행 및 유저 기준)
    logger.info(f"\n[1] actual_plan_days 부호별 분포")
    sign_dist = con.execute(f"""
        SELECT
            CASE
                WHEN actual_plan_days < 0 THEN 'negative'
                WHEN actual_plan_days = 0 THEN 'zero'
                ELSE 'positive'
            END AS sign_category,
            COUNT(*) AS row_count,
            COUNT(DISTINCT user_id) AS user_count
        FROM {seq_table}
        GROUP BY sign_category
        ORDER BY
            CASE sign_category
                WHEN 'negative' THEN 1
                WHEN 'zero' THEN 2
                ELSE 3
            END
    """).fetchall()

    sign_categories = []
    sign_row_counts = []
    sign_user_counts = []
    for category, row_cnt, user_cnt in sign_dist:
        sign_categories.append(category)
        sign_row_counts.append(row_cnt)
        sign_user_counts.append(user_cnt)
        logger.info(f"  {category}: {row_cnt:,} 행 ({row_cnt/total_rows*100:.2f}%), {user_cnt:,} 유저 ({user_cnt/total_users*100:.2f}%)")

    # 부호별 분포 그래프
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 행 기준
    bars1 = axes[0].bar(sign_categories, sign_row_counts, color=['#e74c3c', '#f39c12', '#27ae60'])
    axes[0].set_title('actual_plan_days Sign Distribution (Rows)')
    axes[0].set_xlabel('Sign Category')
    axes[0].set_ylabel('Row Count')
    axes[0].set_ylim(bottom=0)
    for bar, cnt in zip(bars1, sign_row_counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{cnt:,}\n({cnt/total_rows*100:.1f}%)',
                     ha='center', va='bottom', fontsize=9)

    # 유저 기준
    bars2 = axes[1].bar(sign_categories, sign_user_counts, color=['#e74c3c', '#f39c12', '#27ae60'])
    axes[1].set_title('actual_plan_days Sign Distribution (Users)')
    axes[1].set_xlabel('Sign Category')
    axes[1].set_ylabel('User Count')
    axes[1].set_ylim(bottom=0)
    for bar, cnt in zip(bars2, sign_user_counts):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{cnt:,}\n({cnt/total_users*100:.1f}%)',
                     ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    sign_plot_path = os.path.join(output_dir, "actual_plan_days_sign_distribution.png")
    plt.savefig(sign_plot_path, dpi=150)
    plt.close()
    logger.info(f"  그래프 저장: {sign_plot_path}")

    # 2. 음수/0/양수 분포 (유저 기준 - 해당 카테고리만 가진 유저)
    logger.info(f"\n[2] actual_plan_days 부호별 분포 (유저 기준 - 해당 부호만 가진 유저)")
    user_sign_dist = con.execute(f"""
        WITH user_signs AS (
            SELECT
                user_id,
                SUM(CASE WHEN actual_plan_days < 0 THEN 1 ELSE 0 END) AS neg_cnt,
                SUM(CASE WHEN actual_plan_days = 0 THEN 1 ELSE 0 END) AS zero_cnt,
                SUM(CASE WHEN actual_plan_days > 0 THEN 1 ELSE 0 END) AS pos_cnt
            FROM {seq_table}
            GROUP BY user_id
        )
        SELECT
            CASE
                WHEN neg_cnt > 0 AND zero_cnt = 0 AND pos_cnt = 0 THEN 'only_negative'
                WHEN neg_cnt = 0 AND zero_cnt > 0 AND pos_cnt = 0 THEN 'only_zero'
                WHEN neg_cnt = 0 AND zero_cnt = 0 AND pos_cnt > 0 THEN 'only_positive'
                WHEN neg_cnt > 0 AND pos_cnt > 0 THEN 'mixed_neg_pos'
                WHEN zero_cnt > 0 AND pos_cnt > 0 AND neg_cnt = 0 THEN 'mixed_zero_pos'
                WHEN neg_cnt > 0 AND zero_cnt > 0 AND pos_cnt = 0 THEN 'mixed_neg_zero'
                ELSE 'mixed_all'
            END AS user_category,
            COUNT(*) AS user_count
        FROM user_signs
        GROUP BY user_category
        ORDER BY user_count DESC
    """).fetchall()

    user_categories = []
    user_counts = []
    for category, user_cnt in user_sign_dist:
        user_categories.append(category)
        user_counts.append(user_cnt)
        logger.info(f"  {category}: {user_cnt:,} 유저 ({user_cnt/total_users*100:.2f}%)")

    # 유저 부호 조합 그래프
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(user_categories[::-1], user_counts[::-1], color='#3498db')
    ax.set_title('User Distribution by actual_plan_days Sign Combination')
    ax.set_xlabel('User Count')
    ax.set_ylabel('Category')
    for bar, cnt in zip(bars, user_counts[::-1]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                f' {cnt:,} ({cnt/total_users*100:.1f}%)',
                ha='left', va='center', fontsize=9)
    plt.tight_layout()
    user_sign_plot_path = os.path.join(output_dir, "actual_plan_days_user_sign_combination.png")
    plt.savefig(user_sign_plot_path, dpi=150)
    plt.close()
    logger.info(f"  그래프 저장: {user_sign_plot_path}")

    # 3. actual_plan_days 값 분포 (구간별) - 행 및 유저 포함
    logger.info(f"\n[3] actual_plan_days 값 분포 (구간별)")
    bucket_dist = con.execute(f"""
        SELECT
            CASE
                WHEN actual_plan_days < -30 THEN '< -30'
                WHEN actual_plan_days < -7 THEN '[-30, -7)'
                WHEN actual_plan_days < -1 THEN '[-7, -1)'
                WHEN actual_plan_days < 0 THEN '[-1, 0)'
                WHEN actual_plan_days = 0 THEN '0'
                WHEN actual_plan_days <= 7 THEN '(0, 7]'
                WHEN actual_plan_days <= 30 THEN '(7, 30]'
                WHEN actual_plan_days <= 31 THEN '(30, 31]'
                WHEN actual_plan_days <= 60 THEN '(31, 60]'
                WHEN actual_plan_days <= 90 THEN '(60, 90]'
                WHEN actual_plan_days <= 180 THEN '(90, 180]'
                WHEN actual_plan_days <= 365 THEN '(180, 365]'
                ELSE '> 365'
            END AS bucket,
            COUNT(*) AS row_count,
            COUNT(DISTINCT user_id) AS user_count
        FROM {seq_table}
        GROUP BY bucket
        ORDER BY
            CASE bucket
                WHEN '< -30' THEN 1
                WHEN '[-30, -7)' THEN 2
                WHEN '[-7, -1)' THEN 3
                WHEN '[-1, 0)' THEN 4
                WHEN '0' THEN 5
                WHEN '(0, 7]' THEN 6
                WHEN '(7, 30]' THEN 7
                WHEN '(30, 31]' THEN 8
                WHEN '(31, 60]' THEN 9
                WHEN '(60, 90]' THEN 10
                WHEN '(90, 180]' THEN 11
                WHEN '(180, 365]' THEN 12
                ELSE 13
            END
    """).fetchall()

    buckets = []
    bucket_row_counts = []
    bucket_user_counts = []
    for bucket, row_cnt, user_cnt in bucket_dist:
        buckets.append(bucket)
        bucket_row_counts.append(row_cnt)
        bucket_user_counts.append(user_cnt)
        logger.info(f"  {bucket:>12}: {row_cnt:,} 행 ({row_cnt/total_rows*100:.2f}%), {user_cnt:,} 유저 ({user_cnt/total_users*100:.2f}%)")

    # 구간별 분포 그래프
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 행 기준
    colors = ['#e74c3c' if '<' in b or b.startswith('[-') else '#f39c12' if b == '0' else '#27ae60' for b in buckets]
    bars1 = axes[0].bar(buckets, bucket_row_counts, color=colors)
    axes[0].set_title('actual_plan_days Bucket Distribution (Rows)')
    axes[0].set_xlabel('Bucket')
    axes[0].set_ylabel('Row Count')
    axes[0].set_ylim(bottom=0)
    axes[0].tick_params(axis='x', rotation=45)
    for bar, cnt in zip(bars1, bucket_row_counts):
        if cnt > 0:
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                         f'{cnt:,}\n({cnt/total_rows*100:.1f}%)',
                         ha='center', va='bottom', fontsize=8)

    # 유저 기준
    bars2 = axes[1].bar(buckets, bucket_user_counts, color=colors)
    axes[1].set_title('actual_plan_days Bucket Distribution (Unique Users)')
    axes[1].set_xlabel('Bucket')
    axes[1].set_ylabel('User Count')
    axes[1].set_ylim(bottom=0)
    axes[1].tick_params(axis='x', rotation=45)
    for bar, cnt in zip(bars2, bucket_user_counts):
        if cnt > 0:
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                         f'{cnt:,}\n{cnt/total_users*100:.1f}%',
                         ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    bucket_plot_path = os.path.join(output_dir, "actual_plan_days_bucket_distribution.png")
    plt.savefig(bucket_plot_path, dpi=150)
    plt.close()
    logger.info(f"  그래프 저장: {bucket_plot_path}")

    # 4. payment_plan_days vs actual_plan_days 비교
    logger.info(f"\n[4] payment_plan_days vs actual_plan_days 비교")
    comparison = con.execute(f"""
        SELECT
            CASE
                WHEN actual_plan_days < payment_plan_days THEN 'actual < payment'
                WHEN actual_plan_days = payment_plan_days THEN 'actual = payment'
                ELSE 'actual > payment'
            END AS comparison,
            COUNT(*) AS row_count,
            COUNT(DISTINCT user_id) AS user_count
        FROM {seq_table}
        GROUP BY comparison
        ORDER BY
            CASE comparison
                WHEN 'actual < payment' THEN 1
                WHEN 'actual = payment' THEN 2
                ELSE 3
            END
    """).fetchall()

    comp_labels = []
    comp_row_counts = []
    comp_user_counts = []
    for comp, row_cnt, user_cnt in comparison:
        comp_labels.append(comp)
        comp_row_counts.append(row_cnt)
        comp_user_counts.append(user_cnt)
        logger.info(f"  {comp}: {row_cnt:,} 행 ({row_cnt/total_rows*100:.2f}%), {user_cnt:,} 유저 ({user_cnt/total_users*100:.2f}%)")

    # payment vs actual 비교 그래프
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#e74c3c', '#3498db', '#27ae60']

    bars1 = axes[0].bar(comp_labels, comp_row_counts, color=colors)
    axes[0].set_title('payment_plan_days vs actual_plan_days (Rows)')
    axes[0].set_xlabel('Comparison')
    axes[0].set_ylabel('Row Count')
    axes[0].set_ylim(bottom=0)
    for bar, cnt in zip(bars1, comp_row_counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{cnt:,}\n({cnt/total_rows*100:.1f}%)',
                     ha='center', va='bottom', fontsize=9)

    bars2 = axes[1].bar(comp_labels, comp_user_counts, color=colors)
    axes[1].set_title('payment_plan_days vs actual_plan_days (Users)')
    axes[1].set_xlabel('Comparison')
    axes[1].set_ylabel('User Count')
    axes[1].set_ylim(bottom=0)
    for bar, cnt in zip(bars2, comp_user_counts):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                     f'{cnt:,}\n({cnt/total_users*100:.1f}%)',
                     ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    comp_plot_path = os.path.join(output_dir, "actual_vs_payment_plan_days.png")
    plt.savefig(comp_plot_path, dpi=150)
    plt.close()
    logger.info(f"  그래프 저장: {comp_plot_path}")

    # 5. payment_plan_days별 actual_plan_days 평균
    logger.info(f"\n[5] payment_plan_days별 actual_plan_days 평균")
    plan_avg = con.execute(f"""
        SELECT
            payment_plan_days,
            COUNT(*) AS row_count,
            COUNT(DISTINCT user_id) AS user_count,
            ROUND(AVG(actual_plan_days), 1) AS avg_actual,
            ROUND(AVG(actual_plan_days - payment_plan_days), 1) AS avg_diff
        FROM {seq_table}
        GROUP BY payment_plan_days
        ORDER BY row_count DESC
        LIMIT 10
    """).fetchall()

    plan_labels = []
    plan_row_counts = []
    plan_user_counts = []
    plan_avg_diffs = []
    for plan_days, row_cnt, user_cnt, avg_actual, avg_diff in plan_avg:
        plan_labels.append(str(plan_days))
        plan_row_counts.append(row_cnt)
        plan_user_counts.append(user_cnt)
        plan_avg_diffs.append(avg_diff)
        logger.info(f"  plan={plan_days}: {row_cnt:,} 행, {user_cnt:,} 유저, avg_actual={avg_actual}, diff={avg_diff:+.1f}")

    # payment_plan_days별 분포 그래프
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bars1 = axes[0].bar(plan_labels, plan_row_counts, color='#3498db')
    axes[0].set_title('Transactions by payment_plan_days (Top 10)')
    axes[0].set_xlabel('payment_plan_days')
    axes[0].set_ylabel('Row Count')
    axes[0].set_ylim(bottom=0)
    axes[0].tick_params(axis='x', rotation=45)

    # 평균 차이 그래프
    colors = ['#e74c3c' if d < 0 else '#27ae60' for d in plan_avg_diffs]
    bars2 = axes[1].bar(plan_labels, plan_avg_diffs, color=colors)
    axes[1].set_title('Avg Difference (actual - payment) by payment_plan_days')
    axes[1].set_xlabel('payment_plan_days')
    axes[1].set_ylabel('Avg Difference (days)')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].tick_params(axis='x', rotation=45)
    for bar, diff in zip(bars2, plan_avg_diffs):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + (0.5 if diff >= 0 else -1.5),
                     f'{diff:+.1f}',
                     ha='center', va='bottom' if diff >= 0 else 'top', fontsize=9)

    plt.tight_layout()
    plan_plot_path = os.path.join(output_dir, "actual_plan_days_by_payment_plan.png")
    plt.savefig(plan_plot_path, dpi=150)
    plt.close()
    logger.info(f"  그래프 저장: {plan_plot_path}")

    # 6. is_cancel별 actual_plan_days 분포
    logger.info(f"\n[6] is_cancel별 actual_plan_days 분포")
    cancel_dist = con.execute(f"""
        SELECT
            is_cancel,
            COUNT(*) AS row_count,
            COUNT(DISTINCT user_id) AS user_count,
            SUM(CASE WHEN actual_plan_days < 0 THEN 1 ELSE 0 END) AS neg_cnt,
            SUM(CASE WHEN actual_plan_days = 0 THEN 1 ELSE 0 END) AS zero_cnt,
            SUM(CASE WHEN actual_plan_days > 0 THEN 1 ELSE 0 END) AS pos_cnt,
            ROUND(AVG(actual_plan_days), 1) AS avg_actual
        FROM {seq_table}
        GROUP BY is_cancel
        ORDER BY is_cancel
    """).fetchall()

    cancel_data = []
    for is_cancel, row_cnt, user_cnt, neg, zero, pos, avg in cancel_dist:
        cancel_data.append({
            'is_cancel': is_cancel,
            'row_count': row_cnt,
            'user_count': user_cnt,
            'neg': neg, 'zero': zero, 'pos': pos,
            'avg': avg
        })
        logger.info(f"  is_cancel={is_cancel}: {row_cnt:,} 행, {user_cnt:,} 유저")
        logger.info(f"    neg={neg:,} ({neg/row_cnt*100:.1f}%), zero={zero:,} ({zero/row_cnt*100:.1f}%), pos={pos:,} ({pos/row_cnt*100:.1f}%)")
        logger.info(f"    avg_actual={avg}")

    # is_cancel별 분포 그래프
    if len(cancel_data) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 행 수 비교
        cancel_labels = [f'is_cancel={d["is_cancel"]}' for d in cancel_data]
        cancel_rows = [d['row_count'] for d in cancel_data]
        cancel_users = [d['user_count'] for d in cancel_data]

        axes[0].bar(cancel_labels, cancel_rows, color=['#27ae60', '#e74c3c'])
        axes[0].set_title('Transactions by is_cancel')
        axes[0].set_ylabel('Row Count')
        axes[0].set_ylim(bottom=0)

        # 부호별 비율 (stacked bar)
        x = range(len(cancel_data))
        width = 0.6
        neg_pcts = [d['neg']/d['row_count']*100 for d in cancel_data]
        zero_pcts = [d['zero']/d['row_count']*100 for d in cancel_data]
        pos_pcts = [d['pos']/d['row_count']*100 for d in cancel_data]

        axes[1].bar(x, neg_pcts, width, label='negative', color='#e74c3c')
        axes[1].bar(x, zero_pcts, width, bottom=neg_pcts, label='zero', color='#f39c12')
        axes[1].bar(x, pos_pcts, width, bottom=[n+z for n, z in zip(neg_pcts, zero_pcts)], label='positive', color='#27ae60')
        axes[1].set_title('actual_plan_days Sign Distribution by is_cancel')
        axes[1].set_ylabel('Percentage (%)')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(cancel_labels)
        axes[1].legend()
        axes[1].set_ylim(0, 100)

        plt.tight_layout()
        cancel_plot_path = os.path.join(output_dir, "actual_plan_days_by_is_cancel.png")
        plt.savefig(cancel_plot_path, dpi=150)
        plt.close()
        logger.info(f"  그래프 저장: {cancel_plot_path}")

    # 7. 음수 actual_plan_days 상세 분석
    neg_count = con.execute(f"SELECT COUNT(*) FROM {seq_table} WHERE actual_plan_days < 0").fetchone()[0]
    if neg_count > 0:
        logger.info(f"\n[7] 음수 actual_plan_days 상세 분석 (상위 10개 패턴)")
        neg_patterns = con.execute(f"""
            SELECT
                payment_plan_days,
                actual_plan_days,
                is_cancel,
                COUNT(*) AS cnt,
                COUNT(DISTINCT user_id) AS user_cnt
            FROM {seq_table}
            WHERE actual_plan_days < 0
            GROUP BY payment_plan_days, actual_plan_days, is_cancel
            ORDER BY cnt DESC
            LIMIT 10
        """).fetchall()

        for plan, actual, cancel, cnt, user_cnt in neg_patterns:
            logger.info(f"  plan={plan}, actual={actual}, is_cancel={cancel}: {cnt:,} 행, {user_cnt:,} 유저")

    con.close()
    logger.info(f"\n=== actual_plan_days 분석 완료 ===\n")


# ============================================================================
# 8.5.2.1. is_cancel=1 트랜잭션의 actual_plan_days 분포 분석
# ============================================================================
def analyze_cancel_actual_plan_days(
    db_path: str,
    seq_table: str = "transactions_seq",
    output_dir: str = "data/analysis",
) -> None:
    """
    is_cancel=1인 트랜잭션의 actual_plan_days 분포를 분석합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        seq_table: 트랜잭션 시퀀스 테이블명 (기본값: transactions_seq)
        output_dir: 그래프 저장 디렉토리 (기본값: data/analysis)
    """
    import matplotlib.pyplot as plt

    logger.info(f"=== is_cancel=1 actual_plan_days 분석 시작 ===")
    logger.info(f"테이블: {seq_table}")
    logger.info(f"출력 디렉토리: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    # 테이블 존재 확인
    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if seq_table not in existing_tables:
        logger.error(f"테이블 {seq_table}이 존재하지 않습니다.")
        con.close()
        return

    # 전체 및 is_cancel=1 통계
    total_stats = con.execute(f"""
        SELECT
            COUNT(*) AS total_rows,
            SUM(CASE WHEN is_cancel = 1 THEN 1 ELSE 0 END) AS cancel_rows,
            COUNT(DISTINCT user_id) AS total_users,
            COUNT(DISTINCT CASE WHEN is_cancel = 1 THEN user_id END) AS cancel_users
        FROM {seq_table}
    """).fetchone()
    total_rows, cancel_rows, total_users, cancel_users = total_stats
    logger.info(f"전체: {total_rows:,} 행, {total_users:,} 유저")
    logger.info(f"is_cancel=1: {cancel_rows:,} 행 ({cancel_rows/total_rows*100:.2f}%), {cancel_users:,} 유저 ({cancel_users/total_users*100:.2f}%)")

    if cancel_rows == 0:
        logger.info("is_cancel=1인 트랜잭션이 없습니다.")
        con.close()
        logger.info(f"=== is_cancel=1 actual_plan_days 분석 완료 ===\n")
        return

    # 1. 부호별 분포
    logger.info(f"\n[1] is_cancel=1 actual_plan_days 부호별 분포")
    sign_dist = con.execute(f"""
        SELECT
            CASE
                WHEN actual_plan_days < 0 THEN 'negative'
                WHEN actual_plan_days = 0 THEN 'zero'
                ELSE 'positive'
            END AS sign_category,
            COUNT(*) AS row_count,
            COUNT(DISTINCT user_id) AS user_count
        FROM {seq_table}
        WHERE is_cancel = 1
        GROUP BY sign_category
        ORDER BY
            CASE sign_category
                WHEN 'negative' THEN 1
                WHEN 'zero' THEN 2
                ELSE 3
            END
    """).fetchall()

    sign_categories = []
    sign_row_counts = []
    sign_user_counts = []
    for category, row_cnt, user_cnt in sign_dist:
        sign_categories.append(category)
        sign_row_counts.append(row_cnt)
        sign_user_counts.append(user_cnt)
        logger.info(f"  {category}: {row_cnt:,} 행 ({row_cnt/cancel_rows*100:.2f}%), {user_cnt:,} 유저 ({user_cnt/cancel_users*100:.2f}%)")

    # 2. 구간별 분포
    logger.info(f"\n[2] is_cancel=1 actual_plan_days 구간별 분포")
    bucket_dist = con.execute(f"""
        SELECT
            CASE
                WHEN actual_plan_days < -31 THEN '< -31'
                WHEN actual_plan_days < -30 THEN '[-31, -30)'
                WHEN actual_plan_days < -7 THEN '[-30, -7)'
                WHEN actual_plan_days < -1 THEN '[-7, -1)'
                WHEN actual_plan_days < 0 THEN '[-1, 0)'
                WHEN actual_plan_days = 0 THEN '0'
                WHEN actual_plan_days <= 7 THEN '(0, 7]'
                WHEN actual_plan_days <= 30 THEN '(7, 30]'
                WHEN actual_plan_days <= 31 THEN '(30, 31]'
                WHEN actual_plan_days <= 90 THEN '(31, 90]'
                WHEN actual_plan_days <= 180 THEN '(90, 180]'
                WHEN actual_plan_days <= 365 THEN '(180, 365]'
                ELSE '> 365'
            END AS bucket,
            COUNT(*) AS row_count,
            COUNT(DISTINCT user_id) AS user_count
        FROM {seq_table}
        WHERE is_cancel = 1
        GROUP BY bucket
        ORDER BY
            CASE bucket
                WHEN '< -31' THEN 1
                WHEN '[-31, -30)' THEN 2
                WHEN '[-30, -7)' THEN 3
                WHEN '[-7, -1)' THEN 4
                WHEN '[-1, 0)' THEN 5
                WHEN '0' THEN 6
                WHEN '(0, 7]' THEN 7
                WHEN '(7, 30]' THEN 8
                WHEN '(30, 31]' THEN 9
                WHEN '(31, 90]' THEN 10
                WHEN '(90, 180]' THEN 11
                WHEN '(180, 365]' THEN 12
                ELSE 13
            END
    """).fetchall()

    buckets = []
    bucket_row_counts = []
    bucket_user_counts = []
    for bucket, row_cnt, user_cnt in bucket_dist:
        buckets.append(bucket)
        bucket_row_counts.append(row_cnt)
        bucket_user_counts.append(user_cnt)
        logger.info(f"  {bucket:>15}: {row_cnt:,} 행 ({row_cnt/cancel_rows*100:.2f}%), {user_cnt:,} 유저 ({user_cnt/cancel_users*100:.2f}%)")

    # 3. 통계 요약
    logger.info(f"\n[3] is_cancel=1 actual_plan_days 통계 요약")
    stats = con.execute(f"""
        SELECT
            MIN(actual_plan_days) AS min_val,
            MAX(actual_plan_days) AS max_val,
            ROUND(AVG(actual_plan_days), 1) AS avg_val,
            ROUND(MEDIAN(actual_plan_days), 1) AS median_val,
            ROUND(STDDEV(actual_plan_days), 1) AS std_val
        FROM {seq_table}
        WHERE is_cancel = 1
    """).fetchone()
    logger.info(f"  min={stats[0]}, max={stats[1]}, avg={stats[2]}, median={stats[3]}, std={stats[4]}")

    # 4. payment_plan_days별 분포
    logger.info(f"\n[4] is_cancel=1 payment_plan_days별 actual_plan_days 통계")
    plan_stats = con.execute(f"""
        SELECT
            payment_plan_days,
            COUNT(*) AS row_count,
            COUNT(DISTINCT user_id) AS user_count,
            ROUND(AVG(actual_plan_days), 1) AS avg_actual,
            ROUND(MEDIAN(actual_plan_days), 1) AS median_actual,
            MIN(actual_plan_days) AS min_actual,
            MAX(actual_plan_days) AS max_actual
        FROM {seq_table}
        WHERE is_cancel = 1
        GROUP BY payment_plan_days
        ORDER BY row_count DESC
        LIMIT 10
    """).fetchall()

    for plan, row_cnt, user_cnt, avg_val, med_val, min_val, max_val in plan_stats:
        logger.info(f"  plan={plan}: {row_cnt:,} 행, {user_cnt:,} 유저, avg={avg_val}, median={med_val}, range=[{min_val}, {max_val}]")

    # 그래프 생성
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 부호별 분포 (행)
    colors_sign = ['#e74c3c' if c == 'negative' else '#f39c12' if c == 'zero' else '#27ae60' for c in sign_categories]
    bars1 = axes[0, 0].bar(sign_categories, sign_row_counts, color=colors_sign)
    axes[0, 0].set_title('is_cancel=1: Sign Distribution (Rows)')
    axes[0, 0].set_ylabel('Row Count')
    axes[0, 0].set_ylim(bottom=0)
    for bar, cnt in zip(bars1, sign_row_counts):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{cnt:,}\n({cnt/cancel_rows*100:.1f}%)',
                        ha='center', va='bottom', fontsize=9)

    # 부호별 분포 (유저)
    bars2 = axes[0, 1].bar(sign_categories, sign_user_counts, color=colors_sign)
    axes[0, 1].set_title('is_cancel=1: Sign Distribution (Users)')
    axes[0, 1].set_ylabel('User Count')
    axes[0, 1].set_ylim(bottom=0)
    for bar, cnt in zip(bars2, sign_user_counts):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{cnt:,}\n({cnt/cancel_users*100:.1f}%)',
                        ha='center', va='bottom', fontsize=9)

    # 구간별 분포 (행)
    colors_bucket = ['#e74c3c' if '<' in b or b.startswith('[-') or b.startswith('[') else '#f39c12' if b == '0' else '#27ae60' for b in buckets]
    bars3 = axes[1, 0].bar(range(len(buckets)), bucket_row_counts, color=colors_bucket)
    axes[1, 0].set_title('is_cancel=1: Bucket Distribution (Rows)')
    axes[1, 0].set_ylabel('Row Count')
    axes[1, 0].set_xticks(range(len(buckets)))
    axes[1, 0].set_xticklabels(buckets, rotation=45, ha='right')
    axes[1, 0].set_ylim(bottom=0)

    # 구간별 분포 (유저)
    bars4 = axes[1, 1].bar(range(len(buckets)), bucket_user_counts, color=colors_bucket)
    axes[1, 1].set_title('is_cancel=1: Bucket Distribution (Users)')
    axes[1, 1].set_ylabel('User Count')
    axes[1, 1].set_xticks(range(len(buckets)))
    axes[1, 1].set_xticklabels(buckets, rotation=45, ha='right')
    axes[1, 1].set_ylim(bottom=0)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "cancel_actual_plan_days_distribution.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"\n그래프 저장: {plot_path}")

    con.close()
    logger.info(f"=== is_cancel=1 actual_plan_days 분석 완료 ===\n")


# ============================================================================
# 8.5.3. actual_plan_days 범위 밖 유저 제외
# ============================================================================
def exclude_out_of_range_actual_plan_days_users(
    db_path: str,
    seq_table: str = "transactions_seq",
    target_tables: list[str] = None,
    non_cancel_min: int = 1,
    non_cancel_max: int = 410,
    cancel_min: int = -31,
    cancel_max: int = 410,
) -> None:
    """
    actual_plan_days가 지정된 범위를 벗어나는 트랜잭션을 가진 유저를 모든 _merge 테이블에서 제외합니다.

    is_cancel 여부에 따라 다른 범위를 적용합니다:
    - is_cancel=0: [non_cancel_min, non_cancel_max] 범위 적용
    - is_cancel=1: [cancel_min, cancel_max] 범위 적용

    Args:
        db_path: DuckDB 데이터베이스 경로
        seq_table: 트랜잭션 시퀀스 테이블명 (기본값: transactions_seq)
        target_tables: 필터링을 적용할 대상 테이블 이름 리스트
                       기본값: ["train_merge", "transactions_merge", "user_logs_merge", "members_merge"]
        non_cancel_min: is_cancel=0일 때 최소 actual_plan_days (포함, 기본값: 1)
        non_cancel_max: is_cancel=0일 때 최대 actual_plan_days (포함, 기본값: 410)
        cancel_min: is_cancel=1일 때 최소 actual_plan_days (포함, 기본값: -31)
        cancel_max: is_cancel=1일 때 최대 actual_plan_days (포함, 기본값: 410)
    """
    if target_tables is None:
        target_tables = ["train_merge", "transactions_merge", "user_logs_merge", "members_merge"]

    logger.info(f"=== actual_plan_days 범위 밖 유저 제외 시작 ===")
    logger.info(f"시퀀스 테이블: {seq_table}")
    logger.info(f"is_cancel=0 범위: [{non_cancel_min}, {non_cancel_max}]")
    logger.info(f"is_cancel=1 범위: [{cancel_min}, {cancel_max}]")
    logger.info(f"대상 테이블: {target_tables}")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    # 시퀀스 테이블 존재 확인
    if seq_table not in existing_tables:
        logger.error(f"{seq_table} 테이블이 존재하지 않습니다.")
        con.close()
        return

    # 시퀀스 테이블의 user_id 컬럼 확인
    seq_cols = [row[0] for row in con.execute(f"DESCRIBE {seq_table}").fetchall()]
    if "user_id" in seq_cols:
        id_col = "user_id"
    elif "msno" in seq_cols:
        id_col = "msno"
    else:
        logger.error(f"{seq_table}에 user_id/msno 컬럼이 없습니다.")
        con.close()
        return

    # 전체 유저 수
    total_users = con.execute(f"""
        SELECT COUNT(DISTINCT {id_col}) FROM {seq_table}
    """).fetchone()[0]
    logger.info(f"시퀀스 테이블 전체 유저 수: {total_users:,}")

    # 범위 밖 actual_plan_days를 가진 유저 추출 (is_cancel별 다른 조건)
    con.execute(f"""
        CREATE OR REPLACE TABLE _out_of_range_actual_plan_users_temp AS
        SELECT DISTINCT {id_col}
        FROM {seq_table}
        WHERE
            -- is_cancel=0: [{non_cancel_min}, {non_cancel_max}] 범위 밖
            (is_cancel = 0 AND (actual_plan_days < {non_cancel_min} OR actual_plan_days > {non_cancel_max}))
            OR
            -- is_cancel=1: [{cancel_min}, {cancel_max}] 범위 밖
            (is_cancel = 1 AND (actual_plan_days < {cancel_min} OR actual_plan_days > {cancel_max}));
    """)

    out_of_range_count = con.execute("SELECT COUNT(*) FROM _out_of_range_actual_plan_users_temp").fetchone()[0]
    logger.info(f"범위 밖 유저 수: {out_of_range_count:,} ({out_of_range_count/total_users*100:.2f}%)")

    # 범위 밖 상세 통계 (is_cancel별)
    out_of_range_stats = con.execute(f"""
        SELECT
            is_cancel,
            SUM(CASE WHEN is_cancel = 0 AND actual_plan_days < {non_cancel_min} THEN 1
                     WHEN is_cancel = 1 AND actual_plan_days < {cancel_min} THEN 1
                     ELSE 0 END) AS below_min,
            SUM(CASE WHEN is_cancel = 0 AND actual_plan_days > {non_cancel_max} THEN 1
                     WHEN is_cancel = 1 AND actual_plan_days > {cancel_max} THEN 1
                     ELSE 0 END) AS above_max,
            COUNT(*) AS total_out
        FROM {seq_table}
        WHERE
            (is_cancel = 0 AND (actual_plan_days < {non_cancel_min} OR actual_plan_days > {non_cancel_max}))
            OR
            (is_cancel = 1 AND (actual_plan_days < {cancel_min} OR actual_plan_days > {cancel_max}))
        GROUP BY is_cancel
        ORDER BY is_cancel
    """).fetchall()

    for is_cancel, below, above, total in out_of_range_stats:
        if is_cancel == 0:
            logger.info(f"  is_cancel=0: {total:,} 행 (< {non_cancel_min}: {below:,}, > {non_cancel_max}: {above:,})")
        else:
            logger.info(f"  is_cancel=1: {total:,} 행 (< {cancel_min}: {below:,}, > {cancel_max}: {above:,})")

    if out_of_range_count == 0:
        logger.info("제외할 유저가 없습니다.")
        con.execute("DROP TABLE IF EXISTS _out_of_range_actual_plan_users_temp;")
        con.close()
        return

    # 대상 테이블에서 범위 밖 유저 제외
    for t in tqdm(target_tables, desc="actual_plan_days 범위 밖 유저 제외"):
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

        # 범위 밖 유저 제외 (NOT IN)
        con.execute(f"""
            CREATE OR REPLACE TABLE {t}_filtered AS
            SELECT *
            FROM {t}
            WHERE {target_col} NOT IN (SELECT {id_col} FROM _out_of_range_actual_plan_users_temp);
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
    con.execute("DROP TABLE IF EXISTS _out_of_range_actual_plan_users_temp;")

    con.close()
    logger.info(f"=== actual_plan_days 범위 밖 유저 제외 완료 ===\n")


# ============================================================================
# 8.5.3.1. plan_days_diff 컬럼 추가
# ============================================================================
def add_plan_days_diff(
    db_path: str,
    seq_table: str = "transactions_seq",
) -> None:
    """
    transactions_seq 테이블에 plan_days_diff 컬럼을 추가합니다.

    plan_days_diff = payment_plan_days - actual_plan_days
    - 양수: 결제한 플랜보다 실제 연장 기간이 짧음 (손해)
    - 0: 결제한 플랜과 실제 연장 기간이 동일
    - 음수: 결제한 플랜보다 실제 연장 기간이 김 (이득)

    Args:
        db_path: DuckDB 데이터베이스 경로
        seq_table: 트랜잭션 시퀀스 테이블명 (기본값: transactions_seq)
    """
    logger.info(f"=== {seq_table}에 plan_days_diff 추가 시작 ===")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    # 테이블 존재 확인
    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if seq_table not in existing_tables:
        logger.error(f"테이블 {seq_table}이 존재하지 않습니다.")
        con.close()
        return

    # 이미 컬럼이 존재하는지 확인
    columns = [row[0] for row in con.execute(f"DESCRIBE {seq_table}").fetchall()]
    if "plan_days_diff" in columns:
        logger.info("plan_days_diff 컬럼이 이미 존재합니다. 재계산합니다.")
        con.execute(f"ALTER TABLE {seq_table} DROP COLUMN plan_days_diff")

    # plan_days_diff 컬럼 추가
    logger.info("plan_days_diff 계산 중...")
    con.execute(f"""
        ALTER TABLE {seq_table}
        ADD COLUMN plan_days_diff BIGINT;
    """)

    con.execute(f"""
        UPDATE {seq_table}
        SET plan_days_diff = payment_plan_days - actual_plan_days;
    """)

    # 결과 통계
    stats = con.execute(f"""
        SELECT
            COUNT(*) AS total_rows,
            SUM(CASE WHEN plan_days_diff > 0 THEN 1 ELSE 0 END) AS positive_cnt,
            SUM(CASE WHEN plan_days_diff = 0 THEN 1 ELSE 0 END) AS zero_cnt,
            SUM(CASE WHEN plan_days_diff < 0 THEN 1 ELSE 0 END) AS negative_cnt,
            ROUND(AVG(plan_days_diff), 2) AS avg_diff,
            MIN(plan_days_diff) AS min_diff,
            MAX(plan_days_diff) AS max_diff
        FROM {seq_table}
    """).fetchone()

    total, pos, zero, neg, avg, min_val, max_val = stats
    logger.info(f"추가 완료: {total:,} 행")
    logger.info(f"plan_days_diff 통계:")
    logger.info(f"  양수 (payment > actual): {pos:,} ({pos/total*100:.2f}%)")
    logger.info(f"  0 (payment = actual): {zero:,} ({zero/total*100:.2f}%)")
    logger.info(f"  음수 (payment < actual): {neg:,} ({neg/total*100:.2f}%)")
    logger.info(f"  평균: {avg}, 범위: [{min_val}, {max_val}]")

    # is_cancel별 통계
    cancel_stats = con.execute(f"""
        SELECT
            is_cancel,
            COUNT(*) AS cnt,
            ROUND(AVG(plan_days_diff), 2) AS avg_diff,
            MIN(plan_days_diff) AS min_diff,
            MAX(plan_days_diff) AS max_diff
        FROM {seq_table}
        GROUP BY is_cancel
        ORDER BY is_cancel
    """).fetchall()

    logger.info(f"is_cancel별 통계:")
    for is_cancel, cnt, avg, min_val, max_val in cancel_stats:
        logger.info(f"  is_cancel={is_cancel}: {cnt:,} 행, avg={avg}, range=[{min_val}, {max_val}]")

    con.close()
    logger.info(f"=== {seq_table}에 plan_days_diff 추가 완료 ===\n")


# ============================================================================
# 8.5.3.2. is_cancel 트랜잭션의 actual_plan_days 분포 분석
# ============================================================================
def analyze_cancel_actual_plan_days_distribution(
    db_path: str,
    seq_table: str = "transactions_seq",
    output_dir: str = "data/analysis",
) -> None:
    """
    is_cancel=1인 트랜잭션의 actual_plan_days 분포를 상세 분석합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        seq_table: 트랜잭션 시퀀스 테이블명 (기본값: transactions_seq)
        output_dir: 그래프 저장 디렉토리 (기본값: data/analysis)
    """
    import matplotlib.pyplot as plt

    logger.info(f"=== is_cancel 트랜잭션 actual_plan_days 분포 분석 시작 ===")
    logger.info(f"테이블: {seq_table}")
    logger.info(f"출력 디렉토리: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    # 테이블 존재 확인
    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if seq_table not in existing_tables:
        logger.error(f"테이블 {seq_table}이 존재하지 않습니다.")
        con.close()
        return

    # 전체 및 is_cancel=1 통계
    total_stats = con.execute(f"""
        SELECT
            COUNT(*) AS total_rows,
            SUM(CASE WHEN is_cancel = 1 THEN 1 ELSE 0 END) AS cancel_rows,
            COUNT(DISTINCT user_id) AS total_users,
            COUNT(DISTINCT CASE WHEN is_cancel = 1 THEN user_id END) AS cancel_users
        FROM {seq_table}
    """).fetchone()
    total_rows, cancel_rows, total_users, cancel_users = total_stats
    logger.info(f"전체: {total_rows:,} 행, {total_users:,} 유저")
    logger.info(f"is_cancel=1: {cancel_rows:,} 행 ({cancel_rows/total_rows*100:.2f}%), {cancel_users:,} 유저 ({cancel_users/total_users*100:.2f}%)")

    if cancel_rows == 0:
        logger.info("is_cancel=1인 트랜잭션이 없습니다.")
        con.close()
        logger.info(f"=== is_cancel 트랜잭션 actual_plan_days 분포 분석 완료 ===\n")
        return

    # 1. 기본 통계
    logger.info(f"\n[1] is_cancel=1 actual_plan_days 기본 통계")
    basic_stats = con.execute(f"""
        SELECT
            MIN(actual_plan_days) AS min_val,
            MAX(actual_plan_days) AS max_val,
            ROUND(AVG(actual_plan_days), 2) AS avg_val,
            ROUND(MEDIAN(actual_plan_days), 2) AS median_val,
            ROUND(STDDEV(actual_plan_days), 2) AS std_val,
            MODE(actual_plan_days) AS mode_val
        FROM {seq_table}
        WHERE is_cancel = 1
    """).fetchone()
    logger.info(f"  min={basic_stats[0]}, max={basic_stats[1]}")
    logger.info(f"  avg={basic_stats[2]}, median={basic_stats[3]}, std={basic_stats[4]}")
    logger.info(f"  mode={basic_stats[5]}")

    # 2. 값별 분포 (상위 20개)
    logger.info(f"\n[2] is_cancel=1 actual_plan_days 값별 분포 (상위 20개)")
    value_dist = con.execute(f"""
        SELECT
            actual_plan_days,
            COUNT(*) AS row_count,
            COUNT(DISTINCT user_id) AS user_count
        FROM {seq_table}
        WHERE is_cancel = 1
        GROUP BY actual_plan_days
        ORDER BY row_count DESC
        LIMIT 20
    """).fetchall()

    values = []
    value_row_counts = []
    value_user_counts = []
    for val, row_cnt, user_cnt in value_dist:
        values.append(val)
        value_row_counts.append(row_cnt)
        value_user_counts.append(user_cnt)
        logger.info(f"  {val}: {row_cnt:,} 행 ({row_cnt/cancel_rows*100:.2f}%), {user_cnt:,} 유저")

    # 3. 구간별 분포
    logger.info(f"\n[3] is_cancel=1 actual_plan_days 구간별 분포")
    bucket_dist = con.execute(f"""
        SELECT
            CASE
                WHEN actual_plan_days < -365 THEN '< -365'
                WHEN actual_plan_days < -180 THEN '[-365, -180)'
                WHEN actual_plan_days < -90 THEN '[-180, -90)'
                WHEN actual_plan_days < -31 THEN '[-90, -31)'
                WHEN actual_plan_days < 0 THEN '[-31, 0)'
                WHEN actual_plan_days = 0 THEN '0'
                WHEN actual_plan_days <= 30 THEN '(0, 30]'
                WHEN actual_plan_days <= 90 THEN '(30, 90]'
                WHEN actual_plan_days <= 180 THEN '(90, 180]'
                WHEN actual_plan_days <= 365 THEN '(180, 365]'
                ELSE '> 365'
            END AS bucket,
            COUNT(*) AS row_count,
            COUNT(DISTINCT user_id) AS user_count
        FROM {seq_table}
        WHERE is_cancel = 1
        GROUP BY bucket
        ORDER BY
            CASE bucket
                WHEN '< -365' THEN 1
                WHEN '[-365, -180)' THEN 2
                WHEN '[-180, -90)' THEN 3
                WHEN '[-90, -31)' THEN 4
                WHEN '[-31, 0)' THEN 5
                WHEN '0' THEN 6
                WHEN '(0, 30]' THEN 7
                WHEN '(30, 90]' THEN 8
                WHEN '(90, 180]' THEN 9
                WHEN '(180, 365]' THEN 10
                ELSE 11
            END
    """).fetchall()

    buckets = []
    bucket_row_counts = []
    bucket_user_counts = []
    for bucket, row_cnt, user_cnt in bucket_dist:
        buckets.append(bucket)
        bucket_row_counts.append(row_cnt)
        bucket_user_counts.append(user_cnt)
        logger.info(f"  {bucket:>15}: {row_cnt:,} 행 ({row_cnt/cancel_rows*100:.2f}%), {user_cnt:,} 유저 ({user_cnt/cancel_users*100:.2f}%)")

    # 4. payment_plan_days별 분포
    logger.info(f"\n[4] is_cancel=1 payment_plan_days별 actual_plan_days 통계")
    plan_stats = con.execute(f"""
        SELECT
            payment_plan_days,
            COUNT(*) AS row_count,
            ROUND(AVG(actual_plan_days), 2) AS avg_actual,
            ROUND(MEDIAN(actual_plan_days), 2) AS median_actual,
            MIN(actual_plan_days) AS min_actual,
            MAX(actual_plan_days) AS max_actual
        FROM {seq_table}
        WHERE is_cancel = 1
        GROUP BY payment_plan_days
        ORDER BY row_count DESC
        LIMIT 10
    """).fetchall()

    plan_labels = []
    plan_counts = []
    plan_avgs = []
    for plan, cnt, avg, med, min_val, max_val in plan_stats:
        plan_labels.append(str(plan))
        plan_counts.append(cnt)
        plan_avgs.append(avg if avg else 0)
        logger.info(f"  plan={plan}: {cnt:,} 행, avg={avg}, median={med}, range=[{min_val}, {max_val}]")

    # 그래프 생성
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. 상위 값 분포 (행)
    top_n = min(15, len(values))
    colors_val = ['#e74c3c' if v < 0 else '#f39c12' if v == 0 else '#27ae60' for v in values[:top_n]]
    axes[0, 0].bar([str(v) for v in values[:top_n]], value_row_counts[:top_n], color=colors_val)
    axes[0, 0].set_title(f'is_cancel=1: Top {top_n} actual_plan_days Values (Rows)')
    axes[0, 0].set_xlabel('actual_plan_days')
    axes[0, 0].set_ylabel('Row Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylim(bottom=0)

    # 2. 상위 값 분포 (유저)
    axes[0, 1].bar([str(v) for v in values[:top_n]], value_user_counts[:top_n], color=colors_val)
    axes[0, 1].set_title(f'is_cancel=1: Top {top_n} actual_plan_days Values (Users)')
    axes[0, 1].set_xlabel('actual_plan_days')
    axes[0, 1].set_ylabel('User Count')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_ylim(bottom=0)

    # 3. 구간별 분포 (행)
    colors_bucket = ['#e74c3c' if '<' in b or b.startswith('[-') else '#f39c12' if b == '0' else '#27ae60' for b in buckets]
    axes[1, 0].bar(range(len(buckets)), bucket_row_counts, color=colors_bucket)
    axes[1, 0].set_title('is_cancel=1: Bucket Distribution (Rows)')
    axes[1, 0].set_xlabel('Bucket')
    axes[1, 0].set_ylabel('Row Count')
    axes[1, 0].set_xticks(range(len(buckets)))
    axes[1, 0].set_xticklabels(buckets, rotation=45, ha='right')
    axes[1, 0].set_ylim(bottom=0)
    for i, (cnt, bucket) in enumerate(zip(bucket_row_counts, buckets)):
        if cnt > 0:
            axes[1, 0].text(i, cnt, f'{cnt/cancel_rows*100:.1f}%', ha='center', va='bottom', fontsize=8)

    # 4. 구간별 분포 (유저)
    axes[1, 1].bar(range(len(buckets)), bucket_user_counts, color=colors_bucket)
    axes[1, 1].set_title('is_cancel=1: Bucket Distribution (Users)')
    axes[1, 1].set_xlabel('Bucket')
    axes[1, 1].set_ylabel('User Count')
    axes[1, 1].set_xticks(range(len(buckets)))
    axes[1, 1].set_xticklabels(buckets, rotation=45, ha='right')
    axes[1, 1].set_ylim(bottom=0)
    for i, (cnt, bucket) in enumerate(zip(bucket_user_counts, buckets)):
        if cnt > 0:
            axes[1, 1].text(i, cnt, f'{cnt/cancel_users*100:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "cancel_actual_plan_days_detailed_distribution.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"\n그래프 저장: {plot_path}")

    # 히스토그램 (실제 분포)
    cancel_data = con.execute(f"""
        SELECT actual_plan_days FROM {seq_table} WHERE is_cancel = 1
    """).fetchdf()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 전체 범위 히스토그램
    axes[0].hist(cancel_data['actual_plan_days'], bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    axes[0].set_title('is_cancel=1: actual_plan_days Histogram (Full Range)')
    axes[0].set_xlabel('actual_plan_days')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(x=0, color='red', linestyle='--', label='0')
    axes[0].legend()

    # -50 ~ 50 범위 히스토그램 (상세)
    filtered = cancel_data[(cancel_data['actual_plan_days'] >= -50) & (cancel_data['actual_plan_days'] <= 50)]
    axes[1].hist(filtered['actual_plan_days'], bins=50, color='#9b59b6', edgecolor='black', alpha=0.7)
    axes[1].set_title('is_cancel=1: actual_plan_days Histogram ([-50, 50] Range)')
    axes[1].set_xlabel('actual_plan_days')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(x=0, color='red', linestyle='--', label='0')
    axes[1].legend()

    plt.tight_layout()
    hist_path = os.path.join(output_dir, "cancel_actual_plan_days_histogram.png")
    plt.savefig(hist_path, dpi=150)
    plt.close()
    logger.info(f"그래프 저장: {hist_path}")

    con.close()
    logger.info(f"=== is_cancel 트랜잭션 actual_plan_days 분포 분석 완료 ===\n")


# ============================================================================
# 8.5.3.3. is_cancel 트랜잭션 중 expire_date < prev_transaction_date 분석
# ============================================================================
def analyze_cancel_expire_before_prev_transaction(
    db_path: str,
    seq_table: str = "transactions_seq",
    output_dir: str = "data/analysis",
    show_samples: int | None = None,
) -> None:
    """
    is_cancel=1 트랜잭션 중 membership_expire_date가 이전 트랜잭션의
    transaction_date보다 작은 케이스를 분석합니다.

    이 케이스는 취소로 인해 멤버십 만료일이 이전 결제일 이전으로 되돌아가는
    비정상적인 상황을 나타냅니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        seq_table: 트랜잭션 시퀀스 테이블명 (기본값: transactions_seq)
        output_dir: 그래프 저장 디렉토리 (기본값: data/analysis)
        show_samples: 샘플 케이스 수 (None이면 샘플 출력 안 함)
    """
    import matplotlib.pyplot as plt

    logger.info(f"=== is_cancel expire_date < prev_transaction_date 분석 시작 ===")
    logger.info(f"테이블: {seq_table}")
    logger.info(f"출력 디렉토리: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    # 테이블 존재 확인
    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if seq_table not in existing_tables:
        logger.error(f"테이블 {seq_table}이 존재하지 않습니다.")
        con.close()
        return

    # is_cancel=1 전체 통계
    cancel_stats = con.execute(f"""
        SELECT
            COUNT(*) AS cancel_rows,
            COUNT(DISTINCT user_id) AS cancel_users
        FROM {seq_table}
        WHERE is_cancel = 1
    """).fetchone()
    cancel_rows, cancel_users = cancel_stats
    logger.info(f"is_cancel=1 전체: {cancel_rows:,} 행, {cancel_users:,} 유저")

    if cancel_rows == 0:
        logger.info("is_cancel=1인 트랜잭션이 없습니다.")
        con.close()
        logger.info(f"=== is_cancel expire_date < prev_transaction_date 분석 완료 ===\n")
        return

    # 이전 트랜잭션의 transaction_date를 가져와서 비교
    # membership_expire_date < LAG(transaction_date) 인 케이스 찾기
    anomaly_stats = con.execute(f"""
        WITH with_prev AS (
            SELECT
                *,
                LAG(transaction_date) OVER (
                    PARTITION BY user_id
                    ORDER BY transaction_date, membership_expire_date
                ) AS prev_transaction_date
            FROM {seq_table}
        )
        SELECT
            COUNT(*) AS anomaly_rows,
            COUNT(DISTINCT user_id) AS anomaly_users
        FROM with_prev
        WHERE is_cancel = 1
          AND prev_transaction_date IS NOT NULL
          AND membership_expire_date < prev_transaction_date
    """).fetchone()
    anomaly_rows, anomaly_users = anomaly_stats

    logger.info(f"\n[1] expire_date < prev_transaction_date 케이스 통계")
    logger.info(f"  해당 케이스: {anomaly_rows:,} 행 ({anomaly_rows/cancel_rows*100:.2f}% of is_cancel=1)")
    logger.info(f"  영향 유저: {anomaly_users:,} ({anomaly_users/cancel_users*100:.2f}% of is_cancel=1 users)")

    if anomaly_rows == 0:
        logger.info("해당 케이스가 없습니다.")
        con.close()
        logger.info(f"=== is_cancel expire_date < prev_transaction_date 분석 완료 ===\n")
        return

    # 2. 차이 일수 분포
    logger.info(f"\n[2] 차이 일수 (prev_transaction_date - membership_expire_date) 분포")
    diff_stats = con.execute(f"""
        WITH with_prev AS (
            SELECT
                *,
                LAG(transaction_date) OVER (
                    PARTITION BY user_id
                    ORDER BY transaction_date, membership_expire_date
                ) AS prev_transaction_date
            FROM {seq_table}
        ),
        anomaly_cases AS (
            SELECT
                *,
                CAST(prev_transaction_date - membership_expire_date AS BIGINT) AS days_diff
            FROM with_prev
            WHERE is_cancel = 1
              AND prev_transaction_date IS NOT NULL
              AND membership_expire_date < prev_transaction_date
        )
        SELECT
            MIN(days_diff) AS min_diff,
            MAX(days_diff) AS max_diff,
            ROUND(AVG(days_diff), 2) AS avg_diff,
            ROUND(MEDIAN(days_diff), 2) AS median_diff
        FROM anomaly_cases
    """).fetchone()
    logger.info(f"  min={diff_stats[0]}, max={diff_stats[1]}, avg={diff_stats[2]}, median={diff_stats[3]}")

    # 3. 차이 구간별 분포
    logger.info(f"\n[3] 차이 일수 구간별 분포")
    bucket_dist = con.execute(f"""
        WITH with_prev AS (
            SELECT
                *,
                LAG(transaction_date) OVER (
                    PARTITION BY user_id
                    ORDER BY transaction_date, membership_expire_date
                ) AS prev_transaction_date
            FROM {seq_table}
        ),
        anomaly_cases AS (
            SELECT
                *,
                CAST(prev_transaction_date - membership_expire_date AS BIGINT) AS days_diff
            FROM with_prev
            WHERE is_cancel = 1
              AND prev_transaction_date IS NOT NULL
              AND membership_expire_date < prev_transaction_date
        )
        SELECT
            CASE
                WHEN days_diff <= 7 THEN '1-7일'
                WHEN days_diff <= 30 THEN '8-30일'
                WHEN days_diff <= 90 THEN '31-90일'
                WHEN days_diff <= 180 THEN '91-180일'
                WHEN days_diff <= 365 THEN '181-365일'
                ELSE '365일 초과'
            END AS bucket,
            COUNT(*) AS row_count,
            COUNT(DISTINCT user_id) AS user_count
        FROM anomaly_cases
        GROUP BY bucket
        ORDER BY
            CASE bucket
                WHEN '1-7일' THEN 1
                WHEN '8-30일' THEN 2
                WHEN '31-90일' THEN 3
                WHEN '91-180일' THEN 4
                WHEN '181-365일' THEN 5
                ELSE 6
            END
    """).fetchall()

    buckets = []
    bucket_row_counts = []
    bucket_user_counts = []
    for bucket, row_cnt, user_cnt in bucket_dist:
        buckets.append(bucket)
        bucket_row_counts.append(row_cnt)
        bucket_user_counts.append(user_cnt)
        logger.info(f"  {bucket:>12}: {row_cnt:,} 행 ({row_cnt/anomaly_rows*100:.2f}%), {user_cnt:,} 유저")

    # 4. payment_plan_days별 분포
    logger.info(f"\n[4] payment_plan_days별 분포")
    plan_dist = con.execute(f"""
        WITH with_prev AS (
            SELECT
                *,
                LAG(transaction_date) OVER (
                    PARTITION BY user_id
                    ORDER BY transaction_date, membership_expire_date
                ) AS prev_transaction_date
            FROM {seq_table}
        ),
        anomaly_cases AS (
            SELECT
                *,
                CAST(prev_transaction_date - membership_expire_date AS BIGINT) AS days_diff
            FROM with_prev
            WHERE is_cancel = 1
              AND prev_transaction_date IS NOT NULL
              AND membership_expire_date < prev_transaction_date
        )
        SELECT
            payment_plan_days,
            COUNT(*) AS row_count,
            COUNT(DISTINCT user_id) AS user_count,
            ROUND(AVG(days_diff), 2) AS avg_diff
        FROM anomaly_cases
        GROUP BY payment_plan_days
        ORDER BY row_count DESC
        LIMIT 10
    """).fetchall()

    plan_labels = []
    plan_row_counts = []
    for plan, row_cnt, user_cnt, avg_diff in plan_dist:
        plan_labels.append(str(plan))
        plan_row_counts.append(row_cnt)
        logger.info(f"  plan={plan}: {row_cnt:,} 행, {user_cnt:,} 유저, avg_diff={avg_diff}")

    # 5. 샘플 케이스
    if show_samples is not None and show_samples > 0:
        logger.info(f"\n[5] 샘플 케이스 (상위 {show_samples}개)")
        sample_cases = con.execute(f"""
            WITH with_prev AS (
                SELECT
                    *,
                    LAG(transaction_date) OVER (
                        PARTITION BY user_id
                        ORDER BY transaction_date, membership_expire_date
                    ) AS prev_transaction_date
                FROM {seq_table}
            )
            SELECT
                user_id,
                prev_transaction_date,
                transaction_date,
                membership_expire_date,
                CAST(prev_transaction_date - membership_expire_date AS BIGINT) AS days_diff,
                payment_plan_days,
                actual_plan_days
            FROM with_prev
            WHERE is_cancel = 1
              AND prev_transaction_date IS NOT NULL
              AND membership_expire_date < prev_transaction_date
            ORDER BY days_diff DESC
            LIMIT {show_samples}
        """).fetchdf()
        logger.info(f"\n{sample_cases.to_string()}")

    # 그래프 생성
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 구간별 분포 (행)
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']
    axes[0, 0].bar(buckets, bucket_row_counts, color=colors[:len(buckets)])
    axes[0, 0].set_title('Days Diff Bucket Distribution (Rows)\n(prev_tx_date - expire_date)')
    axes[0, 0].set_xlabel('Days Difference Bucket')
    axes[0, 0].set_ylabel('Row Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylim(bottom=0)
    for i, cnt in enumerate(bucket_row_counts):
        axes[0, 0].text(i, cnt, f'{cnt:,}', ha='center', va='bottom', fontsize=9)

    # 2. 구간별 분포 (유저)
    axes[0, 1].bar(buckets, bucket_user_counts, color=colors[:len(buckets)])
    axes[0, 1].set_title('Days Diff Bucket Distribution (Users)\n(prev_tx_date - expire_date)')
    axes[0, 1].set_xlabel('Days Difference Bucket')
    axes[0, 1].set_ylabel('User Count')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_ylim(bottom=0)
    for i, cnt in enumerate(bucket_user_counts):
        axes[0, 1].text(i, cnt, f'{cnt:,}', ha='center', va='bottom', fontsize=9)

    # 3. payment_plan_days별 분포
    if plan_labels:
        axes[1, 0].bar(plan_labels, plan_row_counts, color='#3498db')
        axes[1, 0].set_title('Distribution by payment_plan_days (Rows)')
        axes[1, 0].set_xlabel('payment_plan_days')
        axes[1, 0].set_ylabel('Row Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_ylim(bottom=0)

    # 4. 히스토그램
    diff_data = con.execute(f"""
        WITH with_prev AS (
            SELECT
                *,
                LAG(transaction_date) OVER (
                    PARTITION BY user_id
                    ORDER BY transaction_date, membership_expire_date
                ) AS prev_transaction_date
            FROM {seq_table}
        )
        SELECT CAST(prev_transaction_date - membership_expire_date AS BIGINT) AS days_diff
        FROM with_prev
        WHERE is_cancel = 1
          AND prev_transaction_date IS NOT NULL
          AND membership_expire_date < prev_transaction_date
    """).fetchdf()

    axes[1, 1].hist(diff_data['days_diff'], bins=30, color='#9b59b6', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Days Diff Histogram\n(prev_tx_date - expire_date)')
    axes[1, 1].set_xlabel('Days Difference')
    axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "cancel_expire_before_prev_transaction.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"\n그래프 저장: {plot_path}")

    con.close()
    logger.info(f"=== is_cancel expire_date < prev_transaction_date 분석 완료 ===\n")


# ============================================================================
# 8.5.3.4. is_cancel expire_date < prev_transaction_date 유저 제외
# ============================================================================
def exclude_cancel_expire_before_prev_transaction_users(
    db_path: str,
    seq_table: str = "transactions_seq",
    target_tables: list[str] = None,
) -> None:
    """
    is_cancel=1 트랜잭션 중 membership_expire_date가 이전 트랜잭션의
    transaction_date보다 작은 케이스를 가진 유저를 모든 _merge 테이블에서 제외합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        seq_table: 트랜잭션 시퀀스 테이블명 (기본값: transactions_seq)
        target_tables: 필터링을 적용할 대상 테이블 이름 리스트
                       기본값: ["train_merge", "transactions_merge", "user_logs_merge", "members_merge"]
    """
    if target_tables is None:
        target_tables = ["train_merge", "transactions_merge", "user_logs_merge", "members_merge"]

    logger.info(f"=== is_cancel expire_date < prev_transaction_date 유저 제외 시작 ===")
    logger.info(f"시퀀스 테이블: {seq_table}")
    logger.info(f"대상 테이블: {target_tables}")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    # 시퀀스 테이블 존재 확인
    if seq_table not in existing_tables:
        logger.error(f"{seq_table} 테이블이 존재하지 않습니다.")
        con.close()
        return

    # 시퀀스 테이블의 user_id 컬럼 확인
    seq_cols = [row[0] for row in con.execute(f"DESCRIBE {seq_table}").fetchall()]
    if "user_id" in seq_cols:
        id_col = "user_id"
    elif "msno" in seq_cols:
        id_col = "msno"
    else:
        logger.error(f"{seq_table}에 user_id/msno 컬럼이 없습니다.")
        con.close()
        return

    # 전체 유저 수
    total_users = con.execute(f"""
        SELECT COUNT(DISTINCT {id_col}) FROM {seq_table}
    """).fetchone()[0]
    logger.info(f"시퀀스 테이블 전체 유저 수: {total_users:,}")

    # 해당 조건에 맞는 유저 추출
    con.execute(f"""
        CREATE OR REPLACE TABLE _cancel_expire_anomaly_users_temp AS
        WITH with_prev AS (
            SELECT
                {id_col},
                membership_expire_date,
                is_cancel,
                LAG(transaction_date) OVER (
                    PARTITION BY {id_col}
                    ORDER BY transaction_date, membership_expire_date
                ) AS prev_transaction_date
            FROM {seq_table}
        )
        SELECT DISTINCT {id_col}
        FROM with_prev
        WHERE is_cancel = 1
          AND prev_transaction_date IS NOT NULL
          AND membership_expire_date < prev_transaction_date;
    """)

    anomaly_count = con.execute("SELECT COUNT(*) FROM _cancel_expire_anomaly_users_temp").fetchone()[0]
    logger.info(f"제외 대상 유저 수: {anomaly_count:,} ({anomaly_count/total_users*100:.2f}%)")

    if anomaly_count == 0:
        logger.info("제외할 유저가 없습니다.")
        con.execute("DROP TABLE IF EXISTS _cancel_expire_anomaly_users_temp;")
        con.close()
        return

    # 대상 테이블에서 해당 유저 제외
    for t in tqdm(target_tables, desc="cancel expire anomaly 유저 제외"):
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

        # 해당 유저 제외 (NOT IN)
        con.execute(f"""
            CREATE OR REPLACE TABLE {t}_filtered AS
            SELECT *
            FROM {t}
            WHERE {target_col} NOT IN (SELECT {id_col} FROM _cancel_expire_anomaly_users_temp);
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
    con.execute("DROP TABLE IF EXISTS _cancel_expire_anomaly_users_temp;")

    con.close()
    logger.info(f"=== is_cancel expire_date < prev_transaction_date 유저 제외 완료 ===\n")


# ============================================================================
# 8.5.3.5. is_cancel expire_date < prev_prev_expire_date 분석
# ============================================================================
def analyze_cancel_expire_before_prev_prev_expire(
    db_path: str,
    seq_table: str = "transactions_seq",
    output_dir: str = "data/analysis",
    show_samples: int | None = None,
) -> None:
    """
    is_cancel=1 트랜잭션 중 membership_expire_date가 이전 두 번째 트랜잭션의
    membership_expire_date보다 작은 케이스를 분석합니다.

    이 케이스는 취소로 인해 멤버십 만료일이 두 번째 이전 트랜잭션의 만료일보다
    더 과거로 되돌아가는 비정상적인 상황을 나타냅니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        seq_table: 트랜잭션 시퀀스 테이블명 (기본값: transactions_seq)
        output_dir: 그래프 저장 디렉토리 (기본값: data/analysis)
        show_samples: 샘플 케이스 수 (None이면 샘플 출력 안 함)
    """
    import matplotlib.pyplot as plt

    logger.info(f"=== is_cancel expire_date < prev_prev_expire_date 분석 시작 ===")
    logger.info(f"테이블: {seq_table}")
    logger.info(f"출력 디렉토리: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    # 테이블 존재 확인
    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if seq_table not in existing_tables:
        logger.error(f"테이블 {seq_table}이 존재하지 않습니다.")
        con.close()
        return

    # is_cancel=1 전체 통계
    cancel_stats = con.execute(f"""
        SELECT
            COUNT(*) AS cancel_rows,
            COUNT(DISTINCT user_id) AS cancel_users
        FROM {seq_table}
        WHERE is_cancel = 1
    """).fetchone()
    cancel_rows, cancel_users = cancel_stats
    logger.info(f"is_cancel=1 전체: {cancel_rows:,} 행, {cancel_users:,} 유저")

    if cancel_rows == 0:
        logger.info("is_cancel=1인 트랜잭션이 없습니다.")
        con.close()
        logger.info(f"=== is_cancel expire_date < prev_prev_expire_date 분석 완료 ===\n")
        return

    # 이전 두 번째 트랜잭션의 membership_expire_date를 가져와서 비교
    # membership_expire_date < LAG(membership_expire_date, 2) 인 케이스 찾기
    anomaly_stats = con.execute(f"""
        WITH with_prev_prev AS (
            SELECT
                *,
                LAG(membership_expire_date, 2) OVER (
                    PARTITION BY user_id
                    ORDER BY transaction_date, membership_expire_date
                ) AS prev_prev_expire_date
            FROM {seq_table}
        )
        SELECT
            COUNT(*) AS anomaly_rows,
            COUNT(DISTINCT user_id) AS anomaly_users
        FROM with_prev_prev
        WHERE is_cancel = 1
          AND prev_prev_expire_date IS NOT NULL
          AND membership_expire_date < prev_prev_expire_date
    """).fetchone()
    anomaly_rows, anomaly_users = anomaly_stats

    logger.info(f"\n[1] expire_date < prev_prev_expire_date 케이스 통계")
    logger.info(f"  해당 케이스: {anomaly_rows:,} 행 ({anomaly_rows/cancel_rows*100:.2f}% of is_cancel=1)")
    logger.info(f"  영향 유저: {anomaly_users:,} ({anomaly_users/cancel_users*100:.2f}% of is_cancel=1 users)")

    if anomaly_rows == 0:
        logger.info("해당 케이스가 없습니다.")
        con.close()
        logger.info(f"=== is_cancel expire_date < prev_prev_expire_date 분석 완료 ===\n")
        return

    # 2. 차이 일수 분포
    logger.info(f"\n[2] 차이 일수 (prev_prev_expire_date - membership_expire_date) 분포")
    diff_stats = con.execute(f"""
        WITH with_prev_prev AS (
            SELECT
                *,
                LAG(membership_expire_date, 2) OVER (
                    PARTITION BY user_id
                    ORDER BY transaction_date, membership_expire_date
                ) AS prev_prev_expire_date
            FROM {seq_table}
        ),
        anomaly_cases AS (
            SELECT
                *,
                CAST(prev_prev_expire_date - membership_expire_date AS BIGINT) AS days_diff
            FROM with_prev_prev
            WHERE is_cancel = 1
              AND prev_prev_expire_date IS NOT NULL
              AND membership_expire_date < prev_prev_expire_date
        )
        SELECT
            MIN(days_diff) AS min_diff,
            MAX(days_diff) AS max_diff,
            ROUND(AVG(days_diff), 2) AS avg_diff,
            ROUND(MEDIAN(days_diff), 2) AS median_diff
        FROM anomaly_cases
    """).fetchone()
    logger.info(f"  min={diff_stats[0]}, max={diff_stats[1]}, avg={diff_stats[2]}, median={diff_stats[3]}")

    # 3. 차이 구간별 분포
    logger.info(f"\n[3] 차이 일수 구간별 분포")
    bucket_dist = con.execute(f"""
        WITH with_prev_prev AS (
            SELECT
                *,
                LAG(membership_expire_date, 2) OVER (
                    PARTITION BY user_id
                    ORDER BY transaction_date, membership_expire_date
                ) AS prev_prev_expire_date
            FROM {seq_table}
        ),
        anomaly_cases AS (
            SELECT
                *,
                CAST(prev_prev_expire_date - membership_expire_date AS BIGINT) AS days_diff
            FROM with_prev_prev
            WHERE is_cancel = 1
              AND prev_prev_expire_date IS NOT NULL
              AND membership_expire_date < prev_prev_expire_date
        )
        SELECT
            CASE
                WHEN days_diff <= 7 THEN '1-7일'
                WHEN days_diff <= 30 THEN '8-30일'
                WHEN days_diff <= 90 THEN '31-90일'
                WHEN days_diff <= 180 THEN '91-180일'
                WHEN days_diff <= 365 THEN '181-365일'
                ELSE '365일 초과'
            END AS bucket,
            COUNT(*) AS row_count,
            COUNT(DISTINCT user_id) AS user_count
        FROM anomaly_cases
        GROUP BY bucket
        ORDER BY
            CASE bucket
                WHEN '1-7일' THEN 1
                WHEN '8-30일' THEN 2
                WHEN '31-90일' THEN 3
                WHEN '91-180일' THEN 4
                WHEN '181-365일' THEN 5
                ELSE 6
            END
    """).fetchall()

    buckets = []
    bucket_row_counts = []
    bucket_user_counts = []
    for bucket, row_cnt, user_cnt in bucket_dist:
        buckets.append(bucket)
        bucket_row_counts.append(row_cnt)
        bucket_user_counts.append(user_cnt)
        logger.info(f"  {bucket:>12}: {row_cnt:,} 행 ({row_cnt/anomaly_rows*100:.2f}%), {user_cnt:,} 유저")

    # 4. payment_plan_days별 분포
    logger.info(f"\n[4] payment_plan_days별 분포")
    plan_dist = con.execute(f"""
        WITH with_prev_prev AS (
            SELECT
                *,
                LAG(membership_expire_date, 2) OVER (
                    PARTITION BY user_id
                    ORDER BY transaction_date, membership_expire_date
                ) AS prev_prev_expire_date
            FROM {seq_table}
        ),
        anomaly_cases AS (
            SELECT
                *,
                CAST(prev_prev_expire_date - membership_expire_date AS BIGINT) AS days_diff
            FROM with_prev_prev
            WHERE is_cancel = 1
              AND prev_prev_expire_date IS NOT NULL
              AND membership_expire_date < prev_prev_expire_date
        )
        SELECT
            payment_plan_days,
            COUNT(*) AS row_count,
            COUNT(DISTINCT user_id) AS user_count,
            ROUND(AVG(days_diff), 2) AS avg_diff
        FROM anomaly_cases
        GROUP BY payment_plan_days
        ORDER BY row_count DESC
        LIMIT 10
    """).fetchall()

    plan_labels = []
    plan_row_counts = []
    for plan, row_cnt, user_cnt, avg_diff in plan_dist:
        plan_labels.append(str(plan))
        plan_row_counts.append(row_cnt)
        logger.info(f"  plan={plan}: {row_cnt:,} 행, {user_cnt:,} 유저, avg_diff={avg_diff}")

    # 5. 샘플 케이스
    if show_samples is not None and show_samples > 0:
        logger.info(f"\n[5] 샘플 케이스 (상위 {show_samples}개)")
        sample_cases = con.execute(f"""
            WITH with_prev_prev AS (
                SELECT
                    *,
                    LAG(membership_expire_date, 1) OVER (
                        PARTITION BY user_id
                        ORDER BY transaction_date, membership_expire_date
                    ) AS prev_expire_date,
                    LAG(membership_expire_date, 2) OVER (
                        PARTITION BY user_id
                        ORDER BY transaction_date, membership_expire_date
                    ) AS prev_prev_expire_date
                FROM {seq_table}
            )
            SELECT
                user_id,
                transaction_date,
                prev_prev_expire_date,
                prev_expire_date,
                membership_expire_date,
                CAST(prev_prev_expire_date - membership_expire_date AS BIGINT) AS days_diff,
                payment_plan_days,
                actual_plan_days
            FROM with_prev_prev
            WHERE is_cancel = 1
              AND prev_prev_expire_date IS NOT NULL
              AND membership_expire_date < prev_prev_expire_date
            ORDER BY days_diff DESC
            LIMIT {show_samples}
        """).fetchdf()
        logger.info(f"\n{sample_cases.to_string()}")

    # 그래프 생성
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 구간별 분포 (행)
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']
    axes[0, 0].bar(buckets, bucket_row_counts, color=colors[:len(buckets)])
    axes[0, 0].set_title('Days Diff Bucket Distribution (Rows)\n(prev_prev_expire - expire)')
    axes[0, 0].set_xlabel('Days Difference Bucket')
    axes[0, 0].set_ylabel('Row Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylim(bottom=0)
    for i, cnt in enumerate(bucket_row_counts):
        axes[0, 0].text(i, cnt, f'{cnt:,}', ha='center', va='bottom', fontsize=9)

    # 2. 구간별 분포 (유저)
    axes[0, 1].bar(buckets, bucket_user_counts, color=colors[:len(buckets)])
    axes[0, 1].set_title('Days Diff Bucket Distribution (Users)\n(prev_prev_expire - expire)')
    axes[0, 1].set_xlabel('Days Difference Bucket')
    axes[0, 1].set_ylabel('User Count')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_ylim(bottom=0)
    for i, cnt in enumerate(bucket_user_counts):
        axes[0, 1].text(i, cnt, f'{cnt:,}', ha='center', va='bottom', fontsize=9)

    # 3. payment_plan_days별 분포
    if plan_labels:
        axes[1, 0].bar(plan_labels, plan_row_counts, color='#3498db')
        axes[1, 0].set_title('Distribution by payment_plan_days (Rows)')
        axes[1, 0].set_xlabel('payment_plan_days')
        axes[1, 0].set_ylabel('Row Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_ylim(bottom=0)

    # 4. 히스토그램
    diff_data = con.execute(f"""
        WITH with_prev_prev AS (
            SELECT
                *,
                LAG(membership_expire_date, 2) OVER (
                    PARTITION BY user_id
                    ORDER BY transaction_date, membership_expire_date
                ) AS prev_prev_expire_date
            FROM {seq_table}
        )
        SELECT CAST(prev_prev_expire_date - membership_expire_date AS BIGINT) AS days_diff
        FROM with_prev_prev
        WHERE is_cancel = 1
          AND prev_prev_expire_date IS NOT NULL
          AND membership_expire_date < prev_prev_expire_date
    """).fetchdf()

    axes[1, 1].hist(diff_data['days_diff'], bins=30, color='#9b59b6', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Days Diff Histogram\n(prev_prev_expire - expire)')
    axes[1, 1].set_xlabel('Days Difference')
    axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "cancel_expire_before_prev_prev_expire.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"\n그래프 저장: {plot_path}")

    con.close()
    logger.info(f"=== is_cancel expire_date < prev_prev_expire_date 분석 완료 ===\n")


# ============================================================================
# 8.5.3.6. is_cancel expire_date < prev_prev_expire_date 유저 제외
# ============================================================================
def exclude_cancel_expire_before_prev_prev_expire_users(
    db_path: str,
    seq_table: str = "transactions_seq",
    target_tables: list[str] = None,
) -> None:
    """
    is_cancel=1 트랜잭션 중 membership_expire_date가 이전 두 번째 트랜잭션의
    membership_expire_date보다 작은 케이스를 가진 유저를 모든 _merge 테이블에서 제외합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        seq_table: 트랜잭션 시퀀스 테이블명 (기본값: transactions_seq)
        target_tables: 필터링을 적용할 대상 테이블 이름 리스트
                       기본값: ["train_merge", "transactions_merge", "user_logs_merge", "members_merge"]
    """
    if target_tables is None:
        target_tables = ["train_merge", "transactions_merge", "user_logs_merge", "members_merge"]

    logger.info(f"=== is_cancel expire_date < prev_prev_expire_date 유저 제외 시작 ===")
    logger.info(f"시퀀스 테이블: {seq_table}")
    logger.info(f"대상 테이블: {target_tables}")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    # 시퀀스 테이블 존재 확인
    if seq_table not in existing_tables:
        logger.error(f"{seq_table} 테이블이 존재하지 않습니다.")
        con.close()
        return

    # 시퀀스 테이블의 user_id 컬럼 확인
    seq_cols = [row[0] for row in con.execute(f"DESCRIBE {seq_table}").fetchall()]
    if "user_id" in seq_cols:
        id_col = "user_id"
    elif "msno" in seq_cols:
        id_col = "msno"
    else:
        logger.error(f"{seq_table}에 user_id/msno 컬럼이 없습니다.")
        con.close()
        return

    # 전체 유저 수
    total_users = con.execute(f"""
        SELECT COUNT(DISTINCT {id_col}) FROM {seq_table}
    """).fetchone()[0]
    logger.info(f"시퀀스 테이블 전체 유저 수: {total_users:,}")

    # 해당 조건에 맞는 유저 추출
    con.execute(f"""
        CREATE OR REPLACE TABLE _cancel_prev_prev_anomaly_users_temp AS
        WITH with_prev_prev AS (
            SELECT
                {id_col},
                membership_expire_date,
                is_cancel,
                LAG(membership_expire_date, 2) OVER (
                    PARTITION BY {id_col}
                    ORDER BY transaction_date, membership_expire_date
                ) AS prev_prev_expire_date
            FROM {seq_table}
        )
        SELECT DISTINCT {id_col}
        FROM with_prev_prev
        WHERE is_cancel = 1
          AND prev_prev_expire_date IS NOT NULL
          AND membership_expire_date < prev_prev_expire_date;
    """)

    anomaly_count = con.execute("SELECT COUNT(*) FROM _cancel_prev_prev_anomaly_users_temp").fetchone()[0]
    logger.info(f"제외 대상 유저 수: {anomaly_count:,} ({anomaly_count/total_users*100:.2f}%)")

    if anomaly_count == 0:
        logger.info("제외할 유저가 없습니다.")
        con.execute("DROP TABLE IF EXISTS _cancel_prev_prev_anomaly_users_temp;")
        con.close()
        return

    # 대상 테이블에서 해당 유저 제외
    for t in tqdm(target_tables, desc="cancel prev_prev expire anomaly 유저 제외"):
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

        # 해당 유저 제외 (NOT IN)
        con.execute(f"""
            CREATE OR REPLACE TABLE {t}_filtered AS
            SELECT *
            FROM {t}
            WHERE {target_col} NOT IN (SELECT {id_col} FROM _cancel_prev_prev_anomaly_users_temp);
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
    con.execute("DROP TABLE IF EXISTS _cancel_prev_prev_anomaly_users_temp;")

    con.close()
    logger.info(f"=== is_cancel expire_date < prev_prev_expire_date 유저 제외 완료 ===\n")


# ============================================================================
# 8.5.3.7. 연속 is_cancel=1 트랜잭션 분석
# ============================================================================
def analyze_consecutive_cancel_transactions(
    db_path: str,
    seq_table: str = "transactions_seq",
    output_dir: str = "data/analysis",
    show_samples: int | None = None,
) -> None:
    """
    is_cancel=1 트랜잭션이 두 개 이상 연속으로 나타나는 케이스를 분석합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        seq_table: 트랜잭션 시퀀스 테이블명 (기본값: transactions_seq)
        output_dir: 그래프 저장 디렉토리 (기본값: data/analysis)
        show_samples: 샘플 유저 수 (None이면 샘플 출력 안 함)
    """
    logger.info(f"=== 연속 is_cancel=1 트랜잭션 분석 시작 ===")
    logger.info(f"테이블: {seq_table}")
    logger.info(f"출력 디렉토리: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    # 테이블 존재 확인
    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if seq_table not in existing_tables:
        logger.error(f"테이블 {seq_table}이 존재하지 않습니다.")
        con.close()
        return

    # 전체 통계
    total_stats = con.execute(f"""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT user_id) AS total_users,
            SUM(CASE WHEN is_cancel = 1 THEN 1 ELSE 0 END) AS total_cancel_rows
        FROM {seq_table}
    """).fetchone()
    total_rows, total_users, total_cancel_rows = total_stats
    logger.info(f"전체: {total_rows:,} 행, {total_users:,} 유저")
    logger.info(f"전체 is_cancel=1: {total_cancel_rows:,} 행")

    # 연속 is_cancel=1 케이스 찾기
    # LAG를 사용해 이전 트랜잭션의 is_cancel을 가져오고, 둘 다 1이면 연속
    logger.info(f"\n[1] 연속 is_cancel=1 케이스 통계")
    consecutive_stats = con.execute(f"""
        WITH with_prev_cancel AS (
            SELECT
                *,
                LAG(is_cancel) OVER (
                    PARTITION BY user_id
                    ORDER BY transaction_date, membership_expire_date
                ) AS prev_is_cancel
            FROM {seq_table}
        ),
        consecutive_cancel AS (
            SELECT *
            FROM with_prev_cancel
            WHERE is_cancel = 1 AND prev_is_cancel = 1
        )
        SELECT
            COUNT(*) AS consecutive_count,
            COUNT(DISTINCT user_id) AS consecutive_users
        FROM consecutive_cancel
    """).fetchone()
    consecutive_count, consecutive_users = consecutive_stats

    logger.info(f"  연속 is_cancel=1 행 수: {consecutive_count:,} ({consecutive_count/total_cancel_rows*100:.2f}% of cancel)")
    logger.info(f"  영향받은 유저 수: {consecutive_users:,} ({consecutive_users/total_users*100:.4f}%)")

    if consecutive_count == 0:
        logger.info("연속 is_cancel=1 케이스가 없습니다.")
        con.close()
        logger.info(f"=== 연속 is_cancel=1 트랜잭션 분석 완료 ===\n")
        return

    # 연속 is_cancel 길이 분석 (연속 그룹별)
    logger.info(f"\n[2] 연속 is_cancel=1 길이 분포")
    # 연속 그룹을 식별하기 위해 is_cancel이 바뀌는 지점마다 새 그룹 부여
    streak_dist = con.execute(f"""
        WITH numbered AS (
            SELECT
                user_id,
                is_cancel,
                transaction_date,
                membership_expire_date,
                ROW_NUMBER() OVER (
                    PARTITION BY user_id
                    ORDER BY transaction_date, membership_expire_date
                ) AS rn
            FROM {seq_table}
        ),
        with_group AS (
            SELECT
                *,
                rn - ROW_NUMBER() OVER (
                    PARTITION BY user_id, is_cancel
                    ORDER BY transaction_date, membership_expire_date
                ) AS grp
            FROM numbered
        ),
        cancel_streaks AS (
            SELECT
                user_id,
                grp,
                COUNT(*) AS streak_length
            FROM with_group
            WHERE is_cancel = 1
            GROUP BY user_id, grp
            HAVING COUNT(*) >= 2
        )
        SELECT
            streak_length,
            COUNT(*) AS streak_count,
            COUNT(DISTINCT user_id) AS user_count
        FROM cancel_streaks
        GROUP BY streak_length
        ORDER BY streak_length
    """).fetchall()

    streak_lengths = []
    streak_counts = []
    streak_user_counts = []
    for streak_len, streak_cnt, user_cnt in streak_dist:
        streak_lengths.append(streak_len)
        streak_counts.append(streak_cnt)
        streak_user_counts.append(user_cnt)
        logger.info(f"  연속 {streak_len}개: {streak_cnt:,} 그룹, {user_cnt:,} 유저")

    # payment_plan_days별 분포
    logger.info(f"\n[3] 연속 is_cancel=1의 payment_plan_days별 분포")
    plan_dist = con.execute(f"""
        WITH with_prev_cancel AS (
            SELECT
                *,
                LAG(is_cancel) OVER (
                    PARTITION BY user_id
                    ORDER BY transaction_date, membership_expire_date
                ) AS prev_is_cancel
            FROM {seq_table}
        ),
        consecutive_cancel AS (
            SELECT *
            FROM with_prev_cancel
            WHERE is_cancel = 1 AND prev_is_cancel = 1
        )
        SELECT
            payment_plan_days,
            COUNT(*) AS row_count,
            COUNT(DISTINCT user_id) AS user_count
        FROM consecutive_cancel
        GROUP BY payment_plan_days
        ORDER BY row_count DESC
        LIMIT 10
    """).fetchall()

    plan_labels = []
    plan_row_counts = []
    for plan, row_cnt, user_cnt in plan_dist:
        plan_labels.append(str(plan))
        plan_row_counts.append(row_cnt)
        logger.info(f"  plan={plan}: {row_cnt:,} 행, {user_cnt:,} 유저")

    # actual_plan_days별 분포
    logger.info(f"\n[4] 연속 is_cancel=1의 actual_plan_days 구간별 분포")
    actual_dist = con.execute(f"""
        WITH with_prev_cancel AS (
            SELECT
                *,
                LAG(is_cancel) OVER (
                    PARTITION BY user_id
                    ORDER BY transaction_date, membership_expire_date
                ) AS prev_is_cancel
            FROM {seq_table}
        ),
        consecutive_cancel AS (
            SELECT *
            FROM with_prev_cancel
            WHERE is_cancel = 1 AND prev_is_cancel = 1
        )
        SELECT
            CASE
                WHEN actual_plan_days < 0 THEN '음수'
                WHEN actual_plan_days = 0 THEN '0'
                WHEN actual_plan_days <= 7 THEN '1-7'
                WHEN actual_plan_days <= 30 THEN '8-30'
                WHEN actual_plan_days <= 60 THEN '31-60'
                ELSE '61+'
            END AS bucket,
            COUNT(*) AS row_count,
            COUNT(DISTINCT user_id) AS user_count
        FROM consecutive_cancel
        GROUP BY bucket
        ORDER BY row_count DESC
    """).fetchall()

    actual_buckets = []
    actual_row_counts = []
    for bucket, row_cnt, user_cnt in actual_dist:
        actual_buckets.append(bucket)
        actual_row_counts.append(row_cnt)
        logger.info(f"  {bucket:>8}: {row_cnt:,} 행, {user_cnt:,} 유저")

    # 샘플 케이스 출력
    if show_samples is not None and show_samples > 0:
        logger.info(f"\n[5] 샘플 케이스 (유저별 연속 취소 트랜잭션, {show_samples}명)")
        sample_users = con.execute(f"""
            WITH with_prev_cancel AS (
                SELECT
                    *,
                    LAG(is_cancel) OVER (
                        PARTITION BY user_id
                        ORDER BY transaction_date, membership_expire_date
                    ) AS prev_is_cancel
                FROM {seq_table}
            ),
            users_with_consecutive AS (
                SELECT DISTINCT user_id
                FROM with_prev_cancel
                WHERE is_cancel = 1 AND prev_is_cancel = 1
                LIMIT {show_samples}
            )
            SELECT
                s.user_id,
                s.transaction_date,
                s.membership_expire_date,
                s.payment_plan_days,
                s.actual_plan_days,
                s.is_cancel
            FROM {seq_table} s
            JOIN users_with_consecutive u ON s.user_id = u.user_id
            ORDER BY s.user_id, s.transaction_date, s.membership_expire_date
        """).fetchdf()
        logger.info(f"\n{sample_users.to_string()}")

    # 그래프 생성
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 연속 길이별 분포 (그룹 수)
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']
    if streak_lengths:
        x_labels = [str(x) for x in streak_lengths]
        axes[0, 0].bar(x_labels, streak_counts, color=colors[:len(streak_lengths)])
        axes[0, 0].set_title('Consecutive Cancel Streak Length Distribution\n(Group Count)')
        axes[0, 0].set_xlabel('Streak Length')
        axes[0, 0].set_ylabel('Group Count')
        axes[0, 0].set_ylim(bottom=0)
        for i, cnt in enumerate(streak_counts):
            axes[0, 0].text(i, cnt, f'{cnt:,}', ha='center', va='bottom', fontsize=9)

    # 2. 연속 길이별 분포 (유저 수)
    if streak_lengths:
        axes[0, 1].bar(x_labels, streak_user_counts, color=colors[:len(streak_lengths)])
        axes[0, 1].set_title('Consecutive Cancel Streak Length Distribution\n(User Count)')
        axes[0, 1].set_xlabel('Streak Length')
        axes[0, 1].set_ylabel('User Count')
        axes[0, 1].set_ylim(bottom=0)
        for i, cnt in enumerate(streak_user_counts):
            axes[0, 1].text(i, cnt, f'{cnt:,}', ha='center', va='bottom', fontsize=9)

    # 3. payment_plan_days별 분포
    if plan_labels:
        axes[1, 0].bar(plan_labels, plan_row_counts, color='#3498db')
        axes[1, 0].set_title('Consecutive Cancel by payment_plan_days')
        axes[1, 0].set_xlabel('payment_plan_days')
        axes[1, 0].set_ylabel('Row Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_ylim(bottom=0)

    # 4. actual_plan_days 구간별 분포
    if actual_buckets:
        axes[1, 1].bar(actual_buckets, actual_row_counts, color='#9b59b6')
        axes[1, 1].set_title('Consecutive Cancel by actual_plan_days Bucket')
        axes[1, 1].set_xlabel('actual_plan_days Bucket')
        axes[1, 1].set_ylabel('Row Count')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].set_ylim(bottom=0)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "consecutive_cancel_transactions.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"\n그래프 저장: {plot_path}")

    con.close()
    logger.info(f"=== 연속 is_cancel=1 트랜잭션 분석 완료 ===\n")


# ============================================================================
# 8.5.3.8. 연속 is_cancel=1 유저 제외
# ============================================================================
def exclude_consecutive_cancel_users(
    db_path: str,
    seq_table: str = "transactions_seq",
    target_tables: list[str] = None,
) -> None:
    """
    is_cancel=1 트랜잭션이 두 개 이상 연속으로 나타나는 유저를 모든 _merge 테이블에서 제외합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        seq_table: 트랜잭션 시퀀스 테이블명 (기본값: transactions_seq)
        target_tables: 필터링을 적용할 대상 테이블 이름 리스트
                       기본값: ["train_merge", "transactions_merge", "user_logs_merge", "members_merge"]
    """
    if target_tables is None:
        target_tables = ["train_merge", "transactions_merge", "user_logs_merge", "members_merge"]

    logger.info(f"=== 연속 is_cancel=1 유저 제외 시작 ===")
    logger.info(f"시퀀스 테이블: {seq_table}")
    logger.info(f"대상 테이블: {target_tables}")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    # 시퀀스 테이블 존재 확인
    if seq_table not in existing_tables:
        logger.error(f"{seq_table} 테이블이 존재하지 않습니다.")
        con.close()
        return

    # 시퀀스 테이블의 user_id 컬럼 확인
    seq_cols = [row[0] for row in con.execute(f"DESCRIBE {seq_table}").fetchall()]
    if "user_id" in seq_cols:
        id_col = "user_id"
    elif "msno" in seq_cols:
        id_col = "msno"
    else:
        logger.error(f"{seq_table}에 user_id/msno 컬럼이 없습니다.")
        con.close()
        return

    # 전체 유저 수
    total_users = con.execute(f"""
        SELECT COUNT(DISTINCT {id_col}) FROM {seq_table}
    """).fetchone()[0]
    logger.info(f"시퀀스 테이블 전체 유저 수: {total_users:,}")

    # 연속 is_cancel=1을 가진 유저 추출
    con.execute(f"""
        CREATE OR REPLACE TABLE _consecutive_cancel_users_temp AS
        WITH with_prev_cancel AS (
            SELECT
                {id_col},
                is_cancel,
                LAG(is_cancel) OVER (
                    PARTITION BY {id_col}
                    ORDER BY transaction_date, membership_expire_date
                ) AS prev_is_cancel
            FROM {seq_table}
        )
        SELECT DISTINCT {id_col}
        FROM with_prev_cancel
        WHERE is_cancel = 1 AND prev_is_cancel = 1;
    """)

    anomaly_count = con.execute("SELECT COUNT(*) FROM _consecutive_cancel_users_temp").fetchone()[0]
    logger.info(f"제외 대상 유저 수: {anomaly_count:,} ({anomaly_count/total_users*100:.2f}%)")

    if anomaly_count == 0:
        logger.info("제외할 유저가 없습니다.")
        con.execute("DROP TABLE IF EXISTS _consecutive_cancel_users_temp;")
        con.close()
        return

    # 대상 테이블에서 해당 유저 제외
    for t in tqdm(target_tables, desc="연속 is_cancel 유저 제외"):
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

        # 해당 유저 제외 (NOT IN)
        con.execute(f"""
            CREATE OR REPLACE TABLE {t}_filtered AS
            SELECT *
            FROM {t}
            WHERE {target_col} NOT IN (SELECT {id_col} FROM _consecutive_cancel_users_temp);
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
    con.execute("DROP TABLE IF EXISTS _consecutive_cancel_users_temp;")

    con.close()
    logger.info(f"=== 연속 is_cancel=1 유저 제외 완료 ===\n")


# ============================================================================
# 8.5.4. membership_expire_date 감소 케이스 분석
# ============================================================================
def analyze_expire_date_decrease(
    db_path: str,
    seq_table: str = "transactions_seq",
) -> None:
    """
    같은 sequence_group 내에서 membership_expire_date가 감소하는 케이스를 분석합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        seq_table: 트랜잭션 시퀀스 테이블명 (기본값: transactions_seq)
    """
    logger.info(f"=== membership_expire_date 감소 케이스 분석 시작 ===")
    logger.info(f"테이블: {seq_table}")

    con = duckdb.connect(db_path)
    con.execute("PRAGMA threads=8;")

    # 테이블 존재 확인
    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if seq_table not in existing_tables:
        logger.error(f"테이블 {seq_table}이 존재하지 않습니다.")
        con.close()
        return

    # 전체 통계
    total_stats = con.execute(f"""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT user_id) AS total_users,
            COUNT(DISTINCT (user_id, sequence_group_id)) AS total_seq_groups
        FROM {seq_table}
    """).fetchone()
    total_rows, total_users, total_seq_groups = total_stats
    logger.info(f"전체: {total_rows:,} 행, {total_users:,} 유저, {total_seq_groups:,} 시퀀스 그룹")

    # membership_expire_date가 감소하는 케이스 찾기
    # 같은 user_id, sequence_group_id 내에서 이전 트랜잭션보다 expire_date가 줄어든 경우
    decrease_stats = con.execute(f"""
        WITH with_prev_expire AS (
            SELECT
                *,
                LAG(membership_expire_date) OVER (
                    PARTITION BY user_id, sequence_group_id
                    ORDER BY transaction_date, membership_expire_date
                ) AS prev_expire_date
            FROM {seq_table}
        ),
        decrease_cases AS (
            SELECT *
            FROM with_prev_expire
            WHERE prev_expire_date IS NOT NULL
              AND membership_expire_date < prev_expire_date
        )
        SELECT
            COUNT(*) AS decrease_count,
            COUNT(DISTINCT user_id) AS decrease_users,
            COUNT(DISTINCT (user_id, sequence_group_id)) AS decrease_seq_groups
        FROM decrease_cases
    """).fetchone()
    decrease_count, decrease_users, decrease_seq_groups = decrease_stats

    logger.info(f"\n[1] membership_expire_date 감소 케이스 전체 통계")
    logger.info(f"  감소 케이스: {decrease_count:,} 행 ({decrease_count/total_rows*100:.4f}%)")
    logger.info(f"  영향받은 유저: {decrease_users:,} ({decrease_users/total_users*100:.4f}%)")
    logger.info(f"  영향받은 시퀀스 그룹: {decrease_seq_groups:,} ({decrease_seq_groups/total_seq_groups*100:.4f}%)")

    if decrease_count == 0:
        logger.info("감소 케이스가 없습니다.")
        con.close()
        logger.info(f"=== membership_expire_date 감소 케이스 분석 완료 ===\n")
        return

    # is_cancel 여부별 통계
    logger.info(f"\n[2] is_cancel 여부별 감소 케이스 통계")
    cancel_stats = con.execute(f"""
        WITH with_prev_expire AS (
            SELECT
                *,
                LAG(membership_expire_date) OVER (
                    PARTITION BY user_id, sequence_group_id
                    ORDER BY transaction_date, membership_expire_date
                ) AS prev_expire_date
            FROM {seq_table}
        ),
        decrease_cases AS (
            SELECT *
            FROM with_prev_expire
            WHERE prev_expire_date IS NOT NULL
              AND membership_expire_date < prev_expire_date
        )
        SELECT
            is_cancel,
            COUNT(*) AS cnt,
            COUNT(DISTINCT user_id) AS user_cnt
        FROM decrease_cases
        GROUP BY is_cancel
        ORDER BY is_cancel
    """).fetchall()

    for is_cancel, cnt, user_cnt in cancel_stats:
        pct = cnt / decrease_count * 100
        logger.info(f"  is_cancel={is_cancel}: {cnt:,} 행 ({pct:.2f}%), {user_cnt:,} 유저")

    # 감소 폭 분포
    logger.info(f"\n[3] 감소 폭(일) 분포")
    decrease_dist = con.execute(f"""
        WITH with_prev_expire AS (
            SELECT
                *,
                LAG(membership_expire_date) OVER (
                    PARTITION BY user_id, sequence_group_id
                    ORDER BY transaction_date, membership_expire_date
                ) AS prev_expire_date
            FROM {seq_table}
        ),
        decrease_cases AS (
            SELECT
                *,
                CAST(prev_expire_date - membership_expire_date AS BIGINT) AS decrease_days
            FROM with_prev_expire
            WHERE prev_expire_date IS NOT NULL
              AND membership_expire_date < prev_expire_date
        )
        SELECT
            CASE
                WHEN decrease_days <= 7 THEN '1-7일'
                WHEN decrease_days <= 30 THEN '8-30일'
                WHEN decrease_days <= 90 THEN '31-90일'
                WHEN decrease_days <= 180 THEN '91-180일'
                WHEN decrease_days <= 365 THEN '181-365일'
                ELSE '365일 초과'
            END AS bucket,
            COUNT(*) AS cnt,
            is_cancel
        FROM decrease_cases
        GROUP BY bucket, is_cancel
        ORDER BY
            CASE bucket
                WHEN '1-7일' THEN 1
                WHEN '8-30일' THEN 2
                WHEN '31-90일' THEN 3
                WHEN '91-180일' THEN 4
                WHEN '181-365일' THEN 5
                ELSE 6
            END,
            is_cancel
    """).fetchall()

    # 버킷별로 그룹화하여 출력
    from collections import defaultdict
    bucket_data = defaultdict(lambda: {'cancel_0': 0, 'cancel_1': 0})
    for bucket, cnt, is_cancel in decrease_dist:
        key = f'cancel_{is_cancel}'
        bucket_data[bucket][key] = cnt

    bucket_order = ['1-7일', '8-30일', '31-90일', '91-180일', '181-365일', '365일 초과']
    for bucket in bucket_order:
        if bucket in bucket_data:
            data = bucket_data[bucket]
            total = data['cancel_0'] + data['cancel_1']
            logger.info(f"  {bucket:>10}: 총 {total:,} (is_cancel=0: {data['cancel_0']:,}, is_cancel=1: {data['cancel_1']:,})")

    # 감소 폭 통계
    logger.info(f"\n[4] 감소 폭 상세 통계 (is_cancel별)")
    decrease_summary = con.execute(f"""
        WITH with_prev_expire AS (
            SELECT
                *,
                LAG(membership_expire_date) OVER (
                    PARTITION BY user_id, sequence_group_id
                    ORDER BY transaction_date, membership_expire_date
                ) AS prev_expire_date
            FROM {seq_table}
        ),
        decrease_cases AS (
            SELECT
                *,
                CAST(prev_expire_date - membership_expire_date AS BIGINT) AS decrease_days
            FROM with_prev_expire
            WHERE prev_expire_date IS NOT NULL
              AND membership_expire_date < prev_expire_date
        )
        SELECT
            is_cancel,
            MIN(decrease_days) AS min_decrease,
            MAX(decrease_days) AS max_decrease,
            ROUND(AVG(decrease_days), 1) AS avg_decrease,
            ROUND(MEDIAN(decrease_days), 1) AS median_decrease
        FROM decrease_cases
        GROUP BY is_cancel
        ORDER BY is_cancel
    """).fetchall()

    for is_cancel, min_d, max_d, avg_d, med_d in decrease_summary:
        logger.info(f"  is_cancel={is_cancel}: min={min_d}, max={max_d}, avg={avg_d}, median={med_d}")

    # 샘플 케이스 출력
    logger.info(f"\n[5] 감소 케이스 샘플 (상위 10개)")
    sample_cases = con.execute(f"""
        WITH with_prev_expire AS (
            SELECT
                *,
                LAG(membership_expire_date) OVER (
                    PARTITION BY user_id, sequence_group_id
                    ORDER BY transaction_date, membership_expire_date
                ) AS prev_expire_date
            FROM {seq_table}
        )
        SELECT
            user_id,
            sequence_group_id,
            transaction_date,
            prev_expire_date,
            membership_expire_date,
            CAST(prev_expire_date - membership_expire_date AS BIGINT) AS decrease_days,
            is_cancel,
            payment_plan_days
        FROM with_prev_expire
        WHERE prev_expire_date IS NOT NULL
          AND membership_expire_date < prev_expire_date
        ORDER BY decrease_days DESC
        LIMIT 10
    """).fetchdf()

    logger.info(f"\n{sample_cases.to_string()}")

    con.close()
    logger.info(f"\n=== membership_expire_date 감소 케이스 분석 완료 ===\n")


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
    #         "raw_train",
    #         "raw_transactions",
    #         "raw_user_logs",
    #         "raw_members",
    #     ],
    #     drop_source_tables=False,
    #     # id_col="user_id",
    # )

    # # 4. user_id 매핑 생성 및 변환
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
    #         # "user_logs_with_transactions",
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

    # # 8.3.1. NULL total_secs 유저 제외
    # exclude_null_total_secs_users(
    #     db_path=DB_PATH,
    #     logs_table="user_logs_merge",
    #     target_tables=[
    #         "train_merge",
    #         "transactions_merge",
    #         "user_logs_merge",
    #         "members_merge",
    #     ],
    # )

    # # 8.3.2. plan_days가 30, 31인 트랜잭션만 가진 유저 수 분석
    # analyze_plan_days_30_31_users(
    #     db_path=DB_PATH,
    #     transactions_table="transactions_merge",
    # )

    # # 8.3.3. membership_expire_date가 [2015, 2017] 범위 밖인 유저 제외
    # exclude_out_of_range_expire_date_users(
    #     db_path=DB_PATH,
    #     transactions_table="transactions_merge",
    #     target_tables=[
    #         "train_merge",
    #         "transactions_merge",
    #         "user_logs_merge",
    #         "members_merge",
    #     ],
    #     min_year=2015,
    #     max_year=2017,
    # )

    # # 8.4. analyze
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

    # # 8.5.1. transactions_seq에 actual_plan_days 추가
    # add_actual_plan_days(
    #     db_path=DB_PATH,
    #     seq_table="transactions_seq",
    # )

    # # 8.5.2. actual_plan_days 분석 (음수/0/양수 분포 및 추가 통계)
    # analyze_actual_plan_days(
    #     db_path=DB_PATH,
    #     seq_table="transactions_seq",
    #     output_dir="data/analysis",
    # )

    # # 8.5.2.1. is_cancel=1 트랜잭션의 actual_plan_days 분포 분석
    # analyze_cancel_actual_plan_days(
    #     db_path=DB_PATH,
    #     seq_table="transactions_seq",
    #     output_dir="data/analysis",
    # )

    # # 8.5.3. actual_plan_days 범위 밖 유저 제외
    # exclude_out_of_range_actual_plan_days_users(
    #     db_path=DB_PATH,
    #     seq_table="transactions_seq",
    #     target_tables=[
    #         "train_merge",
    #         "transactions_merge",
    #         "user_logs_merge",
    #         "members_merge",
    #         "transactions_seq"
    #     ],
    #     non_cancel_min=1,
    #     non_cancel_max=410,
    #     cancel_min=-410,
    #     cancel_max=0,
    # )

    # # 8.5.3.1. plan_days_diff 컬럼 추가
    # add_plan_days_diff(
    #     db_path=DB_PATH,
    #     seq_table="transactions_seq",
    # )

    # # 8.5.3.2. is_cancel 트랜잭션의 actual_plan_days 분포 분석
    # analyze_cancel_actual_plan_days_distribution(
    #     db_path=DB_PATH,
    #     seq_table="transactions_seq",
    #     output_dir="data/analysis",
    # )

    # # 8.5.3.3. is_cancel 트랜잭션 중 expire_date < prev_transaction_date 분석
    # analyze_cancel_expire_before_prev_transaction(
    #     db_path=DB_PATH,
    #     seq_table="transactions_seq",
    #     output_dir="data/analysis",
    #     show_samples=10,
    # )

    # # 8.5.3.4. is_cancel expire_date < prev_transaction_date 유저 제외
    # exclude_cancel_expire_before_prev_transaction_users(
    #     db_path=DB_PATH,
    #     seq_table="transactions_seq",
    #     target_tables=[
    #         "train_merge",
    #         "transactions_merge",
    #         "user_logs_merge",
    #         "members_merge",
    #         "transactions_seq",
    #     ],
    # )

    # # 8.5.3.5. is_cancel expire_date < prev_prev_expire_date 분석
    # analyze_cancel_expire_before_prev_prev_expire(
    #     db_path=DB_PATH,
    #     seq_table="transactions_seq",
    #     output_dir="data/analysis",
    #     show_samples=10,
    # )

    # # 8.5.3.6. is_cancel expire_date < prev_prev_expire_date 유저 제외
    # exclude_cancel_expire_before_prev_prev_expire_users(
    #     db_path=DB_PATH,
    #     seq_table="transactions_seq",
    #     target_tables=[
    #         "train_merge",
    #         "transactions_merge",
    #         "user_logs_merge",
    #         "members_merge",
    #         "transactions_seq",
    #     ],
    # )

    # # 8.5.3.7. 연속 is_cancel=1 트랜잭션 분석
    # analyze_consecutive_cancel_transactions(
    #     db_path=DB_PATH,
    #     seq_table="transactions_seq",
    #     output_dir="data/analysis",
    #     show_samples=3,
    # )

    # # 8.5.3.8. 연속 is_cancel=1 유저 제외
    # exclude_consecutive_cancel_users(
    #     db_path=DB_PATH,
    #     seq_table="transactions_seq",
    #     target_tables=[
    #         "train_merge",
    #         "transactions_merge",
    #         "user_logs_merge",
    #         "members_merge",
    #         "transactions_seq",
    #     ],
    # )

    # # 8.5.4. membership_expire_date 감소 케이스 분석
    # analyze_expire_date_decrease(
    #     db_path=DB_PATH,
    #     seq_table="transactions_seq",
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

    # 8.8. user_logs_merge에 total_hours 컬럼 추가
    add_converted_column(
        db_path=DB_PATH,
        table_name="user_logs_merge",
        source_col="total_secs",
        target_col="total_hours",
        divisor=3600,
        # clip_min=0,
        # clip_max=24,
        force_overwrite=True,
    )

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

    # # 8.10. 조건부 데이터 변환 (이상치 처리 등)
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

            "user_id_map",

            # "user_logs_with_transactions"
        ],
    )

    # (선택) 최종 데이터베이스 정보 출력
    show_database_info(db_path=DB_PATH)
