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
# 0. 테이블 삭제 (유틸리티)
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

    # 최종 테이블 목록
    final_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    logger.info(f"삭제된 테이블 수: {dropped_count}")
    logger.info(f"남은 테이블 목록: {final_tables}")

    con.close()
    logger.info(f"=== 테이블 삭제 완료 ===\n")


# ============================================================================
# 0-2. 테이블 복사 (유틸리티)
# ============================================================================
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

    # 원본 테이블 존재 확인
    if source_table not in existing_tables:
        logger.error(f"원본 테이블 {source_table}이 존재하지 않습니다.")
        con.close()
        return

    # 대상 테이블 존재 확인
    if target_table in existing_tables:
        if not force_overwrite:
            logger.warning(f"대상 테이블 {target_table}이 이미 존재, 건너뜀")
            con.close()
            return
        else:
            logger.info(f"대상 테이블 {target_table}이 이미 존재, 덮어쓰기")

    # 복사 실행
    con.execute(f"CREATE OR REPLACE TABLE {target_table} AS SELECT * FROM {source_table}")

    # 결과 로깅
    row_count = con.execute(f"SELECT COUNT(*) FROM {target_table}").fetchone()[0]
    col_count = len(con.execute(f"DESCRIBE {target_table}").fetchall())
    logger.info(f"복사 완료: {row_count:,} 행, {col_count} 열")

    con.close()
    logger.info(f"=== 테이블 복사 완료 ===\n")


# ============================================================================
# 0-3. 여러 테이블 복사 (유틸리티)
# ============================================================================
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
        # 원본 테이블 존재 확인
        if source not in existing_tables:
            logger.warning(f"  {source}: 존재하지 않음, 건너뜀")
            continue

        # 대상 테이블 존재 확인
        if target in existing_tables and not force_overwrite:
            logger.warning(f"  {target}: 이미 존재, 건너뜀")
            continue

        # 복사 실행
        con.execute(f"CREATE OR REPLACE TABLE {target} AS SELECT * FROM {source}")
        row_count = con.execute(f"SELECT COUNT(*) FROM {target}").fetchone()[0]
        logger.info(f"  {source} -> {target}: {row_count:,} 행")
        copied_count += 1

    logger.info(f"복사된 테이블 수: {copied_count}")

    con.close()
    logger.info(f"=== 여러 테이블 복사 완료 ===\n")


# ============================================================================
# 1. CSV를 DuckDB 테이블로 로드
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

        # 테이블 존재 여부 확인
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

        # 청크 단위로 로드 (진행률을 같은 줄에서 덮어쓰기)
        total_rows = 0
        chunk_count = 0
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
            if i == 0:
                con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM chunk")
            else:
                con.execute(f"INSERT INTO {table_name} SELECT * FROM chunk")
            total_rows += len(chunk)
            chunk_count = i + 1
            # 같은 줄에서 덮어쓰기 (carriage return 사용)
            sys.stdout.write(f"\r    청크 {chunk_count:,} 완료 | 누적 행: {total_rows:,}")
            sys.stdout.flush()

        # 줄바꿈으로 청크 로딩 완료 표시
        sys.stdout.write("\n")
        sys.stdout.flush()

        # 결과 로깅
        col_count = len(con.execute(f"DESCRIBE {table_name}").fetchall())
        logger.info(f"    완료: {total_rows:,} 행, {col_count} 열 (총 {chunk_count:,} 청크)")

    con.close()
    logger.info(f"=== CSV -> DuckDB 로드 완료 ===\n")


# ============================================================================
# 2. 테이블 이름 정규화 (_v1 appendix 추가)
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
# 5-7. Feature Correlation 분석 및 히트맵 저장
# ============================================================================
def analyze_feature_correlation(
    db_path: str,
    table_name: str,
    output_dir: str,
    exclude_cols: list[str] = None,
    sample_size: int = None,
    corr_threshold: float = 0.5,
    null_handling: str = "dropna",
    null_values: dict = None,
) -> None:
    """
    테이블의 수치형 컬럼 간 상관관계를 분석하고 히트맵을 저장합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        table_name: 분석할 테이블 이름
        output_dir: 히트맵 이미지 저장 디렉토리
        exclude_cols: 분석에서 제외할 컬럼 리스트 (기본값: user_id, msno, date 관련)
        sample_size: 샘플 크기 (None이면 전체 데이터, 대용량 테이블에서는 샘플링 권장)
        corr_threshold: 로그에 출력할 상관관계 임계값 (기본값: 0.5)
        null_handling: NULL 값 처리 방법 (기본값: "dropna")
            - "dropna": NULL이 있는 행 제외
            - "fillzero": NULL을 0으로 채움
            - "fillmean": NULL을 해당 컬럼 평균으로 채움
        null_values: 특정 값을 NULL로 처리할 규칙 딕셔너리 (기본값: None)
            - 특정 값 리스트: {"gender": [-1], "bd": [0]}
            - 범위 지정: {"total_secs": {"min": 0, "max": 86400}}  # 범위 밖 값을 NULL로
            - 혼합 사용 가능: {"gender": [-1], "total_secs": {"min": 0, "max": 86400}}
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

    # 컬럼 정보 조회
    cols_info = con.execute(f"DESCRIBE {table_name}").fetchall()
    all_cols = {row[0]: row[1] for row in cols_info}

    logger.info(f"전체 컬럼 수: {len(all_cols)}")

    # 기본 제외 컬럼
    if exclude_cols is None:
        exclude_cols = ["user_id", "msno", "date", "date_idx", "transaction_date",
                        "membership_expire_date", "registration_init_time", "expiration_date"]

    # 수치형 컬럼만 선택 (INTEGER, BIGINT, DOUBLE, FLOAT, DECIMAL 등)
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

    # 데이터 로드 (샘플링 옵션)
    cols_str = ", ".join(numeric_cols)
    if sample_size:
        total_rows = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        logger.info(f"전체 행 수: {total_rows:,}, 샘플 크기: {sample_size:,}")
        query = f"SELECT {cols_str} FROM {table_name} USING SAMPLE {sample_size}"
    else:
        query = f"SELECT {cols_str} FROM {table_name}"

    logger.info("데이터 로드 중...")
    df = con.execute(query).fetchdf()
    logger.info(f"로드된 행 수: {len(df):,}")

    # 특정 값을 NULL로 변환
    if null_values:
        logger.info("특정 값 -> NULL 변환 중...")
        for col, rule in null_values.items():
            if col not in df.columns:
                logger.warning(f"  {col}: 컬럼이 존재하지 않음, 건너뜀")
                continue

            if isinstance(rule, list):
                # 특정 값 리스트를 NULL로 변환
                mask = df[col].isin(rule)
                count = mask.sum()
                df.loc[mask, col] = None
                logger.info(f"  {col}: 값 {rule} -> NULL ({count:,}개 변환)")
            elif isinstance(rule, dict):
                # 범위 기반 변환 (범위 밖 값을 NULL로)
                min_val = rule.get("min")
                max_val = rule.get("max")
                mask = False
                if min_val is not None:
                    mask = mask | (df[col] < min_val)
                if max_val is not None:
                    mask = mask | (df[col] > max_val)
                count = mask.sum()
                df.loc[mask, col] = None
                range_str = f"[{min_val}, {max_val}]"
                logger.info(f"  {col}: 범위 {range_str} 밖 -> NULL ({count:,}개 변환)")

    # NULL 값 처리
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()

    if total_nulls == 0:
        logger.info("NULL 값 없음")
    else:
        logger.info(f"NULL 값 발견: 총 {total_nulls:,}개")
        for col in null_counts[null_counts > 0].index:
            logger.info(f"  {col}: {null_counts[col]:,}개 NULL")

        rows_before = len(df)
        if null_handling == "dropna":
            df = df.dropna()
            logger.info(f"NULL 처리 (dropna): {rows_before:,} -> {len(df):,} 행 ({rows_before - len(df):,} 제거)")
        elif null_handling == "fillzero":
            df = df.fillna(0)
            logger.info(f"NULL 처리 (fillzero): NULL을 0으로 대체")
        elif null_handling == "fillmean":
            df = df.fillna(df.mean())
            logger.info(f"NULL 처리 (fillmean): NULL을 컬럼 평균으로 대체")
        else:
            logger.warning(f"알 수 없는 null_handling 옵션: {null_handling}, dropna로 처리")
            df = df.dropna()
            logger.info(f"NULL 처리 (dropna): {rows_before:,} -> {len(df):,} 행 ({rows_before - len(df):,} 제거)")

        if len(df) == 0:
            logger.error("NULL 처리 후 데이터가 없습니다.")
            con.close()
            return

    # 상관관계 계산
    logger.info("상관관계 계산 중...")
    corr_matrix = df.corr()

    # 상관관계가 높은 쌍 찾기 (대각선 제외)
    logger.info(f"\n상관관계 |r| >= {corr_threshold} 인 쌍:")
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= corr_threshold:
                high_corr_pairs.append((col1, col2, corr_val))

    # 상관관계 절대값 기준 정렬
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    if high_corr_pairs:
        for col1, col2, corr_val in high_corr_pairs:
            logger.info(f"  {col1} <-> {col2}: {corr_val:.4f}")
    else:
        logger.info(f"  (없음)")

    # 상관관계 요약 통계
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_values = upper_tri.stack().values

    logger.info(f"\n상관관계 요약 통계:")
    logger.info(f"  평균: {np.mean(corr_values):.4f}")
    logger.info(f"  표준편차: {np.std(corr_values):.4f}")
    logger.info(f"  최소: {np.min(corr_values):.4f}")
    logger.info(f"  최대: {np.max(corr_values):.4f}")

    # 히트맵 생성 (전체 대칭 행렬 표시)
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

    # 상관관계 행렬 CSV 저장
    csv_file = os.path.join(output_dir, f"correlation_{table_name}.csv")
    corr_matrix.to_csv(csv_file)
    logger.info(f"상관관계 행렬 CSV 저장: {csv_file}")

    con.close()
    logger.info(f"=== Feature Correlation 분석 완료 ===\n")


# ============================================================================
# 5-8. 컬럼 값 범위 정제 (범위 외 값을 NULL로 변환)
# ============================================================================
def nullify_out_of_range(
    db_path: str,
    table_name: str,
    column_name: str,
    min_val: float = None,
    max_val: float = None,
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

    # 테이블 존재 확인
    existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    if table_name not in existing_tables:
        logger.error(f"{table_name} 테이블이 존재하지 않습니다.")
        con.close()
        return

    # 컬럼 존재 확인
    cols = [row[0] for row in con.execute(f"DESCRIBE {table_name}").fetchall()]
    if column_name not in cols:
        logger.error(f"{column_name} 컬럼이 존재하지 않습니다.")
        con.close()
        return

    # 변환 전 통계
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

    # 범위 외 값 개수 확인
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

    # CASE 문으로 범위 외 값을 NULL로 변환
    case_conditions = []
    if min_val is not None:
        case_conditions.append(f"WHEN {column_name} < {min_val} THEN NULL")
    if max_val is not None:
        case_conditions.append(f"WHEN {column_name} > {max_val} THEN NULL")
    case_str = " ".join(case_conditions)

    # 임시 테이블로 변환 후 교체
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

    # 변환 후 통계
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
# 9. Parquet 내보내기
# ============================================================================
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

        # 테이블 정보
        row_count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        cols_info = con.execute(f"DESCRIBE {table}").fetchall()
        col_count = len(cols_info)

        logger.info(f"  {table}: {row_count:,} 행, {col_count} 열")

        # 컬럼별 정보
        logger.info(f"    컬럼: {[col[0] for col in cols_info]}")

        # 내보내기
        con.execute(f"""
            COPY {table}
            TO '{output_file}'
            (FORMAT parquet, COMPRESSION {compression});
        """)

        # 파일 크기 확인
        file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
        logger.info(f"    -> {output_file} ({file_size:.2f} MB)")

    con.close()
    logger.info(f"=== Parquet 내보내기 완료 ===\n")


# ============================================================================
# 유틸리티 함수
# ============================================================================
def show_database_info(db_path: str) -> None:
    """데이터베이스의 전체 정보를 출력합니다."""
    logger.info(f"=== 데이터베이스 정보 ===")

    con = duckdb.connect(db_path)

    tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]
    logger.info(f"테이블 수: {len(tables)}")

    for table in tables:
        row_count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        cols = con.execute(f"DESCRIBE {table}").fetchall()
        logger.info(f"  {table}: {row_count:,} 행, {len(cols)} 열")

    con.close()
    logger.info(f"=========================\n")


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

    analyze_feature_correlation(
        db_path=DB_PATH,
        table_name="user_logs_merge",
        output_dir="data/analysis",
        exclude_cols=["user_id"],
        # sample_size=100000,  # 대용량 테이블은 샘플링 권장
        corr_threshold=0.7,
        null_values={
            "total_secs": {"min": 0, "max": 86400},
        },
    )
    analyze_feature_correlation(
        db_path=DB_PATH,
        table_name="members_merge",
        output_dir="data/analysis",
        exclude_cols=["user_id"],
        # sample_size=100000,  # 대용량 테이블은 샘플링 권장
        corr_threshold=0.7,
        null_values={
            "gender": [-1],              # -1을 NULL로
            "bd": {"min": 0, "max": 100},                   # 0-100 범위 밖을 NULL로
        },
    )

    # # 9. Parquet 내보내기
    # export_to_parquet(
    #     db_path=DB_PATH,
    #     output_dir=PARQUET_DIR,
    #     tables=[
    #         # "raw_train_v1",
    #         # "raw_train_v2",
    #         # "raw_transactions_v1",
    #         # "raw_transactions_v2",
    #         # "raw_user_logs_v1",
    #         # "raw_user_logs_v2",
    #         # "raw_members_v3",

    #         # "train_v1",
    #         # "train_v2",
    #         # "transactions_v1",
    #         # "transactions_v2",
    #         # "user_logs_v1",
    #         # "user_logs_v2",
    #         # "members_v3",

    #         "train_merge",
    #         "transactions_merge",
    #         "user_logs_merge",
    #         "members_merge",

    #         "user_id_map",

    #         "user_logs_with_transactions"
    #     ],
    # )

    # (선택) 최종 데이터베이스 정보 출력
    show_database_info(db_path=DB_PATH)
