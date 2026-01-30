# KKBox 프로젝트 코드 가이드

이 문서는 `main.py`와 `feature-engineer.py`의 코드 스타일 및 구현 패턴을 분석하여 정리한 것입니다.
향후 코드 작성 시 참고용으로 사용합니다.

---

## 1. 파일 구조

```python
"""
모듈 docstring - 스크립트의 목적과 수행하는 단계를 번호 목록으로 설명
"""

import os
import sys
import logging
from typing import Optional
# ... 표준 라이브러리

import duckdb
import pandas as pd
from tqdm import tqdm
# ... 서드파티 라이브러리

# ============================================================================
# 로깅 설정
# ============================================================================

# ============================================================================
# 1. 함수명 (기능 설명)
# ============================================================================

# ... 나머지 함수들

# ============================================================================
# 메인 실행
# ============================================================================
if __name__ == "__main__":
    # 상수 설정
    # 파이프라인 실행
```

---

## 2. 구분선 스타일

섹션 구분에 `=` 문자 76개 사용:

```python
# ============================================================================
# 섹션 번호. 섹션 제목
# ============================================================================
```

---

## 3. 로깅 설정

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
```

### 로깅 패턴

- **작업 시작/완료**: `=== 작업명 시작/완료 ===`
- **입력 파라미터 로깅**: 작업 시작 직후
- **중간 통계**: 숫자는 `{value:,}` 포맷 (천 단위 콤마)
- **경고/에러**: `logger.warning()`, `logger.error()`
- **들여쓰기**: 하위 정보는 `  ` (2칸) 또는 `    ` (4칸)

```python
logger.info(f"=== 작업명 시작 ===")
logger.info(f"대상 테이블: {target_tables}")
logger.info(f"  {table_name}: {row_count:,} 행, {col_count} 열")
logger.info(f"=== 작업명 완료 ===\n")  # 완료 후 줄바꿈
```

---

## 4. 함수 설계 패턴

### 4.1 함수 시그니처

```python
def function_name(
    db_path: str,
    table_name: str,
    optional_param: str = "default",
) -> None:
```

- 파라미터마다 줄바꿈
- 마지막 파라미터 뒤에도 콤마 (trailing comma)
- 타입 힌트 필수
- 리스트 타입: `list[str]` (Python 3.9+ 스타일)

### 4.2 Docstring 스타일

Google 스타일 docstring 사용:

```python
def function_name(
    db_path: str,
    target_tables: list[str],
    force_overwrite: bool = True,
) -> None:
    """
    함수의 목적을 한 줄로 설명합니다.
    필요시 추가 설명을 여러 줄에 걸쳐 작성합니다.

    Args:
        db_path: DuckDB 데이터베이스 경로
        target_tables: 대상 테이블 이름 리스트
        force_overwrite: True면 덮어씀 (기본값: True)
    """
```

### 4.3 공통 파라미터

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `db_path` | `str` | DuckDB 데이터베이스 경로 |
| `table_name` / `target_table` | `str` | 단일 테이블명 |
| `target_tables` | `list[str]` | 여러 테이블명 |
| `source_table` | `str` | 원본 테이블 |
| `force_overwrite` | `bool` | 덮어쓰기 여부 (기본값: True) |

---

## 5. DuckDB 사용 패턴

### 5.1 연결 및 설정

```python
con = duckdb.connect(db_path)
con.execute("PRAGMA threads=8;")  # 멀티스레드 설정

# 작업 수행...

con.close()
```

### 5.2 테이블 존재 확인

```python
existing_tables = [row[0] for row in con.execute("SHOW TABLES").fetchall()]

if table_name not in existing_tables:
    logger.warning(f"  {table_name}: 존재하지 않음, 건너뜀")
    continue
```

### 5.3 컬럼 정보 조회

```python
# 컬럼명 리스트
cols = [row[0] for row in con.execute(f"DESCRIBE {table_name}").fetchall()]

# 컬럼명과 타입
cols_info = con.execute(f"DESCRIBE {table_name}").fetchall()
cols = {row[0]: row[1] for row in cols_info}  # {col_name: col_type}
```

### 5.4 임시 테이블 패턴

테이블 변환 시 `_temp` 접미사 사용 후 교체:

```python
con.execute(f"""
    CREATE OR REPLACE TABLE {table_name}_temp AS
    SELECT ... FROM {table_name};
""")

con.execute(f"DROP TABLE {table_name};")
con.execute(f"ALTER TABLE {table_name}_temp RENAME TO {table_name};")
```

### 5.5 통계 쿼리

```python
row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
unique_count = con.execute(f"SELECT COUNT(DISTINCT {col}) FROM {table_name}").fetchone()[0]
```

---

## 6. 진행률 표시

### 6.1 tqdm 사용

```python
from tqdm import tqdm

for table in tqdm(tables, desc="작업 설명"):
    # 작업 수행
```

### 6.2 청크 로딩 시 같은 줄 덮어쓰기

```python
for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
    # 처리...
    sys.stdout.write(f"\r    청크 {i+1:,} 완료 | 누적 행: {total_rows:,}")
    sys.stdout.flush()

sys.stdout.write("\n")  # 완료 후 줄바꿈
sys.stdout.flush()
```

---

## 7. 에러 처리 패턴

### 7.1 존재 확인 후 조기 반환

```python
if table_name not in existing_tables:
    logger.error(f"테이블 {table_name}이 존재하지 않습니다.")
    con.close()
    return
```

### 7.2 경고 후 건너뛰기

```python
if "msno" not in cols:
    logger.warning(f"  {table}: msno 컬럼 없음, 건너뜀")
    continue
```

### 7.3 try-except 최소화

- 예외 처리는 필수적인 경우에만 사용
- 대부분 사전 검증으로 처리

---

## 8. SQL 쿼리 스타일

### 8.1 멀티라인 쿼리

```python
con.execute(f"""
    CREATE OR REPLACE TABLE {target_table} AS
    SELECT
        column1,
        column2,
        CASE
            WHEN condition THEN value1
            ELSE value2
        END AS new_column
    FROM {source_table}
    WHERE condition;
""")
```

### 8.2 동적 SELECT 구문 생성

```python
# COALESCE로 기본값 채우기
feature_selects = ", ".join([
    f"COALESCE(s.{col}, 0) AS {col}" for col in feature_cols
])

# EXCLUDE로 특정 컬럼 제외
con.execute(f"""
    SELECT
        m.user_id,
        t.* EXCLUDE (msno)
    FROM {table_name} t
    JOIN mapping_table m USING (msno);
""")
```

### 8.3 CTE (Common Table Expression) 사용

```python
con.execute(f"""
    CREATE OR REPLACE TABLE {target_table} AS
    WITH date_range AS (
        SELECT UNNEST(generate_series(...)) AS date
    ),
    all_users AS (
        SELECT DISTINCT user_id FROM {source_table}
    ),
    user_date_grid AS (
        SELECT u.user_id, d.date
        FROM all_users u
        CROSS JOIN date_range d
    )
    SELECT ...
    FROM user_date_grid g
    LEFT JOIN {source_table} s ON ...;
""")
```

---

## 9. 결과 검증 패턴

### 9.1 작업 전후 통계 비교

```python
# 작업 전
before_rows = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
logger.info(f"작업 전 행 수: {before_rows:,}")

# 작업 수행...

# 작업 후
after_rows = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
diff = before_rows - after_rows
logger.info(f"작업 후 행 수: {after_rows:,} ({diff:,} 변경)")
```

### 9.2 샘플 데이터 출력

```python
sample = con.execute(f"""
    SELECT * FROM {table_name}
    ORDER BY {sort_col}
    LIMIT 5
""").fetchdf()
logger.info(f"샘플 데이터:\n{sample.to_string()}")
```

---

## 10. 메인 실행 구조

```python
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

    # 1. 첫 번째 단계
    step_one_function(
        db_path=DB_PATH,
        param1=value1,
        param2=value2,
    )

    # 2. 두 번째 단계 (주석처리로 건너뛰기 가능)
    # step_two_function(
    #     db_path=DB_PATH,
    #     ...
    # )
```

---

## 11. 네이밍 컨벤션

### 11.1 함수명

- snake_case 사용
- 동사로 시작: `load_`, `create_`, `convert_`, `filter_`, `export_`
- 명확한 의도 표현: `fill_missing_dates`, `rename_msno_to_user_id`

### 11.2 변수명

- snake_case 사용
- 접미사로 타입 힌트: `_count`, `_list`, `_info`, `_col`, `_table`
- 임시 테이블: `{table_name}_temp`, `{table_name}_filtered`

### 11.3 상수

- UPPER_SNAKE_CASE: `DB_PATH`, `CSV_DIR`, `START_DATE`

---

## 12. 유틸리티 함수 패턴

재사용 가능한 유틸리티는 별도 섹션으로 분리:

```python
# ============================================================================
# 유틸리티 함수
# ============================================================================
def show_database_info(db_path: str) -> None:
    """데이터베이스의 전체 정보를 출력합니다."""
    ...

def show_table_info(db_path: str, table_name: str) -> None:
    """테이블의 상세 정보를 출력합니다."""
    ...
```

---

## 13. 특수 처리 패턴

### 13.1 컬럼명 동적 확인 (msno/user_id)

```python
cols = [row[0] for row in con.execute(f"DESCRIBE {table}").fetchall()]
if "msno" in cols:
    id_col = "msno"
elif "user_id" in cols:
    id_col = "user_id"
else:
    logger.warning(f"  {table}: msno/user_id 컬럼 없음, 건너뜀")
    continue
```

### 13.2 타입별 분기 처리

```python
col_type_upper = col_type.upper()
if "DATE" in col_type_upper or "TIMESTAMP" in col_type_upper:
    # 날짜 타입 처리
    ...
elif "INT" in col_type_upper:
    # 정수 타입 처리
    ...
```

---

## 14. 핵심 원칙 요약

1. **명시적 파라미터**: 함수 호출 시 `param_name=value` 형태로 명시
2. **단일 책임**: 각 함수는 하나의 작업만 수행
3. **풍부한 로깅**: 작업 전/중/후 상태를 상세히 로깅
4. **검증 우선**: 실제 작업 전 조건 검증
5. **점진적 실행**: 메인에서 주석으로 단계별 실행 제어
6. **타입 힌트**: 모든 함수에 타입 힌트 적용
7. **숫자 포맷팅**: `{value:,}` 로 천 단위 콤마 적용
