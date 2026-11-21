import sqlite3
from pathlib import Path
from datetime import datetime, timezone


# --------------------------------------------------
# Database configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATABASE_DIR = PROJECT_ROOT / "data" / "processed"

DATABASE_PATH = DATABASE_DIR / "medical_imaging.db"


def get_connection():
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)

    connection.row_factory = sqlite3.Row

    connection.execute("PRAGMA foreign_keys = ON")

    return connection


# --------------------------------------------------
# Initialize database
# --------------------------------------------------

def init_db():

    with get_connection() as conn:

        conn.execute("""
            CREATE TABLE IF NOT EXISTS studies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_filename TEXT NOT NULL,
                explained_disease TEXT NOT NULL,
                explained_probability REAL NOT NULL,
                created_at TEXT NOT NULL,
                image_path TEXT,
                gradcam_path TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_id INTEGER NOT NULL,
                disease TEXT NOT NULL,
                probability REAL NOT NULL,

                FOREIGN KEY (study_id)
                    REFERENCES studies(id)
                    ON DELETE CASCADE,

                UNIQUE (study_id, disease)
            )
        """)


# --------------------------------------------------
# Save analysis
# --------------------------------------------------

def save_analysis(
    image_filename: str,
    predictions: list,
    explained_disease: str,
    explained_probability: float
) -> int:

    created_at = datetime.now(timezone.utc).isoformat()

    with get_connection() as conn:

        cursor = conn.execute(
            """
            INSERT INTO studies (
                image_filename,
                explained_disease,
                explained_probability,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                image_filename,
                explained_disease,
                float(explained_probability),
                created_at
            )
        )

        study_id = cursor.lastrowid

        prediction_records = [
            (
                study_id,
                item["disease"],
                float(item["probability"])
            )
            for item in predictions
        ]

        conn.executemany(
            """
            INSERT INTO predictions (
                study_id,
                disease,
                probability
            )
            VALUES (?, ?, ?)
            """,
            prediction_records
        )

    return study_id



# --------------------------------------------------
# Retrieve all studies
# --------------------------------------------------

def get_all_studies():

    with get_connection() as conn:

        rows = conn.execute(
            """
            SELECT
                s.id,
                s.image_filename,
                s.explained_disease,
                s.explained_probability,
                s.created_at,
                COUNT(p.id) AS prediction_count

            FROM studies s

            LEFT JOIN predictions p
                ON s.id = p.study_id

            GROUP BY s.id

            ORDER BY s.created_at DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]


# --------------------------------------------------
# Retrieve one study with predictions
# --------------------------------------------------

def get_study_by_id(study_id: int):

    with get_connection() as conn:

        study = conn.execute(
            """
            SELECT *
            FROM studies
            WHERE id = ?
            """,
            (study_id,)
        ).fetchone()

        if study is None:
            return None

        predictions = conn.execute(
            """
            SELECT disease, probability
            FROM predictions
            WHERE study_id = ?
            ORDER BY probability DESC
            """,
            (study_id,)
        ).fetchall()

    result = dict(study)

    result["predictions"] = [
        dict(row) for row in predictions
    ]

    return result


def update_study_image_paths(
    study_id: int,
    image_path: str,
    gradcam_path: str
):
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE studies
            SET image_path = ?,
                gradcam_path = ?
            WHERE id = ?
            """,
            (image_path, gradcam_path, study_id)
        )


if __name__=="__main__":
    init_db()