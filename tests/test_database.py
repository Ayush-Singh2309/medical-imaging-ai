
import pytest

from medical_imaging_ai import database


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """Create an isolated database for each test."""

    db_dir = tmp_path / "processed"
    db_path = db_dir / "test_medical_imaging.db"

    monkeypatch.setattr(database, "DATABASE_DIR", db_dir)
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)

    database.init_db()

    return db_path


@pytest.fixture
def sample_predictions():
    return [
        {"disease": "Effusion", "probability": 0.85},
        {"disease": "Atelectasis", "probability": 0.62},
        {"disease": "Pneumonia", "probability": 0.15},
    ]


@pytest.fixture
def sample_study(test_db, sample_predictions):
    return database.save_analysis(
        image_filename="test_xray.png",
        predictions=sample_predictions,
        explained_disease="Effusion",
        explained_probability=0.85
    )


def test_database_initialization(test_db):
    assert test_db.exists()

    with database.get_connection() as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()

    table_names = {row["name"] for row in tables}

    assert "studies" in table_names
    assert "predictions" in table_names


def test_save_analysis(sample_study):
    assert sample_study > 0


def test_retrieve_all_studies(sample_study):
    studies = database.get_all_studies()

    assert len(studies) == 1
    assert studies[0]["id"] == sample_study
    assert studies[0]["image_filename"] == "test_xray.png"
    assert studies[0]["explained_disease"] == "Effusion"
    assert studies[0]["prediction_count"] == 3


def test_retrieve_study_with_predictions(sample_study):
    study = database.get_study_by_id(sample_study)

    assert study is not None
    assert study["id"] == sample_study
    assert len(study["predictions"]) == 3

    # Predictions should be sorted by probability.
    assert study["predictions"][0]["disease"] == "Effusion"
    assert study["predictions"][0]["probability"] == pytest.approx(0.85)


def test_nonexistent_study():
    study = database.get_study_by_id(99999)

    assert study is None


def test_update_study_image_paths(sample_study):
    image_path = "/tmp/study/original.png"
    gradcam_path = "/tmp/study/gradcam.png"

    database.update_study_image_paths(
        study_id=sample_study,
        image_path=image_path,
        gradcam_path=gradcam_path
    )

    study = database.get_study_by_id(sample_study)

    assert study["image_path"] == image_path
    assert study["gradcam_path"] == gradcam_path


def test_update_nonexistent_study_does_not_create_record(test_db):
    database.update_study_image_paths(
        study_id=99999,
        image_path="/tmp/original.png",
        gradcam_path="/tmp/gradcam.png"
    )

    assert database.get_all_studies() == []
