
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DB_PATH = BASE_DIR / "visual_memory.db"
MODEL_NAME = "yolo11n.pt"

UPLOAD_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Thaif Visual Memory",
    page_icon="🧠",
    layout="wide",
)


def get_connection():
    return sqlite3.connect(str(DB_PATH))


def init_database():
    with get_connection() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                original_name TEXT NOT NULL,
                uploaded_at TEXT NOT NULL,
                objects_json TEXT NOT NULL
            )
        """)


@st.cache_resource
def load_model():
    return YOLO(MODEL_NAME)


def detect_objects(image_path):
    model = load_model()
    result = model.predict(
        source=str(image_path),
        conf=0.25,
        verbose=False,
    )[0]

    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        name = result.names[class_id]
        confidence = float(box.conf[0].item())

        detections.append({
            "name": name,
            "confidence": round(confidence, 3),
        })

    annotated = Image.fromarray(result.plot()[:, :, ::-1])
    return detections, annotated


def save_sighting(filename, original_name, detections):
    timestamp = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    with get_connection() as con:
        con.execute(
            """
            INSERT INTO sightings
            (filename, original_name, uploaded_at, objects_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                filename,
                original_name,
                timestamp,
                json.dumps(detections),
            ),
        )


def find_sightings(query):
    with get_connection() as con:
        rows = con.execute(
            """
            SELECT id, filename, original_name,
                   uploaded_at, objects_json
            FROM sightings
            ORDER BY id DESC
            """
        ).fetchall()

    matches = []
    query = query.strip().casefold()

    for row in rows:
        try:
            objects = json.loads(row[4])
        except (TypeError, json.JSONDecodeError):
            objects = []

        matching_objects = [
            item for item in objects
            if query in item["name"].casefold()
        ]

        if matching_objects:
            matches.append({
                "id": row[0],
                "filename": row[1],
                "original_name": row[2],
                "uploaded_at": row[3],
                "matching_objects": matching_objects,
            })

    return matches


init_database()

st.title("🧠 Thaif Visual Memory")
st.write(
    "Save visual memories, detect objects, and search "
    "for their last recorded sightings."
)

upload_tab, search_tab, history_tab = st.tabs([
    "📷 Scan an Image",
    "🔎 Find an Object",
    "🕘 History",
])

with upload_tab:
    st.subheader("Record a new visual memory")

    uploaded = st.file_uploader(
        "Upload a room, desk, or study-area image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(
            image,
            caption="Image to scan",
            use_container_width=True,
        )

        if st.button("Detect Objects and Save", type="primary"):
            image_id = uuid.uuid4().hex
            image_path = UPLOAD_DIR / f"{image_id}.jpg"

            try:
                image.save(image_path, format="JPEG")

                with st.spinner(
                    "Detecting objects... The first run may "
                    "download the AI model."
                ):
                    detections, annotated = detect_objects(image_path)

                save_sighting(
                    image_path.name,
                    uploaded.name,
                    detections,
                )

                st.success("Visual memory saved!")
                st.image(
                    annotated,
                    caption="Detected objects",
                    use_container_width=True,
                )

                if detections:
                    counts = {}
                    for item in detections:
                        name = item["name"]
                        counts[name] = counts.get(name, 0) + 1

                    st.write("**Objects detected:**")
                    st.write(", ".join(
                        f"{name} ({count})"
                        for name, count in sorted(counts.items())
                    ))
                else:
                    st.info("No supported objects were detected.")

            except Exception as error:
                if image_path.exists():
                    image_path.unlink()

                st.error("The image could not be processed.")
                st.caption(f"Technical details: {error}")


with search_tab:
    st.subheader("Find an object's last recorded sighting")
    st.caption(
        "Search detected categories such as bottle, laptop, "
        "chair, book, or cell phone."
    )

    query = st.text_input(
        "Object name",
        placeholder="e.g. bottle",
        key="object_search",
    )

    if st.button("Search Memories"):
        if not query.strip():
            st.warning("Enter an object name first.")
        else:
            st.session_state["search_results"] = find_sightings(query)

    if "search_results" in st.session_state:
        results = st.session_state["search_results"]

        if not results:
            st.info(
                "No matching sightings found. Try another category "
                "or scan more images."
            )

        for item in results:
            image_path = UPLOAD_DIR / item["filename"]

            with st.container(border=True):
                st.write(f"**Recorded:** {item['uploaded_at']}")
                st.write(f"**Image:** {item['original_name']}")
                st.write(
                    "**Matching objects:** "
                    + ", ".join(
                        f"{obj['name']} ({obj['confidence']:.0%})"
                        for obj in item["matching_objects"]
                    )
                )

                if image_path.exists():
                    st.image(
                        str(image_path),
                        use_container_width=True,
                    )

        if results:
            st.caption(
                "Results show recorded sightings, not verified "
                "current locations."
            )


with history_tab:
    st.subheader("Saved visual memories")

    with get_connection() as con:
        rows = con.execute(
            """
            SELECT id, filename, original_name,
                   uploaded_at, objects_json
            FROM sightings
            ORDER BY id DESC
            """
        ).fetchall()

    if not rows:
        st.info("No memories yet. Scan an image first.")
    else:
        st.write(f"**Total recorded images:** {len(rows)}")

        for row in rows:
            with st.container(border=True):
                st.write(f"**{row[2]}**")
                st.caption(row[3])

                try:
                    objects = json.loads(row[4])
                    names = sorted({
                        item["name"] for item in objects
                    })
                    st.write(
                        "Detected: "
                        + (", ".join(names) if names else "No objects")
                    )
                except (TypeError, json.JSONDecodeError):
                    st.write("Object information unavailable.")

                image_path = UPLOAD_DIR / row[1]
                if image_path.exists():
                    st.image(
                        str(image_path),
                        use_container_width=True,
                    )


st.divider()
st.caption(
    "Thaif Visual Memory | Python · Streamlit · YOLO · SQLite"
)
