import os
import cv2
import face_recognition
import db_manager
import numpy as np

# Cache for registered students
known_face_encodings = []
known_face_ids = []
known_face_names = []

def load_registered_faces():
    """Load all registered students from Excel and compute face encodings."""
    global known_face_encodings, known_face_ids, known_face_names
    
    known_face_encodings = []
    known_face_ids = []
    known_face_names = []
    
    db_manager.init_db()  # Make sure directories exist
    students_df = db_manager.get_registered_students()
    if students_df.empty:
        print("No registered students found in database.")
        return
        
    for _, row in students_df.iterrows():
        photo_path = row["Photo Path"]
        student_id = int(row["ID"])
        name = row["Name"]
        
        if os.path.exists(photo_path):
            try:
                # Load the image file
                image = face_recognition.load_image_file(photo_path)
                # Get face encodings (returns list, take first one if found)
                encodings = face_recognition.face_encodings(image)
                if len(encodings) > 0:
                    known_face_encodings.append(encodings[0])
                    known_face_ids.append(student_id)
                    known_face_names.append(name)
                    print(f"Loaded face encoding for {name} (ID: {student_id})")
                else:
                    print(f"No face found in photo for {name} (ID: {student_id})")
            except Exception as e:
                print(f"Failed to load face for {name} from {photo_path}: {e}")
        else:
            print(f"Photo path does not exist: {photo_path}")

def recognize_faces(rgb_frame):
    """
    Detect and recognize faces in an RGB frame.
    Returns:
        List of dictionaries containing:
        - 'box': (top, right, bottom, left)
        - 'id': Student ID (None if unknown)
        - 'name': Name ('Unknown' if unknown)
    """
    # Detect all face locations
    face_locations = face_recognition.face_locations(rgb_frame)
    if not face_locations:
        return []
        
    # Encode face locations
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    recognized = []
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        student_id = None
        name = "Unknown"
        
        if len(known_face_encodings) > 0:
            # Compare face encoding with known face encodings
            # Lower tolerance is more strict (default is 0.6; 0.5 is very robust against false positives)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            
            if len(face_distances) > 0:
                best_match_idx = np.argmin(face_distances)
                if matches[best_match_idx]:
                    student_id = known_face_ids[best_match_idx]
                    name = known_face_names[best_match_idx]
        
        recognized.append({
            "box": (top, right, bottom, left),
            "id": student_id,
            "name": name
        })
        
    return recognized
