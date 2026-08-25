import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Mock winsound and pyttsx3 before importing main to avoid audio/system errors in headless/test environments
sys.modules['winsound'] = MagicMock()
sys.modules['pyttsx3'] = MagicMock()

import db_manager
import recognizer

class TestHostelGatekeeper(unittest.TestCase):
    def setUp(self):
        # Override Excel path for testing to avoid touching production data
        db_manager.EXCEL_PATH = os.path.join("data", "test_hostel_records.xlsx")
        if os.path.exists(db_manager.EXCEL_PATH):
            try:
                os.remove(db_manager.EXCEL_PATH)
            except Exception:
                pass
        db_manager.init_db()

    def tearDown(self):
        # Clean up test Excel file
        if os.path.exists(db_manager.EXCEL_PATH):
            try:
                os.remove(db_manager.EXCEL_PATH)
            except Exception:
                pass

    @patch('cv2.VideoCapture')
    @patch('face_recognition.face_locations')
    @patch('face_recognition.face_encodings')
    @patch('face_recognition.load_image_file')
    @patch('face_recognition.compare_faces')
    @patch('face_recognition.face_distance')
    def test_full_pipeline(self, mock_distance, mock_compare, mock_load_image, mock_encodings, mock_locations, mock_video_capture):
        # Setup mock video capture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        # Return a dummy frame (height=480, width=640, channels=3)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_video_capture.return_value = mock_cap

        # Mock face locations and encodings for registration
        mock_locations.return_value = [(100, 200, 300, 100)] # One face found
        mock_encodings.return_value = [np.zeros(128)]
        mock_compare.return_value = [True]
        mock_distance.return_value = np.array([0.1])

        # Mock loading image file
        mock_load_image.return_value = np.zeros((200, 200, 3), dtype=np.uint8)

        # Import main and run tests
        import main
        
        # Test 1: Database operations (Registration)
        print("Testing student registration...")
        success, student_id = db_manager.register_student("Alice Doe", "9876543210", "Girl", "Alice_Doe_9876543210.jpg")
        self.assertTrue(success)
        self.assertEqual(student_id, 1)
        
        # Test 2: Inside status checking
        inside = db_manager.get_currently_inside()
        self.assertEqual(len(inside), 1)
        self.assertEqual(inside.iloc[0]["Name"], "Alice Doe")
        
        # Test 3: Load face encodings
        print("Testing face encoding loader...")
        # Create a dummy photo file so loader doesn't skip it
        os.makedirs(db_manager.FACES_DIR, exist_ok=True)
        photo_path = os.path.join(db_manager.FACES_DIR, "Alice_Doe_9876543210.jpg")
        with open(photo_path, "wb") as f:
            f.write(b"dummy image data")
            
        recognizer.load_registered_faces()
        self.assertEqual(len(recognizer.known_face_ids), 1)
        self.assertEqual(recognizer.known_face_names[0], "Alice Doe")

        # Test 4: Face Recognition
        print("Testing face recognition matching...")
        # Recognize a frame
        matches = recognizer.recognize_faces(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["name"], "Alice Doe")
        self.assertEqual(matches[0]["id"], 1)

        # Test 5: Status Toggling (OUT)
        print("Testing status toggle to OUT...")
        res, err = db_manager.toggle_student_status(1)
        self.assertIsNone(err)
        self.assertEqual(res["action"], "OUT")
        
        # Verify student is in Only_Outside sheet and removed from Inside sheet
        inside = db_manager.get_currently_inside()
        self.assertEqual(len(inside), 0)
        
        outside = db_manager.get_currently_outside()
        self.assertEqual(len(outside), 1)
        self.assertEqual(outside.iloc[0]["Name"], "Alice Doe")

        # Test 6: Status Toggling (IN)
        print("Testing status toggle to IN...")
        res, err = db_manager.toggle_student_status(1)
        self.assertIsNone(err)
        self.assertEqual(res["action"], "IN")
        
        # Verify student is back in Inside sheet and removed from Only_Outside sheet
        inside = db_manager.get_currently_inside()
        self.assertEqual(len(inside), 1)
        self.assertEqual(inside.iloc[0]["Name"], "Alice Doe")
        
        outside = db_manager.get_currently_outside()
        self.assertEqual(len(outside), 0)

        # Clean up mock file
        if os.path.exists(photo_path):
            os.remove(photo_path)
            
        print("All verification tests passed successfully!")

if __name__ == '__main__':
    unittest.main()
