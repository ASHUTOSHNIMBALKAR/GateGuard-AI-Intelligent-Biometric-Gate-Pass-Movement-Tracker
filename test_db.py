import os
import db_manager
import pandas as pd

# Override Excel path for testing to avoid touching production data
db_manager.EXCEL_PATH = os.path.join("data", "test_hostel_records.xlsx")

print("=== Starting Database Verification Test (3 Sheets Setup) ===")

# Reset Excel file for clean test
if os.path.exists(db_manager.EXCEL_PATH):
    try:
        os.remove(db_manager.EXCEL_PATH)
        print("Cleaned up existing Excel database.")
    except Exception as e:
        print(f"Warning: Could not remove database file (might be open): {e}")

db_manager.init_db()

# Test 1: Register Boy student
success, info = db_manager.register_student("John Doe", "1234567890", "Boy", "John_Doe_1234567890.jpg")
print(f"Register John Doe: success={success}, info/ID={info}")

# Test 2: Register Girl student
success, info = db_manager.register_student("Jane Smith", "9876543210", "Girl", "Jane_Smith_9876543210.jpg")
print(f"Register Jane Smith: success={success}, info/ID={info}")

# Verify Currently Inside
inside = db_manager.get_currently_inside()
print("\n--- Sheet 3: Inside Students ---")
print(inside[["ID", "Name", "Mobile Number", "Gender", "Registration Date"]])

# Test 4: Toggle John Doe (Inside -> Only_Outside)
print("\nToggling John Doe status (going OUT)...")
res, err = db_manager.toggle_student_status(1)
if err:
    print(f"Error: {err}")
else:
    print(f"Result: Student {res['name']} ({res['mobile']}) has {res['action']} at {res['timestamp']}")

# Verify Only Outside
outside = db_manager.get_currently_outside()
print("\n--- Sheet 2: Only Outside Students ---")
print(outside[["ID", "Name", "Mobile Number", "Gender", "Out Time"]])

# Verify Currently Inside again
inside = db_manager.get_currently_inside()
print("\n--- Sheet 3: Inside Students (after John went outside) ---")
print(inside[["ID", "Name", "Mobile Number", "Gender", "Registration Date"]])

# Verify Logs
logs = db_manager.get_movement_log()
print("\n--- Sheet 1: All Movement Tracking Logs ---")
print(logs)

# Test 5: Toggle John Doe (Only_Outside -> Inside)
print("\nToggling John Doe status (coming IN)...")
res, err = db_manager.toggle_student_status(1)
if err:
    print(f"Error: {err}")
else:
    print(f"Result: Student {res['name']} ({res['mobile']}) has {res['action']} at {res['timestamp']}")

# Verify sheets after John returns
outside = db_manager.get_currently_outside()
print("\n--- Sheet 2: Only Outside Students (after check-in) ---")
print(outside if not outside.empty else "No students outside.")

inside = db_manager.get_currently_inside()
print("\n--- Sheet 3: Inside Students (after check-in) ---")
print(inside[["ID", "Name", "Mobile Number", "Gender", "Registration Date"]])

logs = db_manager.get_movement_log()
print("\n--- Sheet 1: All Movement Tracking Logs (all) ---")
print(logs)

# Clean up test database file at the end
if os.path.exists(db_manager.EXCEL_PATH):
    try:
        os.remove(db_manager.EXCEL_PATH)
        print("\nCleaned up test Excel database.")
    except Exception as e:
        print(f"\nWarning: Could not remove test database file: {e}")

print("\n=== Database Verification Test Completed successfully ===")
