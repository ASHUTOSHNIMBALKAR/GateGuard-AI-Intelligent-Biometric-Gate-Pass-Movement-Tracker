import os
import pandas as pd
from datetime import datetime
import threading

EXCEL_PATH = os.path.join("data", "hostel_records.xlsx")
FACES_DIR = os.path.join("data", "registered_faces")

lock = threading.RLock()

def init_db():
    """Initialize the Excel file and directory structure if they do not exist."""
    with lock:
        os.makedirs("data", exist_ok=True)
        os.makedirs(FACES_DIR, exist_ok=True)
        
        if not os.path.exists(EXCEL_PATH):
            # Create sheets
            log_df = pd.DataFrame(columns=[
                "Timestamp", "ID", "Name", "Mobile Number", "Gender", "Action"
            ])
            outside_df = pd.DataFrame(columns=[
                "ID", "Name", "Mobile Number", "Gender", "Photo Path", "Out Time"
            ])
            inside_df = pd.DataFrame(columns=[
                "ID", "Name", "Mobile Number", "Gender", "Photo Path", "Registration Date"
            ])
            
            with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl') as writer:
                log_df.to_excel(writer, sheet_name="All_Movement_Tracking", index=False)
                outside_df.to_excel(writer, sheet_name="Only_Outside", index=False)
                inside_df.to_excel(writer, sheet_name="Inside", index=False)
            print(f"Database initialized at {EXCEL_PATH}")

def register_student(name, mobile, gender, photo_filename):
    """Register a new student and save them in the Inside sheet."""
    with lock:
        init_db()  # Make sure file exists
        
        try:
            inside_df = pd.read_excel(EXCEL_PATH, sheet_name="Inside")
            outside_df = pd.read_excel(EXCEL_PATH, sheet_name="Only_Outside")
        except Exception:
            inside_df = pd.DataFrame(columns=["ID", "Name", "Mobile Number", "Gender", "Photo Path", "Registration Date"])
            outside_df = pd.DataFrame(columns=["ID", "Name", "Mobile Number", "Gender", "Photo Path", "Out Time"])
            
        mobile_str = str(mobile).strip()
        
        # Check if student is already registered in either Inside or Only_Outside
        existing_mobiles_inside = inside_df["Mobile Number"].astype(str).str.strip().values if not inside_df.empty else []
        existing_mobiles_outside = outside_df["Mobile Number"].astype(str).str.strip().values if not outside_df.empty else []
        
        if mobile_str in existing_mobiles_inside or mobile_str in existing_mobiles_outside:
            return False, "Student with this mobile number is already registered!"
            
        # Get next ID
        max_id_inside = int(inside_df["ID"].max()) if not inside_df.empty else 0
        max_id_outside = int(outside_df["ID"].max()) if not outside_df.empty else 0
        new_id = max(max_id_inside, max_id_outside) + 1
        
        photo_path = os.path.join(FACES_DIR, photo_filename)
        
        new_row = {
            "ID": new_id,
            "Name": name,
            "Mobile Number": mobile_str,
            "Gender": gender,
            "Photo Path": photo_path,
            "Registration Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        inside_df = pd.concat([inside_df, pd.DataFrame([new_row])], ignore_index=True)
        
        try:
            log_df = pd.read_excel(EXCEL_PATH, sheet_name="All_Movement_Tracking")
            
            with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl') as writer:
                log_df.to_excel(writer, sheet_name="All_Movement_Tracking", index=False)
                outside_df.to_excel(writer, sheet_name="Only_Outside", index=False)
                inside_df.to_excel(writer, sheet_name="Inside", index=False)
                
            return True, new_id
        except Exception as e:
            return False, f"Failed to save registration: {e}"

def get_registered_students():
    """Load all registered students by merging Inside and Only_Outside sheets, adding a virtual Status column."""
    with lock:
        try:
            if not os.path.exists(EXCEL_PATH):
                return pd.DataFrame(columns=["ID", "Name", "Mobile Number", "Gender", "Photo Path", "Status", "Registration Date"])
            
            inside_df = pd.read_excel(EXCEL_PATH, sheet_name="Inside")
            outside_df = pd.read_excel(EXCEL_PATH, sheet_name="Only_Outside")
            
            inside_df["Status"] = "Inside"
            outside_df["Status"] = "Outside"
            # Ensure Outside df has the same columns (rename/add Registration Date or empty)
            if "Registration Date" not in outside_df.columns:
                outside_df["Registration Date"] = ""
                
            # Align columns
            cols = ["ID", "Name", "Mobile Number", "Gender", "Photo Path", "Status", "Registration Date"]
            
            # Select only cols that exist
            inside_sel = inside_df[[c for c in cols if c in inside_df.columns]]
            outside_sel = outside_df[[c for c in cols if c in outside_df.columns]]
            
            combined = pd.concat([inside_sel, outside_sel], ignore_index=True)
            return combined
        except Exception as e:
            print(f"Error loading registered students: {e}")
            return pd.DataFrame()

def get_movement_log():
    """Load the movement log."""
    with lock:
        try:
            if not os.path.exists(EXCEL_PATH):
                return pd.DataFrame(columns=["Timestamp", "ID", "Name", "Mobile Number", "Gender", "Action"])
            return pd.read_excel(EXCEL_PATH, sheet_name="All_Movement_Tracking")
        except Exception:
            return pd.DataFrame()

def get_currently_outside():
    """Load the list of students currently outside."""
    with lock:
        try:
            if not os.path.exists(EXCEL_PATH):
                return pd.DataFrame(columns=["ID", "Name", "Mobile Number", "Gender", "Photo Path", "Out Time"])
            return pd.read_excel(EXCEL_PATH, sheet_name="Only_Outside")
        except Exception:
            return pd.DataFrame()

def get_currently_inside():
    """Load the list of students currently inside."""
    with lock:
        try:
            if not os.path.exists(EXCEL_PATH):
                return pd.DataFrame(columns=["ID", "Name", "Mobile Number", "Gender", "Photo Path", "Registration Date"])
            return pd.read_excel(EXCEL_PATH, sheet_name="Inside")
        except Exception:
            return pd.DataFrame()

def toggle_student_status(student_id):
    """Toggle student between Inside and Only_Outside sheets and log to All_Movement_Tracking."""
    with lock:
        # Resolve target details outside of main try block to ensure fallback logger has access to them
        student_name = "Unknown"
        student_mobile = "Unknown"
        student_gender = "Unknown"
        student_photo = "Unknown"
        action = "UNKNOWN"
        
        try:
            # Read sheets
            log_df = pd.read_excel(EXCEL_PATH, sheet_name="All_Movement_Tracking")
            outside_df = pd.read_excel(EXCEL_PATH, sheet_name="Only_Outside")
            inside_df = pd.read_excel(EXCEL_PATH, sheet_name="Inside")
            
            # Find in Inside sheet using safe numeric comparison
            is_inside = True
            inside_ids = pd.to_numeric(inside_df["ID"], errors="coerce")
            student_idx = inside_df.index[inside_ids == int(student_id)]
            
            if len(student_idx) == 0:
                # Look in Outside sheet using safe numeric comparison
                outside_ids = pd.to_numeric(outside_df["ID"], errors="coerce")
                student_idx = outside_df.index[outside_ids == int(student_id)]
                if len(student_idx) == 0:
                    return None, "Student not found in database"
                is_inside = False
                
            idx = student_idx[0]
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if is_inside:
                # Student is inside -> Going OUT
                student = inside_df.loc[idx]
                student_name = student["Name"]
                student_mobile = student["Mobile Number"]
                student_gender = student["Gender"]
                student_photo = student["Photo Path"]
                
                # Remove from Inside
                inside_df = inside_df.drop(idx).reset_index(drop=True)
                
                # Add to Outside
                outside_row = {
                    "ID": student["ID"],
                    "Name": student_name,
                    "Mobile Number": student_mobile,
                    "Gender": student_gender,
                    "Photo Path": student_photo,
                    "Out Time": now_str
                }
                outside_df = pd.concat([outside_df, pd.DataFrame([outside_row])], ignore_index=True)
                action = "OUT"
                
            else:
                # Student is outside -> Coming IN
                student = outside_df.loc[idx]
                student_name = student["Name"]
                student_mobile = student["Mobile Number"]
                student_gender = student["Gender"]
                student_photo = student["Photo Path"]
                
                # Remove from Outside
                outside_df = outside_df.drop(idx).reset_index(drop=True)
                
                # Add to Inside
                inside_row = {
                    "ID": student["ID"],
                    "Name": student_name,
                    "Mobile Number": student_mobile,
                    "Gender": student_gender,
                    "Photo Path": student_photo,
                    "Registration Date": now_str
                }
                inside_df = pd.concat([inside_df, pd.DataFrame([inside_row])], ignore_index=True)
                action = "IN"
                
            # Log movement
            log_row = {
                "Timestamp": now_str,
                "ID": int(student_id),
                "Name": student_name,
                "Mobile Number": student_mobile,
                "Gender": student_gender,
                "Action": action
            }
            log_df = pd.concat([log_df, pd.DataFrame([log_row])], ignore_index=True)
            
            # Save back to Excel
            with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl') as writer:
                log_df.to_excel(writer, sheet_name="All_Movement_Tracking", index=False)
                outside_df.to_excel(writer, sheet_name="Only_Outside", index=False)
                inside_df.to_excel(writer, sheet_name="Inside", index=False)
                
            return {
                "id": int(student_id),
                "name": student_name,
                "mobile": student_mobile,
                "gender": student_gender,
                "action": action,
                "timestamp": now_str
            }, None
            
        except Exception as e:
            # Fallback to local text file to prevent log loss
            try:
                os.makedirs("data", exist_ok=True)
                backup_path = os.path.join("data", "movement_log_backup.txt")
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(backup_path, "a") as f:
                    f.write(f"{now_str} | ID: {student_id} | Name: {student_name} | Action: {action} | Error: Excel locked ({e})\n")
                print(f"Logged to backup file due to Excel write error: {e}")
            except Exception as backup_err:
                print(f"Backup log write failed: {backup_err}")
                
            # Return custom error details
            return None, f"Excel locked. Closed Excel to update. Log saved to backup file."
