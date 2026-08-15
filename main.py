import os
import time
import cv2
import threading
import queue
import winsound
import pyttsx3
import pandas as pd
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw
import face_recognition
import db_manager
import recognizer

# Initialize customtkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Global threads queues & locks
speech_queue = queue.Queue()
gui_event_queue = queue.Queue()
frame_lock = threading.Lock()

latest_frame = None
recognized_faces = []
is_camera_running = True
camera_index = 0
cap = None

# Cooldown for face detections: {student_id: timestamp}
detection_cooldown = {}
COOLDOWN_SECONDS = 15

# TTS Speech Worker Thread function
def speech_worker():
    """Sequential TTS runner in a separate thread to prevent blocking GUI."""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 140)
        engine.setProperty('volume', 0.9)
    except Exception as e:
        print(f"TTS init failed: {e}")
        engine = None
        
    while True:
        text = speech_queue.get()
        if text is None:
            break
        if engine:
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"TTS say error: {e}")
        speech_queue.task_done()

# Beep sound worker
def play_sound(sound_type):
    """Play thread-safe beeps for movement events."""
    def beep():
        try:
            if sound_type == "IN":
                # Pleasant high-low chime
                winsound.Beep(1200, 150)
                winsound.Beep(1500, 150)
            elif sound_type == "OUT":
                # Alert chime
                winsound.Beep(900, 200)
                winsound.Beep(700, 200)
            elif sound_type == "REGISTER":
                # Success chime
                winsound.Beep(1000, 100)
                winsound.Beep(1200, 100)
                winsound.Beep(1400, 150)
            else:
                # Standard warning beep
                winsound.Beep(600, 300)
        except Exception as e:
            print(f"Sound play failed: {e}")
            
    threading.Thread(target=beep, daemon=True).start()

# Face recognition background thread
def face_recognition_thread():
    """Processes latest frames at its own pace and triggers database events."""
    global recognized_faces
    
    # Pre-load encodings on thread startup
    recognizer.load_registered_faces()
    
    while is_camera_running:
        frame_copy = None
        with frame_lock:
            if latest_frame is not None:
                frame_copy = latest_frame.copy()
                
        if frame_copy is None:
            time.sleep(0.03)
            continue
            
        # Run recognition (face_recognition requires RGB)
        rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        results = recognizer.recognize_faces(rgb_frame)
        
        # Save results for GUI thread to draw bounding boxes
        with frame_lock:
            recognized_faces = results
            
        # Process database and voice notifications
        current_time = time.time()
        for face in results:
            student_id = face["id"]
            name = face["name"]
            
            if student_id is not None:
                # Check cooldown to prevent rapid toggling
                last_time = detection_cooldown.get(student_id, 0)
                if current_time - last_time > COOLDOWN_SECONDS:
                    # Mark cooldown immediately to avoid race conditions
                    detection_cooldown[student_id] = current_time
                    
                    # Toggle student status
                    result, err = db_manager.toggle_student_status(student_id)
                    if err:
                        print(f"Error toggling status for {name}: {err}")
                        gui_event_queue.put({
                            "type": "error",
                            "message": f"{err}"
                        })
                        continue
                        
                    action = result["action"]
                    gender = result["gender"]
                    timestamp = result["timestamp"]
                    
                    # Queue sound and TTS
                    play_sound(action)
                    if action == "OUT":
                        msg = f"Goodbye {name}, have a safe trip."
                    else:
                        msg = f"Welcome back, {name}."
                    speech_queue.put(msg)
                    
                    # Send event to GUI thread to refresh tables and show alert
                    gui_event_queue.put({
                        "type": "toggle",
                        "name": name,
                        "action": action,
                        "gender": gender,
                        "time": timestamp
                    })
                    
        time.sleep(0.1)  # Moderate CPU usage

# Tkinter main app class
class HostelGatekeeperApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Initialize DB
        db_manager.init_db()
        
        # Window setup
        self.title("Hostel Gate AI Gatekeeper")
        self.geometry("1200x750")
        self.minsize(1000, 650)
        
        # UI state
        self.active_tab = "dashboard"
        self.toast_label = None
        self.toast_timer = None
        self.last_stats = {"total": 0, "inside": 0, "outside": 0}
        
        # Create Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # 1. Sidebar Frame
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0, fg_color="#121212")
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1) # Push stats to bottom
        
        # Sidebar Title
        self.sidebar_title = ctk.CTkLabel(
            self.sidebar_frame, 
            text="HOSTEL GATE AI", 
            font=ctk.CTkFont(family="Helvetica", size=20, weight="bold"),
            text_color="#2d5cf6"
        )
        self.sidebar_title.grid(row=0, column=0, padx=20, pady=(25, 30))
        
        # Navigation Buttons
        self.btn_dashboard = ctk.CTkButton(
            self.sidebar_frame, text="Dashboard", fg_color="#2d5cf6", hover_color="#1e44c2",
            anchor="w", font=ctk.CTkFont(size=14, weight="bold"), height=40,
            command=lambda: self.switch_tab("dashboard")
        )
        self.btn_dashboard.grid(row=1, column=0, padx=15, pady=8, sticky="ew")
        
        self.btn_register = ctk.CTkButton(
            self.sidebar_frame, text="Register Student", fg_color="transparent", hover_color="#242424",
            anchor="w", font=ctk.CTkFont(size=14), height=40, text_color="#d1d1d1",
            command=lambda: self.switch_tab("register")
        )
        self.btn_register.grid(row=2, column=0, padx=15, pady=8, sticky="ew")
        
        self.btn_directory = ctk.CTkButton(
            self.sidebar_frame, text="Currently Inside", fg_color="transparent", hover_color="#242424",
            anchor="w", font=ctk.CTkFont(size=14), height=40, text_color="#d1d1d1",
            command=lambda: self.switch_tab("directory")
        )
        self.btn_directory.grid(row=3, column=0, padx=15, pady=8, sticky="ew")
        
        self.btn_logs = ctk.CTkButton(
            self.sidebar_frame, text="Movement Log", fg_color="transparent", hover_color="#242424",
            anchor="w", font=ctk.CTkFont(size=14), height=40, text_color="#d1d1d1",
            command=lambda: self.switch_tab("logs")
        )
        self.btn_logs.grid(row=4, column=0, padx=15, pady=8, sticky="ew")
        
        self.btn_outside = ctk.CTkButton(
            self.sidebar_frame, text="Currently Outside", fg_color="transparent", hover_color="#242424",
            anchor="w", font=ctk.CTkFont(size=14), height=40, text_color="#d1d1d1",
            command=lambda: self.switch_tab("outside")
        )
        self.btn_outside.grid(row=5, column=0, padx=15, pady=8, sticky="ew")
        
        # Sidebar Stats Panel
        self.stats_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="#181818", corner_radius=8)
        self.stats_frame.grid(row=8, column=0, padx=15, pady=20, sticky="ew")
        self.stats_frame.columnconfigure(0, weight=1)
        
        self.lbl_stats_title = ctk.CTkLabel(self.stats_frame, text="LIVE STATUS STATS", font=ctk.CTkFont(size=10, weight="bold"), text_color="#888")
        self.lbl_stats_title.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        
        self.lbl_stats_total = ctk.CTkLabel(self.stats_frame, text="Total Students: 0", font=ctk.CTkFont(size=12), text_color="#ccc")
        self.lbl_stats_total.grid(row=1, column=0, padx=10, pady=2, sticky="w")
        
        self.lbl_stats_inside = ctk.CTkLabel(self.stats_frame, text="Inside: 0", font=ctk.CTkFont(size=12), text_color="#2ec4b6")
        self.lbl_stats_inside.grid(row=2, column=0, padx=10, pady=2, sticky="w")
        
        self.lbl_stats_outside = ctk.CTkLabel(self.stats_frame, text="Outside: 0", font=ctk.CTkFont(size=12), text_color="#e71d36")
        self.lbl_stats_outside.grid(row=3, column=0, padx=10, pady=(2, 10), sticky="w")
        
        # 2. Main Content Container Frame
        self.content_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="#181818")
        self.content_frame.grid(row=0, column=1, sticky="nsew")
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)
        
        # Initialize tabs
        self.frames = {}
        self.init_dashboard_tab()
        self.init_register_tab()
        self.init_directory_tab()
        self.init_logs_tab()
        self.init_outside_tab()
        
        # Show dashboard initially
        self.frames["dashboard"].grid(row=0, column=0, sticky="nsew")
        
        # Initialize camera
        self.init_camera()
        
        # Start GUI polling loops
        self.update_webcam_feed()
        self.poll_gui_events()
        self.update_stats()
        
    def init_camera(self):
        """Attempts to open the default system webcam."""
        global cap, camera_index
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            self.show_toast("Warning: No Webcam Detected!", is_error=True)

    def show_toast(self, message, is_error=False, duration=4000):
        """Displays a modern floating notification at the top center of the dashboard."""
        if self.toast_label:
            self.toast_label.destroy()
        if self.toast_timer:
            self.after_cancel(self.toast_timer)
            
        color = "#e71d36" if is_error else "#2ec4b6"
        self.toast_label = ctk.CTkLabel(
            self.content_frame,
            text=message.upper(),
            font=ctk.CTkFont(family="Helvetica", size=15, weight="bold"),
            fg_color=color,
            text_color="white",
            corner_radius=8,
            padx=20,
            pady=12
        )
        self.toast_label.place(relx=0.5, rely=0.05, anchor="n")
        self.toast_timer = self.after(duration, self.hide_toast)

    def hide_toast(self):
        if self.toast_label:
            self.toast_label.destroy()
            self.toast_label = None

    def switch_tab(self, tab_name):
        if tab_name == self.active_tab:
            return
            
        # Hide previous tab
        self.frames[self.active_tab].grid_forget()
        
        # Reset navigation button styles
        buttons = {
            "dashboard": self.btn_dashboard,
            "register": self.btn_register,
            "directory": self.btn_directory,
            "logs": self.btn_logs,
            "outside": self.btn_outside
        }
        
        buttons[self.active_tab].configure(fg_color="transparent", text_color="#d1d1d1", font=ctk.CTkFont(size=14))
        
        # Show new tab
        self.active_tab = tab_name
        self.frames[tab_name].grid(row=0, column=0, sticky="nsew")
        buttons[tab_name].configure(fg_color="#2d5cf6", text_color="white", font=ctk.CTkFont(size=14, weight="bold"))
        
        # Refresh lists if switching to list views
        if tab_name == "directory":
            self.refresh_directory_view()
        elif tab_name == "logs":
            self.refresh_logs_view()
        elif tab_name == "outside":
            self.refresh_outside_view()

    # Tab initialization methods
    def init_dashboard_tab(self):
        dash = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.frames["dashboard"] = dash
        
        dash.grid_columnconfigure(0, weight=3) # Camera
        dash.grid_columnconfigure(1, weight=2) # Info & Logs
        dash.grid_rowconfigure(0, weight=1)
        
        # 1. Left side: Camera Feed Panel
        cam_panel = ctk.CTkFrame(dash, fg_color="#202020", corner_radius=12)
        cam_panel.grid(row=0, column=0, padx=(20, 10), pady=20, sticky="nsew")
        cam_panel.grid_rowconfigure(1, weight=1)
        cam_panel.grid_columnconfigure(0, weight=1)
        
        cam_header = ctk.CTkLabel(cam_panel, text="LIVE GATE CAMERA FEED", font=ctk.CTkFont(size=14, weight="bold"), text_color="#888")
        cam_header.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")
        
        self.lbl_webcam = ctk.CTkLabel(cam_panel, text="", fg_color="#181818", corner_radius=8)
        self.lbl_webcam.grid(row=1, column=0, padx=15, pady=(5, 15), sticky="nsew")
        
        # 2. Right side: Gate Control / Live Log View
        right_panel = ctk.CTkFrame(dash, fg_color="transparent")
        right_panel.grid(row=0, column=1, padx=(10, 20), pady=20, sticky="nsew")
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_rowconfigure(1, weight=1)
        
        # Live Stats / Detection Info Panel
        self.detection_card = ctk.CTkFrame(right_panel, fg_color="#202020", corner_radius=12, height=140)
        self.detection_card.grid(row=0, column=0, pady=(0, 15), sticky="ew")
        self.detection_card.grid_propagate(False)
        self.detection_card.grid_columnconfigure(1, weight=1)
        
        self.lbl_det_title = ctk.CTkLabel(self.detection_card, text="GATE ACTIVITY FEEDBACK", font=ctk.CTkFont(size=12, weight="bold"), text_color="#888")
        self.lbl_det_title.grid(row=0, column=0, columnspan=2, padx=15, pady=(15, 5), sticky="w")
        
        self.lbl_det_name = ctk.CTkLabel(self.detection_card, text="STANDBY", font=ctk.CTkFont(size=26, weight="bold"), text_color="#aaa")
        self.lbl_det_name.grid(row=1, column=0, columnspan=2, padx=15, pady=5, sticky="w")
        
        self.lbl_det_status = ctk.CTkLabel(self.detection_card, text="WAITING FOR DETECTION...", font=ctk.CTkFont(size=14), text_color="#555")
        self.lbl_det_status.grid(row=2, column=0, columnspan=2, padx=15, pady=(0, 15), sticky="w")
        
        # Live Recent Activity list
        activity_panel = ctk.CTkFrame(right_panel, fg_color="#202020", corner_radius=12)
        activity_panel.grid(row=1, column=0, sticky="nsew")
        activity_panel.grid_rowconfigure(1, weight=1)
        activity_panel.grid_columnconfigure(0, weight=1)
        
        activity_header = ctk.CTkLabel(activity_panel, text="RECENT ACTIVITY LOG", font=ctk.CTkFont(size=14, weight="bold"), text_color="#888")
        activity_header.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")
        
        self.dash_log_container = ctk.CTkScrollableFrame(activity_panel, fg_color="transparent")
        self.dash_log_container.grid(row=1, column=0, padx=10, pady=(5, 15), sticky="nsew")
        
        self.refresh_dashboard_logs()

    def init_register_tab(self):
        reg = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.frames["register"] = reg
        
        reg.grid_columnconfigure(0, weight=3) # Register Camera
        reg.grid_columnconfigure(1, weight=2) # Form Fields
        reg.grid_rowconfigure(0, weight=1)
        
        # 1. Left side: Capture preview
        preview_panel = ctk.CTkFrame(reg, fg_color="#202020", corner_radius=12)
        preview_panel.grid(row=0, column=0, padx=(20, 10), pady=20, sticky="nsew")
        preview_panel.grid_rowconfigure(1, weight=1)
        preview_panel.grid_columnconfigure(0, weight=1)
        
        preview_header = ctk.CTkLabel(preview_panel, text="REGISTRATION CAMERA PREVIEW", font=ctk.CTkFont(size=14, weight="bold"), text_color="#888")
        preview_header.grid(row=0, column=0, padx=15, pady=(15, 5), sticky="w")
        
        self.lbl_reg_webcam = ctk.CTkLabel(preview_panel, text="", fg_color="#181818", corner_radius=8)
        self.lbl_reg_webcam.grid(row=1, column=0, padx=15, pady=(5, 15), sticky="nsew")
        
        # 2. Right side: Registration Form fields
        form_panel = ctk.CTkFrame(reg, fg_color="#202020", corner_radius=12)
        form_panel.grid(row=0, column=1, padx=(10, 20), pady=20, sticky="nsew")
        form_panel.grid_columnconfigure(0, weight=1)
        
        form_header = ctk.CTkLabel(form_panel, text="STUDENT DETAILS FORM", font=ctk.CTkFont(size=16, weight="bold"), text_color="#2d5cf6")
        form_header.grid(row=0, column=0, padx=25, pady=(25, 20), sticky="w")
        
        # Fields
        ctk.CTkLabel(form_panel, text="Full Name", font=ctk.CTkFont(size=12, weight="bold"), text_color="#aaa").grid(row=1, column=0, padx=25, pady=(10, 2), sticky="w")
        self.ent_reg_name = ctk.CTkEntry(form_panel, placeholder_text="Enter student's full name", height=40, font=ctk.CTkFont(size=13))
        self.ent_reg_name.grid(row=2, column=0, padx=25, pady=(2, 15), sticky="ew")
        
        ctk.CTkLabel(form_panel, text="Mobile Number", font=ctk.CTkFont(size=12, weight="bold"), text_color="#aaa").grid(row=3, column=0, padx=25, pady=(10, 2), sticky="w")
        self.ent_reg_mobile = ctk.CTkEntry(form_panel, placeholder_text="Enter 10-digit mobile number", height=40, font=ctk.CTkFont(size=13))
        self.ent_reg_mobile.grid(row=4, column=0, padx=25, pady=(2, 15), sticky="ew")
        
        ctk.CTkLabel(form_panel, text="Gender", font=ctk.CTkFont(size=12, weight="bold"), text_color="#aaa").grid(row=5, column=0, padx=25, pady=(10, 2), sticky="w")
        self.cmb_reg_gender = ctk.CTkComboBox(form_panel, values=["Boy", "Girl"], height=40, font=ctk.CTkFont(size=13))
        self.cmb_reg_gender.grid(row=6, column=0, padx=25, pady=(2, 25), sticky="ew")
        
        # Buttons
        self.btn_capture = ctk.CTkButton(
            form_panel, text="CAPTURE & REGISTER STUDENT", fg_color="#2ec4b6", hover_color="#20a396",
            font=ctk.CTkFont(size=14, weight="bold"), height=48,
            command=self.handle_registration
        )
        self.btn_capture.grid(row=7, column=0, padx=25, pady=(20, 10), sticky="ew")
        
        self.lbl_reg_status = ctk.CTkLabel(form_panel, text="", font=ctk.CTkFont(size=12), text_color="#e71d36")
        self.lbl_reg_status.grid(row=8, column=0, padx=25, pady=5)

    def init_directory_tab(self):
        directory = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.frames["directory"] = directory
        directory.grid_columnconfigure(0, weight=1)
        directory.grid_rowconfigure(1, weight=1)
        
        # Header
        hdr_frame = ctk.CTkFrame(directory, fg_color="#202020", corner_radius=12)
        hdr_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        hdr_frame.columnconfigure(0, weight=1)
        
        ctk.CTkLabel(
            hdr_frame, text="STUDENTS CURRENTLY INSIDE THE HOSTEL", 
            font=ctk.CTkFont(size=16, weight="bold"), text_color="#2d5cf6"
        ).grid(row=0, column=0, padx=20, pady=15, sticky="w")
        
        # Scrollable area
        self.dir_scroll = ctk.CTkScrollableFrame(directory, fg_color="#202020", corner_radius=12)
        self.dir_scroll.grid(row=1, column=0, padx=20, pady=(10, 20), sticky="nsew")

    def init_logs_tab(self):
        logs = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.frames["logs"] = logs
        logs.grid_columnconfigure(0, weight=1)
        logs.grid_rowconfigure(1, weight=1)
        
        # Header
        hdr_frame = ctk.CTkFrame(logs, fg_color="#202020", corner_radius=12)
        hdr_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        
        ctk.CTkLabel(
            hdr_frame, text="GATE PASSAGE MOVEMENT LOGS", 
            font=ctk.CTkFont(size=16, weight="bold"), text_color="#2d5cf6"
        ).grid(row=0, column=0, padx=20, pady=15, sticky="w")
        
        # Table Scroll container
        self.logs_scroll = ctk.CTkScrollableFrame(logs, fg_color="#202020", corner_radius=12)
        self.logs_scroll.grid(row=1, column=0, padx=20, pady=(10, 20), sticky="nsew")

    def init_outside_tab(self):
        outside = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        self.frames["outside"] = outside
        outside.grid_columnconfigure(0, weight=1)
        outside.grid_rowconfigure(1, weight=1)
        
        # Header
        hdr_frame = ctk.CTkFrame(outside, fg_color="#202020", corner_radius=12)
        hdr_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        
        ctk.CTkLabel(
            hdr_frame, text="STUDENTS CURRENTLY OUTSIDE THE HOSTEL", 
            font=ctk.CTkFont(size=16, weight="bold"), text_color="#e71d36"
        ).grid(row=0, column=0, padx=20, pady=15, sticky="w")
        
        # Table Scroll
        self.outside_scroll = ctk.CTkScrollableFrame(outside, fg_color="#202020", corner_radius=12)
        self.outside_scroll.grid(row=1, column=0, padx=20, pady=(10, 20), sticky="nsew")

    # Image loading utility
    def load_thumbnail(self, path, size=(45, 45)):
        """Safe thumbnail loader with rounded clipping."""
        try:
            if not path or not os.path.exists(path):
                raise FileNotFoundError
                
            img = Image.open(path).convert("RGBA").resize(size, Image.Resampling.LANCZOS)
        except Exception:
            # Fallback avatar
            img = Image.new("RGBA", size, (40, 40, 40, 255))
            draw = ImageDraw.Draw(img)
            draw.ellipse((2, 2, size[0]-3, size[1]-3), fill=(100, 100, 100, 255))
            
        # Circular Mask
        mask = Image.new("L", size, 0)
        draw_mask = ImageDraw.Draw(mask)
        draw_mask.ellipse((0, 0, size[0], size[1]), fill=255)
        
        rounded_img = Image.new("RGBA", size, (0, 0, 0, 0))
        rounded_img.paste(img, (0, 0), mask=mask)
        return ctk.CTkImage(light_image=rounded_img, dark_image=rounded_img, size=size)

    # Refresh page visual tables
    def refresh_dashboard_logs(self):
        """Displays the recent 5 activity entries in a stack card design."""
        for widget in self.dash_log_container.winfo_children():
            widget.destroy()
            
        logs_df = db_manager.get_movement_log()
        if logs_df.empty:
            lbl_empty = ctk.CTkLabel(self.dash_log_container, text="No gate movements logged yet.", text_color="#555", font=ctk.CTkFont(slant="italic"))
            lbl_empty.pack(pady=20)
            return
            
        # Take recent 5, reverse so latest is top
        recent = logs_df.tail(6).iloc[::-1]
        for _, row in recent.iterrows():
            action = row["Action"]
            color = "#2ec4b6" if action == "IN" else "#e71d36"
            action_text = "CHECKED IN" if action == "IN" else "CHECKED OUT"
            
            card = ctk.CTkFrame(self.dash_log_container, fg_color="#282828", corner_radius=8)
            card.pack(fill="x", pady=4, padx=5)
            
            # Action Badge (green or red side stripe)
            badge = ctk.CTkFrame(card, width=6, fg_color=color, corner_radius=0)
            badge.pack(side="left", fill="y")
            
            # Inner details container
            inner = ctk.CTkFrame(card, fg_color="transparent")
            inner.pack(side="left", fill="both", expand=True, padx=12, pady=8)
            
            lbl_name = ctk.CTkLabel(inner, text=f"{row['Name']}", font=ctk.CTkFont(size=14, weight="bold"), text_color="white")
            lbl_name.pack(anchor="w")
            
            lbl_sub = ctk.CTkLabel(inner, text=f"{action_text}  •  {row['Timestamp']}", font=ctk.CTkFont(size=11), text_color="#aaa")
            lbl_sub.pack(anchor="w")

    def refresh_directory_view(self):
        for widget in self.dir_scroll.winfo_children():
            widget.destroy()
            
        students_df = db_manager.get_currently_inside()
        if students_df.empty:
            ctk.CTkLabel(self.dir_scroll, text="All registered students are currently outside the hostel.", text_color="#666").pack(pady=40)
            return
            
        # Draw table headers
        hdr = ctk.CTkFrame(self.dir_scroll, fg_color="transparent")
        hdr.pack(fill="x", padx=10, pady=(5, 10))
        
        headers = [("Photo", 80), ("ID", 60), ("Student Name", 220), ("Mobile Number", 180), ("Gender", 120), ("Registration Date", 200)]
        for label, w in headers:
            lbl = ctk.CTkLabel(hdr, text=label.upper(), font=ctk.CTkFont(size=11, weight="bold"), text_color="#888", width=w, anchor="w")
            lbl.pack(side="left")
            
        # Draw student rows
        for _, row in students_df.iterrows():
            row_frame = ctk.CTkFrame(self.dir_scroll, fg_color="#262626", corner_radius=6)
            row_frame.pack(fill="x", pady=3, padx=5)
            
            # Load thumbnail
            photo = self.load_thumbnail(row["Photo Path"])
            lbl_photo = ctk.CTkLabel(row_frame, image=photo, text="", width=80, anchor="w")
            lbl_photo.pack(side="left", padx=(10, 0), pady=5)
            
            # Details
            ctk.CTkLabel(row_frame, text=f"#{row['ID']}", font=ctk.CTkFont(weight="bold"), text_color="#888", width=60, anchor="w").pack(side="left")
            ctk.CTkLabel(row_frame, text=row["Name"], font=ctk.CTkFont(size=13, weight="bold"), width=220, anchor="w").pack(side="left")
            ctk.CTkLabel(row_frame, text=row["Mobile Number"], font=ctk.CTkFont(size=13), width=180, anchor="w").pack(side="left")
            ctk.CTkLabel(row_frame, text=row["Gender"], font=ctk.CTkFont(size=13), width=120, anchor="w").pack(side="left")
            ctk.CTkLabel(row_frame, text=row["Registration Date"], font=ctk.CTkFont(size=13), text_color="#aaa", width=200, anchor="w").pack(side="left")

    def refresh_logs_view(self):
        for widget in self.logs_scroll.winfo_children():
            widget.destroy()
            
        logs_df = db_manager.get_movement_log()
        if logs_df.empty:
            ctk.CTkLabel(self.logs_scroll, text="No movement logs recorded yet.", text_color="#666").pack(pady=40)
            return
            
        hdr = ctk.CTkFrame(self.logs_scroll, fg_color="transparent")
        hdr.pack(fill="x", padx=10, pady=(5, 10))
        
        headers = [("Timestamp", 200), ("ID", 60), ("Name", 220), ("Mobile Number", 180), ("Gender", 120), ("Action Triggered", 140)]
        for label, w in headers:
            lbl = ctk.CTkLabel(hdr, text=label.upper(), font=ctk.CTkFont(size=11, weight="bold"), text_color="#888", width=w, anchor="w")
            lbl.pack(side="left")
            
        # Draw log rows (latest on top)
        for _, row in logs_df.iloc[::-1].iterrows():
            action = row["Action"]
            color = "#2ec4b6" if action == "IN" else "#e71d36"
            action_text = "CHECK-IN" if action == "IN" else "CHECK-OUT"
            
            row_frame = ctk.CTkFrame(self.logs_scroll, fg_color="#262626", corner_radius=6)
            row_frame.pack(fill="x", pady=3, padx=5)
            
            ctk.CTkLabel(row_frame, text=row["Timestamp"], font=ctk.CTkFont(size=13), text_color="#aaa", width=200, anchor="w").pack(side="left", padx=(10, 0), pady=8)
            ctk.CTkLabel(row_frame, text=f"#{row['ID']}", font=ctk.CTkFont(weight="bold"), text_color="#888", width=60, anchor="w").pack(side="left")
            ctk.CTkLabel(row_frame, text=row["Name"], font=ctk.CTkFont(size=13, weight="bold"), width=220, anchor="w").pack(side="left")
            ctk.CTkLabel(row_frame, text=row["Mobile Number"], font=ctk.CTkFont(size=13), width=180, anchor="w").pack(side="left")
            ctk.CTkLabel(row_frame, text=row["Gender"], font=ctk.CTkFont(size=13), width=120, anchor="w").pack(side="left")
            
            badge = ctk.CTkLabel(
                row_frame, text=action_text, font=ctk.CTkFont(size=11, weight="bold"), 
                text_color="white", fg_color=color, corner_radius=6, width=90, height=24
            )
            badge.pack(side="left", padx=10)

    def refresh_outside_view(self):
        for widget in self.outside_scroll.winfo_children():
            widget.destroy()
            
        outside_df = db_manager.get_currently_outside()
        if outside_df.empty:
            ctk.CTkLabel(self.outside_scroll, text="All students are currently inside the hostel.", text_color="#2ec4b6", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=40)
            return
            
        hdr = ctk.CTkFrame(self.outside_scroll, fg_color="transparent")
        hdr.pack(fill="x", padx=10, pady=(5, 10))
        
        headers = [("ID", 60), ("Name", 220), ("Mobile Number", 180), ("Gender", 120), ("Out Time", 200)]
        for label, w in headers:
            lbl = ctk.CTkLabel(hdr, text=label.upper(), font=ctk.CTkFont(size=11, weight="bold"), text_color="#888", width=w, anchor="w")
            lbl.pack(side="left")
            
        for _, row in outside_df.iterrows():
            row_frame = ctk.CTkFrame(self.outside_scroll, fg_color="#262626", corner_radius=6)
            row_frame.pack(fill="x", pady=3, padx=5)
            
            ctk.CTkLabel(row_frame, text=f"#{row['ID']}", font=ctk.CTkFont(weight="bold"), text_color="#888", width=60, anchor="w").pack(side="left", padx=(10, 0), pady=8)
            ctk.CTkLabel(row_frame, text=row["Name"], font=ctk.CTkFont(size=13, weight="bold"), width=220, anchor="w").pack(side="left")
            ctk.CTkLabel(row_frame, text=row["Mobile Number"], font=ctk.CTkFont(size=13), width=180, anchor="w").pack(side="left")
            ctk.CTkLabel(row_frame, text=row["Gender"], font=ctk.CTkFont(size=13), width=120, anchor="w").pack(side="left")
            ctk.CTkLabel(row_frame, text=row["Out Time"], font=ctk.CTkFont(size=13), text_color="#e71d36", width=200, anchor="w").pack(side="left")

    # Update functions running in Tkinter loops
    def update_stats(self):
        """Periodically recalculates stats to display in the bottom of the sidebar."""
        students_df = db_manager.get_registered_students()
        outside_df = db_manager.get_currently_outside()
        
        total = len(students_df)
        outside = len(outside_df)
        inside = total - outside
        
        if (total != self.last_stats["total"] or 
            inside != self.last_stats["inside"] or 
            outside != self.last_stats["outside"]):
            
            self.last_stats = {"total": total, "inside": inside, "outside": outside}
            self.lbl_stats_total.configure(text=f"Total Students: {total}")
            self.lbl_stats_inside.configure(text=f"Inside: {inside}")
            self.lbl_stats_outside.configure(text=f"Outside: {outside}")
            
        self.after(2000, self.update_stats)

    def poll_gui_events(self):
        """Checks for status updates from the background face recognition thread."""
        try:
            while True:
                event = gui_event_queue.get_nowait()
                if event["type"] == "toggle":
                    # Update active activity display card
                    name = event["name"]
                    action = event["action"]
                    gender = event["gender"]
                    act_time = event["time"]
                    
                    color = "#2ec4b6" if action == "IN" else "#e71d36"
                    act_text = "CHECKED IN" if action == "IN" else "CHECKED OUT"
                    
                    self.lbl_det_name.configure(text=name.upper(), text_color="white")
                    self.lbl_det_status.configure(text=f"{act_text} AT {act_time}", text_color=color)
                    
                    # Flash background of feedback card temporarily
                    self.detection_card.configure(fg_color="#303030")
                    self.after(500, lambda: self.detection_card.configure(fg_color="#202020"))
                    
                    # Trigger alert toast
                    toast_msg = f"{name} {act_text.lower()} successfully!"
                    self.show_toast(toast_msg, is_error=(action == "OUT"))
                    
                    # Refresh lists if we are currently looking at them
                    self.refresh_dashboard_logs()
                    if self.active_tab == "directory":
                        self.refresh_directory_view()
                    elif self.active_tab == "logs":
                        self.refresh_logs_view()
                    elif self.active_tab == "outside":
                        self.refresh_outside_view()
                elif event["type"] == "error":
                    # Display database warning visual toast in GUI
                    self.show_toast(event["message"], is_error=True)
                        
                gui_event_queue.task_done()
        except queue.Empty:
            pass
        self.after(100, self.poll_gui_events)

    def update_webcam_feed(self):
        """Reads frame, overlay labels/boxes, and sets it to Tkinter widgets."""
        global latest_frame, cap
        
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Mirror frame for intuitive view
                frame = cv2.flip(frame, 1)
                
                # Save frame copy safely for background thread to run face recognition on
                with frame_lock:
                    latest_frame = frame.copy()
                    faces_to_draw = list(recognized_faces) # Copy list safely
                
                # Draw boxes on frame for current frame
                h, w, _ = frame.shape
                for face in faces_to_draw:
                    top, right, bottom, left = face["box"]
                    name = face["name"]
                    student_id = face["id"]
                    
                    # Set border color: green if inside, red if outside, yellow if unknown
                    color = (255, 159, 28) # BGR Yellow
                    status_lbl = "Unknown"
                    
                    if student_id is not None:
                        # Find their status
                        students_df = db_manager.get_registered_students()
                        student_row = students_df[students_df["ID"] == student_id]
                        if not student_row.empty:
                            status = student_row.iloc[0]["Status"]
                            status_lbl = f"{name} ({status})"
                            color = (182, 196, 46) if status == "Inside" else (54, 29, 231) # Green / Red BGR
                            
                    # Draw rectangle around face
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    # Draw solid text bar below
                    cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
                    cv2.putText(
                        frame, status_lbl, (left + 6, bottom - 6), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1
                    )
                
                # Render to Tkinter Label (Dashboard Tab)
                if self.active_tab == "dashboard":
                    # Resize to fit dashboard panel size (~640x480)
                    frame_dash = cv2.resize(frame, (620, 465))
                    rgb_dash = cv2.cvtColor(frame_dash, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(rgb_dash)
                    img_ctk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(620, 465))
                    self.lbl_webcam.configure(image=img_ctk)
                    
                # Render to Tkinter Label (Registration Tab)
                elif self.active_tab == "register":
                    # Resize to fit registration preview panel
                    frame_reg = cv2.resize(frame, (620, 465))
                    rgb_reg = cv2.cvtColor(frame_reg, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(rgb_reg)
                    img_ctk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(620, 465))
                    self.lbl_reg_webcam.configure(image=img_ctk)
                    
        self.after(30, self.update_webcam_feed)

    # Action Handlers
    def handle_registration(self):
        """Validates fields, captures current webcam frame, detects faces, and registers student."""
        name = self.ent_reg_name.get().strip()
        mobile = self.ent_reg_mobile.get().strip()
        gender = self.cmb_reg_gender.get()
        
        self.lbl_reg_status.configure(text="", text_color="#e71d36")
        
        # Validation
        if not name:
            self.lbl_reg_status.configure(text="Name field is required!")
            return
        if not mobile or len(mobile) != 10 or not mobile.isdigit():
            self.lbl_reg_status.configure(text="Please enter a valid 10-digit mobile number!")
            return
            
        # Capture current frame
        frame_to_register = None
        with frame_lock:
            if latest_frame is not None:
                frame_to_register = latest_frame.copy()
                
        if frame_to_register is None:
            self.lbl_reg_status.configure(text="Camera feed not ready. Please wait.")
            return
            
        # Run face detection on the frame to ensure a single clear face is present
        rgb_frame = cv2.cvtColor(frame_to_register, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if len(face_locations) == 0:
            self.lbl_reg_status.configure(text="No face detected! Please look directly at the camera.")
            return
        if len(face_locations) > 1:
            self.lbl_reg_status.configure(text="Multiple faces detected! Only one person should be in view.")
            return
            
        # Save photo
        photo_filename = f"{name.replace(' ', '_')}_{mobile}.jpg"
        db_manager.init_db()  # Make sure folders exist
        photo_path = os.path.join(db_manager.FACES_DIR, photo_filename)
        
        try:
            cv2.imwrite(photo_path, frame_to_register)
        except Exception as e:
            self.lbl_reg_status.configure(text=f"Failed to write image: {e}")
            return
            
        # Add to Excel
        success, info = db_manager.register_student(name, mobile, gender, photo_filename)
        if not success:
            self.lbl_reg_status.configure(text=info)
            # Clean up the photo if Excel write fails
            if os.path.exists(photo_path):
                os.remove(photo_path)
            return
            
        # Success sound & alert
        play_sound("REGISTER")
        self.lbl_reg_status.configure(text="Student registered successfully!", text_color="#2ec4b6")
        self.show_toast(f"Registered {name} successfully!")
        
        # Clear fields
        self.ent_reg_name.delete(0, 'end')
        self.ent_reg_mobile.delete(0, 'end')
        
        # Reload registered faces encodings asynchronously
        threading.Thread(target=recognizer.load_registered_faces, daemon=True).start()

    def on_closing(self):
        """Cleans up camera resource and stops threads before exit."""
        global is_camera_running, cap
        is_camera_running = False
        
        # Stop speech thread
        speech_queue.put(None)
        
        if cap is not None:
            cap.release()
            
        self.destroy()

if __name__ == "__main__":
    # Create speech thread
    speech_thread = threading.Thread(target=speech_worker, daemon=True)
    speech_thread.start()
    
    # Start app
    app = HostelGatekeeperApp()
    
    # Start face recognition worker thread
    rec_thread = threading.Thread(target=face_recognition_thread, daemon=True)
    rec_thread.start()
    
    # Handle window close gracefully
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
