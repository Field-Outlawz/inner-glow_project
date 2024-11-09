import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
import chat_api
import webbrowser
import time
from datetime import datetime
import os
from deepface import DeepFace

# skin and color
class ChatClient:
    def __init__(self, master):
        self.master = master
        self.master.title("Chat Client")
        self.master.geometry("700x500")

        self.load_window()

    def load_window(self):
        title_label = ctk.CTkLabel(self.master, text="Welcome to the Health & Skincare Analysis Tool", font=("Arial", 14))
        title_label.pack()

        camera_button = ctk.CTkButton(self.master, text="Start Health & Skincare Analysis", command = self.capture_and_analyze, font=("Arial", 12))
        camera_button.pack(pady=10)

        
        website_button = ctk.CTkButton(self.master, text="Face Recognition Website", command = self.open_website, font=("Arial", 12))
        website_button.pack(pady=10)

    def display_message(self, prompt):
        message_data = chat_api.message(prompt)

        self.chat_area.configure(state='normal') 
        self.chat_area.insert(ctk.END, message_data)
        self.chat_area.configure(state='disabled') 
        self.chat_area.see(ctk.END)  

    def load_chatbot(self):
        self.remove_screen(self.master)

        
        self.chat_area = ctk.CTkTextbox(self.master, state='disabled', wrap='word')
        self.chat_area.grid(row=0, column=1, sticky="nsew")

        
        self.message_entry = ctk.CTkEntry(self.master, width=300)
        self.message_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        
        self.send_button = ctk.CTkButton(self.master, text="Send", command=lambda: self.display_message(self.message_entry.get()))
        self.send_button.grid(row=1, column=2, padx=10, pady=5)

        self.master.grid_rowconfigure(0, weight=1)  
        self.master.grid_columnconfigure(1, weight=1)  
        self.master.grid_columnconfigure(2, weight=0) 

    def remove_screen(self, frame):
        for child in frame.winfo_children():
            child.destroy()

    def open_analysis(self):
        self.remove_screen(self.master)

        # analysis text
        results = ctk.CTkLabel(self.master, text=self.analysis_text, font=("Arial", 12))
        results.pack()

        AI_analysis = ctk.CTkLabel(self.master, text = chat_api.message(self.analysis_text + "The following data is what we have analysed from a face. Can you summarize it."))
        AI_analysis.pack()

        emotion_text = "The dominant emotion showed by your face is: " + str(self.result["dominant_emotion"])

        emotion = ctk.CTkLabel(self.master, text = emotion_text)
        emotion.pack()

        # Quit button
        quit = ctk.CTkButton(self.master, text="Quit")
        quit.pack()

        chat = ctk.CTkButton(self.master, text="Chat", command = self.load_chatbot)
        chat.pack()

    def analyze_health_and_skin(self, face_roi):
        
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
        avg_saturation = np.mean(hsv[:, :, 1])
        avg_value = np.mean(hsv[:, :, 2])

        # skin critrions
        skin_type = "Unknown"
        if avg_saturation < 60 and avg_value < 70:
            skin_type = "Dry Skin"
        elif avg_saturation > 150 and avg_value > 150:
            skin_type = "Oily Skin"
        else:
            skin_type = "Normal Skin"

        # criterions
        brightness = np.mean(face_roi)
        if brightness < 80:
            health_status = "Low brightness: potential fatigue or dehydration."
        elif brightness > 170:
            health_status = "High brightness: possibly well-hydrated or lighter skin."
        else:
            health_status = "Normal brightness and health tone."

        
        skincare_advice = self.get_skincare_advice(skin_type)
        health_advice = self.get_health_advice(brightness)

        return skin_type, health_status, skincare_advice, health_advice

    def get_skincare_advice(self, skin_type):
        if skin_type == "Dry Skin":
            return (
                "Dry Skin Detected.\n"
                "- Use hydrating cleansers\n"
                "- Avoid hot water on skin\n"
                "- Moisturize with hyaluronic acid\n"
                "- Avoid alcohol-based products\n"
            )
        elif skin_type == "Oily Skin":
            return (
                "Oily Skin Detected.\n"
                "- Use oil-free products\n"
                "- Cleanse with salicylic acid\n"
                "- Avoid heavy creams\n"
                "- Use mattifying moisturizers\n"
            )
        else:
            return (
                "Normal Skin Detected.\n"
                "- Maintain balanced routine\n"
                "- Hydrate regularly\n"
                "- Use sunscreen with SPF 30+\n"
            )

    def get_health_advice(self, brightness):
        if brightness < 80:
            return (
                "Health Note:\n"
                "- Signs of fatigue or dehydration detected\n"
                "- Ensure adequate hydration\n"
                "- Aim for regular sleep patterns\n"
            )
        elif brightness > 170:
            return (
                "Health Note:\n"
                "- Good skin brightness detected\n"
                "- Maintain hydration\n"
                "- Continue a balanced diet\n"
            )
        else:
            return (
                "Health Note:\n"
                "- Healthy skin tone observed\n"
                "- Continue a balanced lifestyle\n"
            )

    
    def capture_and_analyze(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open video stream.")
            return
        
        start_time = time.time()
        face_detected = False
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while True:
            ret, frame = cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture image.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces:
                # drae rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face_detected = True

            # show box
            cv2.imshow("Scanning... Please wait", frame)

            # Capture the face image after 3 seconds-
            if face_detected and (time.time() - start_time) >= 3:
                face_roi = frame[y:y+h, x:x+w]
                skin_type, health_status, skincare_advice, health_advice = self.analyze_health_and_skin(face_roi)

                pic = self.save_captured_image(face_roi)

                self.result = DeepFace.analyze('captured_face.jpg', actions=['emotion'], enforce_detection=False)
                
                self.analysis_text = (
                    f"Skin Type: {skin_type}\n"
                    f"{skincare_advice}\n"
                    f"Health Status: {health_status}\n"
                    f"{health_advice}\n"
                )


                
                cap.release()
                cv2.destroyAllWindows()
                self.open_analysis()
                return

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    
    def save_captured_image(self, face_roi):
        folder_path = "directory2/photos"
        if not os.path.exists("directory2/photos"):
            os.makedirs("directory2/photos")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join("directory2/photos", f"face_capture_{timestamp}.jpg")
        cv2.imwrite(file_path, face_roi)
    
        print(f"Image saved to directory2/photos")

        return file_path

    def open_website(self):
        print("Hello")
        webbrowser.open_new_tab("https://innerglow-f03fcb.webflow.io")
        print("Hello")
        # webbrowser.open("https://innerglow-f03fcb.webflow.io")  # Replace with actual URL

if __name__ == "__main__":
    ctk.set_appearance_mode("dark")  
    ctk.set_default_color_theme("blue")  
    root = ctk.CTk()  
    client = ChatClient(root)
    root.mainloop()