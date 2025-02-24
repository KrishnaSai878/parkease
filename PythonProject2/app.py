from flask import Flask, render_template, redirect, url_for, request
import sqlite3
from datetime import datetime
import cv2
import imutils
import numpy as np
import easyocr
import re
import time

app = Flask(__name__)

# Initialize EasyOCR Reader for English (using CPU)
reader = easyocr.Reader(['en'], gpu=False)

# Database filename
DB_FILENAME = "parking.db"


# Initialize the database and create tables if they don't exist
def init_db():
    conn = sqlite3.connect(DB_FILENAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            plate_number TEXT PRIMARY KEY,
            entry_time TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS parking_history (
            plate_number TEXT,
            entry_time TEXT,
            exit_time TEXT,
            duration INTEGER
        )
    """)

    conn.commit()
    conn.close()


# Initialize the database and create tables if they don't exist
init_db()


# Create operation: Vehicle Entry
def create_vehicle_entry(plate):
    plate = plate.upper()
    conn = sqlite3.connect(DB_FILENAME)
    cursor = conn.cursor()

    cursor.execute("SELECT entry_time FROM vehicles WHERE plate_number = ?", (plate,))
    record = cursor.fetchone()

    if record is None:
        # New entry for vehicle
        entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO vehicles (plate_number, entry_time) VALUES (?, ?)", (plate, entry_time))
        conn.commit()
        print(f"Vehicle {plate} entered at {entry_time}")
    else:
        print(f"Vehicle {plate} is already inside.")

    conn.close()


# Read operation: Viewing Parking History
@app.route('/history')
def history():
    conn = sqlite3.connect(DB_FILENAME)
    cursor = conn.cursor()

    cursor.execute("SELECT plate_number, entry_time, exit_time, duration FROM parking_history ORDER BY rowid DESC")
    history_records = cursor.fetchall()

    conn.close()

    return render_template('history.html', history=history_records)


# Update operation: Modify Exit Time and Duration
@app.route('/update/<plate>', methods=['GET', 'POST'])
def update_exit_time(plate):
    conn = sqlite3.connect(DB_FILENAME)
    cursor = conn.cursor()

    if request.method == 'POST':
        exit_time = request.form['exit_time']

        # Retrieve the entry time of the vehicle
        cursor.execute("SELECT entry_time FROM vehicles WHERE plate_number = ?", (plate,))
        record = cursor.fetchone()

        if record:
            entry_time_str = record[0]
            entry_time_dt = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
            exit_time_dt = datetime.strptime(exit_time, "%Y-%m-%d %H:%M:%S")

            # Calculate the parking duration
            duration = int((exit_time_dt - entry_time_dt).total_seconds() / 60)

            # Insert the updated exit record into the parking history
            cursor.execute("""
                INSERT INTO parking_history (plate_number, entry_time, exit_time, duration)
                VALUES (?, ?, ?, ?)
            """, (plate, entry_time_str, exit_time, duration))

            # Remove the vehicle from the vehicles table (they have exited)
            cursor.execute("DELETE FROM vehicles WHERE plate_number = ?", (plate,))
            conn.commit()
            print(f"Vehicle {plate} exit time updated.")

            return redirect(url_for('history'))
        else:
            print("Vehicle not found.")

    conn.close()
    return render_template('update_exit_time.html', plate=plate)


# Delete operation: Remove a Vehicle Record
@app.route('/delete/<plate>', methods=['POST'])
def delete_vehicle(plate):
    conn = sqlite3.connect(DB_FILENAME)
    cursor = conn.cursor()

    # Delete vehicle from 'vehicles' table
    cursor.execute("DELETE FROM vehicles WHERE plate_number = ?", (plate,))
    conn.commit()

    print(f"Vehicle {plate} has been removed from the parking lot.")
    conn.close()

    return redirect(url_for('history'))


# Process license plate with EasyOCR
def process_license_plate():
    cap = cv2.VideoCapture(0)
    detected_plate = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera.")
            break

        # Resize and preprocess the frame
        img = cv2.resize(frame, (620, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(bfilter, 10, 200)

        # Find contours and sort by area
        contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        screenCnt = None
        for c in contours:
            approx = cv2.approxPolyDP(c, 15, True)
            if len(approx) == 4:
                screenCnt = approx
                break

        if screenCnt is not None:
            cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [screenCnt], 0, 255, -1)
            new_image = cv2.bitwise_and(img, img, mask=mask)

            # Crop the detected region
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))
            cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

            # Use EasyOCR to extract text from the cropped image
            result = reader.readtext(cropped)
            read = "".join([text for (_, text, _) in result])

            # Convert to uppercase and clean the result
            read = read.upper()
            print("Uppercase OCR result:", read)
            read = ''.join(e for e in read if e.isalnum())

            # Remove unwanted country codes
            ignore_list = ["IND", "USA", "UK", "AUS", "CAN", "DEU", "FRA", "ITA", "ESP", "CHN", "JPN", "KOR"]
            for country in ignore_list:
                read = read.replace(country, "")
            print("Modified OCR result:", read)

            # Optionally, try regex to match typical Indian plate pattern (e.g., RJ14CV0002)
            pattern = r"[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}"
            match = re.search(pattern, read)
            if match:
                detected_plate = match.group()
                print("Regex matched plate:", detected_plate)
            else:
                # Fallback: if the filtered result meets criteria, use it
                if 6 <= len(read) <= 11:
                    detected_plate = read
                else:
                    print("Detected plate did not meet length criteria after filtering.")
                    detected_plate = None

            if detected_plate:
                create_vehicle_entry(detected_plate)
                # Save images for debugging (instead of displaying them)
                cv2.imwrite("debug_cropped.jpg", cropped)
                cv2.imwrite("debug_detection.jpg", img)
                time.sleep(3)
                break
        else:
            print("No contour detected")

        # For testing, break after one iteration
        break

    cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error as e:
        print("cv2.destroyAllWindows() failed:", e)
    return detected_plate


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/scan', methods=['POST'])
def scan():
    detected_plate = process_license_plate()
    return redirect(url_for('index'))


@app.route('/auth')
def auth():
    return render_template('authentication.html')


@app.route('/help')
def help_page():
    return render_template('help.html')


@app.route('/homepage')
def homepage():
    return render_template('homepage.html')


@app.route('/status')
def status_page():
    return render_template('statuspage.html')


@app.route('/updatebalance')
def update_balance_page():
    return render_template('updatebalance.html')


@app.route('/admin')
def admin_page():
    return render_template('adminpage.html')


if __name__ == '__main__':
    app.run(debug=True)