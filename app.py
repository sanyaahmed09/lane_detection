from flask import Flask, render_template, Response, request, jsonify, send_from_directory, session, redirect, url_for
import sqlite3
import os
import subprocess
from lines import generate_video_frames

"""
PSUDOCODE:
Initialize Flask app with secret key
Define create_database() to set up user and log tables in SQLite
Call create_database() to ensure tables exist

Define login():
    If POST request:
        Get username, password, and action from form
        Connect to database
        If signup, insert user and handle duplicate usernames
        If login, check credentials and start session
        Close database connection
    Render login page with error if needed

Define logout(): Clear session and redirect to login
Define screen(): Ensure user is logged in, else redirect to login
Define start_lane_detection(): Run lane detection script asynchronously
Define video_feed(): Stream frames from generate_video_frames()
Define serve_video(filename): Serve static video files

Define log_action():
    Get action from JSON request
    Store action in database
    Return success response

Define logs(): Retrieve and return last 10 logged actions

Run app in debug mode if executed directly
"""

app = Flask(__name__)
app.secret_key = "supersecretkey"

def create_database():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

create_database()

@app.route("/", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        action = request.form["action"]

        conn = sqlite3.connect("database.db")
        cursor = conn.cursor()

        if action == "signup":
            try:
                cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                conn.commit()
                session["username"] = username  # Automatically log in after signup
                return redirect(url_for("screen"))
            except sqlite3.IntegrityError:
                error = "Username already exists. Try another one."
        elif action == "login":
            cursor.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
            user = cursor.fetchone()
            if user:
                session["username"] = username
                return redirect(url_for("screen"))
            else:
                error = "Invalid username or password."

        conn.close()

    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))

@app.route("/screen")
def screen():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("screen.html")

@app.route("/start_lane_detection", methods=["POST"])
def start_lane_detection():
    subprocess.Popen(["python", "lines.py"])
    return jsonify({"message": "Lane Detection Started!"})

@app.route("/video_feed")
def video_feed():
    return Response(generate_video_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video/<filename>")
def serve_video(filename):
    return send_from_directory("static", filename, as_attachment=False)

@app.route("/log_action", methods=["POST"])
def log_action():
    data = request.json
    action = data.get("action", "")
    
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO logs (action) VALUES (?)", (action,))
    conn.commit()
    conn.close()
    
    return jsonify({"message": "Action logged!"})

@app.route("/logs")
def logs():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT action FROM logs ORDER BY id DESC LIMIT 10")
    logs = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    return jsonify(logs)

if __name__ == "__main__":
    app.run(debug=True)
