"""User accounts stored in SQLite. Passwords are salted + hashed (never plain text)."""
import hashlib
import hmac
import os
import re
import sqlite3

from .data_utils import ROOT

DB_PATH = ROOT / "users.db"
ITERATIONS = 200_000


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users ("
        "username TEXT PRIMARY KEY, full_name TEXT, email TEXT, "
        "salt BLOB NOT NULL, password_hash BLOB NOT NULL)")
    return conn


def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)


def validate(username, password, email=""):
    """Return an error message, or None if everything is fine."""
    if len(username) < 3:
        return "Username must be at least 3 characters."
    if len(password) < 6:
        return "Password must be at least 6 characters."
    if not (re.search(r"[A-Za-z]", password) and re.search(r"\d", password)):
        return "Password must contain at least one letter and one digit."
    if email and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return "Please enter a valid e-mail address."
    return None


def register(username, password, full_name="", email=""):
    """Create an account. Returns (ok, message)."""
    username = username.strip()
    error = validate(username, password, email.strip())
    if error:
        return False, error
    salt = os.urandom(16)
    try:
        with _connect() as conn:
            conn.execute("INSERT INTO users VALUES (?,?,?,?,?)",
                         (username, full_name.strip(), email.strip(), salt,
                          _hash(password, salt)))
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    return True, "Account created. You can log in now."


def login(username, password):
    """Check credentials. Returns (ok, message)."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT salt, password_hash FROM users WHERE username = ?",
            (username.strip(),)).fetchone()
    if row and hmac.compare_digest(_hash(password, row[0]), row[1]):
        return True, "Login successful."
    return False, "Wrong username or password."
