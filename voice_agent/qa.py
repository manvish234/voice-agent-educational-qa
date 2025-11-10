# voice_agent/qa.py
import os
import re
import uuid
import tempfile
import datetime
import string
from dotenv import load_dotenv
from rapidfuzz import fuzz
from gtts import gTTS
from google import genai
from google.genai import types
import pandas as pd
from typing import Optional, Dict
import json
import requests
import time
import logging
import traceback

# Project paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INFO_FILE = os.path.join(BASE_DIR, "info.txt")
OUTBOX_PATH = os.path.join(BASE_DIR, "outbox.jsonl")

# --------------------------
# Setup
# --------------------------
load_dotenv()

# configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Gemini key (must be set in .env)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing in environment")

# MyClassBoard endpoint + auth (set in .env)
ENQUIRY_API_URL = os.getenv(
    "ENQUIRY_API_URL",
    "https://api.myclassboard.com/api/EnquiryService/Save_EnquiryDetails"
)
MYCLASSBOARD_API_KEY = os.getenv("MYCLASSBOARD_API_KEY", "")   # header value for api_Key
MYCLASSBOARD_AUTH = os.getenv("MYCLASSBOARD_AUTH", "")         # header value for Authorization

# Default numeric IDs (if you don't want to ask these)
DEFAULT_ORGANISATION_ID = int(os.getenv("ORGANISATION_ID", "45"))
DEFAULT_BRANCH_ID = int(os.getenv("BRANCH_ID", "79"))
DEFAULT_ACADEMIC_YEAR_ID = int(os.getenv("ACADEMIC_YEAR_ID", "17"))
DEFAULT_CLASS_ID = int(os.getenv("CLASS_ID", "477"))

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

SYS_PROMPT_QA = (
    "You are a clear, friendly voice assistant. "
    "Always explain things in short, simple sentences. "
    "Make it sound natural, as if spoken to a beginner. "
    "Do not just read. Rephrase to sound human."
)

qa_chat = client.chats.create(
    model="gemini-1.5-flash",
    config=types.GenerateContentConfig(system_instruction=SYS_PROMPT_QA),
)

CSV_FILE = os.path.join(BASE_DIR, "qa.csv")

# Fields to extract
REQUIRED_FIELDS = [
    "What is your full name?",
    "What is your school name?",
    "What is your email address?",
    "Which standard are you studying in?",
    "What is your child's gender? (male/female/other)",
    "What is your father's name?",
    "What is your father's email address?",
    "Mobile number (optional):",
    "Date of birth (YYYY-MM-DD) (optional):"
]

# Function to clean and normalize the extracted data
def clean_field(field_value):
    if not field_value:
        return ""
    # Remove extra spaces, normalize case, and handle common mishearings
    field_value = field_value.strip()
    field_value = re.sub(r"\s+", " ", field_value)  # Remove extra spaces
    return field_value

# Function to parse the text file and extract required fields
def extract_fields(filepath):
    with open(filepath, "r") as file:
        data = file.read()

    # Split entries by "---- New Entry ----"
    entries = data.split("---- New Entry ----")
    parsed_entries = []

    for entry in entries:
        entry_data = {}
        for field in REQUIRED_FIELDS:
            # Extract the field value using regex
            match = re.search(rf"{re.escape(field)}\s*(.*)", entry, re.IGNORECASE)
            if match:
                entry_data[field] = clean_field(match.group(1))
            else:
                entry_data[field] = ""  # Field not found
        parsed_entries.append(entry_data)

    return parsed_entries

class VoiceAgent:
    def __init__(self):
        self.df = None
        self.qa_rows = []
        self._load_csv(CSV_FILE)

        self._mode = "clusters"
        self._active_cluster = None

        self._last_candidates = []
        self._last_cluster_candidates = []
        self._last_answer_index = None

        self._inquiry_state = None
        # Questions the assistant will ask when user says HELP (in order)
        self._inquiry_questions = REQUIRED_FIELDS

        self._message_cache = {}
        try:
            self._seed_inquiry_messages()
        except Exception:
            pass

    # ---------------------------
    # Gemini-driven message helper
    # ---------------------------
    def _gen_message(self, key: str, context: Optional[Dict] = None, fallback: Optional[str] = None) -> str:
        ctx = context or {}
        cache_key = f"{key}:{str(sorted(ctx.items()))}"
        if cache_key in self._message_cache:
            return self._message_cache[cache_key]

        try:
            prompt = (
                f"Produce a short (1-2 sentence), friendly spoken reply for the message key: {key}.\n"
                f"Context: {ctx}\n"
                "Constraints: Keep it concise and suitable for TTS playback. Reply with only the message text."
            )
            resp = qa_chat.send_message(
                [prompt],
                config=types.GenerateContentConfig(
                    system_instruction=SYS_PROMPT_QA,
                    temperature=0.6,
                    max_output_tokens=80,
                ),
            )
            text = (resp.text or "").strip()
            if not text:
                raise ValueError("Empty response from Gemini for message key")
            self._message_cache[cache_key] = text
            return text
        except Exception:
            return fallback or "Sorry, I couldn't prepare a reply right now."

    def _seed_inquiry_messages(self):
        # keep simple TTS-friendly prompts; we cache them to avoid frequent Gemini calls
        self._message_cache.clear()
        self._message_cache["inquiry_intro:[]"] = "I need a few details. You may say 'quit' at any time to stop."
        self._message_cache["inquiry_saved:[]"] = "Thanks — your details are saved. Say a cluster name to continue."
        self._message_cache["inquiry_email_invalid:[]"] = "I didn't catch a valid email. Please spell it using 'at' and 'dot' if needed."

        for i, q in enumerate(self._inquiry_questions, start=1):
            key = f"inquiry_q_{i}:[]"
            self._message_cache[key] = q

    # ===========================
    # Robust CSV loader
    # ===========================
    def _load_csv(self, csv_path: str):
        try:
            full_path = csv_path if os.path.isabs(csv_path) else os.path.join(BASE_DIR, csv_path)
            if not os.path.exists(full_path):
                logger.warning(f"CSV not found at: {full_path}")
                self.df = None
                self.qa_rows = []
                return

            logger.info(f"Loading CSV from: {full_path}")
            df = pd.read_csv(full_path, encoding="utf-8-sig", dtype=str).fillna("")

            colmap = {c.lower().strip(): c for c in df.columns}
            if not any(k in colmap for k in ("cluster", "question", "answer")):
                df2 = pd.read_csv(full_path, header=None, encoding="utf-8-sig", dtype=str).fillna("")
                if len(df2.columns) >= 3:
                    potential_header = df2.iloc[0].tolist()
                    df = df2[1:].copy()
                    df.columns = [str(h).strip() for h in potential_header]
                    colmap = {c.lower().strip(): c for c in df.columns}
                else:
                    logger.error("CSV appears malformed (not enough columns).")
                    self.df = None
                    self.qa_rows = []
                    return

            needed = ["cluster", "question", "answer"]
            if not all(n in colmap for n in needed):
                logger.error(f"CSV must have columns: Cluster, Question, Answer (any case). Got: {list(df.columns)}")
                self.df = None
                self.qa_rows = []
                return

            rename_map = {
                colmap["cluster"]: "Cluster",
                colmap["question"]: "Question",
                colmap["answer"]: "Answer"
            }
            if "resource" in colmap:
                rename_map[colmap["resource"]] = "RESOURCE"

            df = df.rename(columns=rename_map)
            if "RESOURCE" not in df.columns:
                df["RESOURCE"] = ""

            df["Cluster"] = df["Cluster"].astype(str).str.strip()
            df["Question"] = df["Question"].astype(str).str.strip()
            df["Answer"] = df["Answer"].astype(str).str.strip()
            df["RESOURCE"] = df["RESOURCE"].astype(str).str.strip()

            df = df[df["Cluster"].astype(bool) & df["Question"].astype(bool) & df["Answer"].astype(bool)].copy()

            self.df = df.reset_index(drop=True)
            self.qa_rows = list(self.df[["Cluster", "Question", "Answer", "RESOURCE"]].itertuples(index=False, name=None))
            logger.info(f"Loaded {len(self.qa_rows)} QA rows (columns: {list(self.df.columns)})")
        except Exception as e:
            logger.exception(f"CSV load error: {e}")
            self.df = None
            self.qa_rows = []

    # ===========================
    # Normalization + utilities
    # ===========================
    def _normalize(self, text: str) -> str:
        if not text:
            return ""
        if not all(ord(c) < 128 for c in text):
            return ""
        text = text.strip().lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text

    def get_clusters(self):
        if self.df is None:
            return []
        clusters = sorted(self.df["Cluster"].dropna().unique().tolist())
        return clusters

    def get_questions_for_cluster(self, cluster_name: str):
        if not cluster_name or self.df is None:
            return []
        sub = self.df[self.df["Cluster"].str.lower() == str(cluster_name).lower()]
        return sub["Question"].tolist()
    
    
    # Robust gender normalization (handles STT mis-hearings like "mail")
    def _normalize_gender_text(self, text: str) -> Optional[str]:
        if not text:
            return None
        t = str(text).strip().lower()

        # Check numeric (1/2/3)
        m = re.search(r"\b([1-3])\b", t)
        if m:
            mapping = {'1': 'male', '2': 'female', '3': 'other'}
            return mapping.get(m.group(1))

        # Try fuzzy matching against expected words
        candidates = {"male": "male", "female": "female", "other": "other"}
        best = None
        best_score = 0
        for k in candidates:
            score = fuzz.partial_ratio(t, k)
            if score > best_score:
                best_score = score
                best = candidates[k]
        if best_score >= 60:
            return best

        # common STT mis-hearings
        if "mail" in t or "mal" in t or re.search(r"\bm[ea]l\b", t):
            return "male"
        if "female" in t or t.startswith("f ") or re.search(r"\bfem", t):
            return "female"
        if "other" in t or "other" in t:
            return "other"
        return None


    # matching/search helpers
    def score_match(self, user_text, candidate):
        u = self._normalize(user_text)
        v = self._normalize(candidate)
        if not u or not v:
            return 0.0
        tsr = fuzz.token_sort_ratio(u, v)
        pr = fuzz.partial_ratio(u, v)
        wr = fuzz.WRatio(u, v)
        len_ratio = min(len(u), len(v)) / max(len(u), len(v))
        score = 0.35 * tsr + 0.30 * pr + 0.25 * wr + 0.10 * 100 * len_ratio
        return round(score, 2)

    def _best_matches(self, user_text, pool):
        scored = []
        for idx, q in pool:
            s = self.score_match(user_text, q)
            scored.append((q, s, idx))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def find_answer_local(self, user_text, top_k=5):
        pool = [(i, q) for i, (_, q, _, _) in enumerate(self.qa_rows)]
        ranked = self._best_matches(user_text, pool)[:top_k]
        if not ranked:
            return None, []
        best_q, best_score, best_idx = ranked[0]
        if best_score >= 85:
            self._last_answer_index = best_idx
            return self.qa_rows[best_idx][2], []
        elif best_score >= 70:
            return None, ranked
        else:
            return None, []

    def find_answer_in_cluster(self, user_text, cluster_name, top_k=5):
        indices = [(i, q) for i, (c, q, _, _) in enumerate(self.qa_rows) if c.lower() == cluster_name.lower()]
        ranked = self._best_matches(user_text, indices)[:top_k]
        if not ranked:
            return None, []
        best_q, best_score, best_idx = ranked[0]
        if best_score >= 85:
            self._last_answer_index = best_idx
            return self.qa_rows[best_idx][2], []
        elif best_score >= 70:
            return None, ranked
        else:
            return None, []

    def summarize_for_speech(self, answer: str) -> str:
        if not answer:
            return ""
        parts = re.split(r"(?<=[.!?])\s+", answer.strip())
        return " ".join(parts[:2]) if parts else answer

    def query_gemini(self, user_text: str) -> dict:
        try:
            resp = qa_chat.send_message(
                [f"User asked: {user_text}\n\nAnswer in spoken style."],
                config=types.GenerateContentConfig(
                    system_instruction=SYS_PROMPT_QA,
                    temperature=0.7,
                    max_output_tokens=200,
                ),
            )
            answer = (resp.text or "").strip()
            if not answer:
                raise ValueError("Empty response from Gemini")
            short = self.summarize_for_speech(answer)
            return {"status": "ok", "mode": "ai", "text": short, "full_answer": answer, "tts": self.make_tts(short), "resource": ""}
        except Exception:
            not_found_msg = self._gen_message("not_found", {}, fallback="Sorry, I don't know the answer to that. Please try rephrasing or ask something else.")
            return {"status": "ok", "mode": "not_found", "text": not_found_msg, "full_answer": "", "tts": self.make_tts(not_found_msg), "resource": ""}

    def transcribe_audio(self, file_path: str) -> str:
        try:
            with open(file_path, "rb") as f:
                audio_bytes = f.read()

            def _try_mime(mime):
                part = types.Part(inline_data=types.Blob(mime_type=mime, data=audio_bytes))
                resp = client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=[part],
                    config=types.GenerateContentConfig(
                        system_instruction="You are a transcription engine. Return only the transcribed text in English."
                    ),
                )
                return (resp.text or "").strip()

            text = _try_mime("audio/webm") or _try_mime("audio/wav")
            return text
        except Exception as e:
            return f"⚠️ STT error: {e}"

    def make_tts(self, text: str) -> str:
        try:
            tts = gTTS(text or "Sorry, I have nothing to say.")
            fname = f"tts_{uuid.uuid4().hex}.mp3"
            out = os.path.join(tempfile.gettempdir(), fname)
            tts.save(out)
            return fname
        except Exception:
            return ""

    def fuzzy_match(self, user_text: str, candidates: list, threshold: float = 70.0):
        if not user_text or not candidates:
            return None
        scored = [(c, self.score_match(user_text, c)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored and scored[0][1] >= threshold:
            return scored[0][0]
        return None

    # ===========================
    # Helpers for payload building + sending (stdlib only)
    # ===========================
    def _safe_int(self, text: str, default: int = 0) -> int:
        if not text:
            return default
        m = re.search(r"(\d+)", str(text))
        try:
            return int(m.group(1)) if m else default
        except Exception:
            return default

    def _parse_gender(self, text: str) -> int:
        if not text:
            return 1
        t = str(text).strip().lower()
        if t in ("male", "m", "1"):
            return 1
        if t in ("female", "f", "2"):
            return 2
        return 3

    def _build_enquiry_payload(self, answers):
        """
        Build the payload in the exact MyClassBoard key names expected in your curl example.
        Produces keys:
          OrganisationID, BranchID, AcademicYearID, ClassID,
          StudentName, Gender (1/2/3), FatherName, FatherEmailID,
          StudentEmailID, mobileNo, dob, enquirySource, remarks, genderText, schoolName, admissionClass
        """
        # extract using the same question strings your JS uses
        student_name = answers.get("What is your full name?", "") or ""
        school_name = answers.get("What is your school name?", "") or ""
        student_email = answers.get("What is your email address?", "") or ""
        standard = answers.get("Which standard are you studying in?", "") or ""
        gender_text = answers.get("What is your child's gender? (male/female/other)", "") or ""
        father_name = answers.get("What is your father's name?", "") or ""
        father_email = answers.get("What is your father's email address?", "") or ""
        mobile_no = answers.get("Mobile number (optional):", "") or ""
        dob = answers.get("Date of birth (YYYY-MM-DD) (optional):", "") or ""

        # Map text -> numeric gender (1=male,2=female,3=other)
        gender_num = self._parse_gender(gender_text)

        # Compose payload with exact key names expected by the API
        payload = {
            # IDs: use default values if not provided in answers
            "OrganisationID": str(DEFAULT_ORGANISATION_ID),
            "BranchID": str(DEFAULT_BRANCH_ID),
            "AcademicYearID": str(DEFAULT_ACADEMIC_YEAR_ID),
            # ClassID in original curl was numeric; send integer if possible
            "ClassID": int(DEFAULT_CLASS_ID),

            # main student/family data (matching your curl)
            "StudentName": student_name,
            "Gender": gender_num,
            "FatherName": father_name,
            "FatherEmailID": father_email,
            # optional: the API accepts StudentEmailID
            "StudentEmailID": student_email,
            "mobileNo": mobile_no,
            "dob": dob,

            # extra informational fields
            "enquirySource": "Website Chatbot",
            "remarks": "Collected via chatbot",

            # keep both text and standardized fields for traceability
            "genderText": gender_text,
            "schoolName": school_name,
            "admissionClass": standard
        }

        return payload


    def _send_enquiry(self, payload: dict, timeout: int = 10) -> dict:
        """Send enquiry with verbose logging and outbox fallback (no background worker)."""
        if not ENQUIRY_API_URL:
            logger.error("ENQUIRY_API_URL is not configured.")
            return {"ok": False, "reason": "ENQUIRY_API_URL not configured"}

        headers = {
            "Content-Type": "application/json",
            "api_Key": MYCLASSBOARD_API_KEY or "",
            "Authorization": MYCLASSBOARD_AUTH or ""
        }
        
        # Debug: log actual values (redacted)
        logger.info("API Key present: %s", bool(MYCLASSBOARD_API_KEY))
        logger.info("Auth present: %s", bool(MYCLASSBOARD_AUTH))
        logger.info("API URL: %s", ENQUIRY_API_URL)

        logger.info("Attempting to send enquiry to %s", ENQUIRY_API_URL)
        logger.debug("Headers: %s", {k: ("<redacted>" if k in ("X-API-Key","Authorization") else v) for k,v in headers.items()})
        logger.debug("Headers being sent: %s", headers)
        try:
            logger.debug("Payload: %s", json.dumps(payload, default=str))
        except Exception:
            logger.debug("Payload (could not json dumps)")

        try:
            # Validate JSON structure
            json.dumps(payload)
            logger.debug("Payload is valid JSON.")
        except (TypeError, ValueError) as e:
            logger.error("Invalid JSON payload: %s", e)
            return {"ok": False, "reason": "Invalid JSON payload"}

        max_retries = 3
        delay = 1
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    ENQUIRY_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                status = response.status_code
                try:
                    body = response.json()
                except Exception:
                    body = response.text

                logger.info("HTTP %s on attempt %d", status, attempt)
                logger.debug("Response body: %s", str(body)[:1000])
                logger.debug("Response status: %s, body: %s", response.status_code, response.text)

                if status in (200, 201):
                    return {"ok": True, "status_code": status, "body": body}
                else:
                    result = {"ok": False, "status_code": status, "body": body}
                    logger.warning("Enquiry rejected: %s", result)
                    # don't retry for client errors (400s) except maybe 429
                    if 400 <= status < 500 and status not in (429,):
                        return result
                    # else continue to retry for 5xx or 429
            except requests.exceptions.RequestException as e:
                last_exc = e
                logger.error("Request exception on attempt %d: %s", attempt, e)
                logger.debug(traceback.format_exc())

            # backoff
            if attempt < max_retries:
                logger.info("Retrying in %s seconds...", delay)
                time.sleep(delay)
                delay *= 2

        # failed after retries -> write to outbox for manual retry
        try:
            entry = {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "payload": payload,
                "last_result": (str(last_exc) if last_exc else "failed after retries")
            }
            with open(OUTBOX_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
            logger.info("Saved failed enquiry to outbox: %s", OUTBOX_PATH)
        except Exception as e:
            logger.exception("Failed writing outbox entry: %s", e)

        return {"ok": False, "reason": str(last_exc) if last_exc else "failed after retries"}

    # ===========================
    # Flows (clusters/questions/answer)
    # ===========================
    def greeting_and_clusters(self):
        clusters = self.get_clusters()
        if not clusters:
            msg = self._gen_message("no_clusters", {"csv": CSV_FILE}, fallback="No clusters found. Please check your CSV.")
            return {"status": "error", "mode": "clusters", "text": msg, "clusters": [], "tts": self.make_tts(msg)}
        base_text = self._gen_message("clusters_prompt", {"count": len(clusters)}, fallback="These are the clusters. Please say one:")
        instruction = " Say a cluster name, or say 'HELP' to provide a short student info."
        text = base_text + instruction
        self._mode = "clusters"
        self._active_cluster = None
        return {"status": "ok", "mode": "clusters", "text": text, "clusters": clusters, "tts": self.make_tts(text)}

    def process_cluster_choice(self, user_input: str):
        clusters = self.get_clusters()
        match = self.fuzzy_match(user_input, clusters)
        if match:
            self._active_cluster = match
            self._mode = "questions"
            questions = self.get_questions_for_cluster(match)
            text = self._gen_message("cluster_selected", {"cluster": match}, fallback=f"You selected {match}. These are the questions in this cluster. Please say one.")
            return {"status": "ok", "mode": "questions", "cluster": match, "questions": questions, "text": text, "tts": self.make_tts(text)}
        else:
            text = self._gen_message("cluster_not_recognized", {"input": user_input}, fallback="Cluster not recognized. Please say the cluster name.")
            return {"status": "error", "mode": "clusters", "message": text, "text": text, "clusters": clusters, "tts": self.make_tts(text)}

    def process_question_choice(self, user_text: str):
        if not self._active_cluster:
            return self.greeting_and_clusters()

        local_answer, candidates = self.find_answer_in_cluster(user_text, self._active_cluster, top_k=5)
        if local_answer:
            short = self.summarize_for_speech(local_answer)
            resource = ""
            if self._last_answer_index is not None:
                resource = self.qa_rows[self._last_answer_index][3]
            return {"status": "ok", "mode": "answer", "text": short, "full_answer": local_answer, "tts": self.make_tts(short), "resource": resource}

        if candidates:
            self._last_cluster_candidates = candidates
            opts = [c[0] for c in candidates]
            clar_intro = self._gen_message("clarify_in_cluster_intro", {"count": len(opts)}, fallback="I found multiple similar questions. Please say the full question you meant.")
            clar_text = clar_intro + "\n" + "\n".join([f"{i+1}. {q}" for i, q in enumerate(opts)])
            return {"status": "ok", "mode": "clarify_in_cluster", "candidates": opts, "text": clar_text, "tts": self.make_tts(clar_intro)}

        local_answer, candidates = self.find_answer_local(user_text, top_k=5)
        if local_answer:
            short = self.summarize_for_speech(local_answer)
            resource = ""
            if self._last_answer_index is not None:
                resource = self.qa_rows[self._last_answer_index][3]
            self._mode = "clusters"
            self._active_cluster = None
            return {"status": "ok", "mode": "answer", "text": short, "full_answer": local_answer, "tts": self.make_tts(short), "resource": resource}

        if candidates:
            self._last_candidates = candidates
            opts = [c[0] for c in candidates]
            clar_intro = self._gen_message("clarify_intro", {"count": len(opts)}, fallback="I found similar questions. Please say the full question.")
            clar_text = clar_intro + "\n" + "\n".join([f"{i+1}. {q}" for i, q in enumerate(opts)])
            return {"status": "ok", "mode": "clarify", "candidates": opts, "text": clar_text, "tts": self.make_tts(clar_intro)}

        return self.query_gemini(user_text)

    def process_choice(self, user_text: str):
        if user_text and isinstance(user_text, str) and 'help' in user_text.lower():
            return self.start_inquiry()

        if self._mode == "clusters" or not self._active_cluster:
            return self.process_cluster_choice(user_text)

        return self.process_question_choice(user_text)

    def process_turn_from_file(self, file_path: str):
        # If currently in inquiry mode, handle the audio as an inquiry answer
        if self._mode == 'inquiry':
            user_text = self.transcribe_audio(file_path)
            if not user_text:
                msg = self._gen_message("stt_error", {}, fallback="Sorry, I couldn't hear that.")
                return {"status": "error", "message": msg, "text": msg, "tts": self.make_tts(msg)}
            res = self.process_inquiry_answer(user_text)
            res["transcript"] = user_text
            return res

        user_text = self.transcribe_audio(file_path)
        if not user_text:
            msg = self._gen_message("stt_error", {}, fallback="Sorry, I couldn't hear that.")
            return {"status": "error", "message": msg, "text": msg, "tts": self.make_tts(msg)}

        try:
            cleaned = re.sub(r"[^\w\s]", " ", user_text).lower()
            tokens = [t.strip() for t in cleaned.split() if t.strip()]
            if 'help' in tokens:
                res = self.start_inquiry()
                res["transcript"] = user_text
                return res
        except Exception:
            pass

        if self._last_candidates or self._last_cluster_candidates:
            res = self.process_choice(user_text)
            res["transcript"] = user_text
            return res

        res = self.process_turn_direct(user_text)
        res["transcript"] = user_text
        return res

    def process_turn_direct(self, user_text: str):
        if user_text and isinstance(user_text, str) and 'help' in user_text.lower():
            return self.start_inquiry()
        clusters = self.get_clusters()
        ranked = [(c, self.score_match(user_text, c)) for c in clusters]
        ranked.sort(key=lambda x: x[1], reverse=True)
        if ranked and ranked[0][1] >= 70:
            picked = ranked[0][0]
            self._active_cluster = picked
            self._mode = "questions"
            questions = self.get_questions_for_cluster(picked)
            text = self._gen_message("cluster_selected", {"cluster": picked}, fallback=f"You selected {picked}. These are the questions in this cluster. Please say one.")
            return {"status": "ok", "mode": "questions", "cluster": picked,
                    "questions": questions, "text": text,
                    "tts": self.make_tts(text)}

        if self._active_cluster:
            return self.process_question_choice(user_text)

        local_answer, candidates = self.find_answer_local(user_text)
        if local_answer:
            short = self.summarize_for_speech(local_answer)
            self._active_cluster = None
            return {"status": "ok", "mode": "answer", "text": short,
                    "full_answer": local_answer, "tts": self.make_tts(short)}
        if candidates:
            self._last_candidates = candidates
            opts = [c[0] for c in candidates]
            clar_intro = self._gen_message("clarify_intro", {"count": len(opts)}, fallback="I found similar questions. Please say the full question.")
            clar_text = clar_intro + "\n" + "\n".join([f"{i+1}. {q}" for i, q in enumerate(opts)])
            return {"status": "ok", "mode": "clarify",
                    "candidates": opts, "text": clar_text,
                    "tts": self.make_tts(clar_intro)}

        ai = self.query_gemini(f"{user_text}\n\nImportant: Only answer if this is about school domain or MyClassboard clusters")
        return ai

    # ===========================
    # Inquiry Flow
    # ===========================
    def start_inquiry(self):
        """Begin inquiry mode (student info collection)."""
        self._mode = "inquiry"
        self._inquiry_state = {"step": 0, "answers": {}}
        intro = self._gen_message("inquiry_intro", {}, fallback="I need a few details. You can say quit anytime to stop.")
        first_q = self._inquiry_questions[0]
        msg = intro + " " + first_q
        return {
            "status": "ok",
            "mode": "inquiry",
            "text": msg,
            "tts": self.make_tts(msg),
            "question": first_q
        }

    def _is_valid_email(self, text: str) -> bool:
        """Validate email format."""
        pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
        return bool(re.match(pattern, text.strip().lower()))

    def _is_valid_gender(self, text: str) -> bool:
        """Validate gender (male/female/other)."""
        return text.strip().lower() in ["male", "female", "other"]

    def _is_valid_phone(self, text: str) -> bool:
        """Validate phone number (10 digits only)."""
        return bool(re.fullmatch(r"\d{10}", text.strip()))

    def _is_valid_dob(self, text: str) -> bool:
        """Validate date of birth format (YYYY-MM-DD) or allow blank."""
        if not text.strip():
            return True  # optional field
        pattern = r"^\d{4}-\d{2}-\d{2}$"
        return bool(re.match(pattern, text.strip()))

    # Normalize user-provided DOB to YYYY-MM-DD (returns string or None)
    def normalize_dob_input(self, raw_input: str) -> Optional[str]:
        if not raw_input or not str(raw_input).strip():
            return ""
        s = str(raw_input).strip()

        # Try several common formats using datetime.strptime
        from datetime import datetime
        patterns = ["%Y-%m-%d", "%Y %m %d", "%d %m %Y", "%d-%m-%Y", "%d/%m/%Y", "%Y%m%d", "%d%m%Y"]
        for fmt in patterns:
            try:
                dt = datetime.strptime(s, fmt)
                # sanity: year reasonable
                if 1900 <= dt.year <= datetime.now().year:
                    return dt.strftime("%Y-%m-%d")
            except Exception:
                pass

        # Try digits-only heuristics
        s_digits = re.sub(r"[^\d]", "", s)
        if len(s_digits) == 8:
            # prefer YYYYMMDD then DDMMYYYY
            try:
                dt = datetime.strptime(s_digits, "%Y%m%d")
                if 1900 <= dt.year <= datetime.now().year:
                    return dt.strftime("%Y-%m-%d")
            except Exception:
                try:
                    dt = datetime.strptime(s_digits, "%d%m%Y")
                    if 1900 <= dt.year <= datetime.now().year:
                        return dt.strftime("%Y-%m-%d")
                except Exception:
                    pass

        return None  # not parseable

    def process_inquiry_answer(self, text):
        if not self._inquiry_state:
            return {"status": "error", "text": "No active inquiry.", "tts": self.make_tts("No active inquiry.")}

        raw = (text or "").strip()
        normalized = raw.lower().strip()
        step = self._inquiry_state["step"]
        answers = self._inquiry_state["answers"]
        question = self._inquiry_questions[step]

        # 1) Quit
        if normalized in ["quit", "stop", "exit", "cancel", ".", "fullstop", "dot"]:
            self._mode = "clusters"
            self._inquiry_state = None
            msg = "Okay, I’ve stopped collecting details. Say a cluster name to continue."
            return {"status": "ok", "mode": "clusters", "text": msg, "tts": self.make_tts(msg)}

        # 2) Skip
        if normalized in ["skip", "skipp", "skeep", "skipit", "skipped"]:
            answers[question] = ""
            step += 1
            if step >= len(self._inquiry_questions):
                # Instead of sending immediately, we now ask frontend to show a review dialog
                self._inquiry_state["answers"] = answers
                payload = self._build_enquiry_payload(answers)
                review_msg = "Please review the captured details and edit if needed."
                return {
                    "status": "ok",
                    "mode": "review",
                    "text": review_msg,
                    "tts": self.make_tts(review_msg),
                    "answers": answers,
                    "payload": payload,
                    "questions": self._inquiry_questions,        # <-- helpful for frontend ordering
                    "question_order": self._inquiry_questions  # duplicate key but explicit
                }
            self._inquiry_state["step"] = step
            msg = self._inquiry_questions[step]
            return {"status": "ok", "mode": "inquiry", "text": msg, "tts": self.make_tts(msg), "question": msg}

        # 3) If user asks to retry a failed API send (after a previous finalize failure)
        if normalized in ["retry", "try again"] and self._inquiry_state.get("final_payload"):
            # attempt to resend stored payload
            payload = self._inquiry_state["final_payload"]
            api_status = self._send_enquiry(payload)
            success = api_status.get("ok") if isinstance(api_status, dict) else bool(api_status)
            if success:
                # clear inquiry state on success
                self._inquiry_state = None
                self._mode = "clusters"
                msg = "Thanks — your details were resent and submitted successfully."
                return {"status": "ok", "mode": "clusters", "text": msg, "tts": self.make_tts(msg)}
            else:
                logger.warning("Retry failed: %s", api_status)
                msg = "Retry failed. I saved your details and will try again later. Say 'retry' to try again or 'skip' to continue."
                return {"status": "error", "mode": "inquiry", "text": msg, "tts": self.make_tts(msg)}

        # 4) Basic noisy input guard
        if len(normalized) <= 1 or re.fullmatch(r"(.)\1{2,}", normalized):
            msg = "That doesn’t seem valid. Please provide a proper answer."
            return {"status": "error", "mode": "inquiry", "text": msg, "tts": self.make_tts(msg), "question": question}

        raw_input = raw

        # 5) Gender question: accept numeric or fuzzy textual input
        if question.startswith("What is your child's gender"):
            # Try numeric choice first
            m = re.search(r"\b([1-3])\b", normalized)
            if m:
                choice = m.group(1)
                if choice == "1":
                    answers[question] = "male"
                elif choice == "2":
                    answers[question] = "female"
                else:
                    answers[question] = "other"
            else:
                # Try robust textual normalization (handles 'mail' mis-hearings)
                g = self._normalize_gender_text(raw)
                if not g:
                    msg = "Please respond with 1 for male, 2 for female, or 3 for other."
                    return {"status": "error", "mode": "inquiry", "text": msg, "tts": self.make_tts(msg), "question": question}
                answers[question] = g

            step += 1
            if step >= len(self._inquiry_questions):
                # reached end -> provide review to frontend (do NOT auto-send)
                self._inquiry_state["answers"] = answers
                payload = self._build_enquiry_payload(answers)
                review_msg = "Please review the captured details and edit if needed."
                return {
                    "status": "ok",
                    "mode": "review",
                    "text": review_msg,
                    "tts": self.make_tts(review_msg),
                    "answers": answers,
                    "payload": payload,
                    "questions": self._inquiry_questions,
                    "question_order": self._inquiry_questions
                }

            self._inquiry_state["step"] = step
            next_q = self._inquiry_questions[step]
            return {"status": "ok", "mode": "inquiry", "text": next_q, "tts": self.make_tts(next_q), "question": next_q}

        # 6) Email validation
        if "email" in question.lower():
            if not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", normalized):
                msg = "That email doesn’t look right. Please say it again."
                return {"status": "error", "mode": "inquiry", "text": msg, "tts": self.make_tts(msg), "question": question}

        # 7) Mobile normalization (optional)
        if "mobile" in question.lower():
            digits = re.sub(r"\D", "", normalized)
            if digits and not re.fullmatch(r"\d{10}", digits):
                msg = "Please say a valid 10-digit mobile number, or say 'skip' to skip."
                return {"status": "error", "mode": "inquiry", "text": msg, "tts": self.make_tts(msg), "question": question}
            raw_input = digits

        # 8) DOB normalization + validation
        if "date of birth" in question.lower():
            # raw may contain trailing noise; use normalize_dob_input
            dob_norm = self.normalize_dob_input(raw_input)
            if dob_norm is None:
                msg = ("Please provide the date in YYYY-MM-DD format (for example: 2004-07-21). "
                    "You may also say 2004 07 21 or 20040721.")
                return {"status": "error", "mode": "inquiry", "text": msg, "tts": self.make_tts(msg), "question": question}
            raw_input = dob_norm

        # 9) Save answer & advance
        answers[question] = raw_input
        step += 1

        if step >= len(self._inquiry_questions):
            # end of questions: instead of auto-submitting, return a review object for frontend dialog box
            self._inquiry_state["answers"] = answers
            payload = self._build_enquiry_payload(answers)
            review_msg = "Please review the captured details and edit them if needed before submission."
            return {
                "status": "ok",
                "mode": "review",
                "text": review_msg,
                "tts": self.make_tts(review_msg),
                "answers": answers,
                "payload": payload,
                "questions": self._inquiry_questions,
                "question_order": self._inquiry_questions
            }

        self._inquiry_state["step"] = step
        msg = self._inquiry_questions[step]
        return {"status": "ok", "mode": "inquiry", "text": msg, "tts": self.make_tts(msg), "question": msg}


    def _finalize_inquiry(self, answers):
        logger.info(f"Collected Answers: {json.dumps(answers, indent=2)}")

        # Build the payload with proper validation
        payload = self._build_enquiry_payload(answers)
        logger.info(f"Final Payload Sent to API: {json.dumps(payload, indent=2)}")
        logger.debug("Payload being sent: %s", json.dumps(payload, indent=2))

        # Send the payload to the API
        api_status = self._send_enquiry(payload)

        success = api_status.get("ok") if isinstance(api_status, dict) else bool(api_status)
        if success:
            # Clear state and acknowledge success
            self._inquiry_state = None
            self._mode = "clusters"
            msg = "Thank you! Your details have been submitted successfully."
            return {"status": "ok", "mode": "clusters", "text": msg, "tts": self.make_tts(msg), "api_result": api_status}
        else:
            # Persist last payload to allow retries
            logger.warning("Enquiry failed: %s", api_status)
            self._inquiry_state = {"step": len(self._inquiry_questions), "answers": answers}
            self._inquiry_state["final_payload"] = payload
            self._inquiry_state["last_send_result"] = api_status

            msg = ("Something went wrong while submitting. I saved your details and will try later. "
                   "Say 'retry' to try sending again or say a cluster name to continue.")
            return {"status": "error", "mode": "inquiry", "text": msg, "tts": self.make_tts(msg), "api_result": api_status}

    def finalize_review(self, updated_answers: dict):
        """
        Validate and normalize fields, then send to API (calls _finalize_inquiry under the hood).
        Returns the same structure as _finalize_inquiry (success/failure).
        """
        if not isinstance(updated_answers, dict) or not updated_answers:
            return {"status": "error", "text": "No answers provided for submission."}

        # Normalize gender if present
        gender_q = "What is your child's gender? (male/female/other)"
        if gender_q in updated_answers:
            g = self._normalize_gender_text(updated_answers[gender_q])
            if g:
                updated_answers[gender_q] = g
            else:
                updated_answers[gender_q] = updated_answers[gender_q]

        # Normalize DOB if present
        dob_q = "Date of birth (YYYY-MM-DD) (optional):"
        if dob_q in updated_answers:
            dob_norm = self.normalize_dob_input(updated_answers[dob_q])
            if dob_norm is None:
                return {"status": "error", "text": "Date of birth not in a recognized format. Please use YYYY-MM-DD."}
            updated_answers[dob_q] = dob_norm

        # Normalize mobile number
        mobile_q = "Mobile number (optional):"
        if mobile_q in updated_answers:
            if updated_answers[mobile_q]:
                digits = re.sub(r"\D", "", str(updated_answers[mobile_q]))
                if not re.fullmatch(r"\d{10}", digits):
                    return {"status": "error", "text": "Mobile number must be 10 digits."}
                updated_answers[mobile_q] = digits

        # Call same finalize routine to send
        return self._finalize_inquiry(updated_answers)


    # A small safe wrapper to let external callers request a direct send of edited answers.
    def finalize_and_send(self, updated_answers: dict):
        """
        Convenience wrapper that validates and sends updated_answers.
        Returns the same dict structure as finalize_review/_finalize_inquiry.
        """
        if not isinstance(updated_answers, dict) or not updated_answers:
            return {"status": "error", "text": "No answers provided."}
        # Reuse finalize_review validation
        return self.finalize_review(updated_answers)


    def _get_first_question(self, cluster_name: str) -> str:
        """Return the first question for a given cluster (safe, uses self.df if available)."""
        if self.df is None:
            return "No questions found in this cluster."
        try:
            cluster_rows = self.df[self.df["Cluster"].str.lower() == str(cluster_name).lower()]
            if not cluster_rows.empty:
                return cluster_rows.iloc[0]["Question"]
        except Exception:
            logger.debug("Error fetching first question for cluster %s", cluster_name, exc_info=True)
        return "No questions found in this cluster."
