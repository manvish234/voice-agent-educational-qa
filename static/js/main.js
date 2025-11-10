// =======================
// main.js - automatic, buttonless voice agent
// (updated: added wake-word "HEY DSE" and HOLD/PROCEED pause-resume)
// =======================

// =======================
// Global Variables
// =======================
let mediaRecorder,
  chunks = [],
  isRecording = false,
  sessionActive = false;
let audioCtx, analyser, dataArray, source, micStream;
let silenceMonitorInterval = null;
let lastSoundTimestamp = 0;
const SILENCE_TIMEOUT = 3000; // 3 seconds
const VAD_THRESHOLD = 0.01; // RMS threshold for voice activity (tune if needed)

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const quitBtn = document.getElementById("quitBtn");
const statusDiv = document.getElementById("status");
const chatContainer = document.getElementById("chat-container");
const player = document.getElementById("player");
const sessionState = document.getElementById("sessionState");
const orb = document.querySelector(".orb");

// Bootstrap modal instance (will be created when needed)
let reviewModalInstance = null;

// globals used to forward server payloads during review
window.api_payload_from_server = null;
window.review_original_answers = null;

// wake-word recognition (HEY DSE)
let wakeRecognition = null;
let wakeListening = false;

// pause/resume recognition (listens for PROCEED while paused)
let pauseRecognition = null;
let pauseListening = false;

// paused state + last assistant json to replay after resume
let sessionPaused = false;
let lastAssistantJson = null;

// hide visible controls (buttonless UI)
if (startBtn) startBtn.style.display = "none";
if (stopBtn) stopBtn.style.display = "none";
if (quitBtn) quitBtn.style.display = "none";

// =======================
// Utilities
// =======================
function setStatus(s) {
  let displayText = s;
  let extraClass = "";

  if (/listen/i.test(s)) {
    displayText = "Speak now...";
    extraClass = "speak-now";
  } else if (/upload/i.test(s)) {
    displayText = "Fetching...";
    extraClass = "fetching";
  } else if (/ready/i.test(s)) {
    displayText = "Start to roll...";
    extraClass = "start-to-roll";
  }

  if (statusDiv) statusDiv.textContent = displayText;
  if (sessionState) sessionState.textContent = displayText;

  if (statusDiv) {
    statusDiv.className = "status";
    ["ready", "listen", "upload", "processing", "error", "stop"].forEach(
      (c) => {
        if (new RegExp(c, "i").test(s)) {
          statusDiv.classList.add(
            c === "stop"
              ? "stopped"
              : c === "listen"
              ? "listening"
              : c === "upload"
              ? "uploading"
              : c
          );
        }
      }
    );
    if (extraClass) statusDiv.classList.add(extraClass);
  }
}

function appendMessage(sender, text, opts = {}) {
  const messageEl = document.createElement("div");
  messageEl.classList.add("message", sender);

  if (sender === "assistant") {
    const words = (text || "").split(" ").map((word) => {
      const span = document.createElement("span");
      span.textContent = word + " ";
      span.classList.add("word");
      return span;
    });
    words.forEach((w) => messageEl.appendChild(w));
    if (!opts.noAnimate) messageEl.classList.add("speaking");

    if (!opts.noAnimate) {
      let i = 0;
      (function highlightNext() {
        if (i > 0) words[i - 1].classList.remove("active");
        if (i < words.length) {
          words[i].classList.add("active");
          i++;
          setTimeout(highlightNext, 300);
        } else messageEl.classList.remove("speaking");
      })();
    }
  } else {
    messageEl.innerText = text || "";
  }

  if (opts.extraHTML) {
    const wrap = document.createElement("div");
    wrap.innerHTML = opts.extraHTML;
    messageEl.appendChild(wrap);
  }

  chatContainer.appendChild(messageEl);
  chatContainer.scrollTop = chatContainer.scrollHeight;
  return messageEl;
}

// =======================
// Audio visualizer + mic setup
// - keeps micStream available to MediaRecorder
// =======================
function initAudioVisual() {
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 2048;
  dataArray = new Uint8Array(analyser.frequencyBinCount);

  return navigator.mediaDevices
    .getUserMedia({ audio: true })
    .then((stream) => {
      micStream = stream;
      source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);
      animateOrb();
      return true;
    })
    .catch((err) => {
      console.error("Mic access denied:", err);
      setStatus("Mic access denied");
      return false;
    });
}

function animateOrb() {
  requestAnimationFrame(animateOrb);
  if (!analyser) return;

  analyser.getByteFrequencyData(dataArray);
  let peak = 0;
  let sum = 0;
  for (let i = 0; i < dataArray.length; i++) {
    if (dataArray[i] > peak) peak = dataArray[i];
    sum += dataArray[i];
  }
  const avg = sum / dataArray.length;
  const scale = 0.7 + (peak / 255) * 1.2;
  orb.style.transform = `scale(${scale}) rotate(${Date.now() / 2000}rad)`;
  orb.style.filter = `blur(${10 + avg / 30}px)`;
}

// =======================
// VAD helpers (RMS from time domain)
// =======================
function getMicRms() {
  if (!analyser) return 0;
  const td = new Uint8Array(analyser.fftSize);
  analyser.getByteTimeDomainData(td);
  let sumSq = 0;
  for (let i = 0; i < td.length; i++) {
    const v = td[i] / 128 - 1; // -1..1
    sumSq += v * v;
  }
  const rms = Math.sqrt(sumSq / td.length);
  return rms;
}

// =======================
// Recording + silence monitor
// =======================
async function startRecording() {
  if (isRecording) return;
  if (sessionPaused) return; // do not start when paused
  // ensure micStream exists
  if (!micStream) {
    const ok = await initAudioVisual();
    if (!ok) return;
  }

  try {
    mediaRecorder = new MediaRecorder(micStream);
  } catch (e) {
    console.error("Failed to create MediaRecorder:", e);
    setStatus("Recording error");
    return;
  }

  chunks = [];
  mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
  mediaRecorder.onstop = async () => {
    // stop silence monitor
    stopSilenceMonitor();

    const blob = new Blob(chunks, { type: chunks[0]?.type || "audio/webm" });
    const form = new FormData();
    form.append("audio", blob, "recording.webm");
    setStatus("Uploading...");
    try {
      const resp = await fetch("/api/ask", { 
        method: "POST", 
        body: form,
        headers: {
          'X-Requested-With': 'XMLHttpRequest'
        }
      });
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
      }
      const data = await resp.json();
      handleResponse(data);
    } catch (err) {
      console.error('Upload error:', err);
      setStatus("Upload error: " + err.message);
    }
  };

  try {
    mediaRecorder.start();
    isRecording = true;
    setStatus("Listening...");
    startSilenceMonitor();
  } catch (err) {
    console.error("mediaRecorder.start error:", err);
    setStatus("Recording error");
  }
}

function stopRecording() {
  if (mediaRecorder && isRecording) {
    try {
      mediaRecorder.stop();
    } catch (e) {
      console.warn("stopRecording error:", e);
    }
    isRecording = false;
    setStatus("Processing...");
  }
}

function startSilenceMonitor() {
  // reset lastSoundTimestamp to now so short silence doesn't immediately stop
  lastSoundTimestamp = Date.now();
  if (silenceMonitorInterval) clearInterval(silenceMonitorInterval);

  silenceMonitorInterval = setInterval(() => {
    const rms = getMicRms();
    if (rms > VAD_THRESHOLD) {
      lastSoundTimestamp = Date.now();
    } else {
      // check silence duration
      if (Date.now() - lastSoundTimestamp >= SILENCE_TIMEOUT) {
        // prolonged silence -> auto stop
        stopRecording();
      }
    }
  }, 150);
}

function stopSilenceMonitor() {
  if (silenceMonitorInterval) {
    clearInterval(silenceMonitorInterval);
    silenceMonitorInterval = null;
  }
}

// =======================
// TTS playback (with optional end callback)
// =======================
function playTTS(filename, msgEl, onEnded) {
  if (!filename) {
    if (onEnded) onEnded();
    return;
  }

  player.pause();
  player.currentTime = 0;
  window.speechSynthesis.cancel();

  player.src = "/api/tts/" + filename;
  player.play().catch((err) => console.warn('TTS playback failed:', err));

  if (msgEl) msgEl.classList.add("speaking");

  const ttsInterval = setInterval(() => {
    orb.style.transform = `scale(${1 + Math.random() * 0.25}) rotate(${
      Date.now() / 2000
    }rad)`;
  }, 80);

  player.onended = () => {
    clearInterval(ttsInterval);
    if (msgEl) msgEl.classList.remove("speaking");
    if (typeof onEnded === "function") onEnded();
  };
}

// =======================
// Helpers for review mapping
// =======================
function normalizeKey(s) {
  if (s === null || s === undefined) return "";
  return String(s)
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "");
}

function stringifyValue(v) {
  if (v === null || v === undefined) return "";
  if (typeof v === "object") {
    try {
      return JSON.stringify(v);
    } catch {
      return String(v);
    }
  }
  return String(v);
}

// =======================
// Review modal logic (updated mapping + submit)
// (kept from your previous working implementation)
// =======================
function showReviewModal(answersObj = {}, payloadObj = {}) {
  window.review_original_answers = answersObj || {};

  const fieldsContainer = document.getElementById("reviewFields");
  fieldsContainer.innerHTML = "";

  const questionOrder =
    payloadObj?.question_order ||
    payloadObj?.questions ||
    Object.keys(answersObj);

  const apiPayload =
    window.api_payload_from_server || payloadObj?.api_payload || null;
  const apiIndex = {};
  if (apiPayload && typeof apiPayload === "object") {
    Object.keys(apiPayload).forEach((k) => {
      const normalizedKey = normalizeKey(k);
      const valStr = stringifyValue(apiPayload[k]);
      apiIndex[normalizedKey] = apiIndex[normalizedKey] || [];
      apiIndex[normalizedKey].push(k);

      const normalizedVal = normalizeKey(valStr);
      apiIndex[normalizedVal] = apiIndex[normalizedVal] || [];
      apiIndex[normalizedVal].push(k);
    });
  }

  (questionOrder || []).forEach((question) => {
    let value = "";
    if (Object.prototype.hasOwnProperty.call(answersObj, question)) {
      value = answersObj[question];
    } else {
      value = "";
    }

    const displayValue =
      value === null || value === undefined
        ? ""
        : Array.isArray(value)
        ? value.join(", ")
        : typeof value === "object"
        ? JSON.stringify(value)
        : String(value);

    const row = document.createElement("div");
    row.className = "row mb-2 align-items-start";

    const colLabel = document.createElement("div");
    colLabel.className = "col-12 col-md-4";
    const label = document.createElement("label");
    label.className = "form-label fw-bold";
    label.innerText = question;
    colLabel.appendChild(label);

    const colInput = document.createElement("div");
    colInput.className = "col-12 col-md-8";

    const qKeyLower = String(question).toLowerCase();
    let inputEl;

    if (qKeyLower.includes("gender")) {
      inputEl = document.createElement("select");
      inputEl.className = "form-select";
      const options = [
        "",
        "Male",
        "Female",
        "Other",
        "Prefer not to say",
        "Non-binary",
      ];
      options.forEach((opt) => {
        const o = document.createElement("option");
        o.value = opt;
        o.text = opt === "" ? "— Select —" : opt;
        if (
          opt &&
          displayValue &&
          displayValue.toLowerCase() === opt.toLowerCase()
        ) {
          o.selected = true;
        }
        inputEl.appendChild(o);
      });
      inputEl.dataset.origType = Array.isArray(value) ? "array" : typeof value;
    } else if (
      Array.isArray(value) ||
      typeof value === "object" ||
      displayValue.length > 120
    ) {
      inputEl = document.createElement("textarea");
      inputEl.className = "form-control";
      inputEl.rows = 3;
      inputEl.value = displayValue;
      inputEl.dataset.origType = Array.isArray(value) ? "array" : typeof value;
    } else {
      inputEl = document.createElement("input");
      inputEl.type = "text";
      inputEl.className = "form-control";
      inputEl.value = displayValue;
      inputEl.dataset.origType = Array.isArray(value) ? "array" : typeof value;
    }

    inputEl.dataset.qkey = question;

    if (apiPayload && typeof apiPayload === "object") {
      let matchedKey = null;
      const normalizedQ = normalizeKey(question);
      const normalizedVal = normalizeKey(displayValue);

      if (apiIndex[normalizedQ] && apiIndex[normalizedQ].length) {
        matchedKey = apiIndex[normalizedQ][0];
      }
      if (
        !matchedKey &&
        apiIndex[normalizedVal] &&
        apiIndex[normalizedVal].length
      ) {
        matchedKey = apiIndex[normalizedVal][0];
      }
      if (!matchedKey) {
        for (const k of Object.keys(apiPayload)) {
          const vStr = stringifyValue(apiPayload[k]);
          if (vStr === displayValue) {
            matchedKey = k;
            break;
          }
        }
      }
      if (matchedKey) {
        inputEl.dataset.apikey = matchedKey;
      }
    }

    const helper = document.createElement("div");
    helper.className = "form-text review-helper";
    helper.style.display = "none";
    helper.style.color = "#b00";

    colInput.appendChild(inputEl);
    colInput.appendChild(helper);
    row.appendChild(colLabel);
    row.appendChild(colInput);
    fieldsContainer.appendChild(row);
  });

  const modalEl = document.getElementById("reviewModal");
  reviewModalInstance = new bootstrap.Modal(modalEl, { backdrop: "static" });
  reviewModalInstance.show();
}

function wireReviewSubmitButton() {
  const submitBtn = document.getElementById("reviewSubmit");
  if (!submitBtn) return;
  if (submitBtn.dataset.wired === "1") return;
  submitBtn.dataset.wired = "1";

  submitBtn.onclick = async () => {
    const fieldsContainer = document.getElementById("reviewFields");
    const inputs = Array.from(fieldsContainer.querySelectorAll("[data-qkey]"));
    const updatedAnswers = {};

    let updatedApiPayload = null;
    if (
      window.api_payload_from_server &&
      typeof window.api_payload_from_server === "object"
    ) {
      try {
        updatedApiPayload = JSON.parse(
          JSON.stringify(window.api_payload_from_server)
        );
      } catch {
        updatedApiPayload = Object.assign({}, window.api_payload_from_server);
      }
    }

    inputs.forEach((el) => {
      const question = el.dataset.qkey;
      let value = "";

      if (el.tagName === "SELECT") {
        value = el.value || "";
      } else if (el.tagName === "TEXTAREA" || el.tagName === "INPUT") {
        value = (el.value || "").trim();
      } else {
        value = (el.value || "").trim();
      }

      const origType =
        el.dataset.origType ||
        (window.review_original_answers &&
        window.review_original_answers[question] !== undefined
          ? Array.isArray(window.review_original_answers[question])
            ? "array"
            : typeof window.review_original_answers[question]
          : "string");

      let finalValue;
      if (origType === "array") {
        if (!value) {
          finalValue = [];
        } else {
          finalValue = value
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean);
        }
      } else if (origType === "object") {
        try {
          finalValue = JSON.parse(value);
        } catch {
          finalValue = value;
        }
      } else {
        finalValue = value;
      }

      updatedAnswers[question] = finalValue;

      const apikey = el.dataset.apikey;
      if (
        apikey &&
        updatedApiPayload &&
        typeof updatedApiPayload === "object"
      ) {
        updatedApiPayload[apikey] = finalValue;
      }
    });

    const submitBtnEl = document.getElementById("reviewSubmit");
    submitBtnEl.disabled = true;
    const previousText = submitBtnEl.textContent;
    submitBtnEl.textContent = "Submitting...";

    const bodyToSend = { answers: updatedAnswers };
    if (updatedApiPayload) {
      bodyToSend.api_payload = updatedApiPayload;
    } else if (window.api_payload_from_server) {
      bodyToSend.api_payload = window.api_payload_from_server;
    }

    try {
      const response = await fetch("/api/finalize_review", {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          'X-Requested-With': 'XMLHttpRequest'
        },
        body: JSON.stringify(bodyToSend),
      });

      const respJson = await response.json().catch(() => ({}));

      if (!response.ok) {
        console.error(`Error submitting answers (${response.status}):`, respJson);
        alert(`Failed to submit answers (${response.status}): ${respJson.error || 'Unknown error'}`);
        submitBtnEl.disabled = false;
        submitBtnEl.textContent = previousText;
        return;
      }

      console.log("Submission successful:", respJson);

      window.api_payload_from_server = null;
      window.review_original_answers = null;

      const reviewModalInstanceLocal = bootstrap.Modal.getInstance(
        document.getElementById("reviewModal")
      );
      if (reviewModalInstanceLocal) reviewModalInstanceLocal.hide();

      setStatus("Ready");

      try {
        const resp = await fetch("/api/start");
        handleResponse(await resp.json());
      } catch (err) {
        console.error("api/start error after review:", err);
      }
    } catch (error) {
      console.error("Error submitting answers:", error);
      alert(`Network error: ${error.message}. Please check your connection and try again.`);
      submitBtnEl.disabled = false;
      submitBtnEl.textContent = previousText;
    }
  };
}
wireReviewSubmitButton();

// =======================
// Pause / Resume (HOLD ON / PROCEED)
// =======================
function startPauseListener() {
  if (pauseListening) return;
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition || null;
  if (!SpeechRecognition) {
    appendMessage(
      "assistant",
      "Pause-listener unavailable (browser unsupported). Say PROCEED manually.",
      { noAnimate: true }
    );
    return;
  }

  try {
    pauseRecognition = new SpeechRecognition();
    pauseRecognition.continuous = true;
    pauseRecognition.interimResults = true;
    pauseRecognition.lang = "en-US";
    pauseListening = true;

    pauseRecognition.onresult = (event) => {
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const res = event.results[i];
        const transcript = res[0].transcript.trim();
        console.debug("Pause-listen:", transcript, "final:", res.isFinal);
        if (/\b(proceed|resume|continue)\b/i.test(transcript)) {
          appendMessage("assistant", 'Detected "PROCEED" — resuming...', {
            noAnimate: true,
          });
          stopPauseListener();
          // small delay and resume
          setTimeout(() => resumeFlow(), 200);
          break;
        }
        if (/\b(quit|stop|exit)\b/i.test(transcript)) {
          appendMessage(
            "assistant",
            'Detected "QUIT" while paused — ending session.',
            { noAnimate: true }
          );
          stopPauseListener();
          doQuit();
          break;
        }
      }
    };

    pauseRecognition.onerror = (err) => {
      console.warn("Pause recognition error:", err);
      if (err && err.error === "not-allowed") {
        stopPauseListener();
        setStatus("Microphone permission denied for resume-listener");
      }
    };

    pauseRecognition.onend = () => {
      if (pauseListening) {
        try {
          pauseRecognition.start();
        } catch (e) {
          console.warn("Failed to restart pause-recognition:", e);
        }
      }
    };

    try {
      pauseRecognition.start();
      setStatus("Paused - say 'PROCEED' to continue");
    } catch (e) {
      console.warn("pauseRecognition.start error:", e);
    }
  } catch (err) {
    console.error("startPauseListener error:", err);
    pauseListening = false;
  }
}

function stopPauseListener() {
  if (!pauseListening) return;
  pauseListening = false;
  setStatus("Ready");
  try {
    if (pauseRecognition) {
      pauseRecognition.onresult = null;
      pauseRecognition.onend = null;
      pauseRecognition.onerror = null;
      try {
        pauseRecognition.stop();
      } catch (e) {
        console.warn('Error stopping pause recognition:', e);
      }
      pauseRecognition = null;
    }
  } catch (err) {
    console.warn("stopPauseListener error:", err);
  }
}

function pauseFlow() {
  if (sessionPaused) return;
  sessionPaused = true;
  stopRecording();
  stopSilenceMonitor();
  window.speechSynthesis.cancel();
  appendMessage(
    "assistant",
    "⏸️ Paused. Say 'PROCEED' to continue or 'QUIT' to exit.",
    { noAnimate: true }
  );
  setStatus("Paused");
  startPauseListener();
}

function resumeFlow() {
  if (!sessionPaused) return;
  stopPauseListener();
  sessionPaused = false;
  appendMessage("assistant", "▶️ Resuming...", { noAnimate: true });
  setStatus("Resuming...");

  // If we have a stored assistant JSON (pending prompt), replay it by feeding to handleResponse.
  if (lastAssistantJson) {
    try {
      // pass the stored assistant object through handleResponse so existing flow handles TTS & listening.
      handleResponse(lastAssistantJson.full || lastAssistantJson);
    } catch (e) {
      console.warn("Error replaying last assistant message:", e);
      // fallback: just start listening
      sessionActive = true;
      startRecording();
    } finally {
      lastAssistantJson = null;
    }
  } else {
    // Nothing pending — just resume listening
    sessionActive = true;
    startRecording();
  }
}

// =======================
// Wake-word listener (HEY DSE)
// =======================
function startWakeWordListener() {
  if (wakeListening) return;
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition || null;
  if (!SpeechRecognition) {
    console.warn(
      "Wake-word not supported: no SpeechRecognition in this browser."
    );
    appendMessage("assistant", "Wake-word not available in this browser.", {
      noAnimate: true,
    });
    setStatus("Wake-word unsupported");
    return;
  }

  try {
    wakeRecognition = new SpeechRecognition();
    wakeRecognition.continuous = true;
    wakeRecognition.interimResults = true;
    wakeRecognition.lang = "en-US";
    wakeListening = true;

    wakeRecognition.onresult = (event) => {
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const res = event.results[i];
        const transcript = res[0].transcript.trim();
        console.debug("Wake result:", transcript, "final:", res.isFinal);
        if (/\bhey\s+dse\b/i.test(transcript)) {
          console.info("Wake-word detected:", transcript);
          appendMessage(
            "assistant",
            'Detected "HEY DSE" — starting session...',
            { noAnimate: true }
          );
          stopWakeWordListener();
          setTimeout(() => {
            startSession();
          }, 200);
          break;
        }
      }
    };

    wakeRecognition.onerror = (err) => {
      console.warn("Wake recognition error:", err);
      if (err && err.error === "not-allowed") {
        stopWakeWordListener();
        setStatus("Microphone permission denied for wake-word");
      }
    };

    wakeRecognition.onend = () => {
      if (wakeListening) {
        try {
          wakeRecognition.start();
        } catch (e) {
          console.warn("Failed to restart wake-recognition:", e);
        }
      }
    };

    try {
      wakeRecognition.start();
      setStatus('Say "HEY DSE" to start');
      appendMessage("assistant", 'Say "HEY DSE" to start the session.', {
        noAnimate: true,
      });
    } catch (e) {
      console.warn("wakeRecognition.start error:", e);
    }
  } catch (err) {
    console.error("startWakeWordListener error:", err);
    wakeListening = false;
  }
}

function stopWakeWordListener() {
  if (!wakeListening) return;
  wakeListening = false;
  setStatus("Ready");
  try {
    if (wakeRecognition) {
      wakeRecognition.onresult = null;
      wakeRecognition.onend = null;
      wakeRecognition.onerror = null;
      try {
        wakeRecognition.stop();
      } catch {}
      wakeRecognition = null;
    }
  } catch (err) {
    console.warn("stopWakeWordListener error:", err);
  }
}

// =======================
// Graceful quit (triggered by saying "quit")
// =======================
function doQuit() {
  stopRecording();
  window.speechSynthesis.cancel();
  player.pause();
  sessionActive = false;
  appendMessage(
    "assistant",
    '👋 Goodbye. Session stopped. Say "HEY DSE" to start again.',
    {
      noAnimate: true,
    }
  );
  setStatus("Session ended.");
  try {
    const inst = bootstrap.Modal.getInstance(
      document.getElementById("reviewModal")
    );
    if (inst) inst.hide();
  } catch (e) {
    console.warn('Error hiding review modal:', e);
  }
  // clean pause/wake listeners
  stopPauseListener();
  // start wake-word listener to allow restarting
  startWakeWordListener();
}

// =======================
// Auto-start scheduling
// =======================
function scheduleStartListening(opts = {}) {
  const { ttsFile = null, mode = "" } = opts;
  if (!sessionActive) return;
  if (mode && mode.toLowerCase() === "review") {
    return;
  }
  if (isRecording) return;

  if (ttsFile) {
    return;
  } else {
    setTimeout(() => {
      if (!isRecording && sessionActive && !sessionPaused) startRecording();
    }, 400);
  }
}

// =======================
// Robust response handler (main central)
// - includes HOLD/PROCEED detection from transcript
// =======================
async function handleResponse(json) {
  if (!json) {
    setStatus("No response");
    return;
  }
  if (json.error) {
    appendMessage("assistant", "Error: " + json.error, { noAnimate: true });
    setStatus("Error");
    return;
  }

  // If transcript present, treat it as user utterance and check for special commands
  if (json.transcript) {
    appendMessage("user", json.transcript, { noAnimate: true });

    // QUIT has top priority
    if (/(\bquit\b|\bstop\b|\bexit\b)/i.test(json.transcript)) {
      doQuit();
      return;
    }

    // HOLD / PAUSE detection (user wants to pause immediately)
    if (/\b(hold on|hold|pause|holdno|hold-no)\b/i.test(json.transcript)) {
      // store any assistant part of this response to replay after resume
      const assistantPart = {
        text: json.text || null,
        tts: json.tts || null,
        resource: json.resource || json.link || null,
        mode: json.mode || null,
        full: json, // keep full in case handleResponse needs it later
      };
      lastAssistantJson = assistantPart;
      pauseFlow();
      return; // stop further processing; resume will replay stored assistant
    }

    // PROCEED / RESUME detection (user wants to continue)
    if (/\b(proceed|resume|continue)\b/i.test(json.transcript)) {
      // If paused, resume; otherwise ignore
      if (sessionPaused) {
        resumeFlow();
        return;
      }
      // If not paused, just continue normally (don't short-circuit)
    }
  }

  // nested helper to show assistant text and potentially play TTS
  function showAssistantText(text, ttsFile, resource, modeName) {
    // If session is paused, store this assistant response and do not display/play it now
    if (sessionPaused) {
      lastAssistantJson = {
        text,
        tts: ttsFile,
        resource,
        mode: modeName,
        full: { text, tts: ttsFile, resource, mode: modeName },
      };
      return;
    }

    const msgEl = appendMessage("assistant", text || "", { noAnimate: false });
    if (resource) {
      const html = `<div style="margin-top:8px;"><strong>Resource:</strong> <a href="${resource}" target="_blank" rel="noopener noreferrer">${resource}</a></div>`;
      msgEl.insertAdjacentHTML("beforeend", html);
    }

    if (ttsFile) {
      playTTS(ttsFile, msgEl, () => {
        scheduleStartListening({ mode: modeName });
        sessionActive = true;
      });
    } else {
      scheduleStartListening({ mode: modeName });
    }
  }

  const mode = (json.mode || "").toLowerCase();

  if (mode === "clusters") {
    showAssistantText(
      json.text || "Please choose a cluster.",
      json.tts,
      "",
      mode
    );
    const clusters = json.clusters || [];
    if (clusters.length) {
      const msgEl = appendMessage("assistant", "Clusters:", {
        noAnimate: true,
      });
      const listEl = document.createElement("div");
      listEl.style.marginTop = "6px";
      clusters.forEach((c, i) => {
        const row = document.createElement("div");
        row.textContent = `${i + 1}. ${c}`;
        row.classList.add("clickable-cluster");
        row.style.cursor = "pointer";
        row.addEventListener("click", async () => {
          const form = new FormData();
          form.append("choice_text", c);
          setStatus("Uploading...");
          const resp = await fetch("/api/choice", {
            method: "POST",
            body: form,
          });
          handleResponse(await resp.json());
        });
        listEl.appendChild(row);
      });
      msgEl.appendChild(listEl);
    }
    setStatus("Ready");
    return;
  }

  if (mode === "questions") {
    showAssistantText(
      json.text || "Please pick a question.",
      json.tts,
      "",
      mode
    );
    const questions = json.questions || [];
    if (questions.length) {
      const msgEl = appendMessage("assistant", "Questions:", {
        noAnimate: true,
      });
      const listEl = document.createElement("div");
      listEl.style.marginTop = "6px";
      questions.forEach((q, i) => {
        const row = document.createElement("div");
        row.textContent = `${i + 1}. ${q}`;
        row.classList.add("clickable-question");
        row.style.cursor = "pointer";
        row.addEventListener("click", async () => {
          const form = new FormData();
          form.append("choice_text", q);
          setStatus("Uploading...");
          const resp = await fetch("/api/choice", {
            method: "POST",
            body: form,
          });
          handleResponse(await resp.json());
        });
        listEl.appendChild(row);
      });
      msgEl.appendChild(listEl);
    }
    setStatus("Ready");
    return;
  }

  if (mode === "clarify" || mode === "clarify_in_cluster") {
    showAssistantText(
      json.text || "Please clarify which one you meant.",
      json.tts,
      "",
      mode
    );
    const candidates = json.candidates || [];
    if (candidates.length) {
      const msgEl = appendMessage("assistant", "Choose one:", {
        noAnimate: true,
      });
      const listEl = document.createElement("div");
      listEl.style.marginTop = "6px";
      candidates.forEach((q, i) => {
        const row = document.createElement("div");
        row.textContent = `${i + 1}. ${q}`;
        row.style.cursor = "pointer";
        row.addEventListener("click", async () => {
          const form = new FormData();
          form.append("choice_text", q);
          setStatus("Uploading...");
          const resp = await fetch("/api/choice", {
            method: "POST",
            body: form,
          });
          handleResponse(await resp.json());
        });
        listEl.appendChild(row);
      });
      msgEl.appendChild(listEl);
    }
    setStatus("Ready");
    return;
  }

  if (mode === "answer" || mode === "ai" || mode === "not_found") {
    const resource = json.resource || json.link || "";
    showAssistantText(
      json.text || json.full_answer || json.message || "Here is the answer.",
      json.tts,
      resource,
      mode
    );
    setStatus("Ready");
    return;
  }

  if (mode === "inquiry") {
    showAssistantText(
      json.text || "Let's start the inquiry.",
      json.tts,
      "",
      mode
    );
    setStatus("Ready");
    return;
  }

  if (mode === "review") {
    showAssistantText(
      json.text || "Please review the captured details.",
      json.tts,
      "",
      mode
    );

    const answersFromServer = json.answers || {};
    window.api_payload_from_server = json.api_payload || json.payload || null;

    const payloadObj = {
      question_order: json.question_order || json.questions || null,
      api_payload: window.api_payload_from_server,
    };

    showReviewModal(answersFromServer, payloadObj);
    wireReviewSubmitButton();

    setStatus("Ready");
    return;
  }

  // default
  showAssistantText(
    json.text || json.message || "Okay.",
    json.tts || null,
    json.resource || "",
    mode
  );
  setStatus("Ready");
}

// =======================
// Session start (auto) - greet + call /api/start
// Note: when starting session explicitly we stop any wake-word listener
// =======================
async function startSession() {
  stopWakeWordListener();
  // if paused, ensure we are not paused
  sessionPaused = false;
  stopPauseListener();

  if (sessionActive) return;
  sessionActive = true;
  setStatus("Uploading...");
  const greeting = "Hello, welcome to Delhi School of Excellence!";
  appendMessage("assistant", greeting, { noAnimate: true });
  const u = new SpeechSynthesisUtterance(greeting);
  u.onend = async () => {
    setStatus("Uploading...");
    try {
      const resp = await fetch("/api/start", {
        headers: {
          'X-Requested-With': 'XMLHttpRequest'
        }
      });
      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
      }
      const json = await resp.json();
      handleResponse(json);
    } catch (err) {
      console.error("api/start error:", err);
      setStatus("Error: " + err.message);
    }
  };
  window.speechSynthesis.speak(u);
}

// =======================
// Remove visible button handlers (not used)
// =======================
if (startBtn) startBtn.onclick = null;
if (stopBtn) stopBtn.onclick = null;
if (quitBtn) quitBtn.onclick = null;

// =======================
// Initialize on load
// =======================
window.addEventListener("load", async () => {
  await initAudioVisual();
  // keep original behaviour: start session automatically
  startSession();
});
