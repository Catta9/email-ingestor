const startBtn = document.getElementById("start-btn");
const refreshBtn = document.getElementById("refresh-btn");
const statusBadge = document.getElementById("status-badge");
const logList = document.getElementById("log-list");
const contactsBody = document.querySelector("#contacts-table tbody");
const summaryBox = document.getElementById("summary");

const MAX_LOG_ITEMS = 120;
let eventSource = null;

function timestamp() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function setStatus(state, message) {
  statusBadge.dataset.state = state;
  statusBadge.textContent = message;
  statusBadge.className = `status status-${state}`;
  startBtn.disabled = state === "running";
}

function appendLog(message, level = "info") {
  const item = document.createElement("li");
  item.className = `log-item ${level}`;
  const time = document.createElement("time");
  time.textContent = timestamp();
  item.appendChild(time);
  const content = document.createElement("div");
  content.textContent = message;
  item.appendChild(content);
  logList.prepend(item);

  while (logList.children.length > MAX_LOG_ITEMS) {
    logList.removeChild(logList.lastChild);
  }
}

async function loadContacts() {
  try {
    const response = await fetch("/contacts?limit=200");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const contacts = await response.json();
    renderContacts(contacts);
  } catch (error) {
    appendLog(`Errore nel caricamento dei contatti: ${error.message}`, "error");
  }
}

function renderContacts(contacts) {
  contactsBody.innerHTML = "";
  if (!contacts.length) {
    const emptyRow = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 5;
    cell.textContent = "Nessun contatto disponibile";
    cell.className = "muted";
    emptyRow.appendChild(cell);
    contactsBody.appendChild(emptyRow);
    return;
  }

  contacts.forEach((contact) => {
    const row = document.createElement("tr");

    const name = `${contact.first_name || ""} ${contact.last_name || ""}`.trim() || "-";
    row.appendChild(createCell(name));

    const emailCell = document.createElement("td");
    if (contact.email) {
      const link = document.createElement("a");
      link.href = `mailto:${contact.email}`;
      link.textContent = contact.email;
      link.rel = "noopener";
      emailCell.appendChild(link);
    } else {
      emailCell.textContent = "-";
    }
    row.appendChild(emailCell);

    row.appendChild(createCell(contact.org || "-"));
    row.appendChild(createCell(contact.phone || "-"));

    const lastMessage = document.createElement("td");
    const subject = contact.last_message_subject || "-";
    const receivedAt = contact.last_message_received_at
      ? new Date(contact.last_message_received_at).toLocaleString()
      : "";
    lastMessage.innerHTML = `${subject}<div class="muted">${receivedAt}</div>`;
    row.appendChild(lastMessage);

    contactsBody.appendChild(row);
  });
}

function createCell(content) {
  const cell = document.createElement("td");
  cell.textContent = content;
  return cell;
}

function handleEvent(event) {
  const { type, message, data = {} } = event;

  switch (type) {
    case "status": {
      const state = data.state || "idle";
      setStatus(state, message || state);
      if (state !== "running") {
        startBtn.disabled = false;
      }
      break;
    }
    case "run_started": {
      appendLog(message, "info");
      summaryBox.textContent = "Ingestione avviata...";
      break;
    }
    case "run_progress": {
      const processed = data.processed ?? 0;
      const skipped = data.skipped ?? 0;
      const leads = data.leads ?? 0;
      const total = data.total ?? 0;
      summaryBox.textContent = `Elaborate ${processed}/${total} email • Lead nuovi: ${leads} • Saltate: ${skipped}`;
      break;
    }
    case "lead_created": {
      appendLog(`${message} (UID ${data.imap_uid || "?"})`, "success");
      loadContacts();
      break;
    }
    case "email_processed": {
      appendLog(message, "info");
      break;
    }
    case "email_skipped": {
      appendLog(`${message} (${data.reason || "motivo sconosciuto"})`, "warning");
      break;
    }
    case "run_completed": {
      const processed = data.processed ?? 0;
      const leads = data.new_leads ?? 0;
      const skipped = data.skipped ?? 0;
      const total = data.total ?? 0;
      appendLog(`Ingestione completata: ${processed}/${total} email, nuovi lead ${leads}, saltate ${skipped}`, "success");
      summaryBox.textContent = `Ultima esecuzione: ${processed}/${total} email, nuovi lead ${leads}, saltate ${skipped}`;
      loadContacts();
      break;
    }
    case "run_failed": {
      appendLog(message, "error");
      summaryBox.textContent = message;
      setStatus("error", message);
      break;
    }
    default: {
      appendLog(message, "info");
    }
  }
}

function connectEventStream() {
  if (eventSource) {
    eventSource.close();
  }

  eventSource = new EventSource("/ingestion/stream");
  eventSource.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      handleEvent(payload);
    } catch (error) {
      console.error("Impossibile analizzare l'evento SSE", error);
    }
  };

  eventSource.onerror = () => {
    appendLog("Connessione agli eventi interrotta, nuovo tentativo...", "warning");
  };
}

async function triggerIngestion() {
  startBtn.disabled = true;
  try {
    const response = await fetch("/ingestion/run", { method: "POST" });
    if (response.status === 409) {
      const payload = await response.json();
      appendLog(payload.detail || "Ingestione già in esecuzione", "warning");
      return;
    }
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    appendLog("Ingestione avviata", "info");
  } catch (error) {
    appendLog(`Errore nell'avvio dell'ingestione: ${error.message}`, "error");
    startBtn.disabled = false;
  }
}

startBtn.addEventListener("click", triggerIngestion);
refreshBtn.addEventListener("click", loadContacts);

connectEventStream();
loadContacts();
