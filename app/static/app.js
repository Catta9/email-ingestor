const startBtn = document.getElementById("start-btn");
const refreshBtn = document.getElementById("refresh-btn");
const statusBadge = document.getElementById("status-badge");
const logList = document.getElementById("log-list");
const contactsBody = document.querySelector("#contacts-table tbody");
const summaryBox = document.getElementById("summary");
const contactsMeta = document.getElementById("contacts-meta");
const contactsColumnCount = document.querySelectorAll("#contacts-table thead th").length;
const folderLabel = document.getElementById("imap-folder");
const exportBtn = document.getElementById("export-btn");
const statTotal = document.getElementById("stat-total");
const statProcessed = document.getElementById("stat-processed");
const statLeads = document.getElementById("stat-leads");
const statSkipped = document.getElementById("stat-skipped");

const MAX_LOG_ITEMS = 120;
const LEVEL_ICONS = {
  info: "ℹ️",
  success: "✅",
  warning: "⚠️",
  error: "❌",
};

let eventSource = null;
// Stato aggregato mostrato nel riepilogo rapido.
const statsState = {
  total: 0,
  processed: 0,
  leads: 0,
  skipped: 0,
};

// Aggiorna il DOM con i valori correnti del riepilogo.
function renderStats() {
  statTotal.textContent = statsState.total;
  statProcessed.textContent = statsState.processed;
  statLeads.textContent = statsState.leads;
  statSkipped.textContent = statsState.skipped;
}

function setStats(partial = {}) {
  let updated = false;
  Object.entries(partial).forEach(([key, value]) => {
    if (value === undefined || value === null || Number.isNaN(value)) {
      return;
    }
    if (statsState[key] !== value) {
      statsState[key] = value;
      updated = true;
    }
  });
  if (updated) {
    renderStats();
  }
}

function incrementStat(key) {
  if (Object.prototype.hasOwnProperty.call(statsState, key)) {
    statsState[key] += 1;
    renderStats();
  }
}

function timestamp() {
  return new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
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
  const icon = document.createElement("span");
  icon.className = "icon";
  icon.textContent = LEVEL_ICONS[level] || "•";
  item.appendChild(icon);
  const body = document.createElement("div");
  const time = document.createElement("time");
  time.textContent = timestamp();
  const content = document.createElement("div");
  content.textContent = message;
  body.appendChild(time);
  body.appendChild(content);
  item.appendChild(body);
  logList.prepend(item);

  while (logList.children.length > MAX_LOG_ITEMS) {
    logList.removeChild(logList.lastChild);
  }
}

function ensureEventStreamClosed() {
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
}

function connectEventStream() {
  ensureEventStreamClosed();
  eventSource = new EventSource(`/ingestion/stream`);
  eventSource.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      handleEvent(payload);
    } catch (error) {
      console.error("Impossibile analizzare l'evento SSE", error);
    }
  };

  eventSource.onerror = () => {
    setStatus("disconnected", "Scollegato");
    appendLog("Connessione agli eventi interrotta, nuovo tentativo...", "warning");
    setTimeout(connectEventStream, 4000);
  };
}

function updateContactsMeta(contacts) {
  if (!contactsMeta) {
    return;
  }
  if (!contacts.length) {
    contactsMeta.textContent = "Nessun contatto disponibile.";
    return;
  }

  const total = contacts.length;
  contactsMeta.textContent = `Totali: ${total}`;
}

async function loadContacts(options = {}) {
  const { quiet = false } = options;

  try {
    const response = await fetch("/contacts?limit=200");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const contacts = await response.json();
    renderContacts(contacts);
    updateContactsMeta(contacts);
  } catch (error) {
    renderContacts([], { emptyMessage: error.message });
    updateContactsMeta([]);
    if (!quiet) {
      appendLog(`Errore nel caricamento dei contatti: ${error.message}`, "error");
    }
  }
}

function renderContacts(contacts, options = {}) {
  const { emptyMessage } = options;
  contactsBody.innerHTML = "";
  if (!contacts.length) {
    const emptyRow = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = contactsColumnCount;
    cell.className = "muted table-empty";
    cell.textContent = emptyMessage || "Nessun contatto disponibile";
    emptyRow.appendChild(cell);
    contactsBody.appendChild(emptyRow);
    return;
  }

  contacts.forEach((contact) => {
    const row = document.createElement("tr");
    row.dataset.contactId = contact.id;

    const name = `${contact.first_name || ""} ${contact.last_name || ""}`.trim() || "-";
    row.appendChild(createTextCell(name));
    row.appendChild(createEmailCell(contact));
    row.appendChild(createTextCell(contact.org || "-"));
    row.appendChild(createTextCell(contact.phone || "-"));
    row.appendChild(createLastMessageCell(contact));

    contactsBody.appendChild(row);
  });
}

function createTextCell(content) {
  const cell = document.createElement("td");
  cell.textContent = content;
  return cell;
}

function createEmailCell(contact) {
  const cell = document.createElement("td");
  if (contact.email) {
    const link = document.createElement("a");
    link.href = `mailto:${contact.email}`;
    link.textContent = contact.email;
    link.rel = "noopener";
    cell.appendChild(link);
  } else {
    cell.textContent = "-";
  }
  return cell;
}

function createLastMessageCell(contact) {
  const cell = document.createElement("td");
  const subject = contact.last_message_subject || "-";
  const receivedAt = contact.last_message_received_at
    ? new Date(contact.last_message_received_at).toLocaleString()
    : "";
  const excerpt = contact.last_message_excerpt ? contact.last_message_excerpt.slice(0, 140) : "";
  cell.innerHTML = `${subject}<div class="muted">${receivedAt}</div>`;
  if (excerpt) {
    const excerptEl = document.createElement("div");
    excerptEl.className = "muted";
    excerptEl.textContent = excerpt;
    cell.appendChild(excerptEl);
  }
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
      if (folderLabel && data.folder) {
        folderLabel.textContent = data.folder;
      }
      setStats({ total: data.total ?? 0, processed: 0, leads: 0, skipped: 0 });
      break;
    }
    case "run_progress": {
      const processed = data.processed ?? 0;
      const skipped = data.skipped ?? 0;
      const leads = data.leads ?? 0;
      const total = data.total ?? 0;
      summaryBox.textContent = `Elaborate ${processed}/${total} email • Lead nuovi: ${leads} • Saltate: ${skipped}`;
      setStats({ total, processed, leads, skipped });
      break;
    }
    case "lead_created": {
      appendLog(`${message} (UID ${data.imap_uid || "?"})`, "success");
      incrementStat("leads");
      loadContacts({ quiet: true });
      break;
    }
    case "email_processed": {
      appendLog(message, "info");
      incrementStat("processed");
      break;
    }
    case "email_skipped": {
      appendLog(`${message} (${data.reason || "motivo sconosciuto"})`, "warning");
      incrementStat("skipped");
      break;
    }
    case "run_completed": {
      const processed = data.processed ?? 0;
      const leads = data.new_leads ?? 0;
      const skipped = data.skipped ?? 0;
      const total = data.total ?? 0;
      appendLog(
        `Ingestione completata: ${processed}/${total} email, nuovi lead ${leads}, saltate ${skipped}`,
        "success",
      );
      summaryBox.textContent = `Ultima esecuzione: ${processed}/${total} email, nuovi lead ${leads}, saltate ${skipped}`;
      setStats({ total, processed, leads, skipped });
      loadContacts({ quiet: true });
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

async function triggerIngestion() {
  startBtn.disabled = true;
  try {
    const response = await fetch("/ingestion/run", {
      method: "POST",
    });
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
refreshBtn.addEventListener("click", () => loadContacts());
exportBtn.addEventListener("click", () => {
  appendLog("Esportazione Excel avviata", "info");
});

window.addEventListener("beforeunload", ensureEventStreamClosed);

renderStats();
connectEventStream();
loadContacts();
