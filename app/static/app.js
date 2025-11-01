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
const STATUS_LABELS = { new: "Nuovo", reviewed: "In revisione" };
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
  const newCount = contacts.filter((contact) => contact.status === "new").length;
  const reviewedCount = total - newCount;
  contactsMeta.textContent = `Totali: ${total} • Nuovi: ${newCount} • In revisione: ${reviewedCount}`;
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
    row.appendChild(createStatusCell(contact));
    row.appendChild(createTagsCell(contact));
    row.appendChild(createNotesCell(contact));
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

function createStatusCell(contact) {
  const cell = document.createElement("td");
  const select = document.createElement("select");
  select.className = "status-select";
  Object.entries(STATUS_LABELS).forEach(([value, label]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    select.appendChild(option);
  });
  select.value = contact.status || "new";

  select.addEventListener("change", async () => {
    const previous = contact.status || "new";
    const chosen = select.value;
    select.disabled = true;
    try {
      await patchContact(contact.id, { status: chosen });
      appendLog(
        `Lead ${contact.email || contact.id} marcato come ${STATUS_LABELS[chosen] || chosen}`,
        "success",
      );
      await loadContacts({ quiet: true });
    } catch (error) {
      appendLog(`Impossibile aggiornare lo stato: ${error.message}`, "error");
      select.value = previous;
    } finally {
      select.disabled = false;
    }
  });

  cell.appendChild(select);
  return cell;
}

function createTagsCell(contact) {
  const cell = document.createElement("td");
  const tagList = document.createElement("div");
  tagList.className = "tag-list";

  if (contact.tags && contact.tags.length) {
    contact.tags.forEach((tag) => {
      const badge = document.createElement("span");
      badge.className = "tag";
      badge.textContent = tag;
      tagList.appendChild(badge);
    });
  } else {
    const placeholder = document.createElement("span");
    placeholder.className = "muted";
    placeholder.textContent = "Nessun tag";
    tagList.appendChild(placeholder);
  }

  const form = document.createElement("form");
  form.className = "tag-form";
  form.noValidate = true;
  const input = document.createElement("input");
  input.type = "text";
  input.placeholder = "Aggiungi tag";
  input.autocomplete = "off";
  const button = document.createElement("button");
  button.type = "submit";
  button.className = "ghost small";
  button.textContent = "Aggiungi";

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const value = input.value.trim();
    if (!value) {
      return;
    }
    button.disabled = true;
    input.disabled = true;
    try {
      await postTag(contact.id, value);
      appendLog(`Tag "${value}" aggiunto a ${contact.email || contact.id}`, "success");
      input.value = "";
      await loadContacts({ quiet: true });
    } catch (error) {
      appendLog(`Impossibile aggiungere il tag: ${error.message}`, "error");
    } finally {
      button.disabled = false;
      input.disabled = false;
    }
  });

  form.appendChild(input);
  form.appendChild(button);
  cell.appendChild(tagList);
  cell.appendChild(form);
  return cell;
}

function createNotesCell(contact) {
  const cell = document.createElement("td");
  const form = document.createElement("form");
  form.className = "note-form";
  form.noValidate = true;
  const textarea = document.createElement("textarea");
  textarea.placeholder = "Annota follow-up, priorità, ecc.";
  textarea.value = contact.notes || "";
  const footer = document.createElement("footer");
  const button = document.createElement("button");
  button.type = "submit";
  button.className = "ghost small";
  button.textContent = "Salva";

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const current = contact.notes || "";
    const updated = textarea.value;
    if (updated.trim() === current.trim()) {
      appendLog("Nessuna modifica alle note", "info");
      return;
    }
    button.disabled = true;
    textarea.disabled = true;
    try {
      await patchContact(contact.id, { notes: updated });
      appendLog(`Note aggiornate per ${contact.email || contact.id}`, "success");
      await loadContacts({ quiet: true });
    } catch (error) {
      appendLog(`Impossibile salvare le note: ${error.message}`, "error");
    } finally {
      button.disabled = false;
      textarea.disabled = false;
    }
  });

  footer.appendChild(button);
  form.appendChild(textarea);
  form.appendChild(footer);
  cell.appendChild(form);
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

async function patchContact(contactId, payload) {
  const response = await fetch(`/contacts/${contactId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    let detail = "";
    try {
      const data = await response.json();
      detail = data.detail;
    } catch (_) {
      /* ignore */
    }
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return response.json();
}

async function postTag(contactId, tag) {
  const response = await fetch(`/contacts/${contactId}/tags`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tag }),
  });
  if (!response.ok) {
    let detail = "";
    try {
      const data = await response.json();
      detail = data.detail;
    } catch (_) {
      /* ignore */
    }
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return response.json();
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
