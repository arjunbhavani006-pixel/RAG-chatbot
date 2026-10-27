
const uploadForm = document.getElementById("upload-form");
const pdfInput = document.getElementById("pdf-file");
const uploadStatus = document.getElementById("upload-status");
const documentList = document.getElementById("document-list");

const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question");
const chatMessages = document.getElementById("chat-messages");
const sendButton = document.getElementById("send-button");
const welcome = document.getElementById("welcome");

async function apiRequest(url, options = {}) {
    const response = await fetch(url, options);

    let data;
    try {
        data = await response.json();
    } catch {
        throw new Error("Server returned an invalid response.");
    }

    if (!response.ok) {
        throw new Error(data.detail || "Request failed.");
    }

    return data;
}

function addMessage(text, role) {
    welcome.hidden = true;

    const message = document.createElement("div");
    message.className = `message ${role}`;
    message.textContent = text;

    chatMessages.appendChild(message);
    message.scrollIntoView({ behavior: "smooth", block: "nearest" });

    return message;
}

function addSources(message, sources) {
    if (!sources || sources.length === 0) return;

    const container = document.createElement("div");
    container.className = "sources";

    const heading = document.createElement("strong");
    heading.textContent = "Sources";
    container.appendChild(heading);

    sources.forEach((source) => {
        const item = document.createElement("div");
        item.className = "source-item";

        const title = document.createElement("strong");
        title.textContent =
            `[Source ${source.source}] ${source.filename} — Page ${source.page}`;

        const snippet = document.createElement("p");
        snippet.textContent = source.snippet;

        item.appendChild(title);
        item.appendChild(snippet);
        container.appendChild(item);
    });

    message.appendChild(container);
}

async function loadDocuments() {
    documentList.replaceChildren();

    const loading = document.createElement("p");
    loading.className = "muted";
    loading.textContent = "Loading documents...";
    documentList.appendChild(loading);

    try {
        const data = await apiRequest("/api/documents");
        documentList.replaceChildren();

        if (!data.documents.length) {
            const empty = document.createElement("p");
            empty.className = "muted";
            empty.textContent = "No PDFs uploaded yet.";
            documentList.appendChild(empty);
            return;
        }

        data.documents.forEach((doc) => {
            const item = document.createElement("div");
            item.className = "document-item";

            const name = document.createElement("strong");
            name.textContent = doc.filename;

            const details = document.createElement("small");
            details.textContent = `${doc.chunks} text chunks`;

            const removeButton = document.createElement("button");
            removeButton.className = "delete-button";
            removeButton.type = "button";
            removeButton.textContent = "Delete document";

            removeButton.addEventListener("click", async () => {
                const confirmed = window.confirm(
                    `Delete "${doc.filename}" from the knowledge base?`
                );

                if (!confirmed) return;

                removeButton.disabled = true;

                try {
                    await apiRequest(
                        `/api/documents/${encodeURIComponent(doc.document_id)}`,
                        { method: "DELETE" }
                    );
                    uploadStatus.textContent = "Document deleted.";
                    await loadDocuments();
                } catch (error) {
                    uploadStatus.textContent = error.message;
                    removeButton.disabled = false;
                }
            });

            item.append(name, details, removeButton);
            documentList.appendChild(item);
        });
    } catch (error) {
        documentList.replaceChildren();

        const errorMessage = document.createElement("p");
        errorMessage.className = "muted";
        errorMessage.textContent = error.message;
        documentList.appendChild(errorMessage);
    }
}

uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = pdfInput.files[0];

    if (!file) {
        uploadStatus.textContent = "Please select a PDF.";
        return;
    }

    if (file.size > 20 * 1024 * 1024) {
        uploadStatus.textContent = "PDF must be 20 MB or smaller.";
        return;
    }

    const button = uploadForm.querySelector("button");
    button.disabled = true;
    uploadStatus.textContent = "Extracting text and creating embeddings...";

    try {
        const formData = new FormData();
        formData.append("file", file);

        const data = await apiRequest("/api/upload", {
            method: "POST",
            body: formData
        });

        uploadStatus.textContent =
            `${data.filename} processed successfully. ` +
            `${data.chunks_created} chunks created.`;

        pdfInput.value = "";
        await loadDocuments();
    } catch (error) {
        uploadStatus.textContent = `Upload failed: ${error.message}`;
    } finally {
        button.disabled = false;
    }
});

chatForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const question = questionInput.value.trim();
    if (!question) return;

    addMessage(question, "user");
    questionInput.value = "";
    sendButton.disabled = true;

    const loadingMessage = addMessage(
        "Searching your documents and preparing an answer...",
        "assistant"
    );

    try {
        const data = await apiRequest("/api/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question })
        });

        loadingMessage.textContent = data.answer;
        addSources(loadingMessage, data.sources);
    } catch (error) {
        loadingMessage.textContent = `Error: ${error.message}`;
    } finally {
        sendButton.disabled = false;
        questionInput.focus();
    }
});

document.getElementById("refresh-documents")
    .addEventListener("click", loadDocuments);

document.querySelectorAll(".suggestion").forEach((button) => {
    button.addEventListener("click", () => {
        questionInput.value = button.textContent.trim();
        questionInput.focus();
    });
});

loadDocuments();
