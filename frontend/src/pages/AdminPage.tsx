import React, { useState, useRef, useEffect, useCallback } from "react";
import { uploadDocuments, extractYoutube, getCronStatus, setCronStatus, listNamespaces, deleteNamespace, listAdmins, createAdmin, getAuthToken, removeAuthToken, type NamespaceInfo } from "../api";
import "./AdminPage.css";

const MAX_UPLOAD_FILES = parseInt(import.meta.env.VITE_MAX_UPLOAD_FILES || "1", 10);

const AdminPage: React.FC = () => {
  const [namespace, setNamespace] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [youtubeUrls, setYoutubeUrls] = useState("");
  const [mode, setMode] = useState<"upload" | "youtube" | "cron" | "namespaces" | "admins">("upload");
  const [isUploading, setIsUploading] = useState(false);
  const [message, setMessage] = useState<{ text: string; type: "success" | "error" | "info" } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [cronEnabled, setCronEnabled] = useState(false);
  const [isCronLoading, setIsCronLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Namespace management state
  const [namespaces, setNamespaces] = useState<NamespaceInfo[]>([]);
  const [namespacesLoading, setNamespacesLoading] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [indexName, setIndexName] = useState("");
  const [totalVectors, setTotalVectors] = useState(0);

  // Admins management state
  const [adminList, setAdminList] = useState<any[]>([]);
  const [adminsLoading, setAdminsLoading] = useState(false);
  const [newAdminName, setNewAdminName] = useState("");
  const [newAdminEmail, setNewAdminEmail] = useState("");
  const [isCreatingAdmin, setIsCreatingAdmin] = useState(false);

  useEffect(() => {
    const token = getAuthToken();
    if (!token) {
      window.location.href = "/admin/login";
    }
  }, []);

  const fetchNamespaces = useCallback(async () => {
    setNamespacesLoading(true);
    try {
      const res = await listNamespaces();
      setNamespaces(res.namespaces);
      setIndexName(res.index_name);
      setTotalVectors(res.total_vectors);
    } catch (err: any) {
      setMessage({ text: err.message, type: "error" });
    } finally {
      setNamespacesLoading(false);
    }
  }, []);

  const fetchAdmins = useCallback(async () => {
    setAdminsLoading(true);
    try {
      const res = await listAdmins();
      setAdminList(res);
    } catch (err: any) {
      setMessage({ text: err.message, type: "error" });
    } finally {
      setAdminsLoading(false);
    }
  }, []);

  useEffect(() => {
    setMessage(null); // Clear message when switching tabs
    if (mode === "cron") {
      setIsCronLoading(true);
      getCronStatus()
        .then(res => setCronEnabled(res.enabled))
        .catch(err => setMessage({ text: err.message, type: "error" }))
        .finally(() => setIsCronLoading(false));
    }
    if (mode === "namespaces") {
      fetchNamespaces();
    }
    if (mode === "admins") {
      fetchAdmins();
    }
  }, [mode, fetchNamespaces, fetchAdmins]);

  useEffect(() => {
    // Auto-dismiss success and info messages after 5 seconds
    if (message && (message.type === "success" || message.type === "info")) {
      const timer = setTimeout(() => {
        setMessage(null);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [message]);

  const handleCronToggle = async () => {
    setIsCronLoading(true);
    setMessage(null);
    try {
      const res = await setCronStatus(!cronEnabled);
      setCronEnabled(res.enabled);
      setMessage({ text: `Daily summary cron is now ${res.enabled ? "enabled" : "disabled"}.`, type: "success" });
    } catch (err: any) {
      setMessage({ text: err.message, type: "error" });
    } finally {
      setIsCronLoading(false);
    }
  };

  const handleDeleteNamespace = async (ns: string) => {
    setIsDeleting(true);
    setMessage(null);
    try {
      const res = await deleteNamespace(ns, true);
      setMessage({ text: res.message, type: "success" });
      setDeleteConfirm(null);
      // Refresh the list
      await fetchNamespaces();
    } catch (err: any) {
      setMessage({ text: err.message, type: "error" });
    } finally {
      setIsDeleting(false);
    }
  };

  const handleCreateAdmin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newAdminName || !newAdminEmail) return;
    
    setIsCreatingAdmin(true);
    setMessage(null);
    try {
      await createAdmin(newAdminName, newAdminEmail);
      setMessage({ text: `Admin created successfully. Welcome email sent to ${newAdminEmail}.`, type: "success" });
      setNewAdminName("");
      setNewAdminEmail("");
      await fetchAdmins();
    } catch (err: any) {
      setMessage({ text: err.message, type: "error" });
    } finally {
      setIsCreatingAdmin(false);
    }
  };

  const handleLogout = () => {
    removeAuthToken();
    window.location.href = "/admin/login";
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const processFiles = (newFiles: File[]) => {
    setMessage(null);
    const pdfFiles = newFiles.filter(f => f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf"));
    
    if (pdfFiles.length < newFiles.length) {
      setMessage({ text: "Only PDF files are allowed.", type: "error" });
    }

    setFiles((prev) => {
      const combined = [...prev, ...pdfFiles];
      if (combined.length > MAX_UPLOAD_FILES) {
        setMessage({ text: `You can only upload up to ${MAX_UPLOAD_FILES} file(s) at once.`, type: "error" });
        return combined.slice(0, MAX_UPLOAD_FILES);
      }
      return combined;
    });
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFiles(Array.from(e.dataTransfer.files));
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFiles(Array.from(e.target.files));
    }
  };

  const removeFile = (indexToRemove: number) => {
    setFiles((prev) => prev.filter((_, idx) => idx !== indexToRemove));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (mode === "upload" && !namespace.trim()) {
      setMessage({ text: "Please enter a namespace.", type: "error" });
      return;
    }

    if (mode === "upload" && files.length === 0) {
      setMessage({ text: "Please select at least one PDF file.", type: "error" });
      return;
    }

    if (mode === "youtube" && !youtubeUrls.trim()) {
      setMessage({ text: "Please enter at least one YouTube URL.", type: "error" });
      return;
    }

    setIsUploading(true);
    setMessage(null);

    try {
      if (mode === "upload") {
        const response = await uploadDocuments(namespace, files);
        setMessage({
          text: `Success! ${response.message}`,
          type: "success",
        });
        setFiles([]);
        if (fileInputRef.current) fileInputRef.current.value = "";
      } else {
        const urls = youtubeUrls.split("\n").map(u => u.trim()).filter(u => u.length > 0);
        let successCount = 0;
        let errors: string[] = [];

        for (const url of urls) {
          try {
            await extractYoutube(url);
            successCount++;
          } catch (err: any) {
            errors.push(err.message || "Unknown error");
          }
        }
        
        if (errors.length > 0) {
          setMessage({
            text: `Extracted ${successCount} PDFs. Errors: ${errors.join("; ")}`,
            type: successCount > 0 ? "info" : "error"
          });
        } else {
          setMessage({
            text: `Success! Extracted and downloaded ${successCount} PDF(s).`,
            type: "success",
          });
          setYoutubeUrls("");
        }
      }
    } catch (error: any) {
      setMessage({
        text: error.message || "Failed to process request.",
        type: "error",
      });
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="admin-container">
      <div className="admin-card">
        <div className="admin-header" style={{ position: 'relative' }}>
          <button 
            onClick={handleLogout} 
            style={{ position: 'absolute', right: '0', top: '0', background: 'none', border: 'none', color: '#dc2626', cursor: 'pointer', fontWeight: 600, padding: '0.5rem', textDecoration: 'underline' }}
          >
            Logout
          </button>
          <h2>Admin Dashboard</h2>
          <p>Manage knowledge base, daily summaries, and administrators.</p>
        </div>

        <form onSubmit={handleSubmit} className="admin-form">
          {mode === "upload" && (
            <div className="form-group">
              <label htmlFor="namespace">Namespace</label>
              <input
                type="text"
                id="namespace"
                value={namespace}
                onChange={(e) => setNamespace(e.target.value)}
                placeholder="e.g., acme-corp"
                disabled={isUploading}
              />
            </div>
          )}

          <div className="mode-toggle">
            <button
              type="button"
              className={`toggle-btn ${mode === "upload" ? "active" : ""}`}
              onClick={() => setMode("upload")}
              disabled={isUploading}
            >
              Upload PDFs
            </button>
            <button
              type="button"
              className={`toggle-btn ${mode === "youtube" ? "active" : ""}`}
              onClick={() => setMode("youtube")}
              disabled={isUploading}
            >
              YouTube Extraction
            </button>
            <button
              type="button"
              className={`toggle-btn ${mode === "cron" ? "active" : ""}`}
              onClick={() => setMode("cron")}
              disabled={isUploading}
            >
              Daily Summary
            </button>
            <button
              type="button"
              className={`toggle-btn ${mode === "namespaces" ? "active" : ""}`}
              onClick={() => setMode("namespaces")}
              disabled={isUploading}
            >
              Namespaces
            </button>
            <button
              type="button"
              className={`toggle-btn ${mode === "admins" ? "active" : ""}`}
              onClick={() => setMode("admins")}
              disabled={isUploading}
            >
              Admins
            </button>
          </div>

          {mode === "upload" && (
            <div className="form-group">
            <label>Upload PDFs</label>
            <div 
              className={`drop-zone ${isDragging ? "dragging" : ""}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                type="file"
                ref={fileInputRef}
                accept=".pdf"
                multiple={MAX_UPLOAD_FILES > 1}
                onChange={handleFileChange}
                disabled={isUploading}
                style={{ display: "none" }}
              />
              <div className="drop-zone-content">
                <svg className="upload-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                  <polyline points="17 8 12 3 7 8"></polyline>
                  <line x1="12" y1="3" x2="12" y2="15"></line>
                </svg>
                <p>Drag & drop your PDFs here, or <span>click to browse</span></p>
                <small>Max {MAX_UPLOAD_FILES} file(s) allowed.</small>
              </div>
            </div>
          </div>
          )}

          {mode === "upload" && files.length > 0 && (
            <div className="file-preview-list">
              <h4>Selected Files</h4>
              <ul>
                {files.map((file, idx) => (
                  <li key={`${file.name}-${idx}`} className="file-item">
                    <div className="file-info">
                      <svg className="pdf-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                      <span className="file-name">{file.name}</span>
                      <span className="file-size">({(file.size / 1024 / 1024).toFixed(2)} MB)</span>
                    </div>
                    <button 
                      type="button" 
                      className="remove-btn" 
                      onClick={(e) => { e.stopPropagation(); removeFile(idx); }}
                      disabled={isUploading}
                    >
                      &times;
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {mode === "youtube" && (
            <div className="form-group">
              <label htmlFor="youtubeUrls">YouTube URLs (one per line)</label>
              <textarea
                id="youtubeUrls"
                value={youtubeUrls}
                onChange={(e) => setYoutubeUrls(e.target.value)}
                placeholder="https://www.youtube.com/watch?v=...&#10;https://www.youtube.com/watch?v=..."
                rows={5}
                disabled={isUploading}
                style={{ width: "100%", padding: "10px", borderRadius: "8px", border: "1px solid var(--border)", resize: "vertical" }}
              />
              <small style={{display: "block", marginTop: "8px", color: "var(--text-light)"}}>
                Extracts transcripts and downloads them directly as PDF files.
              </small>
            </div>
          )}

          {mode === "cron" && (
            <div className="form-group cron-settings">
              <label>Daily Summary Cron Job</label>
              <div style={{ display: "flex", alignItems: "center", gap: "12px", marginTop: "15px", marginBottom: "15px" }}>
                <span style={{ fontSize: "14px", fontWeight: "600", color: cronEnabled ? "var(--success-color, #00d2a0)" : "var(--text-secondary)" }}>
                  {isCronLoading ? "Updating..." : cronEnabled ? "ENABLED" : "DISABLED"}
                </span>
                <label className="switch">
                  <input 
                    type="checkbox" 
                    checked={cronEnabled} 
                    onChange={handleCronToggle} 
                    disabled={isCronLoading || isUploading}
                  />
                  <span className="slider"></span>
                </label>
              </div>
              <small style={{display: "block", marginTop: "8px", color: "var(--text-secondary)"}}>
                When enabled, the system will generate and email a PDF summary of the day's conversations to administrators every night at midnight.
              </small>
            </div>
          )}

          {mode === "namespaces" && (
            <div className="form-group namespace-manager">
              <div className="ns-header-row">
                <label>Pinecone Namespaces</label>
                <button
                  type="button"
                  className="ns-refresh-btn"
                  onClick={fetchNamespaces}
                  disabled={namespacesLoading}
                  title="Refresh"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="16" height="16">
                    <polyline points="23 4 23 10 17 10"></polyline>
                    <polyline points="1 20 1 14 7 14"></polyline>
                    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
                  </svg>
                </button>
              </div>

              {namespacesLoading ? (
                <div className="ns-loading">
                  <span className="loading-spinner"></span>
                  <span>Loading namespaces...</span>
                </div>
              ) : namespaces.length === 0 ? (
                <div className="ns-empty">
                  <p>No namespaces found in index <strong>{indexName || "—"}</strong>.</p>
                </div>
              ) : (
                <>
                  <div className="ns-stats-bar">
                    <span>Index: <strong>{indexName}</strong></span>
                    <span>Total vectors: <strong>{totalVectors.toLocaleString()}</strong></span>
                  </div>
                  <ul className="ns-list">
                    {namespaces.map((ns) => (
                      <li key={ns.name} className="ns-item">
                        <div className="ns-info">
                          <div className="ns-name-row">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="16" height="16" className="ns-icon">
                              <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
                            </svg>
                            <span className="ns-name">{ns.name}</span>
                          </div>
                          <div className="ns-meta">
                            <span className="ns-badge">{ns.vector_count.toLocaleString()} vectors</span>
                            {ns.has_local_docs && <span className="ns-badge ns-badge-local">Local docs</span>}
                          </div>
                        </div>
                        {deleteConfirm === ns.name ? (
                          <div className="ns-confirm">
                            <span className="ns-confirm-text">Delete?</span>
                            <button
                              type="button"
                              className="ns-confirm-yes"
                              onClick={() => handleDeleteNamespace(ns.name)}
                              disabled={isDeleting}
                            >
                              {isDeleting ? "..." : "Yes"}
                            </button>
                            <button
                              type="button"
                              className="ns-confirm-no"
                              onClick={() => setDeleteConfirm(null)}
                              disabled={isDeleting}
                            >
                              No
                            </button>
                          </div>
                        ) : (
                          <button
                            type="button"
                            className="ns-delete-btn"
                            onClick={() => setDeleteConfirm(ns.name)}
                            title={`Delete namespace '${ns.name}'`}
                          >
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="16" height="16">
                              <polyline points="3 6 5 6 21 6"></polyline>
                              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                              <line x1="10" y1="11" x2="10" y2="17"></line>
                              <line x1="14" y1="11" x2="14" y2="17"></line>
                            </svg>
                          </button>
                        )}
                      </li>
                    ))}
                  </ul>
                </>
              )}
              <small style={{display: "block", marginTop: "12px", color: "var(--text-secondary, #6b7280)"}}>
                Deleting a namespace permanently removes all its vectors from Pinecone and its local document files.
              </small>
            </div>
          )}

          {mode === "admins" && (
            <div className="form-group admins-manager">
              <div className="ns-header-row" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <label>Manage Admins</label>
                <button
                  type="button"
                  className="ns-refresh-btn"
                  onClick={fetchAdmins}
                  disabled={adminsLoading}
                  title="Refresh"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" width="16" height="16">
                    <polyline points="23 4 23 10 17 10"></polyline>
                    <polyline points="1 20 1 14 7 14"></polyline>
                    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path>
                  </svg>
                </button>
              </div>

              {adminsLoading ? (
                <div className="ns-loading" style={{ padding: '1rem', textAlign: 'center' }}>
                  <span>Loading admins...</span>
                </div>
              ) : (
                <ul className="ns-list" style={{ marginBottom: '2rem' }}>
                  {adminList.map((admin) => (
                    <li key={admin._id} className="ns-item" style={{ padding: '1rem', borderBottom: '1px solid #e2e8f0', display: 'flex', justifyContent: 'space-between' }}>
                      <div>
                        <strong>{admin.name}</strong>
                        <div style={{ color: '#64748b', fontSize: '0.875rem' }}>{admin.email}</div>
                      </div>
                    </li>
                  ))}
                </ul>
              )}

              <div style={{ marginTop: '2rem', borderTop: '1px solid #e2e8f0', paddingTop: '1rem' }}>
                <h4 style={{ marginBottom: '1rem' }}>Add New Admin</h4>
                <div style={{ display: 'flex', gap: '1rem', flexDirection: 'column' }}>
                  <input
                    type="text"
                    value={newAdminName}
                    onChange={(e) => setNewAdminName(e.target.value)}
                    placeholder="Admin Name"
                    style={{ padding: '0.75rem', border: '1px solid #e2e8f0', borderRadius: '4px' }}
                  />
                  <input
                    type="email"
                    value={newAdminEmail}
                    onChange={(e) => setNewAdminEmail(e.target.value)}
                    placeholder="Admin Email"
                    style={{ padding: '0.75rem', border: '1px solid #e2e8f0', borderRadius: '4px' }}
                  />
                  <button
                    type="button"
                    onClick={handleCreateAdmin}
                    disabled={isCreatingAdmin || !newAdminName || !newAdminEmail}
                    style={{ padding: '0.75rem', backgroundColor: '#2563eb', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer' }}
                  >
                    {isCreatingAdmin ? "Creating..." : "Create Admin"}
                  </button>
                </div>
              </div>
            </div>
          )}

          {mode !== "cron" && mode !== "namespaces" && mode !== "admins" && (
            <button type="submit" className="submit-btn" disabled={isUploading || (mode === "upload" && (!namespace || files.length === 0)) || (mode === "youtube" && !youtubeUrls.trim())}>
              {isUploading ? (
                <span className="loading-spinner"></span>
              ) : null}
              {isUploading ? "Processing..." : mode === "upload" ? "Upload & Ingest" : "Extract to PDF"}
            </button>
          )}
        </form>

        {message && (
          <div className={`message-box ${message.type}`} style={{ position: 'relative' }}>
            <span>{message.text}</span>
            <button 
              onClick={() => setMessage(null)} 
              style={{
                position: 'absolute',
                top: '10px',
                right: '10px',
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                fontSize: '16px',
                lineHeight: '1',
                padding: '0 5px',
                color: 'inherit',
                opacity: 0.7
              }}
              title="Close"
            >
              &times;
            </button>
          </div>
        )}

        <div className="back-link" style={{ padding: '0 1rem' }}>
          <a href="/">&larr; Back to Chat</a>
        </div>
      </div>
    </div>
  );
};

export default AdminPage;
