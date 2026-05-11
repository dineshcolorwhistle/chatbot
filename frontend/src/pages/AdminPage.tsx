import React, { useState, useRef } from "react";
import { uploadDocuments, extractYoutube } from "../api";
import "./AdminPage.css";

const MAX_UPLOAD_FILES = parseInt(import.meta.env.VITE_MAX_UPLOAD_FILES || "1", 10);

const AdminPage: React.FC = () => {
  const [namespace, setNamespace] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [youtubeUrls, setYoutubeUrls] = useState("");
  const [mode, setMode] = useState<"upload" | "youtube">("upload");
  const [isUploading, setIsUploading] = useState(false);
  const [message, setMessage] = useState<{ text: string; type: "success" | "error" | "info" } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

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
        <div className="admin-header">
          <h2>Knowledge Base Uploader</h2>
          <p>Upload PDF documents to ingest into a specific namespace.</p>
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

          <button type="submit" className="submit-btn" disabled={isUploading || (mode === "upload" && (!namespace || files.length === 0)) || (mode === "youtube" && !youtubeUrls.trim())}>
            {isUploading ? (
              <span className="loading-spinner"></span>
            ) : null}
            {isUploading ? "Processing..." : mode === "upload" ? "Upload & Ingest" : "Extract to PDF"}
          </button>
        </form>

        {message && (
          <div className={`message-box ${message.type}`}>
            {message.text}
          </div>
        )}

        <div className="back-link">
          <a href="/">&larr; Back to Chat</a>
        </div>
      </div>
    </div>
  );
};

export default AdminPage;
