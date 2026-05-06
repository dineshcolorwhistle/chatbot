import React, { useState, useRef } from "react";
import { uploadDocuments } from "../api";
import "./AdminPage.css";

const MAX_UPLOAD_FILES = parseInt(import.meta.env.VITE_MAX_UPLOAD_FILES || "1", 10);

const AdminPage: React.FC = () => {
  const [namespace, setNamespace] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [message, setMessage] = useState<{ text: string; type: "success" | "error" } | null>(null);
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
    if (!namespace.trim()) {
      setMessage({ text: "Please enter a namespace.", type: "error" });
      return;
    }
    if (files.length === 0) {
      setMessage({ text: "Please select at least one PDF file.", type: "error" });
      return;
    }

    setIsUploading(true);
    setMessage(null);

    try {
      const response = await uploadDocuments(namespace, files);
      setMessage({
        text: `Success! ${response.message}`,
        type: "success",
      });
      setFiles([]);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (error: any) {
      setMessage({
        text: error.message || "Failed to upload and ingest documents.",
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

          {files.length > 0 && (
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

          <button type="submit" className="submit-btn" disabled={isUploading || !namespace || files.length === 0}>
            {isUploading ? (
              <span className="loading-spinner"></span>
            ) : null}
            {isUploading ? "Uploading & Ingesting..." : "Upload & Ingest"}
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
