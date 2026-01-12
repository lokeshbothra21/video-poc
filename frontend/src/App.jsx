import { useState, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import axios from 'axios';
import './index.css';

function App() {
    const [file, setFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [processingTime, setProcessingTime] = useState(null);
    const [error, setError] = useState(null);
    const [activeTab, setActiveTab] = useState('upload');
    const [urlInput, setUrlInput] = useState('');
    const fileInputRef = useRef(null);

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
            setError(null);
            setResult(null);
            setProcessingTime(null);
        }
    };

    const handleUrlAnalyze = async () => {
        if (!urlInput) return;

        setLoading(true);
        setError(null);
        setResult(null);
        setProcessingTime(null);

        try {
            const response = await axios.post('http://localhost:8000/analyze_url', {
                url: urlInput
            });

            setResult(response.data);
            setProcessingTime(response.data.processing_time);
        } catch (err) {
            console.error(err);
            setError(err.response?.data?.detail || err.message || 'An error occurred during analysis.');
        } finally {
            setLoading(false);
        }
    };

    const handleUploadAndAnalyze = async () => {
        if (!file) return;

        setLoading(true);
        setError(null);
        setResult(null);
        setProcessingTime(null);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post('http://localhost:8000/upload_and_analyze', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            setResult(response.data);
            setProcessingTime(response.data.processing_time);
        } catch (err) {
            console.error(err);
            setError(err.response?.data?.detail || err.message || 'An error occurred during analysis.');
        } finally {
            setLoading(false);
        }
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        e.currentTarget.classList.add('drag-active');
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        e.currentTarget.classList.remove('drag-active');
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.currentTarget.classList.remove('drag-active');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            // Check if video
            if (e.dataTransfer.files[0].type.startsWith('video/')) {
                setFile(e.dataTransfer.files[0]);
                setError(null);
            } else {
                setError("⚠️ Please upload a valid video file (MP4, MOV, etc.)");
            }
        }
    };

    // Chat State
    const [chatInput, setChatInput] = useState('');
    const [chatHistory, setChatHistory] = useState([]);
    const [chatLoading, setChatLoading] = useState(false);

    const handleChat = async () => {
        if (!chatInput || !result) return;

        // Optimistic UI update
        const userMessage = { role: 'user', content: chatInput };
        setChatHistory(prev => [...prev, userMessage]);
        setChatInput('');
        setChatLoading(true);

        try {
            const response = await axios.post('http://localhost:8000/rag_chat', {
                query: userMessage.content,
                video_uri: result.video_uri || null
            });

            const botMessage = { role: 'assistant', content: response.data.answer };
            setChatHistory(prev => [...prev, botMessage]);
        } catch (err) {
            console.error(err);
            const errorMessage = { role: 'assistant', content: "⚠️ Sorry, I couldn't answer that. Please try again." };
            setChatHistory(prev => [...prev, errorMessage]);
        } finally {
            setChatLoading(false);
        }
    };

    return (
        <div className="container">
            {/* Header Section */}
            <div className="header-section">
                <div className="badge">Gemini 3 Pro + RAG</div>
                <h1 className="title">Video Intelligence</h1>
                <p className="subtitle">
                    Advanced sports analytics powered by multimodal AI. Upload your match footage or paste a YouTube link to get professional-grade tactical breakdowns.
                </p>
            </div>

            {/* Main Glass Card */}
            <div className="glass-card">

                {/* Tabs */}
                <div className="tabs">
                    <button
                        className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
                        onClick={() => setActiveTab('upload')}
                    >
                        Upload Video
                    </button>
                    <button
                        className={`tab-btn ${activeTab === 'url' ? 'active' : ''}`}
                        onClick={() => setActiveTab('url')}
                    >
                        Enter URL
                    </button>
                </div>

                {/* Upload Tab Content */}
                {activeTab === 'upload' && (
                    <div
                        className={`upload-area ${file ? 'has-file' : ''}`}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        onClick={() => !file && fileInputRef.current.click()}
                    >
                        <input
                            type="file"
                            ref={fileInputRef}
                            style={{ display: 'none' }}
                            accept="video/*"
                            onChange={handleFileChange}
                        />

                        {file ? (
                            <div className="file-info" onClick={(e) => e.stopPropagation()}>
                                <div className="file-details">
                                    <span style={{ fontSize: '2rem' }}>📹</span>
                                    <div>
                                        <div className="file-name">{file.name}</div>
                                        <div className="file-size">{(file.size / (1024 * 1024)).toFixed(1)} MB</div>
                                    </div>
                                </div>
                                <button
                                    className="remove-file"
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        setFile(null);
                                        setResult(null);
                                    }}
                                    title="Remove file"
                                >
                                    ✕
                                </button>
                            </div>
                        ) : (
                            <div className="upload-placeholder">
                                <div className="upload-icon-wrapper">
                                    <span className="upload-icon">☁️</span>
                                </div>
                                <div className="upload-text-primary">Click or drop video here</div>
                                <div className="upload-text-secondary">Supported formats: MP4, MOV, AVI (Max 2GB)</div>
                            </div>
                        )}

                        {/* Show button only if file is selected and not loading */}
                        {file && !loading && !result && (
                            <div style={{ marginTop: '1rem' }}>
                                <small style={{ color: 'var(--text-secondary)' }}>Ready to analyze</small>
                            </div>
                        )}
                    </div>
                )}

                {/* URL Tab Content */}
                {activeTab === 'url' && (
                    <div className="url-input-area">
                        <input
                            type="text"
                            className="url-input"
                            placeholder="Paste YouTube or public video URL..."
                            value={urlInput}
                            onChange={(e) => setUrlInput(e.target.value)}
                        />
                    </div>
                )}

                {/* Action Button */}
                <button
                    className="analyze-btn"
                    onClick={activeTab === 'upload' ? handleUploadAndAnalyze : handleUrlAnalyze}
                    disabled={loading || (activeTab === 'upload' ? !file : !urlInput)}
                >
                    {loading ? (
                        <div className="loading-container" style={{ padding: 0, flexDirection: 'row' }}>
                            <div className="loading-spinner" style={{ width: '1.5rem', height: '1.5rem', borderLeftColor: '#fff' }}></div>
                            <span>Processing Video...</span>
                        </div>
                    ) : result ? (
                        'Analyze Another Video'
                    ) : (
                        '🚀 Start Deep Analysis'
                    )}
                </button>

                {/* Error Message */}
                {error && (
                    <div className="error-message">
                        <span>🚨</span>
                        {error}
                    </div>
                )}

                {/* Results Section */}
                {result && (
                    <div className="result-area">
                        {/* Header */}
                        <div className="result-header" style={{ justifyContent: 'space-between' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                                <span style={{ fontSize: '1.5rem' }}>✨</span>
                                <h2 className="result-title">Analysis Report</h2>
                            </div>
                            {processingTime && (
                                <div className="time-badge">
                                    ⏱️ {processingTime.toFixed(2)}s
                                </div>
                            )}
                        </div>

                        {/* Split View: Analysis & Chat */}
                        <div className="analysis-grid">
                            {/* Left: Deep Analysis Markdown */}
                            <div className="analysis-content">
                                <ReactMarkdown>{typeof result === 'string' ? result : result.analysis}</ReactMarkdown>
                            </div>

                            {/* Right: RAG Chat */}
                            <div className="chat-section">
                                <h3 className="chat-title">💬 Ask Deep Questions</h3>
                                <div className="chat-messages">
                                    {chatHistory.length === 0 && (
                                        <p className="chat-placeholder">Ask about player movement, specific shots, or tactical decisions...</p>
                                    )}
                                    {chatHistory.map((msg, idx) => (
                                        <div key={idx} className={`chat-bubble ${msg.role}`}>
                                            {msg.content}
                                        </div>
                                    ))}
                                    {chatLoading && <div className="chat-bubble assistant">Thinking...</div>}
                                </div>
                                <div className="chat-input-area">
                                    <input
                                        type="text"
                                        className="chat-input"
                                        placeholder="e.g. 'Why did Player 1 lose the point at 00:15?'"
                                        value={chatInput}
                                        onChange={(e) => setChatInput(e.target.value)}
                                        onKeyDown={(e) => e.key === 'Enter' && handleChat()}
                                    />
                                    <button onClick={handleChat} disabled={chatLoading} className="chat-send-btn">
                                        ➤
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

export default App;
