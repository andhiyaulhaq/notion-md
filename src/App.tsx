import './styles/utilities.css';
import './styles/notion-md.css';

import { useEffect, useState } from 'react';

import FileSelector from './components/FileSelector';
import NotionMarkdownReader from './components/NotionMarkdownReader';
import { loadMarkdown } from './utils/markdownLoader';

export default function App() {
  const [markdownContent, setMarkdownContent] = useState('');
  const [currentFile, setCurrentFile] = useState('sample.md');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load markdown when file changes
  useEffect(() => {
    const fetchMarkdown = async () => {
      setLoading(true);
      setError(null);

      try {
        const content = await loadMarkdown(currentFile);
        setMarkdownContent(content);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : 'An unknown error occurred',
        );
      } finally {
        setLoading(false);
      }
    };

    fetchMarkdown();
  }, [currentFile]);

  const handleFileChange = (filename: string) => {
    setCurrentFile(filename);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Notion-Style Markdown Reader</h1>
        <FileSelector
          currentFile={currentFile}
          onFileChange={handleFileChange}
        />
      </header>

      <main className="content-container">
        {loading && (
          <div className="loading-state">
            <div className="loader"></div>
            <p>Loading content...</p>
          </div>
        )}

        {error && (
          <div className="error-state">
            <h2>Error Loading Content</h2>
            <p>{error}</p>
            <button
              className="retry-button"
              onClick={() => setCurrentFile(currentFile)}
            >
              Retry
            </button>
          </div>
        )}

        {!loading && !error && (
          <NotionMarkdownReader content={markdownContent} />
        )}
      </main>

      <footer className="app-footer">
        <p>Content loaded from: {currentFile}</p>
        <p>Last updated: {new Date().toLocaleTimeString()}</p>
      </footer>
    </div>
  );
}
