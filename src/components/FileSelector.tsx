import { useEffect, useState } from 'react';

import { getAvailableFiles } from '../utils/markdownLoader';

interface FileSelectorProps {
  currentFile: string;
  onFileChange: (filename: string) => void;
}

export default function FileSelector({
  currentFile,
  onFileChange,
}: FileSelectorProps) {
  const [files, setFiles] = useState<string[]>([]);

  useEffect(() => {
    setFiles(getAvailableFiles());
  }, []);

  return (
    <div className="file-selector">
      <label htmlFor="markdown-files">Select Document: </label>
      <select
        id="markdown-files"
        value={currentFile}
        onChange={(e) => onFileChange(e.target.value)}
        className="file-dropdown"
      >
        {files.map((file) => (
          <option key={file} value={file}>
            {file}
          </option>
        ))}
      </select>
    </div>
  );
}
