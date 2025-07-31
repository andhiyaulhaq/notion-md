import { useEffect, useState } from 'react';

import { marked } from 'marked';

interface NotionMarkdownReaderProps {
  content: string;
}

export default function NotionMarkdownReader({
  content,
}: NotionMarkdownReaderProps) {
  const [html, setHtml] = useState('');

  useEffect(() => {
    // Configure Marked with custom options
    marked.setOptions({
      gfm: true,
      breaks: true,
      // Add any custom renderers here
    });

    // Parse Markdown
    const parsed = marked.parse(content);
    if (parsed instanceof Promise) {
      parsed.then((result) => setHtml(result));
    } else {
      setHtml(parsed);
    }
  }, [content]);

  return (
    <div className="notion-md" dangerouslySetInnerHTML={{ __html: html }} />
  );
}
