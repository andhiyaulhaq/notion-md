import 'katex/dist/katex.min.css';
import '../styles/notion-md.css';

// ...existing code...
import { useEffect, useRef, useState } from 'react';

import { marked } from 'marked';
import mermaid from 'mermaid';
import renderMathInElement from 'katex/contrib/auto-render';

interface NotionMarkdownReaderProps {
  content: string;
}

export default function NotionMarkdownReader({
  content,
}: NotionMarkdownReaderProps) {
  const [html, setHtml] = useState('');
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Mermaid extension for marked
    const mermaidExtension = {
      extensions: [
        {
          name: 'mermaid',
          level: 'block',
          start(src: string) {
            return src.match(/^```mermaid/)?.index;
          },
          tokenizer(src: string) {
            const match = src.match(/^```mermaid\s*\n([\s\S]+?)```/);
            if (match) {
              return {
                type: 'mermaid',
                raw: match[0],
                text: match[1],
                tokens: [],
              };
            }
            return undefined;
          },
          renderer(token: any) {
            // Render as a div for Mermaid to process later
            return `<div class="mermaid">${token.text}</div>`;
          },
        },
      ],
    };

    marked.use(mermaidExtension);

    // Add a custom tokenizer for math
    const mathExtension = {
      extensions: [
        {
          name: 'math',
          level: 'inline',
          start(src: string) {
            return src.match(/\$+/)?.index;
          },
          tokenizer(src: string) {
            // Inline math: $...$
            const match = src.match(/^\$([^\$]+?)\$/);
            if (match) {
              return {
                type: 'math',
                raw: match[0],
                text: match[1],
                tokens: [],
              };
            }
            return undefined;
          },
          renderer(token: any) {
            // Keep as $...$ for KaTeX to pick up
            return `$${token.text}$`;
          },
        },
        {
          name: 'mathBlock',
          level: 'block',
          start(src: string) {
            return src.match(/\$\$/)?.index;
          },
          tokenizer(src: string) {
            // Block math: $$...$$
            const match = src.match(/^\$\$([^$]+?)\$\$/m);
            if (match) {
              return {
                type: 'mathBlock',
                raw: match[0],
                text: match[1],
                tokens: [],
              };
            }
            return undefined;
          },
          renderer(token: any) {
            // Keep as $$...$$ for KaTeX to pick up
            return `$$${token.text}$$`;
          },
        },
      ],
    };

    marked.use(mathExtension);

    marked.setOptions({
      gfm: true,
      breaks: true,
    });

    const parsed = marked.parse(content);

    // Post-process: Replace :icon: Question text with <p>icon Question text</p>
    function replaceIconLine(html: string) {
      // Replace lines starting with :icon: (icon can be any emoji or character)
      html = html.replace(
        /<p>:(.{1,2}):\s*(.*?)<\/p>/g,
        (_match, icon, text) => `<p>${icon} ${text}</p>`,
      );
      // Move icon inside blockquote as <span>icon</span> before <p>text</p>
      html = html.replace(
        /<aside>\s*([^\s<]+)\s*<blockquote>\s*<p>(.*?)<\/p>\s*<\/blockquote>\s*<\/aside>/gs,
        (_match, icon, text) =>
          `<aside><blockquote><span>${icon}</span>&nbsp;&nbsp;&nbsp;<span>${text}</span></blockquote></aside>`,
      );
      return html;
    }

    if (parsed instanceof Promise) {
      parsed.then((result) => setHtml(replaceIconLine(result)));
    } else {
      setHtml(replaceIconLine(parsed));
    }
  }, [content]);

  useEffect(() => {
    if (ref.current) {
      // Render math
      renderMathInElement(ref.current, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false },
        ],
      });
      // Render mermaid diagrams
      mermaid.init(undefined, ref.current.querySelectorAll('.mermaid'));
    }
  }, [html]);

  return (
    <div
      ref={ref}
      className="notion-md"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
