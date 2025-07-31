/**
 * Loads Markdown content from a file in src/contents
 * @param filename - Markdown filename (e.g., "sample.md")
 * @returns Promise with markdown content
 */
export const loadMarkdown = async (filename: string): Promise<string> => {
  // Use Vite's glob to import all markdown files
  const modules = import.meta.glob('../contents/*.md', { as: 'raw' });
  const path = `../contents/${filename}`;
  const loader = modules[path];
  if (!loader) {
    throw new Error(`File not found: ${filename}`);
  }
  return await loader();
};

/**
 * Gets available markdown files in src/contents directory
 */
export const getAvailableFiles = (): string[] => {
  // Use Vite's glob to get all markdown files
  const modules = import.meta.glob('../contents/*.md');
  return Object.keys(modules).map((path) => path.split('/').pop()!);
};
