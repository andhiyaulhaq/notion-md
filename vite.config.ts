import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';

// https://vitejs.dev/config/
// export default defineConfig({
//   plugins: [react()],
// })

export default defineConfig({
  plugins: [react()],
  build: {
    // You can add other build options here
  },
  // Include markdown files in build
  assetsInclude: ['**/*.md'],
});
