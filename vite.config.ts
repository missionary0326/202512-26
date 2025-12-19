import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'fs';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, '.', '');
    return {
      server: {
        port: 3000,
        host: '0.0.0.0',
        fs: {
          // Allow serving files from project root
          allow: ['..']
        },
        middlewareMode: false,
      },
      plugins: [
        react(),
        // Plugin to serve output folder
        {
          name: 'serve-output',
          configureServer(server) {
            server.middlewares.use('/output', (req, res, next) => {
              const filePath = path.join(__dirname, 'output', req.url);
              if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
                res.setHeader('Content-Type', 'text/csv');
                fs.createReadStream(filePath).pipe(res);
              } else {
                next();
              }
            });
          }
        }
      ],
      define: {
        'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
        'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY)
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      },
    };
});
