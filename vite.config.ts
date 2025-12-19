import path from "path";
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import fs from "fs";

export default defineConfig(({ mode, command }) => {
  const env = loadEnv(mode, ".", "");

  return {
    base: "/202512-26/",

    server: {
      port: 3000,
      host: "0.0.0.0",
      fs: { allow: [".."] },
      middlewareMode: false,
    },

    plugins: [
      react(),

      command === "serve" && {
        name: "serve-output",
        configureServer(server) {
          server.middlewares.use("/output", (req, res, next) => {
            const filePath = path.join(__dirname, "output", req.url || "");
            if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
              res.setHeader("Content-Type", "text/csv");
              fs.createReadStream(filePath).pipe(res);
            } else {
              next();
            }
          });
        },
      },

      {
        name: "copy-output-to-dist",
        apply: "build",
        closeBundle() {
          const srcDir = path.resolve(__dirname, "output");
          const dstDir = path.resolve(__dirname, "dist", "output");

          if (!fs.existsSync(srcDir)) return;
          fs.mkdirSync(dstDir, { recursive: true });

          for (const file of fs.readdirSync(srcDir)) {
            const src = path.join(srcDir, file);
            const dst = path.join(dstDir, file);
            if (fs.statSync(src).isFile()) fs.copyFileSync(src, dst);
          }
        },
      },
    ].filter(Boolean),

    define: {
      "process.env.API_KEY": JSON.stringify(env.GEMINI_API_KEY),
      "process.env.GEMINI_API_KEY": JSON.stringify(env.GEMINI_API_KEY),
    },

    resolve: {
      alias: {
        "@": path.resolve(__dirname, "."),
      },
    },
  };
});
