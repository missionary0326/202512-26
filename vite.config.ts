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

      command === "serve" && {
        name: "serve-data",
        configureServer(server) {
          server.middlewares.use("/data", (req, res, next) => {
            const filePath = path.join(__dirname, "data", req.url || "");
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
          // Copy output folder
          const outputSrcDir = path.resolve(__dirname, "output");
          const outputDstDir = path.resolve(__dirname, "dist", "output");

          if (fs.existsSync(outputSrcDir)) {
            fs.mkdirSync(outputDstDir, { recursive: true });
            for (const file of fs.readdirSync(outputSrcDir)) {
              const src = path.join(outputSrcDir, file);
              const dst = path.join(outputDstDir, file);
              if (fs.statSync(src).isFile()) fs.copyFileSync(src, dst);
            }
          }

          // Copy data folder
          const dataSrcDir = path.resolve(__dirname, "data");
          const dataDstDir = path.resolve(__dirname, "dist", "data");

          if (fs.existsSync(dataSrcDir)) {
            fs.mkdirSync(dataDstDir, { recursive: true });
            for (const file of fs.readdirSync(dataSrcDir)) {
              const src = path.join(dataSrcDir, file);
              const dst = path.join(dataDstDir, file);
              if (fs.statSync(src).isFile()) fs.copyFileSync(src, dst);
            }
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
