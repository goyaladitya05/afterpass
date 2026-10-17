import { defineConfig } from "vite";

export default defineConfig({
  // dev only: lets ?src=/@fs/<path> load corpus files that live outside web/
  server: { fs: { allow: [".."] } },
  worker: { format: "es" },
  build: { target: "es2022" },
});
