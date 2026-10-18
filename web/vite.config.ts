import { defineConfig } from "vite";

export default defineConfig({
  // dev only: lets ?src=/@fs/<path> load corpus files that live outside web/
  server: { fs: { allow: [".."] } },
  // relative asset paths: the site works from any folder, such as GitHub Pages' /afterpass/
  base: "./",
  worker: { format: "es" },
  build: { target: "es2022" },
});
