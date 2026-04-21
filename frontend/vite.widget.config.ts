/**
 * Vite Widget Build Configuration
 *
 * Builds the chat widget as a single self-contained IIFE bundle
 * that can be dropped into any website via a <script> tag.
 *
 * Output: dist-widget/widget.js (single file, CSS included)
 *
 * Usage:
 *   npm run build:widget
 *
 * The bundle includes React, all components, and all CSS — inlined.
 * No external dependencies required on the host site.
 */

import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

/**
 * Plugin that injects extracted CSS into the JS bundle at runtime.
 * This ensures the widget is a single .js file with no separate .css.
 */
function cssInjector(): Plugin {
  return {
    name: "css-injector",
    enforce: "post",
    generateBundle(_options, bundle) {
      // Find the CSS asset
      const cssAsset = Object.keys(bundle).find((key) => key.endsWith(".css"));
      if (!cssAsset) return;

      const css = bundle[cssAsset];
      if (css.type !== "asset") return;

      const cssContent = typeof css.source === "string"
        ? css.source
        : new TextDecoder().decode(css.source);

      // Escape backticks and ${} in CSS for template literal
      const escapedCss = cssContent
        .replace(/\\/g, "\\\\")
        .replace(/`/g, "\\`")
        .replace(/\$/g, "\\$");

      // Inject CSS into the JS entry chunk
      const jsEntry = Object.keys(bundle).find((key) => key.endsWith(".js"));
      if (jsEntry && bundle[jsEntry].type === "chunk") {
        const injection = `(function(){var s=document.createElement("style");s.setAttribute("data-cw-widget","");s.textContent=\`${escapedCss}\`;document.head.appendChild(s)})();`;
        bundle[jsEntry].code = injection + "\n" + bundle[jsEntry].code;
      }

      // Remove the CSS asset from the bundle
      delete bundle[cssAsset];
    },
  };
}

export default defineConfig({
  plugins: [react(), cssInjector()],
  build: {
    lib: {
      entry: resolve(__dirname, "src/widget/widget-entry.tsx"),
      name: "ColorWhistleChat",
      fileName: () => "widget.js",
      formats: ["iife"],
    },
    outDir: "dist-widget",
    emptyOutDir: true,
    cssCodeSplit: false,
    cssMinify: true,
    minify: true,
    rollupOptions: {
      external: [],
    },
  },
  publicDir: false,
  define: {
    "process.env.NODE_ENV": JSON.stringify("production"),
  },
});
