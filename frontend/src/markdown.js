import DOMPurify from "dompurify";
import { marked } from "marked";

marked.setOptions({
  breaks: true,
  gfm: true,
});

function slugifyHeading(text) {
  return String(text || "")
    .replace(/<[^>]+>/g, "")
    .trim()
    .toLowerCase()
    .replace(/[^\w\u4e00-\u9fa5\s-]/g, "")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-");
}

function withHeadingIds(html) {
  const counts = new Map();
  return html.replace(/<(h[1-3])>(.*?)<\/\1>/g, (_, tag, content) => {
    const base = slugifyHeading(content) || "section";
    const count = counts.get(base) || 0;
    counts.set(base, count + 1);
    const id = count ? `${base}-${count}` : base;
    return `<${tag} id="${id}">${content}</${tag}>`;
  });
}

export function renderMarkdown(markdown) {
  const html = withHeadingIds(marked.parse(String(markdown ?? "")));
  return DOMPurify.sanitize(html);
}

// Model text follows exactly the same ordinary Markdown renderer as any
// user-provided Markdown.  No protocol stripping or answer-specific rewrite
// is allowed here.
export function renderModelMarkdown(markdown) {
  return renderMarkdown(markdown);
}

export function reportToMarkdown(report) {
  if (!report) return "";
  if (report.nodes?.length) {
    return report.nodes.map((node) => `## ${node.title}\n\n${node.content}`).join("\n\n");
  }
  return report.markdown || "";
}

export function extractMarkdownOutline(markdown) {
  const counts = new Map();
  return String(markdown || "")
    .split("\n")
    .map((line) => /^(#{1,3})\s+(.+)$/.exec(line.trim()))
    .filter(Boolean)
    .slice(0, 30)
    .map((match) => {
      const title = match[2];
      const base = slugifyHeading(title) || "section";
      const count = counts.get(base) || 0;
      counts.set(base, count + 1);
      return {
        id: count ? `${base}-${count}` : base,
        title,
      };
    });
}
