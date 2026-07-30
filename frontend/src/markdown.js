// RWKV-ECRA/frontend/src/markdown.js
import DOMPurify from "dompurify";
import { marked } from "marked";

marked.setOptions({
  breaks: true,
  gfm: true,
});

// 🛡️ 防护 3：MapReduce 并发产生的错乱标题与层级清洗
function normalizeMarkdownHeadings(text) {
  if (!text) return "";
  
  // 1. 层级降维打压：将落单的 H1 (#) 强制降级为 H2 (##)，防止中间突然出现超大号标题破坏排版
  let cleaned = text.replace(/^#\s+/gm, '## ');
  
  // 2. 剥离硬编码序号：精准击杀大模型自己编的 "一、", "1.", "(一)", "第五部分" 等前缀
  // 避免与前端自带的 01, 02 计数器发生 "01 五、" 这种冲突碰撞
  cleaned = cleaned.replace(/^(#{1,6})\s+(?:第?[一二三四五六七八九十百]+[、，：\.]?\s*|\d+[\.、]\s*|（[一二三四五六七八九十百]+）\s*|\(\d+\)\s*)/gm, '$1 ');
  
  return cleaned;
}

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
  let text = String(markdown || "");
  
  // ✨ 新增：先过一遍标题清洗机
  text = normalizeMarkdownHeadings(text);

  // 🛡️ 防护 1：全局禁用波浪号的“删除线”语法
  text = text.replace(/~/g, '&#126;');
  
  // 🛡️ 防护 1：全局禁用波浪号的“删除线”语法
  // 研报场景中没有划掉文本的需求，直接将 ~ 转义为 HTML 实体，彻底掐断 Marked 解析删除线的可能
  text = text.replace(/~/g, '&#126;');

  // 🔴 核心修复：强制处理 Markdown 粗体与中文全角标点连用时不渲染的问题
  text = text.replace(/\*\*([^\n]+?)\*\*/g, '<strong>$1</strong>');
  
  // 🛡️ 防护 2：清理大模型幻觉产生的“嵌套/叠buff角标” 
  // 针对形如 ^{^[2]^}^{^[4]^} 或 ^[^{2}^]^ 的连续错误嵌套
  // 使用显式的前瞻断言 (?:\^(?![{\[]))? 完美避开吃掉相邻角标前缀的贪婪 Bug
  // 循环执行 3 次以脱去多层嵌套外壳
  for (let i = 0; i < 3; i++) {
    // 剥离外层，保留内层的网络角标 ^[x]^
    text = text.replace(/\^[\{\[]\s*\^\[(\d+)\]\^?\s*[\}\]](?:\^(?![{\[]))?/g, '^[$1]^');
    // 剥离外层，保留内层的本地角标 ^{x}^
    text = text.replace(/\^[\{\[]\s*\^\{(\d+)\}\^?\s*[\}\]](?:\^(?![{\[]))?/g, '^{$1}^');
  }

  // 预处理角标，将其转换为漂亮的、可点击跳转的 HTML 并挂载唯一 ID 供双向跳转
  // 处理网络事实角标 ^[1]^
  text = text.replace(/\^\[(\d+)\]\^/g, '<a id="cite-ref-$1" href="#source-$1" class="citation-badge web" title="跳转至网络引用来源"><span>[$1]</span></a>');
  
  // 处理本地文档角标 ^{1}^
  text = text.replace(/\^\{(\d+)\}\^/g, '<a id="cite-ref-$1" href="#source-$1" class="citation-badge local" title="跳转至本地文档来源"><span>[$1]</span></a>');

  const html = withHeadingIds(marked.parse(text));
  // DOMPurify 默认允许 <a> 和 <strong> 等安全标签属性
  return DOMPurify.sanitize(html);
}

function stripModelProtocol(text) {
  return String(text || "")
    .replace(/^\s*User:\s*[\s\S]*?\n\s*Assistant:\s*/i, "")
    .replace(/^\s*(?:Assistant:\s*)+/i, "")
    .replace(/^\s*<think>[\s\S]*?<\/think>\s*/i, "")
    .trim();
}

export function renderModelMarkdown(markdown) {
  return renderMarkdown(stripModelProtocol(markdown));
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
  
  // ✨ 新增：清洗后再提取目录，确保侧边栏目录同样干净
  const cleanMarkdown = normalizeMarkdownHeadings(markdown);

  return String(cleanMarkdown || "")
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
