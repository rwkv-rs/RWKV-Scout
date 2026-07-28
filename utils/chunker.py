# RWKV-ECRA/utils/chunker.py
import re
from utils.rwkv_tokenizer import RWKVTokenizer

_tokenizer = RWKVTokenizer("rwkv_vocab_v20230424.txt")

# ==========================================
# ⚡ 性能核心：将所有沉重的正则全局预编译，极大降低 CPU 开销
# ==========================================
_ABBREV_RE = re.compile(r'\b(mr|mrs|ms|dr|prof|sr|jr|vs|e\.g|i\.e|etc|fig|eq|vol|al)\.', flags=re.IGNORECASE)
_INITIAL_RE = re.compile(r'\b([a-zA-Z])\.')
_DECIMAL_RE = re.compile(r'(\d)\.(\d)')
_SENTENCE_SPLIT_RE = re.compile(r'([。！？]+|[.!?]+[\s\n]+|[!?]+(?=[a-zA-Z\u4e00-\u9fa5])|\.+(?=[A-Z\u4e00-\u9fa5]))')
_FALLBACK_PUNCT_RE = re.compile(r'[,，;；:：]')
_FALLBACK_CAMEL_RE = re.compile(r'[a-z][A-Z]')

def get_token_count(text: str) -> int:
    if not text: return 0
    return len(_tokenizer.encode(text))

def split_into_sentences(text: str) -> list[str]:
    """极速智能断句器（预编译版）"""
    text = _ABBREV_RE.sub(lambda m: m.group(0)[:-1] + '<PRD>', text)
    text = _INITIAL_RE.sub(r'\1<PRD>', text)
    text = _DECIMAL_RE.sub(r'\1<PRD>\2', text)

    parts = _SENTENCE_SPLIT_RE.split(text)
    
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sentence = parts[i] + parts[i+1]
        if sentence.strip(): sentences.append(sentence)
            
    if len(parts) % 2 != 0 and parts[-1].strip():
        sentences.append(parts[-1])
        
    return [s.replace('<PRD>', '.').strip() for s in sentences]

def _smart_truncate(text: str, max_tokens: int) -> tuple[str, str]:
    """极速降级截断器（保留以防其他插件调用）"""
    tokens = _tokenizer.encode(text)
    if len(tokens) <= max_tokens: return text, ""

    raw_bytes = _tokenizer.decodeBytes(tokens[:max_tokens])
    chunk_text = raw_bytes.decode('utf-8', errors='ignore')
    min_length = int(len(chunk_text) * 0.5)

    match_all = list(_FALLBACK_PUNCT_RE.finditer(chunk_text))
    if match_all and match_all[-1].end() > min_length:
        cut_idx = match_all[-1].end()
        return text[:cut_idx], text[cut_idx:]

    last_space = chunk_text.rfind(' ')
    if last_space > min_length:
        return text[:last_space], text[last_space:].lstrip()

    match_all = list(_SENTENCE_SPLIT_RE.finditer(chunk_text))
    if match_all and match_all[-1].end() > min_length:
        cut_idx = match_all[-1].end()
        return text[:cut_idx], text[cut_idx:]

    match_all = list(_FALLBACK_CAMEL_RE.finditer(chunk_text))
    if match_all and match_all[-1].start() + 1 > min_length:
        cut_idx = match_all[-1].start() + 1
        return text[:cut_idx], text[cut_idx:]

    cut_idx = len(chunk_text)
    return text[:cut_idx], text[cut_idx:]


def _hard_split_by_tokens(text: str, max_tokens: int) -> list[str]:
    """Split an oversized sentence without ever returning a larger chunk."""
    tokens = _tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return [text]
    parts: list[str] = []
    for start in range(0, len(tokens), max_tokens):
        raw = _tokenizer.decodeBytes(tokens[start : start + max_tokens])
        part = raw.decode("utf-8", errors="ignore").strip()
        if part:
            parts.append(part)
    return parts

def semantic_chunk_text(text: str, max_tokens: int = 800, overlap_ratio: float = 0.1) -> list[str]:
    """主控流程 (图表物理过滤 + 段落优先 + 句子不可破绝对屏障版)"""
    
    # 1. 语言嗅探与图表占位符准备
    is_chinese = len(re.findall(r'[\u4e00-\u9fa5]', text)) > len(text) * 0.1
    mask_str = "[图片/复杂表格已过滤]" if is_chinese else "[Image/Complex Table Filtered]"
    
    # 💥 直接进行破坏性过滤：替换标准 Markdown 图片/HTML 图片
    text = re.sub(r'!\[[^\]]*\]\([^\)]*\)|<img\b[^>]*>', mask_str, text, flags=re.IGNORECASE)
    # 💥 直接进行破坏性过滤：替换 Markdown 复杂表格（连续至少两行包含 | 的区域）
    # Tables are evidence, not decoration. Keep Markdown rows intact so
    # downstream candidates can preserve row/column relationships.
    
    chunks = []
    current_chunk = []
    current_toks = 0
    overlap_tok_limit = max(1, int(max_tokens * overlap_ratio))
    
    paragraphs = text.split('\n')
    
    for para in paragraphs:
        para = para.strip()
        if not para: continue
        para_toks = get_token_count(para)
        
        # 规则 1：段落整体放得下，直接整段塞入
        if current_toks + para_toks <= max_tokens:
            current_chunk.append(para)
            current_toks += para_toks
        else:
            # 放不下时，如果有存量，先封盘写出一个 Chunk，并保留合法句子级的 Overlap
            if current_chunk:
                chunk_str = "\n".join(current_chunk)
                # ❌ 删除了所有 mask_map 的还原操作，直接封装！
                chunks.append(chunk_str)
                
                # 计算安全的句子级 Overlap
                overlap_sents = []
                ov_toks = 0
                all_sents_in_chunk = split_into_sentences("\n".join(current_chunk))
                for s in reversed(all_sents_in_chunk):
                    s_t = get_token_count(s)
                    if ov_toks + s_t <= overlap_tok_limit:
                        overlap_sents.insert(0, s)
                        ov_toks += s_t
                    else: break
                current_chunk = overlap_sents
                current_toks = ov_toks
            
            # 再检查一遍：经历了 Overlap 遗忘后，新段落现在能放下了吗？
            if current_toks + para_toks <= max_tokens:
                current_chunk.append(para)
                current_toks += para_toks
            else:
                # 规则 2：仍然超限，被迫按句子切分处理段落
                sents = split_into_sentences(para)
                for s in sents:
                    if not s.strip(): continue
                    s_toks = get_token_count(s)
                    
                    if current_toks + s_toks <= max_tokens:
                        current_chunk.append(s)
                        current_toks += s_toks
                    else:
                        if current_chunk:
                            chunk_str = "\n".join(current_chunk)
                            # ❌ 删除了所有 mask_map 的还原操作，直接封装！
                            chunks.append(chunk_str)
                            
                            overlap_sents = []
                            ov_toks = 0
                            for osent in reversed(current_chunk):
                                os_t = get_token_count(osent)
                                if ov_toks + os_t <= overlap_tok_limit:
                                    overlap_sents.insert(0, osent)
                                    ov_toks += os_t
                                else: break
                            current_chunk = overlap_sents
                            current_toks = ov_toks
                        
                        # A single paragraph/sentence can exceed the target
                        # window.  Keep the window a hard invariant instead
                        # of silently returning an oversized chunk.
                        oversized_parts = _hard_split_by_tokens(s, max_tokens)
                        if not oversized_parts:
                            continue
                        chunks.extend(oversized_parts[:-1])
                        current_chunk = [oversized_parts[-1]]
                        current_toks = get_token_count(oversized_parts[-1])

    # 收尾最后一个块
    if current_chunk:
        chunk_str = "\n".join(current_chunk)
        # ❌ 删除了所有 mask_map 的还原操作，直接封装！
        chunks.append(chunk_str)
        
    return chunks
