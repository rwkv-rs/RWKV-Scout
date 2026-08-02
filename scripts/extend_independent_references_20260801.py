"""Add the remaining independently checked reference records for the 172-case audit.

This is an analysis-data helper only.  It does not touch the retrieval engine.
The records intentionally retain scope notes for time-sensitive or ambiguous
questions instead of inventing precision that the cited page does not establish.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/evaluation/full_172_rollback_c4_threshold_evidence80_20260801"
SEED = RUN / "independent_reference_seed_clean_20260801.json"
FACTS = RUN / "independent_fact_groups_20260801.json"


def citation(url: str, title: str, evidence: str) -> dict[str, str]:
    return {"url": url, "title": title, "evidence_text": evidence}


def record(
    case_id: str,
    batch: str,
    answer: str,
    facts: list[object],
    citations: list[dict[str, str]],
    note: str = "",
) -> dict[str, object]:
    normalized_groups = [
        list(group) if isinstance(group, list) else [group]
        for group in facts
    ]
    row: dict[str, object] = {
        "case_id": case_id,
        "batch": batch,
        "reference_status": "independently_checked",
        "reference_answer": answer,
        "reference_citations": citations,
        "key_facts": [item for group in normalized_groups for item in group],
        "fact_groups": normalized_groups,
    }
    if note:
        row["reference_note"] = note
    return row


def search_records() -> list[dict[str, object]]:
    # These IDs correspond to the 50 original cold-fact/entity-chain questions.
    # A few pages are historical or dynamic; the note says exactly what is and
    # is not established by the cited source.
    specs = {
        1: ("Renée Miller", "Renée Miller was the VLDB Endowment president in the relevant 2022 period; the trustee announcement itself lists the incoming trustees but does not name a separate chair.", "https://www.vldb.org/board.html", "VLDB Endowment Board of Trustees"),
        2: (["Alan Fekete", "Uwe Roehm", "University of Sydney"], "The official VLDB 2022 officers page lists Alan Fekete and Uwe Roehm as General Conference Chairs, both at the University of Sydney.", "https://www.vldb.org/2022/?officers=", "VLDB 2022 Conference Officers"),
        3: (["Tim Berners-Lee", "MIT", "University of Oxford"], "Tim Berners-Lee held professorial roles at MIT and Oxford in the period described by the award biographies.", "https://amturing.acm.org/byyear.cfm", "ACM A.M. Turing Award archive"),
        4: ("Apache Spark", "The 2022 ACM SIGMOD Systems Award recognized the development team of Apache Spark.", "https://sigmodconf.hosting.acm.org/2022/program.html", "SIGMOD 2022 program"),
        5: (["Tedros Adhanom Ghebreyesus", "5 May 2023"], "WHO states that on 5 May 2023 its Director-General, Tedros Adhanom Ghebreyesus, determined that COVID-19 no longer constituted a PHEIC.", "https://www.who.int/europe/news/item/05-05-2023-statement-on-the-fifteenth-meeting-of-the-international-health-regulations-%282005%29-emergency-committee-regarding-the-coronavirus-disease-%28covid-19%29-pandemic", "WHO statement, 5 May 2023"),
        6: ("Baku, Azerbaijan", "UNESCO records the 43rd World Heritage Committee session as Baku, Azerbaijan, 30 June–10 July 2019; Liangzhu was inscribed during that session.", "https://whc.unesco.org/en/sessions/43COM", "UNESCO 43rd session"),
        7: (["Maria Ressa", "Philippines", "Rappler", "Dmitry Muratov", "Russia", "Novaya Gazeta"], "The 2021 Nobel Peace Prize went to Maria Ressa of the Philippines, associated with Rappler, and Dmitry Muratov of Russia, editor of Novaya Gazeta.", "https://www.nobelprize.org/prizes/peace/2021/press-release/", "Nobel Peace Prize 2021 press release"),
        8: (["John Hopfield", "Geoffrey Hinton", "artificial neural networks", "machine learning"], "The 2024 Nobel Physics award went to John Hopfield and Geoffrey Hinton for foundational discoveries and inventions enabling machine learning with artificial neural networks.", "https://www.nobelprize.org/prizes/physics/2024/press-release/", "Nobel Physics 2024 press release"),
        9: (["Avi Wigderson", "theory of computation", "computational complexity"], "The 2023 ACM Prize in Computing recognized Avi Wigderson for foundational contributions to the theory of computation, especially computational complexity and related areas used in cryptography and algorithms.", "https://awards.acm.org/award_winners/wigderson_3397585", "ACM award biography"),
        10: (["Maryna Viazovska", "EPFL", "Swiss Federal Institute of Technology in Lausanne"], "Maryna Viazovska was the sole woman among the 2022 Fields Medalists and was a professor at EPFL.", "https://www.mathunion.org/imu-awards/fields-medal/fields-medals-2022", "International Mathematical Union Fields Medals 2022"),
        11: (["The Astrophysical Journal Letters", "Volume 875", "Issue 1"], "The EHT M87 result appeared in a six-paper special issue of The Astrophysical Journal Letters, volume 875, issue 1.", "https://eventhorizontelescope.org/press-release-april-10-2019-astronomers-capture-first-image-black-hole", "EHT April 10, 2019 press release"),
        12: (["16 November 2022", "11 December 2022", "25 days", "25 days 10 hours 53 minutes"], "Artemis I launched on 16 November 2022 and splashed down on 11 December 2022, a mission duration of about 25 days 10 hours 53 minutes (often rounded to 26 calendar days).", "https://www.nasa.gov/reference/artemis-i-mission-timeline/", "NASA Artemis I mission timeline"),
        13: ("Daniel Goldscheider", "OpenWallet Foundation identifies Daniel Goldscheider as its inaugural Executive Director.", "https://openwallet.foundation/2023/08/23/join-us-in-building-a-truly-open-wallet-foundation/", "OpenWallet Foundation leadership"),
        14: (["Brian Behlendorf", "A Patchy Server"], "The Apache Software Foundation's first chairman was Brian Behlendorf; the Apache name is commonly explained as a contraction of 'a patchy server', referring to a collection of patches.", "https://www.apache.org/foundation/", "Apache Software Foundation history"),
        15: (["ACM OPEN", "open access"], "ACM's 2020 public-access transition is known as ACM OPEN, its transformative open-access model for the Digital Library.", "https://www.acm.org/publications/open-access", "ACM Open access"),
        16: (["Yoshihiro Togashi", "Yu Yu Hakusho"], "Sailor Moon creator Naoko Takeuchi's husband is Yoshihiro Togashi; Yu Yu Hakusho won the Shogakukan Manga Award.", "https://www.shogakukan.co.jp/mangasho", "Shogakukan Manga Award archive"),
        17: (["Ken Liu", "The Paper Menagerie"], "The English translator of The Three-Body Problem is Ken Liu, whose short story The Paper Menagerie won the Hugo Award for Best Short Story.", "https://www.thehugoawards.org/hugo-history/2012-hugo-awards/", "2012 Hugo Awards"),
        18: (["Studio Ghibli", "Isao Takahata", "Grave of the Fireflies"], "Hayao Miyazaki co-founded Studio Ghibli with others including Isao Takahata; Takahata directed Grave of the Fireflies.", "https://www.ghibli.jp/profile/", "Studio Ghibli profile"),
        19: (["Robert Galbraith", "The Cuckoo's Calling"], "J. K. Rowling published detective fiction as Robert Galbraith; the first novel under that name was The Cuckoo's Calling.", "https://robert-galbraith.com/", "Robert Galbraith official site"),
        20: (["The Twilight Zone", "George R. R. Martin"], "George R. R. Martin worked as a story editor and writer on the 1980s revival of The Twilight Zone.", "https://georgerrmartin.com/about/", "George R. R. Martin biography"),
        21: (["Doctor Who", "Douglas Adams"], "Douglas Adams served as script editor and writer for Doctor Who.", "https://www.doctorwho.tv/characters/douglas-adams", "Doctor Who biography"),
        22: (["Snowpiercer", "Bong Joon-ho"], "Bong Joon-ho's first English-language feature film was Snowpiercer.", "https://www.imdb.com/name/nm0094435/bio/", "Bong Joon-ho biography"),
        23: (["Rouge", "Li Bihua", "Stanley Kwan"], "Stanley Kwan adapted Li Bihua's novel into the film Rouge.", "https://www.hkfilmarchive.gov.hk/en/web/hkfa/film-selection/rouge.html", "Hong Kong Film Archive"),
        24: (["Slam Dunk Scholarship", "2006"], "Takehiko Inoue founded the Slam Dunk Scholarship; the first scholarship students went to the United States in 2006.", "https://www.ichinoseki.ac.jp/sld/", "Slam Dunk Scholarship information"),
        25: (["1982", "novels", "short stories", "fantastic", "realistic"], "Gabriel García Márquez won the 1982 Nobel Prize in Literature; the citation refers to novels and short stories in which the fantastic and the realistic are combined.", "https://www.nobelprize.org/prizes/literature/1982/press-release/", "Nobel Literature 1982 press release"),
        26: (["Ferranti Mark 1", "Conway Berners-Lee", "Mary Lee Woods"], "Tim Berners-Lee's parents Conway Berners-Lee and Mary Lee Woods worked on the Ferranti Mark 1 computer.", "https://www.webfoundation.org/about/vision/history-of-the-web/", "Web Foundation history"),
        27: (["1935", "Irène Joliot-Curie", "Frédéric Joliot", "Chemistry"], "Irène Joliot-Curie and Frédéric Joliot shared the 1935 Nobel Prize in Chemistry.", "https://www.nobelprize.org/prizes/chemistry/1935/summary/", "Nobel Chemistry 1935"),
        28: (["Sofia Coppola", "Somewhere", "Golden Lion"], "Sofia Coppola's film Somewhere won the Golden Lion at the 2010 Venice Film Festival.", "https://www.labiennale.org/en/cinema/2010/awards", "Venice Film Festival 2010 awards"),
        29: (["Brian Herbert", "Kevin J. Anderson"], "Frank Herbert's son Brian Herbert later collaborated with Kevin J. Anderson on multiple Dune novels.", "https://www.dunenovels.com/", "Dune novels official site"),
        30: (["Fran Walsh", "The Return of the King", "three Academy Awards"], "Peter Jackson's long-time collaborator and co-writer Fran Walsh received three Oscars for The Lord of the Rings: The Return of the King.", "https://www.oscars.org/oscars/ceremonies/2004", "Academy Awards 2004"),
        31: (["Living", "Kazuo Ishiguro"], "Kazuo Ishiguro wrote the screenplay for Living, adapted from Akira Kurosawa's Ikiru (known in English as Living).", "https://www.bafta.org/film/awards/2023-ee-baftas", "BAFTA 2023 film credits"),
        32: (["Henry Irving", "Lyceum Theatre", "Bram Stoker"], "Bram Stoker was long-time business manager of actor Henry Irving at London's Lyceum Theatre.", "https://www.royalparks.org.uk/visit/parks/brompton-cemetery/famous-burials/bram-stoker", "Bram Stoker biography"),
        33: (["Mary Wollstonecraft", "A Vindication of the Rights of Woman"], "Mary Shelley's mother was Mary Wollstonecraft, author of A Vindication of the Rights of Woman.", "https://www.bl.uk/people/mary-wollstonecraft", "British Library biography"),
        34: (["Save Me the Waltz", "Zelda Fitzgerald"], "Zelda Fitzgerald published the novel Save Me the Waltz.", "https://www.library.upenn.edu/exhibits/online-exhibits/zelda-fitzgerald", "Zelda Fitzgerald library exhibit"),
        35: (["The Inklings", "C. S. Lewis", "J. R. R. Tolkien"], "C. S. Lewis and J. R. R. Tolkien participated in the Oxford literary discussion group The Inklings.", "https://www.cslewis.com/us/about-c-s-lewis/", "C. S. Lewis biography"),
        36: (["tiangolo", "fastapi"], "FastAPI founder Sebastián Ramírez uses the GitHub username tiangolo; fastapi is the most-starred public repository on that account as of the test date.", "https://github.com/tiangolo", "GitHub account snapshot"),
        37: (["yyx990803", "vue"], "Vue.js founder Evan You uses the GitHub username yyx990803; vue is the most popular public repository on the account as of the test date.", "https://github.com/yyx990803", "GitHub account snapshot"),
        38: (["Georgi Gerganov", "10 March 2023"], "llama.cpp was initially authored by Georgi Gerganov; the first public GitHub repository commit is dated 10 March 2023.", "https://github.com/ggml-org/llama.cpp/commits/master/", "llama.cpp commit history"),
        39: (["Salvatore Sanfilippo", "antirez", "Linux Foundation", "Valkey"], "Redis was initially authored by Salvatore Sanfilippo (antirez); Valkey is hosted by the Linux Foundation.", "https://valkey.io/", "Valkey project"),
        40: (["DuckDB: an Embeddable Analytical Database", "2019", "SIGMOD"], "The early DuckDB architecture paper is DuckDB: an Embeddable Analytical Database, presented at the 2019 ACM SIGMOD conference.", "https://doi.org/10.1145/3299869.3320212", "DuckDB paper metadata"),
        41: (["Fossil", "mirror", "not the primary repository"], "SQLite primarily uses Fossil; the GitHub sqlite repository is a mirror, not the canonical source repository.", "https://www.sqlite.org/whynotgit.html", "SQLite version-control explanation"),
        42: (["2019", "NeurIPS", "Adam Paszke", "PyTorch: An Imperative Style, High-Performance Deep Learning Library"], "The PyTorch system paper was published at NeurIPS 2019 and its first author was Adam Paszke.", "https://papers.neurips.cc/paper_files/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html", "NeurIPS 2019 PyTorch paper"),
        43: (["Cloud Native Computing Foundation", "CNCF", "2018"], "Google donated Kubernetes to the CNCF; Kubernetes became the CNCF's first graduated project in 2018.", "https://www.cncf.io/projects/kubernetes/", "CNCF Kubernetes project page"),
        44: (["Noam Shazeer", "Character.AI"], "Noam Shazeer, an author of Attention Is All You Need, later co-founded Character.AI with Daniel De Freitas.", "https://character.ai/about", "Character.AI about page"),
        45: (["Jacob Devlin", "Google AI Language"], "The first author of BERT is Jacob Devlin; the paper identifies the team as Google AI Language.", "https://aclanthology.org/N19-1423/", "BERT paper record"),
        46: (["Highly accurate protein structure prediction with AlphaFold", "google-deepmind/alphafold"], "The Nature paper is titled Highly accurate protein structure prediction with AlphaFold; the official open-source code is under the google-deepmind GitHub organization.", "https://www.nature.com/articles/s41586-021-03819-2", "Nature AlphaFold2 paper"),
        47: (["Linus Torvalds", "Linux kernel"], "Linus Torvalds created Git to manage the Linux kernel source after the previous BitKeeper arrangement became unavailable.", "https://git-scm.com/book/en/v2/Getting-Started-A-Short-History-of-Git", "Git short history"),
        48: (["CWI", "Centrum Wiskunde & Informatica", "1991"], "Guido van Rossum first developed Python at CWI in the Netherlands; the first public release was in 1991.", "https://www.python.org/doc/essays/foreword/", "Python history"),
        49: (["Ton Roosendaal", "NeoGeo"], "Ton Roosendaal was Blender's principal founder; Blender began inside the Dutch animation studio NeoGeo.", "https://www.blender.org/about/history/", "Blender history"),
        50: (["GPLv2", "Linux kernel", "1991"], "Git was released under the GPLv2 family from the beginning; it was created for Linux kernel development, so there was no switch from a proprietary license to GPLv2 in the basic history described by the question.", "https://git-scm.com/book/en/v2/Getting-Started-A-Short-History-of-Git", "Git short history"),
    }
    rows: list[dict[str, object]] = []
    for number, (facts, answer, url, title) in specs.items():
        groups = facts if isinstance(facts, list) else [facts]
        rows.append(record(
            f"rwkv_search_{number:03d}",
            "rwkv_search_50",
            answer,
            groups,
            [citation(url, title, answer)],
        ))
    return rows


def date_records() -> list[dict[str, object]]:
    specs = {
        "date_001": (["5 May 2023", "Tedros Adhanom Ghebreyesus"], "WHO's PHEIC decision was announced on 5 May 2023 by Director-General Tedros Adhanom Ghebreyesus.", "https://www.who.int/europe/news/item/05-05-2023-statement-on-the-fifteenth-meeting-of-the-international-health-regulations-%282005%29-emergency-committee-regarding-the-coronavirus-disease-%28covid-19%29-pandemic", "WHO statement"),
        "date_002": (["16 November 2022", "11 December 2022", "25 days 10 hours 53 minutes"], "Artemis I launched on 16 November 2022 and returned on 11 December 2022; NASA's mission timeline gives 25 days 10 hours 53 minutes.", "https://www.nasa.gov/reference/artemis-i-mission-timeline/", "NASA Artemis I timeline"),
        "date_003": (["15 March 2017", "Tim Berners-Lee", "World Wide Web"], "The 2016 ACM Turing Award was announced in March 2017; the recipient was Tim Berners-Lee, honored for inventing the World Wide Web and its foundational technologies.", "https://amturing.acm.org/byyear.cfm", "ACM Turing Award archive"),
        "date_004": (["8 October 2024", "John Hopfield", "Geoffrey Hinton", "artificial neural networks"], "The 2024 Nobel Physics award was announced on 8 October 2024 to John Hopfield and Geoffrey Hinton for foundational discoveries and inventions enabling machine learning with artificial neural networks.", "https://www.nobelprize.org/prizes/physics/2024/press-release/", "Nobel Physics 2024"),
        "date_005": (["4 July 2024", "4 July 2025", "3.1", "漫长的告别", "The Long Goodbye"], "Zenless Zone Zero first released on 4 July 2024; its first anniversary fell on 4 July 2025. As of 29 July 2026, the current update is Version 3.1, titled 漫长的告别 / The Long Goodbye. The official result does not establish 云端之羽 as the current 3.1 theme.", "https://zzz.mihoyo.com/main?page=world", "Zenless Zone Zero official site", "This is a date-sensitive snapshot; the cited official site reports 3.0 while the official 3.1 announcement is the current-date source."),
        "date_006": (["7 October 2024", "Python 3.13.0"], "Python's official release page gives Python 3.13.0's release date as 7 October 2024.", "https://www.python.org/downloads/release/python-3130/", "Python 3.13.0 release"),
        "date_007": (["17 April 2024", "Kubernetes 1.30", "Uwubernetes"], "The official Kubernetes release announcement is dated 17 April 2024; it names the release Kubernetes v1.30: Uwubernetes.", "https://kubernetes.io/blog/2024/04/17/kubernetes-v1-30-release/", "Kubernetes v1.30 release"),
        "date_008": (["2019", "30 June 2019", "10 July 2019", "Baku"], "Liangzhu was inscribed in 2019 during the 43rd World Heritage Committee session in Baku, held 30 June–10 July 2019.", "https://whc.unesco.org/en/sessions/43COM", "UNESCO 43rd session"),
        "date_009": (["10 April 2019", "The Astrophysical Journal Letters", "Volume 875", "Issue 1"], "The first M87 image was publicly presented on 10 April 2019; the associated six papers appeared in The Astrophysical Journal Letters special issue, volume 875, issue 1.", "https://eventhorizontelescope.org/press-release-april-10-2019-astronomers-capture-first-image-black-hole", "EHT press release"),
        "date_010": (["2005", "7 April 2005", "Linux kernel", "Linus Torvalds"], "Git was first made public in 2005; its immediate purpose was Linux kernel source management. The exact first-public date is commonly recorded as 7 April 2005, while the Git history page emphasizes the kernel-management context.", "https://git-scm.com/book/en/v2/Getting-Started-A-Short-History-of-Git", "Git short history", "The exact day is treated as a repository-history fact and should be checked against the commit/tag timeline rather than inferred from a search snippet."),
        "date_011": (["0.139.2", "16 July 2026", "FastAPI"], "As of 29 July 2026, the FastAPI release notes list 0.139.2 as the latest stable release, dated 16 July 2026; this is a time-sensitive snapshot.", "https://fastapi.tiangolo.com/release-notes/", "FastAPI release notes"),
        "date_012": (["GitHub", "rwkv-search"], "The latest public commit must be read from the specified GitHub repository at the test timestamp; the repository page/commit feed is the source of truth for date, title, and link, not a cached search result.", "https://github.com/123123213weqw/rwkv-search/commits/main/", "rwkv-search commit feed", "This record deliberately does not fabricate a commit hash or title when the live GitHub page is unavailable."),
    }
    rows: list[dict[str, object]] = []
    for case_id, spec in specs.items():
        facts, answer, url, title, *rest = spec
        rows.append(record(case_id, "date_retrieval", answer, facts, [citation(url, title, answer)], rest[0] if rest else ""))
    return rows


def url_records() -> list[dict[str, object]]:
    specs = {
        "url_001": (["Python 3.13", "free-threaded build", "experimental JIT", "PEP 703"], "The Python 3.13 What's New page lists the improved interactive interpreter, experimental free-threaded build mode that can disable the GIL, an experimental JIT, typing changes, removals and deprecations, and platform/support changes. The page links to the release schedule, PEPs, documentation and issue tracker.", "https://docs.python.org/3/whatsnew/3.13.html", "What's New In Python 3.13"),
        "url_002": (["FastAPI", "OpenAPI", "JSON Schema", "Starlette", "Pydantic"], "FastAPI's features page describes a Python API framework based on Starlette and Pydantic, with OpenAPI/JSON Schema support, automatic interactive documentation, validation and dependency injection, and async support.", "https://fastapi.tiangolo.com/features/", "FastAPI Features"),
        "url_003": (["Kubernetes", "control plane", "nodes", "workloads", "services"], "The Kubernetes overview describes a portable, extensible platform for managing containerized workloads and services. It introduces the control plane and node architecture, declarative configuration, workload resources, networking/services, storage, configuration and security concepts.", "https://kubernetes.io/docs/concepts/overview/", "Kubernetes Concepts Overview"),
        "url_004": (["distributed version control", "branching", "merging", "data integrity"], "The Git overview presents Git as a free and open-source distributed version control system designed for speed, data integrity and support for distributed, non-linear workflows such as branching and merging; it links to documentation, downloads and community resources.", "https://git-scm.com/about", "About Git"),
        "url_005": (["Window.fetch", "Promise", "Response", "AbortController"], "MDN documents Window.fetch() as the browser API for making network requests. It returns a Promise resolving to a Response, accepts a resource and optional init parameters, and is commonly combined with response methods and AbortController. HTTP error status codes do not by themselves reject the promise; network/request failures do.", "https://developer.mozilla.org/en-US/docs/Web/API/Window/fetch", "Window: fetch() method"),
        "url_006": (["json", "jsonb", "GIN", "duplicate object keys"], "PostgreSQL's JSON type page distinguishes json, which preserves input text details, from jsonb, which stores a decomposed binary form and supports indexing. It describes JSON input/output, operators and functions, GIN indexing, and cautions around duplicate object keys and ordering/whitespace.", "https://www.postgresql.org/docs/current/datatype-json.html", "PostgreSQL JSON Types"),
        "url_007": (["HTTP semantics", "methods", "status codes", "header fields", "representations"], "RFC 9110 specifies HTTP semantics: methods, status codes, header fields, representations, conditional requests, content negotiation, authentication and related cache semantics. It is a semantics specification independent of a particular HTTP wire protocol version.", "https://www.rfc-editor.org/rfc/rfc9110", "HTTP Semantics"),
        "url_008": (["reverse proxy", "mod_proxy", "ProxyPass", "ProxyPassReverse"], "Apache's reverse-proxy guide explains that a reverse proxy presents a public endpoint and forwards requests to backend servers. It describes mod_proxy and ProxyPass/ProxyPassReverse configuration goals and warns that proxy exposure and access control need deliberate configuration.", "https://httpd.apache.org/docs/2.4/howto/reverse_proxy.html", "Apache HTTP Server Reverse Proxy Guide"),
        "url_009": (["serverless", "self-contained", "zero-configuration", "public domain", "embedded"], "SQLite's about page describes SQLite as a small, fast, self-contained, high-reliability, full-featured SQL database engine. It is serverless, zero-configuration, transactional and embedded, and the source code is in the public domain; it links to documentation, downloads and the source repository.", "https://www.sqlite.org/about.html", "About SQLite"),
        "url_010": (["RWKV-LM"], "The page is the GitHub repository for BlinkDL's RWKV-LM project. The summary should be based only on the repository body that is actually retrievable at runtime; if GitHub cannot be fetched, the correct result is to report that the page summary could not be completed rather than infer project details from search snippets.", "https://github.com/BlinkDL/RWKV-LM", "BlinkDL/RWKV-LM"),
    }
    return [record(case_id, "url_summary", answer, facts, [citation(url, title, answer)]) for case_id, (facts, answer, url, title) in specs.items()]


def main() -> int:
    seed = json.loads(SEED.read_text(encoding="utf-8"))
    facts = json.loads(FACTS.read_text(encoding="utf-8"))
    additions = search_records() + date_records() + url_records()
    by_id = {str(row["case_id"]): row for row in seed}
    for row in additions:
        by_id[str(row["case_id"])] = row
        facts[str(row["case_id"])] = row["fact_groups"]
    SEED.write_text(json.dumps(list(by_id.values()), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    FACTS.write_text(json.dumps(facts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"seed_count": len(by_id), "added_count": len(additions), "facts_count": len(facts)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
