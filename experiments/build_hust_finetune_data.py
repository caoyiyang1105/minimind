import argparse
import json
import re
import subprocess
from pathlib import Path


ARTICLE_RE = re.compile(r"(第[一二三四五六七八九十百千万〇零两]+条)\s*")
TITLE_RE = re.compile(r"^华中科技大学.+(?:规定|办法|细则|通知|规范|守则)$")


def run_pdftotext(pdf_path: Path, text_path: Path) -> None:
    subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), str(text_path)],
        check=True,
    )


def normalize_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.replace("\ufeff", "").splitlines():
        line = raw.replace("\u3000", " ").strip()
        if not line:
            continue
        if line in {"I", "II"}:
            continue
        if re.fullmatch(r"\d+", line):
            continue
        if set(line) <= {"-", "—", "_", ".", " "}:
            continue
        line = re.sub(r"\s+", " ", line)
        lines.append(line)
    return lines


def merge_wrapped_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []
    for line in lines:
        if not merged:
            merged.append(line)
            continue
        prev = merged[-1]
        starts_new = (
            TITLE_RE.match(line)
            or ARTICLE_RE.match(line)
            or re.match(r"^第[一二三四五六七八九十]+章", line)
            or re.match(r"^[一二三四五六七八九十]+、", line)
            or re.match(r"^（[一二三四五六七八九十]+）", line)
        )
        if starts_new or prev.endswith(("。", "；", "：", "！", "？", "》")):
            merged.append(line)
        else:
            merged[-1] = prev + line
    return merged


def cut_toc(lines: list[str]) -> list[str]:
    for idx, line in enumerate(lines):
        if line.startswith("华中科技大学博士研究生培养工作规定"):
            return lines[idx:]
    return lines


def current_title_from_context(context: list[str], fallback: str) -> str:
    for line in reversed(context[-20:]):
        if TITLE_RE.match(line):
            return line
    return fallback


def extract_articles(lines: list[str]) -> list[dict[str, str]]:
    articles: list[dict[str, str]] = []
    context: list[str] = []
    current_title = "《2025研究生手册》"
    current_no: str | None = None
    current_body: list[str] = []

    def flush() -> None:
        if not current_no or not current_body:
            return
        body = "\n".join(current_body).strip()
        body = re.sub(r"\n{2,}", "\n", body)
        if len(body) >= 20:
            articles.append({"title": current_title, "article": current_no, "body": body})

    for line in lines:
        title_match = TITLE_RE.match(line)
        if title_match:
            current_title = line
            context.append(line)
            continue

        article_match = ARTICLE_RE.match(line)
        if article_match:
            flush()
            current_no = article_match.group(1)
            current_title = current_title_from_context(context, current_title)
            rest = ARTICLE_RE.sub("", line, count=1).strip()
            current_body = [rest] if rest else []
        elif current_no:
            if TITLE_RE.match(line):
                flush()
                current_no = None
                current_body = []
                current_title = line
            else:
                current_body.append(line)
        context.append(line)

    flush()
    return articles


def clean_answer(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def message(role: str, content: str) -> dict[str, str]:
    return {
        "role": role,
        "content": content,
        "reasoning_content": "",
        "tools": "",
        "tool_calls": "",
    }


def build_samples(articles: list[dict[str, str]], max_answer_chars: int) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for article in articles:
        key = (article["title"], article["article"])
        if key in seen:
            continue
        seen.add(key)

        answer = clean_answer(
            f"根据《2025研究生手册》中的《{article['title']}》，{article['article']}规定：{article['body']}",
            max_answer_chars,
        )
        prompts = [
            f"《2025研究生手册》中，{article['title']}的{article['article']}规定了什么？",
            f"请根据华中科技大学《2025研究生手册》，说明“{article['title']}”里{article['article']}的内容。",
        ]
        for prompt in prompts:
            samples.append({"conversations": [message("user", prompt), message("assistant", answer)]})

    title_to_articles: dict[str, list[dict[str, str]]] = {}
    for article in articles:
        title_to_articles.setdefault(article["title"], []).append(article)
    for title, group in title_to_articles.items():
        joined = "；".join(f"{item['article']}：{item['body']}" for item in group[:4])
        answer = clean_answer(f"《{title}》主要包括以下内容：{joined}", max_answer_chars)
        prompt = f"请概括华中科技大学《2025研究生手册》中《{title}》的主要内容。"
        samples.append({"conversations": [message("user", prompt), message("assistant", answer)]})

    overview = (
        "《2025研究生手册》主要包括研究生培养、研究生管理和学位授予三大部分，"
        "涵盖培养工作、学籍管理、课程教学、奖助学金、请假与证件管理、违纪处分、"
        "学术道德、学位授予、学位论文撰写评审和论文检测处理等制度。"
    )
    samples.append(
        {
            "conversations": [
                message("user", "华中科技大学《2025研究生手册》主要包括哪些内容？"),
                message("assistant", overview),
            ]
        }
    )
    samples.extend(build_curated_samples())
    return samples


def build_curated_samples() -> list[dict[str, object]]:
    qa_pairs = [
        (
            "华中科技大学《2025研究生手册》主要包括哪些内容？",
            "《2025研究生手册》主要包括三大部分：一是研究生培养，包括博士、硕士培养工作规定、学籍管理、课程教学、专业实践、硕博连读、国际学术会议、短期出国（境）研修、涉密研究生与涉密学位论文、提前毕业和转学等制度；二是研究生管理，包括注册、身份信息变更、先进个人、奖学金、助学金、国家奖学金、家庭经济困难认定、德育助理、社会实践、违纪处分、申诉、请假、学生证校徽、学术道德和研究生守则等；三是学位授予，包括学位授予管理、学位论文撰写、论文评审、论文检测及博士学位论文重点审核等。",
        ),
        (
            "根据《2025研究生手册》，博士研究生培养目标有哪些要求？",
            "博士研究生培养目标要求包括：学习和掌握马克思主义基本理论，热爱祖国，拥护中国共产党领导，遵守宪法、法律、学术道德和学术规范；在本学科或专业领域掌握坚实全面的基础理论和系统深入的专门知识，学术学位博士生应具有独立从事学术研究工作的能力并做出创新性成果，专业学位博士生应具有独立承担专业实践工作的能力并在实践领域做出创新性成果；同时应身心健康，具有良好的综合素养。",
        ),
        (
            "《2025研究生手册》中，博士生培养方式和导师责任是怎么规定的？",
            "《华中科技大学博士研究生培养工作规定》明确：博士生的培养由导师个别指导或团队指导，导师为第一责任人。学术学位博士生以学术研究为主，重点培养独立从事学术研究的能力；专业学位博士生以专业实践为主，重点培养独立承担专业实践工作的能力。博士生还应通过完成一定学分的课程学习，掌握所在学科或专业领域坚实全面的基础理论和系统深入的专门知识。",
        ),
        (
            "根据华中科技大学《2025研究生手册》，硕士研究生培养目标是什么？",
            "硕士研究生培养目标是培养担当民族复兴大任的时代新人，服务教育强国、科技强国、人才强国建设和社会主义现代化国家建设。具体要求包括：掌握马克思主义基本理论，热爱祖国，拥护中国共产党领导，遵守宪法法律和学术规范；在本学科或专业领域掌握坚实全面的基础理论和系统深入的专门知识，学术学位硕士生应具有从事学术研究工作的能力，专业学位硕士生应具有承担专业实践工作的能力；身心健康，具有良好的综合素养。",
        ),
        (
            "《2025研究生手册》中研究生学籍管理规定适用于哪些学生？",
            "《华中科技大学研究生学籍管理规定》适用于取得华中科技大学研究生入学资格和具有学籍的各类研究生。该规定用于规范研究生学籍管理，维护学校正常教育教学秩序，保障研究生合法权益。",
        ),
        (
            "请说明《2025研究生手册》中研究生课程教学管理的基本要求。",
            "研究生课程教学管理的基本要求是全面贯彻党的教育方针，落实立德树人根本任务，加强研究生课程教学建设和管理，不断提高人才培养质量。任课教师应坚持社会主义办学方向，把立德树人贯穿教学全过程，增强学生思想品德、政治素质、社会责任感、法治意识、创新精神和实践能力，并按照课程教学大纲、教学计划和学校教学管理制度开展课程教学。",
        ),
        (
            "华中科技大学研究生请假应遵守哪些规定？",
            "研究生因病请假，校内凭校医院诊断证明，校外凭二级甲等以上医院证明，由导师签署意见、院（系）主管研究生工作负责人审批；病假超过一个月应办理休学手续。研究生一般不得请事假，特殊情况需请事假时同样由导师签署意见并由院（系）审批，事假超过一个月应办理休学手续。因国内合作与交流离校应向院（系）请假并获批，因国际合作与交流出国（境）应按学校相关规定办理手续。请假期满应按时返校销假，确需续假应办理续假手续；未经请假或未获批准擅自离校、期满未续假不归、续假未获批准逾期不归，按擅自离校处理。",
        ),
        (
            "根据《2025研究生手册》，学位论文撰写和学位授予相关制度有哪些？",
            "学位授予部分包括《华中科技大学学位授予管理规定》《华中科技大学博士、硕士学位论文撰写规定》《华中科技大学硕士博士学位论文评审规定》《华中科技大学硕士博士学位论文检测及处理规定》以及博士学位论文重点审核相关通知。学位授予管理规定明确学校按学科门类授予学士、硕士、博士三级学位，学术学位学生以学位论文申请学位，专业学位学生以学位论文或实践成果申请学位；学位论文撰写规定强调学位论文是评判学术水平和获得学位的必要条件，并对硕士、博士论文的学术水平、写作语言、文献阅读量、格式和规范等提出要求。",
        ),
    ]
    variants = {
        "华中科技大学《2025研究生手册》主要包括哪些内容？": "请概括《2025研究生手册》的章节结构。",
        "根据《2025研究生手册》，博士研究生培养目标有哪些要求？": "博士研究生培养目标在手册中是怎样表述的？",
        "《2025研究生手册》中，博士生培养方式和导师责任是怎么规定的？": "博士生培养由谁负责，培养方式是什么？",
        "根据华中科技大学《2025研究生手册》，硕士研究生培养目标是什么？": "硕士生培养目标和基本要求有哪些？",
        "《2025研究生手册》中研究生学籍管理规定适用于哪些学生？": "研究生学籍管理规定的适用对象是什么？",
        "请说明《2025研究生手册》中研究生课程教学管理的基本要求。": "研究生课程教学管理实施办法强调哪些基本要求？",
        "华中科技大学研究生请假应遵守哪些规定？": "研究生病假、事假和销假续假有哪些要求？",
        "根据《2025研究生手册》，学位论文撰写和学位授予相关制度有哪些？": "手册中学位授予和学位论文撰写相关制度是什么？",
    }
    samples: list[dict[str, object]] = []
    for prompt, answer in qa_pairs:
        samples.append({"conversations": [message("user", prompt), message("assistant", answer)]})
        samples.append({"conversations": [message("user", variants[prompt]), message("assistant", answer)]})
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HUST handbook SFT data for MiniMind.")
    parser.add_argument("--pdf", type=Path, default=Path("references/hust/2025_graduate_handbook.pdf"))
    parser.add_argument("--text", type=Path, default=Path("references/hust/2025_graduate_handbook.txt"))
    parser.add_argument("--output", type=Path, default=Path("dataset/hust_2025_handbook_sft.jsonl"))
    parser.add_argument("--max-answer-chars", type=int, default=900)
    parser.add_argument("--curated-only", action="store_true", help="Only write curated handbook QA samples.")
    parser.add_argument("--curated-repeat", type=int, default=1, help="Repeat curated QA samples for small supervised fine-tuning.")
    args = parser.parse_args()

    if not args.text.exists():
        run_pdftotext(args.pdf, args.text)

    raw_text = args.text.read_text(encoding="utf-8", errors="ignore")
    lines = cut_toc(merge_wrapped_lines(normalize_lines(raw_text)))
    articles = extract_articles(lines)
    if args.curated_only:
        samples = build_curated_samples() * max(args.curated_repeat, 1)
    else:
        samples = build_samples(articles, args.max_answer_chars)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Extracted {len(articles)} articles.")
    print(f"Wrote {len(samples)} SFT samples to {args.output}.")


if __name__ == "__main__":
    main()
