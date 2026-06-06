from openai import OpenAI

SYSTEM_PROMPT = """\
你是一位深度播客听众，擅长将对话内容提炼成高质量的中文科技/思想类 blog 文章。
你的写作风格：观点明确、逻辑清晰、语言简洁有力，不堆砌废话，善用具体例子和类比。\
"""

USER_PROMPT_TEMPLATE = """\
播客名称：{podcast_title}
单集标题：{episode_title}
主播/嘉宾：{hosts}
时长：{duration}

以下是这期播客的转录稿：
---
{transcript}
---

请将上述内容整理成一篇可以直接发布的 blog 文章，要求：

1. **标题**：提炼出一个抓眼球的标题（非原集名）
2. **导语**（2-3 句）：用最有吸引力的角度切入，让读者想继续读
3. **正文**：
   - 按话题自然分节，每节有 ### 小标题
   - 用散文叙述，不要 bullet list 堆砌
   - 保留关键观点的原话引用（用 > 块引用格式）
   - 对重要观点给出你的简短点评或延伸
4. **结尾**（1 段）：提炼核心价值，给读者一个带走的思考
5. **延伸阅读**（可选）：如提到了具体书籍/工具/人物，列在末尾

风格要求：
- 写给有一定背景的读者，不做过度解释
- 保持播客的原有语气和观点，不过度二创
- 全文 1500-3000 字为宜
- 使用 Markdown 格式输出\
"""


def organize(
    transcript: str,
    episode_meta: dict,
    api_key: str,
    model: str = "gpt-4o",
) -> str:
    """Call OpenAI chat completion to turn a transcript into a blog post."""
    from fetcher import format_duration

    client = OpenAI(api_key=api_key)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        podcast_title=episode_meta.get("podcast_title", ""),
        episode_title=episode_meta.get("title", ""),
        hosts=episode_meta.get("hosts", ""),
        duration=format_duration(episode_meta.get("duration_sec")),
        transcript=transcript,
    )

    print(f"[organizer] Calling {model} to write blog post ...")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content
