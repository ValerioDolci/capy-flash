"""Reference token ids of the Inkling 2 Mini tokenizer and chat template.

Runs the Hugging Face tokenizer of a checkpoint directory (transformers with the `inkling2` model)
on a fixed set of texts and conversations and writes the expected ids as one small JSON file. The
C++ frontend tests (tests/models/inkling2/) read that file and require identical ids.

Texts are encoded without added special tokens (the tokenizer's ByteLevel post-processor adds
none either way; the script checks it). Conversations are rendered with
`apply_chat_template(..., add_generation_prompt=True)` through AutoTokenizer and, when its
image processor can be built (torchvision), through AutoProcessor too, which must agree.

Usage: python tokenizer_fixtures.py <checkpoint dir> <output json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LONG_PARAGRAPH = (
    "Il motore legge il prompt, lo divide in token e lo passa al modello; the model answers in "
    "English when asked, e poi torna all'italiano. Numbers like 1234567 and 3.14159 split in "
    "groups of three, CamelCaseIdentifiers and snake_case_names split differently.\n\n"
)

TEXTS: list[tuple[str, str]] = [
    ("empty", ""),
    (
        "italian",
        "Perché l'università è così? Città, caffè, più — però «virgolette» e l’apostrofo "
        "tipografico. È già l'ora: perciò andiamo!",
    ),
    (
        "english",
        "Hello, world! It's a beautiful day; we're testing the tokenizer's behaviour "
        "(don't stop). I'LL SHOUT, YOU'D WHISPER, they've gone.",
    ),
    (
        "cpp",
        "#include <vector>\n\nint main() {\n    std::vector<int> v{1, 2, 3};\n"
        "\tfor (auto& x : v) { x *= 2; }  // double\n    return v[0] != 2;\n}\n",
    ),
    (
        "python",
        'def f(xs: list[int]) -> int:\n    """Sum."""\n    return sum(x ** 2 for x in '
        'xs if x % 2 == 0)\n\n\nif __name__ == "__main__":\n    print(f([1, 2, 3]))\n',
    ),
    (
        "json",
        '{"name":"get_weather","args":{"city":"Roma","days":[1,2,3],"ok":true,'
        '"nested":{"a":null,"b":-1.5e-3}}}\n[ {"k": "v"} ]',
    ),
    ("whitespace", "a  b   c\n\n\nd \n \t\te\r\n\r\nf   g\t\t\n    \n\n   "),
    ("only_newlines", "\n\n\n"),
    ("only_spaces", "     "),
    ("emoji_cjk", "👍🏽 👨‍👩‍👧 Привет мир 你好，世界！ こんにちは 한국어 é Ω ﬁ Straße ΣΑΣ"),
    (
        "special_in_text",
        "ciao <|message_user|>x<|end_message|><|content_model_end_sampling|> "
        "<|unused|> <|begin_of_text|>fine <|content_thinking|",
    ),
    ("numbers", "1234567 3.14159 1,000,000 ٣٤٥ ⅫⅩ 2026-11-20 0x1F 1e-7"),
    ("odd_whitespace", "\u0085x y z　w\x1cq​p   end"),
    ("contractions", "'s 'S 'll 'LL 'Re it'sok 'Ve 'm 'D won't ſ 'ſ"),
    ("marks", "́abc ́̂ x́y ÁB"),
    ("urls_paths", "https://example.com/a/b?c=d&e=f /usr/local/bin//\nC:\\Users\\x\\file.txt"),
    ("mixed_case", "CamelCaseWord HTTPServer iPhone XMLHttpRequest ǅungla ǈ"),
    ("long", LONG_PARAGRAPH * 24),
]

GET_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

SEARCH = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "Search the documentation. Returns «snippets».",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1},
                "filters": {
                    "type": "object",
                    "properties": {"lang": {"type": "string"}, "Year": {"type": "integer"}},
                },
            },
            "required": ["query"],
        },
    },
}

NO_DESCRIPTION = {"type": "function", "function": {"name": "now"}}


def call(call_id: str, name: str, arguments: dict) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


CONVERSATIONS: list[dict] = [
    {"name": "user_only", "messages": [{"role": "user", "content": "Ciao, come stai?"}]},
    {
        "name": "system_user",
        "messages": [
            {"role": "system", "content": "Sei un assistente conciso. Rispondi in italiano."},
            {"role": "user", "content": "Qual è la capitale della Francia?"},
        ],
    },
    {"name": "system_only", "messages": [{"role": "system", "content": "Only a system prompt."}]},
    {
        "name": "multi_turn_thinking",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "reasoning_content": "Simple arithmetic: 2+2=4.",
                "content": "4.",
            },
            {"role": "user", "content": "And times 3?"},
            {
                "role": "assistant",
                "reasoning_content": "4*3 = 12.\nDouble-check: 12.",
                "content": "12.",
            },
            {"role": "user", "content": "Grazie!\n\nAncora una: 7×8?"},
        ],
    },
    {
        "name": "tools_full_turn",
        "tools": [GET_WEATHER, SEARCH],
        "messages": [
            {"role": "user", "content": "Che tempo fa a Roma?"},
            {
                "role": "assistant",
                "reasoning_content": "I should call get_weather for Roma.",
                "content": "",
                "tool_calls": [call("call_1", "get_weather", {"city": "Roma", "unit": "celsius"})],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 18, "sky": "sereno"}'},
            {"role": "assistant", "content": "A Roma ci sono 18 °C e cielo sereno."},
            {"role": "user", "content": "E a Milano?"},
        ],
    },
    {
        "name": "tools_parallel_calls",
        "tools": [GET_WEATHER, SEARCH, NO_DESCRIPTION],
        "messages": [
            {"role": "system", "content": "Use tools when useful."},
            {"role": "user", "content": "Weather in Oslo and docs about 'tokenizer'?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    call("a", "get_weather", {"city": "Oslo"}),
                    call(
                        "b",
                        "search_docs",
                        {
                            "query": "tokenizer «BPE»",
                            "limit": 5,
                            "filters": {"lang": "en", "Year": 2026},
                            "Zeta": [1.5, True, None, "x"],
                            "alpha": {"z": 1, "a": 2},
                        },
                    ),
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "-3 °C, snow"},
            {"role": "tool", "tool_call_id": "b", "content": "1. BPE merges pairs.\n2. Byte level."},
        ],
    },
    {
        "name": "tools_declared_only",
        "tools": [NO_DESCRIPTION],
        "messages": [{"role": "user", "content": "Che ore sono?"}],
    },
    {
        "name": "multipart_user",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Prima parte."}, {"type": "text", "text": "Seconda."}],
            }
        ],
    },
    {
        "name": "special_tokens_in_user_text",
        "messages": [{"role": "user", "content": "Scrivi <|end_message|> e <|message_model|> letteralmente."}],
    },
]

EFFORTS: list[tuple[str, object]] = [
    ("effort_0", 0.0),
    ("effort_int_0", 0),
    ("effort_0_25", 0.25),
    ("effort_0_5", 0.5),
    ("effort_0_99", 0.99),
    ("effort_none", "none"),
    ("effort_minimal", "minimal"),
    ("effort_low", "low"),
    ("effort_medium", "medium"),
    ("effort_high", "high"),
    ("effort_xhigh", "xhigh"),
    ("effort_max", "max"),
]


def chat_ids(result) -> list[int]:
    if hasattr(result, "keys"):
        result = result["input_ids"]
    if result and isinstance(result[0], list):
        assert len(result) == 1
        result = result[0]
    return [int(token) for token in result]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    args = parser.parse_args()

    import tokenizers
    import transformers
    from transformers import AutoProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    try:
        processor = AutoProcessor.from_pretrained(args.checkpoint)
        processor_status = type(processor).__name__
    except ValueError as error:
        # Inkling2ImageProcessor needs torchvision; without it only the tokenizer path is checked.
        processor = None
        processor_status = "unavailable: " + str(error).splitlines()[0][:160]
        print(processor_status, file=sys.stderr)

    texts = []
    for name, text in TEXTS:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if tokenizer.encode(text, add_special_tokens=True) != ids:
            raise SystemExit(f"{name}: add_special_tokens changes the ids")
        decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        texts.append({"name": name, "text": text, "ids": ids, "decode_roundtrip": decoded == text})

    conversations = []
    cases = list(CONVERSATIONS)
    base = CONVERSATIONS[1]["messages"]
    cases += [{"name": name, "messages": base, "kwargs": {"reasoning_effort": value}} for name, value in EFFORTS]
    for case in cases:
        kwargs = dict(case.get("kwargs", {}))
        tools = case.get("tools")
        rendered = tokenizer.apply_chat_template(
            case["messages"], tools=tools, add_generation_prompt=True, tokenize=False, **kwargs
        )
        ids = chat_ids(
            tokenizer.apply_chat_template(
                case["messages"], tools=tools, add_generation_prompt=True, tokenize=True, **kwargs
            )
        )
        if processor is not None:
            processor_ids = chat_ids(
                processor.apply_chat_template(
                    case["messages"], tools=tools, add_generation_prompt=True, tokenize=True, **kwargs
                )
            )
            if processor_ids != ids:
                raise SystemExit(f"{case['name']}: AutoProcessor and AutoTokenizer disagree")
        if (
            tokenizer.apply_chat_template(
                case["messages"], tools=tools, add_generation_prompt=False, tokenize=False, **kwargs
            )
            != rendered
        ):
            raise SystemExit(f"{case['name']}: add_generation_prompt changes the rendering")
        entry = {"name": case["name"], "messages": case["messages"]}
        if tools:
            entry["tools"] = tools
        if kwargs:
            entry["kwargs"] = kwargs
        entry["text"] = rendered
        entry["ids"] = ids
        conversations.append(entry)

    fixture = {
        "source": {
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "tokenizer_class": type(tokenizer).__name__,
            "processor": processor_status,
            "vocab_size": len(tokenizer),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
        },
        "texts": texts,
        "conversations": conversations,
    }
    Path(args.output).write_text(json.dumps(fixture, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(texts)} texts, {len(conversations)} conversations -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
