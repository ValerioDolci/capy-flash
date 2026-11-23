from __future__ import annotations

import json

import pytest

from tools.convert.resources import load_resources, token_domain

QWEN35_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
    r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
QWEN2_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


@pytest.fixture
def qwen_resources(tmp_path):
    tokenizer = {
        "model": {"type": "BPE", "vocab": {f"token_{i}": i for i in range(248044)}},
        "added_tokens": [
            {"id": i, "content": token, "special": True}
            for i, token in enumerate(("<|endoftext|>", "<|im_start|>", "<|im_end|>"), 248044)
        ],
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": QWEN2_PATTERN}, "behavior": "Isolated", "invert": False},
                {"type": "ByteLevel", "add_prefix_space": False, "use_regex": False},
            ],
        },
    }
    config = {"pretokenize_regex": QWEN35_PATTERN}
    (tmp_path / "chat_template.jinja").write_text("{{ messages }}")
    (tmp_path / "generation_config.json").write_text('{"eos_token_id":248046}')
    return tokenizer, config


def write_tokenizer(path, tokenizer, config):
    (path / "tokenizer.json").write_text(json.dumps(tokenizer))
    (path / "tokenizer_config.json").write_text(json.dumps(config))


def test_qwen_resaved_split_points_to_resource_overrides(tmp_path, qwen_resources):
    write_tokenizer(tmp_path, *qwen_resources)
    with pytest.raises(ValueError, match="Qwen2 Split") as error:
        load_resources(tmp_path, vocab_size=248320)
    message = str(error.value)
    assert "pretokenize_regex" in message
    assert "--resource tokenizer.json=" in message
    assert "--resource tokenizer_config.json=" in message


def test_qwen_corrected_overrides_preserve_checkpoint_template_and_generation(tmp_path, qwen_resources):
    tokenizer, config = qwen_resources
    write_tokenizer(tmp_path, tokenizer, config)
    original = (tmp_path / "tokenizer.json").read_bytes()
    selected = tmp_path / "official"
    selected.mkdir()
    tokenizer["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = QWEN35_PATTERN
    config["add_bos_token"] = False
    config["added_tokens_decoder"] = {str(t["id"]): t for t in tokenizer["added_tokens"]}
    write_tokenizer(selected, tokenizer, config)
    refs, payloads, count, special = load_resources(
        tmp_path,
        vocab_size=248320,
        overrides={name: selected / name for name in ("tokenizer.json", "tokenizer_config.json")},
    )
    assert count == 248047 and special == (248044, 248045, 248046)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        assert payloads[refs["text"][name]] == (selected / name).read_bytes()
    for name in ("chat_template.jinja", "generation_config.json"):
        assert payloads[refs["text"][name]] == (tmp_path / name).read_bytes()
    assert (tmp_path / "tokenizer.json").read_bytes() == original


@pytest.mark.parametrize("different", ["split", "config", "missing_config", "vocabulary", "control_token"])
def test_qwen_diagnostic_requires_the_specific_mismatch(tmp_path, qwen_resources, different):
    tokenizer, config = qwen_resources
    if different == "split":
        tokenizer["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = QWEN35_PATTERN
    elif different == "config":
        config["pretokenize_regex"] = QWEN2_PATTERN
    elif different == "missing_config":
        config.clear()
    elif different == "vocabulary":
        tokenizer["model"]["vocab"]["extra"] = 248047
    else:
        tokenizer["added_tokens"][0]["content"] = "<other>"
    write_tokenizer(tmp_path, tokenizer, config)
    load_resources(tmp_path, vocab_size=248320)


def test_final_resources_override_defaults_without_hash_pinning(tmp_path):
    (tmp_path / "tokenizer.json").write_text("invalid default")
    selected = tmp_path / "custom-tokenizer.json"
    selected.write_text(json.dumps({"model": {"type": "BPE", "vocab": {"a": 0, "b": 1}}}))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"added_tokens_decoder": {"2": {"content": "<end>", "special": True}}})
    )
    (tmp_path / "generation_config.json").write_text('{"eos_token_id":2,"temperature":1.0}')
    template = tmp_path / "custom.jinja"
    template.write_text("custom {{ messages }}")
    references, payloads, count, special = load_resources(
        tmp_path,
        vocab_size=4,
        overrides={"tokenizer.json": selected, "chat_template.jinja": template},
    )
    assert count == 3 and special == (2,)
    assert set(references) == {"text"}
    assert payloads[references["text"]["chat_template.jinja"]] == template.read_bytes()
    assert payloads[references["text"]["generation_config.json"]] == (tmp_path / "generation_config.json").read_bytes()


def test_special_tokens_merge_both_resources_with_consistent_flags():
    tokenizer = {
        "model": {"vocab": {"a": 0}},
        "added_tokens": [{"id": 1, "content": "<start>", "special": True}],
    }
    config = {"added_tokens_decoder": {"2": {"content": "<end>", "special": True}}}
    assert token_domain(tokenizer, config, 4) == (3, (1, 2))
    config["added_tokens_decoder"]["1"] = {"content": "<start>", "special": False}
    with pytest.raises(ValueError, match="special flag"):
        token_domain(tokenizer, config, 4)
