
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
from typing import Any


def clean_notebook_shell_lines(source: str) -> str:
    """Comment out notebook `!command` lines for AST analysis only."""
    return "\n".join(
        ("# " + line) if line.lstrip().startswith("!") else line
        for line in source.splitlines()
    ) + "\n"


def get_constant_str_subscript(node: ast.AST | None) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None
    if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.slice.value
    return None


def find_calls(tree: ast.AST, method_or_function: str) -> list[ast.Call]:
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Name) and f.id == method_or_function) or (
                isinstance(f, ast.Attribute) and f.attr == method_or_function
            ):
                result.append(node)
    return result


def get_kw(call: ast.Call, name: str) -> ast.AST | None:
    return next((k.value for k in call.keywords if k.arg == name), None)


def run_alignment_unit_test(tree: ast.Module) -> dict[str, Any]:
    """Execute ONLY the tokenizer/label-alignment function using a stub tokenizer.

    The stub deterministically splits a made-up test word into two subwords; it
    does not resemble a trained model and never downloads an actual tokenizer.
    """
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == 'tokenize_and_align_labels']
    if not funcs:
        return {"ran": False, "error": "Function tokenize_and_align_labels not found"}

    class StubTokenizer:
        def tokenize(self, word: str) -> list[str]:
            return ["un", "##affordable"] if word == "unaffordable" else [word]

        def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
            vocab = {"un": 101, "##affordable": 102}
            return [vocab.get(tok, 103) for tok in tokens]

    function_only = ast.Module(body=funcs, type_ignores=[])
    ns: dict[str, Any] = {"tokenizer": StubTokenizer()}
    try:
        exec(compile(function_only, "<alignment-only>", "exec"), ns, ns)
        output = ns["tokenize_and_align_labels"]({
            "Tokens": [["unaffordable"]],
            "Tags": [["1"]],
        })
        actual = output["labels"][0]
        expected = [1, 2]  # BIO: first subword B-Term, later subword I-Term
        return {
            "ran": True,
            "synthetic_input": {"Tokens": [["unaffordable"]], "Tags": [["1"]]},
            "stub_subwords": ["un", "##affordable"],
            "actual_labels": actual,
            "expected_bio_labels": expected,
            "bio_alignment_correct": actual == expected,
        }
    except Exception as exc:
        return {"ran": False, "error": repr(exc)}


def audit_pdf(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"checked": False, "error": f"Missing file: {path}"}
    if importlib.util.find_spec('fitz') is None:
        return {"checked": False, "error": "PyMuPDF missing: pip install pymupdf"}
    import fitz
    with fitz.open(str(path)) as document:
        pages = [" ".join(page.get_text().split()) for page in document]
    page10 = pages[9] if len(pages) >= 10 else ""
    page4 = pages[3] if len(pages) >= 4 else ""
    return {
        "checked": True,
        "page_count": len(pages),
        "page10_mentions_ate": "Aspect Term Extraction" in page10,
        "page10_mentions_atsc": "Aspect Term Sentiment Classification" in page10,
        "page4_contains_example": "The food was delicious" in page4
                                  and "service was terrible" in page4,
    }


def audit_source(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding='utf-8')
    original_syntax: dict[str, Any]
    try:
        ast.parse(source, filename=str(path))
        original_syntax = {"valid": True}
    except SyntaxError as error:
        original_syntax = {
            "valid": False, "line": error.lineno, "message": error.msg,
        }
    notebook_lines = [
        i for i, line in enumerate(source.splitlines(), 1)
        if line.lstrip().startswith('!')
    ]
    tree = ast.parse(clean_notebook_shell_lines(source))
    trainer_calls = find_calls(tree, 'Trainer')
    trainer = trainer_calls[0] if trainer_calls else None
    train_split = get_constant_str_subscript(get_kw(trainer, 'train_dataset')) if trainer else None
    eval_split = get_constant_str_subscript(get_kw(trainer, 'eval_dataset')) if trainer else None
    training_args_calls = find_calls(tree, 'TrainingArguments')
    training_args = training_args_calls[0] if training_args_calls else None
    load_best = get_kw(training_args, 'load_best_model_at_end') if training_args else None
    selects_best = isinstance(load_best, ast.Constant) and load_best.value is True
    top_level_train = any(
        isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and isinstance(n.value.func.value, ast.Name)
        and n.value.func.value.id == 'trainer' and n.value.func.attr == 'train'
        for n in tree.body
    )
    dataset_ids = [arg.value for c in find_calls(tree, 'load_dataset')
                   for arg in c.args[:1] if isinstance(arg, ast.Constant)
                   and isinstance(arg.value, str)]
    checkpoint_paths = [arg.value for c in find_calls(tree, 'from_pretrained')
                        for arg in c.args[:1] if isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str) and arg.value.startswith('/')]
    module_availability = {name: importlib.util.find_spec(name) is not None
                           for name in ('torch', 'transformers', 'datasets',
                                        'seqeval', 'accelerate')}
    return {
        "line_count": len(source.splitlines()),
        "original_python_syntax": original_syntax,
        "notebook_shell_command_lines": notebook_lines,
        "parsed_after_commenting_shell_commands": True,
        "source_dataset_identifiers": dataset_ids,
        "training_dataset_split": train_split,
        "evaluation_dataset_split": eval_split,
        "selects_best_checkpoint_on_eval": selects_best,
        "test_data_used_for_checkpoint_selection": eval_split == 'test' and selects_best,
        "training_runs_at_module_top_level": top_level_train,
        "alignment_unit_test": run_alignment_unit_test(tree),
        "source_uses_polarities_field": "Polarities" in source,
        "checkpoint_paths": checkpoint_paths,
        "checkpoint_paths_exist": {p: Path(p).exists() for p in checkpoint_paths},
        "dependencies_found_in_this_runtime": module_availability,
    }


def generate_report(source_path: Path, pdf_path: Path) -> dict[str, Any]:
    result = {
        "phase": 1,
        "scope": "Audit only; no model training or predictions",
        "source_file": source_path.name,
        "pdf_file": pdf_path.name,
        "pdf": audit_pdf(pdf_path),
        "ate_source": audit_source(source_path),
        "model_metrics": None,
        "model_predictions": None,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--json-out', type=Path, default=None)
    args = parser.parse_args()
    if not args.source.is_file() or not args.pdf.is_file():
        parser.error('Both --source and --pdf must exist')
    result = generate_report(args.source, args.pdf)
    if args.json_out:
        args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nAUDIT FINISHED (no model trained; findings are not model metrics)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
