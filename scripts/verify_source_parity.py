"""Read-only AST/byte parity checks for one-way-copied benchmark mechanisms."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
from pathlib import Path


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise LookupError(name)


def _assignment(function: ast.FunctionDef, name: str) -> ast.AST:
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return node.value
    raise LookupError(name)


def _generate_keywords(function: ast.FunctionDef) -> list[ast.keyword]:
    for node in ast.walk(function):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "generate":
                return node.keywords
    raise LookupError("generate call")


def _constant(tree: ast.Module, name: str) -> str:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    return node.value.value
    raise LookupError(name)


def _dump(node: ast.AST | list[ast.AST]) -> str:
    if isinstance(node, list):
        return "\n".join(ast.dump(item, include_attributes=False) for item in node)
    return ast.dump(node, include_attributes=False)


class _NormalizeEngineAttributes(ast.NodeTransformer):
    """Normalize method storage without normalizing generation values."""

    def visit_Attribute(self, node: ast.Attribute):
        node = self.generic_visit(node)
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr == "tokenizer"
        ):
            return ast.copy_location(ast.Name(id="tokenizer", ctx=node.ctx), node)
        return node


def _dump_generation_keywords(keywords: list[ast.keyword]) -> str:
    normalizer = _NormalizeEngineAttributes()
    normalized = [normalizer.visit(copy.deepcopy(keyword)) for keyword in keywords]
    for keyword in normalized:
        ast.fix_missing_locations(keyword)
    return _dump(normalized)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-src", type=Path, required=True)
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "paper_structure",
    )
    args = parser.parse_args()

    old_chunk = _parse(args.benchmark_src / "chunk_and_align.py")
    old_predict = _parse(args.benchmark_src / "predict.py")
    new_chunk = _parse(args.package_dir / "chunking.py")
    new_infer = _parse(args.package_dir / "infer.py")
    new_locate = _parse(args.package_dir / "locate.py")

    checks: list[tuple[str, bool]] = []
    for function_name in ("chunk_text",):
        checks.append((
            function_name,
            _dump(_function(old_chunk, function_name))
            == _dump(_function(new_chunk, function_name)),
        ))
    for function_name in (
        "strip_for_loose",
        "_is_standalone",
        "find_all_positions",
        "select_position",
    ):
        checks.append((
            function_name,
            _dump(_function(old_chunk, function_name))
            == _dump(_function(new_locate, function_name)),
        ))
    checks.append((
        "parse_model_output",
        _dump(_function(old_predict, "parse_model_output"))
        == _dump(_function(new_infer, "parse_model_output")),
    ))

    old_prompt = _constant(old_predict, "SYSTEM_PROMPT")
    new_prompt = _constant(new_infer, "SYSTEM_PROMPT")
    checks.append(("SYSTEM_PROMPT bytes", old_prompt.encode() == new_prompt.encode()))

    old_run = _function(old_predict, "run_inference")
    new_predict = _function(new_infer, "predict_chunk")
    checks.append((
        "prompt assembly AST",
        _dump(_assignment(old_run, "prompt")) == _dump(_assignment(new_predict, "prompt")),
    ))
    checks.append((
        "generation keyword AST",
        _dump_generation_keywords(_generate_keywords(old_run))
        == _dump_generation_keywords(_generate_keywords(new_predict)),
    ))

    for name, equal in checks:
        print(f"SOURCE_PARITY {name}: {'EQUAL' if equal else 'MISMATCH'}")
    print("SYSTEM_PROMPT_SHA256=" + hashlib.sha256(new_prompt.encode()).hexdigest())
    return 0 if all(equal for _, equal in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
