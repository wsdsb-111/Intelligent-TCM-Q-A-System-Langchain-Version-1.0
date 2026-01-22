"""
将已有的 HuggingFace Transformers 模型目录转换为 Sentence-Transformers 结构。

使用方法（在项目根目录运行）:
    python 工具脚本/convert_to_sentence_transformer.py \
        --source "E:/毕业论文和设计/线上智能中医问答项目/Model Layer/model/iic/nlp_gte_sentence-embedding_chinese-base/iic/nlp_gte_sentence-embedding_chinese-base" \
        --target "E:/毕业论文和设计/线上智能中医问答项目/Model Layer/model/sentence-transformers/nlp_gte_sentence-embedding_chinese-base"

如果不指定 --target，将在 source 目录中原地生成 Sentence-Transformers 所需结构。
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer, models


def convert_model(source: Path, target: Path | None = None, pooling: str = "mean") -> Path:
    if not source.exists():
        raise FileNotFoundError(f"源模型目录不存在: {source}")

    if target is None:
        target = source

    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    print("源模型目录:", source.resolve())
    print("目标保存目录:", target)

    print("\n加载 Transformers 模型模块...")
    word_embedding_model = models.Transformer(str(source.resolve()))

    pooling_mode = pooling.lower()
    if pooling_mode == "mean":
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False,
        )
    elif pooling_mode == "cls":
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=False,
            pooling_mode_cls_token=True,
            pooling_mode_max_tokens=False,
        )
    else:
        raise ValueError(f"暂不支持的池化方式: {pooling}")

    print("构建 Sentence-Transformers 管线...")
    sentence_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    print("保存为 Sentence-Transformers 结构...")
    sentence_model.save(str(target))

    print("\n转换完成 ✅")
    return target


def main():
    parser = argparse.ArgumentParser(description="将 Transformers 模型目录转换为 Sentence-Transformers 结构")
    parser.add_argument(
        "--source",
        required=True,
        help="现有 Transformers 模型目录（包含 config.json、pytorch_model.bin 等）",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="可选，输出目录。默认在 source 目录原地生成/覆盖 Sentence-Transformers 结构",
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "cls"],
        default="mean",
        help="池化方式，默认 mean",
    )

    args = parser.parse_args()

    source = Path(args.source).expanduser()
    target = Path(args.target).expanduser() if args.target else None

    convert_model(source, target, pooling=args.pooling)


if __name__ == "__main__":
    main()

