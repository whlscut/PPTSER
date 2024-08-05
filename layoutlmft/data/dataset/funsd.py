# coding=utf-8
'''
Reference: https://huggingface.co/datasets/nielsr/funsd/blob/main/funsd.py
'''
import json
import csv
import os

import datasets

from layoutlmft.data.utils.image_utils import load_image, normalize_bbox


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{Jaume2019FUNSDAD,
  title={FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents},
  author={Guillaume Jaume and H. K. Ekenel and J. Thiran},
  journal={2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)},
  year={2019},
  volume={2},
  pages={1-6}
}
"""

_DESCRIPTION = """\
https://guillaumejaume.github.io/FUNSD/
"""


class FunsdConfig(datasets.BuilderConfig):
    """BuilderConfig for FUNSD"""

    def __init__(self, **kwargs):
        """BuilderConfig for FUNSD.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(FunsdConfig, self).__init__(**kwargs)


class Funsd(datasets.GeneratorBasedBuilder):
    """Conll2003 dataset."""

    BUILDER_CONFIGS = [
        FunsdConfig(name="funsd", version=datasets.Version("1.0.0"), description="FUNSD dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
                        )
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "image_path": datasets.Value("string"),
                    "word_belongs_to_seg": datasets.Sequence(datasets.Value("int64")),
                }
            ),
            supervised_keys=None,
            homepage="https://guillaumejaume.github.io/FUNSD/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        dir_data = self.config.data_dir

        with open(os.path.join(dir_data, "train.txt"), "r", encoding="utf-8") as train_f:
            orig_train_files = train_f.readlines()
            train_files = [f.strip() for f in orig_train_files]
        with open(os.path.join(dir_data, "validate.txt"), "r", encoding="utf-8") as val_f:
            # with open(os.path.join(file_dir, "task2_train.txt"), "r", encoding="utf-8") as val_f:
            orig_val_files = val_f.readlines()
            val_files = [f.strip() for f in orig_val_files]

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": train_files}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepaths": val_files}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepaths": val_files}),
        ]

    def get_line_bbox(self, bboxs):
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

        assert x1 >= x0 and y1 >= y0
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
        return bbox

    def _generate_examples(self, filepaths):
        logger.info("⏳ Generating examples from = %s", filepaths)
        data_root = "/".join(self.config.data_dir.split("/")[:-2])
        example_list = filepaths
        for guid, img_file in enumerate(sorted(example_list)):
            img_path = os.path.join(data_root, img_file)
            ann_path = img_path.replace("images/", "annotations/").split(".")[0] + '.json'

            tokens = []
            bboxes = []
            ner_tags = []
            word_belong_init = 0
            word_belongs_to_seg = []

            with open(ann_path, "r", encoding="utf8") as f:
                data = json.load(f)

            image, size = load_image(img_path)
            for item in data["form"]:
                cur_line_bboxes = []
                words, label = item["words"], item["label"]
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue
                word_belongs_to_seg = word_belongs_to_seg + [word_belong_init] * len(words)
                word_belong_init += 1

                if label == "other":
                    for w in words:
                        tokens.append(w["text"])
                        ner_tags.append("O")
                        cur_line_bboxes.append(normalize_bbox(w["box"], size))
                else:
                    tokens.append(words[0]["text"])
                    ner_tags.append("B-" + label.upper())
                    cur_line_bboxes.append(normalize_bbox(words[0]["box"], size))
                    for w in words[1:]:
                        # pass
                        tokens.append(w["text"])
                        ner_tags.append("I-" + label.upper())
                        cur_line_bboxes.append(normalize_bbox(w["box"], size))
                # by default: --segment_level_layout 1
                # if do not want to use segment_level_layout, comment the following line
                cur_line_bboxes = self.get_line_bbox(cur_line_bboxes)
                bboxes.extend(cur_line_bboxes)
            yield guid, {"id": str(guid), "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags,
                         "image": image, "image_path": img_path, "word_belongs_to_seg": word_belongs_to_seg}