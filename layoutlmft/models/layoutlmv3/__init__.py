from collections import OrderedDict

from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification, \
    AutoModelForQuestionAnswering, AutoModelForSequenceClassification, AutoTokenizer
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, RobertaConverter

from transformers.models.auto.modeling_auto import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update

from .configuration_layoutlmv3 import LayoutLMv3Config
from .modeling_layoutlmv3_fs import (
    LayoutLMv3ForFSMethod,
)
from .tokenization_layoutlmv3 import LayoutLMv3Tokenizer
from .tokenization_layoutlmv3_fast import LayoutLMv3TokenizerFast


AutoConfig.register("layoutlmv3", LayoutLMv3Config)

AutoTokenizer.register(
    LayoutLMv3Config, slow_tokenizer_class=LayoutLMv3Tokenizer, fast_tokenizer_class=LayoutLMv3TokenizerFast
)

CONFIG_MAPPING_NAMES = OrderedDict()
MODEL_FOR_FS_METHOD_NAMES = OrderedDict()
MODEL_FOR_FS_METHOD = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_FS_METHOD_NAMES)
class AutoModelForFSMethod(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_FS_METHOD
AutoModelForFSMethod = auto_class_update(AutoModelForFSMethod)
AutoModelForFSMethod.register(
    LayoutLMv3Config, LayoutLMv3ForFSMethod
)

SLOW_TO_FAST_CONVERTERS.update({"LayoutLMv3Tokenizer": RobertaConverter})
