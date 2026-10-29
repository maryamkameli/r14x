# Placeholders for the optional LLM-grader dependencies imported at module load by the official AMPO repo
# (verl/deepscaler/utils.py: vertexai, google-cloud-aiplatform, sentence_transformers). The grader only runs with
# use_math_orm=True, which AMPO never sets; anything that touches these placeholders raises.
import importlib.util, sys, types

def _missing(name):
    try:
        return importlib.util.find_spec(name) is None
    except (ImportError, ValueError):
        return True

class _Unavailable:
    def __init__(self, *a, **k):
        raise RuntimeError('LLM-grader dependency stubbed out for the AMPO baselines')

def _stub(path, attrs):
    parts = path.split('.')
    for i in range(1, len(parts) + 1):
        name = '.'.join(parts[:i])
        if name not in sys.modules:
            m = types.ModuleType(name); m.__path__ = []
            sys.modules[name] = m
            if i > 1:
                setattr(sys.modules['.'.join(parts[:i - 1])], parts[i - 1], m)
    for a in attrs:
        setattr(sys.modules[path], a, _Unavailable)

if _missing('vertexai'):
    _stub('vertexai', ['init'])
    _stub('vertexai.generative_models', ['GenerationConfig', 'GenerativeModel', 'HarmBlockThreshold', 'HarmCategory'])
if _missing('google.cloud.aiplatform_v1beta1'):
    _stub('google.cloud.aiplatform_v1beta1.types.content', ['SafetySetting'])
if _missing('sentence_transformers'):
    _stub('sentence_transformers', ['SentenceTransformer', 'util'])
