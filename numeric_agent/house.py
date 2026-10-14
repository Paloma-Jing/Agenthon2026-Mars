"""House-model adapter: the ONLY model path allowed inside the competition image.

The evaluation cluster runs network="restricted". The proxy allowlist contains the
organizer-hosted endpoint and nothing else, so every call must go to
MODEL_ENDPOINT + "/v1/chat/completions" with the per-unit MODEL_TOKEN bearer.
No participant key and no external endpoint is reachable, and none is read here.

All four values are read from the environment at run time, never hardcoded:
  MODEL_ENDPOINT  scheme://host:port, no path; the OpenAI-compatible API is under /v1
  MODEL_TOKEN     per-unit bearer credential
  MODEL_NAME      the pinned house model id
  HTTP(S)_PROXY / NO_PROXY  the audited route; the OpenAI client picks these up itself

Budget: 25 admitted requests per unit, at most 4000 output tokens per request, applied
by the route. max_steps is bounded so a run cannot approach that ceiling, and the
smolagents retryer is disabled because an admitted retry spends another slot.
"""
import os
from smolagents import OpenAIModel

#: House documentation pins this model and version. MODEL_NAME wins when it is set;
#: this is only the fallback used to label the controller in local records.
HOUSE_FALLBACK_MODEL = 'nvidia/nemotron-3-super-120b-a12b'
HOUSE_VERSION = 'rl-030326-fp8'
MAX_TOKENS = 4000          # per-request ceiling enforced by the route
REQUEST_BUDGET = 25        # admitted requests per unit, enforced by the route
DEFAULT_MAX_STEPS = 8      # 8 steps + 1 possible final-answer call = 9 admits


def model_id():
    return os.environ.get('MODEL_NAME') or HOUSE_FALLBACK_MODEL


def endpoint():
    return (os.environ.get('MODEL_ENDPOINT') or '').rstrip('/')


def available():
    """True when this process can reach the organizer endpoint."""
    return bool(endpoint() and os.environ.get('MODEL_TOKEN'))


def describe():
    """Environment facts for the run record. Never returns the token."""
    return {'backend': 'house', 'model_id': model_id(), 'endpoint_set': bool(endpoint()),
            'token_set': bool(os.environ.get('MODEL_TOKEN')),
            'request_budget': REQUEST_BUDGET, 'max_output_tokens': MAX_TOKENS,
            'network_mode': os.environ.get('QFBENCH_NETWORK') or 'unset'}


def build_model(thinking=False, seed=None):
    """An OpenAI-compatible client against the house route.

    thinking is off by default: the house model thinks by default through its chat
    template, which spends output tokens on narrative header text before the tool call.
    Vendor-side tools (web search, code execution, retrieval) are never enabled here.
    """
    if not available():
        raise RuntimeError('MODEL_ENDPOINT/MODEL_TOKEN are not set; the house route is unreachable')
    extra = {'chat_template_kwargs': {'enable_thinking': True}} if thinking else {}
    kwargs = {}
    if seed is not None:
        kwargs['seed'] = int(seed)
    return OpenAIModel(model_id=model_id(), api_base=endpoint() + '/v1',
                       api_key=os.environ['MODEL_TOKEN'],
                       client_kwargs={'timeout': 120.0, 'max_retries': 0},
                       max_tokens=MAX_TOKENS, extra_body=extra or None, **kwargs)
