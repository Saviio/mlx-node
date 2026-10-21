"""Independent, pinned upstream MLX oracle. Run in the validation venv, never the server.

Inputs are local checkpoints. The --upstream directory must be the reviewed
jinaai/jina-reranker-v3.5-mlx revision recorded in manifest.json. Its full-sequence
attention implementation deliberately does not share native chunk/cache code.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from tokenizers import Tokenizer


def module(path, name):
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def canonical(key):
    if key.startswith('model.'):
        key = key[6:]
    for source, target in [('linear1.', 'projector.fc1.'), ('linear2.', 'projector.fc2.'),
                           ('projector.layers.0.', 'projector.fc1.'), ('projector.layers.2.', 'projector.fc2.'),
                           ('projector.0.', 'projector.fc1.'), ('projector.2.', 'projector.fc2.')]:
        if key.startswith(source):
            return target + key[len(source):]
    return key if key.startswith('projector.') else 'model.' + key


def load(path, upstream):
    config = json.loads((path / 'config.json').read_text())
    model = upstream.Model(upstream.ModelArgs.from_dict(config))
    index = path / 'model.safetensors.index.json'
    files = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
    if (path / 'projector.safetensors').exists():
        files.add('projector.safetensors')
    weights = {canonical(k): v for f in files for k, v in mx.load(str(path / f)).items() if 'lm_head' not in k}
    quant = config.get('quantization', config.get('quantization_config', {}))
    def predicate(name, layer):
        if name + '.scales' not in weights:
            return False
        bare = name.removeprefix('model.')
        q = quant.get(name, quant.get(bare, quant))
        return {key: q.get(key, quant.get(key, default)) for key, default in [('bits', 4), ('group_size', 64), ('mode', 'affine')]}
    nn.quantize(model, class_predicate=predicate)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    return model, config, Tokenizer.from_file(str(path / 'tokenizer.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--v3-upstream', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    upstream = module(args.upstream / 'modeling.py', 'modeling')
    wrapper = module(args.upstream / 'rerank.py', 'jina_v35_reference')
    v3 = module(args.v3_upstream / 'rerank.py', 'jina_v3_reference')
    model, config, tokenizer = load(args.model, upstream)
    hybrid = bool(config.get('use_sliding_window'))
    specials = {'doc_embed_token': '<|embed_token|>', 'query_embed_token': '<|rerank_token|>'}
    def encode(s):
        return tokenizer.encode(s, add_special_tokens=False).ids
    def prompt(q, docs):
        return wrapper._format_prompt(q, docs) if hybrid else v3.format_docs_prompts_func(q, docs, special_tokens=specials)
    outputs = []
    for case in json.loads(args.cases.read_text()):
        query = case['query']; docs = case['documents']; policy = case.get('policy', {})
        for marker in ['<|embed_token|>', '<|rerank_token|>', '<|score_token|>']:
            query = query.replace(marker, '')
            docs = [d.replace(marker, '') for d in docs]
        blocks = []; at = 0
        while at < len(docs):
            end = at + 1
            if len(encode(prompt(query, docs[at:end]))) > policy.get('maxBlockTokens', 32768):
                raise ValueError('case document too long')
            # Sequential reference planner independent of the native binary search.
            while end < len(docs) and end-at < policy.get('maxDocumentsPerBlock', 125) and len(encode(prompt(query, docs[at:end+1]))) <= policy.get('maxBlockTokens', 32768):
                end += 1
            ids = encode(prompt(query, docs[at:end]))
            positions = [i for i, token in enumerate(ids) if token == tokenizer.token_to_id('<|embed_token|>')]
            query_positions = [i for i, token in enumerate(ids) if token == tokenizer.token_to_id('<|rerank_token|>')]
            hidden = model(mx.array([ids]))[0]
            selected = mx.array(positions + [query_positions[-1]])
            projected = model.projector(hidden[selected]).astype(mx.float32)
            mx.eval(projected)
            features = np.array(projected)
            document_vectors = features[:-1]; query_vector = features[-1]
            scores = np.sum(document_vectors * query_vector, axis=1) / (np.linalg.norm(document_vectors, axis=1) * np.linalg.norm(query_vector))
            blocks.append({'tokens': ids, 'positions': positions, 'queryPosition': query_positions[-1], 'documents': document_vectors.tolist(), 'query': query_vector.tolist(), 'weight': float(np.max((1+scores)/2))})
            at = end
        q = np.average(np.array([b['query'] for b in blocks], dtype=np.float32), axis=0, weights=np.array([b['weight'] for b in blocks], dtype=np.float32))
        d = np.array([d for b in blocks for d in b['documents']], dtype=np.float32)
        scores = (np.sum(d * q, axis=1) / (np.linalg.norm(d, axis=1)*np.linalg.norm(q))).tolist()
        if not np.all(np.isfinite(scores)):
            raise ValueError('nonfinite oracle scores')
        outputs.append({'name': case['name'], 'scores': scores, 'blocks': blocks})
        print(case['name'], 'tokens', sum(len(b['tokens']) for b in blocks), 'blocks', len(blocks), flush=True)
        mx.clear_cache()
    args.output.write_text(json.dumps(outputs))


if __name__ == '__main__':
    main()
