"""Upload an already completed visual analysis directory to Comet."""
import argparse
import json
from pathlib import Path

import comet_ml

parser = argparse.ArgumentParser()
parser.add_argument('--artifact-dir', required=True)
parser.add_argument('--run-name', required=True)
parser.add_argument('--k', type=int, required=True)
args = parser.parse_args()
root = Path(args.artifact_dir)
summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
experiment = comet_ml.Experiment(workspace='team-rl-exp')
experiment.set_name(args.run_name)
experiment.add_tags(['TunedLens', 'CKA', 'SID-F', f'k{args.k}', 'wikitext103', 'reuploaded'])
experiment.log_parameters({
    'source_checkpoint': summary['source_checkpoint'],
    'k': args.k,
    'device': summary['device'],
    'fit_steps': summary['fit_steps'],
    'eval_batches': summary['eval_batches'],
    'checkpoint_optimizer_step': summary.get('checkpoint_optimizer_step'),
    'checkpoint_tokens_processed': summary.get('checkpoint_tokens_processed'),
})
for metric_name, values in summary['metrics'].items():
    for layer, value in enumerate(values, 1):
        experiment.log_metric(f'validation/layer_{layer}/{metric_name}', value)
for layer, value in enumerate(summary.get('fit_forward_kl', []), 1):
    experiment.log_metric(f'fit/layer_{layer}/forward_kl', value)
for figure_path in sorted(root.glob('*.png')):
    experiment.log_image(str(figure_path), name=figure_path.stem, image_format='png', step=0)
experiment.log_asset(str(root / 'summary.json'), overwrite=True)
experiment.end()
print(f'COMET_LOGGED {args.run_name}')
