"""Evaluate one fixed feature-selection parameter within training folds."""
import json
import logging
from datetime import datetime
from pathlib import Path
from conf import parse_args, dataconf
from datasets import load_data
from classification import evaluate_parameter


def main():
    args = parse_args()
    log_dir = Path(__file__).resolve().parent / 'log' / args.dataname
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = '{}-k{}-foldcv-{}'.format(args.model, args.tcdfs_k, datetime.now().strftime('%y%m%d-%H%M%S-%f'))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(message)s',
                        handlers=[logging.FileHandler(log_dir / (stem + '.log'), encoding='utf-8'),
                                  logging.StreamHandler()])
    args.logger = logging.getLogger(__name__)
    args.logger.info('Configuration: %s', vars(args))
    args.dataconf = dataconf()
    data, labels = load_data(args)
    results = evaluate_parameter(args, data.values.T, labels.to_numpy())
    output = log_dir / (stem + '.json')
    output.write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
    args.logger.info('Results saved to %s', output)


if __name__ == '__main__':
    main()
