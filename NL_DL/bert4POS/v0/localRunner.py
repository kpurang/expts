"""
    this is meant to run the trainer, predictor, tester locally
    For now, just the trainer.
"""

import argparse
import trainer
import json

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--what', choices=['train', 'dumpConfig'], default='train',
                    help='Dump the configuration to the output directory')
    ap.add_argument('--bertConfig', required=False, default=None,
                    help='json bert config file')
    ap.add_argument('--trainConfig', required=False, default=None,
                    help='json train config file')
    args = ap.parse_args()
    bertConfig = trainer.defaultBertConfig
    trainConfig = trainer.defaultTrainConfig
    if args.bertConfig is not None:
        with open(args.bertConfig, 'r') as jx:
            jbert = json.load(jx)
            bertConfig.update(jbert)
    if args.trainConfig is not None:
        with open(args.trainConfig, 'r') as jx:
            jtrain = json.load(jx)
            trainConfig.update(jtrain)
    if args.what == 'train':
        trainer.train(trainer.trainConfig, trainer.bertConfig)
    elif args.what == 'dumpConfig':
        with open('bertConfig.json', 'w') as wx:
            json.dump(trainer.bertConfig, wx, indent=4)
        with open('trainConfig.json', 'w') as wx:
            json.dump(trainer.trainConfig, wx, indent=4)
        print('Done dump')

