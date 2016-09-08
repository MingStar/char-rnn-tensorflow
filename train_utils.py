from __future__ import print_function
import os, sys, shutil, yaml

class BestModelSaver():
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.best_dir = os.path.join(self.save_dir, "best")

    def remove_and_initialise_best_dir(self):
        if os.path.exists(self.best_dir):
            shutil.rmtree(self.best_dir)
        os.makedirs(self.best_dir)
        # save checkpoint file
        checkpoint = os.path.join(self.best_dir, "checkpoint")
        if not os.path.exists(checkpoint):
            with open(checkpoint, 'w') as f:
                f.write("model_checkpoint_path: \"model.ckpt\"\n")
        # copy config and chars/vocab file
        for file in ["config.pkl", 'chars_vocab.pkl']:
            shutil.copyfile(os.path.join(self.save_dir, file),
                            os.path.join(self.best_dir, file))

    def keep_best(self, model_path, train_loss, step, total_steps):
        score_filepath = os.path.join(self.best_dir, "model_info.yaml")
        try:
            with open(score_filepath) as f:
                d = yaml.load(f)
                best_score_so_far = d['training_loss']
        except IOError:
            best_score_so_far = sys.float_info.max
        if train_loss >= best_score_so_far:
            return
        # now copy files
        shutil.copyfile(model_path, os.path.join(self.best_dir, "model.ckpt"))
        shutil.copyfile(model_path+".meta", os.path.join(self.best_dir, "model.ckpt.meta"))
        # save model info
        d = {'training_loss': float(train_loss), 'current_step': step, 'total_steps': total_steps}
        with open(score_filepath, 'w') as f:
            yaml.dump(d, f, default_flow_style=False)
        print("model also saved to the 'best' folder")