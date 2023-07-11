import argparse
import os

from ano3ddpm import dist_util
from ano3ddpm.mri_dataset import load_data
from ano3ddpm.resample import create_named_schedule_sampler
from ano3ddpm.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from ano3ddpm.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()
    os.environ["DIFFUSION_BLOB_LOGDIR"] = f"{args.output_dir}/checkpoints"
    os.environ["OPENAI_LOGDIR"] = f"{args.output_dir}/logs"
    os.environ["PLOTDIR"] = f"{args.output_dir}/plots"

    for dir in ["vid", "img"]:
        os.makedirs(f"{args.output_dir}/plots/{dir}", exist_ok=True)

    dist_util.setup_dist()

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    data_train = load_data(
        file_path=args.file_path,
        batch_size=args.batch_size,
        split="train",
        random_slice=True,
    )

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data_train,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        max_steps=args.max_steps,
    ).run_loop()


def create_argparser() -> argparse.ArgumentParser:
    defaults = dict(
        file_path="",
        output_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=1000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        max_steps=500000,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
