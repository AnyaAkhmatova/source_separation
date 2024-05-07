import torch




class SpexPlusShortGRUInferencer:
    def __init__(self,
                 rank,
                 world_size,
                 model,
                 streamer,
                 dataloader,
                 config,
                 logger,
                 device,
                 test_mode=False,
                 criterion=None,
                 metrics=None,
                 save_inference=False,
                 save_dir=None):
        self.rank = rank
        self.world_size = world_size

        self.model = model
        self.streamer = streamer
        self.dataloader = dataloader

        self.config = config
        self.logger = logger
        self.device = device
        
        self.test_mode = test_mode
        self.criterion = criterion
        self.metrics = metrics

        self.save_inference = save_inference
        self.save_dir = save_dir

        self.model = self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def move_batch_to_device(batch, device, test_mode):
        for tensor_for_gpu in ["mix_chunks", "ref", "lens"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device, non_blocking=True)
        if test_mode:
            batch["target"] = batch["target"].to(device, non_blocking=True)
        return batch

    def run(self):
                    batch["mix_chunks"], n_chunks = self.streamer.make_chunks(batch["mix"])
            batch = self.move_batch_to_device(batch, self.device, False)
            
            batch["s1"] = self.model(batch["mix_chunks"], batch["ref"], False)
            length = batch["target"].shape[-1]
            batch["s1"] = self.streamer.apply_overlap_add_method(batch["s1"], n_chunks)
            batch["s1"] = batch["s1"][:, :length]
            
            batch["loss"], batch["sisdr"] = self.criterion(**batch, have_relevant_speakers=False)
        
        metrics.update("loss", batch["loss"].item())
        metrics.update("SISDR", batch["sisdr"].item())

        for met in metric_names:
            batch[met.lower()] = self.metrics[met](**batch)
            metrics.update(met, batch[met.lower()].item())



        return
