from lightning.app import LightningApp, LightningFlow, LightningWork

class TrainingWork(LightningWork):
    def run(self):
        import subprocess
        subprocess.run("python scripts/train_clip.py", shell=True)
        subprocess.run("python scripts/train_llama.py", shell=True)

class ServingWork(LightningWork):
    def run(self):
        import subprocess
        subprocess.run("python main.py", shell=True)

class VQAFlow(LightningFlow):
    def __init__(self):
        super().__init__()
        self.trainer = TrainingWork()
        self.server = ServingWork()
        self.stage = "training"

    def run(self):
        if self.stage == "training":
            self.trainer.run()
            self.stage = "serving"
        elif self.stage == "serving":
            self.server.run()

app = LightningApp(VQAFlow())
