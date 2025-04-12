from premiumModel.train import Trainer

if __name__ == "__main__":
    img_path = "testImages/image4.jpg"
    tokenizer_path = "processed/tokenizer.p"
    model_weights_path = "models/modelV1.h5"

    trainer = Trainer(img_path, tokenizer_path, model_weights_path)
    trainer.train()
