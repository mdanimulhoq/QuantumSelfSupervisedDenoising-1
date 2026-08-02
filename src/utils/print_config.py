import hydra
from omegaconf import OmegaConf

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()
