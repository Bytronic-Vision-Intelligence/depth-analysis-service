from dependencies.state_machine.depth_analysis_sm import DepthAnalysisSM
from dependencies import loadConfig
import argparse

def main(config_path:str):
    config = loadConfig.get_config(config_path)
    depth_analysis_sm = DepthAnalysisSM(config)
    while True:
        depth_analysis_sm.runtime()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="get config directory")
    parser.add_argument("--config", type= str, help="the location of the config file")
    args = parser.parse_args()
    config_path = args.config
    main(config_path)