from dependencies.state_machine.depth_analysis_sm import DepthAnalysisSM
from dependencies import loadConfig

def main():
    config = loadConfig.get_config()
    depth_analysis_sm = DepthAnalysisSM(config)
    while True:
        depth_analysis_sm.runtime()

if __name__ == "__main__":
    main()